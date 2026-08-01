`default_nettype none

module ace2_rope_core #(
    parameter integer LANES = 16,
    parameter integer ACT_WIDTH = 8,
    parameter integer SCALE_WIDTH = 16,
    parameter integer TRIG_WIDTH = 16
) (
    input  wire                                  clk_i,
    input  wire                                  rst_ni,
    input  wire                                  clear_i,

    input  wire                                  start_valid_i,
    output wire                                  start_ready_o,

    input  wire                                  beat_valid_i,
    output wire                                  beat_ready_o,
    input  wire [LANES*ACT_WIDTH-1:0]           act_data_i,
    input  wire [LANES*ACT_WIDTH-1:0]           pair_data_i,
    input  wire [LANES*SCALE_WIDTH-1:0]         scale_data_i,
    input  wire [LANES*SCALE_WIDTH-1:0]         pair_scale_data_i,
    input  wire [LANES*TRIG_WIDTH-1:0]          cos_data_i,
    input  wire [LANES*TRIG_WIDTH-1:0]          sin_data_i,
    input  wire                                  second_half_i,

    output wire                                  out_valid_o,
    input  wire                                  out_ready_i,
    output wire [LANES*ACT_WIDTH-1:0]           out_data_o,
    output wire                                  saturation_seen_o
);
    localparam integer LANE_INDEX_WIDTH = (LANES <= 1) ? 1 : $clog2(LANES);
    localparam [LANE_INDEX_WIDTH-1:0] LAST_LANE = LANE_INDEX_WIDTH'(LANES - 1);
    localparam integer ROPE_SCALE_FRAC = 9;
    localparam integer ROPE_ROUND_SHIFT = ROPE_SCALE_FRAC + (TRIG_WIDTH - 1);
    localparam integer PRE_ROTATE_WIDTH = ACT_WIDTH + SCALE_WIDTH;
    localparam integer ROTATE_PRODUCT_WIDTH = PRE_ROTATE_WIDTH + TRIG_WIDTH;
    localparam integer ROTATE_SUM_WIDTH = ROTATE_PRODUCT_WIDTH + 1;
    localparam integer ROUNDED_BASE_WIDTH = ROTATE_SUM_WIDTH - ROPE_ROUND_SHIFT;
    localparam integer MUL_WIDTH = ROTATE_PRODUCT_WIDTH;
    localparam [21:0] ST_IDLE              = 22'h000001;
    localparam [21:0] ST_LOAD              = 22'h000002;
    localparam [21:0] ST_SETUP_ACT_SCALE   = 22'h000004;
    localparam [21:0] ST_MUL_ACT_SCALE     = 22'h000008;
    localparam [21:0] ST_SETUP_PAIR_SCALE  = 22'h000010;
    localparam [21:0] ST_MUL_PAIR_SCALE    = 22'h000020;
    localparam [21:0] ST_SETUP_COS         = 22'h000040;
    localparam [21:0] ST_MUL_COS           = 22'h000080;
    localparam [21:0] ST_SETUP_SIN         = 22'h000100;
    localparam [21:0] ST_MUL_SIN           = 22'h000200;
    localparam [21:0] ST_ROUND_PREP        = 22'h000400;
    localparam [21:0] ST_ROUND_HIGH        = 22'h000800;
    localparam [21:0] ST_ROUND_RESULT      = 22'h001000;
    localparam [21:0] ST_ROUND             = 22'h002000;
    localparam [21:0] ST_DONE              = 22'h004000;
    localparam [21:0] ST_FINISH_ACT_SCALE  = 22'h008000;
    localparam [21:0] ST_FINISH_PAIR_SCALE = 22'h010000;
    localparam [21:0] ST_FINISH_COS        = 22'h020000;
    localparam [21:0] ST_FINISH_SIN        = 22'h040000;
    localparam [21:0] ST_SELECT_LANE       = 22'h080000;
    localparam [21:0] ST_NEGATE_PRODUCT    = 22'h100000;
    localparam [21:0] ST_ROTATE_CHUNK      = 22'h200000;
    localparam [1:0] MUL_DEST_ACT_SCALE  = 2'd0;
    localparam [1:0] MUL_DEST_PAIR_SCALE = 2'd1;
    localparam [1:0] MUL_DEST_COS        = 2'd2;
    localparam [1:0] MUL_DEST_SIN        = 2'd3;

    (* fsm_encoding = "none" *) reg [21:0] state_q;
    reg [LANE_INDEX_WIDTH-1:0] lane_idx_q;
    reg [LANES-1:0] lane_onehot_q;
    reg second_half_q;
    reg signed [ACT_WIDTH-1:0] act_lane_q;
    reg signed [ACT_WIDTH-1:0] pair_lane_q;
    reg signed [SCALE_WIDTH-1:0] scale_lane_q;
    reg signed [SCALE_WIDTH-1:0] pair_scale_lane_q;
    reg signed [TRIG_WIDTH-1:0] cos_lane_q;
    reg signed [TRIG_WIDTH-1:0] sin_lane_q;
    reg signed [PRE_ROTATE_WIDTH-1:0] act_scaled_q;
    reg signed [PRE_ROTATE_WIDTH-1:0] pair_scaled_q;
    reg signed [ROTATE_PRODUCT_WIDTH-1:0] prod_cos_q;
    reg signed [ROTATE_PRODUCT_WIDTH-1:0] prod_sin_q;
    reg signed [ROTATE_SUM_WIDTH-1:0] round_value_q;
    reg [ROTATE_SUM_WIDTH-1:0] round_abs_q;
    reg [ACT_WIDTH-1:0] rounded_output_q;
    reg round_saturate_q;
    reg [MUL_WIDTH-1:0] mul_acc_q;
    reg [MUL_WIDTH-1:0] mul_multiplicand_q;
    reg [15:0] mul_multiplier_q;
    reg [4:0] mul_count_q;
    reg [1:0] mul_chunk_idx_q;
    reg mul_carry_q;
    reg mul_sign_q;
    reg [1:0] mul_dest_q;
    reg [1:0] mul_negate_chunk_idx_q;
    reg mul_negate_carry_q;
    reg [1:0] rotate_chunk_idx_q;
    reg rotate_carry_q;
    reg [1:0] round_negate_chunk_idx_q;
    reg round_negate_carry_q;
    reg [LANES*ACT_WIDTH-1:0] out_data_q;
    reg out_valid_q;
    reg saturation_seen_q;

    wire signed [ACT_WIDTH-1:0] act_lane_w;
    wire signed [ACT_WIDTH-1:0] pair_lane_w;
    wire signed [SCALE_WIDTH-1:0] scale_lane_w;
    wire signed [SCALE_WIDTH-1:0] pair_scale_lane_w;
    wire signed [TRIG_WIDTH-1:0] cos_lane_w;
    wire signed [TRIG_WIDTH-1:0] sin_lane_w;
    reg signed [ACT_WIDTH-1:0] selected_act_lane_w;
    reg signed [ACT_WIDTH-1:0] selected_pair_lane_w;
    reg signed [SCALE_WIDTH-1:0] selected_scale_lane_w;
    reg signed [SCALE_WIDTH-1:0] selected_pair_scale_lane_w;
    reg signed [TRIG_WIDTH-1:0] selected_cos_lane_w;
    reg signed [TRIG_WIDTH-1:0] selected_sin_lane_w;
    wire [15:0] act_lane_abs_w;
    wire [15:0] pair_lane_abs_w;
    wire [15:0] scale_lane_abs_w;
    wire [15:0] pair_scale_lane_abs_w;
    wire [PRE_ROTATE_WIDTH-1:0] act_scaled_abs_w;
    wire [PRE_ROTATE_WIDTH-1:0] pair_scaled_abs_w;
    wire [15:0] cos_lane_abs_w;
    wire [15:0] sin_lane_abs_w;
    reg [15:0] mul_acc_chunk_w;
    reg [15:0] mul_multiplicand_chunk_w;
    reg [15:0] mul_negate_chunk_w;
    reg [15:0] rotate_lhs_chunk_w;
    reg [15:0] rotate_rhs_chunk_w;
    reg [15:0] round_negate_chunk_w;
    integer lane_sel_idx;
    wire [15:0] mul_addend_chunk_w;
    wire [16:0] mul_chunk_sum_w;
    wire [MUL_WIDTH-1:0] mul_acc_next_w;
    wire [16:0] mul_negate_sum_w;
    wire [MUL_WIDTH-1:0] mul_negate_next_w;
    wire [16:0] rotate_sum_w;
    wire [ROTATE_SUM_WIDTH-1:0] rotate_next_w;
    wire [16:0] round_negate_sum_w;
    wire [ROTATE_SUM_WIDTH-1:0] round_negate_next_w;
    wire signed [ROTATE_SUM_WIDTH-1:0] prod_cos_ext_w;
    wire signed [ROTATE_SUM_WIDTH-1:0] prod_sin_ext_w;
    wire [ROUNDED_BASE_WIDTH-1:0] rounded_base_w;
    wire [ROPE_ROUND_SHIFT-1:0] rounded_remainder_w;
    wire round_increment_w;
    wire [ROUNDED_BASE_WIDTH:0] rounded_abs_w;
    wire output_saturate_w;
    wire [ACT_WIDTH-1:0] rounded_output_w;
    wire state_idle_w;
    wire state_load_w;

    assign state_idle_w = state_q[0];
    assign state_load_w = state_q[1];
    assign start_ready_o = state_idle_w && !out_valid_q;
    assign beat_ready_o = state_load_w;
    assign out_valid_o = out_valid_q;
    assign out_data_o = out_data_q;
    assign saturation_seen_o = saturation_seen_q;

    assign act_lane_w = act_lane_q;
    assign pair_lane_w = pair_lane_q;
    assign scale_lane_w = scale_lane_q;
    assign pair_scale_lane_w = pair_scale_lane_q;
    assign cos_lane_w = cos_lane_q;
    assign sin_lane_w = sin_lane_q;

    assign act_lane_abs_w = act_lane_w[ACT_WIDTH-1] ? {8'd0, (~act_lane_w + {{(ACT_WIDTH-1){1'b0}}, 1'b1})} :
                                                        {8'd0, act_lane_w};
    assign pair_lane_abs_w = pair_lane_w[ACT_WIDTH-1] ? {8'd0, (~pair_lane_w + {{(ACT_WIDTH-1){1'b0}}, 1'b1})} :
                                                        {8'd0, pair_lane_w};
    assign scale_lane_abs_w = scale_lane_w[SCALE_WIDTH-1] ? (~scale_lane_w + {{(SCALE_WIDTH-1){1'b0}}, 1'b1}) :
                                                            scale_lane_w;
    assign pair_scale_lane_abs_w = pair_scale_lane_w[SCALE_WIDTH-1] ? (~pair_scale_lane_w + {{(SCALE_WIDTH-1){1'b0}}, 1'b1}) :
                                                                      pair_scale_lane_w;
    assign act_scaled_abs_w = act_scaled_q[PRE_ROTATE_WIDTH-1] ?
                              (~act_scaled_q + {{(PRE_ROTATE_WIDTH-1){1'b0}}, 1'b1}) :
                              act_scaled_q;
    assign pair_scaled_abs_w = pair_scaled_q[PRE_ROTATE_WIDTH-1] ?
                               (~pair_scaled_q + {{(PRE_ROTATE_WIDTH-1){1'b0}}, 1'b1}) :
                               pair_scaled_q;
    assign cos_lane_abs_w = cos_lane_w[TRIG_WIDTH-1] ? (~cos_lane_w + {{(TRIG_WIDTH-1){1'b0}}, 1'b1}) :
                                                       cos_lane_w;
    assign sin_lane_abs_w = sin_lane_w[TRIG_WIDTH-1] ? (~sin_lane_w + {{(TRIG_WIDTH-1){1'b0}}, 1'b1}) :
                                                       sin_lane_w;
    assign mul_addend_chunk_w = mul_multiplier_q[0] ? mul_multiplicand_chunk_w : 16'd0;
    assign mul_chunk_sum_w = {1'b0, mul_acc_chunk_w} +
                             {1'b0, mul_addend_chunk_w} +
                             {16'd0, (mul_chunk_idx_q != 2'd0) && mul_carry_q};
    assign mul_acc_next_w = (mul_chunk_idx_q == 2'd0) ?
                            {mul_acc_q[MUL_WIDTH-1:16], mul_chunk_sum_w[15:0]} :
                            ((mul_chunk_idx_q == 2'd1) ?
                             {mul_acc_q[MUL_WIDTH-1:32], mul_chunk_sum_w[15:0], mul_acc_q[15:0]} :
                             {mul_chunk_sum_w[7:0], mul_acc_q[31:0]});
    assign mul_negate_sum_w = {1'b0, mul_negate_chunk_w} + {16'd0, mul_negate_carry_q};
    assign mul_negate_next_w = (mul_negate_chunk_idx_q == 2'd0) ?
                               {mul_acc_q[MUL_WIDTH-1:16], mul_negate_sum_w[15:0]} :
                               ((mul_negate_chunk_idx_q == 2'd1) ?
                                {mul_acc_q[MUL_WIDTH-1:32], mul_negate_sum_w[15:0], mul_acc_q[15:0]} :
                                {mul_negate_sum_w[7:0], mul_acc_q[31:0]});
    assign rotate_sum_w = {1'b0, rotate_lhs_chunk_w} +
                          {1'b0, rotate_rhs_chunk_w} +
                          {16'd0, rotate_carry_q};
    assign rotate_next_w = (rotate_chunk_idx_q == 2'd0) ?
                           {round_value_q[ROTATE_SUM_WIDTH-1:16], rotate_sum_w[15:0]} :
                           ((rotate_chunk_idx_q == 2'd1) ?
                            {round_value_q[ROTATE_SUM_WIDTH-1:32], rotate_sum_w[15:0], round_value_q[15:0]} :
                            {rotate_sum_w[8:0], round_value_q[31:0]});
    assign round_negate_sum_w = {1'b0, round_negate_chunk_w} + {16'd0, round_negate_carry_q};
    assign round_negate_next_w = (round_negate_chunk_idx_q == 2'd0) ?
                                 {round_abs_q[ROTATE_SUM_WIDTH-1:16], round_negate_sum_w[15:0]} :
                                 ((round_negate_chunk_idx_q == 2'd1) ?
                                  {round_abs_q[ROTATE_SUM_WIDTH-1:32], round_negate_sum_w[15:0], round_abs_q[15:0]} :
                                  {round_negate_sum_w[8:0], round_abs_q[31:0]});
    assign prod_cos_ext_w = {{(ROTATE_SUM_WIDTH-ROTATE_PRODUCT_WIDTH){prod_cos_q[ROTATE_PRODUCT_WIDTH-1]}}, prod_cos_q};
    assign prod_sin_ext_w = {{(ROTATE_SUM_WIDTH-ROTATE_PRODUCT_WIDTH){prod_sin_q[ROTATE_PRODUCT_WIDTH-1]}}, prod_sin_q};
    assign rounded_base_w = round_abs_q[ROTATE_SUM_WIDTH-1:ROPE_ROUND_SHIFT];
    assign rounded_remainder_w = round_abs_q[ROPE_ROUND_SHIFT-1:0];
    assign round_increment_w = rounded_remainder_w[ROPE_ROUND_SHIFT-1] &&
                              ((|rounded_remainder_w[ROPE_ROUND_SHIFT-2:0]) || rounded_base_w[0]);
    assign rounded_abs_w = {1'b0, rounded_base_w} + {{ROUNDED_BASE_WIDTH{1'b0}}, round_increment_w};
    assign output_saturate_w = (!round_value_q[ROTATE_SUM_WIDTH-1] && (rounded_abs_w > {{(ROUNDED_BASE_WIDTH-7){1'b0}}, 8'd127})) ||
                              (round_value_q[ROTATE_SUM_WIDTH-1] && (rounded_abs_w > {{(ROUNDED_BASE_WIDTH-7){1'b0}}, 8'd128}));
    assign rounded_output_w = output_saturate_w ? (round_value_q[ROTATE_SUM_WIDTH-1] ? 8'h80 : 8'h7f) :
                              (round_value_q[ROTATE_SUM_WIDTH-1] ? (~rounded_abs_w[ACT_WIDTH-1:0] + 8'd1) : rounded_abs_w[ACT_WIDTH-1:0]);

    always @* begin
        selected_act_lane_w = {ACT_WIDTH{1'b0}};
        selected_pair_lane_w = {ACT_WIDTH{1'b0}};
        selected_scale_lane_w = {SCALE_WIDTH{1'b0}};
        selected_pair_scale_lane_w = {SCALE_WIDTH{1'b0}};
        selected_cos_lane_w = {TRIG_WIDTH{1'b0}};
        selected_sin_lane_w = {TRIG_WIDTH{1'b0}};
        for (lane_sel_idx = 0; lane_sel_idx < LANES; lane_sel_idx = lane_sel_idx + 1) begin
            selected_act_lane_w = selected_act_lane_w | ({ACT_WIDTH{lane_onehot_q[lane_sel_idx]}} & act_data_i[lane_sel_idx*ACT_WIDTH +: ACT_WIDTH]);
            selected_pair_lane_w = selected_pair_lane_w | ({ACT_WIDTH{lane_onehot_q[lane_sel_idx]}} & pair_data_i[lane_sel_idx*ACT_WIDTH +: ACT_WIDTH]);
            selected_scale_lane_w = selected_scale_lane_w | ({SCALE_WIDTH{lane_onehot_q[lane_sel_idx]}} & scale_data_i[lane_sel_idx*SCALE_WIDTH +: SCALE_WIDTH]);
            selected_pair_scale_lane_w = selected_pair_scale_lane_w | ({SCALE_WIDTH{lane_onehot_q[lane_sel_idx]}} & pair_scale_data_i[lane_sel_idx*SCALE_WIDTH +: SCALE_WIDTH]);
            selected_cos_lane_w = selected_cos_lane_w | ({TRIG_WIDTH{lane_onehot_q[lane_sel_idx]}} & cos_data_i[lane_sel_idx*TRIG_WIDTH +: TRIG_WIDTH]);
            selected_sin_lane_w = selected_sin_lane_w | ({TRIG_WIDTH{lane_onehot_q[lane_sel_idx]}} & sin_data_i[lane_sel_idx*TRIG_WIDTH +: TRIG_WIDTH]);
        end
    end

    always @* begin
        if (mul_negate_chunk_idx_q == 2'd1) begin
            mul_negate_chunk_w = ~mul_acc_q[31:16];
        end else if (mul_negate_chunk_idx_q == 2'd2) begin
            mul_negate_chunk_w = {8'd0, ~mul_acc_q[39:32]};
        end else begin
            mul_negate_chunk_w = ~mul_acc_q[15:0];
        end
    end

    always @* begin
        if (rotate_chunk_idx_q == 2'd1) begin
            rotate_lhs_chunk_w = prod_cos_ext_w[31:16];
            rotate_rhs_chunk_w = second_half_q ? prod_sin_ext_w[31:16] : ~prod_sin_ext_w[31:16];
        end else if (rotate_chunk_idx_q == 2'd2) begin
            rotate_lhs_chunk_w = {7'd0, prod_cos_ext_w[40:32]};
            rotate_rhs_chunk_w = second_half_q ? {7'd0, prod_sin_ext_w[40:32]} : {7'd0, ~prod_sin_ext_w[40:32]};
        end else begin
            rotate_lhs_chunk_w = prod_cos_ext_w[15:0];
            rotate_rhs_chunk_w = second_half_q ? prod_sin_ext_w[15:0] : ~prod_sin_ext_w[15:0];
        end
    end


    always @* begin
        if (round_negate_chunk_idx_q == 2'd1) begin
            round_negate_chunk_w = ~round_value_q[31:16];
        end else if (round_negate_chunk_idx_q == 2'd2) begin
            round_negate_chunk_w = {7'd0, ~round_value_q[40:32]};
        end else begin
            round_negate_chunk_w = ~round_value_q[15:0];
        end
    end

    always @* begin
        if (mul_chunk_idx_q == 2'd1) begin
            mul_acc_chunk_w = mul_acc_q[31:16];
            mul_multiplicand_chunk_w = mul_multiplicand_q[31:16];
        end else if (mul_chunk_idx_q == 2'd2) begin
            mul_acc_chunk_w = {8'd0, mul_acc_q[39:32]};
            mul_multiplicand_chunk_w = {8'd0, mul_multiplicand_q[39:32]};
        end else begin
            mul_acc_chunk_w = mul_acc_q[15:0];
            mul_multiplicand_chunk_w = mul_multiplicand_q[15:0];
        end
    end

    always @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            state_q <= ST_IDLE;
            lane_idx_q <= {LANE_INDEX_WIDTH{1'b0}};
            lane_onehot_q <= {{(LANES-1){1'b0}}, 1'b1};
            second_half_q <= 1'b0;
            act_lane_q <= {ACT_WIDTH{1'b0}};
            pair_lane_q <= {ACT_WIDTH{1'b0}};
            scale_lane_q <= {SCALE_WIDTH{1'b0}};
            pair_scale_lane_q <= {SCALE_WIDTH{1'b0}};
            cos_lane_q <= {TRIG_WIDTH{1'b0}};
            sin_lane_q <= {TRIG_WIDTH{1'b0}};
            act_scaled_q <= {PRE_ROTATE_WIDTH{1'b0}};
            pair_scaled_q <= {PRE_ROTATE_WIDTH{1'b0}};
            prod_cos_q <= {ROTATE_PRODUCT_WIDTH{1'b0}};
            prod_sin_q <= {ROTATE_PRODUCT_WIDTH{1'b0}};
            round_value_q <= {ROTATE_SUM_WIDTH{1'b0}};
            round_abs_q <= {ROTATE_SUM_WIDTH{1'b0}};
            rounded_output_q <= {ACT_WIDTH{1'b0}};
            round_saturate_q <= 1'b0;
            mul_acc_q <= {MUL_WIDTH{1'b0}};
            mul_multiplicand_q <= {MUL_WIDTH{1'b0}};
            mul_multiplier_q <= 16'd0;
            mul_count_q <= 5'd0;
            mul_chunk_idx_q <= 2'd0;
            mul_carry_q <= 1'b0;
            mul_sign_q <= 1'b0;
            mul_dest_q <= MUL_DEST_ACT_SCALE;
            mul_negate_chunk_idx_q <= 2'd0;
            mul_negate_carry_q <= 1'b0;
            rotate_chunk_idx_q <= 2'd0;
            rotate_carry_q <= 1'b0;
            round_negate_chunk_idx_q <= 2'd0;
            round_negate_carry_q <= 1'b0;
            out_data_q <= {LANES*ACT_WIDTH{1'b0}};
            out_valid_q <= 1'b0;
            saturation_seen_q <= 1'b0;
        end else begin
            if (out_valid_q && out_ready_i) begin
                out_valid_q <= 1'b0;
                state_q <= ST_IDLE;
            end

            case (state_q)
                ST_IDLE: begin
                    if (start_valid_i && start_ready_o) begin
                        lane_idx_q <= {LANE_INDEX_WIDTH{1'b0}};
                        lane_onehot_q <= {{(LANES-1){1'b0}}, 1'b1};
                        out_data_q <= {LANES*ACT_WIDTH{1'b0}};
                        saturation_seen_q <= 1'b0;
                        state_q <= ST_LOAD;
                    end
                end

                ST_LOAD: begin
                    if (beat_valid_i && beat_ready_o) begin
                        second_half_q <= second_half_i;
                        state_q <= ST_SELECT_LANE;
                    end
                end

                ST_SELECT_LANE: begin
                    act_lane_q <= selected_act_lane_w;
                    pair_lane_q <= selected_pair_lane_w;
                    scale_lane_q <= selected_scale_lane_w;
                    pair_scale_lane_q <= selected_pair_scale_lane_w;
                    cos_lane_q <= selected_cos_lane_w;
                    sin_lane_q <= selected_sin_lane_w;
                    state_q <= ST_SETUP_ACT_SCALE;
                end

                ST_SETUP_ACT_SCALE: begin
                    mul_acc_q <= {MUL_WIDTH{1'b0}};
                    mul_multiplicand_q <= {{(MUL_WIDTH-16){1'b0}}, act_lane_abs_w};
                    mul_multiplier_q <= scale_lane_abs_w;
                    mul_count_q <= 5'd16;
                    mul_chunk_idx_q <= 2'd0;
                    mul_carry_q <= 1'b0;
                    mul_sign_q <= act_lane_w[ACT_WIDTH-1] ^ scale_lane_w[SCALE_WIDTH-1];
                    mul_dest_q <= MUL_DEST_ACT_SCALE;
                    state_q <= ST_MUL_ACT_SCALE;
                end

                ST_MUL_ACT_SCALE: begin
                    mul_acc_q <= mul_acc_next_w;
                    if (mul_chunk_idx_q == 2'd2) begin
                        mul_chunk_idx_q <= 2'd0;
                        mul_carry_q <= 1'b0;
                        mul_multiplicand_q <= {mul_multiplicand_q[MUL_WIDTH-2:0], 1'b0};
                        mul_multiplier_q <= {1'b0, mul_multiplier_q[15:1]};
                        if (mul_count_q == 5'd1) begin
                            mul_count_q <= 5'd0;
                            if (mul_sign_q) begin
                                mul_negate_chunk_idx_q <= 2'd0;
                                mul_negate_carry_q <= 1'b1;
                                state_q <= ST_NEGATE_PRODUCT;
                            end else begin
                                state_q <= ST_FINISH_ACT_SCALE;
                            end
                        end else begin
                            mul_count_q <= mul_count_q - 5'd1;
                        end
                    end else begin
                        mul_chunk_idx_q <= mul_chunk_idx_q + 2'd1;
                        mul_carry_q <= mul_chunk_sum_w[16];
                    end
                end

                ST_FINISH_ACT_SCALE: begin
                    act_scaled_q <= mul_acc_q[PRE_ROTATE_WIDTH-1:0];
                    state_q <= ST_SETUP_PAIR_SCALE;
                end

                ST_SETUP_PAIR_SCALE: begin
                    mul_acc_q <= {MUL_WIDTH{1'b0}};
                    mul_multiplicand_q <= {{(MUL_WIDTH-16){1'b0}}, pair_lane_abs_w};
                    mul_multiplier_q <= pair_scale_lane_abs_w;
                    mul_count_q <= 5'd16;
                    mul_chunk_idx_q <= 2'd0;
                    mul_carry_q <= 1'b0;
                    mul_sign_q <= pair_lane_w[ACT_WIDTH-1] ^ pair_scale_lane_w[SCALE_WIDTH-1];
                    mul_dest_q <= MUL_DEST_PAIR_SCALE;
                    state_q <= ST_MUL_PAIR_SCALE;
                end

                ST_MUL_PAIR_SCALE: begin
                    mul_acc_q <= mul_acc_next_w;
                    if (mul_chunk_idx_q == 2'd2) begin
                        mul_chunk_idx_q <= 2'd0;
                        mul_carry_q <= 1'b0;
                        mul_multiplicand_q <= {mul_multiplicand_q[MUL_WIDTH-2:0], 1'b0};
                        mul_multiplier_q <= {1'b0, mul_multiplier_q[15:1]};
                        if (mul_count_q == 5'd1) begin
                            mul_count_q <= 5'd0;
                            if (mul_sign_q) begin
                                mul_negate_chunk_idx_q <= 2'd0;
                                mul_negate_carry_q <= 1'b1;
                                state_q <= ST_NEGATE_PRODUCT;
                            end else begin
                                state_q <= ST_FINISH_PAIR_SCALE;
                            end
                        end else begin
                            mul_count_q <= mul_count_q - 5'd1;
                        end
                    end else begin
                        mul_chunk_idx_q <= mul_chunk_idx_q + 2'd1;
                        mul_carry_q <= mul_chunk_sum_w[16];
                    end
                end

                ST_FINISH_PAIR_SCALE: begin
                    pair_scaled_q <= mul_acc_q[PRE_ROTATE_WIDTH-1:0];
                    state_q <= ST_SETUP_COS;
                end

                ST_SETUP_COS: begin
                    mul_acc_q <= {MUL_WIDTH{1'b0}};
                    mul_multiplicand_q <= {{(MUL_WIDTH-PRE_ROTATE_WIDTH){1'b0}}, act_scaled_abs_w};
                    mul_multiplier_q <= cos_lane_abs_w;
                    mul_count_q <= 5'd16;
                    mul_chunk_idx_q <= 2'd0;
                    mul_carry_q <= 1'b0;
                    mul_sign_q <= act_scaled_q[PRE_ROTATE_WIDTH-1] ^ cos_lane_w[TRIG_WIDTH-1];
                    mul_dest_q <= MUL_DEST_COS;
                    state_q <= ST_MUL_COS;
                end

                ST_MUL_COS: begin
                    mul_acc_q <= mul_acc_next_w;
                    if (mul_chunk_idx_q == 2'd2) begin
                        mul_chunk_idx_q <= 2'd0;
                        mul_carry_q <= 1'b0;
                        mul_multiplicand_q <= {mul_multiplicand_q[MUL_WIDTH-2:0], 1'b0};
                        mul_multiplier_q <= {1'b0, mul_multiplier_q[15:1]};
                        if (mul_count_q == 5'd1) begin
                            mul_count_q <= 5'd0;
                            if (mul_sign_q) begin
                                mul_negate_chunk_idx_q <= 2'd0;
                                mul_negate_carry_q <= 1'b1;
                                state_q <= ST_NEGATE_PRODUCT;
                            end else begin
                                state_q <= ST_FINISH_COS;
                            end
                        end else begin
                            mul_count_q <= mul_count_q - 5'd1;
                        end
                    end else begin
                        mul_chunk_idx_q <= mul_chunk_idx_q + 2'd1;
                        mul_carry_q <= mul_chunk_sum_w[16];
                    end
                end

                ST_FINISH_COS: begin
                    prod_cos_q <= mul_acc_q[ROTATE_PRODUCT_WIDTH-1:0];
                    state_q <= ST_SETUP_SIN;
                end

                ST_SETUP_SIN: begin
                    mul_acc_q <= {MUL_WIDTH{1'b0}};
                    mul_multiplicand_q <= {{(MUL_WIDTH-PRE_ROTATE_WIDTH){1'b0}}, pair_scaled_abs_w};
                    mul_multiplier_q <= sin_lane_abs_w;
                    mul_count_q <= 5'd16;
                    mul_chunk_idx_q <= 2'd0;
                    mul_carry_q <= 1'b0;
                    mul_sign_q <= pair_scaled_q[PRE_ROTATE_WIDTH-1] ^ sin_lane_w[TRIG_WIDTH-1];
                    mul_dest_q <= MUL_DEST_SIN;
                    state_q <= ST_MUL_SIN;
                end

                ST_MUL_SIN: begin
                    mul_acc_q <= mul_acc_next_w;
                    if (mul_chunk_idx_q == 2'd2) begin
                        mul_chunk_idx_q <= 2'd0;
                        mul_carry_q <= 1'b0;
                        mul_multiplicand_q <= {mul_multiplicand_q[MUL_WIDTH-2:0], 1'b0};
                        mul_multiplier_q <= {1'b0, mul_multiplier_q[15:1]};
                        if (mul_count_q == 5'd1) begin
                            mul_count_q <= 5'd0;
                            if (mul_sign_q) begin
                                mul_negate_chunk_idx_q <= 2'd0;
                                mul_negate_carry_q <= 1'b1;
                                state_q <= ST_NEGATE_PRODUCT;
                            end else begin
                                state_q <= ST_FINISH_SIN;
                            end
                        end else begin
                            mul_count_q <= mul_count_q - 5'd1;
                        end
                    end else begin
                        mul_chunk_idx_q <= mul_chunk_idx_q + 2'd1;
                        mul_carry_q <= mul_chunk_sum_w[16];
                    end
                end

                ST_FINISH_SIN: begin
                    prod_sin_q <= mul_acc_q[ROTATE_PRODUCT_WIDTH-1:0];
                    state_q <= ST_ROUND_PREP;
                end

                ST_NEGATE_PRODUCT: begin
                    mul_acc_q <= mul_negate_next_w;
                    if (mul_negate_chunk_idx_q == 2'd2) begin
                        mul_negate_chunk_idx_q <= 2'd0;
                        mul_negate_carry_q <= 1'b0;
                        case (mul_dest_q)
                            MUL_DEST_ACT_SCALE: state_q <= ST_FINISH_ACT_SCALE;
                            MUL_DEST_PAIR_SCALE: state_q <= ST_FINISH_PAIR_SCALE;
                            MUL_DEST_COS: state_q <= ST_FINISH_COS;
                            default: state_q <= ST_FINISH_SIN;
                        endcase
                    end else begin
                        mul_negate_chunk_idx_q <= mul_negate_chunk_idx_q + 2'd1;
                        mul_negate_carry_q <= mul_negate_sum_w[16];
                    end
                end

                ST_ROUND_PREP: begin
                    round_value_q <= {ROTATE_SUM_WIDTH{1'b0}};
                    rotate_chunk_idx_q <= 2'd0;
                    rotate_carry_q <= !second_half_q;
                    state_q <= ST_ROTATE_CHUNK;
                end

                ST_ROTATE_CHUNK: begin
                    round_value_q <= rotate_next_w;
                    if (rotate_chunk_idx_q == 2'd2) begin
                        rotate_chunk_idx_q <= 2'd0;
                        rotate_carry_q <= 1'b0;
                        if (rotate_next_w[ROTATE_SUM_WIDTH-1]) begin
                            round_negate_chunk_idx_q <= 2'd0;
                            round_negate_carry_q <= 1'b1;
                            state_q <= ST_ROUND_HIGH;
                        end else begin
                            round_abs_q <= rotate_next_w;
                            state_q <= ST_ROUND_RESULT;
                        end
                    end else begin
                        rotate_chunk_idx_q <= rotate_chunk_idx_q + 2'd1;
                        rotate_carry_q <= rotate_sum_w[16];
                    end
                end

                ST_ROUND_HIGH: begin
                    round_abs_q <= round_negate_next_w;
                    if (round_negate_chunk_idx_q == 2'd2) begin
                        round_negate_chunk_idx_q <= 2'd0;
                        round_negate_carry_q <= 1'b0;
                        state_q <= ST_ROUND_RESULT;
                    end else begin
                        round_negate_chunk_idx_q <= round_negate_chunk_idx_q + 2'd1;
                        round_negate_carry_q <= round_negate_sum_w[16];
                    end
                end

                ST_ROUND_RESULT: begin
                    rounded_output_q <= rounded_output_w;
                    round_saturate_q <= output_saturate_w;
                    state_q <= ST_ROUND;
                end

                ST_ROUND: begin
                    out_data_q[lane_idx_q*ACT_WIDTH +: ACT_WIDTH] <= rounded_output_q;
                    saturation_seen_q <= saturation_seen_q | round_saturate_q;
                    if (lane_idx_q == LAST_LANE) begin
                        out_valid_q <= 1'b1;
                        state_q <= ST_DONE;
                    end else begin
                        lane_idx_q <= lane_idx_q + {{(LANE_INDEX_WIDTH-1){1'b0}}, 1'b1};
                        lane_onehot_q <= {lane_onehot_q[LANES-2:0], 1'b0};
                        state_q <= ST_SELECT_LANE;
                    end
                end

                ST_DONE: begin
                end

                default: begin
                    state_q <= ST_IDLE;
                    out_valid_q <= 1'b0;
                end
            endcase
            if (clear_i) begin
                state_q <= ST_IDLE;
                out_valid_q <= 1'b0;
                saturation_seen_q <= 1'b0;
            end
        end
    end
endmodule

`default_nettype wire
