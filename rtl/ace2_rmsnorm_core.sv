`default_nettype none

module ace2_rmsnorm_core #(
    parameter integer HIDDEN_SIZE = 896,
    parameter integer LANES = 16,
    parameter integer ACT_WIDTH = 8,
    parameter integer GAIN_WIDTH = 16,
    parameter integer ACC_WIDTH = 48,
    parameter integer INV_RMS_FRAC = 30,
    parameter integer GAIN_FRAC = 8
) (
    input  wire                              clk_i,
    input  wire                              rst_ni,
    input  wire                              clear_i,

    input  wire                              start_valid_i,
    output wire                              start_ready_o,

    input  wire                              in_valid_i,
    output wire                              in_ready_o,
    input  wire [LANES*ACT_WIDTH-1:0]       in_data_i,

    input  wire                              gain_valid_i,
    output wire                              gain_ready_o,
    input  wire [LANES*GAIN_WIDTH-1:0]      gain_data_i,

    input  wire                              scale_act_valid_i,
    output wire                              scale_act_ready_o,
    input  wire [LANES*ACT_WIDTH-1:0]        scale_act_data_i,

    output wire                              out_valid_o,
    input  wire                              out_ready_i,
    output wire [LANES*ACT_WIDTH-1:0]       out_data_o,

    output wire                              done_valid_o,
    input  wire                              done_ready_i,
    output wire [ACC_WIDTH-1:0]             sumsq_o,
    output wire [INV_RMS_FRAC+1:0]          inv_rms_q30_o,
    output wire                              saturation_seen_o
);
    localparam integer BEATS = HIDDEN_SIZE / LANES;
    localparam integer BEAT_INDEX_WIDTH = (BEATS <= 1) ? 1 : $clog2(BEATS + 1);
    localparam integer LANE_INDEX_WIDTH = (LANES <= 1) ? 1 : $clog2(LANES);
    localparam [ACC_WIDTH-1:0] HIDDEN_HALF_ACC = ACC_WIDTH'(HIDDEN_SIZE) >> 1;
    localparam [BEAT_INDEX_WIDTH-1:0] BEATS_VALUE = BEAT_INDEX_WIDTH'(BEATS);
    localparam [BEAT_INDEX_WIDTH-1:0] LAST_BEAT = BEAT_INDEX_WIDTH'(BEATS - 1);
    localparam [LANE_INDEX_WIDTH-1:0] LAST_LANE = LANE_INDEX_WIDTH'(LANES - 1);
    localparam [LANE_INDEX_WIDTH-1:0] LANE_BEFORE_LAST = (LANES <= 1) ? {LANE_INDEX_WIDTH{1'b0}} : LANE_INDEX_WIDTH'(LANES - 2);
    localparam [3:0] ST_IDLE        = 4'd0;
    localparam [3:0] ST_COLLECT     = 4'd1;
    localparam [3:0] ST_MEAN_DIV    = 4'd2;
    localparam [3:0] ST_SQRT        = 4'd3;
    localparam [3:0] ST_SQRT_DECIDE = 4'd4;
    localparam [3:0] ST_INV_DIV     = 4'd5;
    localparam [3:0] ST_SCALE       = 4'd6;
    localparam [3:0] ST_SCALE_PREP  = 4'd7;
    localparam [3:0] ST_SCALE_MUL   = 4'd8;
    localparam [3:0] ST_SCALE_MUL_HI = 4'd9;
    localparam [3:0] ST_SCALE_ROUND = 4'd10;
    localparam [3:0] ST_DONE        = 4'd11;
    localparam integer DIV_COUNT_WIDTH = (ACC_WIDTH <= 1) ? 1 : $clog2(ACC_WIDTH + 1);
    localparam [DIV_COUNT_WIDTH-1:0] DIV_COUNT_INIT = DIV_COUNT_WIDTH'(ACC_WIDTH);
    localparam integer DIV_QUOT_WIDTH = INV_RMS_FRAC + 1;
    localparam integer DIV_QUOT_REG_WIDTH = DIV_QUOT_WIDTH - 1;
    localparam integer DIV_REM_WIDTH = 11;
    localparam integer SCALE_PRODUCT_SHIFT = INV_RMS_FRAC + GAIN_FRAC;
    localparam integer SCALE_BASE_WIDTH = 64 - SCALE_PRODUCT_SHIFT;
    localparam [5:0] SCALE_MUL_COUNT_INIT = 6'd32;

    reg [3:0] state_q;
    reg [BEAT_INDEX_WIDTH-1:0] collect_idx_q;
    reg [BEAT_INDEX_WIDTH-1:0] scale_idx_q;
    reg [LANE_INDEX_WIDTH-1:0] lane_idx_q;
    reg [LANES*ACT_WIDTH-1:0] collect_beat_q;
    reg [LANES*ACT_WIDTH-1:0] out_work_q;
    reg [ACC_WIDTH-1:0] sumsq_q;
    reg [ACC_WIDTH-1:0] mean_square_q;
    reg [2*ACT_WIDTH-1:0] collect_square_q;
    reg [15:0] rms_candidate_q;
    reg [31:0] rms_square_q;
    reg [INV_RMS_FRAC+1:0] inv_rms_q;
    reg [ACC_WIDTH-1:0] div_dividend_q;
    reg [DIV_REM_WIDTH-1:0] div_divisor_q;
    reg [DIV_QUOT_REG_WIDTH-1:0] div_quotient_q;
    /* verilator lint_off UNUSED */
    reg [DIV_REM_WIDTH-1:0] div_remainder_q;
    /* verilator lint_on UNUSED */
    reg [DIV_COUNT_WIDTH-1:0] div_count_q;
    reg out_valid_q;
    reg done_valid_q;
    reg saturation_seen_q;
    reg sqrt_done_q;
    reg collect_active_q;
    reg collect_square_valid_q;
    reg lane_last_q;
    reg scale_active_q;
    reg scale_product_sign_q;
    reg scale_round_active_q;
    reg signed [ACT_WIDTH-1:0] scale_act_q;
    reg signed [GAIN_WIDTH-1:0] scale_gain_q;
    reg [63:0] scale_mul_acc_q;
    reg [63:0] scale_mul_multiplicand_q;
    reg [31:0] scale_mul_multiplier_q;
    reg [31:0] scale_mul_addend_hi_q;
    reg scale_mul_carry_q;
    reg [5:0] scale_mul_count_q;

    assign start_ready_o = (state_q == ST_IDLE) && !done_valid_q;
    assign in_ready_o = (state_q == ST_COLLECT) && !collect_active_q;
    assign scale_act_ready_o = (state_q == ST_SCALE) && (scale_idx_q < BEATS_VALUE) && !out_valid_q && !scale_active_q;
    assign gain_ready_o = scale_act_ready_o;
    assign out_valid_o = out_valid_q;
    assign out_data_o = out_work_q;
    assign done_valid_o = done_valid_q;
    assign sumsq_o = sumsq_q;
    assign inv_rms_q30_o = inv_rms_q;
    assign saturation_seen_o = saturation_seen_q;

    wire [ACC_WIDTH-1:0] next_sumsq_w;
    wire [ACT_WIDTH:0] scaled_lane_w;
    wire [SCALE_BASE_WIDTH-1:0] scale_round_base_w;
    wire [SCALE_PRODUCT_SHIFT-1:0] scale_round_remainder_w;
    wire [SCALE_BASE_WIDTH:0] scale_rounded_base_w;
    wire scale_round_increment_w;
    wire scale_positive_saturate_w;
    wire scale_negative_saturate_w;
    wire [ACT_WIDTH-1:0] scale_negative_value_w;
    wire [ACT_WIDTH-1:0] collect_lane_raw_w;
    wire signed [ACT_WIDTH-1:0] collect_lane_w;
    wire signed [2*ACT_WIDTH-1:0] collect_lane_square_w;
    wire signed [ACT_WIDTH-1:0] scale_act_w;
    wire signed [GAIN_WIDTH-1:0] scale_gain_w;
    wire [ACT_WIDTH-1:0] scale_act_abs_w;
    wire [GAIN_WIDTH-1:0] scale_gain_abs_w;
    wire [ACT_WIDTH+GAIN_WIDTH-1:0] scale_lane_product_w;
    wire scale_product_sign_w;
    wire [DIV_REM_WIDTH-1:0] div_remainder_shift_w;
    wire div_subtract_w;
    wire sqrt_done_w;
    wire [DIV_REM_WIDTH-1:0] div_remainder_next_w;
    wire [DIV_QUOT_WIDTH-1:0] div_quotient_next_w;
    wire [ACC_WIDTH-1:0] div_dividend_next_w;
    wire [31:0] rms_square_next_w;
    wire [32:0] scale_mul_low_sum_w;
    wire [31:0] scale_mul_high_sum_w;
    reg [LANES*ACT_WIDTH-1:0] scaled_work_w;

    assign collect_lane_raw_w = collect_beat_q[ACT_WIDTH-1:0];
    assign collect_lane_w = collect_lane_raw_w;
    assign collect_lane_square_w = collect_lane_w * collect_lane_w;
    assign next_sumsq_w = sumsq_q + {{(ACC_WIDTH-2*ACT_WIDTH){1'b0}}, collect_square_q};
    assign scale_act_w = scale_act_data_i[lane_idx_q*ACT_WIDTH +: ACT_WIDTH];
    assign scale_gain_w = gain_data_i[lane_idx_q*GAIN_WIDTH +: GAIN_WIDTH];
    assign scale_act_abs_w = scale_act_q[ACT_WIDTH-1] ? (~scale_act_q + {{(ACT_WIDTH-1){1'b0}}, 1'b1}) : scale_act_q;
    assign scale_gain_abs_w = scale_gain_q[GAIN_WIDTH-1] ? (~scale_gain_q + {{(GAIN_WIDTH-1){1'b0}}, 1'b1}) : scale_gain_q;
    assign scale_lane_product_w = scale_act_abs_w * scale_gain_abs_w;
    assign scale_product_sign_w = scale_act_q[ACT_WIDTH-1] ^ scale_gain_q[GAIN_WIDTH-1];
    assign scale_round_base_w = scale_mul_acc_q[63:SCALE_PRODUCT_SHIFT];
    assign scale_round_remainder_w = scale_mul_acc_q[SCALE_PRODUCT_SHIFT-1:0];
    assign scale_round_increment_w = scale_round_remainder_w[SCALE_PRODUCT_SHIFT-1] &&
                                     ((|scale_round_remainder_w[SCALE_PRODUCT_SHIFT-2:0]) || scale_round_base_w[0]);
    assign scale_rounded_base_w = {1'b0, scale_round_base_w} + {{SCALE_BASE_WIDTH{1'b0}}, scale_round_increment_w};
    assign scale_positive_saturate_w = scale_rounded_base_w > {{(SCALE_BASE_WIDTH-7){1'b0}}, 8'd127};
    assign scale_negative_saturate_w = scale_rounded_base_w > {{(SCALE_BASE_WIDTH-7){1'b0}}, 8'd128};
    assign scale_negative_value_w = ~scale_rounded_base_w[ACT_WIDTH-1:0] + {{(ACT_WIDTH-1){1'b0}}, 1'b1};
    assign scaled_lane_w = scale_product_sign_q ?
                           (scale_negative_saturate_w ? {1'b1, 8'h80} : {1'b0, scale_negative_value_w}) :
                           (scale_positive_saturate_w ? {1'b1, 8'h7f} : {1'b0, scale_rounded_base_w[ACT_WIDTH-1:0]});
    assign div_remainder_shift_w = {div_remainder_q[DIV_REM_WIDTH-2:0], div_dividend_q[ACC_WIDTH-1]};
    assign div_subtract_w = (div_remainder_shift_w >= div_divisor_q);
    assign sqrt_done_w = ({{(ACC_WIDTH-32){1'b0}}, rms_square_q} >= mean_square_q) || (rms_candidate_q == 16'h00ff);
    assign div_remainder_next_w = div_subtract_w ? (div_remainder_shift_w - div_divisor_q) : div_remainder_shift_w;
    assign div_quotient_next_w = {div_quotient_q, div_subtract_w};
    assign div_dividend_next_w = {div_dividend_q[ACC_WIDTH-2:0], 1'b0};
    assign rms_square_next_w = rms_square_q + {15'd0, rms_candidate_q, 1'b0} + 32'd1;
    assign scale_mul_low_sum_w = {1'b0, scale_mul_acc_q[31:0]} +
                                 (scale_mul_multiplier_q[0] ? {1'b0, scale_mul_multiplicand_q[31:0]} : 33'd0);
    assign scale_mul_high_sum_w = scale_mul_acc_q[63:32] +
                                  scale_mul_addend_hi_q +
                                  {31'd0, scale_mul_carry_q};

    always @* begin
        scaled_work_w = out_work_q;
        scaled_work_w[lane_idx_q*ACT_WIDTH +: ACT_WIDTH] = scaled_lane_w[ACT_WIDTH-1:0];
    end

    always @(posedge clk_i) begin
        if (in_valid_i && in_ready_o) begin
            collect_beat_q <= in_data_i;
        end else if ((state_q == ST_COLLECT) && collect_active_q &&
                     !collect_square_valid_q) begin
            collect_beat_q <= collect_beat_q >> ACT_WIDTH;
        end
    end

    always @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            div_dividend_q <= {ACC_WIDTH{1'b0}};
        end else begin
            case (state_q)
                ST_IDLE: begin
                    if (start_valid_i && start_ready_o) begin
                        div_dividend_q <= {ACC_WIDTH{1'b0}};
                    end
                end
                ST_COLLECT: begin
                    if (collect_active_q && collect_square_valid_q &&
                        lane_last_q && (collect_idx_q == LAST_BEAT)) begin
                        div_dividend_q <= next_sumsq_w + HIDDEN_HALF_ACC;
                    end
                end
                ST_MEAN_DIV,
                ST_INV_DIV:
                    div_dividend_q <= div_dividend_next_w;
                ST_SQRT_DECIDE: begin
                    if (sqrt_done_q) begin
                        div_dividend_q <= ACC_WIDTH'(1) << INV_RMS_FRAC;
                    end
                end
                default: begin
                end
            endcase
        end
    end

    always @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            state_q <= ST_IDLE;
            collect_idx_q <= {BEAT_INDEX_WIDTH{1'b0}};
            scale_idx_q <= {BEAT_INDEX_WIDTH{1'b0}};
            lane_idx_q <= {LANE_INDEX_WIDTH{1'b0}};
            out_work_q <= {LANES*ACT_WIDTH{1'b0}};
            sumsq_q <= {ACC_WIDTH{1'b0}};
            mean_square_q <= {ACC_WIDTH{1'b0}};
            collect_square_q <= {2*ACT_WIDTH{1'b0}};
            rms_candidate_q <= 16'd1;
            rms_square_q <= 32'd1;
            inv_rms_q <= {(INV_RMS_FRAC+2){1'b0}};
            div_divisor_q <= {{(DIV_REM_WIDTH-1){1'b0}}, 1'b1};
            div_quotient_q <= {DIV_QUOT_REG_WIDTH{1'b0}};
            div_remainder_q <= {DIV_REM_WIDTH{1'b0}};
            div_count_q <= {DIV_COUNT_WIDTH{1'b0}};
            out_valid_q <= 1'b0;
            done_valid_q <= 1'b0;
            saturation_seen_q <= 1'b0;
            sqrt_done_q <= 1'b0;
            collect_active_q <= 1'b0;
            collect_square_valid_q <= 1'b0;
            lane_last_q <= (LANES <= 1) ? 1'b1 : 1'b0;
            scale_active_q <= 1'b0;
            scale_product_sign_q <= 1'b0;
            scale_round_active_q <= 1'b0;
            scale_act_q <= {ACT_WIDTH{1'b0}};
            scale_gain_q <= {GAIN_WIDTH{1'b0}};
            scale_mul_acc_q <= 64'd0;
            scale_mul_multiplicand_q <= 64'd0;
            scale_mul_multiplier_q <= 32'd0;
            scale_mul_addend_hi_q <= 32'd0;
            scale_mul_carry_q <= 1'b0;
            scale_mul_count_q <= 6'd0;
        end else begin
                if (out_valid_q && out_ready_i) begin
                    out_valid_q <= 1'b0;
                    if ((state_q == ST_SCALE) && (scale_idx_q == BEATS_VALUE)) begin
                        state_q <= ST_DONE;
                        done_valid_q <= 1'b1;
                    end
                end

                if (scale_round_active_q) begin
                    out_work_q <= scaled_work_w;
                    saturation_seen_q <= saturation_seen_q | scaled_lane_w[ACT_WIDTH];
                    if (lane_last_q) begin
                        out_valid_q <= 1'b1;
                        lane_idx_q <= {LANE_INDEX_WIDTH{1'b0}};
                        lane_last_q <= (LANES <= 1) ? 1'b1 : 1'b0;
                        scale_active_q <= 1'b0;
                    end else begin
                        lane_idx_q <= lane_idx_q + {{(LANE_INDEX_WIDTH-1){1'b0}}, 1'b1};
                        lane_last_q <= (lane_idx_q == LANE_BEFORE_LAST);
                    end
                    scale_round_active_q <= 1'b0;
                    state_q <= ST_SCALE;
                end

                case (state_q)
                ST_IDLE: begin
                    if (start_valid_i && start_ready_o) begin
                        collect_idx_q <= {BEAT_INDEX_WIDTH{1'b0}};
                        scale_idx_q <= {BEAT_INDEX_WIDTH{1'b0}};
                        lane_idx_q <= {LANE_INDEX_WIDTH{1'b0}};
                        lane_last_q <= (LANES <= 1) ? 1'b1 : 1'b0;
                        out_work_q <= {LANES*ACT_WIDTH{1'b0}};
                        sumsq_q <= {ACC_WIDTH{1'b0}};
                        mean_square_q <= {ACC_WIDTH{1'b0}};
                        collect_square_q <= {2*ACT_WIDTH{1'b0}};
                        rms_candidate_q <= 16'd1;
                        rms_square_q <= 32'd1;
                        inv_rms_q <= {(INV_RMS_FRAC+2){1'b0}};
                        div_divisor_q <= {{(DIV_REM_WIDTH-1){1'b0}}, 1'b1};
                        div_quotient_q <= {DIV_QUOT_REG_WIDTH{1'b0}};
                        div_remainder_q <= {DIV_REM_WIDTH{1'b0}};
                        div_count_q <= {DIV_COUNT_WIDTH{1'b0}};
                        out_valid_q <= 1'b0;
                        done_valid_q <= 1'b0;
                        saturation_seen_q <= 1'b0;
                        sqrt_done_q <= 1'b0;
                        collect_active_q <= 1'b0;
                        collect_square_valid_q <= 1'b0;
                        scale_active_q <= 1'b0;
                        scale_product_sign_q <= 1'b0;
                        scale_round_active_q <= 1'b0;
                        scale_act_q <= {ACT_WIDTH{1'b0}};
                        scale_gain_q <= {GAIN_WIDTH{1'b0}};
                        scale_mul_acc_q <= 64'd0;
                        scale_mul_multiplicand_q <= 64'd0;
                        scale_mul_multiplier_q <= 32'd0;
                        scale_mul_addend_hi_q <= 32'd0;
                        scale_mul_carry_q <= 1'b0;
                        scale_mul_count_q <= 6'd0;
                        state_q <= ST_COLLECT;
                    end
                end

                ST_COLLECT: begin
                    if (in_valid_i && in_ready_o) begin
                        lane_idx_q <= {LANE_INDEX_WIDTH{1'b0}};
                        lane_last_q <= (LANES <= 1) ? 1'b1 : 1'b0;
                        collect_active_q <= 1'b1;
                        collect_square_valid_q <= 1'b0;
                    end else if (collect_active_q) begin
                        if (!collect_square_valid_q) begin
                            collect_square_q <= collect_lane_square_w[2*ACT_WIDTH-1:0];
                            collect_square_valid_q <= 1'b1;
                        end else begin
                            sumsq_q <= next_sumsq_w;
                            collect_square_valid_q <= 1'b0;
                            if (lane_last_q) begin
                                collect_active_q <= 1'b0;
                                lane_idx_q <= {LANE_INDEX_WIDTH{1'b0}};
                                lane_last_q <= (LANES <= 1) ? 1'b1 : 1'b0;
                                if (collect_idx_q == LAST_BEAT) begin
                                    div_divisor_q <= DIV_REM_WIDTH'(HIDDEN_SIZE);
                                    div_quotient_q <= {DIV_QUOT_REG_WIDTH{1'b0}};
                                    div_remainder_q <= {DIV_REM_WIDTH{1'b0}};
                                    div_count_q <= DIV_COUNT_INIT;
                                    collect_idx_q <= {BEAT_INDEX_WIDTH{1'b0}};
                                    scale_idx_q <= {BEAT_INDEX_WIDTH{1'b0}};
                                    state_q <= ST_MEAN_DIV;
                                end else begin
                                    collect_idx_q <= collect_idx_q + {{(BEAT_INDEX_WIDTH-1){1'b0}}, 1'b1};
                                end
                            end else begin
                                lane_idx_q <= lane_idx_q + {{(LANE_INDEX_WIDTH-1){1'b0}}, 1'b1};
                                lane_last_q <= (lane_idx_q == LANE_BEFORE_LAST);
                            end
                        end
                    end
                end

                ST_MEAN_DIV: begin
                    div_quotient_q <= div_quotient_next_w[DIV_QUOT_REG_WIDTH-1:0];
                    div_remainder_q <= div_remainder_next_w;
                    if (div_count_q == {{(DIV_COUNT_WIDTH-1){1'b0}}, 1'b1}) begin
                        mean_square_q <= {{(ACC_WIDTH-DIV_QUOT_WIDTH){1'b0}}, div_quotient_next_w};
                        rms_candidate_q <= 16'd1;
                        rms_square_q <= 32'd1;
                        div_count_q <= {DIV_COUNT_WIDTH{1'b0}};
                        state_q <= ST_SQRT;
                    end else begin
                        div_count_q <= div_count_q - {{(DIV_COUNT_WIDTH-1){1'b0}}, 1'b1};
                    end
                end

                ST_SQRT: begin
                    sqrt_done_q <= sqrt_done_w;
                    state_q <= ST_SQRT_DECIDE;
                end

                ST_SQRT_DECIDE: begin
                    if (sqrt_done_q) begin
                        div_divisor_q <= rms_candidate_q[DIV_REM_WIDTH-1:0];
                        div_quotient_q <= {DIV_QUOT_REG_WIDTH{1'b0}};
                        div_remainder_q <= {DIV_REM_WIDTH{1'b0}};
                        div_count_q <= DIV_COUNT_INIT;
                        state_q <= ST_INV_DIV;
                    end else begin
                        rms_candidate_q <= rms_candidate_q + 16'd1;
                        rms_square_q <= rms_square_next_w;
                        state_q <= ST_SQRT;
                    end
                end

                ST_INV_DIV: begin
                    div_quotient_q <= div_quotient_next_w[DIV_QUOT_REG_WIDTH-1:0];
                    div_remainder_q <= div_remainder_next_w;
                    if (div_count_q == {{(DIV_COUNT_WIDTH-1){1'b0}}, 1'b1}) begin
                        inv_rms_q <= {1'b0, div_quotient_next_w};
                        div_count_q <= {DIV_COUNT_WIDTH{1'b0}};
                        state_q <= ST_SCALE;
                    end else begin
                        div_count_q <= div_count_q - {{(DIV_COUNT_WIDTH-1){1'b0}}, 1'b1};
                    end
                end

                ST_SCALE: begin
                    if (!scale_active_q && !out_valid_q && scale_act_valid_i && gain_valid_i && scale_act_ready_o && gain_ready_o) begin
                        out_work_q <= {LANES*ACT_WIDTH{1'b0}};
                        lane_idx_q <= {LANE_INDEX_WIDTH{1'b0}};
                        lane_last_q <= (LANES <= 1) ? 1'b1 : 1'b0;
                        scale_idx_q <= scale_idx_q + {{(BEAT_INDEX_WIDTH-1){1'b0}}, 1'b1};
                        scale_active_q <= 1'b1;
                    end else if (scale_active_q && !out_valid_q) begin
                        scale_act_q <= scale_act_w;
                        scale_gain_q <= scale_gain_w;
                        state_q <= ST_SCALE_PREP;
                    end
                end

                ST_SCALE_PREP: begin
                        scale_product_sign_q <= scale_product_sign_w;
                        scale_mul_acc_q <= 64'd0;
                        scale_mul_multiplicand_q <= {{(64-(ACT_WIDTH+GAIN_WIDTH)){1'b0}}, scale_lane_product_w};
                        scale_mul_multiplier_q <= inv_rms_q[31:0];
                        scale_mul_addend_hi_q <= 32'd0;
                        scale_mul_carry_q <= 1'b0;
                        scale_mul_count_q <= SCALE_MUL_COUNT_INIT;
                        state_q <= ST_SCALE_MUL;
                end

                ST_SCALE_MUL: begin
                    scale_mul_acc_q[31:0] <= scale_mul_low_sum_w[31:0];
                    scale_mul_addend_hi_q <= scale_mul_multiplier_q[0] ? scale_mul_multiplicand_q[63:32] : 32'd0;
                    scale_mul_carry_q <= scale_mul_low_sum_w[32];
                    state_q <= ST_SCALE_MUL_HI;
                end

                ST_SCALE_MUL_HI: begin
                    scale_mul_acc_q[63:32] <= scale_mul_high_sum_w;
                    scale_mul_multiplicand_q <= {scale_mul_multiplicand_q[62:0], 1'b0};
                    scale_mul_multiplier_q <= {1'b0, scale_mul_multiplier_q[31:1]};
                    if (scale_mul_count_q == 6'd1) begin
                        scale_mul_count_q <= 6'd0;
                        scale_round_active_q <= 1'b1;
                        state_q <= ST_SCALE_ROUND;
                    end else begin
                        scale_mul_count_q <= scale_mul_count_q - 6'd1;
                        state_q <= ST_SCALE_MUL;
                    end
                end

                ST_SCALE_ROUND: begin
                end

                ST_DONE: begin
                    out_valid_q <= 1'b0;
                    if (done_valid_q && done_ready_i) begin
                        done_valid_q <= 1'b0;
                        state_q <= ST_IDLE;
                    end
                end

                default: begin
                    state_q <= ST_IDLE;
                    out_valid_q <= 1'b0;
                    done_valid_q <= 1'b0;
                    scale_active_q <= 1'b0;
                    collect_active_q <= 1'b0;
                    collect_square_valid_q <= 1'b0;
                end
            endcase
            if (clear_i) begin
                state_q <= ST_IDLE;
                collect_idx_q <= {BEAT_INDEX_WIDTH{1'b0}};
                scale_idx_q <= {BEAT_INDEX_WIDTH{1'b0}};
                lane_idx_q <= {LANE_INDEX_WIDTH{1'b0}};
                out_valid_q <= 1'b0;
                done_valid_q <= 1'b0;
                saturation_seen_q <= 1'b0;
                sqrt_done_q <= 1'b0;
                collect_active_q <= 1'b0;
                collect_square_valid_q <= 1'b0;
                lane_last_q <= (LANES <= 1) ? 1'b1 : 1'b0;
                scale_active_q <= 1'b0;
                scale_round_active_q <= 1'b0;
                scale_act_q <= {ACT_WIDTH{1'b0}};
                scale_gain_q <= {GAIN_WIDTH{1'b0}};
                scale_mul_count_q <= 6'd0;
                scale_mul_addend_hi_q <= 32'd0;
                scale_mul_carry_q <= 1'b0;
            end
        end
    end
endmodule

`default_nettype wire
