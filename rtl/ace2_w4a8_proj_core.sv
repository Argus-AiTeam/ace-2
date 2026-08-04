`default_nettype none

module ace2_w4a8_proj_core #(
    parameter integer K_SIZE = 896,
    parameter integer MAC_LANES = 4,
    parameter integer ACT_WIDTH = 8,
    parameter integer WGT_WIDTH = 4,
    parameter integer ACC_WIDTH = 32,
    parameter integer GROUP_INDEX_WIDTH = ((K_SIZE / MAC_LANES) <= 1) ? 1 : $clog2((K_SIZE / MAC_LANES) + 1)
) (
    input  wire                              clk_i,
    input  wire                              rst_ni,
    input  wire                              clear_i,

    input  wire                              start_valid_i,
    output wire                              start_ready_o,
    input  wire [GROUP_INDEX_WIDTH-1:0]      last_group_i,

    input  wire                              pair_valid_i,
    output wire                              pair_ready_o,
    input  wire [MAC_LANES*ACT_WIDTH-1:0]   act_data_i,
    input  wire [MAC_LANES*WGT_WIDTH-1:0]   weight_data_i,

    input  wire                              meta_valid_i,
    output wire                              meta_ready_o,
    input  wire signed [31:0]                multiplier_i,
    input  wire [5:0]                        right_shift_i,
    input  wire signed [ACT_WIDTH-1:0]       output_zero_point_i,
    input  wire signed [ACC_WIDTH-1:0]       bias_accumulator_i,

    output wire                              out_valid_o,
    input  wire                              out_ready_i,
    output wire [ACT_WIDTH-1:0]              out_data_o,
    output wire signed [ACC_WIDTH-1:0]       acc_o,
    output wire                              accumulator_overflow_o,
    output wire                              saturation_seen_o
);
    localparam [3:0] ST_IDLE        = 4'd0;
    localparam [3:0] ST_RUN         = 4'd1;
    localparam [3:0] ST_ACCUM_COMMIT = 4'd2;
    localparam [3:0] ST_META        = 4'd3;
    localparam [3:0] ST_MUL         = 4'd4;
    localparam [3:0] ST_ROUND_SHIFT = 4'd5;
    localparam [3:0] ST_ROUND_APPLY = 4'd6;
    localparam [3:0] ST_SATURATE    = 4'd7;
    localparam [3:0] ST_DONE        = 4'd8;
    localparam [3:0] ST_META_PREP   = 4'd9;

    reg [3:0] state_q;
    reg [GROUP_INDEX_WIDTH-1:0] group_idx_q;
    reg signed [ACC_WIDTH-1:0] acc_q;
    reg out_valid_q;
    reg [ACT_WIDTH-1:0] out_data_q;
    reg accumulator_overflow_q;
    reg saturation_seen_q;
    reg product_sign_q;
    reg [5:0] right_shift_q;
    reg signed [ACT_WIDTH-1:0] output_zero_point_q;
    reg [63:0] mul_acc_q;
    reg [63:0] mul_multiplicand_q;
    reg [31:0] mul_multiplier_q;
    reg [31:0] meta_multiplier_q;
    reg [5:0] mul_count_q;
    reg [1:0] mul_chunk_q;
    reg mul_carry_q;
    reg [63:0] round_value_q;
    reg [5:0] round_count_q;
    reg round_guard_q;
    reg round_sticky_q;
    reg [64:0] rounded_magnitude_q;
    reg clear_q;
    integer mac_lane;

    reg signed [ACC_WIDTH-1:0] partial_sum_w;
    reg signed [ACC_WIDTH-1:0] partial_sum_q;
    reg signed [ACT_WIDTH-1:0] mac_act_w;
    reg signed [WGT_WIDTH-1:0] mac_weight_w;
    reg [15:0] mul_acc_chunk_w;
    reg [15:0] mul_multiplicand_chunk_w;
    wire [31:0] acc_abs_w;
    wire [31:0] multiplier_abs_w;
    wire signed [ACC_WIDTH:0] biased_acc_w;
    wire bias_add_overflow_w;
    wire [15:0] mul_addend_chunk_w;
    wire [16:0] mul_chunk_sum_w;
    wire [63:0] mul_next_acc_w;
    wire round_increment_w;
    wire signed [8:0] zero_point_ext_w;
    wire [8:0] positive_limit_w;
    wire [8:0] negative_limit_w;
    wire magnitude_gt_positive_limit_w;
    wire magnitude_gt_negative_limit_w;
    wire signed [8:0] positive_value_w;
    wire signed [8:0] negative_value_w;
    wire [ACT_WIDTH-1:0] positive_output_w;
    wire [ACT_WIDTH-1:0] negative_output_w;

    assign start_ready_o = (state_q == ST_IDLE) && !out_valid_q;
    assign pair_ready_o = (state_q == ST_RUN);
    assign meta_ready_o = (state_q == ST_META);
    assign out_valid_o = out_valid_q;
    assign out_data_o = out_data_q;
    assign acc_o = acc_q;
    assign accumulator_overflow_o = accumulator_overflow_q;
    assign saturation_seen_o = saturation_seen_q;
    assign biased_acc_w =
        $signed({acc_q[ACC_WIDTH-1], acc_q}) +
        $signed({bias_accumulator_i[ACC_WIDTH-1], bias_accumulator_i});
    assign bias_add_overflow_w = biased_acc_w[ACC_WIDTH] != biased_acc_w[ACC_WIDTH-1];
    assign acc_abs_w = acc_q[ACC_WIDTH-1] ? (~acc_q + {{(ACC_WIDTH-1){1'b0}}, 1'b1}) : acc_q;
    assign multiplier_abs_w = meta_multiplier_q[31] ? (~meta_multiplier_q + 32'd1) : meta_multiplier_q;
    assign mul_addend_chunk_w = mul_multiplier_q[0] ? mul_multiplicand_chunk_w : 16'd0;
    assign mul_chunk_sum_w = {1'b0, mul_acc_chunk_w} +
                             {1'b0, mul_addend_chunk_w} +
                             {16'd0, ((mul_chunk_q != 2'd0) && mul_carry_q)};
    assign mul_next_acc_w = (mul_chunk_q == 2'd0) ? {mul_acc_q[63:16], mul_chunk_sum_w[15:0]} :
                            (mul_chunk_q == 2'd1) ? {mul_acc_q[63:32], mul_chunk_sum_w[15:0], mul_acc_q[15:0]} :
                            (mul_chunk_q == 2'd2) ? {mul_acc_q[63:48], mul_chunk_sum_w[15:0], mul_acc_q[31:0]} :
                                                    {mul_chunk_sum_w[15:0], mul_acc_q[47:0]};
    assign round_increment_w = round_guard_q && (round_sticky_q || round_value_q[0]);
    assign zero_point_ext_w = {output_zero_point_q[ACT_WIDTH-1], output_zero_point_q};
    assign positive_limit_w = 9'sd127 - zero_point_ext_w;
    assign negative_limit_w = zero_point_ext_w + 9'sd128;
    assign magnitude_gt_positive_limit_w = (|rounded_magnitude_q[64:9]) || (rounded_magnitude_q[8:0] > positive_limit_w);
    assign magnitude_gt_negative_limit_w = (|rounded_magnitude_q[64:9]) || (rounded_magnitude_q[8:0] > negative_limit_w);
    assign positive_value_w = zero_point_ext_w + $signed({1'b0, rounded_magnitude_q[7:0]});
    assign negative_value_w = zero_point_ext_w - $signed({1'b0, rounded_magnitude_q[7:0]});
    assign positive_output_w = positive_value_w[ACT_WIDTH-1:0] | {ACT_WIDTH{positive_value_w[8] & 1'b0}};
    assign negative_output_w = negative_value_w[ACT_WIDTH-1:0] | {ACT_WIDTH{negative_value_w[8] & 1'b0}};

    always @* begin
        partial_sum_w = {ACC_WIDTH{1'b0}};
        for (mac_lane = 0; mac_lane < MAC_LANES; mac_lane = mac_lane + 1) begin
            mac_act_w = act_data_i[mac_lane*ACT_WIDTH +: ACT_WIDTH];
            mac_weight_w = weight_data_i[mac_lane*WGT_WIDTH +: WGT_WIDTH];
            partial_sum_w = partial_sum_w + (mac_act_w * mac_weight_w);
        end
    end

    always @* begin
        case (mul_chunk_q)
            2'd0: begin
                mul_acc_chunk_w = mul_acc_q[15:0];
                mul_multiplicand_chunk_w = mul_multiplicand_q[15:0];
            end
            2'd1: begin
                mul_acc_chunk_w = mul_acc_q[31:16];
                mul_multiplicand_chunk_w = mul_multiplicand_q[31:16];
            end
            2'd2: begin
                mul_acc_chunk_w = mul_acc_q[47:32];
                mul_multiplicand_chunk_w = mul_multiplicand_q[47:32];
            end
            default: begin
                mul_acc_chunk_w = mul_acc_q[63:48];
                mul_multiplicand_chunk_w = mul_multiplicand_q[63:48];
            end
        endcase
    end

    always @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            state_q <= ST_IDLE;
            group_idx_q <= {GROUP_INDEX_WIDTH{1'b0}};
            acc_q <= {ACC_WIDTH{1'b0}};
            out_valid_q <= 1'b0;
            out_data_q <= {ACT_WIDTH{1'b0}};
            accumulator_overflow_q <= 1'b0;
            saturation_seen_q <= 1'b0;
            product_sign_q <= 1'b0;
            right_shift_q <= 6'd0;
            output_zero_point_q <= {ACT_WIDTH{1'b0}};
            mul_acc_q <= 64'd0;
            mul_multiplicand_q <= 64'd0;
            mul_multiplier_q <= 32'd0;
            meta_multiplier_q <= 32'd0;
            mul_count_q <= 6'd0;
            mul_chunk_q <= 2'd0;
            mul_carry_q <= 1'b0;
            round_value_q <= 64'd0;
            round_count_q <= 6'd0;
            round_guard_q <= 1'b0;
            round_sticky_q <= 1'b0;
            rounded_magnitude_q <= 65'd0;
            clear_q <= 1'b0;
            partial_sum_q <= {ACC_WIDTH{1'b0}};
        end else begin
            clear_q <= clear_i;
            if (clear_q) begin
                state_q <= ST_IDLE;
                group_idx_q <= {GROUP_INDEX_WIDTH{1'b0}};
                acc_q <= {ACC_WIDTH{1'b0}};
                out_valid_q <= 1'b0;
                out_data_q <= {ACT_WIDTH{1'b0}};
                accumulator_overflow_q <= 1'b0;
                saturation_seen_q <= 1'b0;
                product_sign_q <= 1'b0;
                right_shift_q <= 6'd0;
                output_zero_point_q <= {ACT_WIDTH{1'b0}};
                mul_acc_q <= 64'd0;
                mul_multiplicand_q <= 64'd0;
                mul_multiplier_q <= 32'd0;
                meta_multiplier_q <= 32'd0;
                mul_count_q <= 6'd0;
                mul_chunk_q <= 2'd0;
                mul_carry_q <= 1'b0;
                round_value_q <= 64'd0;
                round_count_q <= 6'd0;
                round_guard_q <= 1'b0;
                round_sticky_q <= 1'b0;
                rounded_magnitude_q <= 65'd0;
                partial_sum_q <= {ACC_WIDTH{1'b0}};
            end else begin
                if (out_valid_q && out_ready_i) begin
                    out_valid_q <= 1'b0;
                    state_q <= ST_IDLE;
                end

                case (state_q)
                    ST_IDLE: begin
                        if (start_valid_i && start_ready_o) begin
                            group_idx_q <= {GROUP_INDEX_WIDTH{1'b0}};
                            acc_q <= {ACC_WIDTH{1'b0}};
                            out_data_q <= {ACT_WIDTH{1'b0}};
                            accumulator_overflow_q <= 1'b0;
                            saturation_seen_q <= 1'b0;
                            state_q <= ST_RUN;
                        end
                    end

                ST_RUN: begin
                    if (pair_valid_i && pair_ready_o) begin
                        partial_sum_q <= partial_sum_w;
                        state_q <= ST_ACCUM_COMMIT;
                    end
                end

                ST_ACCUM_COMMIT: begin
                    acc_q <= acc_q + partial_sum_q;
                    if (group_idx_q == last_group_i) begin
                        group_idx_q <= {GROUP_INDEX_WIDTH{1'b0}};
                        state_q <= ST_META;
                    end else begin
                        group_idx_q <= group_idx_q + {{(GROUP_INDEX_WIDTH-1){1'b0}}, 1'b1};
                        state_q <= ST_RUN;
                    end
                end

                ST_META: begin
                    if (meta_valid_i && meta_ready_o) begin
                        acc_q <= biased_acc_w[ACC_WIDTH-1:0];
                        accumulator_overflow_q <= bias_add_overflow_w;
                        product_sign_q <= biased_acc_w[ACC_WIDTH-1] ^ multiplier_i[31];
                        right_shift_q <= right_shift_i;
                        output_zero_point_q <= output_zero_point_i;
                        meta_multiplier_q <= multiplier_i;
                        state_q <= ST_META_PREP;
                    end
                end

                ST_META_PREP: begin
                    mul_acc_q <= 64'd0;
                    mul_multiplicand_q <= {{32{1'b0}}, acc_abs_w};
                    mul_multiplier_q <= multiplier_abs_w;
                    mul_count_q <= 6'd32;
                    mul_chunk_q <= 2'd0;
                    mul_carry_q <= 1'b0;
                    state_q <= ST_MUL;
                end

                ST_MUL: begin
                    mul_acc_q <= mul_next_acc_w;
                    if (mul_chunk_q == 2'd3) begin
                        mul_chunk_q <= 2'd0;
                        mul_carry_q <= 1'b0;
                        mul_multiplicand_q <= {mul_multiplicand_q[62:0], 1'b0};
                        mul_multiplier_q <= {1'b0, mul_multiplier_q[31:1]};
                        if (mul_count_q == 6'd1) begin
                            mul_count_q <= 6'd0;
                            round_value_q <= mul_next_acc_w;
                            round_count_q <= right_shift_q;
                            round_guard_q <= 1'b0;
                            round_sticky_q <= 1'b0;
                            if (right_shift_q == 6'd0) begin
                                state_q <= ST_ROUND_APPLY;
                            end else begin
                                state_q <= ST_ROUND_SHIFT;
                            end
                        end else begin
                            mul_count_q <= mul_count_q - 6'd1;
                        end
                    end else begin
                        mul_chunk_q <= mul_chunk_q + 2'd1;
                        mul_carry_q <= mul_chunk_sum_w[16];
                    end
                end

                ST_ROUND_SHIFT: begin
                    round_sticky_q <= round_sticky_q | round_guard_q;
                    round_guard_q <= round_value_q[0];
                    round_value_q <= {1'b0, round_value_q[63:1]};
                    if (round_count_q == 6'd1) begin
                        round_count_q <= 6'd0;
                        state_q <= ST_ROUND_APPLY;
                    end else begin
                        round_count_q <= round_count_q - 6'd1;
                    end
                end

                ST_ROUND_APPLY: begin
                    rounded_magnitude_q <= {1'b0, round_value_q} + {{64{1'b0}}, round_increment_w};
                    state_q <= ST_SATURATE;
                end

                ST_SATURATE: begin
                    if (product_sign_q) begin
                        if (magnitude_gt_negative_limit_w) begin
                            out_data_q <= 8'h80;
                            saturation_seen_q <= 1'b1;
                        end else begin
                            out_data_q <= negative_output_w;
                            saturation_seen_q <= 1'b0;
                        end
                    end else begin
                        if (magnitude_gt_positive_limit_w) begin
                            out_data_q <= 8'h7f;
                            saturation_seen_q <= 1'b1;
                        end else begin
                            out_data_q <= positive_output_w;
                            saturation_seen_q <= 1'b0;
                        end
                    end
                    out_valid_q <= 1'b1;
                    state_q <= ST_DONE;
                end

                ST_DONE: begin
                end

                default: begin
                    state_q <= ST_IDLE;
                    out_valid_q <= 1'b0;
                end
            endcase
        end
    end
    end
endmodule

`default_nettype wire
