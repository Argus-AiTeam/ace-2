`default_nettype none

module ace2_silu_gate_core (
    input  wire                 clk_i,
    input  wire                 rst_ni,
    input  wire                 clear_i,
    input  wire                 start_valid_i,
    output wire                 start_ready_o,
    input  wire signed [31:0]   multiplier_i,
    input  wire [5:0]           right_shift_i,
    input  wire signed [7:0]    output_zero_point_i,
    input  wire                 beat_valid_i,
    output wire                 beat_ready_o,
    input  wire [3:0]           lane_count_i,
    input  wire [127:0]         gate_data_i,
    input  wire [127:0]         up_data_i,
    output wire                 out_valid_o,
    input  wire                 out_ready_i,
    output wire [63:0]          out_data_o,
    output wire                 saturation_seen_o
);
    localparam [3:0] ST_READY = 4'd0;
    localparam [3:0] ST_INPUT = 4'd1;
    localparam [3:0] ST_SILU = 4'd2;
    localparam [3:0] ST_PRODUCT = 4'd3;
    localparam [3:0] ST_MUL_PREP = 4'd4;
    localparam [3:0] ST_MUL = 4'd5;
    localparam [3:0] ST_ROUND_SHIFT = 4'd6;
    localparam [3:0] ST_ROUND_APPLY = 4'd7;
    localparam [3:0] ST_SATURATE = 4'd8;
    localparam [3:0] ST_DONE = 4'd9;

    reg [3:0] state_q;
    reg [2:0] lane_q;
    reg [3:0] lane_count_q;
    reg signed [31:0] multiplier_q;
    reg [5:0] right_shift_q;
    reg signed [7:0] output_zero_point_q;
    reg [127:0] gate_data_q;
    reg [127:0] up_data_q;
    reg signed [15:0] gate_lane_q;
    reg signed [15:0] up_lane_q;
    reg signed [15:0] silu_q;
    reg signed [31:0] product_q;
    reg product_sign_q;
    reg [63:0] mul_acc_q;
    reg [63:0] mul_multiplicand_q;
    reg [31:0] mul_multiplier_q;
    reg [5:0] mul_count_q;
    reg [1:0] mul_chunk_q;
    reg mul_carry_q;
    reg [63:0] round_value_q;
    reg [5:0] round_count_q;
    reg round_guard_q;
    reg round_sticky_q;
    reg [64:0] rounded_magnitude_q;
    reg clear_q;
    reg [63:0] out_data_q;
    reg out_valid_q;
    reg saturation_seen_q;

    wire signed [15:0] selected_gate_w = gate_data_q[lane_q*16 +: 16];
    wire signed [15:0] selected_up_w = up_data_q[lane_q*16 +: 16];
    wire [2:0] last_lane_w =
        (lane_count_q == 4'd8) ? 3'd7 : (lane_count_q[2:0] - 3'd1);
    wire signed [15:0] lookup_w;
    wire [31:0] product_abs_w =
        product_q[31] ? (~product_q + 32'd1) : product_q;
    wire [31:0] multiplier_abs_w =
        multiplier_q[31] ? (~multiplier_q + 32'd1) : multiplier_q;
    reg [15:0] mul_acc_chunk_w;
    reg [15:0] mul_multiplicand_chunk_w;
    wire [15:0] mul_addend_chunk_w =
        mul_multiplier_q[0] ? mul_multiplicand_chunk_w : 16'd0;
    wire [16:0] mul_chunk_sum_w =
        {1'b0, mul_acc_chunk_w} +
        {1'b0, mul_addend_chunk_w} +
        {16'd0, ((mul_chunk_q != 2'd0) && mul_carry_q)};
    wire [63:0] mul_next_acc_w =
        (mul_chunk_q == 2'd0) ?
            {mul_acc_q[63:16], mul_chunk_sum_w[15:0]} :
        (mul_chunk_q == 2'd1) ?
            {mul_acc_q[63:32], mul_chunk_sum_w[15:0], mul_acc_q[15:0]} :
        (mul_chunk_q == 2'd2) ?
            {mul_acc_q[63:48], mul_chunk_sum_w[15:0], mul_acc_q[31:0]} :
            {mul_chunk_sum_w[15:0], mul_acc_q[47:0]};
    wire round_increment_w =
        round_guard_q && (round_sticky_q || round_value_q[0]);
    wire signed [8:0] zero_point_ext_w =
        {output_zero_point_q[7], output_zero_point_q};
    wire [8:0] positive_limit_w = 9'sd127 - zero_point_ext_w;
    wire [8:0] negative_limit_w = zero_point_ext_w + 9'sd128;
    wire positive_saturation_w =
        !product_sign_q &&
        ((|rounded_magnitude_q[64:9]) ||
         (rounded_magnitude_q[8:0] > positive_limit_w));
    wire negative_saturation_w =
        product_sign_q &&
        ((|rounded_magnitude_q[64:9]) ||
         (rounded_magnitude_q[8:0] > negative_limit_w));
    wire signed [8:0] positive_value_w =
        zero_point_ext_w + $signed({1'b0, rounded_magnitude_q[7:0]});
    wire signed [8:0] negative_value_w =
        zero_point_ext_w - $signed({1'b0, rounded_magnitude_q[7:0]});
    wire [7:0] output_value_w =
        positive_saturation_w ? 8'h7f :
        negative_saturation_w ? 8'h80 :
        product_sign_q ? negative_value_w[7:0] : positive_value_w[7:0];

    `include "generated/ace2_silu_lut.svh"

    assign lookup_w = silu_lookup_q3_12(gate_lane_q);
    assign start_ready_o = (state_q == ST_READY) && !out_valid_q;
    assign beat_ready_o = (state_q == ST_READY) && !out_valid_q;
    assign out_valid_o = out_valid_q;
    assign out_data_o = out_data_q;
    assign saturation_seen_o = saturation_seen_q;

    always @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            state_q <= ST_READY;
            lane_q <= 3'd0;
            lane_count_q <= 4'd0;
            multiplier_q <= 32'sd0;
            right_shift_q <= 6'd0;
            output_zero_point_q <= 8'sd0;
            gate_data_q <= 128'd0;
            up_data_q <= 128'd0;
            gate_lane_q <= 16'sd0;
            up_lane_q <= 16'sd0;
            silu_q <= 16'sd0;
            product_q <= 32'sd0;
            product_sign_q <= 1'b0;
            mul_acc_q <= 64'd0;
            mul_multiplicand_q <= 64'd0;
            mul_multiplier_q <= 32'd0;
            mul_count_q <= 6'd0;
            mul_chunk_q <= 2'd0;
            mul_carry_q <= 1'b0;
            round_value_q <= 64'd0;
            round_count_q <= 6'd0;
            round_guard_q <= 1'b0;
            round_sticky_q <= 1'b0;
            rounded_magnitude_q <= 65'd0;
            clear_q <= 1'b0;
            out_data_q <= 64'd0;
            out_valid_q <= 1'b0;
            saturation_seen_q <= 1'b0;
        end else begin
            clear_q <= clear_i;
            if (clear_q) begin
                state_q <= ST_READY;
                lane_q <= 3'd0;
                lane_count_q <= 4'd0;
                out_data_q <= 64'd0;
                out_valid_q <= 1'b0;
                saturation_seen_q <= 1'b0;
            end else begin
                if (start_valid_i && start_ready_o) begin
                    multiplier_q <= multiplier_i;
                    right_shift_q <= right_shift_i;
                    output_zero_point_q <= output_zero_point_i;
                end
                if (out_valid_q && out_ready_i) begin
                    out_valid_q <= 1'b0;
                    state_q <= ST_READY;
                end
                case (state_q)
                ST_READY: begin
                    if (beat_valid_i && beat_ready_o) begin
                        gate_data_q <= gate_data_i;
                        up_data_q <= up_data_i;
                        lane_count_q <= lane_count_i;
                        lane_q <= 3'd0;
                        out_data_q <= 64'd0;
                        saturation_seen_q <= 1'b0;
                        state_q <= ST_INPUT;
                    end
                end
                ST_INPUT: begin
                    gate_lane_q <= selected_gate_w;
                    up_lane_q <= selected_up_w;
                    state_q <= ST_SILU;
                end
                ST_SILU: begin
                    silu_q <= lookup_w;
                    state_q <= ST_PRODUCT;
                end
                ST_PRODUCT: begin
                    product_q <= silu_q * up_lane_q;
                    state_q <= ST_MUL_PREP;
                end
                ST_MUL_PREP: begin
                    product_sign_q <= product_q[31] ^ multiplier_q[31];
                    mul_acc_q <= 64'd0;
                    mul_multiplicand_q <= {32'd0, product_abs_w};
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
                            state_q <= (right_shift_q == 6'd0) ?
                                       ST_ROUND_APPLY : ST_ROUND_SHIFT;
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
                    rounded_magnitude_q <=
                        {1'b0, round_value_q} +
                        {{64{1'b0}}, round_increment_w};
                    state_q <= ST_SATURATE;
                end
                ST_SATURATE: begin
                    out_data_q[lane_q*8 +: 8] <= output_value_w;
                    saturation_seen_q <= saturation_seen_q |
                                         positive_saturation_w |
                                         negative_saturation_w;
                    state_q <= ST_DONE;
                end
                ST_DONE: begin
                    if (!out_valid_q) begin
                        if (lane_q == last_lane_w) begin
                            out_valid_q <= 1'b1;
                        end else begin
                            lane_q <= lane_q + 3'd1;
                            state_q <= ST_INPUT;
                        end
                    end
                end
                    default: state_q <= ST_READY;
                endcase
            end
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
endmodule

`default_nettype wire
