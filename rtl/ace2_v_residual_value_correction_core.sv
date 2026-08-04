`timescale 1ns/1ps
`default_nettype none
/* verilator lint_off DECLFILENAME */

// Frozen contract: shared_v_residual_value_correction_attention_v1.
// This finalizer preserves the baseline V requantization and emits one
// canonical sign-extended signed-4 residual byte.  Because the quotient is
// clamped to [-7,+7], exact threshold comparisons replace a general divider.
module ace2_v_residual_projection_core (
    input  logic                      clk_i,
    input  logic                      rst_ni,
    input  logic                      clear_i,
    input  logic                      start_valid_i,
    output logic                      start_ready_o,
    input  logic signed [31:0]        accumulator_s32_i,
    input  logic signed [31:0]        multiplier_s32_i,
    input  logic        [5:0]         shift_u6_i,
    input  logic        [31:0]        baseline_v_scale32_i,
    input  logic        [31:0]        residual_v_scale32_i,
    output logic                      out_valid_o,
    input  logic                      out_ready_i,
    output logic signed [7:0]         baseline_v8_o,
    output logic        [7:0]         residual_v_canonical_u8_o,
    output logic                      positive_clamp_o,
    output logic                      negative_clamp_o,
    output logic                      descriptor_error_o,
    output logic                      numeric_overflow_o
);
    logic out_valid_q;
    logic signed [7:0] baseline_v8_q;
    logic [7:0] residual_v_canonical_q;
    logic positive_clamp_q, negative_clamp_q;
    logic descriptor_error_q, numeric_overflow_q;

    logic baseline_scale_valid_w, residual_scale_valid_w;
    logic signed [63:0] product_w;
    logic signed [71:0] product_s72_w;
    logic signed [71:0] baseline_rounded_w;
    logic signed [7:0] baseline_saturated_w;
    logic signed [71:0] error_w;
    logic [71:0] error_magnitude_w;
    logic [87:0] numerator_unshifted_w;
    logic signed [8:0] delta_w;
    logic [127:0] numerator_w, denominator_w;
    logic [3:0] floor_quotient_w;
    logic [127:0] floor_product_w, remainder_w;
    logic [128:0] doubled_remainder_w, extended_denominator_w;
    logic round_increment_w;
    logic [4:0] rounded_magnitude_w;
    logic signed [3:0] residual_s4_w;
    logic positive_clamp_w, negative_clamp_w;

    function automatic logic signed [71:0] round_shift_even_s72(
        input logic signed [71:0] value,
        input logic [5:0] shift
    );
        logic sign;
        logic [71:0] magnitude, quotient, remainder, mask, half;
        begin
            if (shift == 0) begin
                round_shift_even_s72 = value;
            end else begin
                sign = value[71];
                magnitude = sign ? $unsigned(-value) : $unsigned(value);
                quotient = magnitude >> shift;
                mask = {72{1'b1}} >> (72-shift);
                remainder = magnitude & mask;
                half = 72'd1 << (shift-1);
                if ((remainder > half) || ((remainder == half) && quotient[0]))
                    quotient = quotient + 72'd1;
                round_shift_even_s72 = sign ? -$signed(quotient) : $signed(quotient);
            end
        end
    endfunction

    function automatic logic [127:0] multiply_small_u128(
        input logic [127:0] value,
        input logic [3:0] factor
    );
        logic [127:0] result;
        integer bit_index;
        begin
            result = 128'd0;
            for (bit_index = 0; bit_index < 4; bit_index = bit_index + 1)
                if (factor[bit_index])
                    result = result + (value << bit_index);
            multiply_small_u128 = result;
        end
    endfunction

    assign start_ready_o = !out_valid_q || out_ready_i;
    assign out_valid_o = out_valid_q;
    assign baseline_v8_o = baseline_v8_q;
    assign residual_v_canonical_u8_o = residual_v_canonical_q;
    assign positive_clamp_o = positive_clamp_q;
    assign negative_clamp_o = negative_clamp_q;
    assign descriptor_error_o = descriptor_error_q;
    assign numeric_overflow_o = numeric_overflow_q;

    always @* begin
        baseline_scale_valid_w =
            (baseline_v_scale32_i[31:24] == 8'd0) &&
            (baseline_v_scale32_i[15:0] >= 16'h8000) &&
            ($signed(baseline_v_scale32_i[23:16]) >= -8'sd24) &&
            ($signed(baseline_v_scale32_i[23:16]) <= 8'sd4);
        residual_scale_valid_w =
            (residual_v_scale32_i[31:24] == 8'd0) &&
            (residual_v_scale32_i[15:0] >= 16'h8000) &&
            ($signed(residual_v_scale32_i[23:16]) >= -8'sd24) &&
            ($signed(residual_v_scale32_i[23:16]) <= 8'sd4);
        product_w = accumulator_s32_i * multiplier_s32_i;
        product_s72_w = {{8{product_w[63]}}, product_w};
        baseline_rounded_w = round_shift_even_s72(product_s72_w, shift_u6_i);
        if (baseline_rounded_w > 72'sd127)
            baseline_saturated_w = 8'sd127;
        else if (baseline_rounded_w < -72'sd128)
            baseline_saturated_w = -8'sd128;
        else
            baseline_saturated_w = baseline_rounded_w[7:0];
        error_w = product_s72_w -
            ({{64{baseline_saturated_w[7]}}, baseline_saturated_w} <<< shift_u6_i);
        error_magnitude_w = error_w[71] ? $unsigned(-error_w) : $unsigned(error_w);
        numerator_unshifted_w = error_magnitude_w * baseline_v_scale32_i[15:0];
        delta_w = $signed({baseline_v_scale32_i[23], baseline_v_scale32_i[23:16]}) -
                  $signed({residual_v_scale32_i[23], residual_v_scale32_i[23:16]}) -
                  $signed({3'd0, shift_u6_i});
        numerator_w = {{40{1'b0}}, numerator_unshifted_w};
        denominator_w = {{112{1'b0}}, residual_v_scale32_i[15:0]};
        if (delta_w >= 0)
            numerator_w = numerator_w << delta_w;
        else
            denominator_w = denominator_w << (-delta_w);

        floor_quotient_w = 4'd0;
        if (numerator_w >= (denominator_w << 3))
            floor_quotient_w = 4'd8;
        else if (numerator_w >= multiply_small_u128(denominator_w, 4'd7))
            floor_quotient_w = 4'd7;
        else if (numerator_w >= multiply_small_u128(denominator_w, 4'd6))
            floor_quotient_w = 4'd6;
        else if (numerator_w >= multiply_small_u128(denominator_w, 4'd5))
            floor_quotient_w = 4'd5;
        else if (numerator_w >= multiply_small_u128(denominator_w, 4'd4))
            floor_quotient_w = 4'd4;
        else if (numerator_w >= multiply_small_u128(denominator_w, 4'd3))
            floor_quotient_w = 4'd3;
        else if (numerator_w >= multiply_small_u128(denominator_w, 4'd2))
            floor_quotient_w = 4'd2;
        else if (numerator_w >= denominator_w)
            floor_quotient_w = 4'd1;
        floor_product_w = multiply_small_u128(denominator_w, floor_quotient_w);
        remainder_w = numerator_w - floor_product_w;
        doubled_remainder_w = {remainder_w, 1'b0};
        extended_denominator_w = {1'b0, denominator_w};
        round_increment_w =
            (doubled_remainder_w > extended_denominator_w) ||
            ((doubled_remainder_w == extended_denominator_w) && floor_quotient_w[0]);
        rounded_magnitude_w = {1'b0, floor_quotient_w} + {4'd0, round_increment_w};
        positive_clamp_w = !error_w[71] && (rounded_magnitude_w > 5'd7);
        negative_clamp_w = error_w[71] && (rounded_magnitude_w > 5'd7);
        if (rounded_magnitude_w > 5'd7)
            residual_s4_w = error_w[71] ? -4'sd7 : 4'sd7;
        else
            residual_s4_w = error_w[71] ?
                -$signed(rounded_magnitude_w[3:0]) :
                 $signed(rounded_magnitude_w[3:0]);
    end

    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            out_valid_q <= 1'b0;
            baseline_v8_q <= '0;
            residual_v_canonical_q <= '0;
            positive_clamp_q <= 1'b0;
            negative_clamp_q <= 1'b0;
            descriptor_error_q <= 1'b0;
            numeric_overflow_q <= 1'b0;
        end else if (clear_i) begin
            out_valid_q <= 1'b0;
            positive_clamp_q <= 1'b0;
            negative_clamp_q <= 1'b0;
            descriptor_error_q <= 1'b0;
            numeric_overflow_q <= 1'b0;
        end else begin
            if (out_valid_q && out_ready_i)
                out_valid_q <= 1'b0;
            if (start_valid_i && start_ready_o) begin
                positive_clamp_q <= 1'b0;
                negative_clamp_q <= 1'b0;
                descriptor_error_q <= 1'b0;
                numeric_overflow_q <= 1'b0;
                baseline_v8_q <= '0;
                residual_v_canonical_q <= '0;
                if (!(baseline_scale_valid_w && residual_scale_valid_w) ||
                    (multiplier_s32_i <= 0) ||
                    (delta_w < -9'sd91) || (delta_w > 9'sd28)) begin
                    descriptor_error_q <= 1'b1;
                end else begin
                    baseline_v8_q <= baseline_saturated_w;
                    residual_v_canonical_q <= {{4{residual_s4_w[3]}}, residual_s4_w};
                    positive_clamp_q <= positive_clamp_w;
                    negative_clamp_q <= negative_clamp_w;
                end
                out_valid_q <= 1'b1;
            end
        end
    end
endmodule


// One engine is reused across all layers, heads, and output lanes.  It accepts
// one authoritative Q0.15 probability and canonical residual-V byte per cycle,
// then performs one exact Scale32 conversion after the complete lane sum.  The
// radix-16 restoring converter emits 32 quotient bits in exactly eight cycles.
module ace2_v_residual_value_correction_core #(
    parameter integer CONTEXT_MAX = 32768
) (
    input  logic                      clk_i,
    input  logic                      rst_ni,
    input  logic                      clear_i,
    input  logic                      start_valid_i,
    output logic                      start_ready_o,
    input  logic        [15:0]        lane_count_u16_i,
    input  logic signed [31:0]        baseline_value_accumulator_s32_i,
    input  logic        [31:0]        baseline_v_scale32_i,
    input  logic        [31:0]        residual_v_scale32_i,
    input  logic                      sample_valid_i,
    output logic                      sample_ready_o,
    input  logic        [15:0]        probability_q0_15_u16_i,
    input  logic        [7:0]         residual_v_canonical_u8_i,
    output logic                      out_valid_o,
    input  logic                      out_ready_i,
    output logic signed [31:0]        corrected_value_accumulator_s32_o,
    output logic signed [31:0]        correction_baseline_domain_s32_o,
    output logic signed [63:0]        correction_raw_s64_o,
    output logic                      descriptor_error_o,
    output logic                      numeric_overflow_o
);
    typedef enum logic [1:0] {
        VC_IDLE,
        VC_ACCUMULATE,
        VC_DIVIDE
    } state_t;

    localparam logic [15:0] CONTEXT_MAX_U16 = CONTEXT_MAX[15:0];

    state_t state_q;
    logic out_valid_q, descriptor_error_q, numeric_overflow_q;
    logic [15:0] lane_count_q, sample_count_q;
    logic signed [31:0] baseline_accumulator_q;
    logic [15:0] baseline_sig_q, residual_sig_q;
    logic signed [7:0] baseline_exp_q, residual_exp_q;
    logic signed [63:0] correction_raw_q;
    logic signed [31:0] correction_q, corrected_accumulator_q;
    logic [127:0] remainder_q, divisor_q;
    logic [31:0] quotient_q;
    logic [2:0] divide_cycle_q;

    logic baseline_scale_valid_w, residual_scale_valid_w;
    logic residual_canonical_valid_w;
    logic signed [4:0] residual_s5_w;
    logic signed [16:0] probability_s17_w;
    logic signed [21:0] sample_product_s22_w;
    logic signed [64:0] accumulator_next_s65_w;
    logic accumulator_overflow_w;
    logic sample_last_w;
    logic [63:0] correction_magnitude_w;
    logic [79:0] ratio_product_w;
    logic signed [8:0] ratio_delta_w;
    logic [127:0] ratio_numerator_w, ratio_denominator_w;
    logic ratio_quotient_overflow_w;

    logic [4:0] divide_bit0_w, divide_bit1_w, divide_bit2_w, divide_bit3_w;
    logic [127:0] shifted_divisor0_w, shifted_divisor1_w;
    logic [127:0] shifted_divisor2_w, shifted_divisor3_w;
    logic [127:0] remainder_step0_w, remainder_step1_w;
    logic [127:0] remainder_step2_w, remainder_step3_w, remainder_next_w;
    logic [31:0] quotient_next_w;
    logic [128:0] doubled_final_remainder_w, extended_divisor_w;
    logic divide_round_increment_w;
    logic [32:0] rounded_quotient_w;
    logic rounded_quotient_range_error_w;
    logic signed [32:0] correction_signed_s33_w;
    logic signed [33:0] corrected_sum_s34_w;
    logic corrected_sum_overflow_w;

    assign start_ready_o = (state_q == VC_IDLE) && !out_valid_q;
    assign sample_ready_o = (state_q == VC_ACCUMULATE);
    assign out_valid_o = out_valid_q;
    assign corrected_value_accumulator_s32_o = corrected_accumulator_q;
    assign correction_baseline_domain_s32_o = correction_q;
    assign correction_raw_s64_o = correction_raw_q;
    assign descriptor_error_o = descriptor_error_q;
    assign numeric_overflow_o = numeric_overflow_q;

    always @* begin
        baseline_scale_valid_w =
            (baseline_v_scale32_i[31:24] == 8'd0) &&
            (baseline_v_scale32_i[15:0] >= 16'h8000) &&
            ($signed(baseline_v_scale32_i[23:16]) >= -8'sd24) &&
            ($signed(baseline_v_scale32_i[23:16]) <= 8'sd4);
        residual_scale_valid_w =
            (residual_v_scale32_i[31:24] == 8'd0) &&
            (residual_v_scale32_i[15:0] >= 16'h8000) &&
            ($signed(residual_v_scale32_i[23:16]) >= -8'sd24) &&
            ($signed(residual_v_scale32_i[23:16]) <= 8'sd4);
        residual_canonical_valid_w =
            (residual_v_canonical_u8_i[7:4] ==
             {4{residual_v_canonical_u8_i[3]}}) &&
            (residual_v_canonical_u8_i[3:0] != 4'h8);
        residual_s5_w = $signed({residual_v_canonical_u8_i[3],
                                 residual_v_canonical_u8_i[3:0]});
        probability_s17_w = $signed({1'b0, probability_q0_15_u16_i});
        sample_product_s22_w = probability_s17_w * residual_s5_w;
        accumulator_next_s65_w =
            $signed({correction_raw_q[63], correction_raw_q}) +
            $signed({{43{sample_product_s22_w[21]}}, sample_product_s22_w});
        accumulator_overflow_w =
            accumulator_next_s65_w[64] != accumulator_next_s65_w[63];
        sample_last_w = ({1'b0, sample_count_q} + 17'd1) ==
                        {1'b0, lane_count_q};

        correction_magnitude_w = accumulator_next_s65_w[63] ?
            $unsigned(-accumulator_next_s65_w[63:0]) :
            $unsigned(accumulator_next_s65_w[63:0]);
        ratio_product_w = correction_magnitude_w * residual_sig_q;
        ratio_delta_w = $signed({residual_exp_q[7], residual_exp_q}) -
                        $signed({baseline_exp_q[7], baseline_exp_q});
        ratio_numerator_w = {{48{1'b0}}, ratio_product_w};
        ratio_denominator_w = {{112{1'b0}}, baseline_sig_q};
        if (ratio_delta_w >= 0)
            ratio_numerator_w = ratio_numerator_w << ratio_delta_w;
        else
            ratio_denominator_w = ratio_denominator_w << (-ratio_delta_w);
        ratio_quotient_overflow_w =
            ratio_numerator_w >= (ratio_denominator_w << 32);

        divide_bit0_w = 5'd31 - {divide_cycle_q, 2'b00};
        divide_bit1_w = divide_bit0_w - 5'd1;
        divide_bit2_w = divide_bit0_w - 5'd2;
        divide_bit3_w = divide_bit0_w - 5'd3;
        shifted_divisor0_w = divisor_q << divide_bit0_w;
        shifted_divisor1_w = divisor_q << divide_bit1_w;
        shifted_divisor2_w = divisor_q << divide_bit2_w;
        shifted_divisor3_w = divisor_q << divide_bit3_w;
        quotient_next_w = quotient_q;
        remainder_step0_w = remainder_q;
        if (remainder_step0_w >= shifted_divisor0_w) begin
            remainder_step0_w = remainder_step0_w - shifted_divisor0_w;
            quotient_next_w[divide_bit0_w] = 1'b1;
        end
        remainder_step1_w = remainder_step0_w;
        if (remainder_step1_w >= shifted_divisor1_w) begin
            remainder_step1_w = remainder_step1_w - shifted_divisor1_w;
            quotient_next_w[divide_bit1_w] = 1'b1;
        end
        remainder_step2_w = remainder_step1_w;
        if (remainder_step2_w >= shifted_divisor2_w) begin
            remainder_step2_w = remainder_step2_w - shifted_divisor2_w;
            quotient_next_w[divide_bit2_w] = 1'b1;
        end
        remainder_step3_w = remainder_step2_w;
        if (remainder_step3_w >= shifted_divisor3_w) begin
            remainder_step3_w = remainder_step3_w - shifted_divisor3_w;
            quotient_next_w[divide_bit3_w] = 1'b1;
        end
        remainder_next_w = remainder_step3_w;
        doubled_final_remainder_w = {remainder_next_w, 1'b0};
        extended_divisor_w = {1'b0, divisor_q};
        divide_round_increment_w =
            (doubled_final_remainder_w > extended_divisor_w) ||
            ((doubled_final_remainder_w == extended_divisor_w) && quotient_next_w[0]);
        rounded_quotient_w = {1'b0, quotient_next_w} +
                             {{32{1'b0}}, divide_round_increment_w};
        rounded_quotient_range_error_w = correction_raw_q[63] ?
            (rounded_quotient_w > 33'h080000000) :
            (rounded_quotient_w > 33'h07fffffff);
        correction_signed_s33_w = correction_raw_q[63] ?
            -$signed(rounded_quotient_w) : $signed(rounded_quotient_w);
        corrected_sum_s34_w =
            $signed({{2{baseline_accumulator_q[31]}}, baseline_accumulator_q}) +
            $signed({correction_signed_s33_w[32], correction_signed_s33_w});
        corrected_sum_overflow_w =
            corrected_sum_s34_w[33:32] != {2{corrected_sum_s34_w[31]}};
    end

    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            state_q <= VC_IDLE;
            out_valid_q <= 1'b0;
            descriptor_error_q <= 1'b0;
            numeric_overflow_q <= 1'b0;
            lane_count_q <= '0;
            sample_count_q <= '0;
            baseline_accumulator_q <= '0;
            baseline_sig_q <= '0;
            residual_sig_q <= '0;
            baseline_exp_q <= '0;
            residual_exp_q <= '0;
            correction_raw_q <= '0;
            correction_q <= '0;
            corrected_accumulator_q <= '0;
            remainder_q <= '0;
            divisor_q <= '0;
            quotient_q <= '0;
            divide_cycle_q <= '0;
        end else if (clear_i) begin
            state_q <= VC_IDLE;
            out_valid_q <= 1'b0;
            descriptor_error_q <= 1'b0;
            numeric_overflow_q <= 1'b0;
            sample_count_q <= '0;
            correction_raw_q <= '0;
        end else begin
            if (out_valid_q && out_ready_i)
                out_valid_q <= 1'b0;
            case (state_q)
                VC_IDLE: begin
                    if (start_valid_i && start_ready_o) begin
                        descriptor_error_q <= 1'b0;
                        numeric_overflow_q <= 1'b0;
                        correction_raw_q <= '0;
                        correction_q <= '0;
                        corrected_accumulator_q <= '0;
                        sample_count_q <= '0;
                        if ((lane_count_u16_i == 16'd0) ||
                            (lane_count_u16_i > CONTEXT_MAX_U16) ||
                            !(baseline_scale_valid_w && residual_scale_valid_w)) begin
                            descriptor_error_q <= 1'b1;
                            out_valid_q <= 1'b1;
                        end else begin
                            lane_count_q <= lane_count_u16_i;
                            baseline_accumulator_q <= baseline_value_accumulator_s32_i;
                            baseline_sig_q <= baseline_v_scale32_i[15:0];
                            residual_sig_q <= residual_v_scale32_i[15:0];
                            baseline_exp_q <= $signed(baseline_v_scale32_i[23:16]);
                            residual_exp_q <= $signed(residual_v_scale32_i[23:16]);
                            state_q <= VC_ACCUMULATE;
                        end
                    end
                end
                VC_ACCUMULATE: begin
                    if (sample_valid_i && sample_ready_o) begin
                        if (!residual_canonical_valid_w) begin
                            descriptor_error_q <= 1'b1;
                            corrected_accumulator_q <= baseline_accumulator_q;
                            out_valid_q <= 1'b1;
                            state_q <= VC_IDLE;
                        end else if (accumulator_overflow_w) begin
                            numeric_overflow_q <= 1'b1;
                            corrected_accumulator_q <= baseline_accumulator_q;
                            out_valid_q <= 1'b1;
                            state_q <= VC_IDLE;
                        end else begin
                            correction_raw_q <= accumulator_next_s65_w[63:0];
                            sample_count_q <= sample_count_q + 16'd1;
                            if (sample_last_w) begin
                                if (ratio_quotient_overflow_w) begin
                                    numeric_overflow_q <= 1'b1;
                                    corrected_accumulator_q <= baseline_accumulator_q;
                                    out_valid_q <= 1'b1;
                                    state_q <= VC_IDLE;
                                end else begin
                                    remainder_q <= ratio_numerator_w;
                                    divisor_q <= ratio_denominator_w;
                                    quotient_q <= 32'd0;
                                    divide_cycle_q <= 3'd0;
                                    state_q <= VC_DIVIDE;
                                end
                            end
                        end
                    end
                end
                VC_DIVIDE: begin
                    remainder_q <= remainder_next_w;
                    quotient_q <= quotient_next_w;
                    if (divide_cycle_q == 3'd7) begin
                        if (rounded_quotient_range_error_w || corrected_sum_overflow_w) begin
                            numeric_overflow_q <= 1'b1;
                            corrected_accumulator_q <= baseline_accumulator_q;
                        end else begin
                            correction_q <= correction_signed_s33_w[31:0];
                            corrected_accumulator_q <= corrected_sum_s34_w[31:0];
                        end
                        out_valid_q <= 1'b1;
                        state_q <= VC_IDLE;
                    end else begin
                        divide_cycle_q <= divide_cycle_q + 3'd1;
                    end
                end
                default: state_q <= VC_IDLE;
            endcase
        end
    end
endmodule

`default_nettype wire
