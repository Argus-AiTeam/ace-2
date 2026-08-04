`timescale 1ns/1ps
`default_nettype none
/* verilator lint_off DECLFILENAME */

// Frozen contract: shared_down_projection_residual_fusion_v1.
// One engine is shared across output lanes and sequential layers. It consumes
// the authoritative signed-32 down-projection accumulator before baseline
// requantization, aligns it with the signed-int8 post-attention residual in a
// common Scale32 exponent domain, adds in signed-96, rounds once, and saturates
// once to signed int8.
//
// Exact pre-clamp comparisons prove every non-saturated quotient magnitude is
// at most 128. The divider therefore resolves quotient bits 7..0 in eight
// shift/subtract cycles. Valid transactions have ten-cycle start-to-valid
// latency; invalid Scale32 descriptors fail before arithmetic starts.
module ace2_down_projection_residual_fusion_core (
    input  logic                       clk_i,
    input  logic                       rst_ni,
    input  logic                       clear_i,
    input  logic                       start_valid_i,
    output logic                       start_ready_o,
    input  logic signed [31:0]         accumulator_s32_i,
    input  logic signed [7:0]          residual_s8_i,
    input  logic        [31:0]         accumulator_scale32_i,
    input  logic        [31:0]         residual_scale32_i,
    input  logic        [31:0]         destination_scale32_i,
    output logic                       out_valid_o,
    input  logic                       out_ready_i,
    output logic signed [7:0]          fused_s8_o,
    output logic                       positive_saturation_o,
    output logic                       negative_saturation_o,
    output logic                       descriptor_error_o,
    output logic                       numeric_overflow_o,
    output logic signed [95:0]         numerator_s96_o,
    output logic        [63:0]         denominator_u64_o,
    output logic signed [7:0]          common_exponent_s8_o,
    output logic        [4:0]          latency_cycles_u5_o
);
    typedef enum logic [2:0] {
        DPRF_IDLE,
        DPRF_PREPARE,
        DPRF_DIVIDE,
        DPRF_FINALIZE
    } state_t;

    state_t state_q;
    logic out_valid_q;
    logic signed [7:0] fused_q;
    logic positive_saturation_q, negative_saturation_q;
    logic descriptor_error_q, numeric_overflow_q;
    logic signed [95:0] numerator_q;
    logic [95:0] numerator_magnitude_q;
    logic [63:0] denominator_q;
    logic signed [7:0] common_exponent_q;
    logic numerator_negative_q;
    logic [95:0] remainder_q;
    logic [7:0] quotient_q;
    logic [2:0] divide_bit_q;
    logic [4:0] latency_cycles_q;
    logic preclamp_positive_q, preclamp_negative_q;

    logic accumulator_scale_valid_w;
    logic residual_scale_valid_w;
    logic destination_scale_valid_w;
    logic all_scales_valid_w;
    logic [15:0] accumulator_sig_w, residual_sig_w, destination_sig_w;
    logic signed [7:0] accumulator_exp_w, residual_exp_w, destination_exp_w;
    logic signed [7:0] common_exponent_w;
    logic [5:0] accumulator_shift_w, residual_shift_w, destination_shift_w;
    logic signed [47:0] accumulator_product_w;
    logic signed [23:0] residual_product_w;
    logic signed [95:0] accumulator_term_w, residual_term_w;
    logic signed [96:0] numerator_sum_w;
    logic numerator_overflow_w;
    logic signed [95:0] numerator_s96_w;
    logic [95:0] numerator_magnitude_w;
    logic [63:0] denominator_w;
    logic denominator_overflow_w;

    logic [96:0] doubled_magnitude_w;
    logic [96:0] positive_threshold_w, negative_threshold_w;
    logic preclamp_positive_w, preclamp_negative_w;
    logic [95:0] shifted_denominator_w;
    logic [95:0] remainder_next_w;
    logic [7:0] quotient_next_w;
    logic [96:0] doubled_remainder_w;
    logic [96:0] extended_denominator_w;
    logic round_increment_w;
    logic [8:0] rounded_magnitude_w;
    logic final_positive_saturation_w, final_negative_saturation_w;
    logic signed [7:0] rounded_s8_w;

    function automatic logic scale32_valid(input logic [31:0] record);
        logic signed [7:0] exponent;
        begin
            exponent = $signed(record[23:16]);
            scale32_valid =
                (record[31:24] == 8'd0) &&
                (record[15:0] >= 16'h8000) &&
                (exponent >= -8'sd24) &&
                (exponent <= 8'sd4);
        end
    endfunction

    function automatic logic [96:0] multiply_u64_by_255(input logic [63:0] value);
        logic [96:0] extended;
        begin
            extended = {{33{1'b0}}, value};
            multiply_u64_by_255 = (extended << 8) - extended;
        end
    endfunction

    function automatic logic [96:0] multiply_u64_by_257(input logic [63:0] value);
        logic [96:0] extended;
        begin
            extended = {{33{1'b0}}, value};
            multiply_u64_by_257 = (extended << 8) + extended;
        end
    endfunction

    function automatic logic [5:0] exponent_delta(
        input logic signed [7:0] exponent,
        input logic signed [7:0] common_exponent
    );
        begin
            exponent_delta = 6'($unsigned(
                $signed({exponent[7], exponent}) -
                $signed({common_exponent[7], common_exponent})
            ));
        end
    endfunction

    assign start_ready_o = rst_ni && !clear_i &&
                           (state_q == DPRF_IDLE) &&
                           (!out_valid_q || out_ready_i);
    assign out_valid_o = out_valid_q;
    assign fused_s8_o = fused_q;
    assign positive_saturation_o = positive_saturation_q;
    assign negative_saturation_o = negative_saturation_q;
    assign descriptor_error_o = descriptor_error_q;
    assign numeric_overflow_o = numeric_overflow_q;
    assign numerator_s96_o = numerator_q;
    assign denominator_u64_o = denominator_q;
    assign common_exponent_s8_o = common_exponent_q;
    assign latency_cycles_u5_o = latency_cycles_q;

    always @* begin
        accumulator_scale_valid_w = scale32_valid(accumulator_scale32_i);
        residual_scale_valid_w = scale32_valid(residual_scale32_i);
        destination_scale_valid_w = scale32_valid(destination_scale32_i);
        all_scales_valid_w = accumulator_scale_valid_w &&
                             residual_scale_valid_w &&
                             destination_scale_valid_w;

        accumulator_sig_w = accumulator_scale32_i[15:0];
        residual_sig_w = residual_scale32_i[15:0];
        destination_sig_w = destination_scale32_i[15:0];
        accumulator_exp_w = $signed(accumulator_scale32_i[23:16]);
        residual_exp_w = $signed(residual_scale32_i[23:16]);
        destination_exp_w = $signed(destination_scale32_i[23:16]);

        common_exponent_w = accumulator_exp_w;
        if (residual_exp_w < common_exponent_w)
            common_exponent_w = residual_exp_w;
        if (destination_exp_w < common_exponent_w)
            common_exponent_w = destination_exp_w;

        accumulator_shift_w = 6'd0;
        residual_shift_w = 6'd0;
        destination_shift_w = 6'd0;
        accumulator_product_w = '0;
        residual_product_w = '0;
        accumulator_term_w = '0;
        residual_term_w = '0;
        numerator_sum_w = '0;
        numerator_overflow_w = 1'b0;
        numerator_s96_w = '0;
        numerator_magnitude_w = '0;
        denominator_w = '0;
        denominator_overflow_w = 1'b0;

        if (all_scales_valid_w) begin
            accumulator_shift_w = exponent_delta(accumulator_exp_w, common_exponent_w);
            residual_shift_w = exponent_delta(residual_exp_w, common_exponent_w);
            destination_shift_w = exponent_delta(destination_exp_w, common_exponent_w);
            accumulator_product_w =
                accumulator_s32_i * $signed({1'b0, accumulator_sig_w});
            residual_product_w =
                residual_s8_i * $signed({1'b0, residual_sig_w});
            accumulator_term_w =
                $signed({{48{accumulator_product_w[47]}}, accumulator_product_w})
                <<< accumulator_shift_w;
            residual_term_w =
                $signed({{72{residual_product_w[23]}}, residual_product_w})
                <<< residual_shift_w;
            numerator_sum_w =
                $signed({accumulator_term_w[95], accumulator_term_w}) +
                $signed({residual_term_w[95], residual_term_w});
            numerator_overflow_w = numerator_sum_w[96] != numerator_sum_w[95];
            numerator_s96_w = numerator_sum_w[95:0];
            numerator_magnitude_w = numerator_s96_w[95] ?
                $unsigned(-numerator_s96_w) : $unsigned(numerator_s96_w);
            denominator_w = {{48{1'b0}}, destination_sig_w}
                            << destination_shift_w;
            denominator_overflow_w = denominator_w == 64'd0;
        end

        doubled_magnitude_w = {numerator_magnitude_q, 1'b0};
        positive_threshold_w = multiply_u64_by_255(denominator_q);
        negative_threshold_w = multiply_u64_by_257(denominator_q);
        preclamp_positive_w = !numerator_negative_q &&
                              (doubled_magnitude_w >= positive_threshold_w);
        preclamp_negative_w = numerator_negative_q &&
                              (doubled_magnitude_w > negative_threshold_w);

        shifted_denominator_w = {{32{1'b0}}, denominator_q} << divide_bit_q;
        remainder_next_w = remainder_q;
        quotient_next_w = quotient_q;
        if (remainder_q >= shifted_denominator_w) begin
            remainder_next_w = remainder_q - shifted_denominator_w;
            quotient_next_w[divide_bit_q] = 1'b1;
        end

        doubled_remainder_w = {remainder_q, 1'b0};
        extended_denominator_w = {33'd0, denominator_q};
        round_increment_w =
            (doubled_remainder_w > extended_denominator_w) ||
            ((doubled_remainder_w == extended_denominator_w) && quotient_q[0]);
        rounded_magnitude_w = {1'b0, quotient_q} +
                              {{8{1'b0}}, round_increment_w};
        final_positive_saturation_w = preclamp_positive_q ||
            (!numerator_negative_q && (rounded_magnitude_w > 9'd127));
        final_negative_saturation_w = preclamp_negative_q ||
            (numerator_negative_q && (rounded_magnitude_w > 9'd128));
        rounded_s8_w = numerator_negative_q ?
            -$signed(rounded_magnitude_w[7:0]) :
             $signed(rounded_magnitude_w[7:0]);
    end

    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            state_q <= DPRF_IDLE;
            out_valid_q <= 1'b0;
            fused_q <= '0;
            positive_saturation_q <= 1'b0;
            negative_saturation_q <= 1'b0;
            descriptor_error_q <= 1'b0;
            numeric_overflow_q <= 1'b0;
            numerator_q <= '0;
            numerator_magnitude_q <= '0;
            denominator_q <= '0;
            common_exponent_q <= '0;
            numerator_negative_q <= 1'b0;
            remainder_q <= '0;
            quotient_q <= '0;
            divide_bit_q <= '0;
            latency_cycles_q <= '0;
            preclamp_positive_q <= 1'b0;
            preclamp_negative_q <= 1'b0;
        end else if (clear_i) begin
            state_q <= DPRF_IDLE;
            out_valid_q <= 1'b0;
            fused_q <= '0;
            positive_saturation_q <= 1'b0;
            negative_saturation_q <= 1'b0;
            descriptor_error_q <= 1'b0;
            numeric_overflow_q <= 1'b0;
            numerator_q <= '0;
            numerator_magnitude_q <= '0;
            denominator_q <= '0;
            common_exponent_q <= '0;
            numerator_negative_q <= 1'b0;
            remainder_q <= '0;
            quotient_q <= '0;
            divide_bit_q <= '0;
            latency_cycles_q <= '0;
            preclamp_positive_q <= 1'b0;
            preclamp_negative_q <= 1'b0;
        end else begin
            if (out_valid_q && out_ready_i)
                out_valid_q <= 1'b0;

            case (state_q)
                DPRF_IDLE: begin
                    if (start_valid_i && start_ready_o) begin
                        fused_q <= '0;
                        positive_saturation_q <= 1'b0;
                        negative_saturation_q <= 1'b0;
                        descriptor_error_q <= 1'b0;
                        numeric_overflow_q <= 1'b0;
                        numerator_q <= '0;
                        denominator_q <= '0;
                        common_exponent_q <= '0;
                        latency_cycles_q <= '0;
                        if (!all_scales_valid_w) begin
                            descriptor_error_q <= 1'b1;
                            out_valid_q <= 1'b1;
                        end else if (numerator_overflow_w || denominator_overflow_w) begin
                            numeric_overflow_q <= 1'b1;
                            out_valid_q <= 1'b1;
                        end else begin
                            numerator_q <= numerator_s96_w;
                            numerator_magnitude_q <= numerator_magnitude_w;
                            denominator_q <= denominator_w;
                            common_exponent_q <= common_exponent_w;
                            numerator_negative_q <= numerator_s96_w[95];
                            state_q <= DPRF_PREPARE;
                        end
                    end
                end

                DPRF_PREPARE: begin
                    preclamp_positive_q <= preclamp_positive_w;
                    preclamp_negative_q <= preclamp_negative_w;
                    remainder_q <= (preclamp_positive_w || preclamp_negative_w) ?
                                   96'd0 : numerator_magnitude_q;
                    quotient_q <= '0;
                    divide_bit_q <= 3'd7;
                    latency_cycles_q <= latency_cycles_q + 5'd1;
                    state_q <= DPRF_DIVIDE;
                end

                DPRF_DIVIDE: begin
                    remainder_q <= remainder_next_w;
                    quotient_q <= quotient_next_w;
                    latency_cycles_q <= latency_cycles_q + 5'd1;
                    if (divide_bit_q == 3'd0)
                        state_q <= DPRF_FINALIZE;
                    else
                        divide_bit_q <= divide_bit_q - 3'd1;
                end

                DPRF_FINALIZE: begin
                    positive_saturation_q <= final_positive_saturation_w;
                    negative_saturation_q <= final_negative_saturation_w;
                    if (final_positive_saturation_w)
                        fused_q <= 8'sd127;
                    else if (final_negative_saturation_w)
                        fused_q <= -8'sd128;
                    else
                        fused_q <= rounded_s8_w;
                    latency_cycles_q <= latency_cycles_q + 5'd1;
                    out_valid_q <= 1'b1;
                    state_q <= DPRF_IDLE;
                end

                default: begin
                    state_q <= DPRF_IDLE;
                    numeric_overflow_q <= 1'b1;
                    out_valid_q <= 1'b1;
                end
            endcase
        end
    end
endmodule

`default_nettype wire
