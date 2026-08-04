`timescale 1ns/1ps
`default_nettype none
/* verilator lint_off DECLFILENAME */

// Frozen contract: shared_qk_residual_cross_term_attention_v1.
// The sidecar is a client of the already-planned shared unsigned divider.  It
// deliberately contains no division operator and owns one shared multiplier.
module ace2_qk_residual_sidecar_core (
    input  logic                      clk_i,
    input  logic                      rst_ni,
    input  logic                      clear_i,
    input  logic                      start_valid_i,
    output logic                      start_ready_o,
    input  logic                      operation_rope_i,

    input  logic signed [31:0]        accumulator_s32_i,
    input  logic signed [31:0]        multiplier_s32_i,
    input  logic        [5:0]         shift_u6_i,
    input  logic        [31:0]        baseline_scale32_i,
    input  logic        [31:0]        residual_scale32_i,

    input  logic signed [3:0]         residual_real_s4_i,
    input  logic signed [3:0]         residual_imag_s4_i,
    input  logic signed [15:0]        cosine_q1_15_i,
    input  logic signed [15:0]        sine_q1_15_i,

    output logic                      div_req_valid_o,
    input  logic                      div_req_ready_i,
    output logic        [127:0]       div_numerator_u128_o,
    output logic        [127:0]       div_denominator_u128_o,
    input  logic                      div_rsp_valid_i,
    output logic                      div_rsp_ready_o,
    input  logic        [127:0]       div_quotient_u128_i,
    input  logic        [127:0]       div_remainder_u128_i,

    output logic                      out_valid_o,
    input  logic                      out_ready_i,
    output logic signed [7:0]         baseline_q8_o,
    output logic signed [3:0]         residual_s4_o,
    output logic signed [7:0]         residual_real_s8_o,
    output logic signed [7:0]         residual_imag_s8_o,
    output logic                      positive_clamp_o,
    output logic                      negative_clamp_o,
    output logic                      descriptor_error_o,
    output logic                      numeric_overflow_o
);
    typedef enum logic [4:0] {
        ST_IDLE,
        ST_PROJ_PRODUCT,
        ST_PROJ_QUANTIZE,
        ST_PROJ_NUM0,
        ST_PROJ_NUM1,
        ST_PROJ_NUM2,
        ST_PROJ_DIV_PREP,
        ST_PROJ_DIV_REQ,
        ST_PROJ_DIV_WAIT,
        ST_ROPE_RC,
        ST_ROPE_IS,
        ST_ROPE_IC,
        ST_ROPE_RS,
        ST_ROPE_FINISH
    } state_t;

    state_t state_q;
    logic out_valid_q;
    logic descriptor_error_q, numeric_overflow_q;
    logic signed [7:0] baseline_q8_q, residual_real_s8_q, residual_imag_s8_q;
    logic signed [3:0] residual_s4_q;
    logic positive_clamp_q, negative_clamp_q;

    logic signed [31:0] accumulator_q, multiplier_q;
    logic [5:0] shift_q;
    logic [15:0] baseline_sig_q, residual_sig_q;
    logic signed [7:0] baseline_exp_q, residual_exp_q;
    logic signed [3:0] residual_real_s4_q, residual_imag_s4_q;
    logic signed [15:0] cosine_q, sine_q;

    logic signed [63:0] product_q;
    logic signed [71:0] error_q;
    logic error_negative_q;
    logic [127:0] numerator_magnitude_q;
    logic [127:0] div_numerator_q, div_denominator_q;
    logic signed [21:0] rope_rc_q, rope_is_q, rope_ic_q, rope_rs_q;

    logic [31:0] shared_mul_a_u32_w, shared_mul_b_u32_w;
    logic [63:0] shared_mul_product_u64_w;
    logic shared_mul_negative_w;
    logic signed [64:0] shared_mul_signed_s65_w;
    logic baseline_scale_valid_w, residual_scale_valid_w;
    logic signed [8:0] delta_w;
    logic signed [71:0] baseline_rounded_w;
    logic signed [7:0] baseline_saturated_w;
    logic signed [71:0] error_w;
    logic [71:0] error_magnitude_w;
    logic [128:0] doubled_remainder_w, extended_denominator_w;
    logic round_div_increment_w;
    logic [128:0] rounded_quotient_w;
    logic signed [21:0] rope_real_acc_w, rope_imag_acc_w;
    logic signed [21:0] rope_real_round_w, rope_imag_round_w;

    function automatic logic [31:0] abs_s32(input logic signed [31:0] value);
        begin
            abs_s32 = value[31] ? $unsigned(-value) : $unsigned(value);
        end
    endfunction

    function automatic logic [15:0] abs_s16(input logic signed [15:0] value);
        begin
            abs_s16 = value[15] ? $unsigned(-value) : $unsigned(value);
        end
    endfunction

    function automatic logic [3:0] abs_s4(input logic signed [3:0] value);
        begin
            abs_s4 = value[3] ? $unsigned(-value) : $unsigned(value);
        end
    endfunction

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
                mask = ({72{1'b1}} >> (72-shift));
                remainder = magnitude & mask;
                half = 72'd1 << (shift-1);
                if ((remainder > half) || ((remainder == half) && quotient[0]))
                    quotient = quotient + 72'd1;
                round_shift_even_s72 = sign ? -$signed(quotient) : $signed(quotient);
            end
        end
    endfunction

    function automatic logic signed [21:0] round_q15_s22(
        input logic signed [21:0] value
    );
        logic sign;
        logic [21:0] magnitude, quotient;
        logic [14:0] remainder;
        begin
            sign = value[21];
            magnitude = sign ? $unsigned(-value) : $unsigned(value);
            quotient = magnitude >> 15;
            remainder = magnitude[14:0];
            if ((remainder > 15'd16384) ||
                ((remainder == 15'd16384) && quotient[0]))
                quotient = quotient + 22'd1;
            round_q15_s22 = sign ? -$signed(quotient) : $signed(quotient);
        end
    endfunction

    always @* begin
        shared_mul_a_u32_w = 32'd0;
        shared_mul_b_u32_w = 32'd0;
        shared_mul_negative_w = 1'b0;
        case (state_q)
            ST_PROJ_PRODUCT: begin
                shared_mul_a_u32_w = abs_s32(accumulator_q);
                shared_mul_b_u32_w = $unsigned(multiplier_q);
                shared_mul_negative_w = accumulator_q[31];
            end
            ST_PROJ_NUM0: begin
                shared_mul_a_u32_w = error_magnitude_w[31:0];
                shared_mul_b_u32_w = {16'd0, baseline_sig_q};
            end
            ST_PROJ_NUM1: begin
                shared_mul_a_u32_w = error_magnitude_w[63:32];
                shared_mul_b_u32_w = {16'd0, baseline_sig_q};
            end
            ST_PROJ_NUM2: begin
                shared_mul_a_u32_w = {24'd0, error_magnitude_w[71:64]};
                shared_mul_b_u32_w = {16'd0, baseline_sig_q};
            end
            ST_ROPE_RC: begin
                shared_mul_a_u32_w = {28'd0, abs_s4(residual_real_s4_q)};
                shared_mul_b_u32_w = {16'd0, abs_s16(cosine_q)};
                shared_mul_negative_w = residual_real_s4_q[3] ^ cosine_q[15];
            end
            ST_ROPE_IS: begin
                shared_mul_a_u32_w = {28'd0, abs_s4(residual_imag_s4_q)};
                shared_mul_b_u32_w = {16'd0, abs_s16(sine_q)};
                shared_mul_negative_w = residual_imag_s4_q[3] ^ sine_q[15];
            end
            ST_ROPE_IC: begin
                shared_mul_a_u32_w = {28'd0, abs_s4(residual_imag_s4_q)};
                shared_mul_b_u32_w = {16'd0, abs_s16(cosine_q)};
                shared_mul_negative_w = residual_imag_s4_q[3] ^ cosine_q[15];
            end
            ST_ROPE_RS: begin
                shared_mul_a_u32_w = {28'd0, abs_s4(residual_real_s4_q)};
                shared_mul_b_u32_w = {16'd0, abs_s16(sine_q)};
                shared_mul_negative_w = residual_real_s4_q[3] ^ sine_q[15];
            end
            default: begin end
        endcase
    end

    // The sole multiplier operator in this module.
    assign shared_mul_product_u64_w = shared_mul_a_u32_w * shared_mul_b_u32_w;
    assign shared_mul_signed_s65_w = shared_mul_negative_w ?
        -$signed({1'b0, shared_mul_product_u64_w}) :
         $signed({1'b0, shared_mul_product_u64_w});

    always @* begin
        baseline_scale_valid_w =
            (baseline_scale32_i[31:24] == 8'd0) &&
            (baseline_scale32_i[15:0] >= 16'h8000) &&
            ($signed(baseline_scale32_i[23:16]) >= -8'sd24) &&
            ($signed(baseline_scale32_i[23:16]) <= 8'sd4);
        residual_scale_valid_w =
            (residual_scale32_i[31:24] == 8'd0) &&
            (residual_scale32_i[15:0] >= 16'h8000) &&
            ($signed(residual_scale32_i[23:16]) >= -8'sd24) &&
            ($signed(residual_scale32_i[23:16]) <= 8'sd4);
        delta_w = {baseline_exp_q[7], baseline_exp_q} -
                  {residual_exp_q[7], residual_exp_q} -
                  $signed({3'd0, shift_q});
        baseline_rounded_w = round_shift_even_s72({{8{product_q[63]}}, product_q}, shift_q);
        if (baseline_rounded_w > 72'sd127)
            baseline_saturated_w = 8'sd127;
        else if (baseline_rounded_w < -72'sd128)
            baseline_saturated_w = -8'sd128;
        else
            baseline_saturated_w = baseline_rounded_w[7:0];
        error_w = {{8{product_q[63]}}, product_q} -
                  ({{64{baseline_saturated_w[7]}}, baseline_saturated_w} <<< shift_q);
        error_magnitude_w = error_q[71] ? $unsigned(-error_q) : $unsigned(error_q);
        doubled_remainder_w = {div_remainder_u128_i, 1'b0};
        extended_denominator_w = {1'b0, div_denominator_q};
        round_div_increment_w =
            (doubled_remainder_w > extended_denominator_w) ||
            ((doubled_remainder_w == extended_denominator_w) &&
             div_quotient_u128_i[0]);
        rounded_quotient_w = {1'b0, div_quotient_u128_i} +
                             {{128{1'b0}}, round_div_increment_w};
        rope_real_acc_w = $signed(rope_rc_q[21:0]) - $signed(rope_is_q[21:0]);
        rope_imag_acc_w = $signed(rope_ic_q[21:0]) + $signed(rope_rs_q[21:0]);
        rope_real_round_w = round_q15_s22(rope_real_acc_w);
        rope_imag_round_w = round_q15_s22(rope_imag_acc_w);
    end

    assign start_ready_o = (state_q == ST_IDLE) && !out_valid_q;
    assign div_req_valid_o = (state_q == ST_PROJ_DIV_REQ);
    assign div_numerator_u128_o = div_numerator_q;
    assign div_denominator_u128_o = div_denominator_q;
    assign div_rsp_ready_o = (state_q == ST_PROJ_DIV_WAIT);
    assign out_valid_o = out_valid_q;
    assign baseline_q8_o = baseline_q8_q;
    assign residual_s4_o = residual_s4_q;
    assign residual_real_s8_o = residual_real_s8_q;
    assign residual_imag_s8_o = residual_imag_s8_q;
    assign positive_clamp_o = positive_clamp_q;
    assign negative_clamp_o = negative_clamp_q;
    assign descriptor_error_o = descriptor_error_q;
    assign numeric_overflow_o = numeric_overflow_q;

    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            state_q <= ST_IDLE;
            out_valid_q <= 1'b0;
            descriptor_error_q <= 1'b0;
            numeric_overflow_q <= 1'b0;
            baseline_q8_q <= '0;
            residual_s4_q <= '0;
            residual_real_s8_q <= '0;
            residual_imag_s8_q <= '0;
            positive_clamp_q <= 1'b0;
            negative_clamp_q <= 1'b0;
            accumulator_q <= '0;
            multiplier_q <= '0;
            shift_q <= '0;
            baseline_sig_q <= '0;
            residual_sig_q <= '0;
            baseline_exp_q <= '0;
            residual_exp_q <= '0;
            residual_real_s4_q <= '0;
            residual_imag_s4_q <= '0;
            cosine_q <= '0;
            sine_q <= '0;
            product_q <= '0;
            error_q <= '0;
            error_negative_q <= 1'b0;
            numerator_magnitude_q <= '0;
            div_numerator_q <= '0;
            div_denominator_q <= '0;
            rope_rc_q <= '0;
            rope_is_q <= '0;
            rope_ic_q <= '0;
            rope_rs_q <= '0;
        end else if (clear_i) begin
            state_q <= ST_IDLE;
            out_valid_q <= 1'b0;
            descriptor_error_q <= 1'b0;
            numeric_overflow_q <= 1'b0;
            positive_clamp_q <= 1'b0;
            negative_clamp_q <= 1'b0;
        end else begin
            if (out_valid_q && out_ready_i)
                out_valid_q <= 1'b0;
            case (state_q)
                ST_IDLE: begin
                    if (start_valid_i && start_ready_o) begin
                        descriptor_error_q <= 1'b0;
                        numeric_overflow_q <= 1'b0;
                        positive_clamp_q <= 1'b0;
                        negative_clamp_q <= 1'b0;
                        baseline_q8_q <= '0;
                        residual_s4_q <= '0;
                        residual_real_s8_q <= '0;
                        residual_imag_s8_q <= '0;
                        if (operation_rope_i) begin
                            if ((residual_real_s4_i == -4'sd8) ||
                                (residual_imag_s4_i == -4'sd8)) begin
                                descriptor_error_q <= 1'b1;
                                out_valid_q <= 1'b1;
                            end else begin
                                residual_real_s4_q <= residual_real_s4_i;
                                residual_imag_s4_q <= residual_imag_s4_i;
                                cosine_q <= cosine_q1_15_i;
                                sine_q <= sine_q1_15_i;
                                state_q <= ST_ROPE_RC;
                            end
                        end else if (!(baseline_scale_valid_w && residual_scale_valid_w) ||
                                     (multiplier_s32_i <= 0)) begin
                            descriptor_error_q <= 1'b1;
                            out_valid_q <= 1'b1;
                        end else begin
                            accumulator_q <= accumulator_s32_i;
                            multiplier_q <= multiplier_s32_i;
                            shift_q <= shift_u6_i;
                            baseline_sig_q <= baseline_scale32_i[15:0];
                            residual_sig_q <= residual_scale32_i[15:0];
                            baseline_exp_q <= $signed(baseline_scale32_i[23:16]);
                            residual_exp_q <= $signed(residual_scale32_i[23:16]);
                            state_q <= ST_PROJ_PRODUCT;
                        end
                    end
                end
                ST_PROJ_PRODUCT: begin
                    if (shared_mul_signed_s65_w[64] != shared_mul_signed_s65_w[63]) begin
                        numeric_overflow_q <= 1'b1;
                        out_valid_q <= 1'b1;
                        state_q <= ST_IDLE;
                    end else begin
                        product_q <= shared_mul_signed_s65_w[63:0];
                        state_q <= ST_PROJ_QUANTIZE;
                    end
                end
                ST_PROJ_QUANTIZE: begin
                    if ((delta_w < -9'sd91) || (delta_w > 9'sd28)) begin
                        descriptor_error_q <= 1'b1;
                        out_valid_q <= 1'b1;
                        state_q <= ST_IDLE;
                    end else begin
                        baseline_q8_q <= baseline_saturated_w;
                        error_q <= error_w;
                        error_negative_q <= error_w[71];
                        numerator_magnitude_q <= '0;
                        state_q <= ST_PROJ_NUM0;
                    end
                end
                ST_PROJ_NUM0: begin
                    numerator_magnitude_q <= {{64{1'b0}}, shared_mul_product_u64_w};
                    state_q <= ST_PROJ_NUM1;
                end
                ST_PROJ_NUM1: begin
                    numerator_magnitude_q <= numerator_magnitude_q +
                                             ({64'd0, shared_mul_product_u64_w} << 32);
                    state_q <= ST_PROJ_NUM2;
                end
                ST_PROJ_NUM2: begin
                    numerator_magnitude_q <= numerator_magnitude_q +
                                             ({64'd0, shared_mul_product_u64_w} << 64);
                    state_q <= ST_PROJ_DIV_PREP;
                end
                ST_PROJ_DIV_PREP: begin
                    if (delta_w >= 0) begin
                        div_numerator_q <= numerator_magnitude_q << delta_w;
                        div_denominator_q <= {{112{1'b0}}, residual_sig_q};
                    end else begin
                        div_numerator_q <= numerator_magnitude_q;
                        div_denominator_q <= {{112{1'b0}}, residual_sig_q} << (-delta_w);
                    end
                    state_q <= ST_PROJ_DIV_REQ;
                end
                ST_PROJ_DIV_REQ: begin
                    if (div_req_ready_i)
                        state_q <= ST_PROJ_DIV_WAIT;
                end
                ST_PROJ_DIV_WAIT: begin
                    if (div_rsp_valid_i) begin
                        if (div_remainder_u128_i >= div_denominator_q) begin
                            numeric_overflow_q <= 1'b1;
                        end else if (rounded_quotient_w > 129'd7) begin
                            residual_s4_q <= error_negative_q ? -4'sd7 : 4'sd7;
                            positive_clamp_q <= !error_negative_q;
                            negative_clamp_q <= error_negative_q;
                        end else begin
                            residual_s4_q <= error_negative_q ?
                                -$signed(rounded_quotient_w[3:0]) :
                                 $signed(rounded_quotient_w[3:0]);
                        end
                        out_valid_q <= 1'b1;
                        state_q <= ST_IDLE;
                    end
                end
                ST_ROPE_RC: begin rope_rc_q <= shared_mul_signed_s65_w[21:0]; state_q <= ST_ROPE_IS; end
                ST_ROPE_IS: begin rope_is_q <= shared_mul_signed_s65_w[21:0]; state_q <= ST_ROPE_IC; end
                ST_ROPE_IC: begin rope_ic_q <= shared_mul_signed_s65_w[21:0]; state_q <= ST_ROPE_RS; end
                ST_ROPE_RS: begin rope_rs_q <= shared_mul_signed_s65_w[21:0]; state_q <= ST_ROPE_FINISH; end
                ST_ROPE_FINISH: begin
                    if ((rope_real_round_w > 22'sd127) || (rope_real_round_w < -22'sd128) ||
                        (rope_imag_round_w > 22'sd127) || (rope_imag_round_w < -22'sd128)) begin
                        numeric_overflow_q <= 1'b1;
                    end else begin
                        residual_real_s8_q <= rope_real_round_w[7:0];
                        residual_imag_s8_q <= rope_imag_round_w[7:0];
                    end
                    out_valid_q <= 1'b1;
                    state_q <= ST_IDLE;
                end
                default: state_q <= ST_IDLE;
            endcase
        end
    end
endmodule


// The unchanged baseline score path supplies its authoritative signed-Q20.44
// result.  This block sign-extends that result into the checked accumulator and
// owns one serialized multiplier for exactly the three residual cross terms.
module ace2_residual_cross_term_score_core #(
    parameter integer HEAD_DIM = 64
) (
    input  logic                      clk_i,
    input  logic                      rst_ni,
    input  logic                      clear_i,
    input  logic                      start_valid_i,
    output logic                      start_ready_o,
    input  logic [6:0]                lane_count_u7_i,
    input  logic signed [63:0]        base_score_q20_44_s64_i,
    input  logic [31:0]               query_scale32_i,
    input  logic [31:0]               key_scale32_i,
    input  logic [31:0]               query_residual_scale32_i,
    input  logic [31:0]               key_residual_scale32_i,
    input  logic                      lane_valid_i,
    output logic                      lane_ready_o,
    input  logic signed [7:0]         query_q8_i,
    input  logic signed [7:0]         key_q8_i,
    input  logic signed [7:0]         query_residual_s8_i,
    input  logic signed [7:0]         key_residual_s8_i,
    output logic                      score_valid_o,
    input  logic                      score_ready_i,
    output logic signed [63:0]        score_q20_44_s64_o,
    output logic signed [31:0]        dot_q_rk_s32_o,
    output logic signed [31:0]        dot_rq_k_s32_o,
    output logic signed [31:0]        dot_rq_rk_s32_o,
    output logic                      descriptor_error_o,
    output logic                      numeric_overflow_o
);
    localparam integer LANE_INDEX_W = (HEAD_DIM <= 2) ? 1 : $clog2(HEAD_DIM);
    localparam logic [6:0] HEAD_DIM_U7 = HEAD_DIM[6:0];
    typedef enum logic [3:0] {
        SC_IDLE,
        SC_LANE_ACCEPT,
        SC_LANE_MUL,
        SC_SCALE_SIG,
        SC_SCALE_WAIT,
        SC_SCALE_DOT
    } score_state_t;

    score_state_t state_q;
    logic score_valid_q, descriptor_error_q, numeric_overflow_q;
    logic signed [63:0] score_q;
    logic [LANE_INDEX_W-1:0] lane_index_q;
    logic [1:0] correction_index_q, term_index_q;
    logic [2:0] conversion_cycle_q;
    logic signed [7:0] key_q, query_residual_q, key_residual_q;
    logic signed [31:0] dot_q_rk_q, dot_rq_k_q, dot_rq_rk_q;
    logic [15:0] query_sig_q, key_sig_q, query_residual_sig_q, key_residual_sig_q;
    logic signed [7:0] query_exp_q, key_exp_q, query_residual_exp_q, key_residual_exp_q;
    logic [31:0] scale_sig_product_q;
    logic signed [66:0] scaled_accumulator_q;

    logic [31:0] shared_mul_a_u32_w, shared_mul_b_u32_w;
    logic [63:0] shared_mul_product_u64_w;
    logic shared_mul_negative_w;
    logic signed [31:0] selected_dot_w;
    logic [15:0] selected_sig_a_w, selected_sig_b_w;
    logic signed [7:0] selected_exp_a_w, selected_exp_b_w;
    integer selected_shift_w;
    logic [127:0] shifted_magnitude_w;
    logic signed [127:0] scaled_term_s128_w;
    logic signed [67:0] accumulator_next_s68_w;
    logic scale_records_valid_w;

    function automatic logic [7:0] abs_s8(input logic signed [7:0] value);
        begin
            abs_s8 = value[7] ? $unsigned(-value) : $unsigned(value);
        end
    endfunction

    function automatic logic [31:0] abs_s32(input logic signed [31:0] value);
        begin
            abs_s32 = value[31] ? $unsigned(-value) : $unsigned(value);
        end
    endfunction

    function automatic logic [127:0] round_shift_even_u128(
        input logic [127:0] value,
        input integer shift
    );
        logic [127:0] quotient, remainder, mask, half;
        begin
            if (shift <= 0) begin
                round_shift_even_u128 = value;
            end else if (shift >= 128) begin
                round_shift_even_u128 = '0;
            end else begin
                quotient = value >> shift;
                mask = ({128{1'b1}} >> (128-shift));
                remainder = value & mask;
                half = 128'd1 << (shift-1);
                if ((remainder > half) || ((remainder == half) && quotient[0]))
                    quotient = quotient + 128'd1;
                round_shift_even_u128 = quotient;
            end
        end
    endfunction

    function automatic logic scale32_valid(input logic [31:0] record);
        begin
            scale32_valid = (record[31:24] == 8'd0) &&
                            (record[15:0] >= 16'h8000) &&
                            ($signed(record[23:16]) >= -8'sd24) &&
                            ($signed(record[23:16]) <= 8'sd4);
        end
    endfunction

    always @* begin
        selected_dot_w = dot_q_rk_q;
        selected_sig_a_w = query_sig_q;
        selected_sig_b_w = key_residual_sig_q;
        selected_exp_a_w = query_exp_q;
        selected_exp_b_w = key_residual_exp_q;
        case (term_index_q)
            2'd1: begin
                selected_dot_w = dot_rq_k_q;
                selected_sig_a_w = query_residual_sig_q;
                selected_sig_b_w = key_sig_q;
                selected_exp_a_w = query_residual_exp_q;
                selected_exp_b_w = key_exp_q;
            end
            2'd2: begin
                selected_dot_w = dot_rq_rk_q;
                selected_sig_a_w = query_residual_sig_q;
                selected_sig_b_w = key_residual_sig_q;
                selected_exp_a_w = query_residual_exp_q;
                selected_exp_b_w = key_residual_exp_q;
            end
            default: begin end
        endcase
        selected_shift_w = {{24{selected_exp_a_w[7]}}, selected_exp_a_w} +
                           {{24{selected_exp_b_w[7]}}, selected_exp_b_w} + 11;
    end

    always @* begin
        shared_mul_a_u32_w = 32'd0;
        shared_mul_b_u32_w = 32'd0;
        shared_mul_negative_w = 1'b0;
        if (state_q == SC_LANE_ACCEPT) begin
            shared_mul_a_u32_w = {24'd0, abs_s8(query_q8_i)};
            shared_mul_b_u32_w = {24'd0, abs_s8(key_residual_s8_i)};
            shared_mul_negative_w = query_q8_i[7] ^ key_residual_s8_i[7];
        end else if (state_q == SC_LANE_MUL) begin
            case (correction_index_q)
                2'd1: begin
                    shared_mul_a_u32_w = {24'd0, abs_s8(query_residual_q)};
                    shared_mul_b_u32_w = {24'd0, abs_s8(key_q)};
                    shared_mul_negative_w = query_residual_q[7] ^ key_q[7];
                end
                default: begin
                    shared_mul_a_u32_w = {24'd0, abs_s8(query_residual_q)};
                    shared_mul_b_u32_w = {24'd0, abs_s8(key_residual_q)};
                    shared_mul_negative_w = query_residual_q[7] ^ key_residual_q[7];
                end
            endcase
        end else if (state_q == SC_SCALE_SIG) begin
            shared_mul_a_u32_w = {16'd0, selected_sig_a_w};
            shared_mul_b_u32_w = {16'd0, selected_sig_b_w};
        end else if (state_q == SC_SCALE_DOT) begin
            shared_mul_a_u32_w = abs_s32(selected_dot_w);
            shared_mul_b_u32_w = scale_sig_product_q;
            shared_mul_negative_w = selected_dot_w[31];
        end
    end

    // The sole multiplier operator in this module.
    assign shared_mul_product_u64_w = shared_mul_a_u32_w * shared_mul_b_u32_w;

    always @* begin
        if (selected_shift_w >= 0)
            shifted_magnitude_w = {64'd0, shared_mul_product_u64_w} << selected_shift_w;
        else
            shifted_magnitude_w = round_shift_even_u128(
                {64'd0, shared_mul_product_u64_w}, -selected_shift_w
            );
        scaled_term_s128_w = shared_mul_negative_w ?
            -$signed(shifted_magnitude_w) : $signed(shifted_magnitude_w);
        accumulator_next_s68_w = {scaled_accumulator_q[66], scaled_accumulator_q} +
                                 scaled_term_s128_w[67:0];
        scale_records_valid_w = scale32_valid(query_scale32_i) &&
                                scale32_valid(key_scale32_i) &&
                                scale32_valid(query_residual_scale32_i) &&
                                scale32_valid(key_residual_scale32_i);
    end

    assign start_ready_o = (state_q == SC_IDLE) && !score_valid_q;
    assign lane_ready_o = (state_q == SC_LANE_ACCEPT);
    assign score_valid_o = score_valid_q;
    assign score_q20_44_s64_o = score_q;
    assign dot_q_rk_s32_o = dot_q_rk_q;
    assign dot_rq_k_s32_o = dot_rq_k_q;
    assign dot_rq_rk_s32_o = dot_rq_rk_q;
    assign descriptor_error_o = descriptor_error_q;
    assign numeric_overflow_o = numeric_overflow_q;

    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            state_q <= SC_IDLE;
            score_valid_q <= 1'b0;
            descriptor_error_q <= 1'b0;
            numeric_overflow_q <= 1'b0;
            score_q <= '0;
            lane_index_q <= '0;
            correction_index_q <= '0;
            term_index_q <= '0;
            conversion_cycle_q <= '0;
            key_q <= '0;
            query_residual_q <= '0;
            key_residual_q <= '0;
            dot_q_rk_q <= '0;
            dot_rq_k_q <= '0;
            dot_rq_rk_q <= '0;
            query_sig_q <= '0;
            key_sig_q <= '0;
            query_residual_sig_q <= '0;
            key_residual_sig_q <= '0;
            query_exp_q <= '0;
            key_exp_q <= '0;
            query_residual_exp_q <= '0;
            key_residual_exp_q <= '0;
            scale_sig_product_q <= '0;
            scaled_accumulator_q <= '0;
        end else if (clear_i) begin
            state_q <= SC_IDLE;
            score_valid_q <= 1'b0;
            descriptor_error_q <= 1'b0;
            numeric_overflow_q <= 1'b0;
        end else begin
            if (score_valid_q && score_ready_i)
                score_valid_q <= 1'b0;
            case (state_q)
                SC_IDLE: begin
                    if (start_valid_i && start_ready_o) begin
                        descriptor_error_q <= 1'b0;
                        numeric_overflow_q <= 1'b0;
                        score_q <= '0;
                        if ((lane_count_u7_i != HEAD_DIM_U7) || !scale_records_valid_w) begin
                            descriptor_error_q <= 1'b1;
                            score_valid_q <= 1'b1;
                        end else begin
                            query_sig_q <= query_scale32_i[15:0];
                            key_sig_q <= key_scale32_i[15:0];
                            query_residual_sig_q <= query_residual_scale32_i[15:0];
                            key_residual_sig_q <= key_residual_scale32_i[15:0];
                            query_exp_q <= $signed(query_scale32_i[23:16]);
                            key_exp_q <= $signed(key_scale32_i[23:16]);
                            query_residual_exp_q <= $signed(query_residual_scale32_i[23:16]);
                            key_residual_exp_q <= $signed(key_residual_scale32_i[23:16]);
                            dot_q_rk_q <= '0;
                            dot_rq_k_q <= '0;
                            dot_rq_rk_q <= '0;
                            lane_index_q <= '0;
                            correction_index_q <= '0;
                            term_index_q <= '0;
                            conversion_cycle_q <= '0;
                            scaled_accumulator_q <= {
                                {3{base_score_q20_44_s64_i[63]}},
                                base_score_q20_44_s64_i
                            };
                            state_q <= SC_LANE_ACCEPT;
                        end
                    end
                end
                SC_LANE_ACCEPT: begin
                    if (lane_valid_i) begin
                        dot_q_rk_q <= dot_q_rk_q +
                            (shared_mul_negative_w ? -$signed(shared_mul_product_u64_w[31:0]) :
                                                     $signed(shared_mul_product_u64_w[31:0]));
                        key_q <= key_q8_i;
                        query_residual_q <= query_residual_s8_i;
                        key_residual_q <= key_residual_s8_i;
                        correction_index_q <= 2'd1;
                        state_q <= SC_LANE_MUL;
                    end
                end
                SC_LANE_MUL: begin
                    case (correction_index_q)
                        2'd1: dot_rq_k_q <= dot_rq_k_q +
                            (shared_mul_negative_w ? -$signed(shared_mul_product_u64_w[31:0]) :
                                                     $signed(shared_mul_product_u64_w[31:0]));
                        default: dot_rq_rk_q <= dot_rq_rk_q +
                            (shared_mul_negative_w ? -$signed(shared_mul_product_u64_w[31:0]) :
                                                     $signed(shared_mul_product_u64_w[31:0]));
                    endcase
                    if (correction_index_q == 2'd2) begin
                        correction_index_q <= '0;
                        if (lane_index_q == LANE_INDEX_W'(HEAD_DIM-1)) begin
                            term_index_q <= 2'd0;
                            state_q <= SC_SCALE_SIG;
                        end else begin
                            lane_index_q <= lane_index_q + 1'b1;
                            state_q <= SC_LANE_ACCEPT;
                        end
                    end else begin
                        correction_index_q <= correction_index_q + 1'b1;
                    end
                end
                SC_SCALE_SIG: begin
                    scale_sig_product_q <= shared_mul_product_u64_w[31:0];
                    conversion_cycle_q <= 3'd1;
                    state_q <= SC_SCALE_WAIT;
                end
                // Each correction conversion owns eight non-overlapped cycles:
                // one significand multiply, six held pipeline cycles, and one
                // dot/scale multiply plus checked accumulation.
                SC_SCALE_WAIT: begin
                    if (conversion_cycle_q == 3'd6) begin
                        state_q <= SC_SCALE_DOT;
                    end else begin
                        conversion_cycle_q <= conversion_cycle_q + 1'b1;
                    end
                end
                SC_SCALE_DOT: begin
                    if ((scaled_term_s128_w[127:67] !=
                         {61{scaled_term_s128_w[66]}}) ||
                        (accumulator_next_s68_w[67] != accumulator_next_s68_w[66])) begin
                        numeric_overflow_q <= 1'b1;
                        score_valid_q <= 1'b1;
                        state_q <= SC_IDLE;
                    end else if (term_index_q == 2'd2) begin
                        if (accumulator_next_s68_w[66:64] !=
                            {3{accumulator_next_s68_w[63]}}) begin
                            numeric_overflow_q <= 1'b1;
                            score_q <= '0;
                        end else begin
                            score_q <= accumulator_next_s68_w[63:0];
                        end
                        score_valid_q <= 1'b1;
                        state_q <= SC_IDLE;
                    end else begin
                        scaled_accumulator_q <= accumulator_next_s68_w[66:0];
                        term_index_q <= term_index_q + 1'b1;
                        state_q <= SC_SCALE_SIG;
                    end
                end
                default: state_q <= SC_IDLE;
            endcase
        end
    end
endmodule

`default_nettype wire
