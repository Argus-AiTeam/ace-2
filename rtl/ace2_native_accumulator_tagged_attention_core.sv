`timescale 1ns/1ps
// ACE-2 shared native-accumulator tagged attention candidate.
// Exactly one explicit signed 32x32 multiplier is shared by RoPE and score.
/* verilator lint_off DECLFILENAME */

module ace2_native_accumulator_rope_core (
    input  logic                     clk_i,
    input  logic                     rst_ni,
    input  logic                     clear_i,
    input  logic                     in_valid_i,
    output logic                     in_ready_o,
    input  logic signed [31:0]       acc0_s32_i,
    input  logic        [31:0]       scale0_u32_i,
    input  logic signed [31:0]       acc1_s32_i,
    input  logic        [31:0]       scale1_u32_i,
    input  logic signed [15:0]       cosine_q1_15_i,
    input  logic signed [15:0]       sine_q1_15_i,
    output logic                     out_valid_o,
    input  logic                     out_ready_i,
    output logic signed [31:0]       real_mantissa_s32_o,
    output logic signed [7:0]        real_exponent_s8_o,
    output logic signed [31:0]       imag_mantissa_s32_o,
    output logic signed [7:0]        imag_exponent_s8_o,
    output logic                     descriptor_error_o,
    output logic                     numeric_overflow_o,
    input  logic                     score_mul_req_valid_i,
    output logic                     score_mul_req_ready_o,
    input  logic signed [31:0]       score_mul_operand_a_s32_i,
    input  logic signed [31:0]       score_mul_operand_b_s32_i,
    output logic                     score_mul_rsp_valid_o,
    input  logic                     score_mul_rsp_ready_i,
    output logic signed [63:0]       score_mul_product_s64_o
);

    typedef enum logic [3:0] {
        R_IDLE,
        R_T0_COEFF, R_T0_ACC,
        R_T1_COEFF, R_T1_ACC,
        R_T2_COEFF, R_T2_ACC,
        R_T3_COEFF, R_T3_ACC,
        R_CALC
    } rope_state_t;
    rope_state_t state_q;

    logic signed [31:0] acc0_q, acc1_q;
    logic [15:0] sig0_q, sig1_q;
    logic signed [7:0] exp0_q, exp1_q;
    logic signed [15:0] cosine_q, sine_q;
    logic signed [31:0] scale_coeff_q;
    logic signed [63:0] term0_cos_q, term1_sin_q, term1_cos_q, term0_sin_q;

    logic out_valid_q;
    logic signed [31:0] real_mantissa_q, imag_mantissa_q;
    logic signed [7:0] real_exponent_q, imag_exponent_q;
    logic descriptor_error_q, numeric_overflow_q;
    logic score_rsp_valid_q;
    logic signed [63:0] score_rsp_product_q;

    logic signed [31:0] shared_mul_a_w, shared_mul_b_w;
    logic signed [63:0] shared_mul_product_w;
    logic scale0_valid_w, scale1_valid_w;
    logic signed [8:0] common_input_exp_w;
    integer delta0_w, delta1_w;
    logic signed [67:0] aligned0_cos_w, aligned1_sin_w;
    logic signed [67:0] aligned1_cos_w, aligned0_sin_w;
    logic signed [67:0] real_w, imag_w;
    logic [40:0] real_tag_w, imag_tag_w;
    logic scale_coeff_overflow_w;

    // The only multiplier operator in candidate RTL.
    assign shared_mul_product_w = shared_mul_a_w * shared_mul_b_w;

    function automatic logic signed [67:0] rshift_rne_68(
        input logic signed [67:0] value,
        input integer shift
    );
        logic sign;
        logic [67:0] magnitude, quotient, remainder, mask, half;
        begin
            if (shift <= 0) begin
                rshift_rne_68 = value;
            end else if (shift >= 68) begin
                rshift_rne_68 = '0;
            end else begin
                sign = value[67];
                magnitude = sign ? $unsigned(-value) : $unsigned(value);
                quotient = magnitude >> shift;
                mask = ({68{1'b1}} >> (68-shift));
                remainder = magnitude & mask;
                half = 68'd1 << (shift-1);
                if ((remainder > half) || ((remainder == half) && quotient[0]))
                    quotient = quotient + 68'd1;
                rshift_rne_68 = sign ? -$signed(quotient) : $signed(quotient);
            end
        end
    endfunction

    // Return {overflow, exponent[7:0], mantissa[31:0]}.
    function automatic logic [40:0] normalize_tagged(
        input logic signed [67:0] value,
        input logic signed [8:0] base_exponent
    );
        logic [67:0] magnitude;
        logic signed [67:0] rounded;
        logic signed [9:0] exponent, shift_s10;
        integer msb, shift, index;
        logic overflow;
        begin
            if (value == 68'sd0) begin
                normalize_tagged = {1'b0, 8'sd0, 32'sd0};
            end else begin
                magnitude = value[67] ? $unsigned(-value) : $unsigned(value);
                msb = -1;
                for (index = 0; index < 68; index = index + 1)
                    if (magnitude[index]) msb = index;
                shift = (msb > 30) ? (msb - 30) : 0;
                rounded = rshift_rne_68(value, shift);
                if ((rounded > 68'sh0000000007fffffff) ||
                    (rounded < -68'sh00000000080000000)) begin
                    shift = shift + 1;
                    rounded = rshift_rne_68(value, shift);
                end
                shift_s10 = shift[9:0];
                exponent = {base_exponent[8], base_exponent} + shift_s10;
                overflow = (exponent < -10'sd96) || (exponent > 10'sd31) ||
                           (rounded > 68'sh0000000007fffffff) ||
                           (rounded < -68'sh00000000080000000);
                normalize_tagged = {overflow, exponent[7:0], rounded[31:0]};
            end
        end
    endfunction

    always @* begin
        shared_mul_a_w = score_mul_operand_a_s32_i;
        shared_mul_b_w = score_mul_operand_b_s32_i;
        case (state_q)
            R_T0_COEFF: begin shared_mul_a_w = $signed({16'd0, sig0_q}); shared_mul_b_w = {{16{cosine_q[15]}}, cosine_q}; end
            R_T0_ACC:   begin shared_mul_a_w = acc0_q; shared_mul_b_w = scale_coeff_q; end
            R_T1_COEFF: begin shared_mul_a_w = $signed({16'd0, sig1_q}); shared_mul_b_w = {{16{sine_q[15]}}, sine_q}; end
            R_T1_ACC:   begin shared_mul_a_w = acc1_q; shared_mul_b_w = scale_coeff_q; end
            R_T2_COEFF: begin shared_mul_a_w = $signed({16'd0, sig1_q}); shared_mul_b_w = {{16{cosine_q[15]}}, cosine_q}; end
            R_T2_ACC:   begin shared_mul_a_w = acc1_q; shared_mul_b_w = scale_coeff_q; end
            R_T3_COEFF: begin shared_mul_a_w = $signed({16'd0, sig0_q}); shared_mul_b_w = {{16{sine_q[15]}}, sine_q}; end
            R_T3_ACC:   begin shared_mul_a_w = acc0_q; shared_mul_b_w = scale_coeff_q; end
            default: begin end
        endcase
    end

    always @* begin
        scale0_valid_w = (scale0_u32_i[31:24] == 8'd0) &&
                         (scale0_u32_i[15:0] >= 16'h8000) &&
                         ($signed(scale0_u32_i[23:16]) >= -8'sd24) &&
                         ($signed(scale0_u32_i[23:16]) <= 8'sd4);
        scale1_valid_w = (scale1_u32_i[31:24] == 8'd0) &&
                         (scale1_u32_i[15:0] >= 16'h8000) &&
                         ($signed(scale1_u32_i[23:16]) >= -8'sd24) &&
                         ($signed(scale1_u32_i[23:16]) <= 8'sd4);
        common_input_exp_w = (exp0_q >= exp1_q) ?
                             {exp0_q[7], exp0_q} : {exp1_q[7], exp1_q};
        delta0_w = {{23{common_input_exp_w[8]}}, common_input_exp_w};
        delta0_w = delta0_w - {{24{exp0_q[7]}}, exp0_q};
        delta1_w = {{23{common_input_exp_w[8]}}, common_input_exp_w};
        delta1_w = delta1_w - {{24{exp1_q[7]}}, exp1_q};
        aligned0_cos_w = rshift_rne_68({{4{term0_cos_q[63]}}, term0_cos_q}, delta0_w);
        aligned1_sin_w = rshift_rne_68({{4{term1_sin_q[63]}}, term1_sin_q}, delta1_w);
        aligned1_cos_w = rshift_rne_68({{4{term1_cos_q[63]}}, term1_cos_q}, delta1_w);
        aligned0_sin_w = rshift_rne_68({{4{term0_sin_q[63]}}, term0_sin_q}, delta0_w);
        real_w = aligned0_cos_w - aligned1_sin_w;
        imag_w = aligned1_cos_w + aligned0_sin_w;
        real_tag_w = normalize_tagged(real_w, common_input_exp_w - 9'sd30);
        imag_tag_w = normalize_tagged(imag_w, common_input_exp_w - 9'sd30);
        scale_coeff_overflow_w =
            shared_mul_product_w[63:32] != {32{shared_mul_product_w[31]}};
    end

    assign in_ready_o = (state_q == R_IDLE) &&
                        (!out_valid_q || out_ready_i) && !score_rsp_valid_q;
    assign score_mul_req_ready_o = (state_q == R_IDLE) &&
                                   !in_valid_i && !score_rsp_valid_q;
    assign out_valid_o = out_valid_q;
    assign real_mantissa_s32_o = real_mantissa_q;
    assign real_exponent_s8_o = real_exponent_q;
    assign imag_mantissa_s32_o = imag_mantissa_q;
    assign imag_exponent_s8_o = imag_exponent_q;
    assign descriptor_error_o = descriptor_error_q;
    assign numeric_overflow_o = numeric_overflow_q;
    assign score_mul_rsp_valid_o = score_rsp_valid_q;
    assign score_mul_product_s64_o = score_rsp_product_q;

    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            state_q <= R_IDLE;
            out_valid_q <= 1'b0;
            descriptor_error_q <= 1'b0;
            numeric_overflow_q <= 1'b0;
            score_rsp_valid_q <= 1'b0;
            acc0_q <= '0; acc1_q <= '0; sig0_q <= '0; sig1_q <= '0;
            exp0_q <= '0; exp1_q <= '0; cosine_q <= '0; sine_q <= '0;
            scale_coeff_q <= '0;
            term0_cos_q <= '0; term1_sin_q <= '0; term1_cos_q <= '0; term0_sin_q <= '0;
            real_mantissa_q <= '0; real_exponent_q <= '0;
            imag_mantissa_q <= '0; imag_exponent_q <= '0;
            score_rsp_product_q <= '0;
        end else if (clear_i) begin
            state_q <= R_IDLE;
            out_valid_q <= 1'b0;
            descriptor_error_q <= 1'b0;
            numeric_overflow_q <= 1'b0;
            score_rsp_valid_q <= 1'b0;
        end else begin
            if (out_valid_q && out_ready_i)
                out_valid_q <= 1'b0;
            if (score_rsp_valid_q && score_mul_rsp_ready_i)
                score_rsp_valid_q <= 1'b0;
            case (state_q)
                R_IDLE: begin
                    if (in_valid_i && in_ready_o) begin
                        if (!(scale0_valid_w && scale1_valid_w)) begin
                            out_valid_q <= 1'b1;
                            descriptor_error_q <= 1'b1;
                            numeric_overflow_q <= 1'b0;
                            real_mantissa_q <= '0; real_exponent_q <= '0;
                            imag_mantissa_q <= '0; imag_exponent_q <= '0;
                        end else begin
                            acc0_q <= acc0_s32_i;
                            acc1_q <= acc1_s32_i;
                            sig0_q <= scale0_u32_i[15:0];
                            sig1_q <= scale1_u32_i[15:0];
                            exp0_q <= $signed(scale0_u32_i[23:16]);
                            exp1_q <= $signed(scale1_u32_i[23:16]);
                            cosine_q <= cosine_q1_15_i;
                            sine_q <= sine_q1_15_i;
                            descriptor_error_q <= 1'b0;
                            numeric_overflow_q <= 1'b0;
                            state_q <= R_T0_COEFF;
                        end
                    end else if (score_mul_req_valid_i && score_mul_req_ready_o) begin
                        score_rsp_product_q <= shared_mul_product_w;
                        score_rsp_valid_q <= 1'b1;
                    end
                end
                R_T0_COEFF: begin
                    scale_coeff_q <= shared_mul_product_w[31:0];
                    numeric_overflow_q <= numeric_overflow_q || scale_coeff_overflow_w;
                    state_q <= R_T0_ACC;
                end
                R_T0_ACC: begin term0_cos_q <= shared_mul_product_w; state_q <= R_T1_COEFF; end
                R_T1_COEFF: begin
                    scale_coeff_q <= shared_mul_product_w[31:0];
                    numeric_overflow_q <= numeric_overflow_q || scale_coeff_overflow_w;
                    state_q <= R_T1_ACC;
                end
                R_T1_ACC: begin term1_sin_q <= shared_mul_product_w; state_q <= R_T2_COEFF; end
                R_T2_COEFF: begin
                    scale_coeff_q <= shared_mul_product_w[31:0];
                    numeric_overflow_q <= numeric_overflow_q || scale_coeff_overflow_w;
                    state_q <= R_T2_ACC;
                end
                R_T2_ACC: begin term1_cos_q <= shared_mul_product_w; state_q <= R_T3_COEFF; end
                R_T3_COEFF: begin
                    scale_coeff_q <= shared_mul_product_w[31:0];
                    numeric_overflow_q <= numeric_overflow_q || scale_coeff_overflow_w;
                    state_q <= R_T3_ACC;
                end
                R_T3_ACC: begin term0_sin_q <= shared_mul_product_w; state_q <= R_CALC; end
                R_CALC: begin
                    real_mantissa_q <= real_tag_w[31:0];
                    real_exponent_q <= real_tag_w[39:32];
                    imag_mantissa_q <= imag_tag_w[31:0];
                    imag_exponent_q <= imag_tag_w[39:32];
                    numeric_overflow_q <= numeric_overflow_q || real_tag_w[40] || imag_tag_w[40];
                    out_valid_q <= 1'b1;
                    state_q <= R_IDLE;
                end
                default: state_q <= R_IDLE;
            endcase
        end
    end
endmodule


module ace2_tagged_attention_score_core #(
    parameter integer MAX_LANES = 64
) (
    input  logic                     clk_i,
    input  logic                     rst_ni,
    input  logic                     clear_i,
    input  logic                     start_valid_i,
    output logic                     start_ready_o,
    input  logic [6:0]               lane_count_u7_i,
    input  logic                     lane_valid_i,
    output logic                     lane_ready_o,
    input  logic signed [31:0]       query_mantissa_s32_i,
    input  logic signed [7:0]        query_exponent_s8_i,
    input  logic signed [31:0]       key_mantissa_s32_i,
    input  logic signed [7:0]        key_exponent_s8_i,
    output logic                     mul_req_valid_o,
    input  logic                     mul_req_ready_i,
    output logic signed [31:0]       mul_operand_a_s32_o,
    output logic signed [31:0]       mul_operand_b_s32_o,
    input  logic                     mul_rsp_valid_i,
    output logic                     mul_rsp_ready_o,
    input  logic signed [63:0]       mul_product_s64_i,
    output logic                     score_valid_o,
    input  logic                     score_ready_i,
    output logic signed [63:0]       score_q20_44_s64_o,
    output logic                     descriptor_error_o,
    output logic                     numeric_overflow_o
);

    localparam integer INDEX_W = (MAX_LANES <= 2) ? 1 : $clog2(MAX_LANES);
    localparam logic [6:0] MAX_LANES_U7 = MAX_LANES[6:0];
    typedef enum logic [2:0] {S_IDLE, S_LANE, S_MUL_REQ, S_MUL_RSP, S_ACCUM, S_EMIT} score_state_t;
    score_state_t state_q;

    logic signed [63:0] product_mem [0:MAX_LANES-1];
    logic signed [8:0] exponent_mem [0:MAX_LANES-1];
    logic [6:0] lane_count_q;
    logic [INDEX_W-1:0] index_q;
    logic signed [31:0] query_mantissa_q, key_mantissa_q;
    logic signed [8:0] pending_exponent_q;
    logic signed [8:0] max_exponent_q, selected_exponent_w;
    logic signed [127:0] accumulator_q, aligned_product_w, accumulator_next_w;
    logic signed [63:0] score_q;
    logic descriptor_error_q, numeric_overflow_q;
    logic lane_exponent_valid_w;
    logic [64:0] converted_score_w;
    integer score_align_shift_w;

    function automatic logic signed [127:0] rshift_rne_128(
        input logic signed [127:0] value,
        input integer shift
    );
        logic sign;
        logic [127:0] magnitude, quotient, remainder, mask, half;
        begin
            if (shift <= 0) begin
                rshift_rne_128 = value;
            end else if (shift >= 128) begin
                rshift_rne_128 = '0;
            end else begin
                sign = value[127];
                magnitude = sign ? $unsigned(-value) : $unsigned(value);
                quotient = magnitude >> shift;
                mask = ({128{1'b1}} >> (128-shift));
                remainder = magnitude & mask;
                half = 128'd1 << (shift-1);
                if ((remainder > half) || ((remainder == half) && quotient[0]))
                    quotient = quotient + 128'd1;
                rshift_rne_128 = sign ? -$signed(quotient) : $signed(quotient);
            end
        end
    endfunction

    function automatic logic [64:0] convert_q20_44(
        input logic signed [127:0] accumulator,
        input logic signed [8:0] common_exponent
    );
        integer shift;
        logic signed [127:0] rounded;
        logic overflow;
        begin
            shift = {{23{common_exponent[8]}}, common_exponent};
            shift = shift + 41;
            overflow = 1'b0;
            rounded = '0;
            if (shift >= 64) begin
                if (accumulator != 128'sd0) overflow = 1'b1;
            end else if (shift >= 0) begin
                if ((accumulator > (128'sh00000000000000007fffffffffffffff >>> shift)) ||
                    (accumulator < (-128'sh00000000000000008000000000000000 >>> shift)))
                    overflow = 1'b1;
                else
                    rounded = accumulator <<< shift;
            end else begin
                rounded = rshift_rne_128(accumulator, -shift);
                if ((rounded > 128'sh00000000000000007fffffffffffffff) ||
                    (rounded < -128'sh00000000000000008000000000000000))
                    overflow = 1'b1;
            end
            convert_q20_44 = {overflow, rounded[63:0]};
        end
    endfunction

    assign lane_exponent_valid_w =
        (query_exponent_s8_i >= -8'sd96) && (query_exponent_s8_i <= 8'sd31) &&
        (key_exponent_s8_i >= -8'sd96) && (key_exponent_s8_i <= 8'sd31);
    assign selected_exponent_w = exponent_mem[index_q];
    always @* begin
        score_align_shift_w = {{23{max_exponent_q[8]}}, max_exponent_q};
        score_align_shift_w = score_align_shift_w -
                              {{23{selected_exponent_w[8]}}, selected_exponent_w};
    end
    assign aligned_product_w = rshift_rne_128(
        {{64{product_mem[index_q][63]}}, product_mem[index_q]},
        score_align_shift_w
    );
    assign accumulator_next_w = accumulator_q + aligned_product_w;
    assign converted_score_w = convert_q20_44(accumulator_next_w, max_exponent_q);

    assign start_ready_o = (state_q == S_IDLE);
    assign lane_ready_o = (state_q == S_LANE);
    assign mul_req_valid_o = (state_q == S_MUL_REQ);
    assign mul_operand_a_s32_o = query_mantissa_q;
    assign mul_operand_b_s32_o = key_mantissa_q;
    assign mul_rsp_ready_o = (state_q == S_MUL_RSP);
    assign score_valid_o = (state_q == S_EMIT);
    assign score_q20_44_s64_o = score_q;
    assign descriptor_error_o = descriptor_error_q;
    assign numeric_overflow_o = numeric_overflow_q;

    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            state_q <= S_IDLE;
            lane_count_q <= '0; index_q <= '0;
            query_mantissa_q <= '0; key_mantissa_q <= '0;
            pending_exponent_q <= '0; max_exponent_q <= '0;
            accumulator_q <= '0; score_q <= '0;
            descriptor_error_q <= 1'b0; numeric_overflow_q <= 1'b0;
        end else if (clear_i) begin
            state_q <= S_IDLE;
            descriptor_error_q <= 1'b0;
            numeric_overflow_q <= 1'b0;
        end else begin
            case (state_q)
                S_IDLE: begin
                    if (start_valid_i) begin
                        descriptor_error_q <= (lane_count_u7_i == 0) ||
                                              (lane_count_u7_i > MAX_LANES_U7);
                        numeric_overflow_q <= 1'b0;
                        score_q <= '0; accumulator_q <= '0; index_q <= '0;
                        lane_count_q <= lane_count_u7_i;
                        max_exponent_q <= -9'sd192;
                        state_q <= ((lane_count_u7_i == 0) ||
                                    (lane_count_u7_i > MAX_LANES_U7)) ? S_EMIT : S_LANE;
                    end
                end
                S_LANE: begin
                    if (lane_valid_i) begin
                        query_mantissa_q <= query_mantissa_s32_i;
                        key_mantissa_q <= key_mantissa_s32_i;
                        pending_exponent_q <=
                            $signed({query_exponent_s8_i[7], query_exponent_s8_i}) +
                            $signed({key_exponent_s8_i[7], key_exponent_s8_i});
                        if (!lane_exponent_valid_w) descriptor_error_q <= 1'b1;
                        state_q <= S_MUL_REQ;
                    end
                end
                S_MUL_REQ: begin
                    if (mul_req_ready_i) state_q <= S_MUL_RSP;
                end
                S_MUL_RSP: begin
                    if (mul_rsp_valid_i) begin
                        product_mem[index_q] <= mul_product_s64_i;
                        exponent_mem[index_q] <= pending_exponent_q;
                        if ((index_q == 0) || (pending_exponent_q > max_exponent_q))
                            max_exponent_q <= pending_exponent_q;
                        if ({1'b0, index_q} + 7'd1 == lane_count_q) begin
                            index_q <= '0; accumulator_q <= '0; state_q <= S_ACCUM;
                        end else begin
                            index_q <= index_q + 1'b1; state_q <= S_LANE;
                        end
                    end
                end
                S_ACCUM: begin
                    accumulator_q <= accumulator_next_w;
                    if ({1'b0, index_q} + 7'd1 == lane_count_q) begin
                        score_q <= converted_score_w[63:0];
                        numeric_overflow_q <= numeric_overflow_q || converted_score_w[64];
                        state_q <= S_EMIT;
                    end else begin
                        index_q <= index_q + 1'b1;
                    end
                end
                S_EMIT: begin
                    if (score_ready_i) state_q <= S_IDLE;
                end
                default: state_q <= S_IDLE;
            endcase
        end
    end
endmodule
/* verilator lint_on DECLFILENAME */
