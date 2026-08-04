`timescale 1ns/1ps
`default_nettype none

// Contract: design/NUMERICAL_REPLACEMENT_PROPOSAL.md
// contract_id: layer0_tile_bfp_score_attention_v1
/* verilator lint_off DECLFILENAME */
module ace2_tile_bfp_score_core #(
    parameter integer MAX_KEYS = 64
) (
    input  wire                   clk_i,
    input  wire                   rst_ni,
    input  wire                   clear_i,
    input  wire                   start_valid_i,
    output wire                   start_ready_o,
    input  wire [6:0]             key_count_u7_i,
    input  wire signed [7:0]      score_exponent_s8_i,
    input  wire                   score_valid_i,
    output wire                   score_ready_o,
    input  wire signed [105:0]    score_num_s106_i,
    output wire                   metadata_valid_o,
    input  wire                   metadata_ready_i,
    output wire signed [127:0]    tile_max_s128_o,
    output wire [4:0]             fraction_bits_u5_o,
    output wire                   mantissa_valid_o,
    input  wire                   mantissa_ready_i,
    output wire [5:0]             mantissa_index_u6_o,
    output wire signed [23:0]     mantissa_s24_o,
    output wire                   mantissa_last_o,
    output wire                   descriptor_error_o,
    output wire                   numeric_overflow_o
);
    localparam [2:0] ST_IDLE    = 3'd0;
    localparam [2:0] ST_COLLECT = 3'd1;
    localparam [2:0] ST_META    = 3'd2;
    localparam [2:0] ST_EMIT    = 3'd3;
    localparam [6:0] MAX_KEYS_U7 = MAX_KEYS[6:0];

    reg [2:0] state_q;
    reg [6:0] key_count_q;
    reg [5:0] collect_index_q;
    reg [5:0] mantissa_index_q;
    reg signed [105:0] score_mem [0:MAX_KEYS-1];
    reg signed [105:0] tile_max_q;
    reg signed [105:0] tile_min_q;
    reg [4:0] fraction_bits_q;
    reg descriptor_error_q;
    reg numeric_overflow_q;

    function automatic [127:0] rne_shift_u128;
        input [127:0] value;
        input [6:0] shift;
        reg [127:0] quotient;
        reg [127:0] remainder;
        reg [127:0] half;
        reg increment;
        begin
            if (shift == 0) begin
                rne_shift_u128 = value;
            end else begin
                quotient = value >> shift;
                remainder = value - (quotient << shift);
                half = 128'd1 << (shift - 1'b1);
                increment = (remainder > half) ||
                            ((remainder == half) && quotient[0]);
                rne_shift_u128 = quotient + {{127{1'b0}}, increment};
            end
        end
    endfunction

    wire start_metadata_error_w =
        (key_count_u7_i == 0) ||
        (key_count_u7_i > MAX_KEYS_U7) ||
        (score_exponent_s8_i != -8'sd1);
    wire signed [105:0] next_max_w =
        (collect_index_q == 0 || score_num_s106_i > tile_max_q) ?
        score_num_s106_i : tile_max_q;
    wire signed [105:0] next_min_w =
        (collect_index_q == 0 || score_num_s106_i < tile_min_q) ?
        score_num_s106_i : tile_min_q;
    wire signed [127:0] tile_max_ext_w =
        {{22{tile_max_q[105]}}, tile_max_q};
    wire signed [127:0] tile_min_ext_w =
        {{22{tile_min_q[105]}}, tile_min_q};
    wire [127:0] range_num_w = tile_max_ext_w - tile_min_ext_w;

    integer candidate_fraction;
    reg fraction_found_w;
    reg [4:0] selected_fraction_w;
    reg [127:0] candidate_range_w;
    always @* begin
        fraction_found_w = 1'b0;
        selected_fraction_w = 5'd0;
        candidate_range_w = 128'd0;
        for (candidate_fraction = 17; candidate_fraction >= 0; candidate_fraction = candidate_fraction - 1) begin
            if (!fraction_found_w) begin
                candidate_range_w = rne_shift_u128(
                    range_num_w,
                    7'd66 - candidate_fraction[6:0]
                );
                if (candidate_range_w <= 128'd8388607) begin
                    selected_fraction_w = candidate_fraction[4:0];
                    fraction_found_w = 1'b1;
                end
            end
        end
    end

    wire signed [127:0] current_score_w =
        {{22{score_mem[mantissa_index_q][105]}}, score_mem[mantissa_index_q]};
    wire [127:0] current_magnitude_w = tile_max_ext_w - current_score_w;
    wire [6:0] mantissa_shift_w = 7'd66 - {2'd0, fraction_bits_q};
    wire [127:0] rounded_magnitude_w =
        rne_shift_u128(current_magnitude_w, mantissa_shift_w);
    wire mantissa_overflow_w = rounded_magnitude_w > 128'd8388607;
    wire signed [23:0] encoded_mantissa_w =
        -$signed(rounded_magnitude_w[23:0]);

    assign start_ready_o = (state_q == ST_IDLE);
    assign score_ready_o = (state_q == ST_COLLECT);
    assign metadata_valid_o = (state_q == ST_META) && fraction_found_w;
    assign tile_max_s128_o = tile_max_ext_w;
    assign fraction_bits_u5_o =
        (state_q == ST_META) ? selected_fraction_w : fraction_bits_q;
    assign mantissa_valid_o = (state_q == ST_EMIT);
    assign mantissa_index_u6_o = mantissa_index_q;
    assign mantissa_s24_o = mantissa_overflow_w ? 24'sd0 : encoded_mantissa_w;
    assign mantissa_last_o =
        ({1'b0, mantissa_index_q} == key_count_q - 7'd1);
    assign descriptor_error_o = descriptor_error_q;
    assign numeric_overflow_o =
        numeric_overflow_q || ((state_q == ST_EMIT) && mantissa_overflow_w);

    always @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            state_q <= ST_IDLE;
            key_count_q <= 7'd0;
            collect_index_q <= 6'd0;
            mantissa_index_q <= 6'd0;
            tile_max_q <= 106'sd0;
            tile_min_q <= 106'sd0;
            fraction_bits_q <= 5'd0;
            descriptor_error_q <= 1'b0;
            numeric_overflow_q <= 1'b0;
        end else if (clear_i) begin
            state_q <= ST_IDLE;
            key_count_q <= 7'd0;
            collect_index_q <= 6'd0;
            mantissa_index_q <= 6'd0;
            tile_max_q <= 106'sd0;
            tile_min_q <= 106'sd0;
            fraction_bits_q <= 5'd0;
            descriptor_error_q <= 1'b0;
            numeric_overflow_q <= 1'b0;
        end else begin
            case (state_q)
                ST_IDLE: begin
                    if (start_valid_i && start_ready_o) begin
                        key_count_q <= key_count_u7_i;
                        collect_index_q <= 6'd0;
                        mantissa_index_q <= 6'd0;
                        tile_max_q <= 106'sd0;
                        tile_min_q <= 106'sd0;
                        fraction_bits_q <= 5'd0;
                        descriptor_error_q <= start_metadata_error_w;
                        numeric_overflow_q <= 1'b0;
                        state_q <= start_metadata_error_w ? ST_IDLE : ST_COLLECT;
                    end
                end
                ST_COLLECT: begin
                    if (score_valid_i && score_ready_o) begin
                        score_mem[collect_index_q] <= score_num_s106_i;
                        tile_max_q <= next_max_w;
                        tile_min_q <= next_min_w;
                        if ({1'b0, collect_index_q} == key_count_q - 7'd1) begin
                            state_q <= ST_META;
                        end else begin
                            collect_index_q <= collect_index_q + 1'b1;
                        end
                    end
                end
                ST_META: begin
                    if (!fraction_found_w) begin
                        numeric_overflow_q <= 1'b1;
                        state_q <= ST_IDLE;
                    end else if (metadata_valid_o && metadata_ready_i) begin
                        fraction_bits_q <= selected_fraction_w;
                        mantissa_index_q <= 6'd0;
                        state_q <= ST_EMIT;
                    end
                end
                ST_EMIT: begin
                    if (mantissa_valid_o && mantissa_ready_i) begin
                        if (mantissa_overflow_w) begin
                            numeric_overflow_q <= 1'b1;
                        end
                        if (mantissa_last_o) begin
                            state_q <= ST_IDLE;
                        end else begin
                            mantissa_index_q <= mantissa_index_q + 1'b1;
                        end
                    end
                end
                default: state_q <= ST_IDLE;
            endcase
        end
    end
endmodule


module ace2_bfp_hierarchical_softmax_core (
    input  wire                   clk_i,
    input  wire                   rst_ni,
    input  wire                   clear_i,
    input  wire                   start_valid_i,
    output wire                   start_ready_o,
    input  wire [6:0]             key_count_u7_i,
    input  wire signed [7:0]      score_exponent_s8_i,
    input  wire [4:0]             fraction_bits_u5_i,
    input  wire signed [127:0]    row_max_s128_i,
    input  wire signed [127:0]    tile_max_s128_i,
    input  wire                   mantissa_valid_i,
    output wire                   mantissa_ready_o,
    input  wire signed [23:0]     mantissa_s24_i,
    output wire                   weight_valid_o,
    input  wire                   weight_ready_i,
    output wire [31:0]            weight_q1_31_o,
    output wire                   weight_last_o,
    output wire                   sum_valid_o,
    input  wire                   sum_ready_i,
    output wire [47:0]            weight_sum_u48_o,
    input  wire                   norm_valid_i,
    output wire                   norm_ready_o,
    input  wire [31:0]            norm_weight_q1_31_i,
    input  wire [47:0]            norm_denominator_u48_i,
    output wire                   prob_valid_o,
    input  wire                   prob_ready_i,
    output wire [14:0]            probability_q0_15_o,
    output wire                   descriptor_error_o,
    output wire                   numeric_overflow_o
);
    `include "generated/ace2_exp_q31_lut.svh"

    localparam [1:0] ST_IDLE = 2'd0;
    localparam [1:0] ST_RUN  = 2'd1;
    localparam [1:0] ST_SUM  = 2'd2;

    reg [1:0] state_q;
    reg [6:0] key_count_q;
    reg [6:0] accepted_count_q;
    reg [4:0] fraction_bits_q;
    reg signed [31:0] tile_offset_q;
    reg [47:0] sum_q;
    reg weight_valid_q;
    reg [31:0] weight_q;
    reg weight_last_q;
    reg prob_valid_q;
    reg [14:0] probability_q;
    reg descriptor_error_q;
    reg numeric_overflow_q;

    function automatic signed [80:0] rne_shift49_s128;
        input signed [127:0] value;
        reg negative;
        reg [127:0] magnitude;
        reg [78:0] quotient;
        reg [48:0] remainder;
        reg increment;
        reg [79:0] rounded;
        begin
            negative = value[127];
            magnitude = negative ? (~value + 128'd1) : value;
            quotient = magnitude[127:49];
            remainder = magnitude[48:0];
            increment =
                (remainder > 49'h1000000000000) ||
                ((remainder == 49'h1000000000000) && quotient[0]);
            rounded = {1'b0, quotient} + {{79{1'b0}}, increment};
            rne_shift49_s128 = negative ?
                -$signed({1'b0, rounded}) : $signed({1'b0, rounded});
        end
    endfunction

    function automatic signed [32:0] rne_shift16_s49;
        input signed [48:0] value;
        reg negative;
        reg [48:0] magnitude;
        reg [32:0] quotient;
        reg [15:0] remainder;
        reg increment;
        reg [32:0] rounded;
        begin
            negative = value[48];
            magnitude = negative ? (~value + 49'd1) : value;
            quotient = magnitude[48:16];
            remainder = magnitude[15:0];
            increment = (remainder > 16'h8000) ||
                        ((remainder == 16'h8000) && quotient[0]);
            rounded = quotient + {{32{1'b0}}, increment};
            rne_shift16_s49 = negative ?
                -$signed(rounded) : $signed(rounded);
        end
    endfunction

    wire start_metadata_error_w =
        (key_count_u7_i == 0) ||
        (key_count_u7_i > 7'd64) ||
        (score_exponent_s8_i != -8'sd1) ||
        (fraction_bits_u5_i > 5'd17) ||
        (tile_max_s128_i > row_max_s128_i);
    wire signed [127:0] tile_offset_num_w =
        tile_max_s128_i - row_max_s128_i;
    wire signed [80:0] offset_rounded_w =
        rne_shift49_s128(tile_offset_num_w);
    wire offset_overflow_w =
        (offset_rounded_w > 81'sd0) ||
        (offset_rounded_w < -81'sd2147483648);

    wire [5:0] local_shift_w = 6'd17 - {1'b0, fraction_bits_q};
    wire signed [40:0] mantissa_extended_w =
        {{17{mantissa_s24_i[23]}}, mantissa_s24_i};
    wire signed [40:0] local_delta_q17_w =
        mantissa_extended_w <<< local_shift_w;
    wire signed [81:0] merged_delta_w =
        $signed({{50{tile_offset_q[31]}}, tile_offset_q}) +
        $signed({{41{local_delta_q17_w[40]}}, local_delta_q17_w});
    wire reconstruction_overflow_w =
        (merged_delta_w > 82'sd0) ||
        (merged_delta_w < -82'sd2147483648);
    wire signed [31:0] merged_q17_w = merged_delta_w[31:0];
    wire merged_underflow_w = merged_q17_w <= -32'sd2097152;
    wire [30:0] merged_magnitude_w = merged_q17_w[31] ?
        (~merged_q17_w[30:0] + 31'd1) : merged_q17_w[30:0];
    wire [8:0] exp_index_w = merged_magnitude_w[21:13];
    wire exp_range_overflow_w = |merged_magnitude_w[30:22];
    wire [15:0] exp_fraction_w = {merged_magnitude_w[12:0], 3'b000};
    wire [31:0] exp_base_w = ace2_exp_q31_lut(exp_index_w);
    wire [31:0] exp_next_w = ace2_exp_q31_lut(exp_index_w + 1'b1);
    wire signed [32:0] exp_difference_w =
        $signed({1'b0, exp_next_w}) - $signed({1'b0, exp_base_w});
    wire signed [48:0] exp_correction_product_w =
        exp_difference_w * $signed({1'b0, exp_fraction_w});
    wire signed [32:0] exp_correction_w =
        rne_shift16_s49(exp_correction_product_w);
    wire signed [33:0] interpolated_exp_w =
        $signed({1'b0, exp_base_w}) +
        $signed({exp_correction_w[32], exp_correction_w});
    wire exp_interpolation_overflow_w =
        (interpolated_exp_w < 0) ||
        (interpolated_exp_w > 34'sh080000000);
    wire [31:0] next_weight_w =
        reconstruction_overflow_w ? 32'd0 :
        (merged_underflow_w ? 32'd0 :
        (merged_q17_w == 0 ? 32'h80000000 : interpolated_exp_w[31:0]));
    wire [48:0] next_sum_w = {1'b0, sum_q} + {17'd0, next_weight_w};
    wire sum_overflow_w = next_sum_w[48];

    wire [46:0] norm_product_w = norm_weight_q1_31_i * 15'd32767;
    wire [47:0] norm_numerator_w = {1'b0, norm_product_w};
    wire [47:0] norm_quotient_w =
        (norm_denominator_u48_i == 0) ? 48'd0 :
        norm_numerator_w / norm_denominator_u48_i;
    wire [47:0] norm_remainder_w =
        (norm_denominator_u48_i == 0) ? 48'd0 :
        norm_numerator_w % norm_denominator_u48_i;
    wire [48:0] norm_doubled_remainder_w =
        {1'b0, norm_remainder_w} << 1;
    wire norm_increment_w =
        (norm_doubled_remainder_w > {1'b0, norm_denominator_u48_i}) ||
        ((norm_doubled_remainder_w == {1'b0, norm_denominator_u48_i}) &&
         norm_quotient_w[0]);
    wire [48:0] norm_rounded_w =
        {1'b0, norm_quotient_w} + {{48{1'b0}}, norm_increment_w};
    wire norm_error_w =
        (norm_denominator_u48_i == 0) ||
        (norm_rounded_w > 49'd32767);

    assign start_ready_o = (state_q == ST_IDLE) && !weight_valid_q;
    assign mantissa_ready_o = (state_q == ST_RUN) && !weight_valid_q;
    assign weight_valid_o = weight_valid_q;
    assign weight_q1_31_o = weight_q;
    assign weight_last_o = weight_last_q;
    assign sum_valid_o = (state_q == ST_SUM);
    assign weight_sum_u48_o = sum_q;
    assign norm_ready_o = !prob_valid_q;
    assign prob_valid_o = prob_valid_q;
    assign probability_q0_15_o = probability_q;
    assign descriptor_error_o = descriptor_error_q;
    assign numeric_overflow_o = numeric_overflow_q;

    always @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            state_q <= ST_IDLE;
            key_count_q <= 7'd0;
            accepted_count_q <= 7'd0;
            fraction_bits_q <= 5'd0;
            tile_offset_q <= 32'sd0;
            sum_q <= 48'd0;
            weight_valid_q <= 1'b0;
            weight_q <= 32'd0;
            weight_last_q <= 1'b0;
            prob_valid_q <= 1'b0;
            probability_q <= 15'd0;
            descriptor_error_q <= 1'b0;
            numeric_overflow_q <= 1'b0;
        end else if (clear_i) begin
            state_q <= ST_IDLE;
            key_count_q <= 7'd0;
            accepted_count_q <= 7'd0;
            fraction_bits_q <= 5'd0;
            tile_offset_q <= 32'sd0;
            sum_q <= 48'd0;
            weight_valid_q <= 1'b0;
            weight_q <= 32'd0;
            weight_last_q <= 1'b0;
            prob_valid_q <= 1'b0;
            probability_q <= 15'd0;
            descriptor_error_q <= 1'b0;
            numeric_overflow_q <= 1'b0;
        end else begin
            if (weight_valid_q && weight_ready_i) begin
                weight_valid_q <= 1'b0;
                if (weight_last_q) begin
                    state_q <= ST_SUM;
                end
            end
            if (prob_valid_q && prob_ready_i) begin
                prob_valid_q <= 1'b0;
            end
            if (norm_valid_i && norm_ready_o) begin
                descriptor_error_q <= descriptor_error_q || norm_error_w;
                probability_q <= norm_error_w ? 15'd0 : norm_rounded_w[14:0];
                prob_valid_q <= 1'b1;
            end
            case (state_q)
                ST_IDLE: begin
                    if (start_valid_i && start_ready_o) begin
                        descriptor_error_q <= start_metadata_error_w;
                        numeric_overflow_q <= offset_overflow_w;
                        key_count_q <= key_count_u7_i;
                        accepted_count_q <= 7'd0;
                        fraction_bits_q <= fraction_bits_u5_i;
                        tile_offset_q <= offset_overflow_w ?
                            32'sd0 : offset_rounded_w[31:0];
                        sum_q <= 48'd0;
                        if (!start_metadata_error_w && !offset_overflow_w) begin
                            state_q <= ST_RUN;
                        end
                    end
                end
                ST_RUN: begin
                    if (mantissa_valid_i && mantissa_ready_o) begin
                        weight_q <= next_weight_w;
                        weight_last_q <=
                            (accepted_count_q == key_count_q - 1'b1);
                        weight_valid_q <= 1'b1;
                        accepted_count_q <= accepted_count_q + 1'b1;
                        if (
                            reconstruction_overflow_w ||
                            sum_overflow_w ||
                            exp_interpolation_overflow_w ||
                            (exp_range_overflow_w && !merged_underflow_w)
                        ) begin
                            numeric_overflow_q <= 1'b1;
                        end
                        sum_q <= next_sum_w[47:0];
                    end
                end
                ST_SUM: begin
                    if (sum_valid_o && sum_ready_i) begin
                        state_q <= ST_IDLE;
                    end
                end
                default: state_q <= ST_IDLE;
            endcase
        end
    end
endmodule
/* verilator lint_on DECLFILENAME */

`default_nettype wire
