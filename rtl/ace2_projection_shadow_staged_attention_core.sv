`timescale 1ns/1ps
`default_nettype none

// Contract: design/NUMERICAL_REPLACEMENT_PROPOSAL.md
// layer0_projection_shadow_staged_attention_v1 focused arithmetic candidate.
/* verilator lint_off DECLFILENAME */
module ace2_projection_shadow_core (
    input  wire                  clk_i,
    input  wire                  rst_ni,
    input  wire                  clear_i,
    input  wire                  in_valid_i,
    output wire                  in_ready_o,
    input  wire signed [31:0]    dot_s32_i,
    input  wire signed [31:0]    bias_s32_i,
    input  wire signed [31:0]    multiplier_s32_i,
    input  wire [5:0]            right_shift_u6_i,
    input  wire signed [7:0]     output_zero_point_s8_i,
    input  wire [47:0]           reserved_i,
    output wire                  out_valid_o,
    input  wire                  out_ready_i,
    output wire signed [31:0]    shadow_q15_16_o,
    output wire                  descriptor_error_o,
    output wire                  numeric_overflow_o
);
    reg out_valid_q;
    reg signed [31:0] shadow_q;
    reg descriptor_error_q;
    reg numeric_overflow_q;

    wire signed [32:0] accumulator_w =
        $signed({dot_s32_i[31], dot_s32_i}) +
        $signed({bias_s32_i[31], bias_s32_i});
    wire accumulator_overflow_w =
        (accumulator_w > 33'sd2147483647) ||
        (accumulator_w < -33'sd2147483648);
    wire metadata_error_w =
        (multiplier_s32_i <= 0) ||
        (output_zero_point_s8_i != 0) ||
        (reserved_i != 48'd0);
    wire signed [63:0] product_w =
        $signed(accumulator_w[31:0]) * $signed(multiplier_s32_i);
    wire signed [6:0] shadow_shift_w =
        $signed({1'b0, right_shift_u6_i}) - 7'sd16;

    reg product_negative_w;
    reg [79:0] product_magnitude_w;
    reg [79:0] rounded_base_w;
    reg [79:0] rounded_mask_w;
    reg [79:0] rounded_remainder_w;
    reg [79:0] rounded_half_w;
    reg rounded_increment_w;
    reg [80:0] rounded_magnitude_w;
    reg signed [80:0] rounded_value_w;
    reg rounded_overflow_w;
    always @* begin
        product_negative_w = product_w[63];
        product_magnitude_w = product_negative_w ?
            {16'd0, (~product_w + 64'd1)} : {16'd0, product_w};
        rounded_base_w = 80'd0;
        rounded_mask_w = 80'd0;
        rounded_remainder_w = 80'd0;
        rounded_half_w = 80'd0;
        rounded_increment_w = 1'b0;
        rounded_magnitude_w = 81'd0;
        if (shadow_shift_w < 0) begin
            rounded_magnitude_w =
                {1'b0, product_magnitude_w} << (-shadow_shift_w);
        end else if (shadow_shift_w == 0) begin
            rounded_magnitude_w = {1'b0, product_magnitude_w};
        end else begin
            rounded_base_w = product_magnitude_w >> shadow_shift_w;
            rounded_mask_w = (80'd1 << shadow_shift_w) - 80'd1;
            rounded_remainder_w = product_magnitude_w & rounded_mask_w;
            rounded_half_w = 80'd1 << (shadow_shift_w - 7'sd1);
            rounded_increment_w =
                (rounded_remainder_w > rounded_half_w) ||
                ((rounded_remainder_w == rounded_half_w) && rounded_base_w[0]);
            rounded_magnitude_w =
                {1'b0, rounded_base_w} + {{80{1'b0}}, rounded_increment_w};
        end
        rounded_value_w = product_negative_w ?
            -$signed(rounded_magnitude_w) : $signed(rounded_magnitude_w);
        rounded_overflow_w =
            (rounded_value_w > 81'sd2147483647) ||
            (rounded_value_w < -81'sd2147483648);
    end

    assign in_ready_o = !out_valid_q;
    assign out_valid_o = out_valid_q;
    assign shadow_q15_16_o = shadow_q;
    assign descriptor_error_o = descriptor_error_q;
    assign numeric_overflow_o = numeric_overflow_q;

    always @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            out_valid_q <= 1'b0;
            shadow_q <= 32'sd0;
            descriptor_error_q <= 1'b0;
            numeric_overflow_q <= 1'b0;
        end else if (clear_i) begin
            out_valid_q <= 1'b0;
            shadow_q <= 32'sd0;
            descriptor_error_q <= 1'b0;
            numeric_overflow_q <= 1'b0;
        end else begin
            if (out_valid_q && out_ready_i) begin
                out_valid_q <= 1'b0;
            end
            if (in_valid_i && in_ready_o) begin
                descriptor_error_q <= metadata_error_w;
                numeric_overflow_q <= accumulator_overflow_w || rounded_overflow_w;
                shadow_q <= rounded_value_w[31:0];
                out_valid_q <= 1'b1;
            end
        end
    end
endmodule


module ace2_projection_shadow_score_core #(
    parameter integer PAIR_COUNT = 32
) (
    input  wire                  clk_i,
    input  wire                  rst_ni,
    input  wire                  clear_i,
    input  wire                  start_valid_i,
    output wire                  start_ready_o,
    input  wire [31:0]           query_scale32_i,
    input  wire [31:0]           key_scale32_i,
    input  wire                  pair_valid_i,
    output wire                  pair_ready_o,
    input  wire signed [31:0]    query_low_q15_16_i,
    input  wire signed [31:0]    query_high_q15_16_i,
    input  wire signed [31:0]    key_low_q15_16_i,
    input  wire signed [31:0]    key_high_q15_16_i,
    input  wire signed [15:0]    query_cos_q15_i,
    input  wire signed [15:0]    query_sin_q15_i,
    input  wire signed [15:0]    key_cos_q15_i,
    input  wire signed [15:0]    key_sin_q15_i,
    output wire                  out_valid_o,
    input  wire                  out_ready_i,
    output wire signed [73:0]    dot_s74_o,
    output wire signed [15:0]    score_q6_9_o,
    output wire                  descriptor_error_o,
    output wire                  numeric_overflow_o
);
    localparam integer PAIR_INDEX_WIDTH =
        (PAIR_COUNT <= 1) ? 1 : $clog2(PAIR_COUNT);
    localparam [PAIR_INDEX_WIDTH-1:0] LAST_PAIR =
        PAIR_INDEX_WIDTH'(PAIR_COUNT - 1);
    localparam [2:0] ST_IDLE      = 3'd0;
    localparam [2:0] ST_PAIR_WAIT = 3'd1;
    localparam [2:0] ST_MUL_REAL  = 3'd2;
    localparam [2:0] ST_MUL_IMAG  = 3'd3;
    localparam [2:0] ST_SCALE     = 3'd4;
    localparam [2:0] ST_DONE      = 3'd5;

    reg [2:0] state_q;
    reg [PAIR_INDEX_WIDTH-1:0] pair_index_q;
    reg signed [33:0] query_real_q;
    reg signed [33:0] query_imag_q;
    reg signed [33:0] key_real_q;
    reg signed [33:0] key_imag_q;
    reg signed [67:0] first_product_q;
    reg signed [73:0] dot_q;
    reg [31:0] significand_product_q;
    reg [6:0] score_shift_q;
    reg signed [15:0] score_q;
    reg out_valid_q;
    reg descriptor_error_q;
    reg numeric_overflow_q;

    function automatic signed [49:0] rne_shift15_s49;
        input signed [48:0] value;
        reg negative;
        reg [48:0] magnitude;
        reg [33:0] quotient;
        reg [14:0] remainder;
        reg increment;
        reg [49:0] rounded;
        begin
            negative = value[48];
            magnitude = negative ? (~value + 49'd1) : value;
            quotient = magnitude[48:15];
            remainder = magnitude[14:0];
            increment = (remainder > 15'h4000) ||
                        ((remainder == 15'h4000) && quotient[0]);
            rounded = {16'd0, quotient} + {{49{1'b0}}, increment};
            rne_shift15_s49 = negative ?
                -$signed(rounded) : $signed(rounded);
        end
    endfunction

    wire signed [47:0] qlc_w = query_low_q15_16_i * query_cos_q15_i;
    wire signed [47:0] qhs_w = query_high_q15_16_i * query_sin_q15_i;
    wire signed [47:0] qhc_w = query_high_q15_16_i * query_cos_q15_i;
    wire signed [47:0] qls_w = query_low_q15_16_i * query_sin_q15_i;
    wire signed [47:0] klc_w = key_low_q15_16_i * key_cos_q15_i;
    wire signed [47:0] khs_w = key_high_q15_16_i * key_sin_q15_i;
    wire signed [47:0] khc_w = key_high_q15_16_i * key_cos_q15_i;
    wire signed [47:0] kls_w = key_low_q15_16_i * key_sin_q15_i;
    wire signed [48:0] query_real_pre_w =
        $signed({qlc_w[47], qlc_w}) - $signed({qhs_w[47], qhs_w});
    wire signed [48:0] query_imag_pre_w =
        $signed({qhc_w[47], qhc_w}) + $signed({qls_w[47], qls_w});
    wire signed [48:0] key_real_pre_w =
        $signed({klc_w[47], klc_w}) - $signed({khs_w[47], khs_w});
    wire signed [48:0] key_imag_pre_w =
        $signed({khc_w[47], khc_w}) + $signed({kls_w[47], kls_w});
    wire signed [49:0] query_real_round_w = rne_shift15_s49(query_real_pre_w);
    wire signed [49:0] query_imag_round_w = rne_shift15_s49(query_imag_pre_w);
    wire signed [49:0] key_real_round_w = rne_shift15_s49(key_real_pre_w);
    wire signed [49:0] key_imag_round_w = rne_shift15_s49(key_imag_pre_w);
    wire rotation_overflow_w =
        (query_real_round_w > 50'sd8589934591) ||
        (query_real_round_w < -50'sd8589934592) ||
        (query_imag_round_w > 50'sd8589934591) ||
        (query_imag_round_w < -50'sd8589934592) ||
        (key_real_round_w > 50'sd8589934591) ||
        (key_real_round_w < -50'sd8589934592) ||
        (key_imag_round_w > 50'sd8589934591) ||
        (key_imag_round_w < -50'sd8589934592);

    wire scale_exact_w =
        (query_scale32_i == 32'h00ffa245) &&
        (key_scale32_i == 32'h00008307);
    wire signed [8:0] query_exp_w =
        $signed({query_scale32_i[23], query_scale32_i[23:16]});
    wire signed [8:0] key_exp_w =
        $signed({key_scale32_i[23], key_scale32_i[23:16]});
    wire signed [9:0] exponent_sum_w =
        $signed({query_exp_w[8], query_exp_w}) +
        $signed({key_exp_w[8], key_exp_w});
    wire signed [10:0] score_shift_signed_w = 11'sd56 - exponent_sum_w;
    wire scale_valid_w = scale_exact_w &&
        (score_shift_signed_w >= 11'sd48) &&
        (score_shift_signed_w <= 11'sd104);
    wire [31:0] significand_product_w =
        query_scale32_i[15:0] * key_scale32_i[15:0];

    wire signed [67:0] real_product_w = query_real_q * key_real_q;
    wire signed [67:0] imag_product_w = query_imag_q * key_imag_q;
    wire signed [74:0] pair_sum_w =
        $signed({{7{first_product_q[67]}}, first_product_q}) +
        $signed({{7{imag_product_w[67]}}, imag_product_w});
    wire signed [74:0] dot_sum_w =
        $signed({dot_q[73], dot_q}) + pair_sum_w;
    wire dot_overflow_w =
        (dot_sum_w > 75'sh1ffffffffffffffffff) ||
        (dot_sum_w < -75'sh2000000000000000000);

    wire signed [105:0] scaled_product_w =
        $signed(dot_q) * $signed({1'b0, significand_product_q});
    reg scaled_negative_w;
    reg [105:0] scaled_magnitude_w;
    reg [105:0] rounded_base_w;
    reg [105:0] rounded_mask_w;
    reg [105:0] rounded_remainder_w;
    reg [105:0] rounded_half_w;
    reg rounded_increment_w;
    reg [106:0] rounded_magnitude_w;
    reg signed [106:0] rounded_value_w;
    reg signed [15:0] saturated_score_w;
    always @* begin
        scaled_negative_w = scaled_product_w[105];
        scaled_magnitude_w = scaled_negative_w ?
            (~scaled_product_w + 106'd1) : scaled_product_w;
        rounded_base_w = scaled_magnitude_w >> score_shift_q;
        rounded_mask_w = (106'd1 << score_shift_q) - 106'd1;
        rounded_remainder_w = scaled_magnitude_w & rounded_mask_w;
        rounded_half_w = 106'd1 << (score_shift_q - 7'd1);
        rounded_increment_w =
            (rounded_remainder_w > rounded_half_w) ||
            ((rounded_remainder_w == rounded_half_w) && rounded_base_w[0]);
        rounded_magnitude_w =
            {1'b0, rounded_base_w} + {{106{1'b0}}, rounded_increment_w};
        rounded_value_w = scaled_negative_w ?
            -$signed(rounded_magnitude_w) : $signed(rounded_magnitude_w);
        if (rounded_value_w > 107'sd32767) begin
            saturated_score_w = 16'sd32767;
        end else if (rounded_value_w < -107'sd32768) begin
            saturated_score_w = -16'sd32768;
        end else begin
            saturated_score_w = rounded_value_w[15:0];
        end
    end

    assign start_ready_o = (state_q == ST_IDLE) && !out_valid_q;
    assign pair_ready_o = (state_q == ST_PAIR_WAIT);
    assign out_valid_o = out_valid_q;
    assign dot_s74_o = dot_q;
    assign score_q6_9_o = score_q;
    assign descriptor_error_o = descriptor_error_q;
    assign numeric_overflow_o = numeric_overflow_q;

    always @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            state_q <= ST_IDLE;
            pair_index_q <= {PAIR_INDEX_WIDTH{1'b0}};
            query_real_q <= 34'sd0;
            query_imag_q <= 34'sd0;
            key_real_q <= 34'sd0;
            key_imag_q <= 34'sd0;
            first_product_q <= 68'sd0;
            dot_q <= 74'sd0;
            significand_product_q <= 32'd0;
            score_shift_q <= 7'd0;
            score_q <= 16'sd0;
            out_valid_q <= 1'b0;
            descriptor_error_q <= 1'b0;
            numeric_overflow_q <= 1'b0;
        end else if (clear_i) begin
            state_q <= ST_IDLE;
            pair_index_q <= {PAIR_INDEX_WIDTH{1'b0}};
            dot_q <= 74'sd0;
            score_q <= 16'sd0;
            out_valid_q <= 1'b0;
            descriptor_error_q <= 1'b0;
            numeric_overflow_q <= 1'b0;
        end else begin
            if (out_valid_q && out_ready_i) begin
                out_valid_q <= 1'b0;
                state_q <= ST_IDLE;
            end
            case (state_q)
                ST_IDLE: begin
                    if (start_valid_i && start_ready_o) begin
                        pair_index_q <= {PAIR_INDEX_WIDTH{1'b0}};
                        dot_q <= 74'sd0;
                        score_q <= 16'sd0;
                        descriptor_error_q <= !scale_valid_w;
                        numeric_overflow_q <= 1'b0;
                        significand_product_q <= significand_product_w;
                        score_shift_q <= score_shift_signed_w[6:0];
                        if (scale_valid_w) begin
                            state_q <= ST_PAIR_WAIT;
                        end else begin
                            out_valid_q <= 1'b1;
                            state_q <= ST_DONE;
                        end
                    end
                end
                ST_PAIR_WAIT: begin
                    if (pair_valid_i && pair_ready_o) begin
                        if (rotation_overflow_w) begin
                            numeric_overflow_q <= 1'b1;
                            out_valid_q <= 1'b1;
                            state_q <= ST_DONE;
                        end else begin
                            query_real_q <= query_real_round_w[33:0];
                            query_imag_q <= query_imag_round_w[33:0];
                            key_real_q <= key_real_round_w[33:0];
                            key_imag_q <= key_imag_round_w[33:0];
                            state_q <= ST_MUL_REAL;
                        end
                    end
                end
                ST_MUL_REAL: begin
                    first_product_q <= real_product_w;
                    state_q <= ST_MUL_IMAG;
                end
                ST_MUL_IMAG: begin
                    if (dot_overflow_w) begin
                        numeric_overflow_q <= 1'b1;
                        out_valid_q <= 1'b1;
                        state_q <= ST_DONE;
                    end else begin
                        dot_q <= dot_sum_w[73:0];
                        if (pair_index_q == LAST_PAIR) begin
                            state_q <= ST_SCALE;
                        end else begin
                            pair_index_q <= pair_index_q + 1'b1;
                            state_q <= ST_PAIR_WAIT;
                        end
                    end
                end
                ST_SCALE: begin
                    score_q <= saturated_score_w;
                    out_valid_q <= 1'b1;
                    state_q <= ST_DONE;
                end
                ST_DONE: begin
                end
                default: begin
                    state_q <= ST_IDLE;
                end
            endcase
        end
    end
endmodule
/* verilator lint_on DECLFILENAME */

`default_nettype wire
