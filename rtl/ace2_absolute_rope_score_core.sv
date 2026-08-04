`timescale 1ns/1ps
`default_nettype none

// Contract: design/NUMERICAL_REPLACEMENT_PROPOSAL.md
// layer0_absolute_rope_online_attention_v1, exact wide-RoPE score front end.
module ace2_absolute_rope_score_core #(
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
    input  wire signed [7:0]     query_low_i,
    input  wire signed [7:0]     query_high_i,
    input  wire signed [7:0]     key_low_i,
    input  wire signed [7:0]     key_high_i,
    input  wire signed [15:0]    query_cos_q15_i,
    input  wire signed [15:0]    query_sin_q15_i,
    input  wire signed [15:0]    key_cos_q15_i,
    input  wire signed [15:0]    key_sin_q15_i,

    output wire                  out_valid_o,
    input  wire                  out_ready_i,
    output wire signed [53:0]    score_raw_o,
    output wire signed [63:0]    logit_q12_20_o,
    output wire                  descriptor_error_o,
    output wire                  numeric_overflow_o
);
    localparam integer PAIR_INDEX_WIDTH =
        (PAIR_COUNT <= 1) ? 1 : $clog2(PAIR_COUNT);
    localparam [PAIR_INDEX_WIDTH-1:0] LAST_PAIR =
        PAIR_INDEX_WIDTH'(PAIR_COUNT - 1);

    localparam [3:0] ST_IDLE       = 4'd0;
    localparam [3:0] ST_PAIR_WAIT  = 4'd1;
    localparam [3:0] ST_MUL_REAL   = 4'd2;
    localparam [3:0] ST_MUL_IMAG   = 4'd3;
    localparam [3:0] ST_SCALE_HIGH = 4'd4;
    localparam [3:0] ST_SCALE_LOW  = 4'd5;
    localparam [3:0] ST_ROUND      = 4'd6;
    localparam [3:0] ST_DONE       = 4'd7;

    reg [3:0] state_q;
    reg [PAIR_INDEX_WIDTH-1:0] pair_index_q;
    reg signed [24:0] query_real_q;
    reg signed [24:0] query_imag_q;
    reg signed [24:0] key_real_q;
    reg signed [24:0] key_imag_q;
    reg signed [63:0] first_product_q;
    reg signed [53:0] score_raw_q;
    reg [31:0] significand_product_q;
    reg [6:0] score_shift_q;
    reg signed [64:0] scale_high_product_q;
    reg signed [86:0] scaled_product_q;
    reg signed [63:0] logit_q;
    reg out_valid_q;
    reg descriptor_error_q;
    reg numeric_overflow_q;

    wire signed [23:0] query_low_cos_w = query_low_i * query_cos_q15_i;
    wire signed [23:0] query_high_sin_w = query_high_i * query_sin_q15_i;
    wire signed [23:0] query_high_cos_w = query_high_i * query_cos_q15_i;
    wire signed [23:0] query_low_sin_w = query_low_i * query_sin_q15_i;
    wire signed [23:0] key_low_cos_w = key_low_i * key_cos_q15_i;
    wire signed [23:0] key_high_sin_w = key_high_i * key_sin_q15_i;
    wire signed [23:0] key_high_cos_w = key_high_i * key_cos_q15_i;
    wire signed [23:0] key_low_sin_w = key_low_i * key_sin_q15_i;

    wire signed [24:0] query_real_w =
        $signed({query_low_cos_w[23], query_low_cos_w}) -
        $signed({query_high_sin_w[23], query_high_sin_w});
    wire signed [24:0] query_imag_w =
        $signed({query_high_cos_w[23], query_high_cos_w}) +
        $signed({query_low_sin_w[23], query_low_sin_w});
    wire signed [24:0] key_real_w =
        $signed({key_low_cos_w[23], key_low_cos_w}) -
        $signed({key_high_sin_w[23], key_high_sin_w});
    wire signed [24:0] key_imag_w =
        $signed({key_high_cos_w[23], key_high_cos_w}) +
        $signed({key_low_sin_w[23], key_low_sin_w});

    wire query_scale_valid_w =
        (query_scale32_i[31:24] == 8'd0) && query_scale32_i[15];
    wire key_scale_valid_w =
        (key_scale32_i[31:24] == 8'd0) && key_scale32_i[15];
    wire signed [8:0] query_exponent_w =
        $signed({query_scale32_i[23], query_scale32_i[23:16]});
    wire signed [8:0] key_exponent_w =
        $signed({key_scale32_i[23], key_scale32_i[23:16]});
    wire query_exponent_valid_w =
        (query_exponent_w >= -9'sd24) && (query_exponent_w <= 9'sd4);
    wire key_exponent_valid_w =
        (key_exponent_w >= -9'sd24) && (key_exponent_w <= 9'sd4);
    wire signed [9:0] exponent_sum_w =
        $signed({query_exponent_w[8], query_exponent_w}) +
        $signed({key_exponent_w[8], key_exponent_w});
    wire signed [10:0] score_shift_signed_w = 11'sd43 - exponent_sum_w;
    wire scale_valid_w = query_scale_valid_w && key_scale_valid_w &&
                         query_exponent_valid_w && key_exponent_valid_w &&
                         (score_shift_signed_w >= 11'sd35) &&
                         (score_shift_signed_w <= 11'sd91);
    wire [31:0] significand_product_w =
        query_scale32_i[15:0] * key_scale32_i[15:0];

    reg [31:0] multiplier_a_w;
    reg [31:0] multiplier_b_w;
    reg multiplier_negative_w;
    wire [63:0] multiplier_magnitude_w = multiplier_a_w * multiplier_b_w;
    wire signed [64:0] multiplier_signed_w = multiplier_negative_w ?
        -$signed({1'b0, multiplier_magnitude_w}) :
         $signed({1'b0, multiplier_magnitude_w});

    wire [24:0] query_real_magnitude_w = query_real_q[24] ?
        (~query_real_q + 25'd1) : query_real_q;
    wire [24:0] query_imag_magnitude_w = query_imag_q[24] ?
        (~query_imag_q + 25'd1) : query_imag_q;
    wire [24:0] key_real_magnitude_w = key_real_q[24] ?
        (~key_real_q + 25'd1) : key_real_q;
    wire [24:0] key_imag_magnitude_w = key_imag_q[24] ?
        (~key_imag_q + 25'd1) : key_imag_q;
    wire signed [31:0] score_high_w = score_raw_q[53:22];
    wire [31:0] score_high_magnitude_w = score_high_w[31] ?
        (~score_high_w + 32'd1) : score_high_w;
    wire [21:0] score_low_w = score_raw_q[21:0];

    always @* begin
        multiplier_a_w = 32'd0;
        multiplier_b_w = 32'd0;
        multiplier_negative_w = 1'b0;
        case (state_q)
            ST_MUL_REAL: begin
                multiplier_a_w = {7'd0, query_real_magnitude_w};
                multiplier_b_w = {7'd0, key_real_magnitude_w};
                multiplier_negative_w = query_real_q[24] ^ key_real_q[24];
            end
            ST_MUL_IMAG: begin
                multiplier_a_w = {7'd0, query_imag_magnitude_w};
                multiplier_b_w = {7'd0, key_imag_magnitude_w};
                multiplier_negative_w = query_imag_q[24] ^ key_imag_q[24];
            end
            ST_SCALE_HIGH: begin
                multiplier_a_w = score_high_magnitude_w;
                multiplier_b_w = significand_product_q;
                multiplier_negative_w = score_high_w[31];
            end
            ST_SCALE_LOW: begin
                multiplier_a_w = {10'd0, score_low_w};
                multiplier_b_w = significand_product_q;
                multiplier_negative_w = 1'b0;
            end
            default: begin
            end
        endcase
    end

    wire signed [64:0] pair_sum_w =
        $signed({first_product_q[63], first_product_q}) + multiplier_signed_w;
    wire signed [64:0] score_sum_w =
        $signed({{11{score_raw_q[53]}}, score_raw_q}) + pair_sum_w;
    wire score_sum_overflow_w =
        (score_sum_w < -65'sd9007199254740992) ||
        (score_sum_w >  65'sd9007199254740991);

    wire signed [86:0] scaled_product_combined_w =
        ($signed({{22{scale_high_product_q[64]}}, scale_high_product_q}) <<< 22) +
        $signed({23'd0, multiplier_magnitude_w});

    reg scaled_negative_w;
    reg [95:0] scaled_magnitude_w;
    reg [95:0] rounded_base_w;
    reg [95:0] rounded_mask_w;
    reg [95:0] rounded_remainder_w;
    reg [95:0] rounded_half_w;
    reg rounded_increment_w;
    reg [96:0] rounded_magnitude_w;
    reg rounded_overflow_w;
    reg signed [63:0] rounded_logit_w;
    always @* begin
        scaled_negative_w = scaled_product_q[86];
        scaled_magnitude_w = scaled_negative_w ?
            {{9{1'b0}}, (~scaled_product_q + 87'd1)} :
            {{9{1'b0}}, scaled_product_q};
        rounded_base_w = scaled_magnitude_w >> score_shift_q;
        rounded_mask_w = (96'd1 << score_shift_q) - 96'd1;
        rounded_remainder_w = scaled_magnitude_w & rounded_mask_w;
        rounded_half_w = 96'd1 << (score_shift_q - 7'd1);
        rounded_increment_w =
            (rounded_remainder_w > rounded_half_w) ||
            ((rounded_remainder_w == rounded_half_w) && rounded_base_w[0]);
        rounded_magnitude_w =
            {1'b0, rounded_base_w} + {{96{1'b0}}, rounded_increment_w};
        rounded_overflow_w =
            (!scaled_negative_w && (rounded_magnitude_w > 97'h07fffffffffffffff)) ||
            ( scaled_negative_w && (rounded_magnitude_w > 97'h08000000000000000));
        if (scaled_negative_w) begin
            rounded_logit_w = -$signed(rounded_magnitude_w[63:0]);
        end else begin
            rounded_logit_w = $signed(rounded_magnitude_w[63:0]);
        end
    end

    assign start_ready_o = (state_q == ST_IDLE) && !out_valid_q;
    assign pair_ready_o = (state_q == ST_PAIR_WAIT);
    assign out_valid_o = out_valid_q;
    assign score_raw_o = score_raw_q;
    assign logit_q12_20_o = logit_q;
    assign descriptor_error_o = descriptor_error_q;
    assign numeric_overflow_o = numeric_overflow_q;

    always @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            state_q <= ST_IDLE;
            pair_index_q <= {PAIR_INDEX_WIDTH{1'b0}};
            query_real_q <= 25'sd0;
            query_imag_q <= 25'sd0;
            key_real_q <= 25'sd0;
            key_imag_q <= 25'sd0;
            first_product_q <= 64'sd0;
            score_raw_q <= 54'sd0;
            significand_product_q <= 32'd0;
            score_shift_q <= 7'd0;
            scale_high_product_q <= 65'sd0;
            scaled_product_q <= 87'sd0;
            logit_q <= 64'sd0;
            out_valid_q <= 1'b0;
            descriptor_error_q <= 1'b0;
            numeric_overflow_q <= 1'b0;
        end else if (clear_i) begin
            state_q <= ST_IDLE;
            pair_index_q <= {PAIR_INDEX_WIDTH{1'b0}};
            score_raw_q <= 54'sd0;
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
                        score_raw_q <= 54'sd0;
                        logit_q <= 64'sd0;
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
                        query_real_q <= query_real_w;
                        query_imag_q <= query_imag_w;
                        key_real_q <= key_real_w;
                        key_imag_q <= key_imag_w;
                        state_q <= ST_MUL_REAL;
                    end
                end
                ST_MUL_REAL: begin
                    first_product_q <= multiplier_signed_w[63:0];
                    state_q <= ST_MUL_IMAG;
                end
                ST_MUL_IMAG: begin
                    if (score_sum_overflow_w) begin
                        numeric_overflow_q <= 1'b1;
                        out_valid_q <= 1'b1;
                        state_q <= ST_DONE;
                    end else begin
                        score_raw_q <= score_sum_w[53:0];
                        if (pair_index_q == LAST_PAIR) begin
                            state_q <= ST_SCALE_HIGH;
                        end else begin
                            pair_index_q <= pair_index_q +
                                {{(PAIR_INDEX_WIDTH-1){1'b0}}, 1'b1};
                            state_q <= ST_PAIR_WAIT;
                        end
                    end
                end
                ST_SCALE_HIGH: begin
                    scale_high_product_q <= multiplier_signed_w;
                    state_q <= ST_SCALE_LOW;
                end
                ST_SCALE_LOW: begin
                    scaled_product_q <= scaled_product_combined_w;
                    state_q <= ST_ROUND;
                end
                ST_ROUND: begin
                    if (rounded_overflow_w) begin
                        numeric_overflow_q <= 1'b1;
                    end else begin
                        logit_q <= rounded_logit_w;
                    end
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

`default_nettype wire
