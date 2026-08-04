`timescale 1ns/1ps
`default_nettype none

// Contract: layer0_absolute_rope_online_attention_v1.
// One key update for one query head. The caller supplies/commits the private
// SRAM state and streams exactly LANE_COUNT numerator/value lanes. All wide
// rescaling uses a single time-multiplexed unsigned 32x32 multiplier.
module ace2_absolute_rope_online_attention_core #(
    parameter integer LANE_COUNT = 64
) (
    input  wire                 clk_i,
    input  wire                 rst_ni,
    input  wire                 clear_i,

    input  wire                 start_valid_i,
    output wire                 start_ready_o,
    input  wire                 state_valid_i,
    input  wire signed [63:0]   maximum_i,
    input  wire [47:0]          denominator_i,
    input  wire signed [63:0]   logit_q12_20_i,

    input  wire                 lane_valid_i,
    output wire                 lane_ready_o,
    input  wire signed [55:0]   numerator_i,
    input  wire signed [7:0]    value_i,

    output wire                 lane_out_valid_o,
    input  wire                 lane_out_ready_i,
    output wire signed [55:0]   numerator_o,
    output wire                 lane_last_o,
    output wire signed [63:0]   maximum_o,
    output wire [47:0]          denominator_o,
    output wire [31:0]          weight_q31_o,
    output wire                 transaction_done_o,
    output wire                 descriptor_error_o,
    output wire                 numeric_overflow_o
);
    localparam integer LANE_INDEX_WIDTH =
        (LANE_COUNT <= 1) ? 1 : $clog2(LANE_COUNT);
    localparam [LANE_INDEX_WIDTH-1:0] LAST_LANE =
        LANE_INDEX_WIDTH'(LANE_COUNT - 1);
    localparam [31:0] Q31_ONE = 32'h80000000;
    localparam signed [64:0] EXP_CUTOFF = -65'sd16777216;

    localparam [4:0] ST_IDLE        = 5'd0;
    localparam [4:0] ST_EXP_MUL     = 5'd1;
    localparam [4:0] ST_EXP_ROUND   = 5'd2;
    localparam [4:0] ST_DEN_HI      = 5'd3;
    localparam [4:0] ST_DEN_LO      = 5'd4;
    localparam [4:0] ST_DEN_ROUND   = 5'd5;
    localparam [4:0] ST_LANE_WAIT   = 5'd6;
    localparam [4:0] ST_LANE_MUL    = 5'd7;
    localparam [4:0] ST_NUM_HI      = 5'd8;
    localparam [4:0] ST_NUM_LO      = 5'd9;
    localparam [4:0] ST_NUM_ROUND   = 5'd10;
    localparam [4:0] ST_LANE_OUT    = 5'd11;
    localparam [1:0] MODE_FIRST     = 2'd0;
    localparam [1:0] MODE_NO_NEW_MAX = 2'd1;
    localparam [1:0] MODE_NEW_MAX   = 2'd2;

    reg [4:0] state_q;
    reg [1:0] mode_q;
    reg [LANE_INDEX_WIDTH-1:0] lane_index_q;
    reg signed [63:0] maximum_q;
    reg [47:0] denominator_q;
    reg [31:0] weight_q;
    reg [15:0] exp_fraction_q;
    reg [31:0] exp_base_q;
    reg signed [64:0] exp_product_q;
    reg [63:0] high_product_q;
    reg [63:0] low_product_q;
    reg signed [64:0] signed_high_product_q;
    reg [47:0] denominator_input_q;
    reg signed [55:0] lane_numerator_q;
    reg signed [7:0] lane_value_q;
    reg signed [55:0] numerator_out_q;
    reg lane_out_valid_q;
    reg transaction_done_q;
    reg descriptor_error_q;
    reg numeric_overflow_q;

    `include "ace2_exp_q31_lut.svh"

    wire signed [64:0] exp_delta_w =
        (logit_q12_20_i <= maximum_i) ?
        ($signed({logit_q12_20_i[63], logit_q12_20_i}) -
         $signed({maximum_i[63], maximum_i})) :
        ($signed({maximum_i[63], maximum_i}) -
         $signed({logit_q12_20_i[63], logit_q12_20_i}));
    wire [24:0] exp_delta_twos_low_w = ~exp_delta_w[24:0] + 25'd1;
    wire [24:0] exp_magnitude_w = exp_delta_w[64] ?
        exp_delta_twos_low_w : exp_delta_w[24:0];
    wire [8:0] exp_index_w = exp_magnitude_w[24:16];
    wire [15:0] exp_fraction_w = exp_magnitude_w[15:0];
    wire [31:0] exp_base_w = ace2_exp_q31_lut(exp_index_w);
    wire [31:0] exp_next_w = ace2_exp_q31_lut(exp_index_w + 9'd1);
    wire signed [32:0] exp_difference_w =
        $signed({1'b0, exp_next_w}) - $signed({1'b0, exp_base_w});
    wire [31:0] exp_difference_magnitude_w = exp_difference_w[32] ?
        (~exp_difference_w[31:0] + 32'd1) : exp_difference_w[31:0];

    wire [23:0] denominator_low_w = denominator_input_q[23:0];
    wire [23:0] denominator_high_w = denominator_input_q[47:24];
    wire signed [31:0] numerator_high_w = lane_numerator_q[55:24];
    wire [31:0] numerator_high_magnitude_w = numerator_high_w[31] ?
        (~numerator_high_w + 32'd1) : numerator_high_w;
    wire [23:0] numerator_low_w = lane_numerator_q[23:0];
    wire [7:0] value_magnitude_w = lane_value_q[7] ?
        (~lane_value_q + 8'd1) : lane_value_q;

    reg [31:0] multiplier_a_w;
    reg [31:0] multiplier_b_w;
    reg multiplier_negative_w;
    wire [63:0] multiplier_magnitude_w = multiplier_a_w * multiplier_b_w;
    wire signed [64:0] multiplier_signed_w = multiplier_negative_w ?
        -$signed({1'b0, multiplier_magnitude_w}) :
         $signed({1'b0, multiplier_magnitude_w});

    always @* begin
        multiplier_a_w = 32'd0;
        multiplier_b_w = 32'd0;
        multiplier_negative_w = 1'b0;
        case (state_q)
            ST_EXP_MUL: begin
                multiplier_a_w = exp_difference_magnitude_w;
                multiplier_b_w = {16'd0, exp_fraction_q};
                multiplier_negative_w = exp_difference_w[32];
            end
            ST_DEN_HI: begin
                multiplier_a_w = {8'd0, denominator_high_w};
                multiplier_b_w = weight_q;
            end
            ST_DEN_LO: begin
                multiplier_a_w = {8'd0, denominator_low_w};
                multiplier_b_w = weight_q;
            end
            ST_LANE_MUL: begin
                multiplier_a_w = weight_q;
                multiplier_b_w = {24'd0, value_magnitude_w};
                multiplier_negative_w = lane_value_q[7];
            end
            ST_NUM_HI: begin
                multiplier_a_w = numerator_high_magnitude_w;
                multiplier_b_w = weight_q;
                multiplier_negative_w = numerator_high_w[31];
            end
            ST_NUM_LO: begin
                multiplier_a_w = {8'd0, numerator_low_w};
                multiplier_b_w = weight_q;
            end
            default: begin
            end
        endcase
    end

    function automatic signed [64:0] rne_shift16_s65;
        input signed [64:0] value;
        reg negative;
        reg [64:0] magnitude;
        reg [48:0] quotient;
        reg [15:0] remainder;
        reg increment;
        reg [49:0] rounded;
        begin
            negative = value[64];
            magnitude = negative ? (~value + 65'd1) : value;
            quotient = magnitude[64:16];
            remainder = magnitude[15:0];
            increment = (remainder > 16'h8000) ||
                        ((remainder == 16'h8000) && quotient[0]);
            rounded = {1'b0, quotient} + {{49{1'b0}}, increment};
            rne_shift16_s65 = negative ?
                -$signed({{15{1'b0}}, rounded}) :
                 $signed({{15{1'b0}}, rounded});
        end
    endfunction

    wire signed [64:0] exp_correction_w = rne_shift16_s65(exp_product_q);
    wire signed [65:0] exp_weight_signed_w =
        $signed({34'd0, exp_base_q}) +
        $signed({exp_correction_w[64], exp_correction_w});

    wire [79:0] denominator_product_w =
        ({16'd0, high_product_q} << 24) + {16'd0, low_product_q};
    wire [48:0] denominator_quotient_w = denominator_product_w[79:31];
    wire [30:0] denominator_remainder_w = denominator_product_w[30:0];
    wire denominator_increment_w =
        (denominator_remainder_w > 31'h40000000) ||
        ((denominator_remainder_w == 31'h40000000) && denominator_quotient_w[0]);
    wire [49:0] denominator_rounded_w =
        {1'b0, denominator_quotient_w} + {{49{1'b0}}, denominator_increment_w};

    wire signed [88:0] numerator_product_w =
        ($signed({{24{signed_high_product_q[64]}}, signed_high_product_q}) <<< 24) +
        $signed({25'd0, low_product_q});
    wire numerator_product_negative_w = numerator_product_w[88];
    wire [88:0] numerator_product_magnitude_w = numerator_product_negative_w ?
        (~numerator_product_w + 89'd1) : numerator_product_w;
    wire [57:0] numerator_quotient_w = numerator_product_magnitude_w[88:31];
    wire [30:0] numerator_remainder_w = numerator_product_magnitude_w[30:0];
    wire numerator_increment_w =
        (numerator_remainder_w > 31'h40000000) ||
        ((numerator_remainder_w == 31'h40000000) && numerator_quotient_w[0]);
    wire [58:0] numerator_rounded_magnitude_w =
        {1'b0, numerator_quotient_w} + {{58{1'b0}}, numerator_increment_w};
    wire signed [58:0] numerator_rescaled_w = numerator_product_negative_w ?
        -$signed(numerator_rounded_magnitude_w) :
         $signed(numerator_rounded_magnitude_w);
    wire signed [58:0] numerator_new_max_w = numerator_rescaled_w +
        ($signed({{51{lane_value_q[7]}}, lane_value_q}) <<< 31);
    wire signed [64:0] numerator_no_new_w =
        $signed({{9{lane_numerator_q[55]}}, lane_numerator_q}) +
        multiplier_signed_w;

    assign start_ready_o = (state_q == ST_IDLE) && !lane_out_valid_q;
    assign lane_ready_o = state_q == ST_LANE_WAIT;
    assign lane_out_valid_o = lane_out_valid_q;
    assign numerator_o = numerator_out_q;
    assign lane_last_o = lane_index_q == LAST_LANE;
    assign maximum_o = maximum_q;
    assign denominator_o = denominator_q;
    assign weight_q31_o = weight_q;
    assign transaction_done_o = transaction_done_q;
    assign descriptor_error_o = descriptor_error_q;
    assign numeric_overflow_o = numeric_overflow_q;

    always @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            state_q <= ST_IDLE;
            mode_q <= MODE_FIRST;
            lane_index_q <= {LANE_INDEX_WIDTH{1'b0}};
            maximum_q <= 64'sd0;
            denominator_q <= 48'd0;
            weight_q <= 32'd0;
            exp_fraction_q <= 16'd0;
            exp_base_q <= 32'd0;
            exp_product_q <= 65'sd0;
            high_product_q <= 64'd0;
            low_product_q <= 64'd0;
            signed_high_product_q <= 65'sd0;
            denominator_input_q <= 48'd0;
            lane_numerator_q <= 56'sd0;
            lane_value_q <= 8'sd0;
            numerator_out_q <= 56'sd0;
            lane_out_valid_q <= 1'b0;
            transaction_done_q <= 1'b0;
            descriptor_error_q <= 1'b0;
            numeric_overflow_q <= 1'b0;
        end else if (clear_i) begin
            state_q <= ST_IDLE;
            lane_index_q <= {LANE_INDEX_WIDTH{1'b0}};
            lane_out_valid_q <= 1'b0;
            transaction_done_q <= 1'b0;
            descriptor_error_q <= 1'b0;
            numeric_overflow_q <= 1'b0;
        end else begin
            transaction_done_q <= 1'b0;
            case (state_q)
                ST_IDLE: begin
                    if (start_valid_i && start_ready_o) begin
                        lane_index_q <= {LANE_INDEX_WIDTH{1'b0}};
                        descriptor_error_q <=
                            (!state_valid_i && (denominator_i != 48'd0)) ||
                            (state_valid_i && (denominator_i == 48'd0));
                        numeric_overflow_q <= 1'b0;
                        denominator_input_q <= denominator_i;
                        if ((!state_valid_i && (denominator_i != 48'd0)) ||
                            (state_valid_i && (denominator_i == 48'd0))) begin
                            state_q <= ST_IDLE;
                        end else if (!state_valid_i) begin
                            mode_q <= MODE_FIRST;
                            maximum_q <= logit_q12_20_i;
                            denominator_q <= {16'd0, Q31_ONE};
                            weight_q <= Q31_ONE;
                            state_q <= ST_LANE_WAIT;
                        end else begin
                            mode_q <= (logit_q12_20_i <= maximum_i) ?
                                MODE_NO_NEW_MAX : MODE_NEW_MAX;
                            maximum_q <= (logit_q12_20_i <= maximum_i) ?
                                maximum_i : logit_q12_20_i;
                            exp_fraction_q <= exp_fraction_w;
                            exp_base_q <= exp_base_w;
                            if (exp_delta_w == 65'sd0) begin
                                weight_q <= Q31_ONE;
                                if (logit_q12_20_i <= maximum_i) begin
                                    denominator_q <= denominator_i + {16'd0, Q31_ONE};
                                    state_q <= ST_LANE_WAIT;
                                end else begin
                                    state_q <= ST_DEN_HI;
                                end
                            end else if (exp_delta_w <= EXP_CUTOFF) begin
                                weight_q <= 32'd0;
                                if (logit_q12_20_i <= maximum_i) begin
                                    denominator_q <= denominator_i;
                                    state_q <= ST_LANE_WAIT;
                                end else begin
                                    state_q <= ST_DEN_HI;
                                end
                            end else begin
                                state_q <= ST_EXP_MUL;
                            end
                        end
                    end
                end
                ST_EXP_MUL: begin
                    exp_product_q <= multiplier_signed_w;
                    state_q <= ST_EXP_ROUND;
                end
                ST_EXP_ROUND: begin
                    if ((exp_weight_signed_w < 66'sd0) ||
                        (exp_weight_signed_w > 66'sd2147483648)) begin
                        numeric_overflow_q <= 1'b1;
                        state_q <= ST_IDLE;
                    end else begin
                        weight_q <= exp_weight_signed_w[31:0];
                        if (mode_q == MODE_NO_NEW_MAX) begin
                            if ({1'b0, denominator_i} + {17'd0, exp_weight_signed_w[31:0]} >=
                                49'h1000000000000) begin
                                numeric_overflow_q <= 1'b1;
                                state_q <= ST_IDLE;
                            end else begin
                                denominator_q <= denominator_i +
                                    {16'd0, exp_weight_signed_w[31:0]};
                                state_q <= ST_LANE_WAIT;
                            end
                        end else begin
                            state_q <= ST_DEN_HI;
                        end
                    end
                end
                ST_DEN_HI: begin
                    high_product_q <= multiplier_magnitude_w;
                    state_q <= ST_DEN_LO;
                end
                ST_DEN_LO: begin
                    low_product_q <= multiplier_magnitude_w;
                    state_q <= ST_DEN_ROUND;
                end
                ST_DEN_ROUND: begin
                    if (denominator_rounded_w + 50'd2147483648 >= 50'h1000000000000) begin
                        numeric_overflow_q <= 1'b1;
                        state_q <= ST_IDLE;
                    end else begin
                        denominator_q <= denominator_rounded_w[47:0] +
                            {16'd0, Q31_ONE};
                        state_q <= ST_LANE_WAIT;
                    end
                end
                ST_LANE_WAIT: begin
                    if (lane_valid_i && lane_ready_o) begin
                        lane_numerator_q <= numerator_i;
                        lane_value_q <= value_i;
                        if (mode_q == MODE_FIRST) begin
                            numerator_out_q <=
                                $signed({{48{value_i[7]}}, value_i}) <<< 31;
                            lane_out_valid_q <= 1'b1;
                            state_q <= ST_LANE_OUT;
                        end else if (mode_q == MODE_NO_NEW_MAX) begin
                            state_q <= ST_LANE_MUL;
                        end else begin
                            state_q <= ST_NUM_HI;
                        end
                    end
                end
                ST_LANE_MUL: begin
                    if ((numerator_no_new_w < -65'sd36028797018963968) ||
                        (numerator_no_new_w >  65'sd36028797018963967)) begin
                        numeric_overflow_q <= 1'b1;
                        state_q <= ST_IDLE;
                    end else begin
                        numerator_out_q <= numerator_no_new_w[55:0];
                        lane_out_valid_q <= 1'b1;
                        state_q <= ST_LANE_OUT;
                    end
                end
                ST_NUM_HI: begin
                    signed_high_product_q <= multiplier_signed_w;
                    state_q <= ST_NUM_LO;
                end
                ST_NUM_LO: begin
                    low_product_q <= multiplier_magnitude_w;
                    state_q <= ST_NUM_ROUND;
                end
                ST_NUM_ROUND: begin
                    if ((numerator_new_max_w < -59'sd36028797018963968) ||
                        (numerator_new_max_w >  59'sd36028797018963967)) begin
                        numeric_overflow_q <= 1'b1;
                        state_q <= ST_IDLE;
                    end else begin
                        numerator_out_q <= numerator_new_max_w[55:0];
                        lane_out_valid_q <= 1'b1;
                        state_q <= ST_LANE_OUT;
                    end
                end
                ST_LANE_OUT: begin
                    if (lane_out_valid_q && lane_out_ready_i) begin
                        lane_out_valid_q <= 1'b0;
                        if (lane_index_q == LAST_LANE) begin
                            transaction_done_q <= 1'b1;
                            state_q <= ST_IDLE;
                        end else begin
                            lane_index_q <= lane_index_q +
                                {{(LANE_INDEX_WIDTH-1){1'b0}}, 1'b1};
                            state_q <= ST_LANE_WAIT;
                        end
                    end
                end
                default: state_q <= ST_IDLE;
            endcase
        end
    end
endmodule


/* verilator lint_off DECLFILENAME */
// Serialized signed round-to-nearest-even numerator/denominator finalizer.
// It performs one 56-bit restoring divide in 56 steps and saturates to int8.
module ace2_absolute_rope_online_attention_finalize_core (
    input  wire                 clk_i,
    input  wire                 rst_ni,
    input  wire                 clear_i,
    input  wire                 start_valid_i,
    output wire                 start_ready_o,
    input  wire signed [55:0]   numerator_i,
    input  wire [47:0]          denominator_i,
    output wire                 out_valid_o,
    input  wire                 out_ready_i,
    output wire signed [7:0]    value_o,
    output wire                 descriptor_error_o
);
    localparam [1:0] ST_IDLE = 2'd0;
    localparam [1:0] ST_DIV = 2'd1;
    localparam [1:0] ST_ROUND = 2'd2;
    localparam [1:0] ST_OUT = 2'd3;
    reg [1:0] state_q;
    reg negative_q;
    reg [55:0] dividend_q;
    reg [47:0] denominator_q;
    reg [48:0] remainder_q;
    reg [55:0] quotient_q;
    reg [5:0] bit_q;
    reg signed [7:0] value_q;
    reg descriptor_error_q;

    wire [48:0] shifted_remainder_w =
        {remainder_q[47:0], dividend_q[bit_q]};
    wire subtract_w = shifted_remainder_w >= {1'b0, denominator_q};
    wire [48:0] remainder_next_w = subtract_w ?
        shifted_remainder_w - {1'b0, denominator_q} : shifted_remainder_w;
    wire [55:0] quotient_next_w = quotient_q |
        (subtract_w ? (56'd1 << bit_q) : 56'd0);
    wire [49:0] doubled_remainder_w = {remainder_q, 1'b0};
    wire [49:0] extended_denominator_w = {2'd0, denominator_q};
    wire round_increment_w =
        (doubled_remainder_w > extended_denominator_w) ||
        ((doubled_remainder_w == extended_denominator_w) && quotient_q[0]);
    wire [56:0] rounded_magnitude_w =
        {1'b0, quotient_q} + {{56{1'b0}}, round_increment_w};

    assign start_ready_o = state_q == ST_IDLE;
    assign out_valid_o = state_q == ST_OUT;
    assign value_o = value_q;
    assign descriptor_error_o = descriptor_error_q;

    always @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            state_q <= ST_IDLE;
            negative_q <= 1'b0;
            dividend_q <= 56'd0;
            denominator_q <= 48'd0;
            remainder_q <= 49'd0;
            quotient_q <= 56'd0;
            bit_q <= 6'd0;
            value_q <= 8'sd0;
            descriptor_error_q <= 1'b0;
        end else if (clear_i) begin
            state_q <= ST_IDLE;
            descriptor_error_q <= 1'b0;
        end else begin
            case (state_q)
                ST_IDLE: begin
                    if (start_valid_i && start_ready_o) begin
                        descriptor_error_q <= denominator_i == 48'd0;
                        if (denominator_i == 48'd0) begin
                            value_q <= 8'sd0;
                            state_q <= ST_OUT;
                        end else begin
                            negative_q <= numerator_i[55];
                            dividend_q <= numerator_i[55] ?
                                (~numerator_i + 56'd1) : numerator_i;
                            denominator_q <= denominator_i;
                            remainder_q <= 49'd0;
                            quotient_q <= 56'd0;
                            bit_q <= 6'd55;
                            state_q <= ST_DIV;
                        end
                    end
                end
                ST_DIV: begin
                    remainder_q <= remainder_next_w;
                    quotient_q <= quotient_next_w;
                    if (bit_q == 6'd0) begin
                        state_q <= ST_ROUND;
                    end else begin
                        bit_q <= bit_q - 6'd1;
                    end
                end
                ST_ROUND: begin
                    if (!negative_q && (rounded_magnitude_w > 57'd127)) begin
                        value_q <= 8'sd127;
                    end else if (negative_q && (rounded_magnitude_w > 57'd128)) begin
                        value_q <= 8'sh80;
                    end else if (negative_q) begin
                        value_q <= $signed((~rounded_magnitude_w[7:0]) + 8'd1);
                    end else begin
                        value_q <= $signed(rounded_magnitude_w[7:0]);
                    end
                    state_q <= ST_OUT;
                end
                ST_OUT: begin
                    if (out_ready_i) begin
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
