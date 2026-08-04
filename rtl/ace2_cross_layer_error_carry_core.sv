`timescale 1ns/1ps
`default_nettype none
/* verilator lint_off DECLFILENAME */

// Frozen contract: cross_layer_quantization_error_carry_final_output_v1.
// Standalone producer lane. The shell is intentionally not modified or
// admitted by this RTL-stage implementation.
module ace2_quantization_error_carry_lane_core (
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
    output logic signed [7:0]          hidden_s8_o,
    output logic signed [15:0]         carry_s16_q15_o,
    output logic                       descriptor_error_o,
    output logic                       numeric_overflow_o,
    output logic signed [95:0]         numerator_s96_o,
    output logic        [63:0]         denominator_u64_o,
    output logic signed [7:0]          common_exponent_s8_o,
    output logic        [5:0]          latency_cycles_u6_o
);
    localparam logic [2:0] ST_IDLE = 3'd0;
    localparam logic [2:0] ST_PREPARE = 3'd1;
    localparam logic [2:0] ST_INTEGER_DIVIDE = 3'd2;
    localparam logic [2:0] ST_INTEGER_FINALIZE = 3'd3;
    localparam logic [2:0] ST_CARRY_DIVIDE = 3'd4;
    localparam logic [2:0] ST_FINALIZE = 3'd5;

    logic [2:0] state_q;
    logic out_valid_q;
    logic signed [7:0] hidden_q;
    logic signed [15:0] carry_q;
    logic descriptor_error_q, numeric_overflow_q;
    logic signed [95:0] numerator_q;
    logic [95:0] numerator_magnitude_q;
    logic [63:0] denominator_q;
    logic signed [7:0] common_exponent_q;
    logic numerator_negative_q;
    logic [95:0] integer_remainder_q;
    logic [7:0] integer_quotient_q;
    logic [2:0] integer_bit_q;
    logic error_negative_q;
    logic [95:0] carry_remainder_q;
    logic [14:0] carry_quotient_q;
    logic [3:0] carry_bit_q;
    logic [5:0] latency_q;

    logic accumulator_scale_valid_w, residual_scale_valid_w, destination_scale_valid_w;
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
    logic signed [95:0] numerator_w;
    logic [95:0] numerator_magnitude_w;
    logic [63:0] denominator_w;
    logic denominator_overflow_w;

    logic [96:0] doubled_magnitude_w;
    logic [96:0] positive_limit_w, negative_limit_w;
    logic positive_overflow_w, negative_overflow_w;
    logic [95:0] integer_shifted_denominator_w;
    logic [95:0] integer_remainder_next_w;
    logic [7:0] integer_quotient_next_w;
    logic [96:0] doubled_integer_remainder_w;
    logic integer_round_increment_w;
    logic [8:0] integer_rounded_magnitude_w;
    logic signed [8:0] hidden_candidate_w;
    logic signed [95:0] hidden_times_denominator_w;
    logic signed [95:0] error_w;
    logic [95:0] error_magnitude_w;
    logic [95:0] carry_numerator_w;
    logic [95:0] carry_shifted_denominator_w;
    logic [95:0] carry_remainder_next_w;
    logic [14:0] carry_quotient_next_w;
    logic [96:0] doubled_carry_remainder_w;
    logic carry_round_increment_w;
    logic [15:0] carry_rounded_magnitude_w;
    logic signed [16:0] carry_candidate_w;

    function automatic logic scale32_valid(input logic [31:0] record);
        logic signed [7:0] exponent;
        begin
            exponent = $signed(record[23:16]);
            scale32_valid = (record[31:24] == 8'd0) &&
                            (record[15:0] >= 16'h8000) &&
                            (exponent >= -8'sd24) && (exponent <= 8'sd4);
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

    assign start_ready_o = rst_ni && !clear_i && (state_q == ST_IDLE) &&
                           (!out_valid_q || out_ready_i);
    assign out_valid_o = out_valid_q;
    assign hidden_s8_o = hidden_q;
    assign carry_s16_q15_o = carry_q;
    assign descriptor_error_o = descriptor_error_q;
    assign numeric_overflow_o = numeric_overflow_q;
    assign numerator_s96_o = numerator_q;
    assign denominator_u64_o = denominator_q;
    assign common_exponent_s8_o = common_exponent_q;
    assign latency_cycles_u6_o = latency_q;

    always_comb begin
        accumulator_scale_valid_w = scale32_valid(accumulator_scale32_i);
        residual_scale_valid_w = scale32_valid(residual_scale32_i);
        destination_scale_valid_w = scale32_valid(destination_scale32_i);
        all_scales_valid_w = accumulator_scale_valid_w && residual_scale_valid_w &&
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
        numerator_w = '0;
        numerator_magnitude_w = '0;
        denominator_w = '0;
        denominator_overflow_w = 1'b0;
        if (all_scales_valid_w) begin
            accumulator_shift_w = exponent_delta(accumulator_exp_w, common_exponent_w);
            residual_shift_w = exponent_delta(residual_exp_w, common_exponent_w);
            destination_shift_w = exponent_delta(destination_exp_w, common_exponent_w);
            accumulator_product_w = accumulator_s32_i * $signed({1'b0, accumulator_sig_w});
            residual_product_w = residual_s8_i * $signed({1'b0, residual_sig_w});
            accumulator_term_w = $signed({{48{accumulator_product_w[47]}}, accumulator_product_w})
                                 <<< accumulator_shift_w;
            residual_term_w = $signed({{72{residual_product_w[23]}}, residual_product_w})
                              <<< residual_shift_w;
            numerator_sum_w = $signed({accumulator_term_w[95], accumulator_term_w}) +
                              $signed({residual_term_w[95], residual_term_w});
            numerator_overflow_w = numerator_sum_w[96] != numerator_sum_w[95];
            numerator_w = numerator_sum_w[95:0];
            numerator_magnitude_w = numerator_w[95] ? $unsigned(-numerator_w) : $unsigned(numerator_w);
            denominator_w = {{48{1'b0}}, destination_sig_w} << destination_shift_w;
            denominator_overflow_w = denominator_w == 64'd0;
        end

        doubled_magnitude_w = {numerator_magnitude_q, 1'b0};
        positive_limit_w = multiply_u64_by_255(denominator_q);
        negative_limit_w = multiply_u64_by_257(denominator_q);
        positive_overflow_w = !numerator_negative_q && (doubled_magnitude_w >= positive_limit_w);
        negative_overflow_w = numerator_negative_q && (doubled_magnitude_w > negative_limit_w);

        integer_shifted_denominator_w = {{32{1'b0}}, denominator_q} << integer_bit_q;
        integer_remainder_next_w = integer_remainder_q;
        integer_quotient_next_w = integer_quotient_q;
        if (integer_remainder_q >= integer_shifted_denominator_w) begin
            integer_remainder_next_w = integer_remainder_q - integer_shifted_denominator_w;
            integer_quotient_next_w[integer_bit_q] = 1'b1;
        end
        doubled_integer_remainder_w = {integer_remainder_q, 1'b0};
        integer_round_increment_w =
            (doubled_integer_remainder_w > {33'd0, denominator_q}) ||
            ((doubled_integer_remainder_w == {33'd0, denominator_q}) && integer_quotient_q[0]);
        integer_rounded_magnitude_w = {1'b0, integer_quotient_q} +
                                      {{8{1'b0}}, integer_round_increment_w};
        hidden_candidate_w = numerator_negative_q ?
            -$signed(integer_rounded_magnitude_w) : $signed(integer_rounded_magnitude_w);
        hidden_times_denominator_w = $signed(hidden_candidate_w) * $signed({1'b0, denominator_q});
        error_w = numerator_q - hidden_times_denominator_w;
        error_magnitude_w = error_w[95] ? $unsigned(-error_w) : $unsigned(error_w);
        carry_numerator_w = error_magnitude_w << 15;

        carry_shifted_denominator_w = {{32{1'b0}}, denominator_q} << carry_bit_q;
        carry_remainder_next_w = carry_remainder_q;
        carry_quotient_next_w = carry_quotient_q;
        if (carry_remainder_q >= carry_shifted_denominator_w) begin
            carry_remainder_next_w = carry_remainder_q - carry_shifted_denominator_w;
            carry_quotient_next_w[carry_bit_q] = 1'b1;
        end
        doubled_carry_remainder_w = {carry_remainder_q, 1'b0};
        carry_round_increment_w =
            (doubled_carry_remainder_w > {33'd0, denominator_q}) ||
            ((doubled_carry_remainder_w == {33'd0, denominator_q}) && carry_quotient_q[0]);
        carry_rounded_magnitude_w = {1'b0, carry_quotient_q} +
                                    {{15{1'b0}}, carry_round_increment_w};
        carry_candidate_w = error_negative_q ?
            -$signed({1'b0, carry_rounded_magnitude_w}) :
             $signed({1'b0, carry_rounded_magnitude_w});
    end

    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            state_q <= ST_IDLE;
            out_valid_q <= 1'b0;
            hidden_q <= '0;
            carry_q <= '0;
            descriptor_error_q <= 1'b0;
            numeric_overflow_q <= 1'b0;
            numerator_q <= '0;
            numerator_magnitude_q <= '0;
            denominator_q <= '0;
            common_exponent_q <= '0;
            numerator_negative_q <= 1'b0;
            integer_remainder_q <= '0;
            integer_quotient_q <= '0;
            integer_bit_q <= '0;
            error_negative_q <= 1'b0;
            carry_remainder_q <= '0;
            carry_quotient_q <= '0;
            carry_bit_q <= '0;
            latency_q <= '0;
        end else if (clear_i) begin
            state_q <= ST_IDLE;
            out_valid_q <= 1'b0;
            hidden_q <= '0;
            carry_q <= '0;
            descriptor_error_q <= 1'b0;
            numeric_overflow_q <= 1'b0;
            numerator_q <= '0;
            numerator_magnitude_q <= '0;
            denominator_q <= '0;
            common_exponent_q <= '0;
            numerator_negative_q <= 1'b0;
            integer_remainder_q <= '0;
            integer_quotient_q <= '0;
            integer_bit_q <= '0;
            error_negative_q <= 1'b0;
            carry_remainder_q <= '0;
            carry_quotient_q <= '0;
            carry_bit_q <= '0;
            latency_q <= '0;
        end else begin
            if (out_valid_q && out_ready_i)
                out_valid_q <= 1'b0;
            case (state_q)
                ST_IDLE: begin
                    if (start_valid_i && start_ready_o) begin
                        hidden_q <= '0;
                        carry_q <= '0;
                        descriptor_error_q <= 1'b0;
                        numeric_overflow_q <= 1'b0;
                        numerator_q <= '0;
                        denominator_q <= '0;
                        common_exponent_q <= '0;
                        latency_q <= '0;
                        if (!all_scales_valid_w) begin
                            descriptor_error_q <= 1'b1;
                            out_valid_q <= 1'b1;
                        end else if (numerator_overflow_w || denominator_overflow_w) begin
                            numeric_overflow_q <= 1'b1;
                            out_valid_q <= 1'b1;
                        end else begin
                            numerator_q <= numerator_w;
                            numerator_magnitude_q <= numerator_magnitude_w;
                            denominator_q <= denominator_w;
                            common_exponent_q <= common_exponent_w;
                            numerator_negative_q <= numerator_w[95];
                            state_q <= ST_PREPARE;
                        end
                    end
                end
                ST_PREPARE: begin
                    latency_q <= 6'd1;
                    if (positive_overflow_w || negative_overflow_w) begin
                        numeric_overflow_q <= 1'b1;
                        out_valid_q <= 1'b1;
                        state_q <= ST_IDLE;
                    end else begin
                        integer_remainder_q <= numerator_magnitude_q;
                        integer_quotient_q <= '0;
                        integer_bit_q <= 3'd7;
                        state_q <= ST_INTEGER_DIVIDE;
                    end
                end
                ST_INTEGER_DIVIDE: begin
                    integer_remainder_q <= integer_remainder_next_w;
                    integer_quotient_q <= integer_quotient_next_w;
                    latency_q <= latency_q + 6'd1;
                    if (integer_bit_q == 3'd0)
                        state_q <= ST_INTEGER_FINALIZE;
                    else
                        integer_bit_q <= integer_bit_q - 3'd1;
                end
                ST_INTEGER_FINALIZE: begin
                    latency_q <= latency_q + 6'd1;
                    hidden_q <= hidden_candidate_w[7:0];
                    error_negative_q <= error_w[95];
                    carry_remainder_q <= carry_numerator_w;
                    carry_quotient_q <= '0;
                    carry_bit_q <= 4'd14;
                    state_q <= ST_CARRY_DIVIDE;
                end
                ST_CARRY_DIVIDE: begin
                    carry_remainder_q <= carry_remainder_next_w;
                    carry_quotient_q <= carry_quotient_next_w;
                    latency_q <= latency_q + 6'd1;
                    if (carry_bit_q == 4'd0)
                        state_q <= ST_FINALIZE;
                    else
                        carry_bit_q <= carry_bit_q - 4'd1;
                end
                ST_FINALIZE: begin
                    latency_q <= latency_q + 6'd1;
                    if ((carry_candidate_w < -17'sd16384) || (carry_candidate_w > 17'sd16384)) begin
                        numeric_overflow_q <= 1'b1;
                        carry_q <= '0;
                    end else begin
                        carry_q <= carry_candidate_w[15:0];
                    end
                    out_valid_q <= 1'b1;
                    state_q <= ST_IDLE;
                end
                default: state_q <= ST_IDLE;
            endcase
        end
    end
endmodule


// One 896-entry carry buffer plus producer/consumer identity state. Writes are
// provisional until the final producer lane commits the transaction. A valid
// carry is cleared only by a terminal consumer result or by reset/clear.
module ace2_error_carry_state_core #(
    parameter integer HIDDEN_SIZE = 896,
    parameter integer TOKEN_ID_WIDTH = 32,
    parameter integer MODEL_ID_WIDTH = 64,
    parameter integer TAG_WIDTH = 16
) (
    input  logic                         clk_i,
    input  logic                         rst_ni,
    input  logic                         clear_i,
    input  logic                         metadata_valid_i,
    input  logic                         producer_start_valid_i,
    output logic                         producer_start_ready_o,
    input  logic [4:0]                   producer_layer_id_i,
    input  logic [TOKEN_ID_WIDTH-1:0]    producer_token_id_i,
    input  logic [MODEL_ID_WIDTH-1:0]    producer_model_id_i,
    input  logic [TAG_WIDTH-1:0]         producer_completion_tag_i,
    input  logic                         producer_lane_valid_i,
    output logic                         producer_lane_ready_o,
    input  logic signed [15:0]           producer_carry_s16_q15_i,
    output logic                         producer_done_valid_o,
    input  logic                         producer_done_ready_i,
    output logic                         producer_done_descriptor_error_o,
    output logic                         producer_done_numeric_overflow_o,
    input  logic                         consumer_start_valid_i,
    output logic                         consumer_start_ready_o,
    input  logic [4:0]                   consumer_layer_id_i,
    input  logic [TOKEN_ID_WIDTH-1:0]    consumer_token_id_i,
    input  logic [MODEL_ID_WIDTH-1:0]    consumer_model_id_i,
    input  logic [TAG_WIDTH-1:0]         consumer_completion_tag_i,
    output logic                         consumer_active_o,
    input  logic [$clog2(HIDDEN_SIZE)-1:0] consumer_read_index_i,
    output logic signed [15:0]           consumer_carry_s16_q15_o,
    output logic                         consumer_read_index_valid_o,
    input  logic                         consumer_finish_valid_i,
    input  logic                         consumer_finish_success_i,
    output logic                         consumer_done_valid_o,
    input  logic                         consumer_done_ready_i,
    output logic                         consumer_done_descriptor_error_o,
    output logic [TAG_WIDTH-1:0]         consumer_completion_tag_o,
    output logic                         carry_valid_o,
    output logic [4:0]                   carry_producer_layer_id_o,
    output logic [TOKEN_ID_WIDTH-1:0]    carry_token_id_o,
    output logic [MODEL_ID_WIDTH-1:0]    carry_model_id_o,
    output logic [TAG_WIDTH-1:0]         carry_producer_completion_tag_o
);
    localparam integer INDEX_WIDTH = (HIDDEN_SIZE <= 1) ? 1 : $clog2(HIDDEN_SIZE);
    localparam logic [1:0] ST_IDLE = 2'd0;
    localparam logic [1:0] ST_PRODUCE = 2'd1;
    localparam logic [1:0] ST_CONSUME = 2'd2;

    logic [1:0] state_q;
    logic signed [15:0] carry_mem_q [0:HIDDEN_SIZE-1];
    logic [INDEX_WIDTH-1:0] producer_index_q;
    logic carry_valid_q;
    logic [4:0] producer_layer_q;
    logic [TOKEN_ID_WIDTH-1:0] token_id_q;
    logic [MODEL_ID_WIDTH-1:0] model_id_q;
    logic [TAG_WIDTH-1:0] producer_tag_q;
    logic [TAG_WIDTH-1:0] consumer_tag_q;
    logic producer_done_valid_q, producer_descriptor_error_q, producer_numeric_overflow_q;
    logic consumer_done_valid_q, consumer_descriptor_error_q;
    logic command_idle_ready_w;
    logic producer_start_accepted_w, consumer_start_accepted_w;
    logic consumer_read_index_in_range_w;

    assign command_idle_ready_w = rst_ni && !clear_i && (state_q == ST_IDLE) &&
                                  !producer_done_valid_q && !consumer_done_valid_q;
    // Producer has explicit priority when both channels assert valid. Exactly
    // one ready is then asserted, so no unserviced ready/valid handshake exists.
    assign producer_start_ready_o = command_idle_ready_w &&
                                    (!consumer_start_valid_i || producer_start_valid_i);
    assign producer_start_accepted_w = producer_start_valid_i && producer_start_ready_o;
    assign producer_lane_ready_o = (state_q == ST_PRODUCE) && !producer_done_valid_q;
    assign producer_done_valid_o = producer_done_valid_q;
    assign producer_done_descriptor_error_o = producer_descriptor_error_q;
    assign producer_done_numeric_overflow_o = producer_numeric_overflow_q;
    assign consumer_start_ready_o = command_idle_ready_w && !producer_start_valid_i;
    assign consumer_start_accepted_w = consumer_start_valid_i && consumer_start_ready_o;
    assign consumer_active_o = state_q == ST_CONSUME;
    assign consumer_read_index_in_range_w =
        {1'b0, consumer_read_index_i} < (INDEX_WIDTH+1)'(HIDDEN_SIZE);
    assign consumer_read_index_valid_o = consumer_active_o && consumer_read_index_in_range_w;
    assign consumer_carry_s16_q15_o = consumer_read_index_valid_o ?
                                      carry_mem_q[consumer_read_index_i] : 16'sd0;
    assign consumer_done_valid_o = consumer_done_valid_q;
    assign consumer_done_descriptor_error_o = consumer_descriptor_error_q;
    assign consumer_completion_tag_o = consumer_tag_q;
    assign carry_valid_o = carry_valid_q;
    assign carry_producer_layer_id_o = producer_layer_q;
    assign carry_token_id_o = token_id_q;
    assign carry_model_id_o = model_id_q;
    assign carry_producer_completion_tag_o = producer_tag_q;

    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            state_q <= ST_IDLE;
            producer_index_q <= '0;
            carry_valid_q <= 1'b0;
            producer_layer_q <= '0;
            token_id_q <= '0;
            model_id_q <= '0;
            producer_tag_q <= '0;
            consumer_tag_q <= '0;
            producer_done_valid_q <= 1'b0;
            producer_descriptor_error_q <= 1'b0;
            producer_numeric_overflow_q <= 1'b0;
            consumer_done_valid_q <= 1'b0;
            consumer_descriptor_error_q <= 1'b0;
        end else if (clear_i) begin
            state_q <= ST_IDLE;
            producer_index_q <= '0;
            carry_valid_q <= 1'b0;
            producer_layer_q <= '0;
            token_id_q <= '0;
            model_id_q <= '0;
            producer_tag_q <= '0;
            consumer_tag_q <= '0;
            producer_done_valid_q <= 1'b0;
            producer_descriptor_error_q <= 1'b0;
            producer_numeric_overflow_q <= 1'b0;
            consumer_done_valid_q <= 1'b0;
            consumer_descriptor_error_q <= 1'b0;
        end else begin
            if (producer_done_valid_q && producer_done_ready_i)
                producer_done_valid_q <= 1'b0;
            if (consumer_done_valid_q && consumer_done_ready_i)
                consumer_done_valid_q <= 1'b0;
            // The descriptor completion tag belongs to the accepted consumer
            // transaction. Capture it on the ready/valid handshake before any
            // validity decision and hold it through terminal backpressure.
            if (consumer_start_accepted_w)
                consumer_tag_q <= consumer_completion_tag_i;

            case (state_q)
                ST_IDLE: begin
                    if (producer_start_accepted_w) begin
                        producer_descriptor_error_q <= 1'b0;
                        producer_numeric_overflow_q <= 1'b0;
                        if (!metadata_valid_i || carry_valid_q || (producer_layer_id_i > 5'd23)) begin
                            producer_descriptor_error_q <= 1'b1;
                            producer_done_valid_q <= 1'b1;
                        end else begin
                            producer_layer_q <= producer_layer_id_i;
                            token_id_q <= producer_token_id_i;
                            model_id_q <= producer_model_id_i;
                            producer_tag_q <= producer_completion_tag_i;
                            producer_index_q <= '0;
                            state_q <= ST_PRODUCE;
                        end
                    end else if (consumer_start_accepted_w) begin
                        consumer_descriptor_error_q <= 1'b0;
                        if (!metadata_valid_i || !carry_valid_q ||
                            (consumer_layer_id_i != producer_layer_q + 5'd1) ||
                            (consumer_token_id_i != token_id_q) ||
                            (consumer_model_id_i != model_id_q)) begin
                            consumer_descriptor_error_q <= 1'b1;
                            consumer_done_valid_q <= 1'b1;
                        end else begin
                            state_q <= ST_CONSUME;
                        end
                    end
                end
                ST_PRODUCE: begin
                    if (producer_lane_valid_i && producer_lane_ready_o) begin
                        if ((producer_carry_s16_q15_i < -16'sd16384) ||
                            (producer_carry_s16_q15_i > 16'sd16384)) begin
                            carry_valid_q <= 1'b0;
                            producer_numeric_overflow_q <= 1'b1;
                            producer_done_valid_q <= 1'b1;
                            state_q <= ST_IDLE;
                        end else begin
                            carry_mem_q[producer_index_q] <= producer_carry_s16_q15_i;
                            if (producer_index_q == INDEX_WIDTH'(HIDDEN_SIZE - 1)) begin
                                carry_valid_q <= 1'b1;
                                producer_done_valid_q <= 1'b1;
                                state_q <= ST_IDLE;
                            end else begin
                                producer_index_q <= producer_index_q + INDEX_WIDTH'(1);
                            end
                        end
                    end
                end
                ST_CONSUME: begin
                    if (consumer_finish_valid_i) begin
                        carry_valid_q <= 1'b0;
                        consumer_descriptor_error_q <= !consumer_finish_success_i;
                        consumer_done_valid_q <= 1'b1;
                        state_q <= ST_IDLE;
                    end
                end
                default: state_q <= ST_IDLE;
            endcase
        end
    end
endmodule


// Carry-aware RMSNorm datapath. The carry state remains live while this core
// performs its square and scale passes and is cleared by the state core only
// after done_valid is accepted as a successful consumer result.
module ace2_carry_aware_rmsnorm_core #(
    parameter integer HIDDEN_SIZE = 896
) (
    input  logic                       clk_i,
    input  logic                       rst_ni,
    input  logic                       clear_i,
    input  logic                       start_valid_i,
    output logic                       start_ready_o,
    input  logic                       square_valid_i,
    output logic                       square_ready_o,
    input  logic signed [7:0]          square_hidden_s8_i,
    input  logic signed [15:0]         square_carry_s16_q15_i,
    input  logic                       scale_valid_i,
    output logic                       scale_ready_o,
    input  logic signed [7:0]          scale_hidden_s8_i,
    input  logic signed [15:0]         scale_carry_s16_q15_i,
    // Frozen metadata: round(weight / RMSNorm output_scale * 2^8).
    // This is an output-scale-folded gain, not a raw model gain in Q7.8.
    input  logic signed [15:0]         scale_scaled_gain_s16_q8_i,
    output logic                       out_valid_o,
    input  logic                       out_ready_i,
    output logic signed [7:0]          out_data_s8_o,
    output logic                       done_valid_o,
    input  logic                       done_ready_i,
    output logic                       done_numeric_overflow_o,
    output logic                       saturation_seen_o,
    output logic [55:0]                sum_squares_q30_o,
    output logic [55:0]                mean_square_q30_o,
    output logic [23:0]                root_q15_o,
    output logic [45:0]                inverse_q30_o
);
    localparam integer INDEX_WIDTH = (HIDDEN_SIZE <= 1) ? 1 : $clog2(HIDDEN_SIZE);
    localparam logic [3:0] ST_IDLE = 4'd0;
    localparam logic [3:0] ST_SQUARE = 4'd1;
    localparam logic [3:0] ST_MEAN_DIVIDE = 4'd2;
    localparam logic [3:0] ST_SQRT = 4'd3;
    localparam logic [3:0] ST_INV_DIVIDE = 4'd4;
    localparam logic [3:0] ST_SCALE = 4'd5;
    localparam logic [3:0] ST_SCALE_MULTIPLY = 4'd6;
    localparam logic [3:0] ST_DONE = 4'd7;
    localparam logic [55:0] HIDDEN_SIZE_U56 = 56'(HIDDEN_SIZE);

    logic [3:0] state_q;
    logic [INDEX_WIDTH-1:0] square_index_q, scale_index_q;
    logic [55:0] sum_squares_q, mean_square_q;
    logic [55:0] mean_dividend_q;
    // The top bit is a consumed long-division shift-register bit; it is
    // intentionally shifted out before the final quotient is observed.
    /* verilator lint_off UNUSED */
    logic [54:0] mean_quotient_q;
    /* verilator lint_on UNUSED */
    logic [9:0] mean_remainder_q;
    logic [5:0] mean_count_q;
    logic [23:0] sqrt_root_q;
    logic [4:0] sqrt_bit_q;
    logic [45:0] inverse_q;
    logic [45:0] inv_dividend_q;
    /* verilator lint_off UNUSED */
    logic [45:0] inv_quotient_q;
    logic [24:0] inv_remainder_q;
    /* verilator lint_on UNUSED */
    logic [5:0] inv_count_q;
    logic scale_product_negative_q;
    logic [85:0] scale_multiplicand_q, scale_accumulator_q;
    logic [45:0] scale_multiplier_q;
    logic [5:0] scale_count_q;
    logic out_valid_q, done_valid_q, numeric_overflow_q, saturation_seen_q;
    logic signed [7:0] out_data_q;

    logic square_carry_valid_w, scale_carry_valid_w;
    logic signed [23:0] square_x_q15_w, scale_x_q15_w;
    logic signed [47:0] square_product_w;
    logic [47:0] square_unsigned_w;
    logic [55:0] square_sum_next_w;
    logic [10:0] mean_remainder_shift_w;
    logic [9:0] mean_remainder_next_w;
    logic mean_subtract_w;
    logic [54:0] mean_quotient_next_w;
    logic [55:0] mean_dividend_next_w;
    logic [23:0] sqrt_trial_w, sqrt_root_next_w;
    logic [47:0] sqrt_trial_square_w, sqrt_root_square_w;
    logic [24:0] inv_remainder_shift_w, inv_remainder_next_w;
    logic inv_subtract_w;
    logic [45:0] inv_quotient_next_w, inv_dividend_next_w;
    logic signed [39:0] scale_product_w;
    logic [39:0] scale_product_magnitude_w;
    logic [85:0] scale_accumulator_next_w;
    logic [32:0] scale_round_base_w;
    logic [52:0] scale_round_remainder_w;
    logic scale_round_increment_w;
    logic [33:0] scale_rounded_magnitude_w;
    logic scale_positive_saturation_w, scale_negative_saturation_w;
    logic signed [7:0] scale_signed_result_w;

    assign start_ready_o = rst_ni && !clear_i && (state_q == ST_IDLE) && !done_valid_q;
    assign square_ready_o = (state_q == ST_SQUARE);
    assign scale_ready_o = (state_q == ST_SCALE) && !out_valid_q;
    assign out_valid_o = out_valid_q;
    assign out_data_s8_o = out_data_q;
    assign done_valid_o = done_valid_q;
    assign done_numeric_overflow_o = numeric_overflow_q;
    assign saturation_seen_o = saturation_seen_q;
    assign sum_squares_q30_o = sum_squares_q;
    assign mean_square_q30_o = mean_square_q;
    assign root_q15_o = sqrt_root_q;
    assign inverse_q30_o = inverse_q;

    always_comb begin
        square_carry_valid_w = (square_carry_s16_q15_i >= -16'sd16384) &&
                               (square_carry_s16_q15_i <= 16'sd16384);
        scale_carry_valid_w = (scale_carry_s16_q15_i >= -16'sd16384) &&
                              (scale_carry_s16_q15_i <= 16'sd16384);
        square_x_q15_w = ($signed({{16{square_hidden_s8_i[7]}}, square_hidden_s8_i}) <<< 15) +
                         $signed({{8{square_carry_s16_q15_i[15]}}, square_carry_s16_q15_i});
        scale_x_q15_w = ($signed({{16{scale_hidden_s8_i[7]}}, scale_hidden_s8_i}) <<< 15) +
                        $signed({{8{scale_carry_s16_q15_i[15]}}, scale_carry_s16_q15_i});
        square_product_w = square_x_q15_w * square_x_q15_w;
        square_unsigned_w = $unsigned(square_product_w);
        square_sum_next_w = sum_squares_q + {{8{1'b0}}, square_unsigned_w};

        mean_remainder_shift_w = {mean_remainder_q, mean_dividend_q[55]};
        mean_subtract_w = mean_remainder_shift_w >= {1'b0, HIDDEN_SIZE_U56[9:0]};
        mean_remainder_next_w = mean_subtract_w ?
                                mean_remainder_shift_w[9:0] - HIDDEN_SIZE_U56[9:0] :
                                mean_remainder_shift_w[9:0];
        mean_quotient_next_w = {mean_quotient_q[53:0], mean_subtract_w};
        mean_dividend_next_w = {mean_dividend_q[54:0], 1'b0};
        sqrt_trial_w = sqrt_root_q | (24'd1 << sqrt_bit_q);
        sqrt_trial_square_w = sqrt_trial_w * sqrt_trial_w;
        sqrt_root_next_w = ({{8{1'b0}}, sqrt_trial_square_w} <= mean_square_q) ?
                           sqrt_trial_w : sqrt_root_q;
        sqrt_root_square_w = sqrt_root_next_w * sqrt_root_next_w;

        inv_remainder_shift_w = {inv_remainder_q[23:0], inv_dividend_q[45]};
        inv_subtract_w = inv_remainder_shift_w >= {1'b0, sqrt_root_q};
        inv_remainder_next_w = inv_subtract_w ?
                               inv_remainder_shift_w - {1'b0, sqrt_root_q} :
                               inv_remainder_shift_w;
        inv_quotient_next_w = {inv_quotient_q[44:0], inv_subtract_w};
        inv_dividend_next_w = {inv_dividend_q[44:0], 1'b0};

        scale_product_w = scale_x_q15_w * scale_scaled_gain_s16_q8_i;
        scale_product_magnitude_w = scale_product_w[39] ? $unsigned(-scale_product_w) : $unsigned(scale_product_w);
        scale_accumulator_next_w = scale_accumulator_q +
            (scale_multiplier_q[0] ? scale_multiplicand_q : 86'd0);
        scale_round_base_w = scale_accumulator_next_w[85:53];
        scale_round_remainder_w = scale_accumulator_next_w[52:0];
        scale_round_increment_w = scale_round_remainder_w[52] &&
            ((|scale_round_remainder_w[51:0]) || scale_round_base_w[0]);
        scale_rounded_magnitude_w = {1'b0, scale_round_base_w} +
                                    {{33{1'b0}}, scale_round_increment_w};
        scale_positive_saturation_w = !scale_product_negative_q && (scale_rounded_magnitude_w > 34'd127);
        scale_negative_saturation_w = scale_product_negative_q && (scale_rounded_magnitude_w > 34'd128);
        if (scale_product_negative_q)
            scale_signed_result_w = -$signed(scale_rounded_magnitude_w[7:0]);
        else
            scale_signed_result_w = $signed(scale_rounded_magnitude_w[7:0]);
    end

    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            state_q <= ST_IDLE;
            square_index_q <= '0;
            scale_index_q <= '0;
            sum_squares_q <= '0;
            mean_square_q <= '0;
            mean_dividend_q <= '0;
            mean_quotient_q <= '0;
            mean_remainder_q <= '0;
            mean_count_q <= '0;
            sqrt_root_q <= '0;
            sqrt_bit_q <= '0;
            inverse_q <= '0;
            inv_dividend_q <= '0;
            inv_quotient_q <= '0;
            inv_remainder_q <= '0;
            inv_count_q <= '0;
            scale_product_negative_q <= 1'b0;
            scale_multiplicand_q <= '0;
            scale_accumulator_q <= '0;
            scale_multiplier_q <= '0;
            scale_count_q <= '0;
            out_valid_q <= 1'b0;
            done_valid_q <= 1'b0;
            numeric_overflow_q <= 1'b0;
            saturation_seen_q <= 1'b0;
            out_data_q <= '0;
        end else if (clear_i) begin
            state_q <= ST_IDLE;
            square_index_q <= '0;
            scale_index_q <= '0;
            sum_squares_q <= '0;
            mean_square_q <= '0;
            mean_dividend_q <= '0;
            mean_quotient_q <= '0;
            mean_remainder_q <= '0;
            mean_count_q <= '0;
            sqrt_root_q <= '0;
            sqrt_bit_q <= '0;
            inverse_q <= '0;
            inv_dividend_q <= '0;
            inv_quotient_q <= '0;
            inv_remainder_q <= '0;
            inv_count_q <= '0;
            scale_product_negative_q <= 1'b0;
            scale_multiplicand_q <= '0;
            scale_accumulator_q <= '0;
            scale_multiplier_q <= '0;
            scale_count_q <= '0;
            out_valid_q <= 1'b0;
            done_valid_q <= 1'b0;
            numeric_overflow_q <= 1'b0;
            saturation_seen_q <= 1'b0;
            out_data_q <= '0;
        end else begin
            if (done_valid_q && done_ready_i) begin
                done_valid_q <= 1'b0;
                state_q <= ST_IDLE;
            end
            if (out_valid_q && out_ready_i) begin
                out_valid_q <= 1'b0;
                if (scale_index_q == INDEX_WIDTH'(HIDDEN_SIZE - 1)) begin
                    done_valid_q <= 1'b1;
                    state_q <= ST_DONE;
                end else begin
                    scale_index_q <= scale_index_q + INDEX_WIDTH'(1);
                    state_q <= ST_SCALE;
                end
            end
            case (state_q)
                ST_IDLE: begin
                    if (start_valid_i && start_ready_o) begin
                        square_index_q <= '0;
                        scale_index_q <= '0;
                        sum_squares_q <= '0;
                        mean_square_q <= '0;
                        sqrt_root_q <= '0;
                        inverse_q <= '0;
                        numeric_overflow_q <= 1'b0;
                        saturation_seen_q <= 1'b0;
                        state_q <= ST_SQUARE;
                    end
                end
                ST_SQUARE: begin
                    if (square_valid_i && square_ready_o) begin
                        if (!square_carry_valid_w || square_product_w[47]) begin
                            numeric_overflow_q <= 1'b1;
                            done_valid_q <= 1'b1;
                            state_q <= ST_DONE;
                        end else begin
                            sum_squares_q <= square_sum_next_w;
                            if (square_index_q == INDEX_WIDTH'(HIDDEN_SIZE - 1)) begin
                                mean_dividend_q <= square_sum_next_w;
                                mean_quotient_q <= '0;
                                mean_remainder_q <= '0;
                                mean_count_q <= 6'd56;
                                state_q <= ST_MEAN_DIVIDE;
                            end else begin
                                square_index_q <= square_index_q + INDEX_WIDTH'(1);
                            end
                        end
                    end
                end
                ST_MEAN_DIVIDE: begin
                    mean_dividend_q <= mean_dividend_next_w;
                    mean_quotient_q <= mean_quotient_next_w;
                    mean_remainder_q <= mean_remainder_next_w;
                    if (mean_count_q == 6'd1) begin
                        mean_square_q <= {1'b0, mean_quotient_next_w} +
                            {{55{1'b0}},
                             (({mean_remainder_next_w, 1'b0} > {1'b0, HIDDEN_SIZE_U56[9:0]}) ||
                              (({mean_remainder_next_w, 1'b0} == {1'b0, HIDDEN_SIZE_U56[9:0]}) &&
                               mean_quotient_next_w[0]))};
                        sqrt_root_q <= '0;
                        sqrt_bit_q <= 5'd22;
                        state_q <= ST_SQRT;
                    end else begin
                        mean_count_q <= mean_count_q - 6'd1;
                    end
                end
                ST_SQRT: begin
                    sqrt_root_q <= sqrt_root_next_w;
                    if (sqrt_bit_q == 5'd0) begin
                        if (sqrt_root_next_w == 24'd0)
                            sqrt_root_q <= 24'd1;
                        else if ({{8{1'b0}}, sqrt_root_square_w} != mean_square_q)
                            sqrt_root_q <= sqrt_root_next_w + 24'd1;
                        inv_dividend_q <= 46'd1 << 45;
                        inv_quotient_q <= '0;
                        inv_remainder_q <= '0;
                        inv_count_q <= 6'd46;
                        state_q <= ST_INV_DIVIDE;
                    end else begin
                        sqrt_bit_q <= sqrt_bit_q - 5'd1;
                    end
                end
                ST_INV_DIVIDE: begin
                    inv_dividend_q <= inv_dividend_next_w;
                    inv_quotient_q <= inv_quotient_next_w;
                    inv_remainder_q <= inv_remainder_next_w;
                    if (inv_count_q == 6'd1) begin
                        inverse_q <= inv_quotient_next_w;
                        state_q <= ST_SCALE;
                    end else begin
                        inv_count_q <= inv_count_q - 6'd1;
                    end
                end
                ST_SCALE: begin
                    if (scale_valid_i && scale_ready_o) begin
                        if (!scale_carry_valid_w) begin
                            numeric_overflow_q <= 1'b1;
                            done_valid_q <= 1'b1;
                            state_q <= ST_DONE;
                        end else begin
                            scale_product_negative_q <= scale_product_w[39];
                            scale_multiplicand_q <= {{46{1'b0}}, scale_product_magnitude_w};
                            scale_multiplier_q <= inverse_q;
                            scale_accumulator_q <= '0;
                            scale_count_q <= 6'd46;
                            state_q <= ST_SCALE_MULTIPLY;
                        end
                    end
                end
                ST_SCALE_MULTIPLY: begin
                    scale_accumulator_q <= scale_accumulator_next_w;
                    scale_multiplicand_q <= scale_multiplicand_q << 1;
                    scale_multiplier_q <= scale_multiplier_q >> 1;
                    if (scale_count_q == 6'd1) begin
                        saturation_seen_q <= saturation_seen_q ||
                                             scale_positive_saturation_w || scale_negative_saturation_w;
                        if (scale_positive_saturation_w) begin
                            out_data_q <= 8'sd127;
                            numeric_overflow_q <= 1'b1;
                        end else if (scale_negative_saturation_w) begin
                            out_data_q <= -8'sd128;
                            numeric_overflow_q <= 1'b1;
                        end else begin
                            out_data_q <= scale_signed_result_w[7:0];
                        end
                        out_valid_q <= 1'b1;
                        state_q <= ST_SCALE;
                    end else begin
                        scale_count_q <= scale_count_q - 6'd1;
                    end
                end
                ST_DONE: begin
                    // Hold terminal status until done handshake.
                end
                default: state_q <= ST_IDLE;
            endcase
        end
    end
endmodule

/* verilator lint_on DECLFILENAME */
`default_nettype wire
