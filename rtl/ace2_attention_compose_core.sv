`default_nettype none

module ace2_attention_compose_core #(
    parameter integer TILE_MAX = 8,
    parameter integer HEAD_DIM = 64,
    parameter integer CONTEXT_MAX = 32768,
    parameter integer ACC_WIDTH = 40
) (
    input  wire                         clk_i,
    input  wire                         rst_ni,
    input  wire                         clear_i,

    input  wire [7:0]                   command_i,
    input  wire [15:0]                  tile_count_i,
    output wire                         command_allowed_o,
    input  wire                         start_valid_i,
    input  wire                         start_authorized_i,
    output wire                         start_ready_o,
    input  wire [TILE_MAX*16-1:0]       score_data_i,

    output wire                         value_ready_o,
    input  wire                         value_valid_i,
    input  wire [127:0]                 value_data_i,

    output wire                         out_valid_o,
    input  wire                         out_ready_i,
    output wire [127:0]                 out_data_o,
    output wire                         out_last_o,
    output wire                         command_done_o,
    output wire                         saturation_seen_o,
    output wire [15:0]                  context_count_o
);
    localparam [7:0] CMD_MAX_FIRST = 8'h00;
    localparam [7:0] CMD_MAX_MORE = 8'h01;
    localparam [7:0] CMD_SUM_FIRST = 8'h02;
    localparam [7:0] CMD_SUM_MORE = 8'h03;
    localparam [7:0] CMD_VALUE_FIRST = 8'h04;
    localparam [7:0] CMD_VALUE_MORE = 8'h05;
    localparam [7:0] CMD_VALUE_LAST = 8'h06;

    localparam [1:0] PROTOCOL_EMPTY = 2'd0;
    localparam [1:0] PROTOCOL_MAX = 2'd1;
    localparam [1:0] PROTOCOL_SUM = 2'd2;
    localparam [1:0] PROTOCOL_VALUE = 2'd3;

    localparam [3:0] ST_IDLE = 4'd0;
    localparam [3:0] ST_MAX = 4'd1;
    localparam [3:0] ST_SUM = 4'd2;
    localparam [3:0] ST_DIV_INIT = 4'd3;
    localparam [3:0] ST_DIV_PREP = 4'd4;
    localparam [3:0] ST_DIV_STEP = 4'd5;
    localparam [3:0] ST_DIV_ROUND = 4'd6;
    localparam [3:0] ST_VALUE_WAIT = 4'd7;
    localparam [3:0] ST_VALUE_ACCUM = 4'd8;
    localparam [3:0] ST_OUTPUT = 4'd9;
    localparam [3:0] ST_DONE = 4'd10;
    localparam [3:0] ST_FINALIZE = 4'd11;
    localparam [3:0] ST_SUM_ACCUM = 4'd12;
    localparam [3:0] ST_FINALIZE_ROUND = 4'd13;
    localparam [3:0] ST_FINALIZE_STORE = 4'd14;
    localparam [3:0] ST_ACCEPT = 4'd15;
    localparam integer FINALIZE_INDEX_WIDTH =
        (HEAD_DIM <= 1) ? 1 : $clog2(HEAD_DIM);
    localparam [FINALIZE_INDEX_WIDTH-1:0] FINALIZE_LAST_LANE =
        FINALIZE_INDEX_WIDTH'(HEAD_DIM - 1);
    localparam integer ACC_LOW_WIDTH = ACC_WIDTH / 2;
    localparam integer ACC_HIGH_WIDTH = ACC_WIDTH - ACC_LOW_WIDTH;

    reg [3:0] state_q;
    reg [1:0] protocol_q;
    reg [7:0] command_q;
    reg [15:0] tile_count_q;
    reg [127:0] score_data_q;
    reg [2:0] tile_index_q;
    reg signed [15:0] global_max_q;
    reg [30:0] global_sum_q;
    reg [15:0] sum_weight_q;
    reg [15:0] max_count_q;
    reg [15:0] sum_count_q;
    reg [15:0] value_count_q;
    reg [15:0] probability_q [0:TILE_MAX-1];
    reg [ACC_LOW_WIDTH-1:0] accumulator_low_q [0:HEAD_DIM-1];
    reg [ACC_HIGH_WIDTH-1:0] accumulator_high_q [0:HEAD_DIM-1];
    reg [127:0] value_data_q;
    reg [3:0] value_lane_q;
    reg [1:0] value_beat_q;
    reg [2:0] value_token_q;
    reg accumulator_write_q;
    reg accumulator_write_replace_q;
    reg [5:0] accumulator_write_index_q;
    reg signed [ACC_WIDTH-1:0] accumulator_write_data_q;
    reg accumulator_commit_q;
    reg [5:0] accumulator_commit_index_q;
    reg [ACC_LOW_WIDTH:0] accumulator_low_sum_q;
    reg [ACC_HIGH_WIDTH-1:0] accumulator_high_sum_q;
    reg [30:0] div_remaining_q;
    reg [45:0] div_trial_q;
    reg [15:0] div_quotient_q;
    reg [4:0] div_bit_q;
    reg [1:0] out_beat_q;
    reg [FINALIZE_INDEX_WIDTH-1:0] finalize_lane_q;
    reg signed [ACC_WIDTH-1:0] finalize_value_q;
    reg [7:0] finalize_rounded_q;
    reg finalize_saturated_q;
    reg [HEAD_DIM*8-1:0] rounded_output_q;
    reg saturation_seen_q;
    reg command_done_q;

    integer reset_lane;
    integer accumulator_reset_lane;

    wire tile_count_valid_w = (tile_count_i != 16'd0) &&
                              (tile_count_i <= 16'(TILE_MAX));
    wire [16:0] max_count_sum_w = {1'b0, max_count_q} + {1'b0, tile_count_i};
    wire [16:0] sum_count_sum_w = {1'b0, sum_count_q} + {1'b0, tile_count_i};
    wire [16:0] value_count_sum_w = {1'b0, value_count_q} + {1'b0, tile_count_i};
    wire max_more_allowed_w = (protocol_q == PROTOCOL_MAX) &&
                              (max_count_sum_w <= 17'(CONTEXT_MAX));
    wire sum_first_allowed_w = (protocol_q == PROTOCOL_MAX);
    wire sum_more_allowed_w = (protocol_q == PROTOCOL_SUM) &&
                              (sum_count_sum_w <= {1'b0, max_count_q});
    wire value_first_allowed_w = (protocol_q == PROTOCOL_SUM) &&
                                 (sum_count_q == max_count_q);
    wire value_more_allowed_w = (protocol_q == PROTOCOL_VALUE) &&
                                (value_count_sum_w < {1'b0, max_count_q});
    wire value_last_allowed_w = (protocol_q == PROTOCOL_VALUE) &&
                                (value_count_sum_w == {1'b0, max_count_q});

    assign command_allowed_o = tile_count_valid_w &&
        (((command_i == CMD_MAX_FIRST)) ||
         ((command_i == CMD_MAX_MORE) && max_more_allowed_w) ||
         ((command_i == CMD_SUM_FIRST) && sum_first_allowed_w) ||
         ((command_i == CMD_SUM_MORE) && sum_more_allowed_w) ||
         ((command_i == CMD_VALUE_FIRST) && value_first_allowed_w) ||
         ((command_i == CMD_VALUE_MORE) && value_more_allowed_w) ||
         ((command_i == CMD_VALUE_LAST) && value_last_allowed_w));
    assign start_ready_o = (state_q == ST_IDLE);
    assign value_ready_o = (state_q == ST_VALUE_WAIT);
    assign out_valid_o = (state_q == ST_OUTPUT);
    assign out_data_o = rounded_output_q[out_beat_q*128 +: 128];
    assign out_last_o = (out_beat_q == 2'd3);
    assign command_done_o = command_done_q;
    assign saturation_seen_o = saturation_seen_q;
    assign context_count_o = max_count_q;

    wire signed [15:0] current_score_w =
        score_data_q[tile_index_q*16 +: 16];
    wire signed [16:0] max_minus_score_signed_w =
        {global_max_q[15], global_max_q} -
        {current_score_w[15], current_score_w};
    wire [16:0] max_minus_score_w =
        max_minus_score_signed_w[16] ? 17'd0 :
        max_minus_score_signed_w[16:0];
    wire [15:0] current_exp_weight_w = exp_lookup_q15(max_minus_score_w);
    wire [45:0] div_remaining_ext_w = {15'd0, div_remaining_q};
    wire div_take_w = div_remaining_ext_w >= div_trial_q;
    wire [31:0] round_remainder_twice_w = {div_remaining_q, 1'b0};
    wire [31:0] round_denominator_w = {1'b0, global_sum_q};
    wire round_increment_w =
        (round_remainder_twice_w > round_denominator_w) ||
        ((round_remainder_twice_w == round_denominator_w) &&
         div_quotient_q[0]);
    wire [16:0] rounded_probability_w =
        {1'b0, div_quotient_q} + {16'd0, round_increment_w};
    wire signed [8:0] value_lane_signed_w =
        $signed({value_data_q[value_lane_q*8 + 7],
                 value_data_q[value_lane_q*8 +: 8]});
    wire signed [16:0] probability_signed_w =
        $signed({1'b0, probability_q[value_token_q]});
    wire signed [25:0] value_product_w =
        probability_signed_w * value_lane_signed_w;
    wire [5:0] accumulator_index_w =
        {value_beat_q, 4'd0} + {2'd0, value_lane_q};
    wire [ACC_LOW_WIDTH-1:0] accumulator_base_low_w =
        accumulator_write_replace_q ? {ACC_LOW_WIDTH{1'b0}} :
        accumulator_low_q[accumulator_write_index_q];
    wire [ACC_HIGH_WIDTH-1:0] accumulator_base_high_w =
        accumulator_write_replace_q ? {ACC_HIGH_WIDTH{1'b0}} :
        accumulator_high_q[accumulator_write_index_q];
    wire [ACC_LOW_WIDTH:0] accumulator_low_add_w =
        {1'b0, accumulator_base_low_w} +
        {1'b0, accumulator_write_data_q[ACC_LOW_WIDTH-1:0]};
    wire [ACC_HIGH_WIDTH-1:0] accumulator_high_add_w =
        accumulator_base_high_w +
        accumulator_write_data_q[ACC_WIDTH-1:ACC_LOW_WIDTH];

    function automatic [15:0] exp_lookup_q15;
        input [16:0] magnitude_q6_9;
        reg [16:0] table_index;
        begin
            table_index = (magnitude_q6_9 + 17'd32) >> 6;
            if (table_index > 17'd64) begin
                exp_lookup_q15 = 16'd0;
            end else begin
                case (table_index[6:0])
                    7'd0: exp_lookup_q15 = 16'd32768;
                    7'd1: exp_lookup_q15 = 16'd28918;
                    7'd2: exp_lookup_q15 = 16'd25520;
                    7'd3: exp_lookup_q15 = 16'd22521;
                    7'd4: exp_lookup_q15 = 16'd19875;
                    7'd5: exp_lookup_q15 = 16'd17539;
                    7'd6: exp_lookup_q15 = 16'd15479;
                    7'd7: exp_lookup_q15 = 16'd13660;
                    7'd8: exp_lookup_q15 = 16'd12055;
                    7'd9: exp_lookup_q15 = 16'd10638;
                    7'd10: exp_lookup_q15 = 16'd9388;
                    7'd11: exp_lookup_q15 = 16'd8285;
                    7'd12: exp_lookup_q15 = 16'd7312;
                    7'd13: exp_lookup_q15 = 16'd6452;
                    7'd14: exp_lookup_q15 = 16'd5694;
                    7'd15: exp_lookup_q15 = 16'd5025;
                    7'd16: exp_lookup_q15 = 16'd4435;
                    7'd17: exp_lookup_q15 = 16'd3914;
                    7'd18: exp_lookup_q15 = 16'd3454;
                    7'd19: exp_lookup_q15 = 16'd3048;
                    7'd20: exp_lookup_q15 = 16'd2690;
                    7'd21: exp_lookup_q15 = 16'd2374;
                    7'd22: exp_lookup_q15 = 16'd2095;
                    7'd23: exp_lookup_q15 = 16'd1849;
                    7'd24: exp_lookup_q15 = 16'd1631;
                    7'd25: exp_lookup_q15 = 16'd1440;
                    7'd26: exp_lookup_q15 = 16'd1271;
                    7'd27: exp_lookup_q15 = 16'd1121;
                    7'd28: exp_lookup_q15 = 16'd990;
                    7'd29: exp_lookup_q15 = 16'd873;
                    7'd30: exp_lookup_q15 = 16'd771;
                    7'd31: exp_lookup_q15 = 16'd680;
                    7'd32: exp_lookup_q15 = 16'd600;
                    7'd33: exp_lookup_q15 = 16'd530;
                    7'd34: exp_lookup_q15 = 16'd467;
                    7'd35: exp_lookup_q15 = 16'd412;
                    7'd36: exp_lookup_q15 = 16'd364;
                    7'd37: exp_lookup_q15 = 16'd321;
                    7'd38: exp_lookup_q15 = 16'd283;
                    7'd39: exp_lookup_q15 = 16'd250;
                    7'd40: exp_lookup_q15 = 16'd221;
                    7'd41: exp_lookup_q15 = 16'd195;
                    7'd42: exp_lookup_q15 = 16'd172;
                    7'd43: exp_lookup_q15 = 16'd152;
                    7'd44: exp_lookup_q15 = 16'd134;
                    7'd45: exp_lookup_q15 = 16'd118;
                    7'd46: exp_lookup_q15 = 16'd104;
                    7'd47: exp_lookup_q15 = 16'd92;
                    7'd48: exp_lookup_q15 = 16'd81;
                    7'd49: exp_lookup_q15 = 16'd72;
                    7'd50: exp_lookup_q15 = 16'd63;
                    7'd51: exp_lookup_q15 = 16'd56;
                    7'd52: exp_lookup_q15 = 16'd49;
                    7'd53: exp_lookup_q15 = 16'd43;
                    7'd54: exp_lookup_q15 = 16'd38;
                    7'd55: exp_lookup_q15 = 16'd34;
                    7'd56: exp_lookup_q15 = 16'd30;
                    7'd57: exp_lookup_q15 = 16'd26;
                    7'd58: exp_lookup_q15 = 16'd23;
                    7'd59: exp_lookup_q15 = 16'd21;
                    7'd60: exp_lookup_q15 = 16'd18;
                    7'd61: exp_lookup_q15 = 16'd16;
                    7'd62: exp_lookup_q15 = 16'd14;
                    7'd63: exp_lookup_q15 = 16'd12;
                    default: exp_lookup_q15 = 16'd11;
                endcase
            end
        end
    endfunction

    function automatic [7:0] round_saturate_q15;
        input signed [ACC_WIDTH-1:0] value;
        reg negative;
        reg [ACC_WIDTH-1:0] magnitude;
        reg [ACC_WIDTH-16:0] base;
        reg [14:0] remainder;
        reg increment;
        reg [ACC_WIDTH-15:0] rounded;
        begin
            negative = value[ACC_WIDTH-1];
            magnitude = negative ? (~value + ACC_WIDTH'(1)) : value;
            base = magnitude[ACC_WIDTH-1:15];
            remainder = magnitude[14:0];
            increment = (remainder > 15'd16384) ||
                        ((remainder == 15'd16384) && base[0]);
            rounded = {1'b0, base} + {{(ACC_WIDTH-15){1'b0}}, increment};
            if (!negative && (rounded > 127)) begin
                round_saturate_q15 = 8'h7f;
            end else if (negative && (rounded > 128)) begin
                round_saturate_q15 = 8'h80;
            end else if (negative) begin
                round_saturate_q15 = ~rounded[7:0] + 8'd1;
            end else begin
                round_saturate_q15 = rounded[7:0];
            end
        end
    endfunction

    function automatic saturated_q15;
        input signed [ACC_WIDTH-1:0] value;
        reg negative;
        reg [ACC_WIDTH-1:0] magnitude;
        reg [ACC_WIDTH-16:0] base;
        reg [14:0] remainder;
        reg increment;
        reg [ACC_WIDTH-15:0] rounded;
        begin
            negative = value[ACC_WIDTH-1];
            magnitude = negative ? (~value + ACC_WIDTH'(1)) : value;
            base = magnitude[ACC_WIDTH-1:15];
            remainder = magnitude[14:0];
            increment = (remainder > 15'd16384) ||
                        ((remainder == 15'd16384) && base[0]);
            rounded = {1'b0, base} + {{(ACC_WIDTH-15){1'b0}}, increment};
            saturated_q15 = (!negative && (rounded > 127)) ||
                            (negative && (rounded > 128));
        end
    endfunction

    always @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            state_q <= ST_IDLE;
            protocol_q <= PROTOCOL_EMPTY;
            command_q <= 8'd0;
            tile_count_q <= 16'd0;
            score_data_q <= 128'd0;
            tile_index_q <= 3'd0;
            global_max_q <= -16'sd32768;
            global_sum_q <= 31'd0;
            sum_weight_q <= 16'd0;
            max_count_q <= 16'd0;
            sum_count_q <= 16'd0;
            value_count_q <= 16'd0;
            value_data_q <= 128'd0;
            value_lane_q <= 4'd0;
            value_beat_q <= 2'd0;
            value_token_q <= 3'd0;
            accumulator_write_q <= 1'b0;
            accumulator_write_replace_q <= 1'b0;
            accumulator_write_index_q <= 6'd0;
            accumulator_write_data_q <= {ACC_WIDTH{1'b0}};
            div_remaining_q <= 31'd0;
            div_trial_q <= 46'd0;
            div_quotient_q <= 16'd0;
            div_bit_q <= 5'd0;
            out_beat_q <= 2'd0;
            finalize_lane_q <= {FINALIZE_INDEX_WIDTH{1'b0}};
            finalize_value_q <= {ACC_WIDTH{1'b0}};
            finalize_rounded_q <= 8'd0;
            finalize_saturated_q <= 1'b0;
            rounded_output_q <= {HEAD_DIM*8{1'b0}};
            saturation_seen_q <= 1'b0;
            command_done_q <= 1'b0;
            for (reset_lane = 0; reset_lane < TILE_MAX;
                 reset_lane = reset_lane + 1) begin
                probability_q[reset_lane] <= 16'd0;
            end
        end else begin
            command_done_q <= 1'b0;
            accumulator_write_q <= 1'b0;
            accumulator_write_replace_q <= 1'b0;
            case (state_q)
                ST_IDLE: begin
                    if (start_valid_i && start_ready_o && start_authorized_i) begin
                        command_q <= command_i;
                        tile_count_q <= tile_count_i;
                        score_data_q <= score_data_i;
                        tile_index_q <= 3'd0;
                        state_q <= ST_ACCEPT;
                    end
                end

                ST_ACCEPT: begin
                    if (command_q == CMD_MAX_FIRST) begin
                        protocol_q <= PROTOCOL_MAX;
                        global_max_q <= $signed(score_data_q[15:0]);
                        max_count_q <= tile_count_q;
                        sum_count_q <= 16'd0;
                        value_count_q <= 16'd0;
                        state_q <= ST_MAX;
                    end else if (command_q == CMD_MAX_MORE) begin
                        max_count_q <= max_count_q + tile_count_q;
                        state_q <= ST_MAX;
                    end else if ((command_q == CMD_SUM_FIRST) ||
                                 (command_q == CMD_SUM_MORE)) begin
                        protocol_q <= PROTOCOL_SUM;
                        if (command_q == CMD_SUM_FIRST) begin
                            global_sum_q <= 31'd0;
                            sum_count_q <= tile_count_q;
                        end else begin
                            sum_count_q <= sum_count_q + tile_count_q;
                        end
                        state_q <= ST_SUM;
                    end else begin
                        protocol_q <= PROTOCOL_VALUE;
                        if (command_q == CMD_VALUE_FIRST) begin
                            value_count_q <= tile_count_q;
                            saturation_seen_q <= 1'b0;
                        end else begin
                            value_count_q <= value_count_q + tile_count_q;
                        end
                        state_q <= ST_DIV_INIT;
                    end
                end

                ST_MAX: begin
                    if (current_score_w > global_max_q) begin
                        global_max_q <= current_score_w;
                    end
                    if ({13'd0, tile_index_q} + 16'd1 == tile_count_q) begin
                        state_q <= ST_DONE;
                    end else begin
                        tile_index_q <= tile_index_q + 3'd1;
                    end
                end

                ST_SUM: begin
                    sum_weight_q <= current_exp_weight_w;
                    state_q <= ST_SUM_ACCUM;
                end

                ST_SUM_ACCUM: begin
                    global_sum_q <= global_sum_q + {15'd0, sum_weight_q};
                    if ({13'd0, tile_index_q} + 16'd1 == tile_count_q) begin
                        state_q <= ST_DONE;
                    end else begin
                        tile_index_q <= tile_index_q + 3'd1;
                        state_q <= ST_SUM;
                    end
                end

                ST_DIV_INIT: begin
                    div_remaining_q <= {current_exp_weight_w, 15'd0};
                    div_quotient_q <= 16'd0;
                    div_bit_q <= 5'd15;
                    state_q <= ST_DIV_PREP;
                end

                ST_DIV_PREP: begin
                    div_trial_q <= {15'd0, global_sum_q} << div_bit_q;
                    state_q <= ST_DIV_STEP;
                end

                ST_DIV_STEP: begin
                    if (div_take_w) begin
                        div_remaining_q <=
                            div_remaining_q - div_trial_q[30:0];
                        div_quotient_q[div_bit_q[3:0]] <= 1'b1;
                    end
                    if (div_bit_q == 5'd0) begin
                        state_q <= ST_DIV_ROUND;
                    end else begin
                        div_bit_q <= div_bit_q - 5'd1;
                        state_q <= ST_DIV_PREP;
                    end
                end

                ST_DIV_ROUND: begin
                    probability_q[tile_index_q] <= rounded_probability_w[16] ?
                        16'hffff : rounded_probability_w[15:0];
                    if ({13'd0, tile_index_q} + 16'd1 == tile_count_q) begin
                        value_token_q <= 3'd0;
                        value_beat_q <= 2'd0;
                        value_lane_q <= 4'd0;
                        state_q <= ST_VALUE_WAIT;
                    end else begin
                        tile_index_q <= tile_index_q + 3'd1;
                        state_q <= ST_DIV_INIT;
                    end
                end

                ST_VALUE_WAIT: begin
                    if (value_valid_i && value_ready_o) begin
                        value_data_q <= value_data_i;
                        value_lane_q <= 4'd0;
                        state_q <= ST_VALUE_ACCUM;
                    end
                end

                ST_VALUE_ACCUM: begin
                    accumulator_write_q <= 1'b1;
                    accumulator_write_replace_q <=
                        (command_q == CMD_VALUE_FIRST) &&
                        (value_token_q == 3'd0);
                    accumulator_write_index_q <= accumulator_index_w;
                    accumulator_write_data_q <=
                        {{(ACC_WIDTH-26){value_product_w[25]}},
                         value_product_w};
                    if (value_lane_q == 4'd15) begin
                        value_lane_q <= 4'd0;
                        if (value_beat_q == 2'd3) begin
                            value_beat_q <= 2'd0;
                            if ({13'd0, value_token_q} + 16'd1 ==
                                tile_count_q) begin
                                value_token_q <= 3'd0;
                                if (command_q == CMD_VALUE_LAST) begin
                                    out_beat_q <= 2'd0;
                                    finalize_lane_q <=
                                        {FINALIZE_INDEX_WIDTH{1'b0}};
                                    saturation_seen_q <= 1'b0;
                                    state_q <= ST_FINALIZE;
                                end else begin
                                    state_q <= ST_DONE;
                                end
                            end else begin
                                value_token_q <= value_token_q + 3'd1;
                                state_q <= ST_VALUE_WAIT;
                            end
                        end else begin
                            value_beat_q <= value_beat_q + 2'd1;
                            state_q <= ST_VALUE_WAIT;
                        end
                    end else begin
                        value_lane_q <= value_lane_q + 4'd1;
                    end
                end

                ST_FINALIZE: begin
                    finalize_value_q <=
                        $signed({accumulator_high_q[finalize_lane_q],
                                 accumulator_low_q[finalize_lane_q]});
                    state_q <= ST_FINALIZE_ROUND;
                end

                ST_FINALIZE_ROUND: begin
                    finalize_rounded_q <= round_saturate_q15(finalize_value_q);
                    finalize_saturated_q <= saturated_q15(finalize_value_q);
                    state_q <= ST_FINALIZE_STORE;
                end

                ST_FINALIZE_STORE: begin
                    rounded_output_q[finalize_lane_q*8 +: 8] <=
                        finalize_rounded_q;
                    saturation_seen_q <=
                        saturation_seen_q | finalize_saturated_q;
                    if (finalize_lane_q == FINALIZE_LAST_LANE) begin
                        out_beat_q <= 2'd0;
                        state_q <= ST_OUTPUT;
                    end else begin
                        finalize_lane_q <= finalize_lane_q + 1'b1;
                        state_q <= ST_FINALIZE;
                    end
                end

                ST_OUTPUT: begin
                    if (out_valid_o && out_ready_i) begin
                        if (out_beat_q == 2'd3) begin
                            state_q <= ST_DONE;
                        end else begin
                            out_beat_q <= out_beat_q + 2'd1;
                        end
                    end
                end

                ST_DONE: begin
                    command_done_q <= 1'b1;
                    state_q <= ST_IDLE;
                end

                default: state_q <= ST_IDLE;
            endcase
            if (clear_i) begin
                state_q <= ST_IDLE;
                protocol_q <= PROTOCOL_EMPTY;
                global_max_q <= -16'sd32768;
                global_sum_q <= 31'd0;
                sum_weight_q <= 16'd0;
                max_count_q <= 16'd0;
                sum_count_q <= 16'd0;
                value_count_q <= 16'd0;
                command_done_q <= 1'b0;
                saturation_seen_q <= 1'b0;
                accumulator_write_q <= 1'b0;
                accumulator_write_replace_q <= 1'b0;
            end
        end
    end

    always @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            for (accumulator_reset_lane = 0;
                 accumulator_reset_lane < HEAD_DIM;
                 accumulator_reset_lane = accumulator_reset_lane + 1) begin
                accumulator_low_q[accumulator_reset_lane] <=
                    {ACC_LOW_WIDTH{1'b0}};
                accumulator_high_q[accumulator_reset_lane] <=
                    {ACC_HIGH_WIDTH{1'b0}};
            end
            accumulator_commit_q <= 1'b0;
            accumulator_commit_index_q <= 6'd0;
            accumulator_low_sum_q <= {(ACC_LOW_WIDTH+1){1'b0}};
            accumulator_high_sum_q <= {ACC_HIGH_WIDTH{1'b0}};
        end else begin
            accumulator_commit_q <= accumulator_write_q;
            if (accumulator_write_q) begin
                accumulator_commit_index_q <= accumulator_write_index_q;
                accumulator_low_sum_q <= accumulator_low_add_w;
                accumulator_high_sum_q <= accumulator_high_add_w;
            end
            if (accumulator_commit_q) begin
                accumulator_low_q[accumulator_commit_index_q] <=
                    accumulator_low_sum_q[ACC_LOW_WIDTH-1:0];
                accumulator_high_q[accumulator_commit_index_q] <=
                    accumulator_high_sum_q +
                    {{(ACC_HIGH_WIDTH-1){1'b0}},
                     accumulator_low_sum_q[ACC_LOW_WIDTH]};
            end
        end
    end
endmodule

`default_nettype wire
