`default_nettype none

module ace2_softmax_core #(
    parameter integer CONTEXT_MAX = 8,
    parameter integer SCORE_WIDTH = 16,
    parameter integer PROB_WIDTH = 16
) (
    input  wire                                  clk_i,
    input  wire                                  rst_ni,
    input  wire                                  clear_i,

    input  wire                                  start_valid_i,
    output wire                                  start_ready_o,
    input  wire [15:0]                           context_count_i,
    input  wire [CONTEXT_MAX*SCORE_WIDTH-1:0]    score_data_i,

    output wire                                  out_valid_o,
    input  wire                                  out_ready_i,
    output wire [CONTEXT_MAX*PROB_WIDTH-1:0]     prob_data_o,
    output wire                                  saturation_seen_o
);
    localparam integer INDEX_WIDTH = (CONTEXT_MAX <= 1) ? 1 : $clog2(CONTEXT_MAX);
    localparam [INDEX_WIDTH-1:0] LAST_INDEX = INDEX_WIDTH'(CONTEXT_MAX - 1);

    localparam [3:0] ST_IDLE      = 4'd0;
    localparam [3:0] ST_MAX       = 4'd1;
    localparam [3:0] ST_EXP_LOOK  = 4'd2;
    localparam [3:0] ST_EXP_ACCUM  = 4'd3;
    localparam [3:0] ST_DIV_INIT   = 4'd4;
    localparam [3:0] ST_DIV_PREP   = 4'd5;
    localparam [3:0] ST_DIV_CMP    = 4'd6;
    localparam [3:0] ST_DIV_STEP   = 4'd7;
    localparam [3:0] ST_ROUND      = 4'd8;
    localparam [3:0] ST_DONE       = 4'd9;

    reg [3:0] state_q;
    reg [CONTEXT_MAX*SCORE_WIDTH-1:0] score_data_q;
    reg [CONTEXT_MAX*PROB_WIDTH-1:0] exp_weight_q;
    reg [CONTEXT_MAX*PROB_WIDTH-1:0] prob_data_q;
    reg [15:0] context_count_q;
    reg [INDEX_WIDTH-1:0] index_q;
    reg signed [SCORE_WIDTH-1:0] max_score_q;
    reg [18:0] exp_sum_q;
    reg [31:0] div_remaining_q;
    reg [15:0] div_quotient_q;
    reg [33:0] div_trial_q;
    reg [4:0] div_bit_q;
    reg div_take_q;
    reg [INDEX_WIDTH-1:0] exp_index_q;
    reg [15:0] exp_lane_weight_q;
    reg out_valid_q;
    integer store_lane;

    wire index_active_w = ({{(16-INDEX_WIDTH){1'b0}}, index_q} < context_count_q);
    wire signed [SCORE_WIDTH-1:0] current_score_w = score_data_q[index_q*SCORE_WIDTH +: SCORE_WIDTH];
    wire signed [16:0] max_minus_score_signed_w = {max_score_q[SCORE_WIDTH-1], max_score_q} -
                                                  {current_score_w[SCORE_WIDTH-1], current_score_w};
    wire [16:0] max_minus_score_w = max_minus_score_signed_w[16] ? 17'd0 : max_minus_score_signed_w[16:0];
    wire [15:0] current_exp_weight_w = exp_weight_q[index_q*PROB_WIDTH +: PROB_WIDTH];
    wire [15:0] exp_lookup_w = index_active_w ? exp_lookup_q15(max_minus_score_w) : 16'd0;
    wire div_take_trial_w = ({2'd0, div_remaining_q} >= div_trial_q);
    wire [32:0] round_remainder_doubled_w = {div_remaining_q, 1'b0};
    wire [32:0] round_denominator_w = {14'd0, exp_sum_q};
    wire round_increment_w = (round_remainder_doubled_w > round_denominator_w) ||
                             ((round_remainder_doubled_w == round_denominator_w) && div_quotient_q[0]);
    wire [16:0] rounded_prob_w = {1'b0, div_quotient_q} + {16'd0, round_increment_w};

    assign start_ready_o = (state_q == ST_IDLE) && !out_valid_q;
    assign out_valid_o = out_valid_q;
    assign prob_data_o = prob_data_q;
    assign saturation_seen_o = 1'b0;

    function automatic [15:0] exp_lookup_q15;
        input [16:0] magnitude_q6_9;
        reg [16:0] table_index;
        begin
            table_index = (magnitude_q6_9 + 17'd32) >> 6;
            if (table_index > 17'd64) begin
                exp_lookup_q15 = 16'd0;
            end else begin
                case (table_index[6:0])
                    7'd0:  exp_lookup_q15 = 16'd32768;
                    7'd1:  exp_lookup_q15 = 16'd28918;
                    7'd2:  exp_lookup_q15 = 16'd25520;
                    7'd3:  exp_lookup_q15 = 16'd22521;
                    7'd4:  exp_lookup_q15 = 16'd19875;
                    7'd5:  exp_lookup_q15 = 16'd17539;
                    7'd6:  exp_lookup_q15 = 16'd15479;
                    7'd7:  exp_lookup_q15 = 16'd13660;
                    7'd8:  exp_lookup_q15 = 16'd12055;
                    7'd9:  exp_lookup_q15 = 16'd10638;
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

    always @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            state_q <= ST_IDLE;
            score_data_q <= {CONTEXT_MAX*SCORE_WIDTH{1'b0}};
            exp_weight_q <= {CONTEXT_MAX*PROB_WIDTH{1'b0}};
            prob_data_q <= {CONTEXT_MAX*PROB_WIDTH{1'b0}};
            context_count_q <= 16'd0;
            index_q <= {INDEX_WIDTH{1'b0}};
            max_score_q <= {SCORE_WIDTH{1'b0}};
            exp_sum_q <= 19'd0;
            div_remaining_q <= 32'd0;
            div_quotient_q <= 16'd0;
            div_trial_q <= 34'd0;
            div_bit_q <= 5'd0;
            div_take_q <= 1'b0;
            exp_index_q <= {INDEX_WIDTH{1'b0}};
            exp_lane_weight_q <= 16'd0;
            out_valid_q <= 1'b0;
        end else if (clear_i) begin
            state_q <= ST_IDLE;
            index_q <= {INDEX_WIDTH{1'b0}};
            exp_sum_q <= 19'd0;
            exp_index_q <= {INDEX_WIDTH{1'b0}};
            exp_lane_weight_q <= 16'd0;
            div_trial_q <= 34'd0;
            div_take_q <= 1'b0;
            out_valid_q <= 1'b0;
        end else begin
            if (out_valid_q && out_ready_i) begin
                out_valid_q <= 1'b0;
                state_q <= ST_IDLE;
            end else begin
                case (state_q)
                    ST_IDLE: begin
                        if (start_valid_i && start_ready_o) begin
                            score_data_q <= score_data_i;
                            exp_weight_q <= {CONTEXT_MAX*PROB_WIDTH{1'b0}};
                            prob_data_q <= {CONTEXT_MAX*PROB_WIDTH{1'b0}};
                            context_count_q <= context_count_i;
                            index_q <= {INDEX_WIDTH{1'b0}};
                            exp_index_q <= {INDEX_WIDTH{1'b0}};
                            exp_lane_weight_q <= 16'd0;
                            max_score_q <= score_data_i[SCORE_WIDTH-1:0];
                            exp_sum_q <= 19'd0;
                            state_q <= ST_MAX;
                        end
                    end

                    ST_MAX: begin
                        if (index_active_w && (current_score_w > max_score_q)) begin
                            max_score_q <= current_score_w;
                        end
                        if (index_q == LAST_INDEX) begin
                            index_q <= {INDEX_WIDTH{1'b0}};
                            state_q <= ST_EXP_LOOK;
                        end else begin
                            index_q <= index_q + {{(INDEX_WIDTH-1){1'b0}}, 1'b1};
                        end
                    end

                    ST_EXP_LOOK: begin
                        exp_index_q <= index_q;
                        exp_lane_weight_q <= exp_lookup_w;
                        state_q <= ST_EXP_ACCUM;
                    end

                    ST_EXP_ACCUM: begin
                        exp_weight_q[exp_index_q*PROB_WIDTH +: PROB_WIDTH] <= exp_lane_weight_q;
                        exp_sum_q <= exp_sum_q + {3'd0, exp_lane_weight_q};
                        if (index_q == LAST_INDEX) begin
                            index_q <= {INDEX_WIDTH{1'b0}};
                            state_q <= ST_DIV_INIT;
                        end else begin
                            index_q <= index_q + {{(INDEX_WIDTH-1){1'b0}}, 1'b1};
                            state_q <= ST_EXP_LOOK;
                        end
                    end

                    ST_DIV_INIT: begin
                        if (!index_active_w || (current_exp_weight_w == 16'd0) || (exp_sum_q == 19'd0)) begin
                            for (store_lane = 0; store_lane < CONTEXT_MAX;
                                 store_lane = store_lane + 1) begin
                                if (index_q == INDEX_WIDTH'(store_lane)) begin
                                    prob_data_q[store_lane*PROB_WIDTH +: PROB_WIDTH] <=
                                        {PROB_WIDTH{1'b0}};
                                end
                            end
                            if (index_q == LAST_INDEX) begin
                                out_valid_q <= 1'b1;
                                state_q <= ST_DONE;
                            end else begin
                                index_q <= index_q + {{(INDEX_WIDTH-1){1'b0}}, 1'b1};
                            end
                        end else begin
                            div_remaining_q <= {1'b0, current_exp_weight_w, 15'd0};
                            div_quotient_q <= 16'd0;
                            div_bit_q <= 5'd15;
                            div_trial_q <= 34'd0;
                            div_take_q <= 1'b0;
                            state_q <= ST_DIV_PREP;
                        end
                    end

                    ST_DIV_PREP: begin
                        div_trial_q <= {15'd0, exp_sum_q} << div_bit_q;
                        state_q <= ST_DIV_CMP;
                    end

                    ST_DIV_CMP: begin
                        div_take_q <= div_take_trial_w;
                        state_q <= ST_DIV_STEP;
                    end

                    ST_DIV_STEP: begin
                        if (div_take_q) begin
                            div_remaining_q <= div_remaining_q - div_trial_q[31:0];
                            div_quotient_q[div_bit_q[3:0]] <= 1'b1;
                        end
                        if (div_bit_q == 5'd0) begin
                            state_q <= ST_ROUND;
                        end else begin
                            div_bit_q <= div_bit_q - 5'd1;
                            state_q <= ST_DIV_PREP;
                        end
                    end

                    ST_ROUND: begin
                        for (store_lane = 0; store_lane < CONTEXT_MAX;
                             store_lane = store_lane + 1) begin
                            if (index_q == INDEX_WIDTH'(store_lane)) begin
                                prob_data_q[store_lane*PROB_WIDTH +: PROB_WIDTH] <=
                                    rounded_prob_w[16] ?
                                    {PROB_WIDTH{1'b1}} : rounded_prob_w[PROB_WIDTH-1:0];
                            end
                        end
                        if (index_q == LAST_INDEX) begin
                            out_valid_q <= 1'b1;
                            state_q <= ST_DONE;
                        end else begin
                            index_q <= index_q + {{(INDEX_WIDTH-1){1'b0}}, 1'b1};
                            state_q <= ST_DIV_INIT;
                        end
                    end

                    ST_DONE: begin
                    end

                    default: begin
                        state_q <= ST_IDLE;
                    end
                endcase
            end
        end
    end
endmodule

`default_nettype wire
