`default_nettype none

module ace2_attention_score_core #(
    parameter integer HEAD_DIM = 64,
    parameter integer MAC_LANES = 1,
    parameter integer ACT_WIDTH = 8,
    parameter integer ACC_WIDTH = 32,
    parameter integer SCORE_WIDTH = 16
) (
    input  wire                                  clk_i,
    input  wire                                  rst_ni,
    input  wire                                  clear_i,

    input  wire                                  start_valid_i,
    output wire                                  start_ready_o,

    input  wire                                  pair_valid_i,
    output wire                                  pair_ready_o,
    input  wire [MAC_LANES*ACT_WIDTH-1:0]       q_data_i,
    input  wire [MAC_LANES*ACT_WIDTH-1:0]       k_data_i,
    input  wire signed [31:0]                    multiplier_i,
    input  wire [5:0]                            right_shift_i,

    output wire                                  out_valid_o,
    input  wire                                  out_ready_i,
    output wire [SCORE_WIDTH-1:0]               score_o,
    output wire signed [ACC_WIDTH-1:0]          acc_o,
    output wire                                  saturation_seen_o
);
    localparam integer GROUPS = HEAD_DIM / MAC_LANES;
    localparam integer GROUP_INDEX_WIDTH = (GROUPS <= 1) ? 1 : $clog2(GROUPS + 1);
    localparam [GROUP_INDEX_WIDTH-1:0] LAST_GROUP = GROUP_INDEX_WIDTH'(GROUPS - 1);

    localparam [2:0] ST_IDLE  = 3'd0;
    localparam [2:0] ST_RUN   = 3'd1;
    localparam [2:0] ST_SCALE = 3'd2;
    localparam [2:0] ST_ROUND = 3'd3;
    localparam [2:0] ST_DONE  = 3'd4;

    reg [2:0] state_q;
    reg [GROUP_INDEX_WIDTH-1:0] group_idx_q;
    reg signed [ACC_WIDTH-1:0] acc_q;
    reg signed [63:0] scaled_product_q;
    reg [SCORE_WIDTH-1:0] score_q;
    reg out_valid_q;
    reg saturation_seen_q;

    reg signed [ACC_WIDTH-1:0] partial_sum_w;
    reg signed [ACT_WIDTH-1:0] q_lane_w;
    reg signed [ACT_WIDTH-1:0] k_lane_w;
    integer mac_lane;

    wire signed [ACC_WIDTH-1:0] acc_next_w = acc_q + partial_sum_w;
    wire scaled_negative_w = scaled_product_q[63];
    wire [63:0] scaled_abs_w =
        scaled_negative_w ? (~scaled_product_q + 64'd1) : scaled_product_q;
    wire [63:0] rounded_base_w = scaled_abs_w >> right_shift_i;
    wire [63:0] round_mask_w =
        (right_shift_i == 6'd0) ? 64'd0 : ((64'd1 << right_shift_i) - 64'd1);
    wire [63:0] rounded_remainder_w = scaled_abs_w & round_mask_w;
    wire [63:0] round_half_w =
        (right_shift_i == 6'd0) ? 64'd0 : (64'd1 << (right_shift_i - 6'd1));
    wire round_increment_w =
        (right_shift_i != 6'd0) &&
        ((rounded_remainder_w > round_half_w) ||
         ((rounded_remainder_w == round_half_w) && rounded_base_w[0]));
    wire [64:0] rounded_abs_w =
        {1'b0, rounded_base_w} + {{64{1'b0}}, round_increment_w};
    wire positive_saturate_w =
        !scaled_negative_w && (rounded_abs_w > 65'd32767);
    wire negative_saturate_w =
        scaled_negative_w && (rounded_abs_w > 65'd32768);
    wire output_saturate_w = positive_saturate_w || negative_saturate_w;
    wire [SCORE_WIDTH-1:0] rounded_score_w =
        output_saturate_w ? (scaled_negative_w ? 16'h8000 : 16'h7fff) :
        (scaled_negative_w ?
         (~rounded_abs_w[SCORE_WIDTH-1:0] + 16'd1) :
         rounded_abs_w[SCORE_WIDTH-1:0]);

    assign start_ready_o = (state_q == ST_IDLE) && !out_valid_q;
    assign pair_ready_o = (state_q == ST_RUN);
    assign out_valid_o = out_valid_q;
    assign score_o = score_q;
    assign acc_o = acc_q;
    assign saturation_seen_o = saturation_seen_q;

    always @* begin
        partial_sum_w = {ACC_WIDTH{1'b0}};
        for (mac_lane = 0; mac_lane < MAC_LANES; mac_lane = mac_lane + 1) begin
            q_lane_w = q_data_i[mac_lane*ACT_WIDTH +: ACT_WIDTH];
            k_lane_w = k_data_i[mac_lane*ACT_WIDTH +: ACT_WIDTH];
            partial_sum_w = partial_sum_w + (q_lane_w * k_lane_w);
        end
    end

    always @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            state_q <= ST_IDLE;
            group_idx_q <= {GROUP_INDEX_WIDTH{1'b0}};
            acc_q <= {ACC_WIDTH{1'b0}};
            scaled_product_q <= 64'sd0;
            score_q <= {SCORE_WIDTH{1'b0}};
            out_valid_q <= 1'b0;
            saturation_seen_q <= 1'b0;
        end else if (clear_i) begin
            state_q <= ST_IDLE;
            group_idx_q <= {GROUP_INDEX_WIDTH{1'b0}};
            acc_q <= {ACC_WIDTH{1'b0}};
            scaled_product_q <= 64'sd0;
            out_valid_q <= 1'b0;
            saturation_seen_q <= 1'b0;
        end else begin
            if (out_valid_q && out_ready_i) begin
                out_valid_q <= 1'b0;
                state_q <= ST_IDLE;
            end

            case (state_q)
                ST_IDLE: begin
                    if (start_valid_i && start_ready_o) begin
                        group_idx_q <= {GROUP_INDEX_WIDTH{1'b0}};
                        acc_q <= {ACC_WIDTH{1'b0}};
                        score_q <= {SCORE_WIDTH{1'b0}};
                        saturation_seen_q <= 1'b0;
                        state_q <= ST_RUN;
                    end
                end

                ST_RUN: begin
                    if (pair_valid_i && pair_ready_o) begin
                        acc_q <= acc_next_w;
                        if (group_idx_q == LAST_GROUP) begin
                            group_idx_q <= {GROUP_INDEX_WIDTH{1'b0}};
                            state_q <= ST_SCALE;
                        end else begin
                            group_idx_q <= group_idx_q + {{(GROUP_INDEX_WIDTH-1){1'b0}}, 1'b1};
                        end
                    end
                end

                ST_SCALE: begin
                    scaled_product_q <= acc_q * multiplier_i;
                    state_q <= ST_ROUND;
                end

                ST_ROUND: begin
                    score_q <= rounded_score_w;
                    saturation_seen_q <= output_saturate_w;
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
