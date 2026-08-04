`timescale 1ns/1ps
`default_nettype none

module ace2_tile_max_delta_attention_tb;
    `include "tile_max_delta_attention_vectors.svh"
    reg clk = 1'b0;
    always #5 clk = ~clk;
    reg rst_n = 1'b0;
    reg clear = 1'b0;

    reg score_start_valid;
    wire score_start_ready;
    reg [6:0] score_count;
    reg [6:0] score_shift;
    reg score_valid;
    wire score_ready;
    reg signed [105:0] score_num;
    wire max_valid;
    reg max_ready;
    wire signed [127:0] tile_max;
    wire delta_valid;
    reg delta_ready;
    wire [5:0] delta_index;
    wire signed [23:0] delta_value;
    wire delta_last;
    wire score_descriptor_error;
    wire score_numeric_overflow;

    ace2_tile_max_delta_score_core u_score (
        .clk_i(clk), .rst_ni(rst_n), .clear_i(clear),
        .start_valid_i(score_start_valid), .start_ready_o(score_start_ready),
        .key_count_u7_i(score_count), .delta_shift_u7_i(score_shift),
        .score_valid_i(score_valid), .score_ready_o(score_ready),
        .score_num_s106_i(score_num), .max_valid_o(max_valid),
        .max_ready_i(max_ready), .tile_max_s128_o(tile_max),
        .delta_valid_o(delta_valid), .delta_ready_i(delta_ready),
        .delta_index_u6_o(delta_index), .delta_q6_17_s24_o(delta_value),
        .delta_last_o(delta_last), .descriptor_error_o(score_descriptor_error),
        .numeric_overflow_o(score_numeric_overflow)
    );

    reg soft_start_valid;
    wire soft_start_ready;
    reg [6:0] soft_count;
    reg [6:0] soft_shift;
    reg signed [127:0] row_max;
    reg signed [127:0] soft_tile_max;
    reg soft_delta_valid;
    wire soft_delta_ready;
    reg signed [23:0] soft_local_delta;
    wire weight_valid;
    reg weight_ready;
    wire [31:0] weight;
    wire weight_last;
    wire sum_valid;
    reg sum_ready;
    wire [47:0] weight_sum;
    reg norm_valid;
    wire norm_ready;
    reg [31:0] norm_weight;
    reg [47:0] norm_denominator;
    wire prob_valid;
    reg prob_ready;
    wire [14:0] probability;
    wire soft_descriptor_error;
    wire soft_numeric_overflow;

    ace2_hierarchical_softmax_core u_softmax (
        .clk_i(clk), .rst_ni(rst_n), .clear_i(clear),
        .start_valid_i(soft_start_valid), .start_ready_o(soft_start_ready),
        .key_count_u7_i(soft_count), .delta_shift_u7_i(soft_shift),
        .row_max_s128_i(row_max), .tile_max_s128_i(soft_tile_max),
        .delta_valid_i(soft_delta_valid), .delta_ready_o(soft_delta_ready),
        .local_delta_q6_17_s24_i(soft_local_delta),
        .weight_valid_o(weight_valid), .weight_ready_i(weight_ready),
        .weight_q1_31_o(weight), .weight_last_o(weight_last),
        .sum_valid_o(sum_valid), .sum_ready_i(sum_ready),
        .weight_sum_u48_o(weight_sum), .norm_valid_i(norm_valid),
        .norm_ready_o(norm_ready), .norm_weight_q1_31_i(norm_weight),
        .norm_denominator_u48_i(norm_denominator), .prob_valid_o(prob_valid),
        .prob_ready_i(prob_ready), .probability_q0_15_o(probability),
        .descriptor_error_o(soft_descriptor_error),
        .numeric_overflow_o(soft_numeric_overflow)
    );

    task send_score;
        input signed [105:0] value;
        begin
            @(posedge clk);
            while (!score_ready) @(posedge clk);
            score_num <= value;
            score_valid <= 1'b1;
            @(posedge clk);
            score_valid <= 1'b0;
        end
    endtask

    task send_delta;
        input signed [23:0] value;
        begin
            @(posedge clk);
            while (!soft_delta_ready) @(posedge clk);
            soft_local_delta <= value;
            soft_delta_valid <= 1'b1;
            @(posedge clk);
            soft_delta_valid <= 1'b0;
            while (!weight_valid) @(posedge clk);
            @(posedge clk);
        end
    endtask

    integer seen;
    reg [31:0] first_weight;
    initial begin
        score_start_valid = 0; score_count = 0; score_shift = 0;
        score_valid = 0; score_num = 0; max_ready = 1; delta_ready = 1;
        soft_start_valid = 0; soft_count = 0; soft_shift = 0;
        row_max = 0; soft_tile_max = 0; soft_delta_valid = 0;
        soft_local_delta = 0; weight_ready = 1; sum_ready = 0;
        norm_valid = 0; norm_weight = 0; norm_denominator = 0;
        prob_ready = 1; first_weight = 0;
        repeat (3) @(posedge clk);
        rst_n = 1'b1;

        @(posedge clk);
        score_count <= 7'd3; score_shift <= 7'd49; score_start_valid <= 1'b1;
        @(posedge clk); score_start_valid <= 1'b0;
        send_score(TILE_MAX_SCORE0);
        send_score(TILE_MAX_SCORE1);
        send_score(TILE_MAX_SCORE2);
        while (!max_valid) @(posedge clk);
        if (tile_max !== ($signed(128'sd1000) <<< 66)) $fatal(1, "tile max mismatch");
        seen = 0;
        while (seen < 3) begin
            @(posedge clk);
            if (delta_valid) begin
                case (seen)
                    0: if (delta_value !== TILE_MAX_DELTA0) $fatal(1, "delta0 mismatch");
                    1: if (delta_value !== TILE_MAX_DELTA1) $fatal(1, "delta1 mismatch");
                    2: if (delta_value !== TILE_MAX_DELTA2 || !delta_last) $fatal(1, "delta2 mismatch");
                endcase
                seen = seen + 1;
            end
        end
        if (score_descriptor_error || score_numeric_overflow) $fatal(1, "score error");

        @(posedge clk);
        soft_count <= 7'd3; soft_shift <= 7'd49;
        row_max <= ($signed(128'sd1000) <<< 66);
        soft_tile_max <= ($signed(128'sd1000) <<< 66);
        soft_start_valid <= 1'b1;
        @(posedge clk); soft_start_valid <= 1'b0;
        send_delta(24'sd0); first_weight = weight;
        send_delta(-24'sd131072);
        send_delta(-24'sd262144);
        while (!sum_valid) @(posedge clk);
        if (first_weight !== 32'h80000000) $fatal(1, "exp(0) mismatch");
        if (weight_sum <= 48'h80000000) $fatal(1, "sum mismatch");
        norm_weight <= first_weight;
        norm_denominator <= weight_sum;
        norm_valid <= 1'b1;
        @(posedge clk); norm_valid <= 1'b0;
        sum_ready <= 1'b1;
        @(posedge clk); sum_ready <= 1'b0;
        while (!prob_valid) @(posedge clk);
        if (probability == 0 || probability > 15'd32767) $fatal(1, "probability mismatch");
        if (soft_descriptor_error || soft_numeric_overflow) $fatal(1, "softmax error");

        $display("TB_PASS tile_max_delta_attention");
        $finish;
    end
endmodule

`default_nettype wire
