`timescale 1ns/1ps
`default_nettype none

module ace2_tile_bfp_attention_tb;
    `include "tile_bfp_attention_vectors.svh"

    reg clk = 1'b0;
    always #5 clk = ~clk;
    initial begin
        repeat (10000) @(posedge clk);
        $fatal(1, "tile-BFP testbench watchdog expired");
    end
    reg rst_n = 1'b0;
    reg clear = 1'b0;

    reg score_start_valid;
    wire score_start_ready;
    reg [6:0] score_count;
    reg signed [7:0] score_exponent;
    reg score_valid;
    wire score_ready;
    reg signed [105:0] score_num;
    wire metadata_valid;
    reg metadata_ready;
    wire signed [127:0] tile_max;
    wire [4:0] fraction_bits;
    wire mantissa_valid;
    reg mantissa_ready;
    wire [5:0] mantissa_index;
    wire signed [23:0] mantissa;
    wire mantissa_last;
    wire score_descriptor_error;
    wire score_numeric_overflow;

    ace2_tile_bfp_score_core u_score (
        .clk_i(clk), .rst_ni(rst_n), .clear_i(clear),
        .start_valid_i(score_start_valid), .start_ready_o(score_start_ready),
        .key_count_u7_i(score_count), .score_exponent_s8_i(score_exponent),
        .score_valid_i(score_valid), .score_ready_o(score_ready),
        .score_num_s106_i(score_num),
        .metadata_valid_o(metadata_valid), .metadata_ready_i(metadata_ready),
        .tile_max_s128_o(tile_max), .fraction_bits_u5_o(fraction_bits),
        .mantissa_valid_o(mantissa_valid), .mantissa_ready_i(mantissa_ready),
        .mantissa_index_u6_o(mantissa_index), .mantissa_s24_o(mantissa),
        .mantissa_last_o(mantissa_last),
        .descriptor_error_o(score_descriptor_error),
        .numeric_overflow_o(score_numeric_overflow)
    );

    reg soft_start_valid;
    wire soft_start_ready;
    reg [6:0] soft_count;
    reg signed [7:0] soft_exponent;
    reg [4:0] soft_fraction;
    reg signed [127:0] row_max;
    reg signed [127:0] soft_tile_max;
    reg soft_mantissa_valid;
    wire soft_mantissa_ready;
    reg signed [23:0] soft_mantissa;
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

    ace2_bfp_hierarchical_softmax_core u_softmax (
        .clk_i(clk), .rst_ni(rst_n), .clear_i(clear),
        .start_valid_i(soft_start_valid), .start_ready_o(soft_start_ready),
        .key_count_u7_i(soft_count), .score_exponent_s8_i(soft_exponent),
        .fraction_bits_u5_i(soft_fraction), .row_max_s128_i(row_max),
        .tile_max_s128_i(soft_tile_max),
        .mantissa_valid_i(soft_mantissa_valid),
        .mantissa_ready_o(soft_mantissa_ready), .mantissa_s24_i(soft_mantissa),
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

    task pulse_clear;
        begin
            @(posedge clk);
            clear <= 1'b1;
            @(posedge clk);
            clear <= 1'b0;
        end
    endtask

    task start_score;
        input [6:0] count;
        begin
            @(posedge clk);
            while (!score_start_ready) @(posedge clk);
            score_count <= count;
            score_exponent <= -8'sd1;
            score_start_valid <= 1'b1;
            @(posedge clk);
            score_start_valid <= 1'b0;
        end
    endtask

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

    task send_soft_mantissa;
        input signed [23:0] value;
        begin
            @(posedge clk);
            while (!soft_mantissa_ready) @(posedge clk);
            soft_mantissa <= value;
            soft_mantissa_valid <= 1'b1;
            @(posedge clk);
            soft_mantissa_valid <= 1'b0;
            while (!weight_valid) @(posedge clk);
            @(posedge clk);
        end
    endtask

    integer seen;
    reg [31:0] first_weight;
    initial begin
        score_start_valid = 0; score_count = 0; score_exponent = 0;
        score_valid = 0; score_num = 0; metadata_ready = 0; mantissa_ready = 1;
        soft_start_valid = 0; soft_count = 0; soft_exponent = 0;
        soft_fraction = 0; row_max = 0; soft_tile_max = 0;
        soft_mantissa_valid = 0; soft_mantissa = 0; weight_ready = 1;
        sum_ready = 0; norm_valid = 0; norm_weight = 0;
        norm_denominator = 0; prob_ready = 1; first_weight = 0;
        repeat (3) @(posedge clk);
        rst_n = 1'b1;

        start_score(7'd3);
        send_score(BFP_SCORE0);
        send_score(BFP_SCORE1);
        send_score(BFP_SCORE2);
        while (!metadata_valid) @(posedge clk);
        if (tile_max !== ($signed(128'sd1000) <<< 66)) $fatal(1, "tile max mismatch");
        if (fraction_bits !== BFP_FRAC) $fatal(1, "fraction mismatch");
        repeat (2) begin
            @(posedge clk);
            if (!metadata_valid || fraction_bits !== BFP_FRAC) $fatal(1, "metadata backpressure mismatch");
        end
        metadata_ready <= 1'b1;
        @(posedge clk);
        metadata_ready <= 1'b0;
        seen = 0;
        while (seen < 3) begin
            @(posedge clk);
            if (mantissa_valid) begin
                case (seen)
                    0: if (mantissa !== BFP_MANT0) $fatal(1, "mantissa0 mismatch");
                    1: if (mantissa !== BFP_MANT1) $fatal(1, "mantissa1 mismatch");
                    2: if (mantissa !== BFP_MANT2 || !mantissa_last) $fatal(1, "mantissa2 mismatch");
                endcase
                seen = seen + 1;
            end
        end
        if (score_descriptor_error || score_numeric_overflow) $fatal(1, "score core error");

        pulse_clear();
        metadata_ready <= 1'b1;
        start_score(7'd2);
        send_score(BFP_BOUNDARY_SCORE0);
        send_score(BFP_BOUNDARY_SCORE1);
        while (!metadata_valid) @(posedge clk);
        if (fraction_bits !== BFP_BOUNDARY_FRAC) $fatal(1, "boundary fraction mismatch");
        seen = 0;
        while (seen < 2) begin
            @(posedge clk);
            if (mantissa_valid) begin
                if (seen == 1 && mantissa !== BFP_BOUNDARY_MANT1) $fatal(1, "boundary mantissa mismatch");
                seen = seen + 1;
            end
        end

        pulse_clear();
        start_score(7'd3);
        send_score(BFP_TIE_SCORE0);
        send_score(BFP_TIE_SCORE1);
        send_score(BFP_TIE_SCORE2);
        while (!metadata_valid) @(posedge clk);
        seen = 0;
        while (seen < 3) begin
            @(posedge clk);
            if (mantissa_valid) begin
                if (seen == 1 && mantissa !== BFP_TIE_MANT1) $fatal(1, "tie-even mantissa1 mismatch");
                if (seen == 2 && mantissa !== BFP_TIE_MANT2) $fatal(1, "tie-even mantissa2 mismatch");
                seen = seen + 1;
            end
        end

        pulse_clear();
        @(posedge clk);
        score_count <= 7'd1;
        score_exponent <= 8'sd0;
        score_start_valid <= 1'b1;
        @(posedge clk);
        score_start_valid <= 1'b0;
        @(posedge clk);
        if (!score_descriptor_error || !score_start_ready) $fatal(1, "score metadata error not fail-closed");

        pulse_clear();
        @(posedge clk);
        soft_count <= 7'd3;
        soft_exponent <= -8'sd1;
        soft_fraction <= 5'd17;
        row_max <= ($signed(128'sd1000) <<< 66);
        soft_tile_max <= ($signed(128'sd1000) <<< 66);
        soft_start_valid <= 1'b1;
        @(posedge clk);
        soft_start_valid <= 1'b0;
        send_soft_mantissa(BFP_MANT0); first_weight = weight;
        send_soft_mantissa(BFP_MANT1);
        send_soft_mantissa(BFP_MANT2);
        while (!sum_valid) @(posedge clk);
        if (first_weight !== 32'h80000000) $fatal(1, "exp(0) mismatch");
        if (weight_sum <= 48'h80000000) $fatal(1, "sum mismatch");
        norm_weight <= first_weight;
        norm_denominator <= weight_sum;
        norm_valid <= 1'b1;
        @(posedge clk);
        norm_valid <= 1'b0;
        sum_ready <= 1'b1;
        @(posedge clk);
        sum_ready <= 1'b0;
        while (!prob_valid) @(posedge clk);
        if (probability == 0 || probability > 15'd32767) $fatal(1, "probability mismatch");
        if (soft_descriptor_error || soft_numeric_overflow) $fatal(1, "softmax error");

        pulse_clear();
        @(posedge clk);
        soft_count <= 7'd1;
        soft_exponent <= -8'sd1;
        soft_fraction <= 5'd18;
        row_max <= 128'sd0;
        soft_tile_max <= 128'sd0;
        soft_start_valid <= 1'b1;
        @(posedge clk);
        soft_start_valid <= 1'b0;
        @(posedge clk);
        if (!soft_descriptor_error || !soft_start_ready) $fatal(1, "softmax metadata error not fail-closed");

        $display("TB_PASS tile_bfp_attention");
        $finish;
    end
endmodule

`default_nettype wire
