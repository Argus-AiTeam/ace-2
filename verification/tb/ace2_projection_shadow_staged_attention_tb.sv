`timescale 1ns/1ps
`default_nettype none

module ace2_projection_shadow_staged_attention_tb;
    reg clk;
    reg rst_n;
    reg clear;

    reg projection_valid;
    wire projection_ready;
    reg signed [31:0] projection_dot;
    reg signed [31:0] projection_bias;
    reg signed [31:0] projection_multiplier;
    reg [5:0] projection_shift;
    reg signed [7:0] projection_zero_point;
    reg [47:0] projection_reserved;
    wire projection_out_valid;
    reg projection_out_ready;
    wire signed [31:0] projection_shadow;
    wire projection_descriptor_error;
    wire projection_numeric_overflow;

    reg score_start_valid;
    wire score_start_ready;
    reg [31:0] query_scale32;
    reg [31:0] key_scale32;
    reg score_pair_valid;
    wire score_pair_ready;
    reg signed [31:0] query_low;
    reg signed [31:0] query_high;
    reg signed [31:0] key_low;
    reg signed [31:0] key_high;
    reg signed [15:0] query_cos;
    reg signed [15:0] query_sin;
    reg signed [15:0] key_cos;
    reg signed [15:0] key_sin;
    wire score_out_valid;
    reg score_out_ready;
    wire signed [73:0] dot_s74;
    wire signed [15:0] score_q6_9;
    wire score_descriptor_error;
    wire score_numeric_overflow;

    reg [2047:0] case_query;
    reg [2047:0] case_key;
    reg [511:0] case_query_cos;
    reg [511:0] case_query_sin;
    reg [511:0] case_key_cos;
    reg [511:0] case_key_sin;
    reg signed [73:0] expected_dot;
    reg signed [15:0] expected_score;
    reg signed [31:0] expected_shadow;
    integer failures;
    integer case_index;
    integer pair_index;
    integer guard;

    `include "../generated/projection_shadow_vectors.svh"

    ace2_projection_shadow_core projection_dut (
        .clk_i(clk), .rst_ni(rst_n), .clear_i(clear),
        .in_valid_i(projection_valid), .in_ready_o(projection_ready),
        .dot_s32_i(projection_dot), .bias_s32_i(projection_bias),
        .multiplier_s32_i(projection_multiplier),
        .right_shift_u6_i(projection_shift),
        .output_zero_point_s8_i(projection_zero_point),
        .reserved_i(projection_reserved),
        .out_valid_o(projection_out_valid), .out_ready_i(projection_out_ready),
        .shadow_q15_16_o(projection_shadow),
        .descriptor_error_o(projection_descriptor_error),
        .numeric_overflow_o(projection_numeric_overflow)
    );

    ace2_projection_shadow_score_core score_dut (
        .clk_i(clk), .rst_ni(rst_n), .clear_i(clear),
        .start_valid_i(score_start_valid), .start_ready_o(score_start_ready),
        .query_scale32_i(query_scale32), .key_scale32_i(key_scale32),
        .pair_valid_i(score_pair_valid), .pair_ready_o(score_pair_ready),
        .query_low_q15_16_i(query_low), .query_high_q15_16_i(query_high),
        .key_low_q15_16_i(key_low), .key_high_q15_16_i(key_high),
        .query_cos_q15_i(query_cos), .query_sin_q15_i(query_sin),
        .key_cos_q15_i(key_cos), .key_sin_q15_i(key_sin),
        .out_valid_o(score_out_valid), .out_ready_i(score_out_ready),
        .dot_s74_o(dot_s74), .score_q6_9_o(score_q6_9),
        .descriptor_error_o(score_descriptor_error),
        .numeric_overflow_o(score_numeric_overflow)
    );

    initial begin clk = 1'b0; forever #5 clk = ~clk; end

    task apply_reset;
        begin
            rst_n = 1'b0; clear = 1'b0;
            projection_valid = 1'b0; projection_out_ready = 1'b0;
            projection_dot = 0; projection_bias = 0; projection_multiplier = 1;
            projection_shift = 16; projection_zero_point = 0; projection_reserved = 0;
            score_start_valid = 1'b0; score_pair_valid = 1'b0; score_out_ready = 1'b0;
            query_scale32 = 32'h00ffa245; key_scale32 = 32'h00008307;
            query_low = 0; query_high = 0; key_low = 0; key_high = 0;
            query_cos = 0; query_sin = 0; key_cos = 0; key_sin = 0;
            repeat (4) @(posedge clk); rst_n = 1'b1; repeat (2) @(posedge clk);
        end
    endtask

    task run_projection_case;
        input integer selected_case;
        begin
            load_projection_shadow_case(selected_case, projection_dot, projection_bias,
                projection_multiplier, projection_shift, expected_shadow);
            while (!projection_ready) @(posedge clk);
            @(negedge clk); projection_valid = 1'b1;
            @(posedge clk); @(negedge clk); projection_valid = 1'b0;
            if (!projection_out_valid || projection_descriptor_error || projection_numeric_overflow ||
                projection_shadow !== expected_shadow) begin
                $display("PROJECTION_SHADOW_MISMATCH case=%0d got=%0d expected=%0d de=%0d ov=%0d",
                    selected_case, projection_shadow, expected_shadow,
                    projection_descriptor_error, projection_numeric_overflow);
                failures = failures + 1;
            end
            repeat (2) @(posedge clk);
            if (!projection_out_valid) begin
                $display("PROJECTION_SHADOW_BACKPRESSURE_LOST case=%0d", selected_case);
                failures = failures + 1;
            end
            @(negedge clk); projection_out_ready = 1'b1;
            @(posedge clk); @(negedge clk); projection_out_ready = 1'b0;
        end
    endtask

    task run_score_case;
        input integer selected_case;
        begin
            load_projection_shadow_score_case(selected_case, case_query, case_key,
                case_query_cos, case_query_sin, case_key_cos, case_key_sin,
                expected_dot, expected_score);
            while (!score_start_ready) @(posedge clk);
            @(negedge clk); score_start_valid = 1'b1;
            @(posedge clk); @(negedge clk); score_start_valid = 1'b0;
            for (pair_index = 0; pair_index < 32; pair_index = pair_index + 1) begin
                while (!score_pair_ready) @(posedge clk);
                @(negedge clk);
                query_low = case_query[pair_index*32 +: 32];
                query_high = case_query[(pair_index+32)*32 +: 32];
                key_low = case_key[pair_index*32 +: 32];
                key_high = case_key[(pair_index+32)*32 +: 32];
                query_cos = case_query_cos[pair_index*16 +: 16];
                query_sin = case_query_sin[pair_index*16 +: 16];
                key_cos = case_key_cos[pair_index*16 +: 16];
                key_sin = case_key_sin[pair_index*16 +: 16];
                score_pair_valid = 1'b1;
                @(posedge clk); @(negedge clk); score_pair_valid = 1'b0;
            end
            guard = 0;
            while (!score_out_valid && guard < 1024) begin
                guard = guard + 1; @(posedge clk);
            end
            if (!score_out_valid || score_descriptor_error || score_numeric_overflow ||
                dot_s74 !== expected_dot || score_q6_9 !== expected_score) begin
                $display("PROJECTION_SHADOW_SCORE_MISMATCH case=%0d dot=%0d expected_dot=%0d score=%0d expected_score=%0d de=%0d ov=%0d",
                    selected_case, dot_s74, expected_dot, score_q6_9, expected_score,
                    score_descriptor_error, score_numeric_overflow);
                failures = failures + 1;
            end
            repeat (2) @(posedge clk);
            if (!score_out_valid) begin
                $display("PROJECTION_SHADOW_SCORE_BACKPRESSURE_LOST case=%0d", selected_case);
                failures = failures + 1;
            end
            @(negedge clk); score_out_ready = 1'b1;
            @(posedge clk); @(negedge clk); score_out_ready = 1'b0;
        end
    endtask

    task run_bad_metadata;
        begin
            while (!projection_ready) @(posedge clk);
            @(negedge clk); projection_zero_point = 1; projection_valid = 1'b1;
            @(posedge clk); @(negedge clk); projection_valid = 1'b0;
            if (!projection_out_valid || !projection_descriptor_error) begin
                $display("PROJECTION_SHADOW_BAD_METADATA_NOT_REJECTED"); failures = failures + 1;
            end
            @(negedge clk); projection_out_ready = 1'b1;
            @(posedge clk); @(negedge clk); projection_out_ready = 1'b0;
            projection_zero_point = 0;

            while (!score_start_ready) @(posedge clk);
            @(negedge clk); query_scale32 = 32'h01ffa245; score_start_valid = 1'b1;
            @(posedge clk); @(negedge clk); score_start_valid = 1'b0;
            if (!score_out_valid || !score_descriptor_error || score_pair_ready) begin
                $display("PROJECTION_SHADOW_BAD_SCALE_NOT_REJECTED"); failures = failures + 1;
            end
            @(negedge clk); score_out_ready = 1'b1;
            @(posedge clk); @(negedge clk); score_out_ready = 1'b0;
            query_scale32 = 32'h00ffa245;
        end
    endtask

    initial begin
        failures = 0;
        apply_reset();
        for (case_index = 0; case_index < PROJECTION_SHADOW_CASE_COUNT; case_index = case_index + 1)
            run_projection_case(case_index);
        for (case_index = 0; case_index < PROJECTION_SHADOW_SCORE_CASE_COUNT; case_index = case_index + 1)
            run_score_case(case_index);
        run_bad_metadata();
        if (failures != 0) begin
            $display("ACE2_PROJECTION_SHADOW_TB_FAIL failures=%0d", failures);
            $fatal(1, "ACE2_PROJECTION_SHADOW_TB_FAIL");
        end
        $display("ACE2_PROJECTION_SHADOW_TB_PASS projection_cases=%0d score_cases=%0d",
            PROJECTION_SHADOW_CASE_COUNT, PROJECTION_SHADOW_SCORE_CASE_COUNT);
        $finish;
    end
endmodule

`default_nettype wire
