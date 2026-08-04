`timescale 1ns/1ps
`default_nettype none

module ace2_absolute_rope_score_tb;
    reg clk;
    reg rst_n;
    reg clear;
    reg start_valid;
    wire start_ready;
    reg [31:0] query_scale32;
    reg [31:0] key_scale32;
    reg pair_valid;
    wire pair_ready;
    reg signed [7:0] query_low;
    reg signed [7:0] query_high;
    reg signed [7:0] key_low;
    reg signed [7:0] key_high;
    reg signed [15:0] query_cos;
    reg signed [15:0] query_sin;
    reg signed [15:0] key_cos;
    reg signed [15:0] key_sin;
    wire out_valid;
    reg out_ready;
    wire signed [53:0] score_raw;
    wire signed [63:0] logit;
    wire descriptor_error;
    wire numeric_overflow;

    reg [511:0] case_query;
    reg [511:0] case_key;
    reg [511:0] case_query_cos;
    reg [511:0] case_query_sin;
    reg [511:0] case_key_cos;
    reg [511:0] case_key_sin;
    reg signed [53:0] expected_score_raw;
    reg signed [63:0] expected_logit;
    integer failures;
    integer case_index;
    integer pair_index;
    integer guard;

    `include "../generated/absolute_rope_online_attention_vectors.svh"

    ace2_absolute_rope_score_core dut (
        .clk_i(clk), .rst_ni(rst_n), .clear_i(clear),
        .start_valid_i(start_valid), .start_ready_o(start_ready),
        .query_scale32_i(query_scale32), .key_scale32_i(key_scale32),
        .pair_valid_i(pair_valid), .pair_ready_o(pair_ready),
        .query_low_i(query_low), .query_high_i(query_high),
        .key_low_i(key_low), .key_high_i(key_high),
        .query_cos_q15_i(query_cos), .query_sin_q15_i(query_sin),
        .key_cos_q15_i(key_cos), .key_sin_q15_i(key_sin),
        .out_valid_o(out_valid), .out_ready_i(out_ready),
        .score_raw_o(score_raw), .logit_q12_20_o(logit),
        .descriptor_error_o(descriptor_error),
        .numeric_overflow_o(numeric_overflow)
    );

    initial begin clk = 1'b0; forever #5 clk = ~clk; end

    task apply_reset;
        begin
            rst_n = 1'b0; clear = 1'b0; start_valid = 1'b0; pair_valid = 1'b0;
            query_scale32 = 32'h00ffa245; key_scale32 = 32'h00008307;
            query_low = 0; query_high = 0; key_low = 0; key_high = 0;
            query_cos = 0; query_sin = 0; key_cos = 0; key_sin = 0;
            out_ready = 1'b0;
            repeat (4) @(posedge clk);
            rst_n = 1'b1;
            repeat (2) @(posedge clk);
        end
    endtask

    task run_case;
        input integer selected_case;
        begin
            load_abs_rope_score_case(selected_case, case_query, case_key,
                case_query_cos, case_query_sin, case_key_cos, case_key_sin,
                expected_score_raw, expected_logit);
            while (!start_ready) @(posedge clk);
            @(negedge clk); start_valid = 1'b1;
            @(posedge clk); @(negedge clk); start_valid = 1'b0;
            for (pair_index = 0; pair_index < 32; pair_index = pair_index + 1) begin
                while (!pair_ready) @(posedge clk);
                @(negedge clk);
                query_low = case_query[pair_index*8 +: 8];
                query_high = case_query[(pair_index+32)*8 +: 8];
                key_low = case_key[pair_index*8 +: 8];
                key_high = case_key[(pair_index+32)*8 +: 8];
                query_cos = case_query_cos[pair_index*16 +: 16];
                query_sin = case_query_sin[pair_index*16 +: 16];
                key_cos = case_key_cos[pair_index*16 +: 16];
                key_sin = case_key_sin[pair_index*16 +: 16];
                pair_valid = 1'b1;
                @(posedge clk); @(negedge clk); pair_valid = 1'b0;
            end
            guard = 0;
            while (!out_valid && guard < 2048) begin guard = guard + 1; @(posedge clk); end
            if (!out_valid) begin
                $display("ABS_ROPE_SCORE_TIMEOUT case=%0d", selected_case); failures = failures + 1;
            end else begin
                if (descriptor_error || numeric_overflow) begin
                    $display("ABS_ROPE_SCORE_UNEXPECTED_ERROR case=%0d descriptor=%0d overflow=%0d",
                        selected_case, descriptor_error, numeric_overflow); failures = failures + 1;
                end
                if (score_raw !== expected_score_raw) begin
                    $display("ABS_ROPE_SCORE_RAW_MISMATCH case=%0d got=%0d expected=%0d",
                        selected_case, score_raw, expected_score_raw); failures = failures + 1;
                end
                if (logit !== expected_logit) begin
                    $display("ABS_ROPE_SCORE_LOGIT_MISMATCH case=%0d got=%0d expected=%0d",
                        selected_case, logit, expected_logit); failures = failures + 1;
                end
                repeat (3) @(posedge clk);
                if (!out_valid) begin
                    $display("ABS_ROPE_SCORE_BACKPRESSURE_LOST case=%0d", selected_case); failures = failures + 1;
                end
                @(negedge clk); out_ready = 1'b1;
                @(posedge clk); @(negedge clk); out_ready = 1'b0;
            end
        end
    endtask

    task run_bad_scale;
        begin
            while (!start_ready) @(posedge clk);
            @(negedge clk); query_scale32 = 32'h01ffa245; start_valid = 1'b1;
            @(posedge clk); @(negedge clk); start_valid = 1'b0;
            guard = 0;
            while (!out_valid && guard < 32) begin guard = guard + 1; @(posedge clk); end
            if (!out_valid || !descriptor_error || pair_ready) begin
                $display("ABS_ROPE_SCORE_BAD_SCALE_NOT_REJECTED"); failures = failures + 1;
            end
            @(negedge clk); out_ready = 1'b1;
            @(posedge clk); @(negedge clk); out_ready = 1'b0;
            query_scale32 = 32'h00ffa245;
        end
    endtask

    initial begin
        failures = 0;
        apply_reset();
        for (case_index = 0; case_index < ABS_ROPE_SCORE_CASE_COUNT; case_index = case_index + 1)
            run_case(case_index);
        run_bad_scale();
        if (failures != 0) begin
            $display("ACE2_ABSOLUTE_ROPE_SCORE_TB_FAIL failures=%0d", failures);
            $fatal(1, "ACE2_ABSOLUTE_ROPE_SCORE_TB_FAIL");
        end
        $display("ACE2_ABSOLUTE_ROPE_SCORE_TB_PASS cases=%0d", ABS_ROPE_SCORE_CASE_COUNT);
        $finish;
    end
endmodule

`default_nettype wire

