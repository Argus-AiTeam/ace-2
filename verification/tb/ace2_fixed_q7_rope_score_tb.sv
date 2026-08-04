`timescale 1ns/1ps
`default_nettype none

module ace2_fixed_q7_rope_score_tb;
    reg clk = 1'b0;
    reg rst_n = 1'b0;
    reg clear = 1'b0;
    reg start_valid = 1'b0;
    wire start_ready;
    reg score_mode = 1'b0;
    reg [511:0] act_data = 512'd0;
    reg [1023:0] cos_q15 = 1024'd0;
    reg [1023:0] sin_q15 = 1024'd0;
    reg [1023:0] query_q7 = 1024'd0;
    reg [1023:0] key_q7 = 1024'd0;
    reg [31:0] query_scale32 = 32'h00f88000;
    reg [31:0] key_scale32 = 32'h00f88000;
    wire out_valid;
    reg out_ready = 1'b0;
    wire [1023:0] rope_q7;
    wire signed [63:0] precenter;
    wire signed [37:0] dot;
    wire [8:0] multiplier_cycles;
    wire error_valid;
    wire coefficient_error;
    wire numeric_overflow;
    integer lane;
    integer failures = 0;
    integer expected_dot;
    reg signed [7:0] activation;
    reg signed [15:0] expected_wide;
    reg signed [15:0] observed_wide;

    always #5 clk = ~clk;

    ace2_fixed_q7_rope_score_core dut (
        .clk_i(clk), .rst_ni(rst_n), .clear_i(clear),
        .start_valid_i(start_valid), .start_ready_o(start_ready),
        .score_mode_i(score_mode), .act_data_i(act_data),
        .cos_q15_i(cos_q15), .sin_q15_i(sin_q15),
        .query_q7_i(query_q7), .key_q7_i(key_q7),
        .query_scale32_i(query_scale32), .key_scale32_i(key_scale32),
        .out_valid_o(out_valid), .out_ready_i(out_ready),
        .rope_q7_o(rope_q7), .precenter_q6_9_o(precenter),
        .dot_q7xq7_o(dot), .multiplier_cycles_o(multiplier_cycles),
        .error_valid_o(error_valid), .coefficient_error_o(coefficient_error),
        .numeric_overflow_o(numeric_overflow)
    );

    function automatic signed [15:0] rne_div256;
        input integer value;
        integer magnitude;
        integer quotient;
        integer remainder;
        begin
            magnitude = value < 0 ? -value : value;
            quotient = magnitude / 256;
            remainder = magnitude % 256;
            if ((remainder > 128) || ((remainder == 128) && (quotient & 1)))
                quotient = quotient + 1;
            rne_div256 = value < 0 ? -quotient : quotient;
        end
    endfunction

    task automatic launch;
        begin
            while (!start_ready) @(posedge clk);
            start_valid = 1'b1;
            @(posedge clk);
            start_valid = 1'b0;
        end
    endtask

    initial begin
        repeat (3) @(posedge clk);
        rst_n = 1'b1;
        for (lane = 0; lane < 64; lane = lane + 1) begin
            activation = lane - 32;
            act_data[lane*8 +: 8] = activation;
            cos_q15[lane*16 +: 16] = 16'sd32767;
            sin_q15[lane*16 +: 16] = 16'sd0;
        end
        score_mode = 1'b0;
        launch();
        wait (out_valid || error_valid);
        if (error_valid) begin
            $display("ROPE_UNEXPECTED_ERROR coefficient=%0d numeric=%0d", coefficient_error, numeric_overflow);
            failures = failures + 1;
        end
        repeat (3) begin
            @(posedge clk);
            if (!out_valid) begin
                $display("ROPE_BACKPRESSURE_RESULT_DROPPED");
                failures = failures + 1;
            end
        end
        for (lane = 0; lane < 64; lane = lane + 1) begin
            activation = lane - 32;
            expected_wide = rne_div256(activation * 32767);
            observed_wide = rope_q7[lane*16 +: 16];
            if (observed_wide !== expected_wide) begin
                $display("ROPE_MISMATCH lane=%0d expected=%0d observed=%0d", lane, expected_wide, observed_wide);
                failures = failures + 1;
            end
        end
        out_ready = 1'b1;
        @(posedge clk);
        out_ready = 1'b0;

        clear = 1'b1;
        @(posedge clk);
        clear = 1'b0;
        cos_q15[0 +: 16] = 16'sd32767;
        cos_q15[32*16 +: 16] = 16'sd32767;
        sin_q15[0 +: 16] = 16'sd32767;
        sin_q15[32*16 +: 16] = 16'sd32767;
        launch();
        wait (error_valid);
        if (!coefficient_error || numeric_overflow) begin
            $display("ROPE_BAD_COEFFICIENT_CLASSIFICATION");
            failures = failures + 1;
        end

        clear = 1'b1;
        @(posedge clk);
        clear = 1'b0;
        expected_dot = 0;
        for (lane = 0; lane < 64; lane = lane + 1) begin
            query_q7[lane*16 +: 16] = $signed(lane - 32);
            key_q7[lane*16 +: 16] = $signed(31 - lane);
            expected_dot = expected_dot + (lane - 32) * (31 - lane);
        end
        score_mode = 1'b1;
        query_scale32 = 32'h00008000;
        key_scale32 = 32'h00008000;
        launch();
        wait (out_valid || error_valid);
        if (error_valid) begin
            $display("SCORE_UNEXPECTED_ERROR");
            failures = failures + 1;
        end
        if (dot !== expected_dot) begin
            $display("SCORE_DOT_MISMATCH expected=%0d observed=%0d", expected_dot, dot);
            failures = failures + 1;
        end
        if (precenter !== rne_div256(expected_dot)) begin
            $display("SCORE_SCALE_MISMATCH expected=%0d observed=%0d", rne_div256(expected_dot), precenter);
            failures = failures + 1;
        end
        if (multiplier_cycles !== 9'd256) begin
            $display("SCORE_CYCLE_MISMATCH expected=256 observed=%0d", multiplier_cycles);
            failures = failures + 1;
        end
        out_ready = 1'b1;
        @(posedge clk);
        out_ready = 1'b0;

        if (failures != 0) begin
            $display("ACE2_FIXED_Q7_ROPE_SCORE_TB_FAIL failures=%0d", failures);
            $fatal(1);
        end
        $display("ACE2_FIXED_Q7_ROPE_SCORE_TB_PASS multiplier_cycles=256 backpressure=pass reset_clear=pass");
        $finish;
    end
endmodule

`default_nettype wire
