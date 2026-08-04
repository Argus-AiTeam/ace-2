`timescale 1ns/1ps
`default_nettype none

module ace2_absolute_rope_online_attention_tb;
    reg clk;
    reg rst_n;
    reg clear;
    reg start_valid;
    wire start_ready;
    reg state_valid;
    reg signed [63:0] maximum_in;
    reg [47:0] denominator_in;
    reg signed [63:0] logit_in;
    reg lane_valid;
    wire lane_ready;
    reg signed [55:0] numerator_in;
    reg signed [7:0] value_in;
    wire lane_out_valid;
    reg lane_out_ready;
    wire signed [55:0] numerator_out;
    wire lane_last;
    wire signed [63:0] maximum_out;
    wire [47:0] denominator_out;
    wire [31:0] weight_out;
    wire transaction_done;
    wire descriptor_error;
    wire numeric_overflow;

    reg finalize_start;
    wire finalize_ready;
    reg signed [55:0] finalize_numerator;
    reg [47:0] finalize_denominator;
    wire finalize_valid;
    reg finalize_out_ready;
    wire signed [7:0] finalize_value;
    wire finalize_error;

    reg signed [55:0] old_numerator [0:3];
    reg signed [55:0] expected_numerator [0:3];
    reg signed [7:0] lane_value [0:3];
    integer lane;
    integer failures;
    integer guard;

    ace2_absolute_rope_online_attention_core #(.LANE_COUNT(4)) dut (
        .clk_i(clk), .rst_ni(rst_n), .clear_i(clear),
        .start_valid_i(start_valid), .start_ready_o(start_ready),
        .state_valid_i(state_valid), .maximum_i(maximum_in),
        .denominator_i(denominator_in), .logit_q12_20_i(logit_in),
        .lane_valid_i(lane_valid), .lane_ready_o(lane_ready),
        .numerator_i(numerator_in), .value_i(value_in),
        .lane_out_valid_o(lane_out_valid), .lane_out_ready_i(lane_out_ready),
        .numerator_o(numerator_out), .lane_last_o(lane_last),
        .maximum_o(maximum_out), .denominator_o(denominator_out),
        .weight_q31_o(weight_out), .transaction_done_o(transaction_done),
        .descriptor_error_o(descriptor_error),
        .numeric_overflow_o(numeric_overflow)
    );

    ace2_absolute_rope_online_attention_finalize_core finalize_dut (
        .clk_i(clk), .rst_ni(rst_n), .clear_i(clear),
        .start_valid_i(finalize_start), .start_ready_o(finalize_ready),
        .numerator_i(finalize_numerator), .denominator_i(finalize_denominator),
        .out_valid_o(finalize_valid), .out_ready_i(finalize_out_ready),
        .value_o(finalize_value), .descriptor_error_o(finalize_error)
    );

    initial begin clk = 1'b0; forever #5 clk = ~clk; end

    task apply_reset;
        begin
            rst_n = 1'b0;
            clear = 1'b0;
            start_valid = 1'b0;
            state_valid = 1'b0;
            maximum_in = 64'sd0;
            denominator_in = 48'd0;
            logit_in = 64'sd0;
            lane_valid = 1'b0;
            numerator_in = 56'sd0;
            value_in = 8'sd0;
            lane_out_ready = 1'b0;
            finalize_start = 1'b0;
            finalize_numerator = 56'sd0;
            finalize_denominator = 48'd0;
            finalize_out_ready = 1'b0;
            repeat (4) @(posedge clk);
            rst_n = 1'b1;
            repeat (2) @(posedge clk);
        end
    endtask

    task run_update;
        input expected_state_valid;
        input signed [63:0] expected_maximum_in;
        input [47:0] expected_denominator_in;
        input signed [63:0] selected_logit;
        input signed [63:0] expected_maximum;
        input [47:0] expected_denominator;
        input [31:0] expected_weight;
        begin
            while (!start_ready) @(posedge clk);
            @(negedge clk);
            state_valid = expected_state_valid;
            maximum_in = expected_maximum_in;
            denominator_in = expected_denominator_in;
            logit_in = selected_logit;
            start_valid = 1'b1;
            @(posedge clk); @(negedge clk); start_valid = 1'b0;
            for (lane = 0; lane < 4; lane = lane + 1) begin
                guard = 0;
                while (!lane_ready && guard < 100) begin guard = guard + 1; @(posedge clk); end
                if (!lane_ready) begin
                    $display("ABS_ONLINE_LANE_READY_TIMEOUT lane=%0d", lane);
                    failures = failures + 1;
                end
                @(negedge clk);
                numerator_in = old_numerator[lane];
                value_in = lane_value[lane];
                lane_valid = 1'b1;
                @(posedge clk); @(negedge clk); lane_valid = 1'b0;
                guard = 0;
                while (!lane_out_valid && guard < 100) begin guard = guard + 1; @(posedge clk); end
                if (!lane_out_valid) begin
                    $display("ABS_ONLINE_LANE_OUT_TIMEOUT lane=%0d", lane);
                    failures = failures + 1;
                end else begin
                    if (numerator_out !== expected_numerator[lane]) begin
                        $display("ABS_ONLINE_NUM_MISMATCH lane=%0d got=%0d expected=%0d",
                            lane, numerator_out, expected_numerator[lane]);
                        failures = failures + 1;
                    end
                    if (lane_last !== (lane == 3)) begin
                        $display("ABS_ONLINE_LAST_MISMATCH lane=%0d last=%0d", lane, lane_last);
                        failures = failures + 1;
                    end
                    if (maximum_out !== expected_maximum ||
                        denominator_out !== expected_denominator ||
                        weight_out !== expected_weight) begin
                        $display("ABS_ONLINE_STATE_MISMATCH lane=%0d max=%0d den=%0d weight=%0d",
                            lane, maximum_out, denominator_out, weight_out);
                        failures = failures + 1;
                    end
                    if (descriptor_error || numeric_overflow) begin
                        $display("ABS_ONLINE_UNEXPECTED_ERROR lane=%0d desc=%0d ovf=%0d",
                            lane, descriptor_error, numeric_overflow);
                        failures = failures + 1;
                    end
                    repeat (2) @(posedge clk);
                    if (!lane_out_valid) begin
                        $display("ABS_ONLINE_BACKPRESSURE_LOST lane=%0d", lane);
                        failures = failures + 1;
                    end
                    @(negedge clk); lane_out_ready = 1'b1;
                    @(posedge clk); @(negedge clk); lane_out_ready = 1'b0;
                end
            end
        end
    endtask

    task run_finalize;
        input signed [55:0] selected_numerator;
        input [47:0] selected_denominator;
        input signed [7:0] expected_value;
        input expected_error;
        begin
            while (!finalize_ready) @(posedge clk);
            @(negedge clk);
            finalize_numerator = selected_numerator;
            finalize_denominator = selected_denominator;
            finalize_start = 1'b1;
            @(posedge clk); @(negedge clk); finalize_start = 1'b0;
            guard = 0;
            while (!finalize_valid && guard < 100) begin guard = guard + 1; @(posedge clk); end
            if (!finalize_valid) begin
                $display("ABS_FINALIZE_TIMEOUT"); failures = failures + 1;
            end else begin
                if (finalize_value !== expected_value || finalize_error !== expected_error) begin
                    $display("ABS_FINALIZE_MISMATCH num=%0d den=%0d got=%0d err=%0d expected=%0d/%0d",
                        selected_numerator, selected_denominator, finalize_value,
                        finalize_error, expected_value, expected_error);
                    failures = failures + 1;
                end
                @(negedge clk); finalize_out_ready = 1'b1;
                @(posedge clk); @(negedge clk); finalize_out_ready = 1'b0;
            end
        end
    endtask

    initial begin
        failures = 0;
        lane_value[0] = 8'sd1;
        lane_value[1] = -8'sd2;
        lane_value[2] = 8'sd3;
        lane_value[3] = -8'sd4;
        apply_reset();

        old_numerator[0] = 56'sd0; old_numerator[1] = 56'sd0;
        old_numerator[2] = 56'sd0; old_numerator[3] = 56'sd0;
        expected_numerator[0] = 56'sd2147483648;
        expected_numerator[1] = -56'sd4294967296;
        expected_numerator[2] = 56'sd6442450944;
        expected_numerator[3] = -56'sd8589934592;
        run_update(1'b0, 64'sd0, 48'd0, 64'sd0, 64'sd0,
            48'd2147483648, 32'd2147483648);

        old_numerator[0] = 56'sd2147483648;
        old_numerator[1] = -56'sd4294967296;
        old_numerator[2] = 56'sd6442450944;
        old_numerator[3] = -56'sd8589934592;
        expected_numerator[0] = 56'sd4294967296;
        expected_numerator[1] = -56'sd8589934592;
        expected_numerator[2] = 56'sd12884901888;
        expected_numerator[3] = -56'sd17179869184;
        run_update(1'b1, 64'sd0, 48'd2147483648, 64'sd0, 64'sd0,
            48'd4294967296, 32'd2147483648);

        old_numerator[0] = 56'sd4294967296;
        old_numerator[1] = -56'sd8589934592;
        old_numerator[2] = 56'sd12884901888;
        old_numerator[3] = -56'sd17179869184;
        expected_numerator[0] = 56'sd3727513816;
        expected_numerator[1] = -56'sd7455027632;
        expected_numerator[2] = 56'sd11182541448;
        expected_numerator[3] = -56'sd14910055264;
        run_update(1'b1, 64'sd0, 48'd4294967296, 64'sd1048576,
            64'sd1048576, 48'd3727513816, 32'd790015084);

        run_finalize(56'sd2, 48'd4, 8'sd0, 1'b0);
        run_finalize(56'sd6, 48'd4, 8'sd2, 1'b0);
        run_finalize(-56'sd2, 48'd4, 8'sd0, 1'b0);
        run_finalize(-56'sd6, 48'd4, -8'sd2, 1'b0);
        run_finalize(56'sd1000, 48'd3, 8'sd127, 1'b0);
        run_finalize(-56'sd1000, 48'd3, 8'sh80, 1'b0);
        run_finalize(56'sd1, 48'd0, 8'sd0, 1'b1);

        if (failures != 0) begin
            $display("ACE2_ABSOLUTE_ROPE_ONLINE_ATTENTION_TB_FAIL failures=%0d", failures);
            $fatal(1, "ACE2_ABSOLUTE_ROPE_ONLINE_ATTENTION_TB_FAIL");
        end
        $display("ACE2_ABSOLUTE_ROPE_ONLINE_ATTENTION_TB_PASS");
        $finish;
    end
endmodule

`default_nettype wire
