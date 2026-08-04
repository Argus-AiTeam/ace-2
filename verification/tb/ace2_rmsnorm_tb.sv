`timescale 1ns/1ps
`default_nettype none

module ace2_rmsnorm_tb;
    localparam integer LANES = 16;
    localparam integer ACT_WIDTH = 8;
    localparam integer GAIN_WIDTH = 16;

    reg clk;
    reg rst_n;
    reg start_valid;
    wire start_ready;
    reg in_valid;
    wire in_ready;
    reg [LANES*ACT_WIDTH-1:0] in_data;
    reg gain_valid;
    wire gain_ready;
    reg [LANES*GAIN_WIDTH-1:0] gain_data;
    reg scale_act_valid;
    wire scale_act_ready;
    reg [LANES*ACT_WIDTH-1:0] scale_act_data;
    wire out_valid;
    reg out_ready;
    wire [LANES*ACT_WIDTH-1:0] out_data;
    wire done_valid;
    reg done_ready;
    wire [47:0] sumsq;
    wire [31:0] inv_rms_q30;
    wire saturation_seen;

    integer failures;
    integer case_index;
    integer beat_index;
    integer output_index;
    integer cycle_guard;

    `include "../generated/rmsnorm_vectors.svh"

    ace2_rmsnorm_core dut (
        .clk_i(clk),
        .rst_ni(rst_n),
        .clear_i(1'b0),
        .start_valid_i(start_valid),
        .start_ready_o(start_ready),
        .in_valid_i(in_valid),
        .in_ready_o(in_ready),
        .in_data_i(in_data),
        .gain_valid_i(gain_valid),
        .gain_ready_o(gain_ready),
        .gain_data_i(gain_data),
        .scale_act_valid_i(scale_act_valid),
        .scale_act_ready_o(scale_act_ready),
        .scale_act_data_i(scale_act_data),
        .out_valid_o(out_valid),
        .out_ready_i(out_ready),
        .out_data_o(out_data),
        .done_valid_o(done_valid),
        .done_ready_i(done_ready),
        .sumsq_o(sumsq),
        .inv_rms_q30_o(inv_rms_q30),
        .saturation_seen_o(saturation_seen)
    );

    initial begin
        clk = 1'b0;
        forever #5 clk = ~clk;
    end

    task apply_reset;
        begin
            rst_n = 1'b0;
            start_valid = 1'b0;
            in_valid = 1'b0;
            in_data = {LANES*ACT_WIDTH{1'b0}};
            gain_valid = 1'b0;
            gain_data = {LANES*GAIN_WIDTH{1'b0}};
            scale_act_valid = 1'b0;
            scale_act_data = {LANES*ACT_WIDTH{1'b0}};
            out_ready = 1'b0;
            done_ready = 1'b0;
            repeat (4) @(posedge clk);
            rst_n = 1'b1;
            repeat (2) @(posedge clk);
        end
    endtask

    task reset_mid_collect_check;
        begin
            @(negedge clk);
            start_valid = 1'b1;
            @(posedge clk);
            @(negedge clk);
            start_valid = 1'b0;
            repeat (3) begin
                in_valid = 1'b1;
                in_data = test_input_beats[0];
                @(posedge clk);
                @(negedge clk);
            end
            rst_n = 1'b0;
            @(posedge clk);
            if (out_valid || done_valid || !start_ready) begin
                $display("RESET_MID_COLLECT_FAIL out_valid=%0d done_valid=%0d start_ready=%0d", out_valid, done_valid, start_ready);
                failures = failures + 1;
            end
            in_valid = 1'b0;
            rst_n = 1'b1;
            repeat (2) @(posedge clk);
        end
    endtask

    task run_case;
        input integer selected_case;
        begin
            while (!start_ready) @(posedge clk);
            @(negedge clk);
            start_valid = 1'b1;
            @(posedge clk);
            @(negedge clk);
            start_valid = 1'b0;

            for (beat_index = 0; beat_index < TEST_BEATS; beat_index = beat_index + 1) begin
                while (!in_ready) @(posedge clk);
                @(negedge clk);
                in_data = test_input_beats[selected_case*TEST_BEATS + beat_index];
                in_valid = 1'b1;
                @(posedge clk);
                if (!in_ready) begin
                    $display("INPUT_NOT_READY case=%0d beat=%0d", selected_case, beat_index);
                    failures = failures + 1;
                end
                @(negedge clk);
                in_valid = 1'b0;
            end
            in_valid = 1'b0;

            beat_index = 0;
            output_index = 0;
            cycle_guard = 0;
            gain_valid = 1'b0;
            out_ready = 1'b1;
            while (output_index < TEST_BEATS) begin
                if ((beat_index < TEST_BEATS) && scale_act_ready && gain_ready) begin
                    @(negedge clk);
                    scale_act_data = test_input_beats[selected_case*TEST_BEATS + beat_index];
                    gain_data = test_gain_beats[selected_case*TEST_BEATS + beat_index];
                    scale_act_valid = 1'b1;
                    gain_valid = 1'b1;
                    @(posedge clk);
                    @(negedge clk);
                    scale_act_valid = 1'b0;
                    gain_valid = 1'b0;
                    beat_index = beat_index + 1;
                end
                if (out_valid) begin
                    if (out_data !== test_expected_beats[selected_case*TEST_BEATS + output_index]) begin
                        $display("OUTPUT_MISMATCH case=%0d beat=%0d got=%032x expected=%032x",
                                 selected_case, output_index, out_data,
                                 test_expected_beats[selected_case*TEST_BEATS + output_index]);
                        failures = failures + 1;
                    end
                    output_index = output_index + 1;
                    @(posedge clk);
                end
                cycle_guard = cycle_guard + 1;
                if (cycle_guard > 100000) begin
                    $display("CASE_TIMEOUT case=%0d output_index=%0d gain_index=%0d", selected_case, output_index, beat_index);
                    failures = failures + 1;
                    output_index = TEST_BEATS;
                end
                @(negedge clk);
            end
            gain_valid = 1'b0;
            scale_act_valid = 1'b0;
            out_ready = 1'b1;
            cycle_guard = 0;
            while (!done_valid && (cycle_guard < 100000)) begin
                cycle_guard = cycle_guard + 1;
                @(posedge clk);
            end
            if (!done_valid) begin
                $display("DONE_TIMEOUT case=%0d", selected_case);
                failures = failures + 1;
            end
            if (sumsq !== test_expected_sumsq[selected_case]) begin
                $display("SUMSQ_MISMATCH case=%0d got=%0d expected=%0d", selected_case, sumsq, test_expected_sumsq[selected_case]);
                failures = failures + 1;
            end
            if (inv_rms_q30 !== test_expected_inv[selected_case]) begin
                $display("INV_MISMATCH case=%0d got=%0d expected=%0d", selected_case, inv_rms_q30, test_expected_inv[selected_case]);
                failures = failures + 1;
            end
            if (saturation_seen !== test_expected_saturation[selected_case]) begin
                $display("SAT_MISMATCH case=%0d got=%0d expected=%0d", selected_case, saturation_seen, test_expected_saturation[selected_case]);
                failures = failures + 1;
            end
            done_ready = 1'b1;
            @(posedge clk);
            @(negedge clk);
            done_ready = 1'b0;
            out_ready = 1'b0;
        end
    endtask

    initial begin
        failures = 0;
        apply_reset();
        reset_mid_collect_check();
        for (case_index = 0; case_index < TEST_COUNT; case_index = case_index + 1) begin
            run_case(case_index);
        end
        if (failures != 0) begin
            $display("ACE2_RMSNORM_TB_FAIL failures=%0d", failures);
            $fatal(1, "ACE2_RMSNORM_TB_FAIL");
        end
        $display("ACE2_RMSNORM_TB_PASS cases=%0d beats_per_case=%0d", TEST_COUNT, TEST_BEATS);
        $finish;
    end
endmodule

`default_nettype wire
