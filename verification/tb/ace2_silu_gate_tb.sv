`timescale 1ns/1ps
`default_nettype none

module ace2_silu_gate_tb;
    reg clk;
    reg rst_n;
    reg clear;
    reg start_valid;
    wire start_ready;
    reg signed [31:0] multiplier;
    reg [5:0] right_shift;
    reg signed [7:0] zero_point;
    reg beat_valid;
    wire beat_ready;
    reg [3:0] lane_count;
    reg [127:0] gate_data;
    reg [127:0] up_data;
    wire out_valid;
    reg out_ready;
    wire [63:0] out_data;
    wire saturation_seen;
    integer failures;
    integer case_index;
    integer beat;
    integer lane;
    integer input_beats;
    integer expected_index;
    reg expected_saturation;
    reg [5:0] executed_boundary_coverage;

    `include "../generated/silu_gate_vectors.svh"

    ace2_silu_gate_core dut (
        .clk_i(clk),
        .rst_ni(rst_n),
        .clear_i(clear),
        .start_valid_i(start_valid),
        .start_ready_o(start_ready),
        .multiplier_i(multiplier),
        .right_shift_i(right_shift),
        .output_zero_point_i(zero_point),
        .beat_valid_i(beat_valid),
        .beat_ready_o(beat_ready),
        .lane_count_i(lane_count),
        .gate_data_i(gate_data),
        .up_data_i(up_data),
        .out_valid_o(out_valid),
        .out_ready_i(out_ready),
        .out_data_o(out_data),
        .saturation_seen_o(saturation_seen)
    );

    initial begin
        clk = 1'b0;
        forever #5 clk = ~clk;
    end

    initial begin
        failures = 0;
        rst_n = 1'b0;
        clear = 1'b0;
        start_valid = 1'b0;
        multiplier = 32'sd0;
        right_shift = 6'd0;
        zero_point = 8'sd0;
        beat_valid = 1'b0;
        lane_count = 4'd0;
        gate_data = 128'd0;
        up_data = 128'd0;
        out_ready = 1'b0;
        executed_boundary_coverage = 6'd0;
        repeat (4) @(posedge clk);
        rst_n = 1'b1;

        for (case_index = 0; case_index < SILU_CASE_COUNT; case_index = case_index + 1) begin
            @(negedge clk);
            multiplier = silu_case_multiplier[case_index];
            right_shift = silu_case_right_shift[case_index];
            zero_point = silu_case_zero_point[case_index];
            start_valid = 1'b1;
            while (!start_ready) @(posedge clk);
            @(posedge clk);
            @(negedge clk);
            start_valid = 1'b0;
            input_beats = (silu_case_length[case_index] + 7) / 8;
            expected_saturation = 1'b0;
            for (beat = 0; beat < input_beats; beat = beat + 1) begin
                gate_data = silu_gate_beats[case_index*SILU_MAX_INPUT_BEATS + beat];
                up_data = silu_up_beats[case_index*SILU_MAX_INPUT_BEATS + beat];
                lane_count = ((beat + 1) * 8 <= silu_case_length[case_index]) ?
                             4'd8 : silu_case_length[case_index] - beat*8;
                beat_valid = 1'b1;
                while (!beat_ready) @(posedge clk);
                @(posedge clk);
                @(negedge clk);
                beat_valid = 1'b0;
                out_ready = 1'b1;
                while (!out_valid) @(posedge clk);
                expected_index = case_index*SILU_MAX_OUTPUT_BEATS + (beat / 2);
                for (lane = 0; lane < lane_count; lane = lane + 1) begin
                    if (out_data[lane*8 +: 8] !==
                        silu_expected_beats[expected_index][((beat % 2)*8+lane)*8 +: 8]) begin
                        $display("SILU_CORE_MISMATCH case=%0d beat=%0d lane=%0d", case_index, beat, lane);
                        failures = failures + 1;
                    end
                end
                expected_saturation = expected_saturation | saturation_seen;
                @(posedge clk);
                @(negedge clk);
                out_ready = 1'b0;
            end
            if (expected_saturation !== silu_expected_saturation[case_index]) begin
                $display("SILU_CORE_SAT_MISMATCH case=%0d got=%0d expected=%0d",
                         case_index, expected_saturation, silu_expected_saturation[case_index]);
                failures = failures + 1;
            end
            executed_boundary_coverage =
                executed_boundary_coverage | silu_case_boundary_coverage[case_index];
        end
        if (executed_boundary_coverage !== SILU_REQUIRED_BOUNDARY_COVERAGE) begin
            $display("SILU_CORE_BOUNDARY_COVERAGE_MISMATCH got=%02x expected=%02x",
                     executed_boundary_coverage, SILU_REQUIRED_BOUNDARY_COVERAGE);
            failures = failures + 1;
        end
        if (failures != 0) $fatal(1, "ACE2_SILU_GATE_TB_FAIL failures=%0d", failures);
        $display("ACE2_SILU_GATE_TB_PASS cases=%0d boundary_mask=%02x",
                 SILU_CASE_COUNT, executed_boundary_coverage);
        $finish;
    end
endmodule

`default_nettype wire
