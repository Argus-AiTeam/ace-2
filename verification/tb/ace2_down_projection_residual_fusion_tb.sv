`timescale 1ns/1ps
`default_nettype none

module ace2_down_projection_residual_fusion_tb;
    logic clk_i = 1'b0;
    logic rst_ni = 1'b0;
    logic clear_i = 1'b0;
    logic start_valid_i = 1'b0;
    logic start_ready_o;
    logic signed [31:0] accumulator_s32_i = '0;
    logic signed [7:0] residual_s8_i = '0;
    logic [31:0] accumulator_scale32_i = '0;
    logic [31:0] residual_scale32_i = '0;
    logic [31:0] destination_scale32_i = '0;
    logic out_valid_o;
    logic out_ready_i = 1'b0;
    logic signed [7:0] fused_s8_o;
    logic positive_saturation_o, negative_saturation_o;
    logic descriptor_error_o, numeric_overflow_o;
    logic signed [95:0] numerator_s96_o;
    logic [63:0] denominator_u64_o;
    logic signed [7:0] common_exponent_s8_o;
    logic [4:0] latency_cycles_u5_o;

    integer executed_cases = 0;
    integer cycles;

    always #5 clk_i = ~clk_i;

    ace2_down_projection_residual_fusion_core dut (.*);

    task automatic run_case(
        input string name,
        input logic signed [31:0] accumulator,
        input logic signed [7:0] residual,
        input logic [31:0] accumulator_scale,
        input logic [31:0] residual_scale,
        input logic [31:0] destination_scale,
        input logic signed [7:0] expected_output,
        input logic signed [95:0] expected_numerator,
        input logic [63:0] expected_denominator,
        input logic signed [7:0] expected_common_exponent,
        input logic expected_positive_saturation,
        input logic expected_negative_saturation,
        input logic expected_descriptor_error,
        input logic expected_numeric_overflow,
        input logic [4:0] expected_latency
    );
        logic signed [7:0] held_output;
        begin
            @(negedge clk_i);
            if (!start_ready_o)
                $fatal(1, "%s: start interface was not ready", name);
            accumulator_s32_i = accumulator;
            residual_s8_i = residual;
            accumulator_scale32_i = accumulator_scale;
            residual_scale32_i = residual_scale;
            destination_scale32_i = destination_scale;
            start_valid_i = 1'b1;
            @(posedge clk_i);
            #1;
            cycles = 0;
            @(negedge clk_i);
            start_valid_i = 1'b0;
            while (!out_valid_o) begin
                @(posedge clk_i);
                #1;
                cycles = cycles + 1;
                if (cycles > 12)
                    $fatal(1, "%s: timed out", name);
            end
            if (cycles !== expected_latency)
                $fatal(1, "%s: latency=%0d expected=%0d", name, cycles, expected_latency);
            if (fused_s8_o !== expected_output ||
                numerator_s96_o !== expected_numerator ||
                denominator_u64_o !== expected_denominator ||
                common_exponent_s8_o !== expected_common_exponent ||
                positive_saturation_o !== expected_positive_saturation ||
                negative_saturation_o !== expected_negative_saturation ||
                descriptor_error_o !== expected_descriptor_error ||
                numeric_overflow_o !== expected_numeric_overflow ||
                latency_cycles_u5_o !== expected_latency)
                $fatal(1, "%s: output mismatch", name);
            held_output = fused_s8_o;
            @(posedge clk_i);
            #1;
            if (!out_valid_o || fused_s8_o !== held_output)
                $fatal(1, "%s: output did not hold under backpressure", name);
            @(negedge clk_i);
            out_ready_i = 1'b1;
            @(posedge clk_i);
            #1;
            @(negedge clk_i);
            out_ready_i = 1'b0;
            executed_cases = executed_cases + 1;
        end
    endtask

    initial begin
        repeat (3) @(posedge clk_i);
        rst_ni = 1'b1;

        `include "down_projection_residual_fusion_vectors.svh"

        @(negedge clk_i);
        accumulator_s32_i = 32'sd3;
        residual_s8_i = 8'sd0;
        accumulator_scale32_i = 32'h00ff8000;
        residual_scale32_i = 32'h00008000;
        destination_scale32_i = 32'h00008000;
        start_valid_i = 1'b1;
        @(posedge clk_i);
        #1;
        @(negedge clk_i);
        start_valid_i = 1'b0;
        if (start_ready_o)
            $fatal(1, "busy core incorrectly accepted a second start");
        clear_i = 1'b1;
        @(posedge clk_i);
        #1;
        @(negedge clk_i);
        clear_i = 1'b0;
        #1;
        if (out_valid_o || !start_ready_o)
            $fatal(1, "clear did not restore idle state");

        @(negedge clk_i);
        start_valid_i = 1'b1;
        @(posedge clk_i);
        #1;
        @(negedge clk_i);
        start_valid_i = 1'b0;
        rst_ni = 1'b0;
        #1;
        if (out_valid_o || start_ready_o)
            $fatal(1, "asynchronous reset did not clear active state");
        @(negedge clk_i);
        rst_ni = 1'b1;
        @(posedge clk_i);
        #1;
        if (!start_ready_o)
            $fatal(1, "core did not return ready after reset release");

        $display("ACE2_DOWN_PROJECTION_RESIDUAL_FUSION_RTL_PASS cases=%0d valid_latency=10 divide_bits=8", executed_cases);
        $finish;
    end
endmodule

`default_nettype wire
