`timescale 1ns/1ps
`default_nettype none
module ace2_dynamic_rope_head_tb;
    reg clk_i=0, rst_ni=1, clear_i=0, start_valid_i=0, out_ready_i=0;
    reg [511:0] act_data_i=0;
    reg [31:0] producer_scale32_i=0;
    reg [1023:0] cos_q15_i=0, sin_q15_i=0;
    wire start_ready_o, out_valid_o, error_valid_o, numeric_overflow_o;
    wire [511:0] out_data_o;
    wire [31:0] output_scale32_o;
    integer case_index, cycle_count=0, accept_cycle, latency;
    reg [31:0] case_producer_scale32, case_expected_scale32;
    reg [511:0] case_activations, case_expected_outputs, held_output;
    reg [1023:0] case_cos_q15, case_sin_q15;
`include "dynamic_rope_vectors.svh"

    ace2_dynamic_rope_head_core dut(.*);
    always #5 clk_i=~clk_i;
    always @(posedge clk_i) cycle_count=cycle_count+1;

    task reset_dut; begin
        rst_ni=0; repeat(2) @(posedge clk_i); @(negedge clk_i); rst_ni=1; repeat(2) @(posedge clk_i);
    end endtask
    task start_case; input integer selected; begin
        load_dynamic_rope_case(selected,case_producer_scale32,case_activations,case_cos_q15,case_sin_q15,case_expected_outputs,case_expected_scale32);
        while(!start_ready_o) @(posedge clk_i);
        @(negedge clk_i); act_data_i=case_activations; producer_scale32_i=case_producer_scale32; cos_q15_i=case_cos_q15; sin_q15_i=case_sin_q15; start_valid_i=1;
        @(posedge clk_i); #1 accept_cycle=cycle_count; @(negedge clk_i); start_valid_i=0;
    end endtask
    task run_case; input integer selected; begin
        out_ready_i=0; start_case(selected); wait(out_valid_o || error_valid_o); #1;
        if(error_valid_o) $fatal(1,"case %0d unexpected error overflow=%0d",selected,numeric_overflow_o);
        latency=cycle_count-accept_cycle; if(latency!=54) $fatal(1,"case %0d latency=%0d",selected,latency);
        if(out_data_o!==case_expected_outputs) $fatal(1,"case %0d payload mismatch",selected);
        if(output_scale32_o!==case_expected_scale32) $fatal(1,"case %0d scale mismatch got=%08x expected=%08x",selected,output_scale32_o,case_expected_scale32);
        held_output=out_data_o; repeat(3) begin @(posedge clk_i); #1; if(!out_valid_o || out_data_o!==held_output) $fatal(1,"backpressure instability"); end
        @(negedge clk_i); out_ready_i=1; @(posedge clk_i); @(negedge clk_i); out_ready_i=0;
        if(!start_ready_o) $fatal(1,"did not return idle");
    end endtask

    initial begin
        reset_dut();
        for(case_index=0;case_index<DYNAMIC_ROPE_CASE_COUNT;case_index=case_index+1) run_case(case_index);
        load_dynamic_rope_case(1,case_producer_scale32,case_activations,case_cos_q15,case_sin_q15,case_expected_outputs,case_expected_scale32);
        @(negedge clk_i); act_data_i=case_activations; producer_scale32_i=case_producer_scale32|32'h01000000; cos_q15_i=case_cos_q15; sin_q15_i=case_sin_q15; start_valid_i=1;
        @(posedge clk_i); @(negedge clk_i); start_valid_i=0;
        if(!error_valid_o || numeric_overflow_o || out_valid_o) $fatal(1,"invalid producer handling failed");
        clear_i=1; @(posedge clk_i); @(negedge clk_i); clear_i=0;
        start_case(3); repeat(7) @(posedge clk_i); @(negedge clk_i); rst_ni=0; @(posedge clk_i); @(negedge clk_i); rst_ni=1; repeat(2) @(posedge clk_i);
        if(!start_ready_o || out_valid_o || error_valid_o) $fatal(1,"busy reset cancellation failed");
        $display("PASS dynamic_rope_head_scale_v1 cases=%0d latency=54 reset=pass backpressure=pass",DYNAMIC_ROPE_CASE_COUNT); $finish;
    end
endmodule
`default_nettype wire
