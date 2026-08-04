`timescale 1ns/1ps
`default_nettype none

module ace2_v_residual_value_correction_stage_tb;
    localparam logic [31:0] SCALE_ONE = 32'h00008000;

    logic clk_i = 1'b0;
    logic rst_ni = 1'b0;
    logic clear_i = 1'b0;

    logic projection_start_valid, projection_start_ready;
    logic signed [31:0] projection_accumulator, projection_multiplier;
    logic [5:0] projection_shift;
    logic [31:0] projection_baseline_scale, projection_residual_scale;
    logic projection_out_valid, projection_out_ready;
    logic signed [7:0] projection_baseline_v8;
    logic [7:0] projection_residual_u8;
    logic projection_positive_clamp, projection_negative_clamp;
    logic projection_descriptor_error, projection_numeric_overflow;

    logic correction_start_valid, correction_start_ready;
    logic [15:0] correction_lane_count;
    logic signed [31:0] correction_baseline_accumulator;
    logic [31:0] correction_baseline_scale, correction_residual_scale;
    logic correction_sample_valid, correction_sample_ready;
    logic [15:0] correction_probability;
    logic [7:0] correction_residual_u8;
    logic correction_out_valid, correction_out_ready;
    logic signed [31:0] correction_result, correction_value;
    logic signed [63:0] correction_raw;
    logic correction_descriptor_error, correction_numeric_overflow;

    integer cycles;

    always #5 clk_i = ~clk_i;

    ace2_v_residual_projection_core projection_dut (
        .clk_i,
        .rst_ni,
        .clear_i,
        .start_valid_i(projection_start_valid),
        .start_ready_o(projection_start_ready),
        .accumulator_s32_i(projection_accumulator),
        .multiplier_s32_i(projection_multiplier),
        .shift_u6_i(projection_shift),
        .baseline_v_scale32_i(projection_baseline_scale),
        .residual_v_scale32_i(projection_residual_scale),
        .out_valid_o(projection_out_valid),
        .out_ready_i(projection_out_ready),
        .baseline_v8_o(projection_baseline_v8),
        .residual_v_canonical_u8_o(projection_residual_u8),
        .positive_clamp_o(projection_positive_clamp),
        .negative_clamp_o(projection_negative_clamp),
        .descriptor_error_o(projection_descriptor_error),
        .numeric_overflow_o(projection_numeric_overflow)
    );

    ace2_v_residual_value_correction_core correction_dut (
        .clk_i,
        .rst_ni,
        .clear_i,
        .start_valid_i(correction_start_valid),
        .start_ready_o(correction_start_ready),
        .lane_count_u16_i(correction_lane_count),
        .baseline_value_accumulator_s32_i(correction_baseline_accumulator),
        .baseline_v_scale32_i(correction_baseline_scale),
        .residual_v_scale32_i(correction_residual_scale),
        .sample_valid_i(correction_sample_valid),
        .sample_ready_o(correction_sample_ready),
        .probability_q0_15_u16_i(correction_probability),
        .residual_v_canonical_u8_i(correction_residual_u8),
        .out_valid_o(correction_out_valid),
        .out_ready_i(correction_out_ready),
        .corrected_value_accumulator_s32_o(correction_result),
        .correction_baseline_domain_s32_o(correction_value),
        .correction_raw_s64_o(correction_raw),
        .descriptor_error_o(correction_descriptor_error),
        .numeric_overflow_o(correction_numeric_overflow)
    );

    task automatic check_known(input [255:0] phase);
        begin
            if ($isunknown({
                projection_start_ready,
                projection_out_valid,
                projection_baseline_v8,
                projection_residual_u8,
                projection_positive_clamp,
                projection_negative_clamp,
                projection_descriptor_error,
                projection_numeric_overflow,
                correction_start_ready,
                correction_sample_ready,
                correction_out_valid,
                correction_result,
                correction_value,
                correction_raw,
                correction_descriptor_error,
                correction_numeric_overflow
            }))
                $fatal(1, "unknown candidate output during %0s", phase);
        end
    endtask

    task automatic start_projection(input logic [31:0] baseline_scale);
        begin
            @(negedge clk_i);
            projection_accumulator = 32'sd3;
            projection_multiplier = 32'sd5;
            projection_shift = 6'd1;
            projection_baseline_scale = baseline_scale;
            projection_residual_scale = 32'h00ff8000;
            projection_start_valid = 1'b1;
            @(negedge clk_i);
            projection_start_valid = 1'b0;
        end
    endtask

    task automatic start_correction(
        input logic [15:0] lane_count,
        input logic signed [31:0] baseline_accumulator
    );
        begin
            @(negedge clk_i);
            correction_lane_count = lane_count;
            correction_baseline_accumulator = baseline_accumulator;
            correction_baseline_scale = SCALE_ONE;
            correction_residual_scale = SCALE_ONE;
            correction_start_valid = 1'b1;
            @(negedge clk_i);
            correction_start_valid = 1'b0;
        end
    endtask

    task automatic send_sample(
        input logic [15:0] probability,
        input logic [7:0] residual
    );
        begin
            @(negedge clk_i);
            if (!correction_sample_ready)
                $fatal(1, "correction sample interface unexpectedly stalled");
            correction_probability = probability;
            correction_residual_u8 = residual;
            correction_sample_valid = 1'b1;
            @(negedge clk_i);
            correction_sample_valid = 1'b0;
        end
    endtask

    task automatic wait_correction_output;
        begin
            cycles = 0;
            while (!correction_out_valid && cycles < 12) begin
                @(posedge clk_i);
                #1;
                cycles = cycles + 1;
            end
            if (!correction_out_valid)
                $fatal(1, "correction output timeout");
        end
    endtask

    initial begin
        projection_start_valid = 1'b0;
        projection_accumulator = '0;
        projection_multiplier = '0;
        projection_shift = '0;
        projection_baseline_scale = '0;
        projection_residual_scale = '0;
        projection_out_ready = 1'b0;
        correction_start_valid = 1'b0;
        correction_lane_count = '0;
        correction_baseline_accumulator = '0;
        correction_baseline_scale = '0;
        correction_residual_scale = '0;
        correction_sample_valid = 1'b0;
        correction_probability = '0;
        correction_residual_u8 = '0;
        correction_out_ready = 1'b0;

        repeat (3) @(posedge clk_i);
        rst_ni = 1'b1;
        @(posedge clk_i);
        #1;
        check_known("post_reset");

        start_projection(SCALE_ONE);
        if (!projection_out_valid)
            $fatal(1, "projection did not produce a held output");
        repeat (2) begin
            @(posedge clk_i);
            #1;
            if (!projection_out_valid)
                $fatal(1, "projection output did not hold under backpressure");
            check_known("post_backpressure");
        end
        @(negedge clk_i);
        projection_out_ready = 1'b1;
        @(negedge clk_i);
        projection_out_ready = 1'b0;

        start_projection(32'h01008000);
        if (!projection_out_valid || !projection_descriptor_error)
            $fatal(1, "invalid projection Scale32 did not fail closed");
        @(negedge clk_i);
        projection_out_ready = 1'b1;
        @(negedge clk_i);
        projection_out_ready = 1'b0;

        start_correction(16'd1, 32'sd0);
        send_sample(16'd1, 8'h09);
        if (!correction_out_valid || !correction_descriptor_error)
            $fatal(1, "noncanonical residual byte did not raise descriptor error");
        @(negedge clk_i);
        correction_out_ready = 1'b1;
        @(negedge clk_i);
        correction_out_ready = 1'b0;

        start_correction(16'h8001, 32'sd0);
        if (!correction_out_valid || !correction_descriptor_error)
            $fatal(1, "lane-count boundary did not raise descriptor error");
        @(negedge clk_i);
        correction_out_ready = 1'b1;
        @(negedge clk_i);
        correction_out_ready = 1'b0;

        start_correction(16'd1, 32'sh7fffffff);
        send_sample(16'd1, 8'h01);
        wait_correction_output();
        if (!correction_numeric_overflow || correction_descriptor_error ||
            correction_result !== 32'sh7fffffff)
            $fatal(1, "checked corrected-accumulator overflow did not fail closed");
        @(negedge clk_i);
        correction_out_ready = 1'b1;
        @(negedge clk_i);
        correction_out_ready = 1'b0;

        start_correction(16'd2, 32'sd0);
        send_sample(16'd2, 8'h01);
        @(negedge clk_i);
        clear_i = 1'b1;
        @(negedge clk_i);
        clear_i = 1'b0;
        @(posedge clk_i);
        #1;
        if (correction_out_valid || correction_sample_ready)
            $fatal(1, "clear did not cancel the active correction");
        check_known("post_clear");

        start_correction(16'd2, 32'sd0);
        send_sample(16'd2, 8'h01);
        @(negedge clk_i);
        rst_ni = 1'b0;
        @(negedge clk_i);
        rst_ni = 1'b1;
        @(posedge clk_i);
        #1;
        if (projection_out_valid || correction_out_valid || correction_sample_ready)
            $fatal(1, "mid-flight reset did not return both cores to idle");
        check_known("post_midflight_reset");

        $display("XZ_OUTPUT_CHECK_PASS phases=post_reset,post_backpressure,post_clear,post_midflight_reset");
        $display("RESET_MIDFLIGHT_CHECK_PASS modules=projection,correction");
        $display("ERROR_COVERAGE_PASS cases=invalid_scale,noncanonical_s4,lane_count,checked_overflow");
        $display("ACE2_V_RESIDUAL_STAGE_STRESS_PASS");
        $finish;
    end
endmodule

`default_nettype wire
