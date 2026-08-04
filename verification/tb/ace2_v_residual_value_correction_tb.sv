`timescale 1ns/1ps
`default_nettype none

module ace2_v_residual_value_correction_tb;
    `include "v_residual_value_correction_vectors.svh"

    logic clk_i = 1'b0;
    logic rst_ni = 1'b0;
    logic clear_i = 1'b0;

    logic projection_start_valid;
    logic projection_start_ready;
    logic signed [31:0] projection_accumulator;
    logic signed [31:0] projection_multiplier;
    logic [5:0] projection_shift;
    logic [31:0] projection_baseline_scale, projection_residual_scale;
    logic projection_out_valid, projection_out_ready;
    logic signed [7:0] projection_baseline_v8;
    logic [7:0] projection_residual_u8;
    logic projection_positive_clamp, projection_negative_clamp;
    logic projection_descriptor_error, projection_numeric_overflow;

    logic correction_start_valid;
    logic correction_start_ready;
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

    integer index;

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

    task automatic run_projection_case(
        input logic signed [31:0] accumulator,
        input logic signed [31:0] multiplier,
        input logic [5:0] shift,
        input logic [31:0] baseline_scale,
        input logic [31:0] residual_scale,
        input logic signed [7:0] expected_baseline,
        input logic [7:0] expected_residual,
        input logic expected_positive_clamp,
        input logic expected_negative_clamp
    );
        begin
            @(negedge clk_i);
            projection_accumulator = accumulator;
            projection_multiplier = multiplier;
            projection_shift = shift;
            projection_baseline_scale = baseline_scale;
            projection_residual_scale = residual_scale;
            projection_start_valid = 1'b1;
            @(negedge clk_i);
            projection_start_valid = 1'b0;
            if (!projection_out_valid)
                $fatal(1, "projection result did not assert in one lane cycle");
            if (projection_baseline_v8 !== expected_baseline ||
                projection_residual_u8 !== expected_residual ||
                projection_positive_clamp !== expected_positive_clamp ||
                projection_negative_clamp !== expected_negative_clamp ||
                projection_descriptor_error || projection_numeric_overflow)
                $fatal(1, "projection vector mismatch");
            @(posedge clk_i);
            #1;
            if (!projection_out_valid)
                $fatal(1, "projection output did not hold under backpressure");
            @(negedge clk_i);
            projection_out_ready = 1'b1;
            @(negedge clk_i);
            projection_out_ready = 1'b0;
        end
    endtask

    task automatic start_correction(
        input logic [15:0] lane_count,
        input logic signed [31:0] baseline_accumulator,
        input logic [31:0] baseline_scale,
        input logic [31:0] residual_scale
    );
        begin
            @(negedge clk_i);
            correction_lane_count = lane_count;
            correction_baseline_accumulator = baseline_accumulator;
            correction_baseline_scale = baseline_scale;
            correction_residual_scale = residual_scale;
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
                $fatal(1, "correction sample interface was not ready");
            correction_probability = probability;
            correction_residual_u8 = residual;
            correction_sample_valid = 1'b1;
            @(negedge clk_i);
            correction_sample_valid = 1'b0;
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

        run_projection_case(
            VRP_0_ACC, VRP_0_MUL, VRP_0_SHIFT, VRP_0_BASE_SCALE,
            VRP_0_RES_SCALE, VRP_0_BASE_V8, VRP_0_RESIDUAL_U8,
            VRP_0_POS_CLAMP, VRP_0_NEG_CLAMP
        );
        run_projection_case(
            VRP_1_ACC, VRP_1_MUL, VRP_1_SHIFT, VRP_1_BASE_SCALE,
            VRP_1_RES_SCALE, VRP_1_BASE_V8, VRP_1_RESIDUAL_U8,
            VRP_1_POS_CLAMP, VRP_1_NEG_CLAMP
        );
        run_projection_case(
            VRP_2_ACC, VRP_2_MUL, VRP_2_SHIFT, VRP_2_BASE_SCALE,
            VRP_2_RES_SCALE, VRP_2_BASE_V8, VRP_2_RESIDUAL_U8,
            VRP_2_POS_CLAMP, VRP_2_NEG_CLAMP
        );
        run_projection_case(
            VRP_3_ACC, VRP_3_MUL, VRP_3_SHIFT, VRP_3_BASE_SCALE,
            VRP_3_RES_SCALE, VRP_3_BASE_V8, VRP_3_RESIDUAL_U8,
            VRP_3_POS_CLAMP, VRP_3_NEG_CLAMP
        );

        start_correction(VRC_LANE_COUNT, VRC_BASE_ACC, VRC_BASE_SCALE, VRC_RES_SCALE);
        send_sample(VRC_P_0, VRC_R_0);
        send_sample(VRC_P_1, VRC_R_1);
        send_sample(VRC_P_2, VRC_R_2);
        send_sample(VRC_P_3, VRC_R_3);
        send_sample(VRC_P_4, VRC_R_4);
        for (index = 0; index < 7; index = index + 1) begin
            @(posedge clk_i);
            #1;
            if (correction_out_valid)
                $fatal(1, "correction converter completed before eight cycles");
        end
        @(posedge clk_i);
        #1;
        if (!correction_out_valid)
            $fatal(1, "correction converter did not complete in eight cycles");
        if (correction_raw !== VRC_CORR_RAW || correction_value !== VRC_CORR ||
            correction_result !== VRC_RESULT || correction_descriptor_error ||
            correction_numeric_overflow)
            $fatal(1, "correction vector mismatch");
        @(posedge clk_i);
        #1;
        if (!correction_out_valid)
            $fatal(1, "correction output did not hold under backpressure");
        @(negedge clk_i);
        correction_out_ready = 1'b1;
        @(negedge clk_i);
        correction_out_ready = 1'b0;

        start_correction(16'd1, 32'sd0, VRC_BASE_SCALE, VRC_RES_SCALE);
        send_sample(16'd1, 8'hf8);
        if (!correction_out_valid || !correction_descriptor_error ||
            correction_numeric_overflow)
            $fatal(1, "reserved signed-4 code did not raise descriptor error");
        @(negedge clk_i);
        correction_out_ready = 1'b1;
        @(negedge clk_i);
        correction_out_ready = 1'b0;

        start_correction(16'd0, 32'sd0, VRC_BASE_SCALE, VRC_RES_SCALE);
        if (!correction_out_valid || !correction_descriptor_error)
            $fatal(1, "zero lane count did not raise descriptor error");
        @(negedge clk_i);
        correction_out_ready = 1'b1;
        @(negedge clk_i);
        correction_out_ready = 1'b0;

        start_correction(16'd2, 32'sd0, VRC_BASE_SCALE, VRC_RES_SCALE);
        send_sample(16'd2, 8'h01);
        @(negedge clk_i);
        clear_i = 1'b1;
        @(negedge clk_i);
        clear_i = 1'b0;
        if (correction_out_valid || correction_sample_ready)
            $fatal(1, "clear did not return correction core to idle");

        @(negedge clk_i);
        rst_ni = 1'b0;
        @(negedge clk_i);
        rst_ni = 1'b1;
        if (projection_out_valid || correction_out_valid)
            $fatal(1, "reset did not clear candidate outputs");

        $display("ACE2_V_RESIDUAL_RTL_PASS projection_cases=4 correction_cycles=8");
        $finish;
    end
endmodule

`default_nettype wire
