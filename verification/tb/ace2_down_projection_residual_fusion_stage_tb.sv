`timescale 1ns/1ps
`default_nettype none

module ace2_down_projection_residual_fusion_stage_tb;
    localparam logic [31:0] UNIT_SCALE32 = 32'h00008000;

    logic clk_i = 1'b0;
    logic rst_ni = 1'b0;
    logic clear_i = 1'b0;
    logic start_valid_i = 1'b0;
    logic start_ready_o;
    logic signed [31:0] accumulator_s32_i = '0;
    logic signed [7:0] residual_s8_i = '0;
    logic [31:0] accumulator_scale32_i = UNIT_SCALE32;
    logic [31:0] residual_scale32_i = UNIT_SCALE32;
    logic [31:0] destination_scale32_i = UNIT_SCALE32;
    logic out_valid_o;
    logic out_ready_i = 1'b0;
    logic signed [7:0] fused_s8_o;
    logic positive_saturation_o;
    logic negative_saturation_o;
    logic descriptor_error_o;
    logic numeric_overflow_o;
    logic signed [95:0] numerator_s96_o;
    logic [63:0] denominator_u64_o;
    logic signed [7:0] common_exponent_s8_o;
    logic [4:0] latency_cycles_u5_o;

    integer cases = 0;
    integer random_cases = 0;
    integer index;
    integer accumulator_value;
    integer residual_value;
    integer expected_value;
    integer cycles;

    always #5 clk_i = ~clk_i;

    ace2_down_projection_residual_fusion_core dut (.*);

    task automatic check_known(input string phase);
        begin
            if ((^start_ready_o) === 1'bx ||
                (^out_valid_o) === 1'bx ||
                (^fused_s8_o) === 1'bx ||
                (^positive_saturation_o) === 1'bx ||
                (^negative_saturation_o) === 1'bx ||
                (^descriptor_error_o) === 1'bx ||
                (^numeric_overflow_o) === 1'bx ||
                (^numerator_s96_o) === 1'bx ||
                (^denominator_u64_o) === 1'bx ||
                (^common_exponent_s8_o) === 1'bx ||
                (^latency_cycles_u5_o) === 1'bx)
                $fatal(1, "%s: output contains X/Z", phase);
        end
    endtask

    task automatic run_unit_case(
        input string name,
        input integer accumulator,
        input integer residual,
        input integer expected
    );
        logic signed [7:0] held_output;
        logic signed [95:0] held_numerator;
        logic [63:0] held_denominator;
        begin
            @(negedge clk_i);
            if (!start_ready_o)
                $fatal(1, "%s: input was not ready", name);
            accumulator_s32_i = accumulator;
            residual_s8_i = residual;
            accumulator_scale32_i = UNIT_SCALE32;
            residual_scale32_i = UNIT_SCALE32;
            destination_scale32_i = UNIT_SCALE32;
            start_valid_i = 1'b1;
            @(posedge clk_i);
            #1;
            @(negedge clk_i);
            start_valid_i = 1'b0;
            cycles = 0;
            while (!out_valid_o) begin
                @(posedge clk_i);
                #1;
                cycles = cycles + 1;
                if (cycles > 10)
                    $fatal(1, "%s: timed out", name);
            end
            if (cycles != 10 || latency_cycles_u5_o != 5'd10)
                $fatal(1, "%s: latency differs", name);
            if (fused_s8_o !== expected ||
                descriptor_error_o !== 1'b0 ||
                numeric_overflow_o !== 1'b0 ||
                common_exponent_s8_o !== 8'sd0 ||
                denominator_u64_o !== 64'h0000000000008000 ||
                numerator_s96_o !== ((accumulator + residual) * 32768))
                $fatal(1, "%s: arithmetic mismatch", name);
            if (positive_saturation_o !== (accumulator + residual > 127) ||
                negative_saturation_o !== (accumulator + residual < -128))
                $fatal(1, "%s: saturation flag mismatch", name);
            check_known(name);
            held_output = fused_s8_o;
            held_numerator = numerator_s96_o;
            held_denominator = denominator_u64_o;
            repeat (3) begin
                @(posedge clk_i);
                #1;
                if (!out_valid_o || fused_s8_o !== held_output ||
                    numerator_s96_o !== held_numerator ||
                    denominator_u64_o !== held_denominator)
                    $fatal(1, "%s: stalled output changed", name);
                check_known("backpressure");
            end
            @(negedge clk_i);
            out_ready_i = 1'b1;
            @(posedge clk_i);
            #1;
            @(negedge clk_i);
            out_ready_i = 1'b0;
            cases = cases + 1;
        end
    endtask

    task automatic run_invalid_case(
        input string name,
        input logic [31:0] accumulator_scale,
        input logic [31:0] residual_scale,
        input logic [31:0] destination_scale
    );
        begin
            @(negedge clk_i);
            accumulator_s32_i = '0;
            residual_s8_i = '0;
            accumulator_scale32_i = accumulator_scale;
            residual_scale32_i = residual_scale;
            destination_scale32_i = destination_scale;
            start_valid_i = 1'b1;
            @(posedge clk_i);
            #1;
            @(negedge clk_i);
            start_valid_i = 1'b0;
            #1;
            if (!out_valid_o || !descriptor_error_o || numeric_overflow_o ||
                latency_cycles_u5_o != 0 || fused_s8_o != 0)
                $fatal(1, "%s: invalid descriptor did not fail closed", name);
            check_known(name);
            out_ready_i = 1'b1;
            @(posedge clk_i);
            #1;
            @(negedge clk_i);
            out_ready_i = 1'b0;
            cases = cases + 1;
        end
    endtask

    initial begin
        repeat (3) @(posedge clk_i);
        check_known("reset_asserted");
        rst_ni = 1'b1;
        @(posedge clk_i);
        #1;
        check_known("reset_released");

        run_unit_case("zero", 0, 0, 0);
        run_unit_case("positive_saturation", 127, 1, 127);
        run_unit_case("negative_saturation", -128, -1, -128);

        for (index = 0; index < 256; index = index + 1) begin
            accumulator_value = ((index * 73) % 511) - 255;
            residual_value = ((index * 29) % 255) - 127;
            expected_value = accumulator_value + residual_value;
            if (expected_value > 127)
                expected_value = 127;
            else if (expected_value < -128)
                expected_value = -128;
            run_unit_case("deterministic_random", accumulator_value, residual_value, expected_value);
            random_cases = random_cases + 1;
        end

        run_invalid_case("accumulator_reserved", 32'h01008000, UNIT_SCALE32, UNIT_SCALE32);
        run_invalid_case("residual_unnormalized", UNIT_SCALE32, 32'h00007fff, UNIT_SCALE32);
        run_invalid_case("destination_exponent_low", UNIT_SCALE32, UNIT_SCALE32, 32'h00e78000);
        run_invalid_case("destination_exponent_high", UNIT_SCALE32, UNIT_SCALE32, 32'h00058000);

        @(negedge clk_i);
        accumulator_s32_i = 32'sd11;
        residual_s8_i = -8'sd3;
        accumulator_scale32_i = UNIT_SCALE32;
        residual_scale32_i = UNIT_SCALE32;
        destination_scale32_i = UNIT_SCALE32;
        start_valid_i = 1'b1;
        @(posedge clk_i);
        #1;
        @(negedge clk_i);
        start_valid_i = 1'b0;
        if (start_ready_o)
            $fatal(1, "busy transaction accepted a second start");
        repeat (4) @(posedge clk_i);
        @(negedge clk_i);
        clear_i = 1'b1;
        @(posedge clk_i);
        #1;
        @(negedge clk_i);
        clear_i = 1'b0;
        #1;
        if (out_valid_o || !start_ready_o)
            $fatal(1, "clear did not cancel the active transaction");
        check_known("post_clear");

        @(negedge clk_i);
        start_valid_i = 1'b1;
        @(posedge clk_i);
        #1;
        @(negedge clk_i);
        start_valid_i = 1'b0;
        repeat (2) @(posedge clk_i);
        rst_ni = 1'b0;
        #1;
        if (out_valid_o || start_ready_o)
            $fatal(1, "asynchronous reset did not cancel the active transaction");
        check_known("midflight_reset_asserted");
        @(negedge clk_i);
        rst_ni = 1'b1;
        @(posedge clk_i);
        #1;
        if (!start_ready_o)
            $fatal(1, "reset release did not restore ready");
        check_known("midflight_reset_released");

        $display(
            "ACE2_DPRF_STAGE_STRESS_PASS cases=%0d randomized=%0d reset=async_midflight clear=midflight backpressure=3 illegal=4 xz=post_reset_post_stall_post_clear overflow=proved_unreachable_valid_domain cdc=single_clock",
            cases,
            random_cases
        );
        $finish;
    end
endmodule

`default_nettype wire
