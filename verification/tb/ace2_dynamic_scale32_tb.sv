`timescale 1ns/1ps
`default_nettype none

module ace2_dynamic_scale32_tb;
    `include "dynamic_scale32_vectors.svh"

    logic clk_i = 1'b0;
    logic rst_ni = 1'b0;
    logic clear_i = 1'b0;

    logic group_start_valid;
    logic group_start_ready;
    logic [7:0] group_lanes;
    logic [31:0] group_base_scale;
    logic [5:0] group_index;
    logic [15:0] group_tag;
    logic group_value_valid;
    logic group_value_ready;
    logic signed [639:0] group_values;
    logic group_payload_valid;
    logic group_payload_ready;
    logic signed [127:0] group_payload;
    logic group_payload_last;
    logic group_commit_valid;
    logic group_commit_ready;
    logic [5:0] group_index_out;
    logic [15:0] group_tag_out;
    logic signed [6:0] group_delta;
    logic [31:0] group_effective_scale;
    logic group_descriptor_error;
    logic group_numeric_overflow;

    logic sidecar_start_valid;
    logic sidecar_start_ready;
    logic [63:0] sidecar_payload_addr;
    logic [7:0] sidecar_group_lanes;
    logic [5:0] sidecar_group_count;
    logic [15:0] sidecar_tag;
    logic [7:0] sidecar_layer;
    logic [7:0] sidecar_opcode;
    logic [31:0] sidecar_elements;
    logic [63:0] sidecar_identity;
    logic sidecar_delta_valid;
    logic sidecar_delta_ready;
    logic signed [6:0] sidecar_delta;
    logic [31:0] sidecar_base_scale;
    logic sidecar_valid;
    logic sidecar_ready;
    logic [511:0] sidecar_value;
    logic sidecar_descriptor_error;
    logic sidecar_numeric_overflow;

    logic validator_start_valid;
    logic validator_start_ready;
    logic [511:0] validator_sidecar;
    logic [1215:0] validator_base_scales;
    logic validator_result_valid;
    logic validator_result_ready;
    logic validator_descriptor_error;
    logic validator_numeric_overflow;

    logic accumulator_start_valid;
    logic accumulator_start_ready;
    logic [15:0] accumulator_start_tag;
    logic accumulator_event_valid;
    logic accumulator_event_ready;
    logic signed [31:0] accumulator_partial;
    logic [31:0] accumulator_scale_a;
    logic [31:0] accumulator_scale_b;
    logic accumulator_event_last;
    logic accumulator_result_valid;
    logic accumulator_result_ready;
    logic signed [159:0] accumulator_result;
    logic signed [7:0] accumulator_exponent;
    logic [15:0] accumulator_result_tag;
    logic accumulator_descriptor_error;
    logic accumulator_numeric_overflow;

    integer lane;
    integer beat;
    integer global_lane;
    integer value_integer;
    integer expected_integer;
    integer cycles;
    logic signed [127:0] held_group_payload;
    logic held_group_payload_last;
    logic [5:0] held_group_index;
    logic [15:0] held_group_tag;
    logic signed [6:0] held_group_delta;
    logic [31:0] held_group_scale;
    logic held_group_descriptor_error;
    logic held_group_numeric_overflow;
    logic [511:0] held_sidecar;
    logic signed [159:0] held_accumulator;

    always #5 clk_i = ~clk_i;

    ace2_dynamic_scale32_group_core group_dut (
        .clk_i,
        .rst_ni,
        .clear_i,
        .group_start_valid_i(group_start_valid),
        .group_start_ready_o(group_start_ready),
        .group_lanes_u8_i(group_lanes),
        .base_scale32_i(group_base_scale),
        .group_index_u6_i(group_index),
        .tensor_tag_u16_i(group_tag),
        .value_valid_i(group_value_valid),
        .value_ready_o(group_value_ready),
        .value_s40_i(group_values),
        .payload_valid_o(group_payload_valid),
        .payload_ready_i(group_payload_ready),
        .payload_s8_o(group_payload),
        .payload_last_o(group_payload_last),
        .group_commit_valid_o(group_commit_valid),
        .group_commit_ready_i(group_commit_ready),
        .group_index_u6_o(group_index_out),
        .tensor_tag_u16_o(group_tag_out),
        .exponent_delta_s7_o(group_delta),
        .effective_scale32_o(group_effective_scale),
        .descriptor_error_o(group_descriptor_error),
        .numeric_overflow_o(group_numeric_overflow)
    );

    ace2_dynamic_scale32_sidecar_builder_core sidecar_builder_dut (
        .clk_i,
        .rst_ni,
        .clear_i,
        .start_valid_i(sidecar_start_valid),
        .start_ready_o(sidecar_start_ready),
        .payload_addr_u64_i(sidecar_payload_addr),
        .group_lanes_u8_i(sidecar_group_lanes),
        .group_count_u6_i(sidecar_group_count),
        .producer_tag_u16_i(sidecar_tag),
        .layer_id_u8_i(sidecar_layer),
        .producer_opcode_u8_i(sidecar_opcode),
        .tensor_elements_u32_i(sidecar_elements),
        .model_identity_u64_i(sidecar_identity),
        .delta_valid_i(sidecar_delta_valid),
        .delta_ready_o(sidecar_delta_ready),
        .exponent_delta_s7_i(sidecar_delta),
        .base_scale32_i(sidecar_base_scale),
        .sidecar_valid_o(sidecar_valid),
        .sidecar_ready_i(sidecar_ready),
        .sidecar_o(sidecar_value),
        .descriptor_error_o(sidecar_descriptor_error),
        .numeric_overflow_o(sidecar_numeric_overflow)
    );

    ace2_dynamic_scale32_sidecar_validator_core validator_dut (
        .clk_i,
        .rst_ni,
        .clear_i,
        .start_valid_i(validator_start_valid),
        .start_ready_o(validator_start_ready),
        .sidecar_i(validator_sidecar),
        .base_scale32_packed_i(validator_base_scales),
        .payload_addr_u64_i(64'h1000),
        .expected_group_lanes_u8_i(8'd64),
        .expected_group_count_u6_i(6'd2),
        .expected_producer_tag_u16_i(16'h1234),
        .expected_layer_id_u8_i(8'd7),
        .expected_producer_opcode_u8_i(8'd1),
        .expected_tensor_elements_u32_i(32'd128),
        .expected_model_identity_u64_i(64'h0123_4567_89ab_cdef),
        .result_valid_o(validator_result_valid),
        .result_ready_i(validator_result_ready),
        .descriptor_error_o(validator_descriptor_error),
        .numeric_overflow_o(validator_numeric_overflow)
    );

    ace2_scale32_tagged_accumulator_core accumulator_dut (
        .clk_i,
        .rst_ni,
        .clear_i,
        .start_valid_i(accumulator_start_valid),
        .start_ready_o(accumulator_start_ready),
        .accumulator_tag_u16_i(accumulator_start_tag),
        .event_valid_i(accumulator_event_valid),
        .event_ready_o(accumulator_event_ready),
        .partial_s32_i(accumulator_partial),
        .scale_a32_i(accumulator_scale_a),
        .scale_b32_i(accumulator_scale_b),
        .event_last_i(accumulator_event_last),
        .result_valid_o(accumulator_result_valid),
        .result_ready_i(accumulator_result_ready),
        .accumulator_s160_o(accumulator_result),
        .canonical_exponent_s8_o(accumulator_exponent),
        .accumulator_tag_u16_o(accumulator_result_tag),
        .descriptor_error_o(accumulator_descriptor_error),
        .numeric_overflow_o(accumulator_numeric_overflow)
    );

    task automatic start_group(
        input logic [7:0] lanes,
        input logic [31:0] scale,
        input logic [5:0] index,
        input logic [15:0] tag
    );
        begin
            while (!group_start_ready)
                @(posedge clk_i);
            @(negedge clk_i);
            group_lanes = lanes;
            group_base_scale = scale;
            group_index = index;
            group_tag = tag;
            group_start_valid = 1'b1;
            @(negedge clk_i);
            group_start_valid = 1'b0;
        end
    endtask

    task automatic send_group_case(input integer case_id, input integer beats);
        begin
            for (beat = 0; beat < beats; beat = beat + 1) begin
                while (!group_value_ready)
                    @(posedge clk_i);
                @(negedge clk_i);
                group_values = '0;
                for (lane = 0; lane < 16; lane = lane + 1) begin
                    global_lane = beat*16 + lane;
                    case (case_id)
                        0: value_integer = 0;
                        1: begin
                            case (global_lane % 5)
                                0: value_integer = -2;
                                1: value_integer = -1;
                                2: value_integer = 0;
                                3: value_integer = 1;
                                default: value_integer = 2;
                            endcase
                        end
                        2: begin
                            if (global_lane == 0) value_integer = 255;
                            else if (global_lane == 1) value_integer = -255;
                            else value_integer = 0;
                        end
                        3: value_integer = global_lane[0] ? 1 : -1;
                        default: value_integer = ((global_lane * 73) % 2001) - 1000;
                    endcase
                    group_values[lane*40 +: 40] = value_integer;
                end
                group_value_valid = 1'b1;
                @(negedge clk_i);
                group_value_valid = 1'b0;
            end
        end
    endtask

    task automatic check_group_case(
        input integer case_id,
        input integer beats,
        input logic signed [6:0] expected_delta,
        input logic [31:0] expected_scale,
        input logic [5:0] expected_index,
        input logic [15:0] expected_tag
    );
        begin
            group_payload_ready = 1'b0;
            for (beat = 0; beat < beats; beat = beat + 1) begin
                cycles = 0;
                while (!group_payload_valid && cycles < 20) begin
                    @(posedge clk_i);
                    cycles = cycles + 1;
                end
                if (!group_payload_valid)
                    $fatal(1, "group payload timeout case=%0d beat=%0d", case_id, beat);
                held_group_payload = group_payload;
                held_group_payload_last = group_payload_last;
                repeat (2) begin
                    @(posedge clk_i);
                    if (!group_payload_valid || group_payload !== held_group_payload ||
                        group_payload_last !== held_group_payload_last)
                        $fatal(1, "group payload changed under backpressure case=%0d beat=%0d",
                               case_id, beat);
                end
                for (lane = 0; lane < 16; lane = lane + 1) begin
                    global_lane = beat*16 + lane;
                    case (case_id)
                        0: expected_integer = 0;
                        1: begin
                            case (global_lane % 5)
                                0: expected_integer = -64;
                                1: expected_integer = -32;
                                2: expected_integer = 0;
                                3: expected_integer = 32;
                                default: expected_integer = 64;
                            endcase
                        end
                        2: begin
                            if (global_lane == 0) expected_integer = 64;
                            else if (global_lane == 1) expected_integer = -64;
                            else expected_integer = 0;
                        end
                        3: expected_integer = global_lane[0] ? 1 : -1;
                        default: expected_integer =
                            $signed(DS32_EXPECTED_MANTISSAS_4[global_lane*8 +: 8]);
                    endcase
                    if ($signed(group_payload[lane*8 +: 8]) !== expected_integer)
                        $fatal(1, "group mismatch case=%0d lane=%0d got=%0d expected=%0d",
                               case_id, global_lane, $signed(group_payload[lane*8 +: 8]), expected_integer);
                    if ($signed(group_payload[lane*8 +: 8]) == -128)
                        $fatal(1, "reserved -128 mantissa emitted");
                end
                if (group_payload_last !== (beat == beats-1))
                    $fatal(1, "group payload last mismatch");
                @(negedge clk_i);
                group_payload_ready = 1'b1;
                @(posedge clk_i);
                @(negedge clk_i);
                group_payload_ready = 1'b0;
            end

            cycles = 0;
            while (!group_commit_valid && cycles < 10) begin
                @(posedge clk_i);
                cycles = cycles + 1;
            end
            if (!group_commit_valid)
                $fatal(1, "group commit timeout");
            if (group_delta !== expected_delta || group_effective_scale !== expected_scale ||
                group_index_out !== expected_index || group_tag_out !== expected_tag ||
                group_descriptor_error || group_numeric_overflow)
                $fatal(1, "group commit mismatch case=%0d delta=%0d scale=%h", case_id, group_delta, group_effective_scale);
            held_group_index = group_index_out;
            held_group_tag = group_tag_out;
            held_group_delta = group_delta;
            held_group_scale = group_effective_scale;
            held_group_descriptor_error = group_descriptor_error;
            held_group_numeric_overflow = group_numeric_overflow;
            repeat (2) begin
                @(posedge clk_i);
                if (!group_commit_valid || group_index_out !== held_group_index ||
                    group_tag_out !== held_group_tag || group_delta !== held_group_delta ||
                    group_effective_scale !== held_group_scale ||
                    group_descriptor_error !== held_group_descriptor_error ||
                    group_numeric_overflow !== held_group_numeric_overflow)
                    $fatal(1, "group commit changed under backpressure case=%0d", case_id);
            end
            @(negedge clk_i);
            group_commit_ready = 1'b1;
            @(negedge clk_i);
            group_commit_ready = 1'b0;
        end
    endtask

    task automatic send_sidecar_delta(input logic signed [6:0] delta);
        begin
            while (!sidecar_delta_ready)
                @(posedge clk_i);
            @(negedge clk_i);
            sidecar_delta = delta;
            sidecar_base_scale = DS32_SCALE_ONE;
            sidecar_delta_valid = 1'b1;
            @(negedge clk_i);
            sidecar_delta_valid = 1'b0;
        end
    endtask

    task automatic send_accumulator_event(input integer partial, input logic last);
        begin
            while (!accumulator_event_ready)
                @(posedge clk_i);
            @(negedge clk_i);
            accumulator_partial = partial;
            accumulator_scale_a = DS32_SCALE_ONE;
            accumulator_scale_b = DS32_SCALE_ONE;
            accumulator_event_last = last;
            accumulator_event_valid = 1'b1;
            @(negedge clk_i);
            accumulator_event_valid = 1'b0;
            accumulator_event_last = 1'b0;
        end
    endtask

    initial begin
        group_start_valid = 1'b0;
        group_lanes = '0;
        group_base_scale = '0;
        group_index = '0;
        group_tag = '0;
        group_value_valid = 1'b0;
        group_values = '0;
        group_payload_ready = 1'b0;
        group_commit_ready = 1'b0;

        sidecar_start_valid = 1'b0;
        sidecar_payload_addr = '0;
        sidecar_group_lanes = '0;
        sidecar_group_count = '0;
        sidecar_tag = '0;
        sidecar_layer = '0;
        sidecar_opcode = '0;
        sidecar_elements = '0;
        sidecar_identity = '0;
        sidecar_delta_valid = 1'b0;
        sidecar_delta = '0;
        sidecar_base_scale = '0;
        sidecar_ready = 1'b0;

        validator_start_valid = 1'b0;
        validator_sidecar = '0;
        validator_base_scales = '0;
        validator_result_ready = 1'b0;

        accumulator_start_valid = 1'b0;
        accumulator_start_tag = '0;
        accumulator_event_valid = 1'b0;
        accumulator_partial = '0;
        accumulator_scale_a = '0;
        accumulator_scale_b = '0;
        accumulator_event_last = 1'b0;
        accumulator_result_ready = 1'b0;

        repeat (3) @(posedge clk_i);
        rst_ni = 1'b1;
        repeat (2) @(posedge clk_i);

        start_group(8'd64, DS32_SCALE_ONE, 6'd0, 16'h1000);
        send_group_case(0, 4);
        check_group_case(0, 4, 7'sd0, DS32_EXPECTED_SCALE_0, 6'd0, 16'h1000);

        start_group(8'd64, DS32_SCALE_ONE, 6'd1, 16'h1001);
        send_group_case(1, 4);
        check_group_case(1, 4, -7'sd5, DS32_EXPECTED_SCALE_1, 6'd1, 16'h1001);

        start_group(8'd64, DS32_SCALE_ONE, 6'd2, 16'h1002);
        send_group_case(2, 4);
        check_group_case(2, 4, 7'sd2, DS32_EXPECTED_SCALE_2, 6'd2, 16'h1002);

        start_group(8'd64, DS32_SCALE_FLOOR, 6'd3, 16'h1003);
        send_group_case(3, 4);
        check_group_case(3, 4, 7'sd0, DS32_EXPECTED_SCALE_3, 6'd3, 16'h1003);

        start_group(8'd128, DS32_SCALE_WIDE, 6'd4, 16'h1004);
        send_group_case(4, 8);
        check_group_case(4, 8, 7'sd3, DS32_EXPECTED_SCALE_4, 6'd4, 16'h1004);

        // Invalid Scale32 fails before accepting payload and publishes no success.
        start_group(8'd64, 32'h0100_8000, 6'd5, 16'h1005);
        cycles = 0;
        while (!group_commit_valid && cycles < 10) begin
            @(posedge clk_i);
            cycles = cycles + 1;
        end
        if (!group_commit_valid || !group_descriptor_error || group_numeric_overflow || group_payload_valid)
            $fatal(1, "invalid Scale32 did not fail closed");
        @(negedge clk_i);
        group_commit_ready = 1'b1;
        @(negedge clk_i);
        group_commit_ready = 1'b0;

        // Group indices outside the frozen [0,37] range fail before payload.
        start_group(8'd64, DS32_SCALE_ONE, 6'd38, 16'h1006);
        cycles = 0;
        while (!group_commit_valid && cycles < 10) begin
            @(posedge clk_i);
            cycles = cycles + 1;
        end
        if (!group_commit_valid || !group_descriptor_error || group_numeric_overflow || group_payload_valid)
            $fatal(1, "out-of-range group index did not fail closed");
        @(negedge clk_i);
        group_commit_ready = 1'b1;
        @(negedge clk_i);
        group_commit_ready = 1'b0;

        // Build and stall one exact sidecar.
        while (!sidecar_start_ready) @(posedge clk_i);
        @(negedge clk_i);
        sidecar_payload_addr = 64'h1000;
        sidecar_group_lanes = 8'd64;
        sidecar_group_count = 6'd2;
        sidecar_tag = 16'h1234;
        sidecar_layer = 8'd7;
        sidecar_opcode = 8'd1;
        sidecar_elements = 32'd128;
        sidecar_identity = 64'h0123_4567_89ab_cdef;
        sidecar_start_valid = 1'b1;
        @(negedge clk_i);
        sidecar_start_valid = 1'b0;
        send_sidecar_delta(-7'sd5);
        send_sidecar_delta(7'sd2);
        while (!sidecar_valid) @(posedge clk_i);
        if (sidecar_value !== DS32_SIDECAR || sidecar_descriptor_error || sidecar_numeric_overflow)
            $fatal(1, "sidecar builder mismatch");
        held_sidecar = sidecar_value;
        repeat (2) begin
            @(posedge clk_i);
            if (!sidecar_valid || sidecar_value !== held_sidecar)
                $fatal(1, "sidecar changed under backpressure");
        end
        @(negedge clk_i);
        sidecar_ready = 1'b1;
        @(negedge clk_i);
        sidecar_ready = 1'b0;

        // Validate the exact sidecar, then reject a corrupt magic byte.
        validator_base_scales[31:0] = DS32_SCALE_ONE;
        validator_base_scales[63:32] = DS32_SCALE_ONE;
        validator_sidecar = DS32_SIDECAR;
        while (!validator_start_ready) @(posedge clk_i);
        @(negedge clk_i);
        validator_start_valid = 1'b1;
        @(negedge clk_i);
        validator_start_valid = 1'b0;
        while (!validator_result_valid) @(posedge clk_i);
        if (validator_descriptor_error || validator_numeric_overflow)
            $fatal(1, "valid sidecar rejected");
        @(negedge clk_i);
        validator_result_ready = 1'b1;
        @(negedge clk_i);
        validator_result_ready = 1'b0;

        validator_sidecar = DS32_SIDECAR ^ 512'h1;
        while (!validator_start_ready) @(posedge clk_i);
        @(negedge clk_i);
        validator_start_valid = 1'b1;
        @(negedge clk_i);
        validator_start_valid = 1'b0;
        while (!validator_result_valid) @(posedge clk_i);
        if (!validator_descriptor_error || validator_numeric_overflow)
            $fatal(1, "corrupt sidecar did not fail closed");
        @(negedge clk_i);
        validator_result_ready = 1'b1;
        @(negedge clk_i);
        validator_result_ready = 1'b0;

        // Two exact Scale32-tagged terms accumulate at canonical exponent -78.
        while (!accumulator_start_ready) @(posedge clk_i);
        @(negedge clk_i);
        accumulator_start_tag = 16'h55aa;
        accumulator_start_valid = 1'b1;
        @(negedge clk_i);
        accumulator_start_valid = 1'b0;
        send_accumulator_event(3, 1'b0);
        send_accumulator_event(-1, 1'b1);
        while (!accumulator_result_valid) @(posedge clk_i);
        if (accumulator_result !== DS32_ACC_EXPECTED || accumulator_exponent !== -8'sd78 ||
            accumulator_result_tag !== 16'h55aa || accumulator_descriptor_error || accumulator_numeric_overflow)
            $fatal(1, "tagged accumulator mismatch got=%h", accumulator_result);
        held_accumulator = accumulator_result;
        repeat (2) begin
            @(posedge clk_i);
            if (!accumulator_result_valid || accumulator_result !== held_accumulator)
                $fatal(1, "accumulator result changed under backpressure");
        end
        @(negedge clk_i);
        accumulator_result_ready = 1'b1;
        @(negedge clk_i);
        accumulator_result_ready = 1'b0;

        // Excess tagged events saturate the bounded counter and fail closed;
        // they cannot wrap and reopen a nominal transaction.
        while (!accumulator_start_ready) @(posedge clk_i);
        @(negedge clk_i);
        accumulator_start_tag = 16'h55ab;
        accumulator_start_valid = 1'b1;
        @(negedge clk_i);
        accumulator_start_valid = 1'b0;
        for (cycles = 0; cycles < 40; cycles = cycles + 1)
            send_accumulator_event(1, cycles == 39);
        while (!accumulator_result_valid) @(posedge clk_i);
        if (!accumulator_descriptor_error || accumulator_numeric_overflow)
            $fatal(1, "excess tagged events did not fail closed");
        @(negedge clk_i);
        accumulator_result_ready = 1'b1;
        @(negedge clk_i);
        accumulator_result_ready = 1'b0;

        if (((group_start_ready !== 1'b0) && (group_start_ready !== 1'b1)) ||
            ((group_value_ready !== 1'b0) && (group_value_ready !== 1'b1)) ||
            ((group_payload_valid !== 1'b0) && (group_payload_valid !== 1'b1)) ||
            ((group_commit_valid !== 1'b0) && (group_commit_valid !== 1'b1)) ||
            ((sidecar_start_ready !== 1'b0) && (sidecar_start_ready !== 1'b1)) ||
            ((sidecar_delta_ready !== 1'b0) && (sidecar_delta_ready !== 1'b1)) ||
            ((sidecar_valid !== 1'b0) && (sidecar_valid !== 1'b1)) ||
            ((validator_start_ready !== 1'b0) && (validator_start_ready !== 1'b1)) ||
            ((validator_result_valid !== 1'b0) && (validator_result_valid !== 1'b1)) ||
            ((accumulator_start_ready !== 1'b0) && (accumulator_start_ready !== 1'b1)) ||
            ((accumulator_event_ready !== 1'b0) && (accumulator_event_ready !== 1'b1)) ||
            ((accumulator_result_valid !== 1'b0) && (accumulator_result_valid !== 1'b1))) begin
            $fatal(1, "unknown control output after reset release");
        end

        $display("ACE2_DYNAMIC_SCALE32_RTL_PASS groups=5 lanes=64,128 sidecar=pass accumulator=pass group_index_bound=pass counter_bound=pass stalls=pass errors=pass");
        $finish;
    end
endmodule

`default_nettype wire
