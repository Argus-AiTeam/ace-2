`timescale 1ns/1ps
`default_nettype none

module ace2_cross_layer_error_carry_tb;
    `include "cross_layer_error_carry_vectors.svh"

    logic clk = 1'b0;
    logic rst_ni = 1'b0;
    logic clear_i = 1'b0;
    always #5 clk = ~clk;

    logic lane_start_valid, lane_start_ready;
    logic signed [31:0] lane_accumulator;
    logic signed [7:0] lane_residual;
    logic [31:0] lane_acc_scale, lane_res_scale, lane_dst_scale;
    logic lane_out_valid, lane_out_ready;
    logic signed [7:0] lane_hidden;
    logic signed [15:0] lane_carry;
    logic lane_descriptor_error, lane_numeric_overflow;
    logic signed [95:0] lane_numerator;
    logic [63:0] lane_denominator;
    logic signed [7:0] lane_common_exponent;
    logic [5:0] lane_latency;

    ace2_quantization_error_carry_lane_core lane_dut (
        .clk_i(clk), .rst_ni(rst_ni), .clear_i(clear_i),
        .start_valid_i(lane_start_valid), .start_ready_o(lane_start_ready),
        .accumulator_s32_i(lane_accumulator), .residual_s8_i(lane_residual),
        .accumulator_scale32_i(lane_acc_scale), .residual_scale32_i(lane_res_scale),
        .destination_scale32_i(lane_dst_scale), .out_valid_o(lane_out_valid),
        .out_ready_i(lane_out_ready), .hidden_s8_o(lane_hidden),
        .carry_s16_q15_o(lane_carry), .descriptor_error_o(lane_descriptor_error),
        .numeric_overflow_o(lane_numeric_overflow), .numerator_s96_o(lane_numerator),
        .denominator_u64_o(lane_denominator), .common_exponent_s8_o(lane_common_exponent),
        .latency_cycles_u6_o(lane_latency)
    );

    logic state_metadata_valid;
    logic producer_start_valid, producer_start_ready;
    logic [4:0] producer_layer;
    logic [31:0] producer_token;
    logic [63:0] producer_model;
    logic [15:0] producer_tag;
    logic producer_lane_valid, producer_lane_ready;
    logic signed [15:0] producer_lane_carry;
    logic producer_done_valid, producer_done_ready;
    logic producer_done_descriptor_error, producer_done_numeric_overflow;
    logic consumer_start_valid, consumer_start_ready;
    logic [4:0] consumer_layer;
    logic [31:0] consumer_token;
    logic [63:0] consumer_model;
    logic [15:0] consumer_tag;
    logic consumer_active;
    logic [2:0] consumer_read_index;
    logic signed [15:0] consumer_read_carry;
    logic consumer_read_index_valid;
    logic consumer_finish_valid, consumer_finish_success;
    logic consumer_done_valid, consumer_done_ready, consumer_done_descriptor_error;
    logic [15:0] consumer_completion_tag;
    logic carry_valid;
    logic [4:0] carry_layer;
    logic [31:0] carry_token;
    logic [63:0] carry_model;
    logic [15:0] carry_tag;

    ace2_error_carry_state_core #(.HIDDEN_SIZE(5)) state_dut (
        .clk_i(clk), .rst_ni(rst_ni), .clear_i(clear_i), .metadata_valid_i(state_metadata_valid),
        .producer_start_valid_i(producer_start_valid), .producer_start_ready_o(producer_start_ready),
        .producer_layer_id_i(producer_layer), .producer_token_id_i(producer_token),
        .producer_model_id_i(producer_model), .producer_completion_tag_i(producer_tag),
        .producer_lane_valid_i(producer_lane_valid), .producer_lane_ready_o(producer_lane_ready),
        .producer_carry_s16_q15_i(producer_lane_carry), .producer_done_valid_o(producer_done_valid),
        .producer_done_ready_i(producer_done_ready),
        .producer_done_descriptor_error_o(producer_done_descriptor_error),
        .producer_done_numeric_overflow_o(producer_done_numeric_overflow),
        .consumer_start_valid_i(consumer_start_valid), .consumer_start_ready_o(consumer_start_ready),
        .consumer_layer_id_i(consumer_layer), .consumer_token_id_i(consumer_token),
        .consumer_model_id_i(consumer_model), .consumer_completion_tag_i(consumer_tag),
        .consumer_active_o(consumer_active), .consumer_read_index_i(consumer_read_index),
        .consumer_carry_s16_q15_o(consumer_read_carry),
        .consumer_read_index_valid_o(consumer_read_index_valid),
        .consumer_finish_valid_i(consumer_finish_valid),
        .consumer_finish_success_i(consumer_finish_success),
        .consumer_done_valid_o(consumer_done_valid), .consumer_done_ready_i(consumer_done_ready),
        .consumer_done_descriptor_error_o(consumer_done_descriptor_error),
        .consumer_completion_tag_o(consumer_completion_tag),
        .carry_valid_o(carry_valid), .carry_producer_layer_id_o(carry_layer),
        .carry_token_id_o(carry_token), .carry_model_id_o(carry_model),
        .carry_producer_completion_tag_o(carry_tag)
    );

    logic rms_start_valid, rms_start_ready;
    logic rms_square_valid, rms_square_ready;
    logic signed [7:0] rms_square_hidden;
    logic signed [15:0] rms_square_carry;
    logic rms_scale_valid, rms_scale_ready;
    logic signed [7:0] rms_scale_hidden;
    logic signed [15:0] rms_scale_carry, rms_scale_scaled_gain;
    logic rms_out_valid, rms_out_ready;
    logic signed [7:0] rms_out_data;
    logic rms_done_valid, rms_done_ready, rms_done_numeric_overflow, rms_saturation;
    logic [55:0] rms_sumsq, rms_mean;
    logic [23:0] rms_root;
    logic [45:0] rms_inverse;

    ace2_carry_aware_rmsnorm_core #(.HIDDEN_SIZE(4)) rms_dut (
        .clk_i(clk), .rst_ni(rst_ni), .clear_i(clear_i),
        .start_valid_i(rms_start_valid), .start_ready_o(rms_start_ready),
        .square_valid_i(rms_square_valid), .square_ready_o(rms_square_ready),
        .square_hidden_s8_i(rms_square_hidden), .square_carry_s16_q15_i(rms_square_carry),
        .scale_valid_i(rms_scale_valid), .scale_ready_o(rms_scale_ready),
        .scale_hidden_s8_i(rms_scale_hidden), .scale_carry_s16_q15_i(rms_scale_carry),
        .scale_scaled_gain_s16_q8_i(rms_scale_scaled_gain), .out_valid_o(rms_out_valid),
        .out_ready_i(rms_out_ready), .out_data_s8_o(rms_out_data),
        .done_valid_o(rms_done_valid), .done_ready_i(rms_done_ready),
        .done_numeric_overflow_o(rms_done_numeric_overflow),
        .saturation_seen_o(rms_saturation), .sum_squares_q30_o(rms_sumsq),
        .mean_square_q30_o(rms_mean), .root_q15_o(rms_root), .inverse_q30_o(rms_inverse)
    );

    task automatic lane_transaction(
        input logic signed [31:0] accumulator,
        input logic signed [7:0] residual,
        input logic [31:0] accumulator_scale,
        input logic [31:0] residual_scale,
        input logic [31:0] destination_scale,
        input logic signed [7:0] expected_hidden,
        input logic signed [15:0] expected_carry
    );
        integer timeout;
        logic signed [7:0] held_hidden;
        logic signed [15:0] held_carry;
        begin
            while (!lane_start_ready) @(posedge clk);
            lane_accumulator <= accumulator;
            lane_residual <= residual;
            lane_acc_scale <= accumulator_scale;
            lane_res_scale <= residual_scale;
            lane_dst_scale <= destination_scale;
            lane_start_valid <= 1'b1;
            @(posedge clk);
            lane_start_valid <= 1'b0;
            timeout = 0;
            while (!lane_out_valid && timeout < 80) begin
                @(posedge clk);
                timeout = timeout + 1;
            end
            if (!lane_out_valid) $fatal(1, "producer timeout");
            if (lane_descriptor_error || lane_numeric_overflow) $fatal(1, "unexpected producer error");
            if (lane_hidden !== expected_hidden || lane_carry !== expected_carry)
                $fatal(1, "producer mismatch hidden=%0d/%0d carry=%0d/%0d", lane_hidden,
                       expected_hidden, lane_carry, expected_carry);
            if (lane_latency !== 6'd26) $fatal(1, "producer latency mismatch %0d", lane_latency);
            held_hidden = lane_hidden;
            held_carry = lane_carry;
            repeat (2) begin
                @(posedge clk);
                if (!lane_out_valid || lane_hidden !== held_hidden || lane_carry !== held_carry)
                    $fatal(1, "producer output changed under backpressure");
            end
            lane_out_ready <= 1'b1;
            @(posedge clk);
            lane_out_ready <= 1'b0;
        end
    endtask

    task automatic lane_error_transaction(
        input logic signed [31:0] accumulator,
        input logic [31:0] accumulator_scale,
        input logic expect_descriptor
    );
        integer timeout;
        begin
            while (!lane_start_ready) @(posedge clk);
            lane_accumulator <= accumulator;
            lane_residual <= 8'sd0;
            lane_acc_scale <= accumulator_scale;
            lane_res_scale <= 32'h00008000;
            lane_dst_scale <= 32'h00008000;
            lane_start_valid <= 1'b1;
            @(posedge clk);
            lane_start_valid <= 1'b0;
            timeout = 0;
            while (!lane_out_valid && timeout < 20) begin @(posedge clk); timeout = timeout + 1; end
            if (!lane_out_valid) $fatal(1, "producer error timeout");
            if (expect_descriptor ? !lane_descriptor_error : !lane_numeric_overflow)
                $fatal(1, "producer expected error missing");
            lane_out_ready <= 1'b1;
            @(posedge clk);
            lane_out_ready <= 1'b0;
        end
    endtask

    task automatic feed_rms_square(
        input logic signed [7:0] hidden,
        input logic signed [15:0] carry
    );
        begin
            while (!rms_square_ready) @(posedge clk);
            rms_square_hidden <= hidden;
            rms_square_carry <= carry;
            rms_square_valid <= 1'b1;
            @(posedge clk);
            rms_square_valid <= 1'b0;
        end
    endtask

    task automatic feed_rms_scale(
        input logic signed [7:0] hidden,
        input logic signed [15:0] carry,
        input logic signed [15:0] gain,
        input logic signed [7:0] expected
    );
        integer timeout;
        begin
            while (!rms_scale_ready) @(posedge clk);
            rms_scale_hidden <= hidden;
            rms_scale_carry <= carry;
            rms_scale_scaled_gain <= gain;
            rms_scale_valid <= 1'b1;
            @(posedge clk);
            rms_scale_valid <= 1'b0;
            timeout = 0;
            while (!rms_out_valid && timeout < 80) begin @(posedge clk); timeout = timeout + 1; end
            if (!rms_out_valid) $fatal(1, "RMSNorm output timeout");
            if (rms_out_data !== expected)
                $fatal(1, "RMSNorm output mismatch got=%0d expected=%0d", rms_out_data, expected);
            rms_out_ready <= 1'b1;
            @(posedge clk);
            rms_out_ready <= 1'b0;
        end
    endtask

    initial begin
        lane_start_valid = 1'b0;
        lane_out_ready = 1'b0;
        lane_accumulator = '0;
        lane_residual = '0;
        lane_acc_scale = '0;
        lane_res_scale = '0;
        lane_dst_scale = '0;
        state_metadata_valid = 1'b1;
        producer_start_valid = 1'b0;
        producer_layer = '0;
        producer_token = '0;
        producer_model = '0;
        producer_tag = '0;
        producer_lane_valid = 1'b0;
        producer_lane_carry = '0;
        producer_done_ready = 1'b0;
        consumer_start_valid = 1'b0;
        consumer_layer = '0;
        consumer_token = '0;
        consumer_model = '0;
        consumer_tag = '0;
        consumer_read_index = '0;
        consumer_finish_valid = 1'b0;
        consumer_finish_success = 1'b0;
        consumer_done_ready = 1'b0;
        rms_start_valid = 1'b0;
        rms_square_valid = 1'b0;
        rms_square_hidden = '0;
        rms_square_carry = '0;
        rms_scale_valid = 1'b0;
        rms_scale_hidden = '0;
        rms_scale_carry = '0;
        rms_scale_scaled_gain = '0;
        rms_out_ready = 1'b0;
        rms_done_ready = 1'b0;

        repeat (3) @(posedge clk);
        rst_ni <= 1'b1;
        repeat (2) @(posedge clk);

        lane_transaction(QECR_ACC_0, QECR_RES_0, QECR_AS_0, QECR_RS_0, QECR_DS_0,
                         QECR_HIDDEN_0, QECR_CARRY_0);
        lane_transaction(QECR_ACC_1, QECR_RES_1, QECR_AS_1, QECR_RS_1, QECR_DS_1,
                         QECR_HIDDEN_1, QECR_CARRY_1);
        lane_transaction(QECR_ACC_2, QECR_RES_2, QECR_AS_2, QECR_RS_2, QECR_DS_2,
                         QECR_HIDDEN_2, QECR_CARRY_2);
        lane_transaction(QECR_ACC_3, QECR_RES_3, QECR_AS_3, QECR_RS_3, QECR_DS_3,
                         QECR_HIDDEN_3, QECR_CARRY_3);
        lane_transaction(QECR_ACC_4, QECR_RES_4, QECR_AS_4, QECR_RS_4, QECR_DS_4,
                         QECR_HIDDEN_4, QECR_CARRY_4);
        lane_transaction(QECR_ACC_5, QECR_RES_5, QECR_AS_5, QECR_RS_5, QECR_DS_5,
                         QECR_HIDDEN_5, QECR_CARRY_5);
        lane_transaction(QECR_ACC_6, QECR_RES_6, QECR_AS_6, QECR_RS_6, QECR_DS_6,
                         QECR_HIDDEN_6, QECR_CARRY_6);
        lane_transaction(QECR_ACC_7, QECR_RES_7, QECR_AS_7, QECR_RS_7, QECR_DS_7,
                         QECR_HIDDEN_7, QECR_CARRY_7);
        lane_transaction(QECR_ACC_8, QECR_RES_8, QECR_AS_8, QECR_RS_8, QECR_DS_8,
                         QECR_HIDDEN_8, QECR_CARRY_8);
        lane_error_transaction(32'sd128, 32'h00008000, 1'b0);
        lane_error_transaction(-32'sd129, 32'h00008000, 1'b0);
        lane_error_transaction(32'sd1, 32'h00007fff, 1'b1);

        producer_layer <= 5'd7;
        producer_token <= 32'h12345678;
        producer_model <= 64'h8899aabbccddeeff;
        producer_tag <= 16'h55aa;
        consumer_layer <= 5'd1;
        consumer_token <= 32'hdeadbeef;
        consumer_model <= 64'h0123456789abcdef;
        consumer_tag <= 16'hbad0;
        producer_start_valid <= 1'b1;
        consumer_start_valid <= 1'b1;
        #1;
        if (!producer_start_ready || consumer_start_ready)
            $fatal(1, "dual-valid arbitration did not grant producer exclusively");
        @(posedge clk);
        producer_start_valid <= 1'b0;
        consumer_start_valid <= 1'b0;
        for (integer lane = 0; lane < 5; lane = lane + 1) begin
            while (!producer_lane_ready) @(posedge clk);
            producer_lane_carry <= $signed(lane * 100 - 150);
            producer_lane_valid <= 1'b1;
            @(posedge clk);
            producer_lane_valid <= 1'b0;
        end
        while (!producer_done_valid) @(posedge clk);
        if (producer_done_descriptor_error || producer_done_numeric_overflow || !carry_valid)
            $fatal(1, "carry state producer commit failed");
        if (carry_layer != 5'd7 || carry_token != 32'h12345678 ||
            carry_model != 64'h8899aabbccddeeff || carry_tag != 16'h55aa)
            $fatal(1, "carry state identity mismatch");
        producer_done_ready <= 1'b1;
        @(posedge clk);
        producer_done_ready <= 1'b0;

        while (!consumer_start_ready) @(posedge clk);
        consumer_layer <= 5'd8;
        consumer_token <= 32'h12345678;
        consumer_model <= 64'h8899aabbccddeeff;
        consumer_tag <= QECR_VALID_CONSUMER_ACCEPTED_TAG;
        consumer_start_valid <= 1'b1;
        @(posedge clk);
        consumer_start_valid <= 1'b0;
        consumer_tag <= QECR_VALID_CONSUMER_CHANGED_TAG;
        if (!consumer_active) @(posedge clk);
        for (integer lane = 0; lane < 5; lane = lane + 1) begin
            consumer_read_index = lane[2:0];
            #1;
            if (!consumer_read_index_valid ||
                consumer_read_carry !== $signed(lane * 100 - 150))
                $fatal(1, "carry state read mismatch lane=%0d", lane);
        end
        consumer_read_index = 3'd7;
        #1;
        if (consumer_read_index_valid || consumer_read_carry !== 16'sd0)
            $fatal(1, "out-of-range carry read was not fail-closed");
        consumer_finish_success <= 1'b1;
        consumer_finish_valid <= 1'b1;
        @(posedge clk);
        consumer_finish_valid <= 1'b0;
        consumer_finish_success <= 1'b0;
        while (!consumer_done_valid) @(posedge clk);
        if (consumer_done_descriptor_error || carry_valid ||
            consumer_completion_tag != QECR_VALID_CONSUMER_EXPECTED_TAG)
            $fatal(1, "carry state consume/clear failed");
        repeat (3) begin
            consumer_tag <= consumer_tag + 16'h0101;
            @(posedge clk);
            if (!consumer_done_valid ||
                consumer_completion_tag != QECR_VALID_CONSUMER_EXPECTED_TAG)
                $fatal(1, "accepted consumer tag changed under terminal backpressure");
        end
        consumer_done_ready <= 1'b1;
        @(posedge clk);
        consumer_done_ready <= 1'b0;

        while (!consumer_start_ready) @(posedge clk);
        consumer_layer <= 5'd9;
        consumer_token <= 32'hfeedface;
        consumer_model <= 64'h1111222233334444;
        consumer_tag <= QECR_INVALID_CONSUMER_ACCEPTED_TAG;
        consumer_start_valid <= 1'b1;
        @(posedge clk);
        consumer_start_valid <= 1'b0;
        consumer_tag <= QECR_INVALID_CONSUMER_CHANGED_TAG;
        while (!consumer_done_valid) @(posedge clk);
        if (!consumer_done_descriptor_error ||
            consumer_completion_tag != QECR_INVALID_CONSUMER_EXPECTED_TAG)
            $fatal(1, "invalid consumer start did not return its accepted completion tag");
        consumer_done_ready <= 1'b1;
        @(posedge clk);
        consumer_done_ready <= 1'b0;

        while (!rms_start_ready) @(posedge clk);
        rms_start_valid <= 1'b1;
        @(posedge clk);
        rms_start_valid <= 1'b0;
        feed_rms_square(QECR_RMS_H_0, QECR_RMS_C_0);
        feed_rms_square(QECR_RMS_H_1, QECR_RMS_C_1);
        feed_rms_square(QECR_RMS_H_2, QECR_RMS_C_2);
        feed_rms_square(QECR_RMS_H_3, QECR_RMS_C_3);
        feed_rms_scale(QECR_RMS_H_0, QECR_RMS_C_0, QECR_RMS_SCALED_G_0, QECR_RMS_Y_0);
        feed_rms_scale(QECR_RMS_H_1, QECR_RMS_C_1, QECR_RMS_SCALED_G_1, QECR_RMS_Y_1);
        feed_rms_scale(QECR_RMS_H_2, QECR_RMS_C_2, QECR_RMS_SCALED_G_2, QECR_RMS_Y_2);
        feed_rms_scale(QECR_RMS_H_3, QECR_RMS_C_3, QECR_RMS_SCALED_G_3, QECR_RMS_Y_3);
        while (!rms_done_valid) @(posedge clk);
        if (rms_done_numeric_overflow || rms_saturation) $fatal(1, "unexpected RMSNorm overflow");
        if (rms_sumsq !== QECR_RMS_SUMSQ || rms_mean !== QECR_RMS_MEAN ||
            rms_root !== QECR_RMS_ROOT || rms_inverse !== QECR_RMS_INV)
            $fatal(1, "RMSNorm aggregate mismatch sum=%0d mean=%0d root=%0d inv=%0d",
                   rms_sumsq, rms_mean, rms_root, rms_inverse);
        rms_done_ready <= 1'b1;
        @(posedge clk);

        $display("ACE2_CROSS_LAYER_ERROR_CARRY_RTL_PASS producer_vectors=%0d state_lanes=5 rms_lanes=4 adversarial_arbitration=pass read_bounds=pass accepted_consumer_tag=pass invalid_consumer_tag=pass",
                 QECR_PRODUCER_VECTOR_COUNT);
        $finish;
    end

    initial begin
        #200000;
        $fatal(1, "cross-layer error-carry testbench timeout");
    end
endmodule

`default_nettype wire
