`timescale 1ns/1ps
`default_nettype none

module ace2_c4_rope_q_replay_tb;
    localparam integer MAC_LANES = 4;
    localparam integer ACT_WIDTH = 8;
    localparam integer PROJ_GROUP_INDEX_WIDTH = 8;
    localparam integer ROPE_LANES = 16;

    reg clk;
    reg rst_n;

    reg proj_start_valid;
    wire proj_start_ready;
    reg [PROJ_GROUP_INDEX_WIDTH-1:0] proj_last_group;
    reg proj_pair_valid;
    wire proj_pair_ready;
    reg [MAC_LANES*ACT_WIDTH-1:0] proj_act_data;
    reg [MAC_LANES*4-1:0] proj_weight_data;
    reg proj_meta_valid;
    wire proj_meta_ready;
    reg signed [31:0] proj_multiplier;
    reg [5:0] proj_right_shift;
    reg signed [7:0] proj_zero_point;
    reg signed [31:0] proj_bias;
    wire proj_out_valid;
    reg proj_out_ready;
    wire [7:0] proj_out_data;
    wire signed [31:0] proj_acc;
    wire proj_acc_overflow;
    wire proj_saturation;

    reg rope_start_valid;
    wire rope_start_ready;
    reg rope_beat_valid;
    wire rope_beat_ready;
    reg [ROPE_LANES*ACT_WIDTH-1:0] rope_act_data;
    reg [ROPE_LANES*ACT_WIDTH-1:0] rope_pair_data;
    reg [ROPE_LANES*16-1:0] rope_scale_data;
    reg [ROPE_LANES*16-1:0] rope_pair_scale_data;
    reg [ROPE_LANES*16-1:0] rope_cos_data;
    reg [ROPE_LANES*16-1:0] rope_sin_data;
    reg rope_second_half;
    wire rope_out_valid;
    reg rope_out_ready;
    wire [ROPE_LANES*ACT_WIDTH-1:0] rope_out_data;
    wire rope_saturation;

    reg signed [7:0] q_actual [0:63];
    reg signed [7:0] rope_actual [0:63];
    integer failures;
    integer output_index;
    integer group_index;
    integer beat_index;
    integer lane_index;
    integer pair_beat;
    integer guard;
    reg [127:0] input_word;
    reg [63:0] weight_word;
    reg [127:0] meta_word;

    `include "c4_rope_q_witness.svh"

    ace2_w4a8_proj_core #(
        .K_SIZE(896),
        .MAC_LANES(MAC_LANES)
    ) projection_dut (
        .clk_i(clk),
        .rst_ni(rst_n),
        .clear_i(1'b0),
        .start_valid_i(proj_start_valid),
        .start_ready_o(proj_start_ready),
        .last_group_i(proj_last_group),
        .pair_valid_i(proj_pair_valid),
        .pair_ready_o(proj_pair_ready),
        .act_data_i(proj_act_data),
        .weight_data_i(proj_weight_data),
        .meta_valid_i(proj_meta_valid),
        .meta_ready_o(proj_meta_ready),
        .multiplier_i(proj_multiplier),
        .right_shift_i(proj_right_shift),
        .output_zero_point_i(proj_zero_point),
        .bias_accumulator_i(proj_bias),
        .out_valid_o(proj_out_valid),
        .out_ready_i(proj_out_ready),
        .out_data_o(proj_out_data),
        .acc_o(proj_acc),
        .accumulator_overflow_o(proj_acc_overflow),
        .saturation_seen_o(proj_saturation)
    );

    ace2_rope_core rope_dut (
        .clk_i(clk),
        .rst_ni(rst_n),
        .clear_i(1'b0),
        .start_valid_i(rope_start_valid),
        .start_ready_o(rope_start_ready),
        .beat_valid_i(rope_beat_valid),
        .beat_ready_o(rope_beat_ready),
        .act_data_i(rope_act_data),
        .pair_data_i(rope_pair_data),
        .scale_data_i(rope_scale_data),
        .pair_scale_data_i(rope_pair_scale_data),
        .cos_data_i(rope_cos_data),
        .sin_data_i(rope_sin_data),
        .second_half_i(rope_second_half),
        .out_valid_o(rope_out_valid),
        .out_ready_i(rope_out_ready),
        .out_data_o(rope_out_data),
        .saturation_seen_o(rope_saturation)
    );

    initial begin
        clk = 1'b0;
        forever #5 clk = ~clk;
    end

    task apply_reset;
        begin
            rst_n = 1'b0;
            proj_start_valid = 1'b0;
            proj_last_group = 8'd0;
            proj_pair_valid = 1'b0;
            proj_act_data = 32'd0;
            proj_weight_data = 16'd0;
            proj_meta_valid = 1'b0;
            proj_multiplier = 32'sd0;
            proj_right_shift = 6'd0;
            proj_zero_point = 8'sd0;
            proj_bias = 32'sd0;
            proj_out_ready = 1'b0;
            rope_start_valid = 1'b0;
            rope_beat_valid = 1'b0;
            rope_act_data = 128'd0;
            rope_pair_data = 128'd0;
            rope_scale_data = 256'd0;
            rope_pair_scale_data = 256'd0;
            rope_cos_data = 256'd0;
            rope_sin_data = 256'd0;
            rope_second_half = 1'b0;
            rope_out_ready = 1'b0;
            repeat (4) @(posedge clk);
            rst_n = 1'b1;
            repeat (2) @(posedge clk);
        end
    endtask

    task run_projection_output;
        input integer selected_output;
        begin
            while (!proj_start_ready) @(posedge clk);
            proj_last_group = WITNESS_GROUPS - 1;
            @(negedge clk);
            proj_start_valid = 1'b1;
            @(posedge clk);
            @(negedge clk);
            proj_start_valid = 1'b0;

            for (group_index = 0; group_index < WITNESS_GROUPS; group_index = group_index + 1) begin
                while (!proj_pair_ready) @(posedge clk);
                input_word = witness_input_beats[group_index / 4];
                weight_word = witness_weight_beats[
                    selected_output*WITNESS_WEIGHT_BEATS_PER_OUTPUT + (group_index / 4)
                ];
                proj_act_data = input_word[(group_index % 4)*32 +: 32];
                proj_weight_data = weight_word[(group_index % 4)*16 +: 16];
                @(negedge clk);
                proj_pair_valid = 1'b1;
                @(posedge clk);
                @(negedge clk);
                proj_pair_valid = 1'b0;
            end

            while (!proj_meta_ready) @(posedge clk);
            meta_word = witness_meta[selected_output];
            proj_multiplier = meta_word[31:0];
            proj_right_shift = meta_word[37:32];
            proj_zero_point = meta_word[47:40];
            proj_bias = meta_word[79:48];
            @(negedge clk);
            proj_meta_valid = 1'b1;
            @(posedge clk);
            @(negedge clk);
            proj_meta_valid = 1'b0;

            guard = 0;
            while (!proj_out_valid && guard < 4096) begin
                guard = guard + 1;
                @(posedge clk);
            end
            if (!proj_out_valid) begin
                $display("ROPE_Q_PROJECTION_TIMEOUT output=%0d", selected_output);
                failures = failures + 1;
            end else begin
                q_actual[selected_output] = proj_out_data;
                if ((proj_out_data !== witness_q_expected[selected_output]) ||
                    (proj_acc !== witness_biased_expected[selected_output]) ||
                    (proj_saturation !== witness_projection_sat_expected[selected_output]) ||
                    proj_acc_overflow) begin
                    $display("ROPE_Q_PROJECTION_MISMATCH output=%0d got=%0d expected=%0d acc=%0d expected_acc=%0d sat=%0d expected_sat=%0d overflow=%0d",
                             selected_output, $signed(proj_out_data),
                             $signed(witness_q_expected[selected_output]), proj_acc,
                             witness_biased_expected[selected_output], proj_saturation,
                             witness_projection_sat_expected[selected_output], proj_acc_overflow);
                    failures = failures + 1;
                end
                proj_out_ready = 1'b1;
                @(posedge clk);
                @(negedge clk);
                proj_out_ready = 1'b0;
            end
        end
    endtask

    task run_rope_beat;
        input integer selected_beat;
        begin
            pair_beat = selected_beat ^ 2;
            while (!rope_start_ready) @(posedge clk);
            @(negedge clk);
            rope_start_valid = 1'b1;
            @(posedge clk);
            @(negedge clk);
            rope_start_valid = 1'b0;

            while (!rope_beat_ready) @(posedge clk);
            for (lane_index = 0; lane_index < 16; lane_index = lane_index + 1) begin
                rope_act_data[lane_index*8 +: 8] = q_actual[selected_beat*16 + lane_index];
                rope_pair_data[lane_index*8 +: 8] = q_actual[pair_beat*16 + lane_index];
            end
            rope_scale_data = witness_rope_scale_beats[selected_beat];
            rope_pair_scale_data = witness_rope_scale_beats[pair_beat];
            rope_cos_data = witness_rope_cos_beats[selected_beat];
            rope_sin_data = witness_rope_sin_beats[selected_beat];
            rope_second_half = selected_beat[1];
            @(negedge clk);
            rope_beat_valid = 1'b1;
            @(posedge clk);
            @(negedge clk);
            rope_beat_valid = 1'b0;

            guard = 0;
            while (!rope_out_valid && guard < 4096) begin
                guard = guard + 1;
                @(posedge clk);
            end
            if (!rope_out_valid) begin
                $display("ROPE_Q_CORE_TIMEOUT beat=%0d", selected_beat);
                failures = failures + 1;
            end else begin
                for (lane_index = 0; lane_index < 16; lane_index = lane_index + 1) begin
                    rope_actual[selected_beat*16 + lane_index] = rope_out_data[lane_index*8 +: 8];
                end
                if ((rope_out_data !== witness_rope_expected_beats[selected_beat]) ||
                    (rope_saturation !== witness_rope_sat_expected[selected_beat])) begin
                    $display("ROPE_Q_CORE_MISMATCH beat=%0d got=%032x expected=%032x sat=%0d expected_sat=%0d",
                             selected_beat, rope_out_data,
                             witness_rope_expected_beats[selected_beat],
                             rope_saturation, witness_rope_sat_expected[selected_beat]);
                    failures = failures + 1;
                end
                rope_out_ready = 1'b1;
                @(posedge clk);
                @(negedge clk);
                rope_out_ready = 1'b0;
            end
        end
    endtask

    initial begin
        failures = 0;
        initialize_witness();
        apply_reset();
        for (output_index = 0; output_index < 64; output_index = output_index + 1) begin
            run_projection_output(output_index);
        end
        for (beat_index = 0; beat_index < 4; beat_index = beat_index + 1) begin
            run_rope_beat(beat_index);
        end
        for (lane_index = 0; lane_index < 64; lane_index = lane_index + 1) begin
            meta_word = witness_meta[lane_index];
            $display("ROPE_Q_WITNESS lane=%0d q_expected=%0d q_actual=%0d rope_expected=%0d rope_actual=%0d dot=%0d bias=%0d biased=%0d multiplier=%0d shift=%0d zp=%0d",
                     lane_index, $signed(witness_q_expected[lane_index]), q_actual[lane_index],
                     $signed(witness_rope_expected_beats[lane_index/16][(lane_index%16)*8 +: 8]),
                     rope_actual[lane_index], witness_dot_expected[lane_index],
                     $signed(meta_word[79:48]), witness_biased_expected[lane_index],
                     $signed(meta_word[31:0]), meta_word[37:32], $signed(meta_word[47:40]));
        end
        if (failures != 0) begin
            $display("ACE2_C4_ROPE_Q_REPLAY_FAIL failures=%0d", failures);
            $fatal(1, "ACE2_C4_ROPE_Q_REPLAY_FAIL");
        end
        $display("ACE2_C4_ROPE_Q_REPLAY_PASS projection_outputs=64 rope_outputs=64 mismatches=0");
        $finish;
    end
endmodule

`default_nettype wire
