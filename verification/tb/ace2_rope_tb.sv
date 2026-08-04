`timescale 1ns/1ps
`default_nettype none

module ace2_rope_tb;
    localparam integer LANES = 16;
    localparam integer ACT_WIDTH = 8;
    localparam integer SCALE_WIDTH = 16;

    reg clk;
    reg rst_n;
    reg start_valid;
    wire start_ready;
    reg beat_valid;
    wire beat_ready;
    reg [LANES*ACT_WIDTH-1:0] act_data;
    reg [LANES*ACT_WIDTH-1:0] pair_data;
    reg [LANES*SCALE_WIDTH-1:0] scale_data;
    reg [LANES*SCALE_WIDTH-1:0] pair_scale_data;
    reg [LANES*SCALE_WIDTH-1:0] cos_data;
    reg [LANES*SCALE_WIDTH-1:0] sin_data;
    reg second_half;
    wire out_valid;
    reg out_ready;
    wire [LANES*ACT_WIDTH-1:0] out_data;
    wire saturation_seen;

    integer failures;
    integer case_index;
    integer beat_index;
    integer guard;
    integer pair_beat;

    `include "../generated/rope_vectors.svh"

    ace2_rope_core dut (
        .clk_i(clk),
        .rst_ni(rst_n),
        .clear_i(1'b0),
        .start_valid_i(start_valid),
        .start_ready_o(start_ready),
        .beat_valid_i(beat_valid),
        .beat_ready_o(beat_ready),
        .act_data_i(act_data),
        .pair_data_i(pair_data),
        .scale_data_i(scale_data),
        .pair_scale_data_i(pair_scale_data),
        .cos_data_i(cos_data),
        .sin_data_i(sin_data),
        .second_half_i(second_half),
        .out_valid_o(out_valid),
        .out_ready_i(out_ready),
        .out_data_o(out_data),
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
            beat_valid = 1'b0;
            act_data = {LANES*ACT_WIDTH{1'b0}};
            pair_data = {LANES*ACT_WIDTH{1'b0}};
            scale_data = {LANES*SCALE_WIDTH{1'b0}};
            pair_scale_data = {LANES*SCALE_WIDTH{1'b0}};
            cos_data = {LANES*SCALE_WIDTH{1'b0}};
            sin_data = {LANES*SCALE_WIDTH{1'b0}};
            second_half = 1'b0;
            out_ready = 1'b0;
            repeat (4) @(posedge clk);
            rst_n = 1'b1;
            repeat (2) @(posedge clk);
        end
    endtask

    task run_beat;
        input integer selected_case;
        input integer selected_beat;
        begin
            pair_beat = selected_beat ^ 2;
            while (!start_ready) @(posedge clk);
            @(negedge clk);
            start_valid = 1'b1;
            @(posedge clk);
            @(negedge clk);
            start_valid = 1'b0;

            while (!beat_ready) @(posedge clk);
            act_data = rope_input_beats[selected_case*ROPE_BEATS + selected_beat];
            pair_data = rope_input_beats[selected_case*ROPE_BEATS + pair_beat];
            scale_data = {
                rope_scale_beats[selected_case*ROPE_BEATS*2 + selected_beat*2 + 1],
                rope_scale_beats[selected_case*ROPE_BEATS*2 + selected_beat*2]
            };
            pair_scale_data = {
                rope_scale_beats[selected_case*ROPE_BEATS*2 + pair_beat*2 + 1],
                rope_scale_beats[selected_case*ROPE_BEATS*2 + pair_beat*2]
            };
            cos_data = {
                rope_cos_beats[selected_case*ROPE_BEATS*2 + selected_beat*2 + 1],
                rope_cos_beats[selected_case*ROPE_BEATS*2 + selected_beat*2]
            };
            sin_data = {
                rope_sin_beats[selected_case*ROPE_BEATS*2 + selected_beat*2 + 1],
                rope_sin_beats[selected_case*ROPE_BEATS*2 + selected_beat*2]
            };
            second_half = selected_beat[1];
            @(negedge clk);
            beat_valid = 1'b1;
            @(posedge clk);
            @(negedge clk);
            beat_valid = 1'b0;

            guard = 0;
            while (!out_valid && (guard < 4096)) begin
                guard = guard + 1;
                @(posedge clk);
            end
            if (!out_valid) begin
                $display("ROPE_CORE_TIMEOUT case=%0d beat=%0d", selected_case, selected_beat);
                failures = failures + 1;
            end else begin
                if (out_data !== rope_expected_beats[selected_case*ROPE_BEATS + selected_beat]) begin
                    $display("ROPE_CORE_MISMATCH case=%0d beat=%0d got=%032x expected=%032x",
                             selected_case, selected_beat, out_data,
                             rope_expected_beats[selected_case*ROPE_BEATS + selected_beat]);
                    failures = failures + 1;
                end
                if (saturation_seen !== rope_expected_saturation[selected_case]) begin
                    $display("ROPE_CORE_SAT_MISMATCH case=%0d beat=%0d got=%0d expected=%0d",
                             selected_case, selected_beat, saturation_seen, rope_expected_saturation[selected_case]);
                    failures = failures + 1;
                end
                out_ready = 1'b1;
                @(posedge clk);
                @(negedge clk);
                out_ready = 1'b0;
            end
        end
    endtask

    initial begin
        failures = 0;
        apply_reset();
        for (case_index = 0; case_index < ROPE_CASE_COUNT; case_index = case_index + 1) begin
            for (beat_index = 0; beat_index < ROPE_BEATS; beat_index = beat_index + 1) begin
                run_beat(case_index, beat_index);
            end
        end
        if (failures != 0) begin
            $display("ACE2_ROPE_TB_FAIL failures=%0d", failures);
            $fatal(1, "ACE2_ROPE_TB_FAIL");
        end
        $display("ACE2_ROPE_TB_PASS cases=%0d beats_per_case=%0d", ROPE_CASE_COUNT, ROPE_BEATS);
        $finish;
    end
endmodule

`default_nettype wire
