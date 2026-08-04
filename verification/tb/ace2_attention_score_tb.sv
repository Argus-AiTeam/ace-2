`timescale 1ns/1ps
`default_nettype none

module ace2_attention_score_tb;
    localparam integer HEAD_DIM = 64;
    localparam integer MAC_LANES = 1;
    localparam integer ACT_WIDTH = 8;
    localparam integer GROUPS = HEAD_DIM / MAC_LANES;

    reg clk;
    reg rst_n;
    reg start_valid;
    wire start_ready;
    reg pair_valid;
    wire pair_ready;
    reg [MAC_LANES*ACT_WIDTH-1:0] q_data;
    reg [MAC_LANES*ACT_WIDTH-1:0] k_data;
    reg signed [31:0] multiplier;
    reg [5:0] right_shift;
    wire out_valid;
    reg out_ready;
    wire [15:0] score;
    wire signed [31:0] acc;
    wire saturation_seen;

    integer failures;
    integer case_index;
    integer token_index;
    integer group_index;
    integer guard;
    integer beat_index;
    integer group_lane;
    reg [127:0] q_beat;
    reg [127:0] k_beat;
    reg [15:0] expected_score;
    reg signed [31:0] expected_acc;

    `include "../generated/attention_score_vectors.svh"

    ace2_attention_score_core #(
        .HEAD_DIM(HEAD_DIM),
        .MAC_LANES(MAC_LANES),
        .ACT_WIDTH(ACT_WIDTH),
        .ACC_WIDTH(32),
        .SCORE_WIDTH(16)
    ) dut (
        .clk_i(clk),
        .rst_ni(rst_n),
        .clear_i(1'b0),
        .start_valid_i(start_valid),
        .start_ready_o(start_ready),
        .pair_valid_i(pair_valid),
        .pair_ready_o(pair_ready),
        .q_data_i(q_data),
        .k_data_i(k_data),
        .multiplier_i(multiplier),
        .right_shift_i(right_shift),
        .out_valid_o(out_valid),
        .out_ready_i(out_ready),
        .score_o(score),
        .acc_o(acc),
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
            pair_valid = 1'b0;
            q_data = {MAC_LANES*ACT_WIDTH{1'b0}};
            k_data = {MAC_LANES*ACT_WIDTH{1'b0}};
            multiplier = 32'sd0;
            right_shift = 6'd0;
            out_ready = 1'b0;
            repeat (4) @(posedge clk);
            rst_n = 1'b1;
            repeat (2) @(posedge clk);
        end
    endtask

    task run_score;
        input integer selected_case;
        input integer selected_token;
        begin
            while (!start_ready) @(posedge clk);
            @(negedge clk);
            multiplier = attn_score_multiplier[selected_case];
            right_shift = attn_score_right_shift[selected_case];
            start_valid = 1'b1;
            @(posedge clk);
            @(negedge clk);
            start_valid = 1'b0;

            for (group_index = 0; group_index < GROUPS; group_index = group_index + 1) begin
                beat_index = group_index / (16 / MAC_LANES);
                group_lane = (group_index % (16 / MAC_LANES)) * MAC_LANES;
                q_beat = attn_score_q_beats[selected_case*ATTN_SCORE_BEATS_PER_VECTOR + beat_index];
                k_beat = attn_score_k_beats[(selected_case*ATTN_SCORE_CONTEXT_MAX + selected_token)*ATTN_SCORE_BEATS_PER_VECTOR + beat_index];
                while (!pair_ready) @(posedge clk);
                @(negedge clk);
                q_data = q_beat[group_lane*ACT_WIDTH +: MAC_LANES*ACT_WIDTH];
                k_data = k_beat[group_lane*ACT_WIDTH +: MAC_LANES*ACT_WIDTH];
                pair_valid = 1'b1;
                @(posedge clk);
                @(negedge clk);
                pair_valid = 1'b0;
            end

            guard = 0;
            while (!out_valid && (guard < 4096)) begin
                guard = guard + 1;
                @(posedge clk);
            end
            if (!out_valid) begin
                $display("ATTN_SCORE_CORE_TIMEOUT case=%0d token=%0d", selected_case, selected_token);
                failures = failures + 1;
            end else begin
                expected_score = attn_score_expected_core_word[selected_case][selected_token*16 +: 16];
                expected_acc = attn_score_expected_acc[selected_case*ATTN_SCORE_CONTEXT_MAX + selected_token];
                if (score !== expected_score) begin
                    $display("ATTN_SCORE_CORE_SCORE_MISMATCH case=%0d token=%0d got=%04x expected=%04x",
                             selected_case, selected_token, score, expected_score);
                    failures = failures + 1;
                end
                if (acc !== expected_acc) begin
                    $display("ATTN_SCORE_CORE_ACC_MISMATCH case=%0d token=%0d got=%0d expected=%0d",
                             selected_case, selected_token, acc, expected_acc);
                    failures = failures + 1;
                end
                if (saturation_seen !== attn_score_expected_core_saturation_token[selected_case*ATTN_SCORE_CONTEXT_MAX + selected_token]) begin
                    $display("ATTN_SCORE_CORE_SAT_MISMATCH case=%0d token=%0d got=%0d expected=%0d",
                             selected_case, selected_token, saturation_seen,
                             attn_score_expected_core_saturation_token[selected_case*ATTN_SCORE_CONTEXT_MAX + selected_token]);
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
        for (case_index = 0; case_index < ATTN_SCORE_CASE_COUNT; case_index = case_index + 1) begin
            for (token_index = 0; token_index < attn_score_context_count[case_index]; token_index = token_index + 1) begin
                run_score(case_index, token_index);
            end
        end
        if (failures != 0) begin
            $display("ACE2_ATTN_SCORE_TB_FAIL failures=%0d", failures);
            $fatal(1, "ACE2_ATTN_SCORE_TB_FAIL");
        end
        $display("ACE2_ATTN_SCORE_TB_PASS cases=%0d context_max=%0d", ATTN_SCORE_CASE_COUNT, ATTN_SCORE_CONTEXT_MAX);
        $finish;
    end
endmodule

`default_nettype wire
