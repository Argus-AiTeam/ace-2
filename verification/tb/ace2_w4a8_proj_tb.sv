`timescale 1ns/1ps
`default_nettype none

module ace2_w4a8_proj_tb;
    localparam integer ACT_WIDTH = 8;
    localparam integer MAC_LANES = 4;
    localparam integer TB_PROJ_GROUP_INDEX_WIDTH = 11;

    reg clk;
    reg rst_n;
    reg start_valid;
    wire start_ready;
    reg pair_valid;
    wire pair_ready;
    reg [MAC_LANES*ACT_WIDTH-1:0] act_data;
    reg [MAC_LANES*4-1:0] weight_data;
    reg [TB_PROJ_GROUP_INDEX_WIDTH-1:0] last_group;
    reg meta_valid;
    wire meta_ready;
    reg signed [31:0] multiplier;
    reg [5:0] right_shift;
    reg signed [ACT_WIDTH-1:0] output_zero_point;
    wire out_valid;
    reg out_ready;
    wire [ACT_WIDTH-1:0] out_data;
    wire signed [31:0] acc;
    wire saturation_seen;

    integer failures;
    integer group_index;
    integer case_index;
    integer out_index;
    integer row_index;
    integer guard;
    integer expected_flat;
    reg [127:0] expected_beat;

    `include "../generated/projection_vectors.svh"

    ace2_w4a8_proj_core #(
        .K_SIZE(PROJ_MAX_K),
        .MAC_LANES(MAC_LANES)
    ) dut (
        .clk_i(clk),
        .rst_ni(rst_n),
        .clear_i(1'b0),
        .start_valid_i(start_valid),
        .start_ready_o(start_ready),
        .last_group_i(last_group),
        .pair_valid_i(pair_valid),
        .pair_ready_o(pair_ready),
        .act_data_i(act_data),
        .weight_data_i(weight_data),
        .meta_valid_i(meta_valid),
        .meta_ready_o(meta_ready),
        .multiplier_i(multiplier),
        .right_shift_i(right_shift),
        .output_zero_point_i(output_zero_point),
        .out_valid_o(out_valid),
        .out_ready_i(out_ready),
        .out_data_o(out_data),
        .acc_o(acc),
        .saturation_seen_o(saturation_seen)
    );

    initial begin
        clk = 1'b0;
        forever #5 clk = ~clk;
    end

    initial begin
        $dumpfile("build/projection-waveform.vcd");
        $dumpvars(0, clk, rst_n, start_valid, start_ready, pair_valid, pair_ready,
                  meta_valid, meta_ready, out_valid, out_ready, out_data, acc,
                  saturation_seen);
    end

    task apply_reset;
        begin
            rst_n = 1'b0;
            start_valid = 1'b0;
            pair_valid = 1'b0;
            act_data = {MAC_LANES*ACT_WIDTH{1'b0}};
            weight_data = {MAC_LANES*4{1'b0}};
            last_group = {TB_PROJ_GROUP_INDEX_WIDTH{1'b0}};
            meta_valid = 1'b0;
            multiplier = 32'sd0;
            right_shift = 6'd0;
            output_zero_point = 8'sd0;
            out_ready = 1'b0;
            repeat (4) @(posedge clk);
            rst_n = 1'b1;
            repeat (2) @(posedge clk);
        end
    endtask

    task run_output;
        input integer selected_case;
        input integer selected_row;
        input integer selected_out;
        reg [127:0] act_word;
        reg [127:0] weight_word;
        reg [127:0] meta_word;
        begin
            while (!start_ready) @(posedge clk);
            last_group = TB_PROJ_GROUP_INDEX_WIDTH'(proj_case_groups[selected_case] - 1);
            @(negedge clk);
            start_valid = 1'b1;
            @(posedge clk);
            @(negedge clk);
            start_valid = 1'b0;

            for (group_index = 0; group_index < proj_case_groups[selected_case]; group_index = group_index + 1) begin
                while (!pair_ready) @(posedge clk);
                weight_word = proj_weight_beats[proj_case_weight_offset[selected_case] + selected_out*proj_case_weight_beats_per_output[selected_case] + (group_index / PROJ_GROUPS_PER_WEIGHT_BEAT)];
                act_word = proj_input_beats[selected_case*PROJ_MAX_ROWS*PROJ_MAX_INPUT_BEATS + selected_row*PROJ_MAX_INPUT_BEATS + (group_index / PROJ_GROUPS_PER_WEIGHT_BEAT)];
                weight_data = weight_word[((group_index % PROJ_GROUPS_PER_WEIGHT_BEAT)*MAC_LANES*4) +: MAC_LANES*4];
                act_data = act_word[((group_index % PROJ_GROUPS_PER_WEIGHT_BEAT)*MAC_LANES*ACT_WIDTH) +: MAC_LANES*ACT_WIDTH];
                @(negedge clk);
                pair_valid = 1'b1;
                @(posedge clk);
                @(negedge clk);
                pair_valid = 1'b0;
            end

            while (!meta_ready) @(posedge clk);
            meta_word = proj_meta_beats[proj_case_meta_offset[selected_case] + selected_out];
            multiplier = meta_word[31:0];
            right_shift = meta_word[37:32];
            output_zero_point = meta_word[47:40];
            @(negedge clk);
            meta_valid = 1'b1;
            @(posedge clk);
            @(negedge clk);
            meta_valid = 1'b0;

            guard = 0;
            while (!out_valid && (guard < 4096)) begin
                guard = guard + 1;
                @(posedge clk);
            end
            if (!out_valid) begin
                $display("PROJ_CORE_TIMEOUT case=%0d row=%0d out=%0d", selected_case, selected_row, selected_out);
                failures = failures + 1;
            end else begin
                expected_flat = selected_case*PROJ_MAX_ROWS*PROJ_MAX_OUTPUT_BEATS +
                                selected_row*PROJ_MAX_OUTPUT_BEATS + (selected_out >> 4);
                expected_beat = proj_expected_beats[expected_flat];
                if (out_data !== expected_beat[(selected_out & 15)*8 +: 8]) begin
                    $display("PROJ_CORE_MISMATCH case=%0d row=%0d out=%0d got=%02x expected=%02x acc=%0d",
                             selected_case, selected_row, selected_out, out_data,
                             expected_beat[(selected_out & 15)*8 +: 8], acc);
                    failures = failures + 1;
                end
                if (proj_expected_saturation[selected_case] && !saturation_seen) begin
                    $display("PROJ_CORE_EXPECTED_SATURATION_MISSING out=%0d", selected_out);
                    failures = failures + 1;
                end
                out_ready = 1'b1;
                @(posedge clk);
                @(negedge clk);
                out_ready = 1'b0;
            end
        end
    endtask

    task run_semantic_case;
        input integer semantic_id;
        input signed [7:0] selected_act;
        input signed [3:0] selected_weight;
        input signed [31:0] selected_multiplier;
        input [5:0] selected_shift;
        input signed [7:0] selected_zero_point;
        input [7:0] selected_expected;
        reg signed [31:0] selected_acc;
        begin
            while (!start_ready) @(posedge clk);
            last_group = {TB_PROJ_GROUP_INDEX_WIDTH{1'b0}};
            @(negedge clk);
            start_valid = 1'b1;
            @(posedge clk);
            @(negedge clk);
            start_valid = 1'b0;

            while (!pair_ready) @(posedge clk);
            act_data = {MAC_LANES*ACT_WIDTH{1'b0}};
            weight_data = {MAC_LANES*4{1'b0}};
            act_data[0 +: ACT_WIDTH] = selected_act;
            weight_data[0 +: 4] = selected_weight;
            selected_acc = selected_act * selected_weight;
            @(negedge clk);
            pair_valid = 1'b1;
            @(posedge clk);
            @(negedge clk);
            pair_valid = 1'b0;

            while (!meta_ready) @(posedge clk);
            multiplier = selected_multiplier;
            right_shift = selected_shift;
            output_zero_point = selected_zero_point;
            @(negedge clk);
            meta_valid = 1'b1;
            @(posedge clk);
            @(negedge clk);
            meta_valid = 1'b0;

            guard = 0;
            while (!out_valid && (guard < 4096)) begin
                guard = guard + 1;
                @(posedge clk);
            end
            if (!out_valid) begin
                $display("PROJ_SEMANTIC_TIMEOUT case=%0d shift=%0d",
                         semantic_id, selected_shift);
                failures = failures + 1;
            end else begin
                if ((out_data !== selected_expected) || (acc !== selected_acc) ||
                    saturation_seen) begin
                    $display("PROJ_SEMANTIC_MISMATCH case=%0d shift=%0d got=%02x expected=%02x acc=%0d expected_acc=%0d saturation=%0d",
                             semantic_id, selected_shift, out_data, selected_expected,
                             acc, selected_acc, saturation_seen);
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
        run_output(0, 0, 0);
        run_output(0, 1, 17);
        run_output(0, 1, 895);
        run_output(1, 0, 0);
        run_output(1, 0, 63);
        run_output(1, 0, 895);
        run_output(PROJ_CASE_MLP_GATE, 0, 0);
        run_output(PROJ_CASE_MLP_GATE, 0, 1023);
        run_output(PROJ_CASE_MLP_GATE, 0, 4863);
        run_output(PROJ_CASE_MLP_DOWN, 0, 0);
        run_output(PROJ_CASE_MLP_DOWN, 0, 127);
        run_output(PROJ_CASE_MLP_DOWN, 0, 895);
        run_output(PROJ_CASE_RMSNORM_CONSUMER, 0, 0);
        run_output(PROJ_CASE_RMSNORM_CONSUMER, 0, 17);
        run_output(PROJ_CASE_RMSNORM_CONSUMER, 0, 31);
        run_semantic_case(0, 8'sd3, 4'sd1, 32'sd2, 6'd0, 8'sd0, 8'h06);
        run_semantic_case(1, 8'sd1, 4'sd1, 32'sd1, 6'd63, 8'sd0, 8'h00);
        run_semantic_case(2, 8'sd10, 4'sd1, 32'sd1, 6'd2, 8'sd0, 8'h02);
        run_semantic_case(3, 8'sd14, 4'sd1, 32'sd1, 6'd2, 8'sd0, 8'h04);
        run_semantic_case(4, -8'sd10, 4'sd1, 32'sd1, 6'd2, 8'sd0, 8'hfe);
        run_semantic_case(5, -8'sd14, 4'sd1, 32'sd1, 6'd2, 8'sd0, 8'hfc);
        if (failures != 0) begin
            $display("ACE2_W4A8_PROJ_TB_FAIL failures=%0d", failures);
            $fatal(1, "ACE2_W4A8_PROJ_TB_FAIL");
        end
        $display("ACE2_PROJ_SEMANTIC_BINS_PASS round_half_ties=4 shift_0=1 shift_63=1 semantic_cases=6");
        $display("ACE2_W4A8_PROJ_TB_PASS cases=%0d checked_outputs=21 max_groups=%0d", PROJ_CASE_COUNT, PROJ_MAX_GROUPS);
        $finish;
    end
endmodule

`default_nettype wire
