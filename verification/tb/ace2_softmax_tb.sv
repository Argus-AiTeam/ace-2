`timescale 1ns/1ps
`default_nettype none

module ace2_softmax_tb;
    localparam integer CONTEXT_MAX = 8;

    reg clk;
    reg rst_n;
    reg start_valid;
    wire start_ready;
    reg [15:0] context_count;
    reg [CONTEXT_MAX*16-1:0] score_data;
    wire out_valid;
    reg out_ready;
    wire [CONTEXT_MAX*16-1:0] prob_data;
    wire saturation_seen;

    integer failures;
    integer case_index;
    integer guard;

    `include "../generated/softmax_vectors.svh"

    ace2_softmax_core #(
        .CONTEXT_MAX(CONTEXT_MAX),
        .SCORE_WIDTH(16),
        .PROB_WIDTH(16)
    ) dut (
        .clk_i(clk),
        .rst_ni(rst_n),
        .clear_i(1'b0),
        .start_valid_i(start_valid),
        .start_ready_o(start_ready),
        .context_count_i(context_count),
        .score_data_i(score_data),
        .out_valid_o(out_valid),
        .out_ready_i(out_ready),
        .prob_data_o(prob_data),
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
            context_count = 16'd0;
            score_data = {CONTEXT_MAX*16{1'b0}};
            out_ready = 1'b0;
            repeat (4) @(posedge clk);
            rst_n = 1'b1;
            repeat (2) @(posedge clk);
        end
    endtask

    task run_softmax;
        input integer selected_case;
        begin
            while (!start_ready) @(posedge clk);
            @(negedge clk);
            context_count = softmax_context_count[selected_case];
            score_data = softmax_score_word[selected_case];
            start_valid = 1'b1;
            @(posedge clk);
            @(negedge clk);
            start_valid = 1'b0;

            guard = 0;
            while (!out_valid && (guard < 4096)) begin
                guard = guard + 1;
                @(posedge clk);
            end
            if (!out_valid) begin
                $display("SOFTMAX_CORE_TIMEOUT case=%0d", selected_case);
                failures = failures + 1;
            end else begin
                if (prob_data !== softmax_expected_word[selected_case]) begin
                    $display("SOFTMAX_CORE_MISMATCH case=%0d got=%032x expected=%032x",
                             selected_case, prob_data, softmax_expected_word[selected_case]);
                    failures = failures + 1;
                end
                if (saturation_seen !== 1'b0) begin
                    $display("SOFTMAX_CORE_UNEXPECTED_SAT case=%0d", selected_case);
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
        for (case_index = 0; case_index < SOFTMAX_CASE_COUNT; case_index = case_index + 1) begin
            run_softmax(case_index);
        end
        if (failures != 0) begin
            $display("ACE2_SOFTMAX_TB_FAIL failures=%0d", failures);
            $fatal(1, "ACE2_SOFTMAX_TB_FAIL");
        end
        $display("ACE2_SOFTMAX_TB_PASS cases=%0d context_max=%0d", SOFTMAX_CASE_COUNT, SOFTMAX_CONTEXT_MAX);
        $finish;
    end
endmodule

`default_nettype wire
