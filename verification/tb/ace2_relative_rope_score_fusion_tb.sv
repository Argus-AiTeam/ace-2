`timescale 1ns/1ps
`default_nettype none

module ace2_relative_rope_score_fusion_tb;
    reg clk = 1'b0;
    always #5 clk = ~clk;
    reg rst_n = 1'b0;

    reg arithmetic_start;
    wire arithmetic_start_ready;
    reg pair_valid;
    wire pair_ready;
    reg signed [7:0] q0;
    reg signed [7:0] q1;
    reg signed [7:0] k0;
    reg signed [7:0] k1;
    reg signed [15:0] cosine;
    reg signed [15:0] sine;
    reg pair_last;
    wire arithmetic_valid;
    wire signed [63:0] precenter;
    wire signed [37:0] phase_acc;
    wire arithmetic_error;

    reg center_start;
    wire center_start_ready;
    reg [1:0] center_phase;
    reg [3:0] center_head;
    reg [15:0] center_query_position;
    reg [15:0] center_key_base;
    reg [8:0] center_key_count;
    reg center_score_valid;
    wire center_score_ready;
    reg signed [63:0] center_precenter;
    wire centered_valid;
    wire signed [15:0] centered;
    wire center_done;
    wire center_error;
    wire [1:0] debug_state;
    wire signed [63:0] debug_max;
    wire [15:0] debug_next;

    integer index;
    integer output_index;

    ace2_relative_rope_score_core u_arithmetic (
        .clk_i(clk), .rst_ni(rst_n), .clear_i(1'b0),
        .start_valid_i(arithmetic_start), .start_ready_o(arithmetic_start_ready),
        .query_scale32_i(32'h00008000), .key_scale32_i(32'h00008000),
        .pair_valid_i(pair_valid), .pair_ready_o(pair_ready),
        .q0_i(q0), .q1_i(q1), .k0_i(k0), .k1_i(k1),
        .cosine_i(cosine), .sine_i(sine), .pair_last_i(pair_last),
        .out_valid_o(arithmetic_valid), .out_ready_i(1'b1),
        .precenter_o(precenter), .phase_acc_o(phase_acc), .error_o(arithmetic_error)
    );

    ace2_global_score_center_core u_center (
        .clk_i(clk), .rst_ni(rst_n), .clear_i(1'b0),
        .start_valid_i(center_start), .start_ready_o(center_start_ready),
        .phase_i(center_phase), .query_head_i(center_head),
        .query_position_i(center_query_position), .key_base_i(center_key_base),
        .key_count_i(center_key_count), .score_valid_i(center_score_valid),
        .score_ready_o(center_score_ready), .precenter_i(center_precenter),
        .centered_valid_o(centered_valid), .centered_ready_i(1'b1),
        .centered_o(centered), .done_valid_o(center_done), .done_ready_i(1'b1),
        .done_error_o(center_error), .debug_state_o(debug_state),
        .debug_global_max_o(debug_max), .debug_next_key_base_o(debug_next)
    );

    task start_center_descriptor;
        input [1:0] phase;
        input [15:0] key_base;
        input [8:0] key_count;
        begin
            while (!center_start_ready) @(posedge clk);
            center_phase <= phase;
            center_key_base <= key_base;
            center_key_count <= key_count;
            center_start <= 1'b1;
            @(posedge clk);
            center_start <= 1'b0;
        end
    endtask

    task send_center_score;
        input signed [63:0] value;
        begin
            while (!center_score_ready) @(posedge clk);
            center_precenter <= value;
            center_score_valid <= 1'b1;
            @(posedge clk);
            center_score_valid <= 1'b0;
        end
    endtask

    task wait_center_done;
        begin
            while (!center_done) @(posedge clk);
            if (center_error) $fatal(1, "unexpected center descriptor error");
            @(posedge clk);
        end
    endtask

    initial begin
        arithmetic_start = 0;
        pair_valid = 0;
        q0 = 0;
        q1 = 0;
        k0 = 0;
        k1 = 0;
        cosine = 0;
        sine = 0;
        pair_last = 0;
        center_start = 0;
        center_phase = 0;
        center_head = 0;
        center_query_position = 16'd256;
        center_key_base = 0;
        center_key_count = 0;
        center_score_valid = 0;
        center_precenter = 0;
        repeat (4) @(posedge clk);
        rst_n = 1'b1;

        while (!arithmetic_start_ready) @(posedge clk);
        arithmetic_start <= 1'b1;
        @(posedge clk);
        arithmetic_start <= 1'b0;
        for (index = 0; index < 32; index = index + 1) begin
            while (!pair_ready) @(posedge clk);
            q0 <= (index == 0) ? 8'sd1 : 8'sd0;
            q1 <= 8'sd0;
            k0 <= (index == 0) ? 8'sd1 : 8'sd0;
            k1 <= 8'sd0;
            cosine <= (index == 0) ? 16'sd32767 : 16'sd0;
            sine <= 16'sd0;
            pair_last <= index == 31;
            pair_valid <= 1'b1;
            @(posedge clk);
            pair_valid <= 1'b0;
        end
        while (!arithmetic_valid) @(posedge clk);
        if (arithmetic_error) $fatal(1, "arithmetic core reported error");
        if (phase_acc !== 38'sd32767) $fatal(1, "phase mismatch %0d", phase_acc);
        if (precenter !== 64'sd64) $fatal(1, "precenter mismatch %0d", precenter);
        @(posedge clk);

        start_center_descriptor(2'd1, 16'd0, 9'd256);
        send_center_score(64'sd10);
        send_center_score(64'sd9);
        for (index = 2; index < 256; index = index + 1)
            send_center_score(-64'sd20);
        wait_center_done();
        if (debug_state !== 2'd1 || debug_max !== 64'sd10 || debug_next !== 16'd256)
            $fatal(1, "first MAX_SCAN state mismatch");

        start_center_descriptor(2'd1, 16'd256, 9'd1);
        send_center_score(64'sd100);
        wait_center_done();
        if (debug_state !== 2'd2 || debug_max !== 64'sd100 || debug_next !== 16'd257)
            $fatal(1, "SEALED state mismatch");

        output_index = 0;
        start_center_descriptor(2'd2, 16'd0, 9'd256);
        fork
            begin
                send_center_score(64'sd10);
                send_center_score(64'sd9);
                for (index = 2; index < 256; index = index + 1)
                    send_center_score(-64'sd20);
            end
            begin
                while (output_index < 256) begin
                    @(posedge clk);
                    if (centered_valid) begin
                        if (output_index == 0 && centered !== -16'sd90)
                            $fatal(1, "centered[0] mismatch %0d", centered);
                        if (output_index == 1 && centered !== -16'sd91)
                            $fatal(1, "centered[1] mismatch %0d", centered);
                        if (output_index >= 2 && centered !== -16'sd120)
                            $fatal(1, "centered[%0d] mismatch %0d", output_index, centered);
                        output_index = output_index + 1;
                    end
                end
            end
        join
        wait_center_done();

        output_index = 0;
        start_center_descriptor(2'd2, 16'd256, 9'd1);
        fork
            send_center_score(64'sd100);
            begin
                while (output_index < 1) begin
                    @(posedge clk);
                    if (centered_valid) begin
                        if (centered !== 16'sd0) $fatal(1, "last centered mismatch");
                        output_index = output_index + 1;
                    end
                end
            end
        join
        wait_center_done();
        if (debug_state !== 2'd0 || debug_max !== 64'sd0 || debug_next !== 16'd0)
            $fatal(1, "final EMPTY state mismatch");

        $display("ACE2_RELATIVE_ROPE_SCORE_FUSION_TB_PASS");
        $finish;
    end
endmodule

`default_nettype wire
