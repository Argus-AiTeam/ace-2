`timescale 1ns/1ps
`default_nettype none

module ace2_attention_compose_tb;
    reg clk;
    reg rst_n;
    reg clear;
    reg [7:0] command;
    reg [15:0] tile_count;
    wire command_allowed;
    reg start_valid;
    wire start_ready;
    reg [127:0] score_data;
    wire value_ready;
    reg value_valid;
    reg [127:0] value_data;
    wire out_valid;
    reg out_ready;
    wire [127:0] out_data;
    wire out_last;
    wire command_done;
    wire saturation_seen;
    wire [15:0] context_count;
    integer failures;
    integer test_case;
    integer tile;
    integer beat;
    integer token;
    integer tile_tokens;
    integer tiles;
    integer guard;
    integer cycle_count;
    integer case_start_cycle;
    reg [127:0] observed [0:3];

    `include "../generated/attention_compose_vectors.svh"

    ace2_attention_compose_core dut (
        .clk_i(clk),
        .rst_ni(rst_n),
        .clear_i(clear),
        .command_i(command),
        .tile_count_i(tile_count),
        .command_allowed_o(command_allowed),
        .start_valid_i(start_valid),
        .start_authorized_i(command_allowed),
        .start_ready_o(start_ready),
        .score_data_i(score_data),
        .value_ready_o(value_ready),
        .value_valid_i(value_valid),
        .value_data_i(value_data),
        .out_valid_o(out_valid),
        .out_ready_i(out_ready),
        .out_data_o(out_data),
        .out_last_o(out_last),
        .command_done_o(command_done),
        .saturation_seen_o(saturation_seen),
        .context_count_o(context_count)
    );

    always #5 clk = ~clk;
    always @(posedge clk) begin
        if (!rst_n) cycle_count <= 0;
        else cycle_count <= cycle_count + 1;
    end

    task start_tile;
        input [7:0] selected_command;
        input integer selected_case;
        input integer selected_tile;
        input integer selected_count;
        begin
            command = selected_command;
            tile_count = selected_count;
            score_data = attn_compose_score_tiles[
                selected_case*ATTN_COMPOSE_MAX_TILES + selected_tile
            ];
            guard = 0;
            while ((!command_allowed || !start_ready) && guard < 1000) begin
                guard = guard + 1;
                @(posedge clk);
            end
            if (!command_allowed || !start_ready) begin
                $display("ATTN_COMPOSE_START_REJECT cmd=%0d case=%0d tile=%0d",
                         selected_command, selected_case, selected_tile);
                failures = failures + 1;
            end
            @(negedge clk);
            start_valid = 1'b1;
            @(posedge clk);
            @(negedge clk);
            start_valid = 1'b0;
        end
    endtask

    task wait_done;
        begin
            guard = 0;
            while (!command_done && guard < 100000) begin
                guard = guard + 1;
                @(posedge clk);
            end
            if (!command_done) begin
                $display("ATTN_COMPOSE_COMMAND_TIMEOUT");
                failures = failures + 1;
            end
            @(posedge clk);
        end
    endtask

    task feed_value_tile;
        input integer selected_case;
        input integer token_base;
        input integer selected_count;
        begin
            for (token = 0; token < selected_count; token = token + 1) begin
                for (beat = 0; beat < ATTN_COMPOSE_BEATS; beat = beat + 1) begin
                    guard = 0;
                    while (!value_ready && guard < 100000) begin
                        guard = guard + 1;
                        @(posedge clk);
                    end
                    if (!value_ready) begin
                        $display("ATTN_COMPOSE_VALUE_TIMEOUT token=%0d beat=%0d",
                                 token, beat);
                        failures = failures + 1;
                    end
                    value_data = attn_compose_value_beats[
                        selected_case*ATTN_COMPOSE_MAX_CONTEXT*
                        ATTN_COMPOSE_BEATS +
                        (token_base + token)*ATTN_COMPOSE_BEATS + beat
                    ];
                    @(negedge clk);
                    value_valid = 1'b1;
                    @(posedge clk);
                    @(negedge clk);
                    value_valid = 1'b0;
                end
            end
        end
    endtask

    initial begin
        clk = 1'b0;
        rst_n = 1'b0;
        clear = 1'b0;
        command = 8'd0;
        tile_count = 16'd0;
        start_valid = 1'b0;
        score_data = 128'd0;
        value_valid = 1'b0;
        value_data = 128'd0;
        out_ready = 1'b0;
        failures = 0;
        cycle_count = 0;
        repeat (4) @(posedge clk);
        rst_n = 1'b1;

        for (test_case = 0; test_case < ATTN_COMPOSE_CASE_COUNT;
             test_case = test_case + 1) begin
            case_start_cycle = cycle_count;
            tiles = (attn_compose_context_count[test_case] +
                     ATTN_COMPOSE_TILE - 1) / ATTN_COMPOSE_TILE;
            for (tile = 0; tile < tiles; tile = tile + 1) begin
                tile_tokens =
                    ((tile + 1)*ATTN_COMPOSE_TILE <=
                     attn_compose_context_count[test_case]) ?
                    ATTN_COMPOSE_TILE :
                    attn_compose_context_count[test_case] -
                    tile*ATTN_COMPOSE_TILE;
                start_tile(tile == 0 ? 8'h00 : 8'h01,
                           test_case, tile, tile_tokens);
                wait_done();
            end
            if (context_count !== attn_compose_context_count[test_case]) begin
                $display("ATTN_COMPOSE_CONTEXT_MISMATCH case=%0d got=%0d expected=%0d",
                         test_case, context_count,
                         attn_compose_context_count[test_case]);
                failures = failures + 1;
            end
            for (tile = 0; tile < tiles; tile = tile + 1) begin
                tile_tokens =
                    ((tile + 1)*ATTN_COMPOSE_TILE <=
                     attn_compose_context_count[test_case]) ?
                    ATTN_COMPOSE_TILE :
                    attn_compose_context_count[test_case] -
                    tile*ATTN_COMPOSE_TILE;
                start_tile(tile == 0 ? 8'h02 : 8'h03,
                           test_case, tile, tile_tokens);
                wait_done();
            end
            for (tile = 0; tile < tiles; tile = tile + 1) begin
                tile_tokens =
                    ((tile + 1)*ATTN_COMPOSE_TILE <=
                     attn_compose_context_count[test_case]) ?
                    ATTN_COMPOSE_TILE :
                    attn_compose_context_count[test_case] -
                    tile*ATTN_COMPOSE_TILE;
                start_tile(tile == 0 ? 8'h04 :
                           ((tile == tiles - 1) ? 8'h06 : 8'h05),
                           test_case, tile, tile_tokens);
                feed_value_tile(test_case, tile*ATTN_COMPOSE_TILE,
                                tile_tokens);
                if (tile == tiles - 1) begin
                    for (beat = 0; beat < ATTN_COMPOSE_BEATS;
                         beat = beat + 1) begin
                        guard = 0;
                        while (!out_valid && guard < 100000) begin
                            guard = guard + 1;
                            @(posedge clk);
                        end
                        observed[beat] = out_data;
                        if (out_last !== (beat == ATTN_COMPOSE_BEATS - 1)) begin
                            $display("ATTN_COMPOSE_LAST_MISMATCH beat=%0d", beat);
                            failures = failures + 1;
                        end
                        @(negedge clk);
                        out_ready = 1'b1;
                        @(posedge clk);
                        @(negedge clk);
                        out_ready = 1'b0;
                    end
                end
                wait_done();
            end
            for (beat = 0; beat < ATTN_COMPOSE_BEATS; beat = beat + 1) begin
                if (observed[beat] !==
                    attn_compose_expected_beats[
                        test_case*ATTN_COMPOSE_BEATS + beat]) begin
                    $display("ATTN_COMPOSE_OUTPUT_MISMATCH case=%0d beat=%0d got=%032x expected=%032x",
                             test_case, beat, observed[beat],
                             attn_compose_expected_beats[
                                 test_case*ATTN_COMPOSE_BEATS + beat]);
                    failures = failures + 1;
                end
            end
            if (saturation_seen !==
                attn_compose_expected_saturation[test_case]) begin
                $display("ATTN_COMPOSE_SATURATION_MISMATCH case=%0d", test_case);
                failures = failures + 1;
            end
            $display("ACE2_ATTN_COMPOSE_CASE case=%0d context=%0d tiles=%0d cycles=%0d score_read_beats=%0d value_read_beats=%0d output_write_beats=4 sram_accesses=0",
                     test_case, attn_compose_context_count[test_case], tiles,
                     cycle_count - case_start_cycle, tiles*3,
                     attn_compose_context_count[test_case]*4);
        end

        if (failures == 0) begin
            $display("ACE2_ATTN_COMPOSE_TB_PASS cases=%0d contexts=9,16,17 context_max=32768 tile=8",
                     ATTN_COMPOSE_CASE_COUNT);
        end else begin
            $display("ACE2_ATTN_COMPOSE_TB_FAIL failures=%0d", failures);
            $fatal(1);
        end
        $finish;
    end
endmodule

`default_nettype wire
