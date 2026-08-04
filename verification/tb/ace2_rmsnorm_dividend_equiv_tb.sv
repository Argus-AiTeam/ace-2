`timescale 1ns/1ps
`default_nettype none

module ace2_rmsnorm_dividend_equiv_tb;
    localparam integer HIDDEN_SIZE = 8;
    localparam integer LANES = 2;
    localparam integer ACT_WIDTH = 8;
    localparam integer GAIN_WIDTH = 16;
    localparam integer BEATS = HIDDEN_SIZE / LANES;

    localparam [3:0] ST_IDLE         = 4'd0;
    localparam [3:0] ST_COLLECT      = 4'd1;
    localparam [3:0] ST_MEAN_DIV     = 4'd2;
    localparam [3:0] ST_SQRT         = 4'd3;
    localparam [3:0] ST_SQRT_DECIDE  = 4'd4;
    localparam [3:0] ST_INV_DIV      = 4'd5;
    localparam [3:0] ST_SCALE        = 4'd6;
    localparam [3:0] ST_SCALE_PREP   = 4'd7;
    localparam [3:0] ST_SCALE_MUL    = 4'd8;
    localparam [3:0] ST_SCALE_MUL_HI = 4'd9;
    localparam [3:0] ST_SCALE_ROUND  = 4'd10;
    localparam [3:0] ST_DONE         = 4'd11;

    reg clk;
    reg rst_n;
    reg clear;
    reg start_valid;
    reg in_valid;
    reg [LANES*ACT_WIDTH-1:0] in_data;
    reg gain_valid;
    reg [LANES*GAIN_WIDTH-1:0] gain_data;
    reg scale_act_valid;
    reg [LANES*ACT_WIDTH-1:0] scale_act_data;
    reg out_ready;
    reg done_ready;

    wire pre_start_ready;
    wire pre_in_ready;
    wire pre_gain_ready;
    wire pre_scale_act_ready;
    wire pre_out_valid;
    wire [LANES*ACT_WIDTH-1:0] pre_out_data;
    wire pre_done_valid;
    wire [47:0] pre_sumsq;
    wire [31:0] pre_inv_rms;
    wire pre_saturation;

    wire post_start_ready;
    wire post_in_ready;
    wire post_gain_ready;
    wire post_scale_act_ready;
    wire post_out_valid;
    wire [LANES*ACT_WIDTH-1:0] post_out_data;
    wire post_done_valid;
    wire [47:0] post_sumsq;
    wire [31:0] post_inv_rms;
    wire post_saturation;

    reg signed [ACT_WIDTH-1:0] act_mem [0:HIDDEN_SIZE-1];
    reg signed [GAIN_WIDTH-1:0] gain_mem [0:HIDDEN_SIZE-1];
    reg [47:0] expected_sumsq;

    integer failures;
    integer cycle_count;
    integer external_checks;
    integer collect_cone_checks;
    integer default_shift_checks;
    integer divider_transition_checks;
    integer mean_entry_checks;
    integer inv_entry_checks;
    integer random_case_count;
    integer start_fire_count;
    integer done_fire_count;
    reg [11:0] clear_seen_mask;
    reg input_backpressure_seen;
    reg output_backpressure_seen;
    reg done_hold_seen;
    reg early_start_seen;
    reg history_valid;

    reg [3:0] sampled_state;
    reg sampled_final_collect;
    reg sampled_sqrt_load;
    reg [47:0] sampled_dividend;
    reg [47:0] sampled_collect_load;
    reg [10:0] sampled_divisor;
    reg [29:0] sampled_quotient;
    reg [10:0] sampled_remainder;
    reg [5:0] sampled_count;
    reg [10:0] sampled_remainder_shift;
    reg sampled_subtract;
    reg [10:0] sampled_remainder_next;
    reg [29:0] sampled_quotient_next;

    ace2_rmsnorm_core_prechange #(
        .HIDDEN_SIZE(HIDDEN_SIZE),
        .LANES(LANES)
    ) pre_dut (
        .clk_i(clk),
        .rst_ni(rst_n),
        .clear_i(clear),
        .start_valid_i(start_valid),
        .start_ready_o(pre_start_ready),
        .in_valid_i(in_valid),
        .in_ready_o(pre_in_ready),
        .in_data_i(in_data),
        .gain_valid_i(gain_valid),
        .gain_ready_o(pre_gain_ready),
        .gain_data_i(gain_data),
        .scale_act_valid_i(scale_act_valid),
        .scale_act_ready_o(pre_scale_act_ready),
        .scale_act_data_i(scale_act_data),
        .out_valid_o(pre_out_valid),
        .out_ready_i(out_ready),
        .out_data_o(pre_out_data),
        .done_valid_o(pre_done_valid),
        .done_ready_i(done_ready),
        .sumsq_o(pre_sumsq),
        .inv_rms_q30_o(pre_inv_rms),
        .saturation_seen_o(pre_saturation)
    );

    ace2_rmsnorm_core #(
        .HIDDEN_SIZE(HIDDEN_SIZE),
        .LANES(LANES)
    ) post_dut (
        .clk_i(clk),
        .rst_ni(rst_n),
        .clear_i(clear),
        .start_valid_i(start_valid),
        .start_ready_o(post_start_ready),
        .in_valid_i(in_valid),
        .in_ready_o(post_in_ready),
        .in_data_i(in_data),
        .gain_valid_i(gain_valid),
        .gain_ready_o(post_gain_ready),
        .gain_data_i(gain_data),
        .scale_act_valid_i(scale_act_valid),
        .scale_act_ready_o(post_scale_act_ready),
        .scale_act_data_i(scale_act_data),
        .out_valid_o(post_out_valid),
        .out_ready_i(out_ready),
        .out_data_o(post_out_data),
        .done_valid_o(post_done_valid),
        .done_ready_i(done_ready),
        .sumsq_o(post_sumsq),
        .inv_rms_q30_o(post_inv_rms),
        .saturation_seen_o(post_saturation)
    );

    initial begin
        clk = 1'b0;
        forever #5 clk = ~clk;
    end

    task record_failure;
        input [8*96-1:0] label;
        begin
            $display("RMSNORM_DIVIDEND_CHECK_FAIL cycle=%0d label=%0s pre_state=%0d post_state=%0d",
                     cycle_count, label, pre_dut.state_q, post_dut.state_q);
            failures = failures + 1;
        end
    endtask

    function [LANES*ACT_WIDTH-1:0] packed_act_beat;
        input integer selected_beat;
        integer lane;
        begin
            packed_act_beat = {LANES*ACT_WIDTH{1'b0}};
            for (lane = 0; lane < LANES; lane = lane + 1)
                packed_act_beat[lane*ACT_WIDTH +: ACT_WIDTH] =
                    act_mem[selected_beat*LANES + lane];
        end
    endfunction

    function [LANES*GAIN_WIDTH-1:0] packed_gain_beat;
        input integer selected_beat;
        integer lane;
        begin
            packed_gain_beat = {LANES*GAIN_WIDTH{1'b0}};
            for (lane = 0; lane < LANES; lane = lane + 1)
                packed_gain_beat[lane*GAIN_WIDTH +: GAIN_WIDTH] =
                    gain_mem[selected_beat*LANES + lane];
        end
    endfunction

    task prepare_case;
        input integer case_kind;
        input integer seed;
        integer index;
        integer signed_value;
        integer magnitude;
        reg [31:0] random_word;
        begin
            random_word = seed;
            expected_sumsq = 48'd0;
            for (index = 0; index < HIDDEN_SIZE; index = index + 1) begin
                case (case_kind)
                    0: begin
                        act_mem[index] = 8'sd0;
                        gain_mem[index] = 16'sd256;
                    end
                    1: begin
                        act_mem[index] = index[0] ? 8'sd127 : -8'sd128;
                        gain_mem[index] = 16'sd256;
                    end
                    2: begin
                        act_mem[index] = -8'sd128;
                        gain_mem[index] = 16'sd256;
                    end
                    3: begin
                        act_mem[index] = 8'sd1;
                        gain_mem[index] = index[0] ? 16'sd384 : 16'sd128;
                    end
                    4: begin
                        act_mem[index] = 8'sd127;
                        gain_mem[index] = 16'sd32767;
                    end
                    default: begin
                        random_word = random_word * 32'd1664525 + 32'd1013904223;
                        act_mem[index] = random_word[31:24];
                        gain_mem[index] = 16'sd128 + $signed({1'b0, random_word[14:8]});
                    end
                endcase
                signed_value = $signed(act_mem[index]);
                magnitude = (signed_value < 0) ? -signed_value : signed_value;
                expected_sumsq = expected_sumsq + magnitude * magnitude;
            end
        end
    endtask

    task drive_idle_inputs;
        begin
            in_valid = 1'b0;
            gain_valid = 1'b0;
            scale_act_valid = 1'b0;
            out_ready = 1'b0;
            done_ready = 1'b0;
        end
    endtask

    task apply_reset;
        begin
            rst_n = 1'b0;
            clear = 1'b0;
            start_valid = 1'b0;
            in_data = {LANES*ACT_WIDTH{1'b0}};
            gain_data = {LANES*GAIN_WIDTH{1'b0}};
            scale_act_data = {LANES*ACT_WIDTH{1'b0}};
            drive_idle_inputs();
            repeat (3) @(posedge clk);
            @(negedge clk);
            rst_n = 1'b1;
            repeat (2) @(posedge clk);
            #2;
            if (!post_start_ready)
                record_failure("reset_did_not_return_idle");
        end
    endtask

    task start_operation;
        begin
            while (!post_start_ready) @(negedge clk);
            @(negedge clk);
            start_valid = 1'b1;
            @(posedge clk);
            if (!(start_valid && post_start_ready))
                record_failure("start_handshake_missing");
            #2;
            @(negedge clk);
            start_valid = 1'b0;
        end
    endtask

    task drive_started_operation;
        input integer traffic_mode;
        input integer done_hold_cycles;
        input integer expect_saturation;
        integer collect_beat;
        integer scale_beat;
        integer output_beat;
        integer guard;
        integer accepted;
        integer scale_accepted;
        begin
            collect_beat = 0;
            accepted = 0;
            in_valid = 1'b0;
            while (collect_beat < BEATS) begin
                @(negedge clk);
                if (accepted)
                    in_valid = 1'b0;
                if (!in_valid && ((traffic_mode == 0) || (((cycle_count + collect_beat) % 3) != 0))) begin
                    in_data = packed_act_beat(collect_beat);
                    in_valid = 1'b1;
                end
                accepted = 0;
                @(posedge clk);
                if (in_valid && post_in_ready) begin
                    collect_beat = collect_beat + 1;
                    accepted = 1;
                end
                #2;
            end
            @(negedge clk);
            in_valid = 1'b0;

            scale_beat = 0;
            output_beat = 0;
            scale_accepted = 0;
            guard = 0;
            gain_valid = 1'b0;
            scale_act_valid = 1'b0;
            while (output_beat < BEATS) begin
                @(negedge clk);
                if (scale_accepted) begin
                    gain_valid = 1'b0;
                    scale_act_valid = 1'b0;
                end
                if (!scale_act_valid && (scale_beat < BEATS) &&
                    ((traffic_mode == 0) || (((cycle_count + scale_beat) % 4) != 0))) begin
                    scale_act_data = packed_act_beat(scale_beat);
                    gain_data = packed_gain_beat(scale_beat);
                    scale_act_valid = 1'b1;
                    gain_valid = 1'b1;
                end
                out_ready = (traffic_mode == 0) || ((cycle_count % 5) != 0);
                scale_accepted = 0;
                @(posedge clk);
                if (scale_act_valid && gain_valid && post_scale_act_ready && post_gain_ready) begin
                    scale_beat = scale_beat + 1;
                    scale_accepted = 1;
                end
                if (post_out_valid && out_ready)
                    output_beat = output_beat + 1;
                guard = guard + 1;
                if (guard > 20000) begin
                    record_failure("output_timeout");
                    output_beat = BEATS;
                end
                #2;
            end
            @(negedge clk);
            gain_valid = 1'b0;
            scale_act_valid = 1'b0;
            out_ready = 1'b1;

            guard = 0;
            while (!post_done_valid && (guard < 20000)) begin
                @(posedge clk);
                guard = guard + 1;
                #2;
            end
            if (!post_done_valid)
                record_failure("done_timeout");
            if (post_sumsq !== expected_sumsq) begin
                $display("DIRECTED_SUMSQ_DETAIL got=%0d expected=%0d", post_sumsq, expected_sumsq);
                record_failure("directed_sumsq_mismatch");
            end
            if ((^post_inv_rms) === 1'bx)
                record_failure("inv_rms_unknown");
            if (expect_saturation && !post_saturation)
                record_failure("saturation_not_seen");
            repeat (done_hold_cycles) begin
                @(posedge clk);
                #2;
                if (!post_done_valid)
                    record_failure("done_valid_not_held");
            end
        end
    endtask

    task complete_done;
        begin
            @(negedge clk);
            done_ready = 1'b1;
            @(posedge clk);
            #2;
            @(negedge clk);
            done_ready = 1'b0;
            out_ready = 1'b0;
            if (!post_start_ready)
                record_failure("done_handshake_did_not_return_idle");
        end
    endtask

    task run_case;
        input integer case_kind;
        input integer seed;
        input integer traffic_mode;
        input integer done_hold_cycles;
        input integer expect_saturation;
        begin
            prepare_case(case_kind, seed);
            start_operation();
            drive_started_operation(traffic_mode, done_hold_cycles, expect_saturation);
            complete_done();
        end
    endtask

    task reach_state;
        input [3:0] target_state;
        integer collect_beat;
        integer scale_beat;
        integer guard;
        begin
            collect_beat = 0;
            scale_beat = 0;
            guard = 0;
            drive_idle_inputs();
            if (target_state != ST_IDLE)
                start_operation();
            while (post_dut.state_q != target_state) begin
                @(negedge clk);
                in_valid = 1'b0;
                gain_valid = 1'b0;
                scale_act_valid = 1'b0;
                out_ready = 1'b1;
                done_ready = 1'b0;
                if ((post_dut.state_q == ST_COLLECT) && post_in_ready && (collect_beat < BEATS)) begin
                    in_data = packed_act_beat(collect_beat);
                    in_valid = 1'b1;
                end
                if ((post_dut.state_q == ST_SCALE) && post_scale_act_ready &&
                    post_gain_ready && (scale_beat < BEATS)) begin
                    scale_act_data = packed_act_beat(scale_beat);
                    gain_data = packed_gain_beat(scale_beat);
                    scale_act_valid = 1'b1;
                    gain_valid = 1'b1;
                end
                @(posedge clk);
                if (in_valid && post_in_ready)
                    collect_beat = collect_beat + 1;
                if (scale_act_valid && gain_valid && post_scale_act_ready && post_gain_ready)
                    scale_beat = scale_beat + 1;
                guard = guard + 1;
                if (guard > 20000) begin
                    record_failure("reach_state_timeout");
                    disable reach_state;
                end
                #2;
            end
        end
    endtask

    task clear_in_state;
        input [3:0] target_state;
        begin
            prepare_case(1, 32'h12345678 + target_state);
            reach_state(target_state);
            @(negedge clk);
            start_valid = 1'b0;
            drive_idle_inputs();
            clear = 1'b1;
            @(posedge clk);
            #2;
            if (post_dut.state_q != ST_IDLE)
                record_failure("clear_did_not_return_idle");
            @(negedge clk);
            clear = 1'b0;
        end
    endtask

    task reset_in_divide;
        begin
            prepare_case(2, 32'h5a5a5a5a);
            reach_state(ST_INV_DIV);
            @(negedge clk);
            drive_idle_inputs();
            rst_n = 1'b0;
            #2;
            if (!post_start_ready || post_done_valid || post_out_valid)
                record_failure("async_reset_divide_outputs");
            @(posedge clk);
            @(negedge clk);
            rst_n = 1'b1;
            repeat (2) @(posedge clk);
            #2;
            if (!post_start_ready)
                record_failure("async_reset_divide_not_idle");
        end
    endtask

    task run_back_to_back;
        begin
            prepare_case(0, 32'h0);
            start_operation();
            drive_started_operation(1, 2, 0);

            prepare_case(3, 32'h0);
            @(negedge clk);
            start_valid = 1'b1;
            done_ready = 1'b1;
            @(posedge clk);
            #2;
            @(negedge clk);
            done_ready = 1'b0;
            if (!post_start_ready)
                record_failure("back_to_back_idle_gap_missing");
            @(posedge clk);
            if (!(start_valid && post_start_ready))
                record_failure("back_to_back_start_missing");
            #2;
            @(negedge clk);
            start_valid = 1'b0;
            drive_started_operation(1, 3, 0);
            complete_done();
        end
    endtask

    always @(posedge clk) begin
        cycle_count = cycle_count + 1;
        sampled_state = post_dut.state_q;
        sampled_final_collect = (post_dut.state_q == ST_COLLECT) &&
                                post_dut.collect_active_q &&
                                post_dut.collect_square_valid_q &&
                                post_dut.lane_last_q &&
                                (post_dut.collect_idx_q == BEATS-1);
        sampled_sqrt_load = (post_dut.state_q == ST_SQRT_DECIDE) &&
                            post_dut.sqrt_done_q;
        sampled_dividend = post_dut.div_dividend_q;
        sampled_collect_load = post_dut.next_sumsq_w + (HIDDEN_SIZE >> 1);
        sampled_divisor = post_dut.div_divisor_q;
        sampled_quotient = post_dut.div_quotient_q;
        sampled_remainder = post_dut.div_remainder_q;
        sampled_count = post_dut.div_count_q;
        sampled_remainder_shift = {post_dut.div_remainder_q[9:0],
                                   post_dut.div_dividend_q[47]};
        sampled_subtract = sampled_remainder_shift >= post_dut.div_divisor_q;
        sampled_remainder_next = sampled_subtract ?
                                 (sampled_remainder_shift - post_dut.div_divisor_q) :
                                 sampled_remainder_shift;
        sampled_quotient_next = {post_dut.div_quotient_q[28:0], sampled_subtract};

        if (rst_n && clear)
            clear_seen_mask[post_dut.state_q] = 1'b1;
        if (rst_n && start_valid && post_start_ready)
            start_fire_count = start_fire_count + 1;
        if (rst_n && post_done_valid && done_ready)
            done_fire_count = done_fire_count + 1;
        if (rst_n && in_valid && !post_in_ready)
            input_backpressure_seen = 1'b1;
        if (rst_n && post_out_valid && !out_ready)
            output_backpressure_seen = 1'b1;
        if (rst_n && post_done_valid && !done_ready)
            done_hold_seen = 1'b1;
        if (rst_n && start_valid && !post_start_ready)
            early_start_seen = 1'b1;

        #1;
        if (!rst_n) begin
            history_valid = 1'b0;
        end else begin
            external_checks = external_checks + 1;
            collect_cone_checks = collect_cone_checks + 1;
            if ((pre_start_ready !== post_start_ready) ||
                (pre_in_ready !== post_in_ready) ||
                (pre_gain_ready !== post_gain_ready) ||
                (pre_scale_act_ready !== post_scale_act_ready) ||
                (pre_out_valid !== post_out_valid) ||
                (pre_out_data !== post_out_data) ||
                (pre_done_valid !== post_done_valid) ||
                (pre_sumsq !== post_sumsq) ||
                (pre_inv_rms !== post_inv_rms) ||
                (pre_saturation !== post_saturation))
                record_failure("external_cycle_equivalence");

            if ((pre_dut.collect_beat_q !== post_dut.collect_beat_q) ||
                (pre_dut.collect_active_q !== post_dut.collect_active_q) ||
                (pre_dut.collect_square_valid_q !== post_dut.collect_square_valid_q))
                record_failure("collect_cone_cycle_equivalence");
            if (post_dut.collect_active_q && (post_dut.state_q != ST_COLLECT))
                record_failure("collect_active_state_invariant");

            if (pre_dut.state_q !== post_dut.state_q)
                record_failure("state_mismatch");
            if ((pre_dut.div_quotient_q !== post_dut.div_quotient_q) ||
                (pre_dut.div_remainder_q !== post_dut.div_remainder_q) ||
                (pre_dut.div_count_q !== post_dut.div_count_q))
                record_failure("divider_transition_mismatch");

            if ((post_dut.state_q == ST_MEAN_DIV) ||
                (post_dut.state_q == ST_INV_DIV)) begin
                if (pre_dut.div_dividend_q !== post_dut.div_dividend_q)
                    record_failure("divide_state_dividend_mismatch");
                if ((^post_dut.div_dividend_q) === 1'bx)
                    record_failure("divide_state_dividend_unknown");
            end

            if (history_valid) begin
                default_shift_checks = default_shift_checks + 1;
                if (sampled_final_collect) begin
                    if (post_dut.div_dividend_q !== sampled_collect_load)
                        record_failure("final_collect_load_priority");
                end else if (sampled_sqrt_load) begin
                    if (post_dut.div_dividend_q !== (48'd1 << 30))
                        record_failure("sqrt_complete_load_priority");
                end else if (post_dut.div_dividend_q !== {sampled_dividend[46:0], 1'b0}) begin
                    record_failure("default_shift_relation");
                end

                if ((post_dut.state_q == ST_MEAN_DIV) &&
                    (sampled_state != ST_MEAN_DIV)) begin
                    mean_entry_checks = mean_entry_checks + 1;
                    if (!sampled_final_collect ||
                        (post_dut.div_dividend_q !== sampled_collect_load))
                        record_failure("mean_div_entry_not_loaded");
                end
                if ((post_dut.state_q == ST_INV_DIV) &&
                    (sampled_state != ST_INV_DIV)) begin
                    inv_entry_checks = inv_entry_checks + 1;
                    if (!sampled_sqrt_load ||
                        (post_dut.div_dividend_q !== (48'd1 << 30)))
                        record_failure("inv_div_entry_not_loaded");
                end

                if ((sampled_state == ST_MEAN_DIV) ||
                    (sampled_state == ST_INV_DIV)) begin
                    divider_transition_checks = divider_transition_checks + 1;
                    if (post_dut.div_dividend_q !== {sampled_dividend[46:0], 1'b0})
                        record_failure("divider_dividend_transition");
                    if (post_dut.div_remainder_q !== sampled_remainder_next)
                        record_failure("divider_remainder_transition");
                    if (post_dut.div_quotient_q !== sampled_quotient_next)
                        record_failure("divider_quotient_transition");
                    if (sampled_count == 6'd1) begin
                        if (post_dut.div_count_q !== 6'd0)
                            record_failure("divider_terminal_count_transition");
                    end else if (post_dut.div_count_q !== (sampled_count - 6'd1)) begin
                        record_failure("divider_count_transition");
                    end
                end
            end
            history_valid = 1'b1;
        end
    end

    integer state_index;
    integer random_index;
    initial begin
        failures = 0;
        cycle_count = 0;
        external_checks = 0;
        collect_cone_checks = 0;
        default_shift_checks = 0;
        divider_transition_checks = 0;
        mean_entry_checks = 0;
        inv_entry_checks = 0;
        random_case_count = 0;
        start_fire_count = 0;
        done_fire_count = 0;
        clear_seen_mask = 12'd0;
        input_backpressure_seen = 1'b0;
        output_backpressure_seen = 1'b0;
        done_hold_seen = 1'b0;
        early_start_seen = 1'b0;
        history_valid = 1'b0;

        apply_reset();
        reset_in_divide();

        for (state_index = 0; state_index < 12; state_index = state_index + 1)
            clear_in_state(state_index[3:0]);

        run_case(0, 32'h00000000, 1, 3, 0);
        run_case(1, 32'h11111111, 1, 2, 0);
        run_case(2, 32'h22222222, 0, 2, 0);
        if (expected_sumsq !== HIDDEN_SIZE * 48'd16384) begin
            $display("MAX_SUMSQ_DETAIL expected_sumsq=%0d contract=%0d",
                     expected_sumsq, HIDDEN_SIZE * 48'd16384);
            record_failure("maximum_accumulated_sum_case");
        end
        run_case(3, 32'h33333333, 1, 2, 0);
        run_case(4, 32'h44444444, 1, 4, 1);

        for (random_index = 0; random_index < 12; random_index = random_index + 1) begin
            run_case(5, 32'hace20000 + random_index, 1, random_index % 4, 0);
            random_case_count = random_case_count + 1;
        end

        run_back_to_back();

        if (clear_seen_mask !== 12'hfff)
            record_failure("clear_state_coverage");
        if (!input_backpressure_seen)
            record_failure("input_backpressure_coverage");
        if (!output_backpressure_seen)
            record_failure("output_backpressure_coverage");
        if (!done_hold_seen)
            record_failure("done_hold_coverage");
        if (!early_start_seen)
            record_failure("back_to_back_early_start_coverage");
        if ((mean_entry_checks == 0) || (inv_entry_checks == 0) ||
            (divider_transition_checks == 0) || (external_checks == 0) ||
            (collect_cone_checks == 0))
            record_failure("assertion_coverage_empty");

        if (failures != 0) begin
            $display("ACE2_RMSNORM_DIVIDEND_EQUIV_FAIL failures=%0d", failures);
            $fatal(1, "ACE2_RMSNORM_DIVIDEND_EQUIV_FAIL");
        end

        $display("ACE2_RMSNORM_DIVIDEND_EQUIV_PASS cycles=%0d external_checks=%0d collect_cone_checks=%0d default_shift_checks=%0d divider_transition_checks=%0d mean_entries=%0d inv_entries=%0d random_cases=%0d clear_mask=%03x starts=%0d dones=%0d",
                 cycle_count, external_checks, collect_cone_checks, default_shift_checks,
                 divider_transition_checks, mean_entry_checks, inv_entry_checks,
                 random_case_count, clear_seen_mask, start_fire_count,
                 done_fire_count);
        $display("ACE2_RMSNORM_DIVIDEND_COVERAGE reset=1 clear_all_states=1 back_to_back=1 input_backpressure=1 output_backpressure=1 zero=1 signed_extrema=1 maximum_accumulated_sum=1 rounding=1 saturation=1 done_handshake=1 deterministic_random=12");
        $finish;
    end
endmodule

`default_nettype wire
