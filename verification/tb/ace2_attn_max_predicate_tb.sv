`timescale 1ns/1ps
`default_nettype none

module ace2_attn_max_predicate_tb;
    localparam [6:0] ST_IDLE = 7'd0;
    localparam [6:0] ST_ATTN_START = 7'd62;
    localparam [6:0] ST_ATTN_WAIT_OUT = 7'd68;
    localparam [6:0] ST_ATTN_CENTER = 7'd121;
    localparam [6:0] ST_ATTN_ROUND = 7'd124;

    reg clk;
    reg rst_n;
    integer checks;

    ace2_shell dut (
        .clk_i(clk),
        .rst_ni(rst_n),
        .csr_valid_i(1'b0),
        .csr_ready_o(),
        .csr_write_i(1'b0),
        .csr_addr_i(32'd0),
        .csr_wdata_i(64'd0),
        .csr_wstrb_i(8'd0),
        .csr_rvalid_o(),
        .csr_rready_i(1'b1),
        .csr_rdata_o(),
        .csr_error_o(),
        .irq_o(),
        .cmd_valid_i(1'b0),
        .cmd_ready_o(),
        .cmd_opcode_i(8'd0),
        .cmd_flags_i(8'd0),
        .cmd_layer_id_i(8'd0),
        .cmd_m_i(16'd0),
        .cmd_n_i(16'd0),
        .cmd_k_i(16'd0),
        .cmd_sequence_position_i(16'd0),
        .cmd_completion_tag_i(16'd0),
        .cmd_src0_addr_i(64'd0),
        .cmd_src1_addr_i(64'd0),
        .cmd_dst_addr_i(64'd0),
        .cmd_scale_addr_i(64'd0),
        .cmd_scratch_addr_i(64'd0),
        .mem_req_valid_o(),
        .mem_req_ready_i(1'b0),
        .mem_req_write_o(),
        .mem_req_addr_o(),
        .mem_req_len_o(),
        .mem_req_tag_o(),
        .mem_wvalid_o(),
        .mem_wready_i(1'b0),
        .mem_wdata_o(),
        .mem_wstrb_o(),
        .mem_wtag_o(),
        .mem_rvalid_i(1'b0),
        .mem_rready_o(),
        .mem_rdata_i(128'd0),
        .mem_rtag_i(8'd0),
        .mem_rerror_i(1'b0),
        .mem_bvalid_i(1'b0),
        .mem_bready_o(),
        .mem_btag_i(8'd0),
        .mem_berror_i(1'b0),
        .sram_req_valid_o(),
        .sram_req_ready_i(8'hff),
        .sram_write_o(),
        .sram_addr_o(),
        .sram_wdata_o(),
        .sram_wstrb_o(),
        .sram_rdata_i(1024'd0),
        .sram_rvalid_i(8'd0),
        .busy_o(),
        .cmd_done_valid_o(),
        .cmd_done_ready_i(1'b1),
        .cmd_done_tag_o(),
        .cmd_done_error_o(),
        .cmd_done_sumsq_o(),
        .cmd_done_inv_rms_q30_o(),
        .cmd_done_saturation_seen_o()
    );

    initial begin
        clk = 1'b0;
        forever #5 clk = ~clk;
    end

    task check_bit;
        input actual;
        input expected;
        input [8*64-1:0] label;
        begin
            checks = checks + 1;
            if (actual !== expected) begin
                $display("ATTN_MAX_RETIME_CHECK_FAIL label=%0s got=%b expected=%b",
                         label, actual, expected);
                $fatal(1, "ATTN_MAX_RETIME_CHECK_FAIL");
            end
        end
    endtask

    task check_u7;
        input [6:0] actual;
        input [6:0] expected;
        input [8*64-1:0] label;
        begin
            checks = checks + 1;
            if (actual !== expected) begin
                $display("ATTN_MAX_RETIME_CHECK_FAIL label=%0s got=%0d expected=%0d",
                         label, actual, expected);
                $fatal(1, "ATTN_MAX_RETIME_CHECK_FAIL");
            end
        end
    endtask

    task check_u65;
        input [64:0] actual;
        input [64:0] expected;
        input [8*64-1:0] label;
        begin
            checks = checks + 1;
            if (actual !== expected) begin
                $display("ATTN_MAX_RETIME_CHECK_FAIL label=%0s got=%h expected=%h",
                         label, actual, expected);
                $fatal(1, "ATTN_MAX_RETIME_CHECK_FAIL");
            end
        end
    endtask

    function automatic expected_update;
        input [2:0] token_idx;
        input score_negative;
        input [64:0] rounded_abs;
        input max_negative;
        input [64:0] max_magnitude;
        begin
            expected_update =
                (token_idx == 3'd0) ||
                (!score_negative && max_negative) ||
                ((score_negative == max_negative) &&
                 (score_negative ?
                  (rounded_abs < max_magnitude) :
                  (rounded_abs > max_magnitude)));
        end
    endfunction

    task run_case;
        input [8*40-1:0] case_name;
        input [2:0] token_idx;
        input [15:0] context_count;
        input score_negative;
        input [63:0] shift_value;
        input guard_bit;
        input sticky_bit;
        input max_negative;
        input [64:0] max_magnitude;
        reg round_increment;
        reg [64:0] rounded_abs;
        reg update_max;
        reg [6:0] expected_next_state;
        begin
            round_increment = guard_bit && (sticky_bit || shift_value[0]);
            rounded_abs = {1'b0, shift_value} + round_increment;
            update_max = expected_update(token_idx, score_negative, rounded_abs,
                                         max_negative, max_magnitude);
            expected_next_state =
                (token_idx == (context_count[2:0] - 3'd1)) ?
                ST_ATTN_CENTER : ST_ATTN_START;

            @(negedge clk);
            dut.state_low_q = ST_ATTN_ROUND;
            dut.n_q = context_count;
            dut.attn_token_idx_q = token_idx;
            dut.attn_shift_value_q = shift_value;
            dut.attn_shift_guard_q = guard_bit;
            dut.attn_shift_sticky_q = sticky_bit;
            dut.attn_score_negative_q = score_negative;
            dut.attn_score_max_magnitude_q = max_magnitude;
            dut.attn_score_max_negative_q = max_negative;
            dut.read_fault_q = 1'b0;
            dut.write_fault_q = 1'b0;
            dut.watchdog_fire_q = 1'b0;
            dut.control_q[1] = 1'b0;

            @(posedge clk);
            #1;
            check_u7(dut.state_q, ST_ATTN_WAIT_OUT, "round_to_wait_state");
            check_u65(dut.attn_rounded_abs_q, rounded_abs,
                      "rounded_value_capture");
`ifdef ACE2_ATTN_MAX_RETIME_POSTCHANGE
            check_bit(dut.attn_score_update_max_q, update_max,
                      "predicate_same_transaction_alignment");
`endif
            check_u65(dut.attn_score_max_magnitude_q, max_magnitude,
                      "round_cycle_max_hold");
            check_bit(dut.attn_score_max_negative_q, max_negative,
                      "round_cycle_sign_hold");

            @(posedge clk);
            #1;
            check_u7(dut.state_q, expected_next_state,
                     "wait_out_next_state");
            check_u65(
                dut.attn_score_max_magnitude_q,
                update_max ? rounded_abs : max_magnitude,
                "wait_out_magnitude_matches_legacy"
            );
            check_bit(
                dut.attn_score_max_negative_q,
                update_max ? score_negative : max_negative,
                "wait_out_sign_matches_legacy"
            );
            if (expected_next_state == ST_ATTN_CENTER) begin
                check_u7({4'd0, dut.attn_center_idx_q}, 7'd0,
                         "last_token_center_index_zero");
            end else begin
                check_u7({4'd0, dut.attn_token_idx_q},
                         {4'd0, token_idx + 3'd1},
                         "nonlast_token_increment");
            end
            $display(
                "ATTN_MAX_RETIME_CASE_PASS name=%0s rounded=%h update=%0d max=%h max_neg=%0d next_state=%0d",
                case_name, rounded_abs, update_max,
                dut.attn_score_max_magnitude_q,
                dut.attn_score_max_negative_q, dut.state_q
            );
        end
    endtask

    task test_response_fault_hold;
        begin
            @(negedge clk);
            dut.state_low_q = ST_ATTN_WAIT_OUT;
            dut.attn_rounded_abs_q = 65'd77;
            dut.attn_score_negative_q = 1'b0;
            dut.attn_score_max_magnitude_q = 65'd11;
            dut.attn_score_max_negative_q = 1'b1;
`ifdef ACE2_ATTN_MAX_RETIME_POSTCHANGE
            dut.attn_score_update_max_q = 1'b1;
`endif
            dut.read_fault_q = 1'b1;
            dut.write_fault_q = 1'b0;
            dut.watchdog_fire_q = 1'b0;
            @(posedge clk);
            #1;
            check_u65(dut.attn_score_max_magnitude_q, 65'd11,
                      "response_fault_magnitude_hold");
            check_bit(dut.attn_score_max_negative_q, 1'b1,
                      "response_fault_sign_hold");
`ifdef ACE2_ATTN_MAX_RETIME_POSTCHANGE
            check_bit(dut.attn_score_update_max_q, 1'b1,
                      "response_fault_predicate_hold");
`endif
            $display("ATTN_MAX_RETIME_RESPONSE_FAULT_PASS");
        end
    endtask

    task test_watchdog_clear;
        begin
            @(negedge clk);
            dut.state_low_q = ST_ATTN_WAIT_OUT;
            dut.attn_rounded_abs_q = 65'd77;
            dut.attn_score_negative_q = 1'b1;
            dut.attn_score_max_magnitude_q = 65'd11;
            dut.attn_score_max_negative_q = 1'b1;
`ifdef ACE2_ATTN_MAX_RETIME_POSTCHANGE
            dut.attn_score_update_max_q = 1'b1;
`endif
            dut.read_fault_q = 1'b0;
            dut.write_fault_q = 1'b0;
            dut.watchdog_fire_q = 1'b1;
            @(posedge clk);
            #1;
            check_u65(dut.attn_rounded_abs_q, 65'd0,
                      "watchdog_rounded_clear");
            check_u65(dut.attn_score_max_magnitude_q, 65'd0,
                      "watchdog_magnitude_clear");
            check_bit(dut.attn_score_max_negative_q, 1'b0,
                      "watchdog_sign_clear");
`ifdef ACE2_ATTN_MAX_RETIME_POSTCHANGE
            check_bit(dut.attn_score_update_max_q, 1'b0,
                      "watchdog_predicate_clear");
`endif
            dut.watchdog_fire_q = 1'b0;
            $display("ATTN_MAX_RETIME_WATCHDOG_PASS");
        end
    endtask

    task test_soft_reset;
        begin
            @(negedge clk);
            dut.state_low_q = ST_ATTN_WAIT_OUT;
            dut.attn_rounded_abs_q = 65'd91;
            dut.attn_score_negative_q = 1'b0;
            dut.attn_score_max_magnitude_q = 65'd17;
            dut.attn_score_max_negative_q = 1'b1;
`ifdef ACE2_ATTN_MAX_RETIME_POSTCHANGE
            dut.attn_score_update_max_q = 1'b1;
`endif
            dut.read_fault_q = 1'b0;
            dut.write_fault_q = 1'b0;
            dut.watchdog_fire_q = 1'b0;
            dut.control_q[1] = 1'b1;
            @(posedge clk);
            #1;
            check_u7(dut.state_q, ST_IDLE, "soft_reset_enters_idle");
            check_u65(dut.attn_score_max_magnitude_q, 65'd91,
                      "soft_reset_wait_edge_legacy_update");
            check_bit(dut.attn_score_max_negative_q, 1'b0,
                      "soft_reset_wait_edge_legacy_sign");
            @(posedge clk);
            #1;
            check_u65(dut.attn_rounded_abs_q, 65'd0,
                      "soft_reset_idle_rounded_clear");
            check_u65(dut.attn_score_max_magnitude_q, 65'd0,
                      "soft_reset_idle_magnitude_clear");
            check_bit(dut.attn_score_max_negative_q, 1'b0,
                      "soft_reset_idle_sign_clear");
`ifdef ACE2_ATTN_MAX_RETIME_POSTCHANGE
            check_bit(dut.attn_score_update_max_q, 1'b0,
                      "soft_reset_idle_predicate_clear");
`endif
            $display("ATTN_MAX_RETIME_SOFT_RESET_PASS");
        end
    endtask

    task test_async_reset;
        begin
            @(negedge clk);
            dut.attn_rounded_abs_q = 65'd123;
            dut.attn_score_max_magnitude_q = 65'd456;
            dut.attn_score_max_negative_q = 1'b1;
`ifdef ACE2_ATTN_MAX_RETIME_POSTCHANGE
            dut.attn_score_update_max_q = 1'b1;
`endif
            #1 rst_n = 1'b0;
            #1;
            check_u65(dut.attn_rounded_abs_q, 65'd0,
                      "async_reset_rounded_clear");
            check_u65(dut.attn_score_max_magnitude_q, 65'd0,
                      "async_reset_magnitude_clear");
            check_bit(dut.attn_score_max_negative_q, 1'b0,
                      "async_reset_sign_clear");
`ifdef ACE2_ATTN_MAX_RETIME_POSTCHANGE
            check_bit(dut.attn_score_update_max_q, 1'b0,
                      "async_reset_predicate_clear");
`endif
            #1 rst_n = 1'b1;
            $display("ATTN_MAX_RETIME_ASYNC_RESET_PASS");
        end
    endtask

    initial begin
        checks = 0;
        rst_n = 1'b0;
        repeat (3) @(posedge clk);
        #1 rst_n = 1'b1;

        run_case("token_zero_override", 3'd0, 16'd3, 1'b1,
                 64'd50, 1'b0, 1'b0, 1'b0, 65'd100);
        run_case("positive_over_negative", 3'd1, 16'd3, 1'b0,
                 64'd5, 1'b0, 1'b0, 1'b1, 65'd1);
        run_case("negative_below_positive", 3'd1, 16'd3, 1'b1,
                 64'd1, 1'b0, 1'b0, 1'b0, 65'd100);
        run_case("positive_smaller", 3'd1, 16'd3, 1'b0,
                 64'd9, 1'b0, 1'b0, 1'b0, 65'd10);
        run_case("positive_equal", 3'd1, 16'd3, 1'b0,
                 64'd10, 1'b0, 1'b0, 1'b0, 65'd10);
        run_case("positive_larger_no_increment", 3'd1, 16'd3, 1'b0,
                 64'd11, 1'b0, 1'b0, 1'b0, 65'd10);
        run_case("positive_larger_round_increment", 3'd1, 16'd3, 1'b0,
                 64'd10, 1'b1, 1'b1, 1'b0, 65'd10);
        run_case("round_tie_even_no_increment", 3'd1, 16'd3, 1'b0,
                 64'd10, 1'b1, 1'b0, 1'b0, 65'd10);
        run_case("round_tie_odd_increment", 3'd1, 16'd3, 1'b0,
                 64'd11, 1'b1, 1'b0, 1'b0, 65'd11);
        run_case("negative_smaller_is_greater", 3'd1, 16'd3, 1'b1,
                 64'd9, 1'b0, 1'b0, 1'b1, 65'd10);
        run_case("negative_equal", 3'd1, 16'd3, 1'b1,
                 64'd10, 1'b0, 1'b0, 1'b1, 65'd10);
        run_case("last_token_larger", 3'd2, 16'd3, 1'b0,
                 64'd12, 1'b0, 1'b0, 1'b0, 65'd10);

        test_response_fault_hold();
        test_watchdog_clear();
        test_soft_reset();
        test_async_reset();

        $display("ACE2_ATTN_MAX_PREDICATE_TB_PASS checks=%0d", checks);
        $finish;
    end
endmodule

`default_nettype wire
