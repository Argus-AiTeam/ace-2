`timescale 1ns/1ps

module ace2_qk_residual_cross_term_tb;
    logic clk = 1'b0;
    logic rst_n = 1'b0;
    logic clear = 1'b0;
    always #5 clk = ~clk;

    logic side_start_valid, side_start_ready, side_operation_rope;
    logic signed [31:0] side_accumulator, side_multiplier;
    logic [5:0] side_shift;
    logic [31:0] side_baseline_scale, side_residual_scale;
    logic signed [3:0] side_real_s4, side_imag_s4;
    logic signed [15:0] side_cosine, side_sine;
    logic div_req_valid, div_req_ready, div_rsp_valid, div_rsp_ready;
    logic [127:0] div_numerator, div_denominator, div_quotient, div_remainder;
    logic side_out_valid, side_out_ready;
    logic signed [7:0] side_q8, side_real_s8, side_imag_s8;
    logic signed [3:0] side_r4;
    logic side_pos_clamp, side_neg_clamp, side_desc_error, side_overflow;

    logic score_start_valid, score_start_ready;
    logic [6:0] score_lane_count;
    logic signed [63:0] score_base_score;
    logic [31:0] score_q_scale, score_k_scale, score_rq_scale, score_rk_scale;
    logic score_lane_valid, score_lane_ready;
    logic signed [7:0] score_q, score_k, score_rq, score_rk;
    logic score_valid, score_ready;
    logic signed [63:0] score_value;
    logic signed [31:0] score_dot_q_rk, score_dot_rq_k, score_dot_rq_rk;
    logic score_desc_error, score_overflow;
    integer watchdog_cycles = 0;
    logic [255:0] divider_result_w;
    logic div_force_bad_remainder;
    localparam integer SCORE_EXPECTED_CYCLES = 216;

    `include "qk_residual_cross_term_vectors.svh"

    function automatic [255:0] divide_u128(
        input logic [127:0] numerator,
        input logic [127:0] denominator
    );
        logic [127:0] quotient;
        logic [128:0] remainder;
        integer bit_index;
        begin
            quotient = '0;
            remainder = '0;
            if (denominator == 0) begin
                divide_u128 = {{128{1'b1}}, numerator};
            end else begin
                for (bit_index = 127; bit_index >= 0; bit_index = bit_index - 1) begin
                    remainder = {remainder[127:0], numerator[bit_index]};
                    if (remainder >= {1'b0, denominator}) begin
                        remainder = remainder - {1'b0, denominator};
                        quotient[bit_index] = 1'b1;
                    end
                end
                divide_u128 = {quotient, remainder[127:0]};
            end
        end
    endfunction

    always @* divider_result_w = divide_u128(div_numerator, div_denominator);

    always @(posedge clk) begin
        watchdog_cycles <= watchdog_cycles + 1;
        if (watchdog_cycles > 5000) begin
            $display("WATCHDOG side_state=%0d side_valid=%0b div_req=%0b div_rsp=%0b score_state=%0d score_valid=%0b lane_ready=%0b",
                u_sidecar.state_q, side_out_valid, div_req_valid, div_rsp_valid,
                u_score.state_q, score_valid, score_lane_ready);
            $fatal(1, "TB_TIMEOUT");
        end
    end

    ace2_qk_residual_sidecar_core u_sidecar (
        .clk_i(clk), .rst_ni(rst_n), .clear_i(clear),
        .start_valid_i(side_start_valid), .start_ready_o(side_start_ready),
        .operation_rope_i(side_operation_rope),
        .accumulator_s32_i(side_accumulator),
        .multiplier_s32_i(side_multiplier), .shift_u6_i(side_shift),
        .baseline_scale32_i(side_baseline_scale),
        .residual_scale32_i(side_residual_scale),
        .residual_real_s4_i(side_real_s4), .residual_imag_s4_i(side_imag_s4),
        .cosine_q1_15_i(side_cosine), .sine_q1_15_i(side_sine),
        .div_req_valid_o(div_req_valid), .div_req_ready_i(div_req_ready),
        .div_numerator_u128_o(div_numerator),
        .div_denominator_u128_o(div_denominator),
        .div_rsp_valid_i(div_rsp_valid), .div_rsp_ready_o(div_rsp_ready),
        .div_quotient_u128_i(div_quotient),
        .div_remainder_u128_i(div_remainder),
        .out_valid_o(side_out_valid), .out_ready_i(side_out_ready),
        .baseline_q8_o(side_q8), .residual_s4_o(side_r4),
        .residual_real_s8_o(side_real_s8), .residual_imag_s8_o(side_imag_s8),
        .positive_clamp_o(side_pos_clamp), .negative_clamp_o(side_neg_clamp),
        .descriptor_error_o(side_desc_error), .numeric_overflow_o(side_overflow)
    );

    ace2_residual_cross_term_score_core u_score (
        .clk_i(clk), .rst_ni(rst_n), .clear_i(clear),
        .start_valid_i(score_start_valid), .start_ready_o(score_start_ready),
        .lane_count_u7_i(score_lane_count),
        .base_score_q20_44_s64_i(score_base_score),
        .query_scale32_i(score_q_scale), .key_scale32_i(score_k_scale),
        .query_residual_scale32_i(score_rq_scale),
        .key_residual_scale32_i(score_rk_scale),
        .lane_valid_i(score_lane_valid), .lane_ready_o(score_lane_ready),
        .query_q8_i(score_q), .key_q8_i(score_k),
        .query_residual_s8_i(score_rq), .key_residual_s8_i(score_rk),
        .score_valid_o(score_valid), .score_ready_i(score_ready),
        .score_q20_44_s64_o(score_value),
        .dot_q_rk_s32_o(score_dot_q_rk), .dot_rq_k_s32_o(score_dot_rq_k),
        .dot_rq_rk_s32_o(score_dot_rq_rk),
        .descriptor_error_o(score_desc_error), .numeric_overflow_o(score_overflow)
    );

    assign div_req_ready = 1'b1;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            div_rsp_valid <= 1'b0;
            div_quotient <= '0;
            div_remainder <= '0;
        end else begin
            if (div_rsp_valid && div_rsp_ready)
                div_rsp_valid <= 1'b0;
            if (div_req_valid && div_req_ready) begin
                div_quotient <= divider_result_w[255:128];
                div_remainder <= div_force_bad_remainder ?
                    div_denominator : divider_result_w[127:0];
                div_rsp_valid <= 1'b1;
            end
        end
    end

    task automatic set_projection_case(input integer which);
        begin
            case (which)
                0: begin side_accumulator=QKR_P0_ACC; side_multiplier=QKR_P0_MUL; side_shift=QKR_P0_SHIFT; side_baseline_scale=QKR_P0_BASE_SCALE; side_residual_scale=QKR_P0_RES_SCALE; end
                1: begin side_accumulator=QKR_P1_ACC; side_multiplier=QKR_P1_MUL; side_shift=QKR_P1_SHIFT; side_baseline_scale=QKR_P1_BASE_SCALE; side_residual_scale=QKR_P1_RES_SCALE; end
                2: begin side_accumulator=QKR_P2_ACC; side_multiplier=QKR_P2_MUL; side_shift=QKR_P2_SHIFT; side_baseline_scale=QKR_P2_BASE_SCALE; side_residual_scale=QKR_P2_RES_SCALE; end
                default: begin side_accumulator=QKR_P3_ACC; side_multiplier=QKR_P3_MUL; side_shift=QKR_P3_SHIFT; side_baseline_scale=QKR_P3_BASE_SCALE; side_residual_scale=QKR_P3_RES_SCALE; end
            endcase
        end
    endtask

    task automatic check_projection_case(input integer which);
        logic signed [7:0] expected_q8;
        logic signed [3:0] expected_r4;
        logic expected_pos, expected_neg;
        begin
            case (which)
                0: begin expected_q8=QKR_P0_Q8; expected_r4=QKR_P0_R4; expected_pos=QKR_P0_POS_CLAMP; expected_neg=QKR_P0_NEG_CLAMP; end
                1: begin expected_q8=QKR_P1_Q8; expected_r4=QKR_P1_R4; expected_pos=QKR_P1_POS_CLAMP; expected_neg=QKR_P1_NEG_CLAMP; end
                2: begin expected_q8=QKR_P2_Q8; expected_r4=QKR_P2_R4; expected_pos=QKR_P2_POS_CLAMP; expected_neg=QKR_P2_NEG_CLAMP; end
                default: begin expected_q8=QKR_P3_Q8; expected_r4=QKR_P3_R4; expected_pos=QKR_P3_POS_CLAMP; expected_neg=QKR_P3_NEG_CLAMP; end
            endcase
            @(negedge clk);
            while (!side_start_ready) @(negedge clk);
            set_projection_case(which);
            side_operation_rope = 1'b0;
            side_start_valid = 1'b1;
            @(negedge clk);
            side_start_valid = 1'b0;
            while (!side_out_valid) @(posedge clk);
            if (side_desc_error || side_overflow || side_q8 !== expected_q8 ||
                side_r4 !== expected_r4 || side_pos_clamp !== expected_pos ||
                side_neg_clamp !== expected_neg) begin
                $display("PROJECTION_FAIL case=%0d q8=%0d/%0d r4=%0d/%0d clamp=%0b%0b/%0b%0b de=%0b ov=%0b",
                    which, side_q8, expected_q8, side_r4, expected_r4,
                    side_pos_clamp, side_neg_clamp, expected_pos, expected_neg,
                    side_desc_error, side_overflow);
                $fatal(1);
            end
            @(posedge clk);
        end
    endtask

    task automatic set_rope_case(input integer which);
        begin
            case (which)
                0: begin side_real_s4=QKR_R0_REAL; side_imag_s4=QKR_R0_IMAG; side_cosine=QKR_R0_COS; side_sine=QKR_R0_SIN; end
                1: begin side_real_s4=QKR_R1_REAL; side_imag_s4=QKR_R1_IMAG; side_cosine=QKR_R1_COS; side_sine=QKR_R1_SIN; end
                2: begin side_real_s4=QKR_R2_REAL; side_imag_s4=QKR_R2_IMAG; side_cosine=QKR_R2_COS; side_sine=QKR_R2_SIN; end
                default: begin side_real_s4=QKR_R3_REAL; side_imag_s4=QKR_R3_IMAG; side_cosine=QKR_R3_COS; side_sine=QKR_R3_SIN; end
            endcase
        end
    endtask

    task automatic check_rope_case(input integer which);
        logic signed [7:0] expected_real, expected_imag;
        begin
            case (which)
                0: begin expected_real=QKR_R0_REAL_OUT; expected_imag=QKR_R0_IMAG_OUT; end
                1: begin expected_real=QKR_R1_REAL_OUT; expected_imag=QKR_R1_IMAG_OUT; end
                2: begin expected_real=QKR_R2_REAL_OUT; expected_imag=QKR_R2_IMAG_OUT; end
                default: begin expected_real=QKR_R3_REAL_OUT; expected_imag=QKR_R3_IMAG_OUT; end
            endcase
            @(negedge clk);
            while (!side_start_ready) @(negedge clk);
            set_rope_case(which);
            side_operation_rope = 1'b1;
            side_start_valid = 1'b1;
            @(negedge clk);
            side_start_valid = 1'b0;
            while (!side_out_valid) @(posedge clk);
            if (side_desc_error || side_overflow || side_real_s8 !== expected_real ||
                side_imag_s8 !== expected_imag) begin
                $display("ROPE_FAIL case=%0d real=%0d/%0d imag=%0d/%0d de=%0b ov=%0b",
                    which, side_real_s8, expected_real, side_imag_s8, expected_imag,
                    side_desc_error, side_overflow);
                $fatal(1);
            end
            @(posedge clk);
        end
    endtask

    task automatic set_score_lane(input integer lane);
        integer q_value, k_value, rq_value, rk_value;
        begin
            q_value = ((lane * 7) % 31) - 15;
            k_value = ((lane * 5) % 29) - 14;
            rq_value = ((lane * 3) % 15) - 7;
            rk_value = ((lane * 11) % 15) - 7;
            score_q = q_value;
            score_k = k_value;
            score_rq = rq_value;
            score_rk = rk_value;
        end
    endtask

    task automatic assert_known_outputs(input integer phase);
        begin
            if ((^side_start_ready) === 1'bx || (^side_out_valid) === 1'bx ||
                (^side_q8) === 1'bx || (^side_r4) === 1'bx ||
                (^side_real_s8) === 1'bx || (^side_imag_s8) === 1'bx ||
                (^side_pos_clamp) === 1'bx || (^side_neg_clamp) === 1'bx ||
                (^side_desc_error) === 1'bx || (^side_overflow) === 1'bx ||
                (^score_start_ready) === 1'bx || (^score_lane_ready) === 1'bx ||
                (^score_valid) === 1'bx || (^score_value) === 1'bx ||
                (^score_dot_q_rk) === 1'bx || (^score_dot_rq_k) === 1'bx ||
                (^score_dot_rq_rk) === 1'bx || (^score_desc_error) === 1'bx ||
                (^score_overflow) === 1'bx)
                $fatal(1, "XZ_OUTPUT_FAIL phase=%0d", phase);
        end
    endtask

    task automatic run_score_case;
        integer lane;
        integer latency_cycles;
        logic signed [63:0] held_score;
        begin
            @(negedge clk);
            while (!score_start_ready) @(negedge clk);
            score_lane_count = 7'd64;
            score_base_score = QKR_S_BASE_SCORE;
            score_q_scale = QKR_S_Q_SCALE;
            score_k_scale = QKR_S_K_SCALE;
            score_rq_scale = QKR_S_RQ_SCALE;
            score_rk_scale = QKR_S_RK_SCALE;
            score_ready = 1'b0;
            score_start_valid = 1'b1;
            @(posedge clk);
            if (!score_start_ready)
                $fatal(1, "SCORE_START_HANDSHAKE_FAIL");
            latency_cycles = 0;
            @(negedge clk);
            score_start_valid = 1'b0;
            fork
                begin : score_latency_monitor
                    while (!score_valid) begin
                        @(posedge clk);
                        #1;
                        latency_cycles = latency_cycles + 1;
                    end
                    assert (latency_cycles == SCORE_EXPECTED_CYCLES)
                        else $fatal(1, "SCORE_LATENCY_FAIL cycles=%0d expected=%0d",
                                    latency_cycles, SCORE_EXPECTED_CYCLES);
                    $display("SCORE_LATENCY_PASS cycles=%0d", latency_cycles);
                end
                begin : score_lane_driver
                    for (lane = 0; lane < 64; lane = lane + 1) begin
                        while (!score_lane_ready) @(negedge clk);
                        set_score_lane(lane);
                        score_lane_valid = 1'b1;
                        @(negedge clk);
                        score_lane_valid = 1'b0;
                    end
                end
            join
            if (score_desc_error || score_overflow || score_value !== QKR_S_SCORE ||
                score_dot_q_rk !== QKR_S_DOT_Q_RK ||
                score_dot_rq_k !== QKR_S_DOT_RQ_K ||
                score_dot_rq_rk !== QKR_S_DOT_RQ_RK) begin
                $display("SCORE_FAIL score=%0d/%0d dots=%0d,%0d,%0d expected=%0d,%0d,%0d de=%0b ov=%0b",
                    score_value, QKR_S_SCORE, score_dot_q_rk, score_dot_rq_k,
                    score_dot_rq_rk, QKR_S_DOT_Q_RK, QKR_S_DOT_RQ_K,
                    QKR_S_DOT_RQ_RK, score_desc_error, score_overflow);
                $fatal(1);
            end
            held_score = score_value;
            repeat (3) begin
                @(posedge clk);
                if (!score_valid || score_value !== held_score)
                    $fatal(1, "SCORE_BACKPRESSURE_FAIL");
            end
            score_ready = 1'b1;
            @(posedge clk);
        end
    endtask

    task automatic run_sidecar_overflow_case;
        begin
            @(negedge clk);
            while (!side_start_ready) @(negedge clk);
            set_projection_case(0);
            side_operation_rope = 1'b0;
            div_force_bad_remainder = 1'b1;
            side_start_valid = 1'b1;
            @(negedge clk);
            side_start_valid = 1'b0;
            while (!side_out_valid) @(posedge clk);
            if (side_desc_error || !side_overflow)
                $fatal(1, "SIDECAR_OVERFLOW_FAIL");
            div_force_bad_remainder = 1'b0;
            @(posedge clk);
            $display("SIDECAR_OVERFLOW_CHECK_PASS source=divider_remainder_invariant");
        end
    endtask

    task automatic run_score_overflow_case;
        integer lane;
        begin
            @(negedge clk);
            while (!score_start_ready) @(negedge clk);
            score_lane_count = 7'd64;
            score_base_score = 64'sd0;
            score_q_scale = 32'h0004ffff;
            score_k_scale = 32'h0004ffff;
            score_rq_scale = 32'h0004ffff;
            score_rk_scale = 32'h0004ffff;
            score_ready = 1'b0;
            score_start_valid = 1'b1;
            @(negedge clk);
            score_start_valid = 1'b0;
            for (lane = 0; lane < 64; lane = lane + 1) begin
                while (!score_lane_ready) @(negedge clk);
                score_q = 8'sd127;
                score_k = 8'sd127;
                score_rq = 8'sd127;
                score_rk = 8'sd127;
                score_lane_valid = 1'b1;
                @(negedge clk);
                score_lane_valid = 1'b0;
            end
            while (!score_valid) @(posedge clk);
            if (score_desc_error || !score_overflow ||
                score_dot_q_rk !== 32'sd1032256 ||
                score_dot_rq_k !== 32'sd1032256 ||
                score_dot_rq_rk !== 32'sd1032256)
                $fatal(1, "SCORE_OVERFLOW_FAIL dots=%0d,%0d,%0d de=%0b ov=%0b",
                    score_dot_q_rk, score_dot_rq_k, score_dot_rq_rk,
                    score_desc_error, score_overflow);
            score_ready = 1'b1;
            @(posedge clk);
            $display("SCORE_OVERFLOW_CHECK_PASS source=checked_q20_44_accumulator");
        end
    endtask

    task automatic run_reset_midflight_case;
        begin
            @(negedge clk);
            while (!(side_start_ready && score_start_ready)) @(negedge clk);
            set_projection_case(0);
            side_operation_rope = 1'b0;
            score_lane_count = 7'd64;
            score_base_score = QKR_S_BASE_SCORE;
            score_q_scale = QKR_S_Q_SCALE;
            score_k_scale = QKR_S_K_SCALE;
            score_rq_scale = QKR_S_RQ_SCALE;
            score_rk_scale = QKR_S_RK_SCALE;
            side_start_valid = 1'b1;
            score_start_valid = 1'b1;
            @(negedge clk);
            side_start_valid = 1'b0;
            score_start_valid = 1'b0;
            rst_n = 1'b0;
            #1;
            if (side_out_valid || score_valid || side_desc_error || score_desc_error ||
                side_overflow || score_overflow)
                $fatal(1, "RESET_MIDFLIGHT_ASYNC_FAIL");
            @(negedge clk);
            rst_n = 1'b1;
            repeat (2) @(posedge clk);
            if (!side_start_ready || !score_start_ready || side_out_valid || score_valid)
                $fatal(1, "RESET_MIDFLIGHT_RECOVERY_FAIL");
            assert_known_outputs(3);
            $display("RESET_MIDFLIGHT_CHECK_PASS modules=sidecar,score");
        end
    endtask

    initial begin
        side_start_valid = 0; side_operation_rope = 0; side_out_ready = 1;
        side_accumulator = 0; side_multiplier = 0; side_shift = 0;
        side_baseline_scale = 0; side_residual_scale = 0;
        side_real_s4 = 0; side_imag_s4 = 0; side_cosine = 0; side_sine = 0;
        div_force_bad_remainder = 0;
        score_start_valid = 0; score_lane_valid = 0; score_ready = 1;
        score_lane_count = 0; score_base_score = 0;
        score_q_scale = 0; score_k_scale = 0; score_rq_scale = 0; score_rk_scale = 0;
        score_q = 0; score_k = 0; score_rq = 0; score_rk = 0;
        repeat (4) @(posedge clk);
        rst_n <= 1'b1;
        repeat (2) @(posedge clk);
        assert_known_outputs(0);

        check_projection_case(0);
        check_projection_case(1);
        check_projection_case(2);
        check_projection_case(3);
        check_rope_case(0);
        check_rope_case(1);
        check_rope_case(2);
        check_rope_case(3);
        run_score_case();

        // Sidecar result and error metadata remain stable under backpressure.
        side_out_ready = 1'b0;
        @(negedge clk);
        while (!side_start_ready) @(negedge clk);
        set_rope_case(0);
        side_operation_rope = 1'b1;
        side_start_valid = 1'b1;
        @(negedge clk);
        side_start_valid = 1'b0;
        while (!side_out_valid) @(posedge clk);
        repeat (3) begin
            @(posedge clk);
            if (!side_out_valid || side_real_s8 !== QKR_R0_REAL_OUT ||
                side_imag_s8 !== QKR_R0_IMAG_OUT || side_desc_error || side_overflow)
                $fatal(1, "SIDECAR_BACKPRESSURE_FAIL");
        end
        side_out_ready = 1'b1;
        @(posedge clk);
        assert_known_outputs(1);

        // Reserved signed-4 code -8 fails closed before multiplier activity.
        @(negedge clk);
        while (!side_start_ready) @(negedge clk);
        side_operation_rope = 1'b1;
        side_real_s4 = -4'sd8;
        side_imag_s4 = 4'sd0;
        side_start_valid = 1'b1;
        @(negedge clk);
        side_start_valid = 1'b0;
        while (!side_out_valid) @(posedge clk);
        if (!side_desc_error || side_overflow)
            $fatal(1, "RESERVED_S4_FAIL");
        @(posedge clk);

        // Invalid lane count fails closed without accepting lane payload.
        @(negedge clk);
        while (!score_start_ready) @(negedge clk);
        score_lane_count = 7'd63;
        score_q_scale = QKR_S_Q_SCALE;
        score_k_scale = QKR_S_K_SCALE;
        score_rq_scale = QKR_S_RQ_SCALE;
        score_rk_scale = QKR_S_RK_SCALE;
        score_start_valid = 1'b1;
        @(negedge clk);
        score_start_valid = 1'b0;
        while (!score_valid) @(posedge clk);
        if (!score_desc_error || score_overflow || score_lane_ready)
            $fatal(1, "SCORE_DESCRIPTOR_FAIL");

        run_sidecar_overflow_case();
        run_score_overflow_case();

        // Clear cancels in-flight multiplier/divider state and removes valids.
        @(negedge clk);
        while (!side_start_ready) @(negedge clk);
        set_projection_case(0);
        side_operation_rope = 1'b0;
        side_start_valid = 1'b1;
        @(negedge clk);
        side_start_valid = 1'b0;
        repeat (2) @(posedge clk);
        @(negedge clk);
        clear = 1'b1;
        @(negedge clk);
        clear = 1'b0;
        @(posedge clk);
        if (side_out_valid || score_valid || !side_start_ready)
            $fatal(1, "CLEAR_FAIL");
        assert_known_outputs(2);

        run_reset_midflight_case();

        $display("XZ_OUTPUT_CHECK_PASS phases=post_reset,post_backpressure,post_clear,post_midflight_reset");
        $display("TB_PASS qk_residual_cross_term");
        $finish;
    end
endmodule
