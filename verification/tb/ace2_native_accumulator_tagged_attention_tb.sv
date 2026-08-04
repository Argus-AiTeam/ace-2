`timescale 1ns/1ps

module ace2_native_accumulator_tagged_attention_tb;
    logic clk = 1'b0;
    logic rst_n = 1'b0;
    logic clear = 1'b0;
    always #5 clk = ~clk;

    logic rope_in_valid, rope_in_ready;
    logic signed [31:0] acc0, acc1;
    logic [31:0] scale0, scale1;
    logic signed [15:0] cos_q15, sin_q15;
    logic rope_out_valid, rope_out_ready;
    logic signed [31:0] real_m, imag_m;
    logic signed [7:0] real_e, imag_e;
    logic rope_desc_error, rope_overflow;

    logic score_start_valid, score_start_ready;
    logic [6:0] lane_count;
    logic lane_valid, lane_ready;
    logic signed [31:0] q_m, k_m;
    logic signed [7:0] q_e, k_e;
    logic score_valid, score_ready;
    logic signed [63:0] score;
    logic score_desc_error, score_overflow;
    logic shared_mul_req_valid, shared_mul_req_ready;
    logic signed [31:0] shared_mul_a, shared_mul_b;
    logic shared_mul_rsp_valid, shared_mul_rsp_ready;
    logic signed [63:0] shared_mul_product;

    `include "native_accumulator_tagged_attention_vectors.svh"

    ace2_native_accumulator_rope_core u_rope (
        .clk_i(clk), .rst_ni(rst_n), .clear_i(clear),
        .in_valid_i(rope_in_valid), .in_ready_o(rope_in_ready),
        .acc0_s32_i(acc0), .scale0_u32_i(scale0),
        .acc1_s32_i(acc1), .scale1_u32_i(scale1),
        .cosine_q1_15_i(cos_q15), .sine_q1_15_i(sin_q15),
        .out_valid_o(rope_out_valid), .out_ready_i(rope_out_ready),
        .real_mantissa_s32_o(real_m), .real_exponent_s8_o(real_e),
        .imag_mantissa_s32_o(imag_m), .imag_exponent_s8_o(imag_e),
        .descriptor_error_o(rope_desc_error), .numeric_overflow_o(rope_overflow),
        .score_mul_req_valid_i(shared_mul_req_valid),
        .score_mul_req_ready_o(shared_mul_req_ready),
        .score_mul_operand_a_s32_i(shared_mul_a),
        .score_mul_operand_b_s32_i(shared_mul_b),
        .score_mul_rsp_valid_o(shared_mul_rsp_valid),
        .score_mul_rsp_ready_i(shared_mul_rsp_ready),
        .score_mul_product_s64_o(shared_mul_product)
    );

    ace2_tagged_attention_score_core u_score (
        .clk_i(clk), .rst_ni(rst_n), .clear_i(clear),
        .start_valid_i(score_start_valid), .start_ready_o(score_start_ready),
        .lane_count_u7_i(lane_count),
        .lane_valid_i(lane_valid), .lane_ready_o(lane_ready),
        .query_mantissa_s32_i(q_m), .query_exponent_s8_i(q_e),
        .key_mantissa_s32_i(k_m), .key_exponent_s8_i(k_e),
        .mul_req_valid_o(shared_mul_req_valid), .mul_req_ready_i(shared_mul_req_ready),
        .mul_operand_a_s32_o(shared_mul_a), .mul_operand_b_s32_o(shared_mul_b),
        .mul_rsp_valid_i(shared_mul_rsp_valid), .mul_rsp_ready_o(shared_mul_rsp_ready),
        .mul_product_s64_i(shared_mul_product),
        .score_valid_o(score_valid), .score_ready_i(score_ready),
        .score_q20_44_s64_o(score),
        .descriptor_error_o(score_desc_error), .numeric_overflow_o(score_overflow)
    );

    task automatic set_rope_case(input integer which);
        begin
            case (which)
                0: begin acc0=NAT_R0_ACC0; scale0=NAT_R0_SCALE0; acc1=NAT_R0_ACC1; scale1=NAT_R0_SCALE1; cos_q15=NAT_R0_COS; sin_q15=NAT_R0_SIN; end
                1: begin acc0=NAT_R1_ACC0; scale0=NAT_R1_SCALE0; acc1=NAT_R1_ACC1; scale1=NAT_R1_SCALE1; cos_q15=NAT_R1_COS; sin_q15=NAT_R1_SIN; end
                2: begin acc0=NAT_R2_ACC0; scale0=NAT_R2_SCALE0; acc1=NAT_R2_ACC1; scale1=NAT_R2_SCALE1; cos_q15=NAT_R2_COS; sin_q15=NAT_R2_SIN; end
                default: begin acc0=NAT_R3_ACC0; scale0=NAT_R3_SCALE0; acc1=NAT_R3_ACC1; scale1=NAT_R3_SCALE1; cos_q15=NAT_R3_COS; sin_q15=NAT_R3_SIN; end
            endcase
        end
    endtask

    task automatic check_rope_case(input integer which);
        logic signed [31:0] expected_real_m, expected_imag_m;
        logic signed [7:0] expected_real_e, expected_imag_e;
        begin
            case (which)
                0: begin expected_real_m=NAT_R0_REAL_M; expected_real_e=NAT_R0_REAL_E; expected_imag_m=NAT_R0_IMAG_M; expected_imag_e=NAT_R0_IMAG_E; end
                1: begin expected_real_m=NAT_R1_REAL_M; expected_real_e=NAT_R1_REAL_E; expected_imag_m=NAT_R1_IMAG_M; expected_imag_e=NAT_R1_IMAG_E; end
                2: begin expected_real_m=NAT_R2_REAL_M; expected_real_e=NAT_R2_REAL_E; expected_imag_m=NAT_R2_IMAG_M; expected_imag_e=NAT_R2_IMAG_E; end
                default: begin expected_real_m=NAT_R3_REAL_M; expected_real_e=NAT_R3_REAL_E; expected_imag_m=NAT_R3_IMAG_M; expected_imag_e=NAT_R3_IMAG_E; end
            endcase
            @(negedge clk);
            while (!rope_in_ready) @(negedge clk);
            set_rope_case(which);
            rope_in_valid = 1'b1;
            @(negedge clk);
            rope_in_valid = 1'b0;
            while (!rope_out_valid) @(posedge clk);
            if (rope_desc_error || rope_overflow || real_m !== expected_real_m ||
                real_e !== expected_real_e || imag_m !== expected_imag_m ||
                imag_e !== expected_imag_e) begin
                $display("ROPE_FAIL case=%0d real=%0d/%0d exp=%0d/%0d imag=%0d/%0d iexp=%0d/%0d de=%0b ov=%0b",
                    which, real_m, expected_real_m, real_e, expected_real_e,
                    imag_m, expected_imag_m, imag_e, expected_imag_e,
                    rope_desc_error, rope_overflow);
                $fatal(1);
            end
            @(posedge clk);
        end
    endtask

    task automatic set_score_lane(input integer which, input integer lane);
        begin
            if (which == 0) begin
                case (lane)
                    0: begin q_m=NAT_S0_Q_M_0; q_e=NAT_S0_Q_E_0; k_m=NAT_S0_K_M_0; k_e=NAT_S0_K_E_0; end
                    1: begin q_m=NAT_S0_Q_M_1; q_e=NAT_S0_Q_E_1; k_m=NAT_S0_K_M_1; k_e=NAT_S0_K_E_1; end
                    2: begin q_m=NAT_S0_Q_M_2; q_e=NAT_S0_Q_E_2; k_m=NAT_S0_K_M_2; k_e=NAT_S0_K_E_2; end
                    default: begin q_m=NAT_S0_Q_M_3; q_e=NAT_S0_Q_E_3; k_m=NAT_S0_K_M_3; k_e=NAT_S0_K_E_3; end
                endcase
            end else if (which == 1) begin
                if (lane == 0) begin q_m=NAT_S1_Q_M_0; q_e=NAT_S1_Q_E_0; k_m=NAT_S1_K_M_0; k_e=NAT_S1_K_E_0; end
                else begin q_m=NAT_S1_Q_M_1; q_e=NAT_S1_Q_E_1; k_m=NAT_S1_K_M_1; k_e=NAT_S1_K_E_1; end
            end else begin
                q_m=NAT_S2_Q_M_0; q_e=NAT_S2_Q_E_0;
                k_m=NAT_S2_K_M_0; k_e=NAT_S2_K_E_0;
            end
        end
    endtask

    task automatic run_score_case(input integer which, input integer lanes, input logic signed [63:0] expected);
        integer lane;
        begin
            @(negedge clk);
            while (!score_start_ready) @(negedge clk);
            lane_count = lanes[6:0];
            score_start_valid = 1'b1;
            @(negedge clk);
            score_start_valid = 1'b0;
            for (lane = 0; lane < lanes; lane = lane + 1) begin
                while (!lane_ready) @(negedge clk);
                set_score_lane(which, lane);
                lane_valid = 1'b1;
                @(negedge clk);
                lane_valid = 1'b0;
            end
            while (!score_valid) @(posedge clk);
            if (score_desc_error || score_overflow || score !== expected) begin
                $display("SCORE_FAIL case=%0d got=%0d expected=%0d de=%0b ov=%0b",
                         which, score, expected, score_desc_error, score_overflow);
                $display("SCORE_DEBUG max_exp=%0d p0=%0d/e%0d p1=%0d/e%0d p2=%0d/e%0d p3=%0d/e%0d acc=%0d",
                         u_score.max_exponent_q,
                         u_score.product_mem[0], u_score.exponent_mem[0],
                         u_score.product_mem[1], u_score.exponent_mem[1],
                         u_score.product_mem[2], u_score.exponent_mem[2],
                         u_score.product_mem[3], u_score.exponent_mem[3],
                         u_score.accumulator_q);
                $fatal(1);
            end
            @(posedge clk);
        end
    endtask

    initial begin
        rope_in_valid = 0; rope_out_ready = 1;
        score_start_valid = 0; lane_valid = 0; score_ready = 1;
        acc0 = 0; acc1 = 0; scale0 = 0; scale1 = 0; cos_q15 = 0; sin_q15 = 0;
        lane_count = 0; q_m = 0; q_e = 0; k_m = 0; k_e = 0;
        repeat (4) @(posedge clk);
        rst_n <= 1'b1;
        repeat (2) @(posedge clk);

        check_rope_case(0);
        check_rope_case(1);
        check_rope_case(2);
        check_rope_case(3);

        // Backpressure must hold a complete tagged result stable.
        rope_out_ready <= 1'b0;
        @(negedge clk);
        while (!rope_in_ready) @(negedge clk);
        set_rope_case(0);
        rope_in_valid = 1'b1;
        @(negedge clk);
        rope_in_valid = 1'b0;
        while (!rope_out_valid) @(posedge clk);
        repeat (3) begin
            @(posedge clk);
            if (!rope_out_valid || real_m !== NAT_R0_REAL_M || imag_m !== NAT_R0_IMAG_M)
                $fatal(1, "ROPE_BACKPRESSURE_FAIL");
        end
        rope_out_ready <= 1'b1;
        @(posedge clk);

        run_score_case(0, NAT_S0_LANES, NAT_S0_EXPECTED);
        run_score_case(1, NAT_S1_LANES, NAT_S1_EXPECTED);
        run_score_case(2, NAT_S2_LANES, NAT_S2_EXPECTED);

        @(negedge clk);
        while (!score_start_ready) @(negedge clk);
        lane_count = 0;
        score_start_valid = 1'b1;
        @(negedge clk);
        score_start_valid = 1'b0;
        while (!score_valid) @(posedge clk);
        if (!score_desc_error || score_overflow)
            $fatal(1, "SCORE_DESCRIPTOR_ERROR_FAIL");
        @(posedge clk);

        // Tagged exponents outside [-96,+31] fail closed.
        @(negedge clk);
        while (!score_start_ready) @(negedge clk);
        lane_count = 1;
        score_start_valid = 1'b1;
        @(negedge clk);
        score_start_valid = 1'b0;
        while (!lane_ready) @(negedge clk);
        q_m = 1; q_e = -8'sd97; k_m = 1; k_e = 0; lane_valid = 1'b1;
        @(negedge clk);
        lane_valid = 1'b0;
        while (!score_valid) @(posedge clk);
        if (!score_desc_error)
            $fatal(1, "SCORE_EXPONENT_ERROR_FAIL");

        $display("TB_PASS native_accumulator_tagged_attention");
        $finish;
    end
endmodule
