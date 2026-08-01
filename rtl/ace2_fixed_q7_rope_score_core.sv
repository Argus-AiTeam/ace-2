`default_nettype none

module ace2_fixed_q7_rope_score_core (
    input  wire             clk_i,
    input  wire             rst_ni,
    input  wire             clear_i,
    input  wire             start_valid_i,
    output wire             start_ready_o,
    input  wire             score_mode_i,
    input  wire [511:0]     act_data_i,
    input  wire [1023:0]    cos_q15_i,
    input  wire [1023:0]    sin_q15_i,
    input  wire [1023:0]    query_q7_i,
    input  wire [1023:0]    key_q7_i,
    input  wire [31:0]      query_scale32_i,
    input  wire [31:0]      key_scale32_i,
    output wire             out_valid_o,
    input  wire             out_ready_i,
    output wire [1023:0]    rope_q7_o,
    output wire signed [63:0] precenter_q6_9_o,
    output wire signed [37:0] dot_q7xq7_o,
    output wire [8:0]       multiplier_cycles_o,
    output wire             error_valid_o,
    output wire             coefficient_error_o,
    output wire             numeric_overflow_o
);
    localparam [2:0] ST_IDLE  = 3'd0;
    localparam [2:0] ST_ROPE  = 3'd1;
    localparam [2:0] ST_SCORE = 3'd2;
    localparam [2:0] ST_SCALE = 3'd3;
    localparam [2:0] ST_DONE  = 3'd4;
    localparam [2:0] ST_ERROR = 3'd5;

    reg [2:0] state_q;
    reg [5:0] index_q;
    reg [1:0] phase_q;
    reg [511:0] act_data_q;
    reg [1023:0] cos_q15_q;
    reg [1023:0] sin_q15_q;
    reg [1023:0] query_q7_q;
    reg [1023:0] key_q7_q;
    reg [31:0] query_scale32_q;
    reg [31:0] key_scale32_q;
    reg [1023:0] rope_q7_q;
    reg signed [32:0] lane_partial_q;
    reg signed [37:0] dot_q;
    reg signed [63:0] precenter_q;
    reg [8:0] multiplier_cycles_q;
    reg out_valid_q;
    reg error_valid_q;
    reg coefficient_error_q;
    reg numeric_overflow_q;

    reg signed [7:0] rope_x0_w;
    reg signed [7:0] rope_x1_w;
    reg signed [15:0] rope_c0_w;
    reg signed [15:0] rope_s0_w;
    reg signed [15:0] rope_c1_w;
    reg signed [15:0] rope_s1_w;
    reg signed [24:0] rope_r0_w;
    reg signed [24:0] rope_r1_w;
    reg [16:0] rope_abs_c_w;
    reg [16:0] rope_abs_s_w;
    reg signed [15:0] rope_w0_w;
    reg signed [15:0] rope_w1_w;
    reg rope_coeff_invalid_w;
    reg rope_numeric_invalid_w;

    reg signed [15:0] score_q_w;
    reg signed [15:0] score_k_w;
    reg signed [7:0] score_ql_w;
    reg signed [7:0] score_qh_w;
    reg signed [7:0] score_kl_w;
    reg signed [7:0] score_kh_w;
    reg score_cq_w;
    reg score_ck_w;
    reg signed [15:0] score_mul_w;
    reg signed [32:0] score_term_w;
    reg signed [32:0] score_lane_product_w;
    reg signed [37:0] score_dot_next_w;

    reg [31:0] scale_sig_product_w;
    reg [16:0] scale_pair_sig_w;
    reg signed [8:0] scale_exp_sum_w;
    reg [6:0] scale_shift_w;
    reg signed [54:0] scale_product_w;
    reg signed [63:0] scale_rounded_w;
    reg signed [8:0] scale_shift_signed_w;
    reg scale_invalid_w;

    function automatic signed [15:0] rne_shift8_s25;
        input signed [24:0] value;
        reg negative;
        reg [24:0] magnitude;
        reg [16:0] quotient;
        reg [7:0] remainder;
        reg increment;
        begin
            negative = value[24];
            magnitude = negative ? (~value + 25'd1) : value;
            quotient = magnitude[24:8];
            remainder = magnitude[7:0];
            increment = (remainder > 8'h80) ||
                        ((remainder == 8'h80) && quotient[0]);
            quotient = quotient + {{16{1'b0}}, increment};
            rne_shift8_s25 = negative ? -$signed(quotient[15:0]) : $signed(quotient[15:0]);
        end
    endfunction

    function automatic signed [63:0] rne_shift_s55;
        input signed [54:0] value;
        input [6:0] shift;
        reg negative;
        reg [127:0] magnitude;
        reg [127:0] quotient;
        reg [127:0] mask;
        reg [127:0] remainder;
        reg [127:0] half;
        reg increment;
        begin
            negative = value[54];
            magnitude = negative ? {{73{1'b0}}, (~value + 55'd1)} : {{73{1'b0}}, value};
            quotient = magnitude >> shift;
            mask = (128'd1 << shift) - 128'd1;
            remainder = magnitude & mask;
            half = 128'd1 << (shift - 1'b1);
            increment = (remainder > half) ||
                        ((remainder == half) && quotient[0]);
            quotient = quotient + {{127{1'b0}}, increment};
            rne_shift_s55 = negative ? -$signed(quotient[63:0]) : $signed(quotient[63:0]);
        end
    endfunction

    wire signed [7:0] query_exp_w = query_scale32_q[23:16];
    wire signed [7:0] key_exp_w = key_scale32_q[23:16];
    wire query_scale_valid_w =
        (query_scale32_q[31:24] == 8'h00) &&
        (query_scale32_q[15:0] >= 16'h8000) &&
        (query_exp_w >= -8'sd24) && (query_exp_w <= 8'sd4);
    wire key_scale_valid_w =
        (key_scale32_q[31:24] == 8'h00) &&
        (key_scale32_q[15:0] >= 16'h8000) &&
        (key_exp_w >= -8'sd24) && (key_exp_w <= 8'sd4);

    assign start_ready_o = (state_q == ST_IDLE);
    assign out_valid_o = out_valid_q;
    assign rope_q7_o = rope_q7_q;
    assign precenter_q6_9_o = precenter_q;
    assign dot_q7xq7_o = dot_q;
    assign multiplier_cycles_o = multiplier_cycles_q;
    assign error_valid_o = error_valid_q;
    assign coefficient_error_o = coefficient_error_q;
    assign numeric_overflow_o = numeric_overflow_q;

    always @* begin
        rope_x0_w = $signed(act_data_q[index_q*8 +: 8]);
        rope_x1_w = $signed(act_data_q[(index_q + 6'd32)*8 +: 8]);
        rope_c0_w = $signed(cos_q15_q[index_q*16 +: 16]);
        rope_s0_w = $signed(sin_q15_q[index_q*16 +: 16]);
        rope_c1_w = $signed(cos_q15_q[(index_q + 6'd32)*16 +: 16]);
        rope_s1_w = $signed(sin_q15_q[(index_q + 6'd32)*16 +: 16]);
        rope_abs_c_w = rope_c0_w[15] ?
            -$signed({rope_c0_w[15], rope_c0_w}) : {1'b0, rope_c0_w};
        rope_abs_s_w = rope_s0_w[15] ?
            -$signed({rope_s0_w[15], rope_s0_w}) : {1'b0, rope_s0_w};
        rope_r0_w = rope_x0_w * rope_c0_w - rope_x1_w * rope_s0_w;
        rope_r1_w = rope_x0_w * rope_s0_w + rope_x1_w * rope_c0_w;
        rope_w0_w = rne_shift8_s25(rope_r0_w);
        rope_w1_w = rne_shift8_s25(rope_r1_w);
        rope_coeff_invalid_w = (rope_c0_w != rope_c1_w) ||
                               (rope_s0_w != rope_s1_w) ||
                               ((rope_abs_c_w + rope_abs_s_w) > 17'd46342);
        rope_numeric_invalid_w =
            ($signed(rope_r0_w) > 25'sd5931776) ||
            ($signed(rope_r0_w) < -25'sd5931776) ||
            ($signed(rope_r1_w) > 25'sd5931776) ||
            ($signed(rope_r1_w) < -25'sd5931776) ||
            ($signed(rope_w0_w) > 16'sd23171) ||
            ($signed(rope_w0_w) < -16'sd23171) ||
            ($signed(rope_w1_w) > 16'sd23171) ||
            ($signed(rope_w1_w) < -16'sd23171);
    end

    always @* begin
        score_q_w = $signed(query_q7_q[index_q*16 +: 16]);
        score_k_w = $signed(key_q7_q[index_q*16 +: 16]);
        score_ql_w = score_q_w[7:0];
        score_qh_w = score_q_w[15:8];
        score_kl_w = score_k_w[7:0];
        score_kh_w = score_k_w[15:8];
        score_cq_w = score_q_w[7];
        score_ck_w = score_k_w[7];
        case (phase_q)
            2'd0: begin
                score_mul_w = score_ql_w * score_kl_w;
                score_term_w = {{17{score_mul_w[15]}}, score_mul_w} +
                    (score_cq_w ? ($signed({{25{score_kl_w[7]}}, score_kl_w}) <<< 8) : 33'sd0) +
                    (score_ck_w ? ($signed({{25{score_ql_w[7]}}, score_ql_w}) <<< 8) : 33'sd0) +
                    ((score_cq_w && score_ck_w) ? 33'sd65536 : 0);
            end
            2'd1: begin
                score_mul_w = score_qh_w * score_kl_w;
                score_term_w = {{17{score_mul_w[15]}}, score_mul_w} +
                    (score_ck_w ? ($signed({{25{score_qh_w[7]}}, score_qh_w}) <<< 8) : 33'sd0);
            end
            2'd2: begin
                score_mul_w = score_ql_w * score_kh_w;
                score_term_w = {{17{score_mul_w[15]}}, score_mul_w} +
                    (score_cq_w ? ($signed({{25{score_kh_w[7]}}, score_kh_w}) <<< 8) : 33'sd0);
            end
            default: begin
                score_mul_w = score_qh_w * score_kh_w;
                score_term_w = {{17{score_mul_w[15]}}, score_mul_w};
            end
        endcase
        score_lane_product_w = lane_partial_q + ($signed(score_term_w) <<<
            ((phase_q == 2'd1 || phase_q == 2'd2) ? 8 :
             (phase_q == 2'd3 ? 16 : 0)));
        score_dot_next_w = dot_q + {{5{score_lane_product_w[32]}}, score_lane_product_w};
    end

    always @* begin
        scale_sig_product_w = query_scale32_q[15:0] * key_scale32_q[15:0];
        scale_pair_sig_w = scale_sig_product_w[31:15] +
            {{16{1'b0}},
            ((scale_sig_product_w[14:0] > 15'h4000) ||
             ((scale_sig_product_w[14:0] == 15'h4000) && scale_sig_product_w[15]))};
        scale_exp_sum_w = $signed(query_exp_w) + $signed(key_exp_w);
        scale_shift_signed_w = 9'sd23 - scale_exp_sum_w;
        scale_shift_w = scale_shift_signed_w[6:0];
        scale_product_w = $signed(dot_q) * $signed({1'b0, scale_pair_sig_w});
        scale_rounded_w = rne_shift_s55(scale_product_w, scale_shift_w);
        scale_invalid_w = !query_scale_valid_w || !key_scale_valid_w ||
                          (scale_exp_sum_w < -9'sd48) ||
                          (scale_exp_sum_w > 9'sd8) ||
                          (scale_shift_signed_w < 9'sd15) ||
                          (scale_shift_signed_w > 9'sd71);
    end

    always @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            state_q <= ST_IDLE;
            index_q <= 6'd0;
            phase_q <= 2'd0;
            rope_q7_q <= 1024'd0;
            lane_partial_q <= 33'sd0;
            dot_q <= 38'sd0;
            precenter_q <= 64'sd0;
            multiplier_cycles_q <= 9'd0;
            out_valid_q <= 1'b0;
            error_valid_q <= 1'b0;
            coefficient_error_q <= 1'b0;
            numeric_overflow_q <= 1'b0;
        end else if (clear_i) begin
            state_q <= ST_IDLE;
            index_q <= 6'd0;
            phase_q <= 2'd0;
            rope_q7_q <= 1024'd0;
            lane_partial_q <= 33'sd0;
            dot_q <= 38'sd0;
            precenter_q <= 64'sd0;
            multiplier_cycles_q <= 9'd0;
            out_valid_q <= 1'b0;
            error_valid_q <= 1'b0;
            coefficient_error_q <= 1'b0;
            numeric_overflow_q <= 1'b0;
        end else begin
            case (state_q)
                ST_IDLE: begin
                    out_valid_q <= 1'b0;
                    error_valid_q <= 1'b0;
                    coefficient_error_q <= 1'b0;
                    numeric_overflow_q <= 1'b0;
                    if (start_valid_i) begin
                        act_data_q <= act_data_i;
                        cos_q15_q <= cos_q15_i;
                        sin_q15_q <= sin_q15_i;
                        query_q7_q <= query_q7_i;
                        key_q7_q <= key_q7_i;
                        query_scale32_q <= query_scale32_i;
                        key_scale32_q <= key_scale32_i;
                        index_q <= 6'd0;
                        phase_q <= 2'd0;
                        rope_q7_q <= 1024'd0;
                        lane_partial_q <= 33'sd0;
                        dot_q <= 38'sd0;
                        precenter_q <= 64'sd0;
                        multiplier_cycles_q <= 9'd0;
                        state_q <= score_mode_i ? ST_SCORE : ST_ROPE;
                    end
                end
                ST_ROPE: begin
                    if (rope_coeff_invalid_w) begin
                        error_valid_q <= 1'b1;
                        coefficient_error_q <= 1'b1;
                        state_q <= ST_ERROR;
                    end else if (rope_numeric_invalid_w) begin
                        error_valid_q <= 1'b1;
                        numeric_overflow_q <= 1'b1;
                        state_q <= ST_ERROR;
                    end else begin
                        rope_q7_q[index_q*16 +: 16] <= rope_w0_w;
                        rope_q7_q[(index_q + 6'd32)*16 +: 16] <= rope_w1_w;
                        if (index_q == 6'd31) begin
                            out_valid_q <= 1'b1;
                            state_q <= ST_DONE;
                        end else begin
                            index_q <= index_q + 6'd1;
                        end
                    end
                end
                ST_SCORE: begin
                    multiplier_cycles_q <= multiplier_cycles_q + 9'd1;
                    if (phase_q == 2'd0) begin
                        lane_partial_q <= score_term_w;
                        phase_q <= 2'd1;
                    end else if (phase_q == 2'd3) begin
                        dot_q <= score_dot_next_w;
                        lane_partial_q <= 33'sd0;
                        phase_q <= 2'd0;
                        if (index_q == 6'd63) begin
                            state_q <= ST_SCALE;
                        end else begin
                            index_q <= index_q + 6'd1;
                        end
                    end else begin
                        lane_partial_q <= score_lane_product_w;
                        phase_q <= phase_q + 2'd1;
                    end
                end
                ST_SCALE: begin
                    if (scale_invalid_w) begin
                        error_valid_q <= 1'b1;
                        numeric_overflow_q <= 1'b1;
                        state_q <= ST_ERROR;
                    end else begin
                        precenter_q <= scale_rounded_w;
                        out_valid_q <= 1'b1;
                        state_q <= ST_DONE;
                    end
                end
                ST_DONE: begin
                    if (out_valid_q && out_ready_i) begin
                        out_valid_q <= 1'b0;
                        state_q <= ST_IDLE;
                    end
                end
                ST_ERROR: error_valid_q <= 1'b1;
                default: begin
                    error_valid_q <= 1'b1;
                    numeric_overflow_q <= 1'b1;
                    state_q <= ST_ERROR;
                end
            endcase
        end
    end

`ifndef SYNTHESIS
    always @(posedge clk_i) begin
        if (out_valid_q && error_valid_q)
            $error("fixed-Q7 core exposed success and error simultaneously");
        if ((state_q == ST_DONE) && !out_ready_i && !out_valid_q)
            $error("fixed-Q7 core dropped a backpressured result");
    end
`endif
endmodule

`default_nettype wire
