`default_nettype none
/* verilator lint_off DECLFILENAME */

// Arithmetic half of layer0_relative_rope_score_fusion_v1. One descriptor
// streams exactly 32 split-half Q/K pairs and produces one signed-64
// precenter score. The signed-70 Scale32 product is rounded without wrap.
module ace2_relative_rope_score_core (
    input  wire                 clk_i,
    input  wire                 rst_ni,
    input  wire                 clear_i,
    input  wire                 start_valid_i,
    output wire                 start_ready_o,
    input  wire [31:0]          query_scale32_i,
    input  wire [31:0]          key_scale32_i,
    input  wire                 pair_valid_i,
    output wire                 pair_ready_o,
    input  wire signed [7:0]    q0_i,
    input  wire signed [7:0]    q1_i,
    input  wire signed [7:0]    k0_i,
    input  wire signed [7:0]    k1_i,
    input  wire signed [15:0]   cosine_i,
    input  wire signed [15:0]   sine_i,
    input  wire                 pair_last_i,
    output wire                 out_valid_o,
    input  wire                 out_ready_i,
    output wire signed [63:0]   precenter_o,
    output wire signed [37:0]   phase_acc_o,
    output wire                 error_o
);
    localparam [1:0] ST_IDLE = 2'd0;
    localparam [1:0] ST_RUN  = 2'd1;
    localparam [1:0] ST_OUT  = 2'd2;

    reg [1:0] state_q;
    reg [5:0] pair_count_q;
    reg signed [37:0] phase_acc_q;
    reg [31:0] significand_product_q;
    reg [6:0] right_shift_q;
    reg signed [63:0] precenter_q;
    reg signed [37:0] phase_result_q;
    reg error_q;

    wire query_scale_valid_w =
        (query_scale32_i[31:24] == 8'd0) &&
        (query_scale32_i[15:0] >= 16'h8000) &&
        ($signed(query_scale32_i[23:16]) >= -8'sd24) &&
        ($signed(query_scale32_i[23:16]) <= 8'sd4);
    wire key_scale_valid_w =
        (key_scale32_i[31:24] == 8'd0) &&
        (key_scale32_i[15:0] >= 16'h8000) &&
        ($signed(key_scale32_i[23:16]) >= -8'sd24) &&
        ($signed(key_scale32_i[23:16]) <= 8'sd4);
    wire signed [8:0] exponent_sum_w =
        $signed({query_scale32_i[23], query_scale32_i[23:16]}) +
        $signed({key_scale32_i[23], key_scale32_i[23:16]});
    wire signed [8:0] right_shift_start_w = 9'sd39 - exponent_sum_w;

    wire signed [15:0] p00_w = q0_i * k0_i;
    wire signed [15:0] p11_w = q1_i * k1_i;
    wire signed [15:0] p10_w = q1_i * k0_i;
    wire signed [15:0] p01_w = q0_i * k1_i;
    wire signed [16:0] a_term_w =
        $signed({p00_w[15], p00_w}) + $signed({p11_w[15], p11_w});
    wire signed [16:0] b_term_w =
        $signed({p10_w[15], p10_w}) - $signed({p01_w[15], p01_w});
    wire signed [32:0] ac_product_w = a_term_w * cosine_i;
    wire signed [32:0] bs_product_w = b_term_w * sine_i;
    wire signed [33:0] pair_term_w =
        $signed({ac_product_w[32], ac_product_w}) -
        $signed({bs_product_w[32], bs_product_w});
    wire signed [37:0] pair_term_ext_w =
        {{4{pair_term_w[33]}}, pair_term_w};
    wire signed [37:0] completed_phase_w = phase_acc_q + pair_term_ext_w;
    wire signed [32:0] significand_product_signed_w =
        $signed({1'b0, significand_product_q});
    wire signed [70:0] scaled_full_w =
        completed_phase_w * significand_product_signed_w;
    wire signed [69:0] scaled_w = scaled_full_w[69:0];
    wire scaled_width_error_w = scaled_full_w[70] != scaled_full_w[69];

    function automatic signed [63:0] rne_shift_s70;
        input signed [69:0] value;
        input [6:0] shift;
        reg negative;
        reg [69:0] magnitude;
        reg [63:0] quotient;
        reg [69:0] remainder;
        reg [69:0] mask;
        reg [69:0] half;
        reg increment;
        reg [63:0] rounded;
        begin
            negative = value[69];
            magnitude = negative ? (~value + 70'd1) : value;
            quotient = 64'd0;
            remainder = 70'd0;
            mask = 70'd0;
            half = 70'd0;
            if (shift == 7'd0) begin
                quotient = magnitude[63:0];
            end else if (shift <= 7'd70) begin
                quotient = 64'(magnitude >> shift);
                if (shift == 7'd70) begin
                    remainder = magnitude;
                    half = 70'd1 << 69;
                end else begin
                    mask = (70'd1 << shift) - 70'd1;
                    remainder = magnitude & mask;
                    half = 70'd1 << (shift - 7'd1);
                end
            end
            increment = (shift != 7'd0) && (shift <= 7'd70) &&
                ((remainder > half) ||
                 ((remainder == half) && quotient[0]));
            // The architecture proof bounds every conforming precenter to
            // signed 64 bits before this narrowing point.
            rounded = quotient + {{63{1'b0}}, increment};
            rne_shift_s70 = negative ? -$signed(rounded) : $signed(rounded);
        end
    endfunction

    assign start_ready_o = state_q == ST_IDLE;
    assign pair_ready_o = state_q == ST_RUN;
    assign out_valid_o = state_q == ST_OUT;
    assign precenter_o = precenter_q;
    assign phase_acc_o = phase_result_q;
    assign error_o = error_q;

    always @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            state_q <= ST_IDLE;
            pair_count_q <= 6'd0;
            phase_acc_q <= 38'sd0;
            significand_product_q <= 32'd0;
            right_shift_q <= 7'd0;
            precenter_q <= 64'sd0;
            phase_result_q <= 38'sd0;
            error_q <= 1'b0;
        end else if (clear_i) begin
            state_q <= ST_IDLE;
            pair_count_q <= 6'd0;
            phase_acc_q <= 38'sd0;
            error_q <= 1'b0;
        end else begin
            case (state_q)
                ST_IDLE: begin
                    if (start_valid_i) begin
                        pair_count_q <= 6'd0;
                        phase_acc_q <= 38'sd0;
                        significand_product_q <=
                            query_scale32_i[15:0] * key_scale32_i[15:0];
                        right_shift_q <= right_shift_start_w[6:0];
                        error_q <= !query_scale_valid_w || !key_scale_valid_w ||
                            (right_shift_start_w < 9'sd31) ||
                            (right_shift_start_w > 9'sd87);
                        state_q <= ST_RUN;
                    end
                end
                ST_RUN: begin
                    if (pair_valid_i) begin
                        if (pair_last_i != (pair_count_q == 6'd31)) begin
                            error_q <= 1'b1;
                        end
                        if (pair_count_q == 6'd31) begin
                            phase_result_q <= completed_phase_w;
                            precenter_q <= rne_shift_s70(scaled_w, right_shift_q);
                            error_q <= error_q || scaled_width_error_w ||
                                (pair_last_i != 1'b1);
                            state_q <= ST_OUT;
                        end else begin
                            phase_acc_q <= completed_phase_w;
                            pair_count_q <= pair_count_q + 6'd1;
                        end
                    end
                end
                ST_OUT: begin
                    if (out_ready_i) begin
                        state_q <= ST_IDLE;
                    end
                end
                default: state_q <= ST_IDLE;
            endcase
        end
    end
endmodule


// Complete-row maximum/emit half. MAX_SCAN never publishes a score;
// CENTER_EMIT uses the unchanged SEALED maximum, and SINGLE stores at most
// 256 signed-64 precenters before publishing a centered row.
module ace2_global_score_center_core (
    input  wire                 clk_i,
    input  wire                 rst_ni,
    input  wire                 clear_i,
    input  wire                 start_valid_i,
    output wire                 start_ready_o,
    input  wire [1:0]           phase_i,
    input  wire [3:0]           query_head_i,
    input  wire [15:0]          query_position_i,
    input  wire [15:0]          key_base_i,
    input  wire [8:0]           key_count_i,
    input  wire                 score_valid_i,
    output wire                 score_ready_o,
    input  wire signed [63:0]   precenter_i,
    output wire                 centered_valid_o,
    input  wire                 centered_ready_i,
    output wire signed [15:0]   centered_o,
    output wire                 done_valid_o,
    input  wire                 done_ready_i,
    output wire                 done_error_o,
    output wire [1:0]           debug_state_o,
    output wire signed [63:0]   debug_global_max_o,
    output wire [15:0]          debug_next_key_base_o
);
    localparam [1:0] PHASE_SINGLE = 2'd0;
    localparam [1:0] PHASE_MAX_SCAN = 2'd1;
    localparam [1:0] PHASE_CENTER_EMIT = 2'd2;
    localparam [1:0] ROW_EMPTY = 2'd0;
    localparam [1:0] ROW_SCAN = 2'd1;
    localparam [1:0] ROW_SEALED = 2'd2;
    localparam [1:0] ROW_EMIT = 2'd3;
    localparam [1:0] ST_IDLE = 2'd0;
    localparam [1:0] ST_INGEST = 2'd1;
    localparam [1:0] ST_SINGLE_EMIT = 2'd2;
    localparam [1:0] ST_DONE = 2'd3;

    reg [1:0] state_q;
    reg [1:0] phase_q;
    reg [3:0] head_q;
    reg [15:0] query_position_q;
    reg [15:0] key_base_q;
    reg [8:0] key_count_q;
    reg [8:0] score_index_q;
    reg [8:0] emit_index_q;
    reg signed [63:0] local_max_q;
    reg local_max_valid_q;
    reg signed [63:0] row_store_q [0:255];
    reg signed [63:0] global_max_q [0:13];
    reg [15:0] global_query_position_q [0:13];
    reg [15:0] global_next_key_base_q [0:13];
    reg [1:0] global_state_q [0:13];
    reg centered_valid_q;
    reg signed [15:0] centered_q;
    reg final_emit_pending_q;
    reg done_error_q;
    integer index;

    wire [16:0] descriptor_end_w =
        {1'b0, key_base_i} + {8'd0, key_count_i};
    wire [16:0] expected_total_w = {1'b0, query_position_i} + 17'd1;
    wire canonical_tile_w =
        (key_base_i[7:0] == 8'd0) &&
        (descriptor_end_w <= expected_total_w) &&
        (((expected_total_w - {1'b0, key_base_i}) > 17'd256) ?
         (key_count_i == 9'd256) :
         ({8'd0, key_count_i} ==
          (expected_total_w - {1'b0, key_base_i})));
    wire phase_single_valid_w =
        (phase_i == PHASE_SINGLE) &&
        (expected_total_w <= 17'd256) &&
        (key_base_i == 16'd0) &&
        (key_count_i == expected_total_w[8:0]);
    wire phase_scan_valid_w =
        (phase_i == PHASE_MAX_SCAN) &&
        (expected_total_w >= 17'd257) && canonical_tile_w &&
        (((key_base_i == 16'd0) &&
          (global_state_q[query_head_i] == ROW_EMPTY)) ||
         ((key_base_i != 16'd0) &&
          (global_state_q[query_head_i] == ROW_SCAN) &&
          (global_query_position_q[query_head_i] == query_position_i) &&
          (global_next_key_base_q[query_head_i] == key_base_i)));
    wire phase_emit_valid_w =
        (phase_i == PHASE_CENTER_EMIT) &&
        (expected_total_w >= 17'd257) && canonical_tile_w &&
        (((key_base_i == 16'd0) &&
          (global_state_q[query_head_i] == ROW_SEALED) &&
          (global_query_position_q[query_head_i] == query_position_i)) ||
         ((key_base_i != 16'd0) &&
          (global_state_q[query_head_i] == ROW_EMIT) &&
          (global_query_position_q[query_head_i] == query_position_i) &&
          (global_next_key_base_q[query_head_i] == key_base_i)));
    wire descriptor_valid_w =
        (query_head_i < 4'd14) && (key_count_i != 9'd0) &&
        (phase_single_valid_w || phase_scan_valid_w || phase_emit_valid_w);
    wire signed [64:0] center_delta_w =
        $signed({precenter_i[63], precenter_i}) -
        $signed({global_max_q[head_q][63], global_max_q[head_q]});
    wire signed [64:0] single_center_delta_w =
        $signed({row_store_q[emit_index_q[7:0]][63],
                 row_store_q[emit_index_q[7:0]]}) -
        $signed({local_max_q[63], local_max_q});

    function automatic signed [15:0] clamp_center;
        input signed [64:0] value;
        begin
            if (value > 65'sd0)
                clamp_center = 16'sd0;
            else if (value < -65'sd32768)
                clamp_center = -16'sd32768;
            else
                clamp_center = value[15:0];
        end
    endfunction

    assign start_ready_o = state_q == ST_IDLE;
    assign score_ready_o = (state_q == ST_INGEST) &&
        ((phase_q != PHASE_CENTER_EMIT) ||
         !centered_valid_q || centered_ready_i);
    assign centered_valid_o = centered_valid_q;
    assign centered_o = centered_q;
    assign done_valid_o = state_q == ST_DONE;
    assign done_error_o = done_error_q;
    assign debug_state_o = global_state_q[query_head_i];
    assign debug_global_max_o = global_max_q[query_head_i];
    assign debug_next_key_base_o = global_next_key_base_q[query_head_i];

    always @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            state_q <= ST_IDLE;
            centered_valid_q <= 1'b0;
            final_emit_pending_q <= 1'b0;
            done_error_q <= 1'b0;
            for (index = 0; index < 14; index = index + 1) begin
                global_max_q[index] <= 64'sd0;
                global_query_position_q[index] <= 16'd0;
                global_next_key_base_q[index] <= 16'd0;
                global_state_q[index] <= ROW_EMPTY;
            end
        end else if (clear_i) begin
            state_q <= ST_IDLE;
            centered_valid_q <= 1'b0;
            final_emit_pending_q <= 1'b0;
            done_error_q <= 1'b0;
            for (index = 0; index < 14; index = index + 1) begin
                global_max_q[index] <= 64'sd0;
                global_query_position_q[index] <= 16'd0;
                global_next_key_base_q[index] <= 16'd0;
                global_state_q[index] <= ROW_EMPTY;
            end
        end else begin
            if (centered_valid_q && centered_ready_i) begin
                centered_valid_q <= 1'b0;
                if (final_emit_pending_q) begin
                    final_emit_pending_q <= 1'b0;
                    if (phase_q == PHASE_CENTER_EMIT) begin
                        if ({1'b0, key_base_q} + {8'd0, key_count_q} ==
                            {1'b0, query_position_q} + 17'd1) begin
                            global_max_q[head_q] <= 64'sd0;
                            global_query_position_q[head_q] <= 16'd0;
                            global_next_key_base_q[head_q] <= 16'd0;
                            global_state_q[head_q] <= ROW_EMPTY;
                        end else begin
                            global_next_key_base_q[head_q] <=
                                key_base_q + {7'd0, key_count_q};
                            global_state_q[head_q] <= ROW_EMIT;
                        end
                    end
                    state_q <= ST_DONE;
                end
            end

            case (state_q)
                ST_IDLE: begin
                    if (start_valid_i) begin
                        phase_q <= phase_i;
                        head_q <= query_head_i;
                        query_position_q <= query_position_i;
                        key_base_q <= key_base_i;
                        key_count_q <= key_count_i;
                        score_index_q <= 9'd0;
                        emit_index_q <= 9'd0;
                        local_max_q <= 64'sd0;
                        local_max_valid_q <= 1'b0;
                        centered_valid_q <= 1'b0;
                        final_emit_pending_q <= 1'b0;
                        done_error_q <= !descriptor_valid_w;
                        state_q <= descriptor_valid_w ? ST_INGEST : ST_DONE;
                    end
                end
                ST_INGEST: begin
                    if (score_valid_i && score_ready_o) begin
                        if (phase_q == PHASE_SINGLE) begin
                            row_store_q[score_index_q[7:0]] <= precenter_i;
                        end
                        if ((phase_q == PHASE_SINGLE) ||
                            (phase_q == PHASE_MAX_SCAN)) begin
                            if (!local_max_valid_q || precenter_i > local_max_q) begin
                                local_max_q <= precenter_i;
                            end
                            local_max_valid_q <= 1'b1;
                        end
                        if (phase_q == PHASE_CENTER_EMIT) begin
                            centered_q <= clamp_center(center_delta_w);
                            centered_valid_q <= 1'b1;
                            if (score_index_q + 9'd1 == key_count_q) begin
                                final_emit_pending_q <= 1'b1;
                            end
                        end
                        if (score_index_q + 9'd1 == key_count_q) begin
                            if (phase_q == PHASE_SINGLE) begin
                                state_q <= ST_SINGLE_EMIT;
                                emit_index_q <= 9'd0;
                            end else if (phase_q == PHASE_MAX_SCAN) begin
                                global_max_q[head_q] <=
                                    (!local_max_valid_q || precenter_i > local_max_q) ?
                                    precenter_i : local_max_q;
                                global_query_position_q[head_q] <= query_position_q;
                                global_next_key_base_q[head_q] <=
                                    key_base_q + {7'd0, key_count_q};
                                global_state_q[head_q] <=
                                    ({1'b0, key_base_q} + {8'd0, key_count_q} ==
                                     {1'b0, query_position_q} + 17'd1) ?
                                    ROW_SEALED : ROW_SCAN;
                                state_q <= ST_DONE;
                            end
                        end else begin
                            score_index_q <= score_index_q + 9'd1;
                        end
                    end
                end
                ST_SINGLE_EMIT: begin
                    if (!centered_valid_q || centered_ready_i) begin
                        centered_q <= clamp_center(single_center_delta_w);
                        centered_valid_q <= 1'b1;
                        if (emit_index_q + 9'd1 == key_count_q) begin
                            final_emit_pending_q <= 1'b1;
                        end else begin
                            emit_index_q <= emit_index_q + 9'd1;
                        end
                    end
                end
                ST_DONE: begin
                    if (done_ready_i) begin
                        state_q <= ST_IDLE;
                        done_error_q <= 1'b0;
                    end
                end
                default: state_q <= ST_IDLE;
            endcase
        end
    end
endmodule

/* verilator lint_on DECLFILENAME */
`default_nettype wire
