`default_nettype none

module ace2_dynamic_rope_head_core (
    input  wire             clk_i,
    input  wire             rst_ni,
    input  wire             clear_i,
    input  wire             start_valid_i,
    output wire             start_ready_o,
    input  wire [511:0]     act_data_i,
    input  wire [31:0]      producer_scale32_i,
    input  wire [1023:0]    cos_q15_i,
    input  wire [1023:0]    sin_q15_i,
    output wire             out_valid_o,
    input  wire             out_ready_i,
    output wire [511:0]     out_data_o,
    output wire [31:0]      output_scale32_o,
    output wire             error_valid_o,
    output wire             numeric_overflow_o
);
    localparam [2:0] ST_IDLE    = 3'd0;
    localparam [2:0] ST_ROTATE  = 3'd1;
    localparam [2:0] ST_ENCODE  = 3'd2;
    localparam [2:0] ST_REQUANT = 3'd3;
    localparam [2:0] ST_DONE    = 3'd4;
    localparam [2:0] ST_ERROR   = 3'd5;
    localparam [31:0] SCALE32_ZERO = 32'h00e88000;

    reg [2:0] state_q;
    reg [4:0] pair_index_q;
    reg [4:0] encode_cycle_q;
    reg [1:0] requant_group_q;
    reg [511:0] act_data_q;
    reg [1023:0] cos_q15_q;
    reg [1023:0] sin_q15_q;
    reg [31:0] producer_scale32_q;
    reg signed [24:0] rotated_q [0:63];
    reg [24:0] maximum_q;
    reg [31:0] output_scale32_q;
    reg scale_encode_valid_q;
    reg [511:0] out_data_q;
    reg out_valid_q;
    reg error_valid_q;
    reg numeric_overflow_q;

    wire signed [7:0] producer_exp_w = producer_scale32_q[23:16];
    wire producer_scale_valid_w =
        (producer_scale32_q[31:24] == 8'h00) &&
        (producer_scale32_q[15:0] >= 16'h8000) &&
        (producer_exp_w >= -8'sd24) &&
        (producer_exp_w <= 8'sd4);

    reg signed [7:0] rotate_x0_w;
    reg signed [7:0] rotate_x1_w;
    reg signed [15:0] rotate_c0_w;
    reg signed [15:0] rotate_s0_w;
    reg signed [15:0] rotate_c1_w;
    reg signed [15:0] rotate_s1_w;
    reg signed [23:0] rotate_x0c0_w;
    reg signed [23:0] rotate_x1s0_w;
    reg signed [23:0] rotate_x1c1_w;
    reg signed [23:0] rotate_x0s1_w;
    reg signed [24:0] rotate_r0_w;
    reg signed [24:0] rotate_r1_w;
    reg [24:0] rotate_abs0_w;
    reg [24:0] rotate_abs1_w;
    reg [24:0] rotate_max_w;

    wire [32:0] encoded_scale_w;
    reg [127:0] requant_group_data_w;
    reg requant_overflow_w;
    integer requant_lane;
    integer requant_index;
    integer requant_delta;
    reg [24:0] requant_magnitude;
    reg [95:0] requant_numerator;
    reg [95:0] requant_denominator;
    reg [95:0] requant_quotient;
    reg [95:0] requant_remainder;
    reg requant_increment;
    reg [7:0] requant_byte;
    integer reset_lane;

    function automatic [32:0] encode_scale32;
        input [15:0] producer_significand;
        input signed [7:0] producer_exponent;
        input [24:0] maximum_magnitude;
        integer exponent_value;
        reg found;
        reg [95:0] numerator;
        reg [95:0] denominator;
        reg [95:0] scaled_numerator;
        reg [95:0] candidate_significand;
        reg [7:0] exponent_bits;
        begin
            encode_scale32 = 33'd0;
            if (maximum_magnitude == 25'd0) begin
                encode_scale32 = {1'b1, SCALE32_ZERO};
            end else begin
                numerator = producer_significand * maximum_magnitude;
                denominator = 96'd127 << 30;
                if (producer_exponent >= 0) begin
                    numerator = numerator << producer_exponent;
                end else begin
                    denominator = denominator << (-producer_exponent);
                end
                found = 1'b0;
                for (exponent_value = -24; exponent_value <= 4; exponent_value = exponent_value + 1) begin
                    if (!found) begin
                        scaled_numerator = numerator << (15 - exponent_value);
                        candidate_significand =
                            (scaled_numerator + denominator - 96'd1) / denominator;
                        if (candidate_significand < 96'h8000) begin
                            candidate_significand = 96'h8000;
                        end
                        if (candidate_significand <= 96'hffff) begin
                            exponent_bits = exponent_value[7:0];
                            encode_scale32 = {
                                1'b1, 8'h00, exponent_bits,
                                candidate_significand[15:0]
                            };
                            found = 1'b1;
                        end
                    end
                end
            end
        end
    endfunction

    assign encoded_scale_w = encode_scale32(
        producer_scale32_q[15:0], producer_exp_w, maximum_q
    );
    assign start_ready_o = (state_q == ST_IDLE);
    assign out_valid_o = out_valid_q;
    assign out_data_o = out_data_q;
    assign output_scale32_o = output_scale32_q;
    assign error_valid_o = error_valid_q;
    assign numeric_overflow_o = numeric_overflow_q;

    always @* begin
        rotate_x0_w = $signed(act_data_q[pair_index_q*8 +: 8]);
        rotate_x1_w = $signed(act_data_q[(pair_index_q + 6'd32)*8 +: 8]);
        rotate_c0_w = $signed(cos_q15_q[pair_index_q*16 +: 16]);
        rotate_s0_w = $signed(sin_q15_q[pair_index_q*16 +: 16]);
        rotate_c1_w = $signed(cos_q15_q[(pair_index_q + 6'd32)*16 +: 16]);
        rotate_s1_w = $signed(sin_q15_q[(pair_index_q + 6'd32)*16 +: 16]);
        rotate_x0c0_w = rotate_x0_w * rotate_c0_w;
        rotate_x1s0_w = rotate_x1_w * rotate_s0_w;
        rotate_x1c1_w = rotate_x1_w * rotate_c1_w;
        rotate_x0s1_w = rotate_x0_w * rotate_s1_w;
        rotate_r0_w = $signed(rotate_x0c0_w) - $signed(rotate_x1s0_w);
        rotate_r1_w = $signed(rotate_x1c1_w) + $signed(rotate_x0s1_w);
        rotate_abs0_w = rotate_r0_w[24] ? (~rotate_r0_w + 25'd1) : rotate_r0_w;
        rotate_abs1_w = rotate_r1_w[24] ? (~rotate_r1_w + 25'd1) : rotate_r1_w;
        rotate_max_w = maximum_q;
        if (rotate_abs0_w > rotate_max_w) rotate_max_w = rotate_abs0_w;
        if (rotate_abs1_w > rotate_max_w) rotate_max_w = rotate_abs1_w;
    end

    always @* begin
        requant_group_data_w = 128'd0;
        requant_overflow_w = 1'b0;
        requant_delta = $signed(producer_scale32_q[23:16]) -
                         $signed(output_scale32_q[23:16]) - 15;
        requant_magnitude = 25'd0;
        requant_numerator = 96'd0;
        requant_denominator = 96'd1;
        requant_quotient = 96'd0;
        requant_remainder = 96'd0;
        requant_increment = 1'b0;
        requant_byte = 8'd0;
        for (requant_lane = 0; requant_lane < 16; requant_lane = requant_lane + 1) begin
            requant_index = requant_group_q * 16 + requant_lane;
            requant_magnitude = rotated_q[requant_index][24] ?
                (~rotated_q[requant_index] + 25'd1) : rotated_q[requant_index];
            requant_numerator = requant_magnitude * producer_scale32_q[15:0];
            requant_denominator = output_scale32_q[15:0];
            if (requant_delta >= 0) begin
                requant_numerator = requant_numerator << requant_delta;
            end else begin
                requant_denominator = requant_denominator << (-requant_delta);
            end
            requant_quotient = requant_numerator / requant_denominator;
            requant_remainder = requant_numerator % requant_denominator;
            requant_increment =
                ((requant_remainder << 1) > requant_denominator) ||
                (((requant_remainder << 1) == requant_denominator) && requant_quotient[0]);
            requant_quotient = requant_quotient + requant_increment;
            if (requant_quotient > 96'd127) begin
                requant_overflow_w = 1'b1;
                requant_byte = rotated_q[requant_index][24] ? 8'h81 : 8'h7f;
            end else if (rotated_q[requant_index][24] && (requant_quotient != 0)) begin
                requant_byte = (~requant_quotient[7:0]) + 8'd1;
            end else begin
                requant_byte = requant_quotient[7:0];
            end
            requant_group_data_w[requant_lane*8 +: 8] = requant_byte;
        end
    end

    always @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            state_q <= ST_IDLE;
            pair_index_q <= 5'd0;
            encode_cycle_q <= 5'd0;
            requant_group_q <= 2'd0;
            act_data_q <= 512'd0;
            cos_q15_q <= 1024'd0;
            sin_q15_q <= 1024'd0;
            producer_scale32_q <= 32'd0;
            maximum_q <= 25'd0;
            output_scale32_q <= 32'd0;
            scale_encode_valid_q <= 1'b0;
            out_data_q <= 512'd0;
            out_valid_q <= 1'b0;
            error_valid_q <= 1'b0;
            numeric_overflow_q <= 1'b0;
            for (reset_lane = 0; reset_lane < 64; reset_lane = reset_lane + 1) begin
                rotated_q[reset_lane] <= 25'sd0;
            end
        end else if (clear_i) begin
            state_q <= ST_IDLE;
            pair_index_q <= 5'd0;
            encode_cycle_q <= 5'd0;
            requant_group_q <= 2'd0;
            maximum_q <= 25'd0;
            output_scale32_q <= 32'd0;
            scale_encode_valid_q <= 1'b0;
            out_data_q <= 512'd0;
            out_valid_q <= 1'b0;
            error_valid_q <= 1'b0;
            numeric_overflow_q <= 1'b0;
        end else begin
            case (state_q)
                ST_IDLE: begin
                    out_valid_q <= 1'b0;
                    error_valid_q <= 1'b0;
                    numeric_overflow_q <= 1'b0;
                    if (start_valid_i) begin
                        act_data_q <= act_data_i;
                        cos_q15_q <= cos_q15_i;
                        sin_q15_q <= sin_q15_i;
                        producer_scale32_q <= producer_scale32_i;
                        pair_index_q <= 5'd0;
                        maximum_q <= 25'd0;
                        out_data_q <= 512'd0;
                        if ((producer_scale32_i[31:24] != 8'h00) ||
                            (producer_scale32_i[15:0] < 16'h8000) ||
                            ($signed(producer_scale32_i[23:16]) < -8'sd24) ||
                            ($signed(producer_scale32_i[23:16]) > 8'sd4)) begin
                            error_valid_q <= 1'b1;
                            state_q <= ST_ERROR;
                        end else begin
                            state_q <= ST_ROTATE;
                        end
                    end
                end
                ST_ROTATE: begin
                    rotated_q[pair_index_q] <= rotate_r0_w;
                    rotated_q[pair_index_q + 6'd32] <= rotate_r1_w;
                    maximum_q <= rotate_max_w;
                    if (pair_index_q == 5'd31) begin
                        encode_cycle_q <= 5'd0;
                        state_q <= ST_ENCODE;
                    end else begin
                        pair_index_q <= pair_index_q + 5'd1;
                    end
                end
                ST_ENCODE: begin
                    if (encode_cycle_q == 5'd0) begin
                        output_scale32_q <= encoded_scale_w[31:0];
                        scale_encode_valid_q <= encoded_scale_w[32];
                    end
                    if (encode_cycle_q == 5'd17) begin
                        if (!scale_encode_valid_q) begin
                            error_valid_q <= 1'b1;
                            numeric_overflow_q <= 1'b1;
                            state_q <= ST_ERROR;
                        end else begin
                            requant_group_q <= 2'd0;
                            state_q <= ST_REQUANT;
                        end
                    end else begin
                        encode_cycle_q <= encode_cycle_q + 5'd1;
                    end
                end
                ST_REQUANT: begin
                    out_data_q[requant_group_q*128 +: 128] <= requant_group_data_w;
                    if (requant_overflow_w) begin
                        error_valid_q <= 1'b1;
                        numeric_overflow_q <= 1'b1;
                        state_q <= ST_ERROR;
                    end else if (requant_group_q == 2'd3) begin
                        out_valid_q <= 1'b1;
                        state_q <= ST_DONE;
                    end else begin
                        requant_group_q <= requant_group_q + 2'd1;
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
        if (rst_ni && (state_q != ST_IDLE) && !producer_scale_valid_w &&
            (state_q != ST_ERROR)) begin
            $error("dynamic RoPE accepted an invalid ProducerScale32 record");
        end
        if (rst_ni && out_valid_q && error_valid_q) begin
            $error("dynamic RoPE exposed success and error simultaneously");
        end
    end
`endif
endmodule

`default_nettype wire
