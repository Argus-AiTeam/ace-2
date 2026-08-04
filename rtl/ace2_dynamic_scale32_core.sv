`timescale 1ns/1ps
`default_nettype none
/* verilator lint_off DECLFILENAME */

// Shared token/group dynamic Scale32 primitives for
// shared_token_group_dynamic_scale32_v1.
//
// The group core buffers exact signed wide values expressed in units of the
// immutable baseline Scale32 record, selects the smallest legal power-of-two
// exponent delta, and emits symmetric signed-int8 mantissas.  Two 128x40-bit
// banks implement the frozen 1,280-byte ping-pong allocation.
module ace2_dynamic_scale32_group_core #(
    parameter integer VALUE_WIDTH = 40,
    parameter integer LANES_PER_BEAT = 16,
    parameter integer MAX_GROUP_LANES = 128
) (
    input  logic                              clk_i,
    input  logic                              rst_ni,
    input  logic                              clear_i,

    input  logic                              group_start_valid_i,
    output logic                              group_start_ready_o,
    input  logic [7:0]                        group_lanes_u8_i,
    input  logic [31:0]                       base_scale32_i,
    input  logic [5:0]                        group_index_u6_i,
    input  logic [15:0]                       tensor_tag_u16_i,

    input  logic                              value_valid_i,
    output logic                              value_ready_o,
    input  logic signed [LANES_PER_BEAT*VALUE_WIDTH-1:0] value_s40_i,

    output logic                              payload_valid_o,
    input  logic                              payload_ready_i,
    output logic signed [LANES_PER_BEAT*8-1:0] payload_s8_o,
    output logic                              payload_last_o,

    output logic                              group_commit_valid_o,
    input  logic                              group_commit_ready_i,
    output logic [5:0]                        group_index_u6_o,
    output logic [15:0]                       tensor_tag_u16_o,
    output logic signed [6:0]                 exponent_delta_s7_o,
    output logic [31:0]                       effective_scale32_o,
    output logic                              descriptor_error_o,
    output logic                              numeric_overflow_o
);
    localparam logic [2:0] BANK_EMPTY   = 3'd0;
    localparam logic [2:0] BANK_CAPTURE = 3'd1;
    localparam logic [2:0] BANK_READY   = 3'd2;
    localparam logic [2:0] BANK_EMIT    = 3'd3;
    localparam logic [2:0] BANK_COMMIT  = 3'd4;

    logic signed [VALUE_WIDTH-1:0] value_mem_q [0:1][0:MAX_GROUP_LANES-1];
    logic [2:0] bank_state_q [0:1];
    logic [7:0] group_lanes_q [0:1];
    logic [31:0] base_scale32_q [0:1];
    logic [5:0] group_index_q [0:1];
    logic [15:0] tensor_tag_q [0:1];
    logic [VALUE_WIDTH-1:0] maximum_abs_q [0:1];
    logic signed [6:0] selected_delta_q [0:1];
    logic [31:0] effective_scale32_q [0:1];
    logic descriptor_error_q [0:1];
    logic numeric_overflow_q [0:1];

    logic write_bank_q;
    logic read_bank_q;
    logic capture_active_q;
    logic capture_bank_q;
    logic [2:0] capture_beat_q;
    logic [2:0] emit_beat_q [0:1];

    logic scale_valid_comb;
    logic signed [7:0] base_exponent_comb;
    logic select_found_comb;
    logic signed [6:0] select_delta_comb;
    logic [63:0] selected_magnitude_comb;
    logic [63:0] quotient_comb;
    logic [63:0] remainder_comb;
    logic [63:0] half_comb;
    logic signed [8:0] effective_exponent_comb;
    logic signed [6:0] delta_scan_s7;
    logic signed [LANES_PER_BEAT*64-1:0] quantized_lane_comb;
    logic signed [MAX_GROUP_LANES*VALUE_WIDTH-1:0] value_mem_bank0_comb;
    logic signed [MAX_GROUP_LANES*VALUE_WIDTH-1:0] value_mem_bank1_comb;
    logic signed [MAX_GROUP_LANES*VALUE_WIDTH-1:0] read_value_mem_comb;
    logic [2:0] write_bank_state_comb;
    logic [2:0] read_bank_state_comb;
    logic [7:0] read_group_lanes_comb;
    logic [31:0] read_base_scale32_comb;
    logic [5:0] read_group_index_comb;
    logic [15:0] read_tensor_tag_comb;
    logic [VALUE_WIDTH-1:0] read_maximum_abs_comb;
    logic signed [6:0] read_selected_delta_comb;
    logic [31:0] read_effective_scale32_comb;
    logic read_descriptor_error_comb;
    logic read_numeric_overflow_comb;
    logic [2:0] read_emit_beat_comb;
    integer delta_scan;
    integer lane_index;
    genvar value_index;

    function automatic logic scale32_valid(input logic [31:0] record);
        logic signed [7:0] exponent;
        begin
            exponent = $signed(record[23:16]);
            scale32_valid =
                (record[31:24] == 8'h00) &&
                (record[15:0] >= 16'h8000) &&
                (exponent >= -8'sd24) &&
                (exponent <= 8'sd4);
        end
    endfunction

    function automatic logic [VALUE_WIDTH-1:0] absolute_value(
        input logic signed [VALUE_WIDTH-1:0] value
    );
        begin
            if (value[VALUE_WIDTH-1])
                absolute_value = $unsigned(-value);
            else
                absolute_value = $unsigned(value);
        end
    endfunction

    function automatic logic [VALUE_WIDTH-1:0] beat_maximum_abs(
        input logic signed [LANES_PER_BEAT*VALUE_WIDTH-1:0] values
    );
        logic [VALUE_WIDTH-1:0] result;
        logic [VALUE_WIDTH-1:0] magnitude;
        integer index;
        begin
            result = '0;
            for (index = 0; index < LANES_PER_BEAT; index = index + 1) begin
                magnitude = absolute_value(values[index*VALUE_WIDTH +: VALUE_WIDTH]);
                if (magnitude > result)
                    result = magnitude;
            end
            beat_maximum_abs = result;
        end
    endfunction

    function automatic logic signed [63:0] round_shift_even_signed(
        input logic signed [VALUE_WIDTH-1:0] value,
        input logic [5:0] shift
    );
        logic [63:0] magnitude;
        logic [63:0] quotient;
        logic [63:0] remainder;
        logic [63:0] half;
        logic [63:0] rounded;
        begin
            magnitude = {{(64-VALUE_WIDTH){1'b0}}, absolute_value(value)};
            if (shift == 0) begin
                rounded = magnitude;
            end else begin
                quotient = magnitude >> shift;
                remainder = magnitude & ((64'h1 << shift) - 64'h1);
                half = 64'h1 << (shift - 1'b1);
                rounded = quotient;
                if ((remainder > half) || ((remainder == half) && quotient[0]))
                    rounded = quotient + 64'd1;
            end
            if (value[VALUE_WIDTH-1])
                round_shift_even_signed = -$signed(rounded);
            else
                round_shift_even_signed = $signed(rounded);
        end
    endfunction

    function automatic logic signed [63:0] quantize_value(
        input logic signed [VALUE_WIDTH-1:0] value,
        input logic signed [6:0] delta
    );
        logic signed [63:0] extended;
        begin
            extended = {{(64-VALUE_WIDTH){value[VALUE_WIDTH-1]}}, value};
            if (delta < 0)
                quantize_value = extended <<< -delta;
            else
                quantize_value = round_shift_even_signed(value, delta[5:0]);
        end
    endfunction

    assign write_bank_state_comb = write_bank_q ? bank_state_q[1] : bank_state_q[0];
    assign read_bank_state_comb = read_bank_q ? bank_state_q[1] : bank_state_q[0];
    assign read_group_lanes_comb = read_bank_q ? group_lanes_q[1] : group_lanes_q[0];
    assign read_base_scale32_comb = read_bank_q ? base_scale32_q[1] : base_scale32_q[0];
    assign read_group_index_comb = read_bank_q ? group_index_q[1] : group_index_q[0];
    assign read_tensor_tag_comb = read_bank_q ? tensor_tag_q[1] : tensor_tag_q[0];
    assign read_maximum_abs_comb = read_bank_q ? maximum_abs_q[1] : maximum_abs_q[0];
    assign read_selected_delta_comb = read_bank_q ? selected_delta_q[1] : selected_delta_q[0];
    assign read_effective_scale32_comb =
        read_bank_q ? effective_scale32_q[1] : effective_scale32_q[0];
    assign read_descriptor_error_comb =
        read_bank_q ? descriptor_error_q[1] : descriptor_error_q[0];
    assign read_numeric_overflow_comb =
        read_bank_q ? numeric_overflow_q[1] : numeric_overflow_q[0];
    assign read_emit_beat_comb = read_bank_q ? emit_beat_q[1] : emit_beat_q[0];

    generate
        for (value_index = 0; value_index < MAX_GROUP_LANES; value_index = value_index + 1) begin : gen_value_mem_views
            assign value_mem_bank0_comb[value_index*VALUE_WIDTH +: VALUE_WIDTH] =
                value_mem_q[0][value_index];
            assign value_mem_bank1_comb[value_index*VALUE_WIDTH +: VALUE_WIDTH] =
                value_mem_q[1][value_index];
        end
    endgenerate
    assign read_value_mem_comb = read_bank_q ? value_mem_bank1_comb : value_mem_bank0_comb;

    always @* begin
        scale_valid_comb = scale32_valid(read_base_scale32_comb);
        base_exponent_comb = $signed(read_base_scale32_comb[23:16]);
        select_found_comb = 1'b0;
        select_delta_comb = 7'sd0;
        selected_magnitude_comb = 64'd0;
        quotient_comb = 64'd0;
        remainder_comb = 64'd0;
        half_comb = 64'd0;
        effective_exponent_comb = 9'sd0;
        delta_scan_s7 = 7'sd0;

        if (scale_valid_comb && (read_maximum_abs_comb == '0)) begin
            select_found_comb = 1'b1;
            select_delta_comb = 7'sd0;
        end else if (scale_valid_comb) begin
            for (delta_scan = -24; delta_scan <= 24; delta_scan = delta_scan + 1) begin
                delta_scan_s7 = 7'(delta_scan);
                effective_exponent_comb =
                    {{1{base_exponent_comb[7]}}, base_exponent_comb} +
                    {{2{delta_scan_s7[6]}}, delta_scan_s7};
                if (!select_found_comb &&
                    (effective_exponent_comb >= -9'sd24) &&
                    (effective_exponent_comb <= 9'sd4)) begin
                    if (delta_scan < 0) begin
                        selected_magnitude_comb =
                            {{(64-VALUE_WIDTH){1'b0}}, read_maximum_abs_comb}
                            << -delta_scan;
                    end else if (delta_scan == 0) begin
                        selected_magnitude_comb =
                            {{(64-VALUE_WIDTH){1'b0}}, read_maximum_abs_comb};
                    end else begin
                        quotient_comb =
                            {{(64-VALUE_WIDTH){1'b0}}, read_maximum_abs_comb}
                            >> delta_scan;
                        remainder_comb =
                            {{(64-VALUE_WIDTH){1'b0}}, read_maximum_abs_comb}
                            & ((64'h1 << delta_scan) - 64'h1);
                        half_comb = 64'h1 << (delta_scan - 1);
                        selected_magnitude_comb = quotient_comb;
                        if ((remainder_comb > half_comb) ||
                            ((remainder_comb == half_comb) && quotient_comb[0]))
                            selected_magnitude_comb = quotient_comb + 64'd1;
                    end
                    if (selected_magnitude_comb <= 64'd127) begin
                        select_found_comb = 1'b1;
                        select_delta_comb = delta_scan_s7;
                    end
                end
            end
        end
    end

    always @* begin
        group_start_ready_o = rst_ni && !clear_i && !capture_active_q &&
                              (write_bank_state_comb == BANK_EMPTY);
        value_ready_o = rst_ni && !clear_i && capture_active_q;

        payload_valid_o = 1'b0;
        payload_s8_o = '0;
        payload_last_o = 1'b0;
        group_commit_valid_o = 1'b0;
        group_index_u6_o = read_group_index_comb;
        tensor_tag_u16_o = read_tensor_tag_comb;
        exponent_delta_s7_o = read_selected_delta_comb;
        effective_scale32_o = read_effective_scale32_comb;
        descriptor_error_o = read_descriptor_error_comb;
        numeric_overflow_o = read_numeric_overflow_comb &&
                             !read_descriptor_error_comb;
        for (lane_index = 0; lane_index < LANES_PER_BEAT; lane_index = lane_index + 1)
            quantized_lane_comb[lane_index*64 +: 64] = quantize_value(
                $signed(read_value_mem_comb[
                    (read_emit_beat_comb*LANES_PER_BEAT + lane_index)*VALUE_WIDTH
                    +: VALUE_WIDTH
                ]),
                read_selected_delta_comb
            );

        if (read_bank_state_comb == BANK_EMIT) begin
            payload_valid_o = 1'b1;
            payload_last_o =
                (read_emit_beat_comb ==
                 ((read_group_lanes_comb == 8'd64) ? 3'd3 : 3'd7));
            for (lane_index = 0; lane_index < LANES_PER_BEAT; lane_index = lane_index + 1) begin
                payload_s8_o[lane_index*8 +: 8] = quantized_lane_comb[lane_index*64 +: 8];
            end
        end

        if (read_bank_state_comb == BANK_COMMIT)
            group_commit_valid_o = 1'b1;
    end

    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            bank_state_q[0] <= BANK_EMPTY;
            bank_state_q[1] <= BANK_EMPTY;
            group_lanes_q[0] <= '0;
            group_lanes_q[1] <= '0;
            base_scale32_q[0] <= '0;
            base_scale32_q[1] <= '0;
            group_index_q[0] <= '0;
            group_index_q[1] <= '0;
            tensor_tag_q[0] <= '0;
            tensor_tag_q[1] <= '0;
            maximum_abs_q[0] <= '0;
            maximum_abs_q[1] <= '0;
            selected_delta_q[0] <= '0;
            selected_delta_q[1] <= '0;
            effective_scale32_q[0] <= '0;
            effective_scale32_q[1] <= '0;
            descriptor_error_q[0] <= 1'b0;
            descriptor_error_q[1] <= 1'b0;
            numeric_overflow_q[0] <= 1'b0;
            numeric_overflow_q[1] <= 1'b0;
            emit_beat_q[0] <= '0;
            emit_beat_q[1] <= '0;
            write_bank_q <= 1'b0;
            read_bank_q <= 1'b0;
            capture_active_q <= 1'b0;
            capture_bank_q <= 1'b0;
            capture_beat_q <= '0;
        end else if (clear_i) begin
            bank_state_q[0] <= BANK_EMPTY;
            bank_state_q[1] <= BANK_EMPTY;
            descriptor_error_q[0] <= 1'b0;
            descriptor_error_q[1] <= 1'b0;
            numeric_overflow_q[0] <= 1'b0;
            numeric_overflow_q[1] <= 1'b0;
            write_bank_q <= 1'b0;
            read_bank_q <= 1'b0;
            capture_active_q <= 1'b0;
            capture_bank_q <= 1'b0;
            capture_beat_q <= '0;
        end else begin
            if (group_start_valid_i && group_start_ready_o) begin
                group_lanes_q[write_bank_q] <= group_lanes_u8_i;
                base_scale32_q[write_bank_q] <= base_scale32_i;
                group_index_q[write_bank_q] <= group_index_u6_i;
                tensor_tag_q[write_bank_q] <= tensor_tag_u16_i;
                maximum_abs_q[write_bank_q] <= '0;
                selected_delta_q[write_bank_q] <= '0;
                effective_scale32_q[write_bank_q] <= '0;
                descriptor_error_q[write_bank_q] <= 1'b0;
                numeric_overflow_q[write_bank_q] <= 1'b0;
                emit_beat_q[write_bank_q] <= '0;
                if (((group_lanes_u8_i != 8'd64) &&
                     (group_lanes_u8_i != 8'd128)) ||
                    (group_index_u6_i >= 6'd38) ||
                    !scale32_valid(base_scale32_i)) begin
                    bank_state_q[write_bank_q] <= BANK_COMMIT;
                    descriptor_error_q[write_bank_q] <= 1'b1;
                    write_bank_q <= ~write_bank_q;
                end else begin
                    bank_state_q[write_bank_q] <= BANK_CAPTURE;
                    capture_active_q <= 1'b1;
                    capture_bank_q <= write_bank_q;
                    capture_beat_q <= '0;
                end
            end

            if (value_valid_i && value_ready_o) begin
                for (lane_index = 0; lane_index < LANES_PER_BEAT; lane_index = lane_index + 1)
                    value_mem_q[capture_bank_q]
                        [capture_beat_q*LANES_PER_BEAT + lane_index] <=
                        value_s40_i[lane_index*VALUE_WIDTH +: VALUE_WIDTH];

                if (beat_maximum_abs(value_s40_i) > maximum_abs_q[capture_bank_q])
                    maximum_abs_q[capture_bank_q] <= beat_maximum_abs(value_s40_i);

                if (capture_beat_q ==
                    ((group_lanes_q[capture_bank_q] == 8'd64) ? 3'd3 : 3'd7)) begin
                    bank_state_q[capture_bank_q] <= BANK_READY;
                    capture_active_q <= 1'b0;
                    capture_beat_q <= '0;
                    write_bank_q <= ~write_bank_q;
                end else begin
                    capture_beat_q <= capture_beat_q + 3'd1;
                end
            end

            if (bank_state_q[read_bank_q] == BANK_READY) begin
                if (!scale_valid_comb) begin
                    descriptor_error_q[read_bank_q] <= 1'b1;
                    bank_state_q[read_bank_q] <= BANK_COMMIT;
                end else if (!select_found_comb) begin
                    numeric_overflow_q[read_bank_q] <= 1'b1;
                    bank_state_q[read_bank_q] <= BANK_COMMIT;
                end else begin
                    selected_delta_q[read_bank_q] <= select_delta_comb;
                    effective_scale32_q[read_bank_q] <= {
                        8'h00,
                        base_exponent_comb + select_delta_comb,
                        base_scale32_q[read_bank_q][15:0]
                    };
                    emit_beat_q[read_bank_q] <= '0;
                    bank_state_q[read_bank_q] <= BANK_EMIT;
                end
            end

            if (payload_valid_o && payload_ready_i) begin
                if (payload_last_o)
                    bank_state_q[read_bank_q] <= BANK_COMMIT;
                else
                    emit_beat_q[read_bank_q] <= emit_beat_q[read_bank_q] + 3'd1;
            end

            if (group_commit_valid_o && group_commit_ready_i) begin
                bank_state_q[read_bank_q] <= BANK_EMPTY;
                descriptor_error_q[read_bank_q] <= 1'b0;
                numeric_overflow_q[read_bank_q] <= 1'b0;
                read_bank_q <= ~read_bank_q;
            end
        end
    end
endmodule


// Build one exact 64-byte BFP1 tensor sidecar.  One base Scale32 record is
// supplied with every dynamic delta so effective-exponent validation happens
// before the sidecar can be published.
module ace2_dynamic_scale32_sidecar_builder_core (
    input  logic                clk_i,
    input  logic                rst_ni,
    input  logic                clear_i,
    input  logic                start_valid_i,
    output logic                start_ready_o,
    input  logic [63:0]         payload_addr_u64_i,
    input  logic [7:0]          group_lanes_u8_i,
    input  logic [5:0]          group_count_u6_i,
    input  logic [15:0]         producer_tag_u16_i,
    input  logic [7:0]          layer_id_u8_i,
    input  logic [7:0]          producer_opcode_u8_i,
    input  logic [31:0]         tensor_elements_u32_i,
    input  logic [63:0]         model_identity_u64_i,
    input  logic                delta_valid_i,
    output logic                delta_ready_o,
    input  logic signed [6:0]   exponent_delta_s7_i,
    input  logic [31:0]         base_scale32_i,
    output logic                sidecar_valid_o,
    input  logic                sidecar_ready_i,
    output logic [511:0]        sidecar_o,
    output logic                descriptor_error_o,
    output logic                numeric_overflow_o
);
    logic busy_q;
    logic sidecar_valid_q;
    logic [511:0] sidecar_q;
    logic [5:0] group_count_q;
    logic [5:0] delta_index_q;
    logic descriptor_error_q;
    logic numeric_overflow_q;
    logic signed [7:0] base_exponent;
    logic signed [8:0] effective_exponent;
    logic [31:0] expected_group_count;

    function automatic logic scale32_valid(input logic [31:0] record);
        logic signed [7:0] exponent;
        begin
            exponent = $signed(record[23:16]);
            scale32_valid =
                (record[31:24] == 8'h00) &&
                (record[15:0] >= 16'h8000) &&
                (exponent >= -8'sd24) &&
                (exponent <= 8'sd4);
        end
    endfunction

    always @* begin
        start_ready_o = rst_ni && !clear_i && !busy_q && !sidecar_valid_q;
        delta_ready_o = rst_ni && !clear_i && busy_q;
        sidecar_valid_o = sidecar_valid_q;
        sidecar_o = sidecar_q;
        descriptor_error_o = descriptor_error_q;
        numeric_overflow_o = numeric_overflow_q && !descriptor_error_q;
        base_exponent = $signed(base_scale32_i[23:16]);
        effective_exponent =
            {{1{base_exponent[7]}}, base_exponent} +
            {{2{exponent_delta_s7_i[6]}}, exponent_delta_s7_i};
        if (group_lanes_u8_i == 8'd64)
            expected_group_count = (tensor_elements_u32_i + 32'd63) >> 6;
        else if (group_lanes_u8_i == 8'd128)
            expected_group_count = (tensor_elements_u32_i + 32'd127) >> 7;
        else
            expected_group_count = 32'hffff_ffff;
    end

    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            busy_q <= 1'b0;
            sidecar_valid_q <= 1'b0;
            sidecar_q <= '0;
            group_count_q <= '0;
            delta_index_q <= '0;
            descriptor_error_q <= 1'b0;
            numeric_overflow_q <= 1'b0;
        end else if (clear_i) begin
            busy_q <= 1'b0;
            sidecar_valid_q <= 1'b0;
            sidecar_q <= '0;
            descriptor_error_q <= 1'b0;
            numeric_overflow_q <= 1'b0;
        end else begin
            if (start_valid_i && start_ready_o) begin
                sidecar_q <= '0;
                sidecar_q[31:0] <= 32'h3150_4642;
                sidecar_q[39:32] <= 8'd1;
                sidecar_q[47:40] <= group_lanes_u8_i;
                sidecar_q[55:48] <= {2'b00, group_count_u6_i};
                sidecar_q[63:56] <= 8'h00;
                sidecar_q[79:64] <= producer_tag_u16_i;
                sidecar_q[87:80] <= layer_id_u8_i;
                sidecar_q[95:88] <= producer_opcode_u8_i;
                sidecar_q[127:96] <= tensor_elements_u32_i;
                sidecar_q[191:128] <= model_identity_u64_i;
                group_count_q <= group_count_u6_i;
                delta_index_q <= '0;
                descriptor_error_q <= 1'b0;
                numeric_overflow_q <= 1'b0;
                if ((payload_addr_u64_i < 64'd64) ||
                    (payload_addr_u64_i[5:0] != 6'd0) ||
                    ((group_lanes_u8_i != 8'd64) &&
                     (group_lanes_u8_i != 8'd128)) ||
                    (group_count_u6_i == 0) ||
                    (group_count_u6_i > 6'd38) ||
                    (expected_group_count != {26'd0, group_count_u6_i})) begin
                    descriptor_error_q <= 1'b1;
                    sidecar_valid_q <= 1'b1;
                    busy_q <= 1'b0;
                end else begin
                    busy_q <= 1'b1;
                end
            end

            if (delta_valid_i && delta_ready_o) begin
                sidecar_q[(24+delta_index_q)*8 +: 8] <=
                    {exponent_delta_s7_i[6], exponent_delta_s7_i};
                if (!scale32_valid(base_scale32_i))
                    descriptor_error_q <= 1'b1;
                if ((exponent_delta_s7_i < -7'sd24) ||
                    (exponent_delta_s7_i > 7'sd24) ||
                    (effective_exponent < -9'sd24) ||
                    (effective_exponent > 9'sd4))
                    numeric_overflow_q <= 1'b1;

                if (delta_index_q + 6'd1 == group_count_q) begin
                    busy_q <= 1'b0;
                    sidecar_valid_q <= 1'b1;
                end else begin
                    delta_index_q <= delta_index_q + 6'd1;
                end
            end

            if (sidecar_valid_q && sidecar_ready_i) begin
                sidecar_valid_q <= 1'b0;
                descriptor_error_q <= 1'b0;
                numeric_overflow_q <= 1'b0;
            end
        end
    end
endmodule


// Validate one candidate sidecar before payload issue.  Packed base Scale32
// records are ordered by group index; unused records are ignored.
module ace2_dynamic_scale32_sidecar_validator_core #(
    parameter integer MAX_GROUPS = 38
) (
    input  logic                         clk_i,
    input  logic                         rst_ni,
    input  logic                         clear_i,
    input  logic                         start_valid_i,
    output logic                         start_ready_o,
    input  logic [511:0]                 sidecar_i,
    input  logic [MAX_GROUPS*32-1:0]     base_scale32_packed_i,
    input  logic [63:0]                  payload_addr_u64_i,
    input  logic [7:0]                   expected_group_lanes_u8_i,
    input  logic [5:0]                   expected_group_count_u6_i,
    input  logic [15:0]                  expected_producer_tag_u16_i,
    input  logic [7:0]                   expected_layer_id_u8_i,
    input  logic [7:0]                   expected_producer_opcode_u8_i,
    input  logic [31:0]                  expected_tensor_elements_u32_i,
    input  logic [63:0]                  expected_model_identity_u64_i,
    output logic                         result_valid_o,
    input  logic                         result_ready_i,
    output logic                         descriptor_error_o,
    output logic                         numeric_overflow_o
);
    logic result_valid_q;
    logic descriptor_error_q;
    logic numeric_overflow_q;
    logic descriptor_error_comb;
    logic numeric_overflow_comb;
    logic [31:0] expected_group_count_comb;
    logic [31:0] base_scale;
    logic signed [7:0] base_exponent;
    logic signed [7:0] delta;
    logic signed [8:0] effective_exponent;
    integer group_index;

    function automatic logic scale32_valid(input logic [31:0] record);
        logic signed [7:0] exponent;
        begin
            exponent = $signed(record[23:16]);
            scale32_valid =
                (record[31:24] == 8'h00) &&
                (record[15:0] >= 16'h8000) &&
                (exponent >= -8'sd24) &&
                (exponent <= 8'sd4);
        end
    endfunction

    always @* begin
        start_ready_o = rst_ni && !clear_i && !result_valid_q;
        result_valid_o = result_valid_q;
        descriptor_error_o = descriptor_error_q;
        numeric_overflow_o = numeric_overflow_q && !descriptor_error_q;
        descriptor_error_comb = 1'b0;
        numeric_overflow_comb = 1'b0;
        expected_group_count_comb = 32'hffff_ffff;
        base_scale = '0;
        base_exponent = '0;
        delta = '0;
        effective_exponent = '0;

        if (expected_group_lanes_u8_i == 8'd64)
            expected_group_count_comb = (expected_tensor_elements_u32_i + 32'd63) >> 6;
        else if (expected_group_lanes_u8_i == 8'd128)
            expected_group_count_comb = (expected_tensor_elements_u32_i + 32'd127) >> 7;

        if ((payload_addr_u64_i < 64'd64) ||
            (payload_addr_u64_i[5:0] != 6'd0) ||
            (sidecar_i[31:0] != 32'h3150_4642) ||
            (sidecar_i[39:32] != 8'd1) ||
            (sidecar_i[47:40] != expected_group_lanes_u8_i) ||
            (sidecar_i[55:48] != {2'b00, expected_group_count_u6_i}) ||
            (sidecar_i[63:56] != 8'h00) ||
            (sidecar_i[79:64] != expected_producer_tag_u16_i) ||
            (sidecar_i[87:80] != expected_layer_id_u8_i) ||
            (sidecar_i[95:88] != expected_producer_opcode_u8_i) ||
            (sidecar_i[127:96] != expected_tensor_elements_u32_i) ||
            (sidecar_i[191:128] != expected_model_identity_u64_i) ||
            (sidecar_i[511:496] != 16'h0000) ||
            (expected_group_count_u6_i == 0) ||
            (expected_group_count_u6_i > 6'd38) ||
            (expected_group_count_comb != {26'd0, expected_group_count_u6_i}))
            descriptor_error_comb = 1'b1;

        for (group_index = 0; group_index < MAX_GROUPS; group_index = group_index + 1) begin
            if (group_index < expected_group_count_u6_i) begin
                base_scale = base_scale32_packed_i[group_index*32 +: 32];
                base_exponent = $signed(base_scale[23:16]);
                delta = $signed(sidecar_i[(24+group_index)*8 +: 8]);
                effective_exponent = base_exponent + delta;
                if (!scale32_valid(base_scale))
                    descriptor_error_comb = 1'b1;
                if ((delta < -8'sd24) || (delta > 8'sd24) ||
                    (effective_exponent < -9'sd24) ||
                    (effective_exponent > 9'sd4))
                    numeric_overflow_comb = 1'b1;
            end else if (sidecar_i[(24+group_index)*8 +: 8] != 8'h00) begin
                descriptor_error_comb = 1'b1;
            end
        end
    end

    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            result_valid_q <= 1'b0;
            descriptor_error_q <= 1'b0;
            numeric_overflow_q <= 1'b0;
        end else if (clear_i) begin
            result_valid_q <= 1'b0;
            descriptor_error_q <= 1'b0;
            numeric_overflow_q <= 1'b0;
        end else begin
            if (start_valid_i && start_ready_o) begin
                result_valid_q <= 1'b1;
                descriptor_error_q <= descriptor_error_comb;
                numeric_overflow_q <= numeric_overflow_comb;
            end else if (result_valid_q && result_ready_i) begin
                result_valid_q <= 1'b0;
                descriptor_error_q <= 1'b0;
                numeric_overflow_q <= 1'b0;
            end
        end
    end
endmodule


// One exact shared tagged align/accumulate service.  Scale32 products are
// represented at canonical binary exponent -78:
//   partial * sig_a * sig_b * 2^(exp_a + exp_b - 30)
// becomes
//   partial * sig_a * sig_b << (exp_a + exp_b + 48).
// No rounding occurs before the operator-defined output boundary.
module ace2_scale32_tagged_accumulator_core #(
    parameter integer ACC_WIDTH = 160,
    parameter integer MAX_EVENTS = 38
) (
    input  logic                         clk_i,
    input  logic                         rst_ni,
    input  logic                         clear_i,
    input  logic                         start_valid_i,
    output logic                         start_ready_o,
    input  logic [15:0]                  accumulator_tag_u16_i,
    input  logic                         event_valid_i,
    output logic                         event_ready_o,
    input  logic signed [31:0]           partial_s32_i,
    input  logic [31:0]                  scale_a32_i,
    input  logic [31:0]                  scale_b32_i,
    input  logic                         event_last_i,
    output logic                         result_valid_o,
    input  logic                         result_ready_i,
    output logic signed [ACC_WIDTH-1:0]  accumulator_s160_o,
    output logic signed [7:0]            canonical_exponent_s8_o,
    output logic [15:0]                  accumulator_tag_u16_o,
    output logic                         descriptor_error_o,
    output logic                         numeric_overflow_o
);
    localparam logic [6:0] MAX_EVENTS_U7 = 7'(MAX_EVENTS);
    logic busy_q;
    logic result_valid_q;
    logic signed [ACC_WIDTH-1:0] accumulator_q;
    logic signed [ACC_WIDTH-1:0] result_accumulator_q;
    logic [15:0] accumulator_tag_q;
    logic [6:0] event_count_q;
    logic descriptor_error_q;
    logic numeric_overflow_q;

    logic scale_a_valid;
    logic scale_b_valid;
    logic signed [7:0] exponent_a;
    logic signed [7:0] exponent_b;
    logic signed [8:0] alignment_shift;
    logic signed [65:0] scale_product_comb;
    logic signed [ACC_WIDTH-1:0] aligned_term_comb;
    logic signed [ACC_WIDTH-1:0] sum_comb;
    logic add_overflow_comb;

    function automatic logic scale32_valid(input logic [31:0] record);
        logic signed [7:0] exponent;
        begin
            exponent = $signed(record[23:16]);
            scale32_valid =
                (record[31:24] == 8'h00) &&
                (record[15:0] >= 16'h8000) &&
                (exponent >= -8'sd24) &&
                (exponent <= 8'sd4);
        end
    endfunction

    always @* begin
        start_ready_o = rst_ni && !clear_i && !busy_q && !result_valid_q;
        event_ready_o = rst_ni && !clear_i && busy_q;
        result_valid_o = result_valid_q;
        accumulator_s160_o = result_accumulator_q;
        canonical_exponent_s8_o = -8'sd78;
        accumulator_tag_u16_o = accumulator_tag_q;
        descriptor_error_o = descriptor_error_q;
        numeric_overflow_o = numeric_overflow_q && !descriptor_error_q;

        scale_a_valid = scale32_valid(scale_a32_i);
        scale_b_valid = scale32_valid(scale_b32_i);
        exponent_a = $signed(scale_a32_i[23:16]);
        exponent_b = $signed(scale_b32_i[23:16]);
        alignment_shift = exponent_a + exponent_b + 9'sd48;
        scale_product_comb =
            $signed(partial_s32_i) *
            $signed({1'b0, scale_a32_i[15:0]}) *
            $signed({1'b0, scale_b32_i[15:0]});
        aligned_term_comb = {{(ACC_WIDTH-66){scale_product_comb[65]}}, scale_product_comb};
        if ((alignment_shift >= 0) && (alignment_shift <= 9'sd56))
            aligned_term_comb = aligned_term_comb <<< alignment_shift;
        else
            aligned_term_comb = '0;
        sum_comb = accumulator_q + aligned_term_comb;
        add_overflow_comb =
            (accumulator_q[ACC_WIDTH-1] == aligned_term_comb[ACC_WIDTH-1]) &&
            (sum_comb[ACC_WIDTH-1] != accumulator_q[ACC_WIDTH-1]);
    end

    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            busy_q <= 1'b0;
            result_valid_q <= 1'b0;
            accumulator_q <= '0;
            result_accumulator_q <= '0;
            accumulator_tag_q <= '0;
            event_count_q <= '0;
            descriptor_error_q <= 1'b0;
            numeric_overflow_q <= 1'b0;
        end else if (clear_i) begin
            busy_q <= 1'b0;
            result_valid_q <= 1'b0;
            accumulator_q <= '0;
            descriptor_error_q <= 1'b0;
            numeric_overflow_q <= 1'b0;
        end else begin
            if (start_valid_i && start_ready_o) begin
                busy_q <= 1'b1;
                accumulator_q <= '0;
                result_accumulator_q <= '0;
                accumulator_tag_q <= accumulator_tag_u16_i;
                event_count_q <= '0;
                descriptor_error_q <= 1'b0;
                numeric_overflow_q <= 1'b0;
            end

            if (event_valid_i && event_ready_o) begin
                if (!scale_a_valid || !scale_b_valid ||
                    (alignment_shift < 0) || (alignment_shift > 9'sd56) ||
                    (event_count_q >= MAX_EVENTS_U7)) begin
                    descriptor_error_q <= 1'b1;
                end else begin
                    accumulator_q <= sum_comb;
                    if (add_overflow_comb)
                        numeric_overflow_q <= 1'b1;
                end
                if (event_count_q < MAX_EVENTS_U7)
                    event_count_q <= event_count_q + 7'd1;
                else
                    event_count_q <= event_count_q;

                if (event_last_i) begin
                    busy_q <= 1'b0;
                    result_valid_q <= 1'b1;
                    if (scale_a_valid && scale_b_valid &&
                        (alignment_shift >= 0) && (alignment_shift <= 9'sd56) &&
                        (event_count_q < MAX_EVENTS_U7))
                        result_accumulator_q <= sum_comb;
                    else
                        result_accumulator_q <= accumulator_q;
                end
            end

            if (result_valid_q && result_ready_i) begin
                result_valid_q <= 1'b0;
                descriptor_error_q <= 1'b0;
                numeric_overflow_q <= 1'b0;
            end
        end
    end
endmodule

/* verilator lint_on DECLFILENAME */
`default_nettype wire
