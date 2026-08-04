`default_nettype none

import ace2_pkg::*;

module ace2_shell_oproj_harness (
    input  wire        clk_i,
    input  wire        rst_ni,
    output reg         done_o,
    output reg [31:0]  failures_o,
    output reg [31:0]  vectors_o,
    output reg [31:0]  writes_o,
    output reg [63:0]  signature_o,
    output reg [63:0]  total_cycles_o,
    output reg [31:0]  max_cycles_o
);
    localparam [63:0] OPROJ_ACT_BASE = 64'h0000_0000_0200_0000;
    localparam [63:0] OPROJ_WEIGHT_BASE = 64'h0000_0000_0210_0000;
    localparam [63:0] OPROJ_META_BASE = 64'h0000_0000_0220_0000;
    localparam [63:0] OPROJ_OUT_BASE = 64'h0000_0000_0230_0000;
    localparam integer OPROJ_CASE_BALANCED = 0;
    localparam integer OPROJ_CASE_SATURATION = 1;
    localparam integer OPROJ_CASE_COUNT = 2;
    localparam integer OPROJ_REDUCTION = 896;
    localparam integer OPROJ_OUTPUTS = 896;
    localparam integer OPROJ_BEATS = OPROJ_OUTPUTS / 16;
    localparam integer OPROJ_WEIGHT_BEATS_PER_OUTPUT = OPROJ_REDUCTION / 16;
    localparam integer PROJ_SIGNATURE_BEATS = 304;

    localparam [2:0] ST_CSR_READ = 3'd0;
    localparam [2:0] ST_CSR_READ_WAIT = 3'd1;
    localparam [2:0] ST_CSR_ENABLE = 3'd2;
    localparam [2:0] ST_COMMAND = 3'd3;
    localparam [2:0] ST_WAIT_DONE = 3'd4;
    localparam [2:0] ST_NEXT_CASE = 3'd5;
    localparam [2:0] ST_FINISHED = 3'd6;
    localparam [2:0] ST_COMMAND_GAP = 3'd7;

    wire csr_ready;
    wire csr_rvalid;
    wire [63:0] csr_rdata;
    wire csr_error;
    wire irq;
    wire cmd_ready;
    wire mem_req_valid;
    reg mem_req_ready;
    wire mem_req_write;
    wire [63:0] mem_req_addr;
    wire [15:0] mem_req_len;
    wire [7:0] mem_req_tag;
    wire mem_wvalid;
    reg mem_wready;
    wire [127:0] mem_wdata;
    wire [15:0] mem_wstrb;
    wire [7:0] mem_wtag;
    reg mem_rvalid;
    wire mem_rready;
    reg [127:0] mem_rdata;
    reg [7:0] mem_rtag;
    reg mem_rerror;
    reg mem_bvalid;
    wire mem_bready;
    reg [7:0] mem_btag;
    reg mem_berror;
    wire [7:0] sram_req_valid;
    wire [7:0] sram_write;
    wire [8*12-1:0] sram_addr;
    wire [8*128-1:0] sram_wdata;
    wire [8*16-1:0] sram_wstrb;
    wire busy;
    wire cmd_done_valid;
    wire [15:0] cmd_done_tag;
    wire cmd_done_error;
    wire [47:0] cmd_done_sumsq;
    wire [31:0] cmd_done_inv;
    wire cmd_done_saturation;

    reg [2:0] state;
    integer current_case;
    integer cycle_count;
    integer command_phase_cycle;
    integer command_start_cycle;
    integer observed_count;
    integer pending_read;
    integer read_delay;
    integer pending_write_response;
    integer write_response_delay;
    integer beat;
    integer expected_count;
    integer expected_saturation;
    reg [63:0] pending_addr;
    reg [63:0] pending_write_addr;
    reg [7:0] pending_tag;
    reg [7:0] pending_write_tag;
    reg [127:0] observed_output [0:127];
    reg [63:0] vector_signature;

    function [15:0] oproj_case_rows;
        input integer selected_case;
        begin
            oproj_case_rows =
                (selected_case == OPROJ_CASE_BALANCED) ? 16'd2 : 16'd1;
        end
    endfunction

    function oproj_case_saturation;
        input integer selected_case;
        begin
            oproj_case_saturation =
                (selected_case == OPROJ_CASE_SATURATION) ? 1'b1 : 1'b0;
        end
    endfunction

    function [7:0] oproj_activation;
        input integer selected_case;
        input integer selected_row;
        input integer selected_index;
        integer value;
        begin
            if (selected_case == OPROJ_CASE_SATURATION) begin
                value = selected_index[0] ? 127 : -128;
            end else begin
                value = ((selected_row * 19 + selected_index * 7 + 3) % 128) - 64;
            end
            oproj_activation = value[7:0];
        end
    endfunction

    function [3:0] oproj_weight;
        input integer selected_case;
        input integer selected_output;
        input integer selected_index;
        integer value;
        begin
            if (selected_case == OPROJ_CASE_SATURATION) begin
                value = ((selected_output + selected_index * 7) % 16) - 8;
            end else begin
                value = ((selected_output * 5 + selected_index * 3 + 1) % 5) - 2;
            end
            oproj_weight = value[3:0];
        end
    endfunction

    function [127:0] oproj_activation_beat;
        input integer selected_case;
        input integer selected_row;
        input integer selected_beat;
        integer lane;
        begin
            oproj_activation_beat = 128'd0;
            for (lane = 0; lane < 16; lane = lane + 1) begin
                oproj_activation_beat[lane*8 +: 8] =
                    oproj_activation(selected_case, selected_row,
                                      selected_beat * 16 + lane);
            end
        end
    endfunction

    function [127:0] oproj_weight_beat;
        input integer selected_case;
        input integer selected_output;
        input integer selected_beat;
        integer lane;
        begin
            oproj_weight_beat = 128'd0;
            for (lane = 0; lane < 16; lane = lane + 1) begin
                oproj_weight_beat[lane*4 +: 4] =
                    oproj_weight(selected_case, selected_output,
                                 selected_beat * 16 + lane);
            end
        end
    endfunction

    function [127:0] oproj_meta_beat;
        input integer selected_case;
        input integer selected_output;
        integer multiplier;
        integer right_shift;
        integer zero_point;
        begin
            if (selected_case == OPROJ_CASE_SATURATION) begin
                multiplier = 16 + (selected_output & 3);
                right_shift = 3;
                zero_point = 0;
            end else begin
                multiplier = (selected_output % 3) + 1;
                right_shift = 9 + (selected_output & 1);
                zero_point = (selected_output % 5) - 2;
            end
            oproj_meta_beat = 128'd0;
            oproj_meta_beat[31:0] = multiplier[31:0];
            oproj_meta_beat[37:32] = right_shift[5:0];
            oproj_meta_beat[47:40] = zero_point[7:0];
        end
    endfunction

    wire csr_valid = (state == ST_CSR_READ) || (state == ST_CSR_ENABLE);
    wire csr_write = (state == ST_CSR_ENABLE);
    wire [31:0] csr_addr = (state == ST_CSR_ENABLE) ?
                           ACE2_CSR_CONTROL : ACE2_CSR_ID;
    wire [63:0] csr_wdata = 64'h1;
    wire csr_rready = (state == ST_CSR_READ_WAIT);
    wire cmd_valid = (state == ST_COMMAND);
    wire cmd_done_ready = (state == ST_WAIT_DONE);

    function [127:0] lookup_read_data;
        input [63:0] addr;
        integer index;
        integer row;
        integer output_index;
        integer beat_index;
        begin
            lookup_read_data = 128'd0;
            if ((addr >= OPROJ_ACT_BASE) &&
                (addr < (OPROJ_ACT_BASE +
                         oproj_case_rows(current_case)*OPROJ_BEATS*16))) begin
                index = (addr - OPROJ_ACT_BASE) >> 4;
                row = index / OPROJ_BEATS;
                beat_index = index % OPROJ_BEATS;
                lookup_read_data = oproj_activation_beat(
                    current_case, row, beat_index
                );
            end else if ((addr >= OPROJ_WEIGHT_BASE) &&
                         (addr < (OPROJ_WEIGHT_BASE +
                                  OPROJ_OUTPUTS*
                                  OPROJ_WEIGHT_BEATS_PER_OUTPUT*16))) begin
                index = (addr - OPROJ_WEIGHT_BASE) >> 4;
                output_index = index / OPROJ_WEIGHT_BEATS_PER_OUTPUT;
                beat_index = index % OPROJ_WEIGHT_BEATS_PER_OUTPUT;
                lookup_read_data = oproj_weight_beat(
                    current_case, output_index, beat_index
                );
            end else if ((addr >= OPROJ_META_BASE) &&
                         (addr < (OPROJ_META_BASE + OPROJ_OUTPUTS*16))) begin
                index = (addr - OPROJ_META_BASE) >> 4;
                lookup_read_data = oproj_meta_beat(current_case, index);
            end else begin
                lookup_read_data =
                    128'hbad0_bad0_bad0_bad0_bad0_bad0_bad0_bad0;
            end
        end
    endfunction

    function [63:0] fold_oproj_word;
        input [63:0] signature;
        input [127:0] word;
        input integer selected_case;
        input integer selected_beat;
        reg [63:0] ordinal;
        begin
            ordinal = selected_case*PROJ_SIGNATURE_BEATS + selected_beat;
            fold_oproj_word = {signature[62:0], signature[63]} ^
                              word[63:0] ^ word[127:64] ^ ordinal;
        end
    endfunction

    ace2_shell dut (
        .clk_i(clk_i),
        .rst_ni(rst_ni),
        .csr_valid_i(csr_valid),
        .csr_ready_o(csr_ready),
        .csr_write_i(csr_write),
        .csr_addr_i(csr_addr),
        .csr_wdata_i(csr_wdata),
        .csr_wstrb_i(8'hff),
        .csr_rvalid_o(csr_rvalid),
        .csr_rready_i(csr_rready),
        .csr_rdata_o(csr_rdata),
        .csr_error_o(csr_error),
        .irq_o(irq),
        .cmd_valid_i(cmd_valid),
        .cmd_ready_o(cmd_ready),
        .cmd_opcode_i(ACE2_OPCODE_W4A8_PROJ),
        .cmd_flags_i(8'h09),
        .cmd_layer_id_i(8'd0),
        .cmd_m_i(oproj_case_rows(current_case)),
        .cmd_n_i(16'd896),
        .cmd_k_i(16'd896),
        .cmd_sequence_position_i(16'd0),
        .cmd_completion_tag_i(16'h3f00 + current_case),
        .cmd_src0_addr_i(OPROJ_ACT_BASE),
        .cmd_src1_addr_i(OPROJ_WEIGHT_BASE),
        .cmd_dst_addr_i(OPROJ_OUT_BASE),
        .cmd_scale_addr_i(OPROJ_META_BASE),
        .cmd_scratch_addr_i(64'd0),
        .mem_req_valid_o(mem_req_valid),
        .mem_req_ready_i(mem_req_ready),
        .mem_req_write_o(mem_req_write),
        .mem_req_addr_o(mem_req_addr),
        .mem_req_len_o(mem_req_len),
        .mem_req_tag_o(mem_req_tag),
        .mem_wvalid_o(mem_wvalid),
        .mem_wready_i(mem_wready),
        .mem_wdata_o(mem_wdata),
        .mem_wstrb_o(mem_wstrb),
        .mem_wtag_o(mem_wtag),
        .mem_rvalid_i(mem_rvalid),
        .mem_rready_o(mem_rready),
        .mem_rdata_i(mem_rdata),
        .mem_rtag_i(mem_rtag),
        .mem_rerror_i(mem_rerror),
        .mem_bvalid_i(mem_bvalid),
        .mem_bready_o(mem_bready),
        .mem_btag_i(mem_btag),
        .mem_berror_i(mem_berror),
        .sram_req_valid_o(sram_req_valid),
        .sram_req_ready_i(8'hff),
        .sram_write_o(sram_write),
        .sram_addr_o(sram_addr),
        .sram_wdata_o(sram_wdata),
        .sram_wstrb_o(sram_wstrb),
        .sram_rdata_i({8*128{1'b0}}),
        .sram_rvalid_i(8'd0),
        .busy_o(busy),
        .cmd_done_valid_o(cmd_done_valid),
        .cmd_done_ready_i(cmd_done_ready),
        .cmd_done_tag_o(cmd_done_tag),
        .cmd_done_error_o(cmd_done_error),
        .cmd_done_sumsq_o(cmd_done_sumsq),
        .cmd_done_inv_rms_q30_o(cmd_done_inv),
        .cmd_done_saturation_seen_o(cmd_done_saturation)
    );

    always @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            cycle_count <= 0;
            command_phase_cycle <= 0;
            mem_req_ready <= 1'b0;
            mem_wready <= 1'b0;
            mem_rvalid <= 1'b0;
            mem_rdata <= 128'd0;
            mem_rtag <= 8'd0;
            mem_rerror <= 1'b0;
            mem_bvalid <= 1'b0;
            mem_btag <= 8'd0;
            mem_berror <= 1'b0;
            pending_read <= 0;
            pending_addr <= 64'd0;
            pending_tag <= 8'd0;
            read_delay <= 0;
            pending_write_addr <= 64'd0;
            pending_write_response <= 0;
            pending_write_tag <= 8'd0;
            write_response_delay <= 0;
            observed_count <= 0;
        end else begin
            cycle_count <= cycle_count + 1;
            if ((state == ST_CSR_ENABLE) && csr_ready) begin
                command_phase_cycle <= 0;
            end else if ((state == ST_WAIT_DONE) && cmd_done_valid &&
                         (current_case != (OPROJ_CASE_COUNT - 1))) begin
                command_phase_cycle <= 0;
            end else begin
                command_phase_cycle <= command_phase_cycle + 1;
            end
            mem_req_ready <= (command_phase_cycle[1:0] != 2'b01);
            mem_wready <= (command_phase_cycle[2:0] != 3'b011);

            if (state == ST_NEXT_CASE) begin
                observed_count <= 0;
            end else if (mem_wvalid && mem_wready) begin
                if (observed_count < 128) begin
                    observed_output[observed_count] <= mem_wdata;
                end
                observed_count <= observed_count + 1;
                pending_write_response <= 1;
                pending_write_tag <= mem_wtag;
                write_response_delay <= 0;
            end

            if (mem_rvalid && mem_rready) begin
                mem_rvalid <= 1'b0;
            end
            if (!mem_rvalid && pending_read) begin
                if (read_delay != 0) begin
                    read_delay <= read_delay - 1;
                end else begin
                    mem_rvalid <= 1'b1;
                    mem_rdata <= lookup_read_data(pending_addr);
                    mem_rtag <= pending_tag;
                    pending_read <= 0;
                end
            end
            if (mem_req_valid && mem_req_ready && !mem_req_write) begin
                pending_read <= 1;
                pending_addr <= mem_req_addr;
                pending_tag <= mem_req_tag;
                read_delay <= {1'b0, command_phase_cycle[0]};
            end
            if (mem_req_valid && mem_req_ready && mem_req_write) begin
                pending_write_addr <= mem_req_addr;
            end
            if (mem_bvalid && mem_bready) begin
                mem_bvalid <= 1'b0;
            end
            if (!mem_bvalid && pending_write_response) begin
                if (write_response_delay != 0) begin
                    write_response_delay <= write_response_delay - 1;
                end else begin
                    mem_bvalid <= 1'b1;
                    mem_btag <= pending_write_tag;
                    pending_write_response <= 0;
                end
            end
        end
    end

    always @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            state <= ST_CSR_READ;
            current_case <= 0;
            command_start_cycle <= 0;
            done_o <= 1'b0;
            failures_o = 32'd0;
            vectors_o <= 32'd0;
            writes_o <= 32'd0;
            signature_o = 64'd0;
            vector_signature = 64'd0;
            total_cycles_o <= 64'd0;
            max_cycles_o <= 32'd0;
        end else begin
            case (state)
                ST_CSR_READ: begin
                    if (csr_ready) begin
                        state <= ST_CSR_READ_WAIT;
                    end
                end
                ST_CSR_READ_WAIT: begin
                    if (csr_rvalid) begin
                        if (csr_error ||
                            (csr_rdata !== 64'h4143453200000001)) begin
                            failures_o = failures_o + 1;
                        end
                        state <= ST_CSR_ENABLE;
                    end
                end
                ST_CSR_ENABLE: begin
                    if (csr_ready) begin
                        state <= ST_COMMAND_GAP;
                    end
                end
                ST_COMMAND_GAP: begin
                    state <= ST_COMMAND;
                end
                ST_COMMAND: begin
                    if (cmd_ready) begin
                        command_start_cycle <= cycle_count;
                        state <= ST_WAIT_DONE;
                    end
                end
                ST_WAIT_DONE: begin
                    if (cmd_done_valid) begin
                        expected_count =
                            oproj_case_rows(current_case)*OPROJ_BEATS;
                        expected_saturation =
                            oproj_case_saturation(current_case);
                        if (cmd_done_tag !== (16'h3f00 + current_case)) begin
                            failures_o = failures_o + 1;
                        end
                        if (cmd_done_error !== expected_saturation[0]) begin
                            failures_o = failures_o + 1;
                        end
                        if (cmd_done_saturation !== expected_saturation[0]) begin
                            failures_o = failures_o + 1;
                        end
                        if (observed_count != expected_count) begin
                            failures_o = failures_o + 1;
                        end
                        vector_signature = 64'd0;
                        for (beat = 0; beat < expected_count; beat = beat + 1) begin
                            signature_o = fold_oproj_word(
                                signature_o,
                                observed_output[beat],
                                current_case,
                                beat
                            );
                            vector_signature = fold_oproj_word(
                                vector_signature,
                                observed_output[beat],
                                current_case,
                                beat
                            );
                        end
                        $display("ACE2_SHELL_VECTOR_RESULT simulator=verilator opcode=01 vector=%0d writes=%0d signature=%016x command_accept_cycle=%0d completion_accept_cycle=%0d cycles=%0d",
                                 current_case, observed_count, vector_signature,
                                 command_start_cycle, cycle_count,
                                 cycle_count - command_start_cycle);
                        vectors_o <= vectors_o + 1;
                        writes_o <= writes_o + observed_count;
                        total_cycles_o <= total_cycles_o +
                                          (cycle_count - command_start_cycle);
                        if ((cycle_count - command_start_cycle) >
                            max_cycles_o) begin
                            max_cycles_o <= cycle_count - command_start_cycle;
                        end
                        if (current_case == (OPROJ_CASE_COUNT - 1)) begin
                            done_o <= 1'b1;
                            state <= ST_FINISHED;
                        end else begin
                            current_case <= current_case + 1;
                            state <= ST_NEXT_CASE;
                        end
                    end
                end
                ST_NEXT_CASE: begin
                    state <= ST_COMMAND;
                end
                default: begin
                    state <= ST_FINISHED;
                end
            endcase
        end
    end
endmodule

`default_nettype wire
