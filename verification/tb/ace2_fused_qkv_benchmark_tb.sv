`timescale 1ns/1ps
`default_nettype none

module ace2_fused_qkv_benchmark_tb;
    localparam [7:0] OPCODE_W4A8_PROJ = 8'h01;
    localparam [7:0] OPCODE_FUSED_QKV = 8'h0b;
    localparam [63:0] ACT_BASE = 64'h0000_0000_0000_1000;
    localparam [63:0] WGT_BASE = 64'h0000_0000_0000_2000;
    localparam [63:0] META_BASE = 64'h0000_0000_0009_0000;
    localparam [63:0] LEGACY_DST_BASE = 64'h0000_0000_000a_0000;
    localparam [63:0] FUSED_DST_BASE = 64'h0000_0000_000a_1000;
    localparam integer K_SIZE = 896;
    localparam integer Q_OUTPUTS = 896;
    localparam integer KV_OUTPUTS = 128;
    localparam integer TOTAL_OUTPUTS = Q_OUTPUTS + 2*KV_OUTPUTS;
    localparam integer WEIGHT_BYTES_PER_OUTPUT = K_SIZE / 2;
    localparam integer Q_WEIGHT_BYTES = Q_OUTPUTS * WEIGHT_BYTES_PER_OUTPUT;
    localparam integer KV_WEIGHT_BYTES = KV_OUTPUTS * WEIGHT_BYTES_PER_OUTPUT;
    localparam integer Q_META_BYTES = Q_OUTPUTS * 16;
    localparam integer KV_META_BYTES = KV_OUTPUTS * 16;

    reg clk;
    reg rst_n;
    reg csr_valid;
    wire csr_ready;
    reg csr_write;
    reg [31:0] csr_addr;
    reg [63:0] csr_wdata;
    reg [7:0] csr_wstrb;
    wire csr_rvalid;
    reg csr_rready;
    wire [63:0] csr_rdata;
    wire csr_error;
    wire irq;
    reg cmd_valid;
    wire cmd_ready;
    reg [7:0] cmd_opcode;
    reg [7:0] cmd_flags;
    reg [7:0] cmd_layer_id;
    reg [15:0] cmd_m;
    reg [15:0] cmd_n;
    reg [15:0] cmd_k;
    reg [15:0] cmd_sequence_position;
    reg [15:0] cmd_completion_tag;
    reg [63:0] cmd_src0_addr;
    reg [63:0] cmd_src1_addr;
    reg [63:0] cmd_dst_addr;
    reg [63:0] cmd_scale_addr;
    reg [63:0] cmd_scratch_addr;
    wire mem_req_valid;
    wire mem_req_ready;
    wire mem_req_write;
    wire [63:0] mem_req_addr;
    wire [15:0] mem_req_len;
    wire [7:0] mem_req_tag;
    wire mem_wvalid;
    wire mem_wready;
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
    wire busy;
    wire cmd_done_valid;
    reg cmd_done_ready;
    wire [15:0] cmd_done_tag;
    wire cmd_done_error;
    wire [47:0] cmd_done_sumsq;
    wire [31:0] cmd_done_inv_rms;
    wire cmd_done_saturation;
    wire [7:0] sram_req_valid_unused;
    wire [7:0] sram_write_unused;
    wire [8*12-1:0] sram_addr_unused;
    wire [8*128-1:0] sram_wdata_unused;
    wire [8*16-1:0] sram_wstrb_unused;

    reg write_request_pending;
    reg [63:0] write_request_addr;
    reg [7:0] write_request_tag;
    reg [7:0] output_mem [0:TOTAL_OUTPUTS-1];
    reg [7:0] expected_mem [0:TOTAL_OUTPUTS-1];
    reg [1023:0] expected_path;
    reg corrupt_mode;
    reg corrupt_sent;
    integer cycle_count;
    integer read_requests;
    integer activation_read_requests;
    integer weight_read_requests;
    integer metadata_read_requests;
    integer write_requests;
    integer output_writes;
    integer request_stalls;
    integer write_stalls;
    integer host_commands;
    integer command_start_cycle;
    integer sequence_start_cycle;
    integer output_offset;
    integer byte_lane;
    integer failures;
    integer expected_tag;
    integer final_cycles;

    function automatic [7:0] activation_byte(input integer element_index);
        integer signed value;
        begin
            value = ((element_index * 13 + 7) % 255) - 127;
            activation_byte = value[7:0];
        end
    endfunction

    function automatic [3:0] weight_nibble(
        input integer projection_index,
        input integer channel_index,
        input integer element_index
    );
        integer signed value;
        begin
            value = ((projection_index * 11 + channel_index * 5 +
                      element_index * 3 + 1) % 15) - 7;
            weight_nibble = value[3:0];
        end
    endfunction

    function automatic [127:0] memory_beat(input [63:0] address);
        reg [127:0] beat;
        integer lane;
        integer offset;
        integer projection_index;
        integer channel_index;
        integer local_byte;
        integer element_index;
        integer signed bias;
        begin
            beat = 128'd0;
            projection_index = 0;
            channel_index = 0;
            if ((address >= ACT_BASE) && (address < ACT_BASE + K_SIZE)) begin
                offset = address - ACT_BASE;
                for (lane = 0; lane < 16; lane = lane + 1)
                    beat[lane*8 +: 8] = activation_byte(offset + lane);
            end else if ((address >= WGT_BASE) &&
                         (address < WGT_BASE + Q_WEIGHT_BYTES +
                                           2*KV_WEIGHT_BYTES)) begin
                offset = address - WGT_BASE;
                if (offset < Q_WEIGHT_BYTES) begin
                    projection_index = 0;
                    channel_index = offset / WEIGHT_BYTES_PER_OUTPUT;
                    local_byte = offset % WEIGHT_BYTES_PER_OUTPUT;
                end else if (offset < Q_WEIGHT_BYTES + KV_WEIGHT_BYTES) begin
                    projection_index = 1;
                    offset = offset - Q_WEIGHT_BYTES;
                    channel_index = offset / WEIGHT_BYTES_PER_OUTPUT;
                    local_byte = offset % WEIGHT_BYTES_PER_OUTPUT;
                end else begin
                    projection_index = 2;
                    offset = offset - Q_WEIGHT_BYTES - KV_WEIGHT_BYTES;
                    channel_index = offset / WEIGHT_BYTES_PER_OUTPUT;
                    local_byte = offset % WEIGHT_BYTES_PER_OUTPUT;
                end
                for (lane = 0; lane < 16; lane = lane + 1) begin
                    element_index = (local_byte + lane) * 2;
                    beat[lane*8 +: 4] =
                        weight_nibble(projection_index, channel_index,
                                      element_index);
                    beat[lane*8+4 +: 4] =
                        weight_nibble(projection_index, channel_index,
                                      element_index + 1);
                end
            end else if ((address >= META_BASE) &&
                         (address < META_BASE + Q_META_BYTES +
                                            2*KV_META_BYTES)) begin
                offset = address - META_BASE;
                if (offset < Q_META_BYTES) begin
                    projection_index = 0;
                    channel_index = offset / 16;
                end else if (offset < Q_META_BYTES + KV_META_BYTES) begin
                    projection_index = 1;
                    channel_index = (offset - Q_META_BYTES) / 16;
                end else begin
                    projection_index = 2;
                    channel_index =
                        (offset - Q_META_BYTES - KV_META_BYTES) / 16;
                end
                bias = channel_index * 31 + projection_index * 257 - 1984;
                beat[31:0] = 32'd1000003 +
                              projection_index * 32'd4099 +
                              channel_index * 32'd97;
                beat[37:32] = 6'd25;
                beat[47:40] = 8'd0;
                beat[79:48] = bias[31:0];
            end
            memory_beat = beat;
        end
    endfunction

    assign mem_req_ready = rst_n && ((cycle_count % 17) != 3) &&
                           ((cycle_count % 23) != 9);
    assign mem_wready = rst_n && ((cycle_count % 19) != 5);

    ace2_shell dut (
        .clk_i(clk),
        .rst_ni(rst_n),
        .csr_valid_i(csr_valid),
        .csr_ready_o(csr_ready),
        .csr_write_i(csr_write),
        .csr_addr_i(csr_addr),
        .csr_wdata_i(csr_wdata),
        .csr_wstrb_i(csr_wstrb),
        .csr_rvalid_o(csr_rvalid),
        .csr_rready_i(csr_rready),
        .csr_rdata_o(csr_rdata),
        .csr_error_o(csr_error),
        .irq_o(irq),
        .cmd_valid_i(cmd_valid),
        .cmd_ready_o(cmd_ready),
        .cmd_opcode_i(cmd_opcode),
        .cmd_flags_i(cmd_flags),
        .cmd_layer_id_i(cmd_layer_id),
        .cmd_m_i(cmd_m),
        .cmd_n_i(cmd_n),
        .cmd_k_i(cmd_k),
        .cmd_sequence_position_i(cmd_sequence_position),
        .cmd_completion_tag_i(cmd_completion_tag),
        .cmd_src0_addr_i(cmd_src0_addr),
        .cmd_src1_addr_i(cmd_src1_addr),
        .cmd_dst_addr_i(cmd_dst_addr),
        .cmd_scale_addr_i(cmd_scale_addr),
        .cmd_scratch_addr_i(cmd_scratch_addr),
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
        .sram_req_valid_o(sram_req_valid_unused),
        .sram_req_ready_i(8'hff),
        .sram_write_o(sram_write_unused),
        .sram_addr_o(sram_addr_unused),
        .sram_wdata_o(sram_wdata_unused),
        .sram_wstrb_o(sram_wstrb_unused),
        .sram_rdata_i({8*128{1'b0}}),
        .sram_rvalid_i(8'h00),
        .busy_o(busy),
        .cmd_done_valid_o(cmd_done_valid),
        .cmd_done_ready_i(cmd_done_ready),
        .cmd_done_tag_o(cmd_done_tag),
        .cmd_done_error_o(cmd_done_error),
        .cmd_done_sumsq_o(cmd_done_sumsq),
        .cmd_done_inv_rms_q30_o(cmd_done_inv_rms),
        .cmd_done_saturation_seen_o(cmd_done_saturation)
    );

    initial begin
        clk = 1'b0;
        forever #5 clk = ~clk;
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            cycle_count <= 0;
            read_requests <= 0;
            activation_read_requests <= 0;
            weight_read_requests <= 0;
            metadata_read_requests <= 0;
            write_requests <= 0;
            output_writes <= 0;
            request_stalls <= 0;
            write_stalls <= 0;
            mem_rvalid <= 1'b0;
            mem_rdata <= 128'd0;
            mem_rtag <= 8'd0;
            mem_rerror <= 1'b0;
            mem_bvalid <= 1'b0;
            mem_btag <= 8'd0;
            mem_berror <= 1'b0;
            write_request_pending <= 1'b0;
            write_request_addr <= 64'd0;
            write_request_tag <= 8'd0;
            corrupt_sent <= 1'b0;
            for (output_offset = 0; output_offset < TOTAL_OUTPUTS;
                 output_offset = output_offset + 1)
                output_mem[output_offset] <= 8'd0;
        end else begin
            cycle_count <= cycle_count + 1;
            if (mem_req_valid && !mem_req_ready)
                request_stalls <= request_stalls + 1;
            if (mem_wvalid && !mem_wready)
                write_stalls <= write_stalls + 1;
            if (mem_rvalid && mem_rready)
                mem_rvalid <= 1'b0;
            if (mem_bvalid && mem_bready)
                mem_bvalid <= 1'b0;

            if (mem_req_valid && mem_req_ready) begin
                if (mem_req_len != 16'd1)
                    $fatal(1, "unexpected memory request length");
                if (mem_req_write) begin
                    if (write_request_pending)
                        $fatal(1, "overlapping write requests");
                    write_request_pending <= 1'b1;
                    write_request_addr <= mem_req_addr;
                    write_request_tag <= mem_req_tag;
                    write_requests <= write_requests + 1;
                end else begin
                    if (mem_rvalid)
                        $fatal(1, "overlapping read responses");
                    mem_rdata <= memory_beat(mem_req_addr);
                    mem_rtag <= (corrupt_mode && !corrupt_sent &&
                                 (activation_read_requests == 4)) ?
                                (mem_req_tag ^ 8'h01) : mem_req_tag;
                    mem_rerror <= 1'b0;
                    mem_rvalid <= 1'b1;
                    read_requests <= read_requests + 1;
                    if ((mem_req_addr >= ACT_BASE) &&
                        (mem_req_addr < ACT_BASE + K_SIZE))
                        activation_read_requests <=
                            activation_read_requests + 1;
                    else if ((mem_req_addr >= WGT_BASE) &&
                             (mem_req_addr < WGT_BASE + Q_WEIGHT_BYTES +
                                               2*KV_WEIGHT_BYTES))
                        weight_read_requests <= weight_read_requests + 1;
                    else if ((mem_req_addr >= META_BASE) &&
                             (mem_req_addr < META_BASE + Q_META_BYTES +
                                                2*KV_META_BYTES))
                        metadata_read_requests <= metadata_read_requests + 1;
                    else
                        $fatal(1, "read outside QKV ranges address=%016x",
                               mem_req_addr);
                    if (corrupt_mode && !corrupt_sent &&
                        (activation_read_requests == 4))
                        corrupt_sent <= 1'b1;
                end
            end

            if (mem_wvalid && mem_wready) begin
                if (!write_request_pending || (mem_wtag != write_request_tag))
                    $fatal(1, "write data without matching request");
                if (mem_wstrb != 16'hffff)
                    $fatal(1, "partial QKV output write");
                if (((write_request_addr < LEGACY_DST_BASE) ||
                     (write_request_addr >= LEGACY_DST_BASE + TOTAL_OUTPUTS)) &&
                    ((write_request_addr < FUSED_DST_BASE) ||
                     (write_request_addr >= FUSED_DST_BASE + TOTAL_OUTPUTS)))
                    $fatal(1, "write outside QKV destination ranges");
                if ((write_request_addr >= LEGACY_DST_BASE) &&
                    (write_request_addr < LEGACY_DST_BASE + TOTAL_OUTPUTS))
                    output_offset = write_request_addr - LEGACY_DST_BASE;
                else
                    output_offset = write_request_addr - FUSED_DST_BASE;
                for (byte_lane = 0; byte_lane < 16;
                     byte_lane = byte_lane + 1)
                    output_mem[output_offset + byte_lane] <=
                        mem_wdata[byte_lane*8 +: 8];
                output_writes <= output_writes + 16;
                write_request_pending <= 1'b0;
                mem_bvalid <= 1'b1;
                mem_btag <= mem_wtag;
                mem_berror <= 1'b0;
            end
        end
    end

    task automatic enable_shell;
        begin
            @(negedge clk);
            csr_valid = 1'b1;
            csr_write = 1'b1;
            csr_addr = 32'h18;
            csr_wdata = 64'd1;
            csr_wstrb = 8'hff;
            while (!csr_ready)
                @(negedge clk);
            @(posedge clk);
            @(negedge clk);
            csr_valid = 1'b0;
            csr_write = 1'b0;
            csr_wstrb = 8'd0;
            repeat (2) @(posedge clk);
        end
    endtask

    task automatic launch_command(
        input [7:0] opcode,
        input [15:0] outputs,
        input [15:0] tag,
        input [63:0] weight_base,
        input [63:0] output_base,
        input [63:0] metadata_base
    );
        begin
            while (!cmd_ready)
                @(negedge clk);
            cmd_opcode = opcode;
            cmd_n = outputs;
            cmd_completion_tag = tag;
            cmd_src1_addr = weight_base;
            cmd_dst_addr = output_base;
            cmd_scale_addr = metadata_base;
            cmd_valid = 1'b1;
            @(posedge clk);
            command_start_cycle = cycle_count;
            host_commands = host_commands + 1;
            @(negedge clk);
            cmd_valid = 1'b0;
        end
    endtask

    task automatic await_completion(
        input [15:0] tag,
        input integer expect_fault
    );
        begin
            while (!cmd_done_valid &&
                   (cycle_count - command_start_cycle < 10000000))
                @(negedge clk);
            if (!cmd_done_valid)
                $fatal(1, "QKV command timeout tag=%04x", tag);
            if (cmd_done_tag !== tag)
                $fatal(1, "completion tag mismatch got=%04x expected=%04x",
                       cmd_done_tag, tag);
            if (expect_fault) begin
                if (!cmd_done_error)
                    $fatal(1, "corrupted command completed without error");
            end else if (cmd_done_error !== cmd_done_saturation) begin
                $fatal(1, "unexpected command error=%0d saturation=%0d",
                       cmd_done_error, cmd_done_saturation);
            end
            cmd_done_ready = 1'b1;
            @(posedge clk);
            @(negedge clk);
            cmd_done_ready = 1'b0;
        end
    endtask

    task automatic check_outputs;
        begin
            failures = 0;
            for (output_offset = 0; output_offset < TOTAL_OUTPUTS;
                 output_offset = output_offset + 1) begin
                if (output_mem[output_offset] !==
                    expected_mem[output_offset]) begin
                    if (failures < 8)
                        $display("QKV_MISMATCH output=%0d got=%02x expected=%02x",
                                 output_offset, output_mem[output_offset],
                                 expected_mem[output_offset]);
                    failures = failures + 1;
                end
            end
            if (failures != 0)
                $fatal(1, "QKV reference mismatch count=%0d", failures);
            if ((write_requests != TOTAL_OUTPUTS/16) ||
                (output_writes != TOTAL_OUTPUTS))
                $fatal(1, "QKV write coverage requests=%0d bytes=%0d",
                       write_requests, output_writes);
        end
    endtask

    task automatic run_fused;
        begin
            launch_command(OPCODE_FUSED_QKV, Q_OUTPUTS, 16'hf301,
                           WGT_BASE, FUSED_DST_BASE, META_BASE);
            await_completion(16'hf301, 0);
        end
    endtask

    initial begin
        if (!$value$plusargs("EXPECTED=%s", expected_path))
            $fatal(1, "missing +EXPECTED=<path>");
        $readmemh(expected_path, expected_mem);
        rst_n = 1'b0;
        csr_valid = 1'b0;
        csr_write = 1'b0;
        csr_addr = 32'd0;
        csr_wdata = 64'd0;
        csr_wstrb = 8'd0;
        csr_rready = 1'b0;
        cmd_valid = 1'b0;
        cmd_opcode = 8'd0;
        cmd_flags = 8'd0;
        cmd_layer_id = 8'd0;
        cmd_m = 16'd1;
        cmd_n = 16'd0;
        cmd_k = K_SIZE;
        cmd_sequence_position = 16'd0;
        cmd_completion_tag = 16'd0;
        cmd_src0_addr = ACT_BASE;
        cmd_src1_addr = WGT_BASE;
        cmd_dst_addr = FUSED_DST_BASE;
        cmd_scale_addr = META_BASE;
        cmd_scratch_addr = 64'd0;
        cmd_done_ready = 1'b0;
        corrupt_mode = $test$plusargs("CORRUPT");
        host_commands = 0;
        failures = 0;

        repeat (5) @(posedge clk);
        @(negedge clk);
        rst_n = 1'b1;
        repeat (2) @(posedge clk);
        enable_shell();
        sequence_start_cycle = cycle_count;

        if ($test$plusargs("CORRUPT")) begin
            launch_command(OPCODE_FUSED_QKV, Q_OUTPUTS, 16'hfc01,
                           WGT_BASE, FUSED_DST_BASE, META_BASE);
            await_completion(16'hfc01, 1);
            if (!corrupt_sent || (write_requests != 0))
                $fatal(1, "corruption coverage missing sent=%0d writes=%0d",
                       corrupt_sent, write_requests);
            $display("ACE2_FUSED_QKV_CORRUPTION_PASS commands=%0d reads=%0d writes=%0d",
                     host_commands, read_requests, write_requests);
            $finish;
        end

        if ($test$plusargs("RESET")) begin
            launch_command(OPCODE_FUSED_QKV, Q_OUTPUTS, 16'hfa01,
                           WGT_BASE, FUSED_DST_BASE, META_BASE);
            while (activation_read_requests < 8)
                @(negedge clk);
            rst_n = 1'b0;
            #1;
            if (busy || cmd_done_valid || mem_req_valid || mem_wvalid)
                $fatal(1, "mid-command reset did not quiesce shell");
            repeat (4) @(posedge clk);
            @(negedge clk);
            rst_n = 1'b1;
            corrupt_mode = 1'b0;
            host_commands = 0;
            repeat (2) @(posedge clk);
            enable_shell();
            sequence_start_cycle = cycle_count;
            run_fused();
            check_outputs();
            $display("ACE2_FUSED_QKV_RESET_PASS commands=%0d outputs_checked=%0d cycles=%0d reads=%0d",
                     host_commands, TOTAL_OUTPUTS,
                     cycle_count - sequence_start_cycle, read_requests);
            $finish;
        end

        if ($test$plusargs("LEGACY")) begin
            launch_command(OPCODE_W4A8_PROJ, Q_OUTPUTS, 16'hf101,
                           WGT_BASE, LEGACY_DST_BASE, META_BASE);
            await_completion(16'hf101, 0);
            launch_command(OPCODE_W4A8_PROJ, KV_OUTPUTS, 16'hf102,
                           WGT_BASE + Q_WEIGHT_BYTES,
                           LEGACY_DST_BASE + Q_OUTPUTS,
                           META_BASE + Q_META_BYTES);
            await_completion(16'hf102, 0);
            launch_command(OPCODE_W4A8_PROJ, KV_OUTPUTS, 16'hf103,
                           WGT_BASE + Q_WEIGHT_BYTES + KV_WEIGHT_BYTES,
                           LEGACY_DST_BASE + Q_OUTPUTS + KV_OUTPUTS,
                           META_BASE + Q_META_BYTES + KV_META_BYTES);
            await_completion(16'hf103, 0);
        end else if ($test$plusargs("FUSED")) begin
            run_fused();
        end else begin
            $fatal(1, "select +LEGACY or +FUSED");
        end

        check_outputs();
        final_cycles = cycle_count - sequence_start_cycle;
        if ((request_stalls == 0) || (write_stalls == 0))
            $fatal(1, "backpressure coverage missing request=%0d write=%0d",
                   request_stalls, write_stalls);
        $display("ACE2_FUSED_QKV_PASS mode=%s commands=%0d outputs_checked=%0d simulator_cycles=%0d memory_read_requests=%0d activation_read_requests=%0d weight_read_requests=%0d metadata_read_requests=%0d memory_write_requests=%0d backpressure_stalls=%0d",
                 $test$plusargs("LEGACY") ? "legacy" : "fused",
                 host_commands, TOTAL_OUTPUTS, final_cycles, read_requests,
                 activation_read_requests, weight_read_requests,
                 metadata_read_requests, write_requests,
                 request_stalls + write_stalls);
        $finish;
    end
endmodule

`default_nettype wire
