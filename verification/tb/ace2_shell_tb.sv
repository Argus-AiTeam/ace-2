`timescale 1ns/1ps
`default_nettype none

import ace2_pkg::*;

module ace2_shell_tb;
    localparam integer LANES = 16;
    localparam integer ACT_WIDTH = 8;
    localparam integer GAIN_WIDTH = 16;
    localparam [63:0] ACT_BASE = 64'h0000_0000_0000_1000;
    localparam [63:0] GAIN_BASE = 64'h0000_0000_0000_4000;
    localparam [63:0] OUT_BASE = 64'h0000_0000_0000_8000;
    localparam [63:0] PROJ_ACT_BASE = 64'h0000_0000_0010_0000;
    localparam [63:0] PROJ_WEIGHT_BASE = 64'h0000_0000_0020_0000;
    localparam [63:0] PROJ_META_BASE = 64'h0000_0000_0030_0000;
    localparam [63:0] PROJ_OUT_BASE = 64'h0000_0000_0040_0000;
    localparam [63:0] ROPE_ACT_BASE = 64'h0000_0000_0050_0000;
    localparam [63:0] ROPE_SCALE_BASE = 64'h0000_0000_0060_0000;
    localparam [63:0] ROPE_TABLE_BASE = 64'h0000_0000_0080_0000;
    localparam [63:0] ROPE_OUT_BASE = 64'h0000_0000_0100_0000;
    localparam [7:0] ROPE_BASIS_ROTATION_HALF_DEGREES = 8'd45;
    localparam [63:0] KV_K_BASE = 64'h0000_0000_0110_0000;
    localparam [63:0] KV_V_BASE = 64'h0000_0000_0120_0000;
    localparam [63:0] KV_META_BASE = 64'h0000_0000_0130_0000;
    localparam [63:0] KV_CACHE_BASE = 64'h0000_0000_0140_0000;
    localparam [63:0] ATTN_Q_BASE = 64'h0000_0000_0150_0000;
    localparam [63:0] ATTN_K_BASE = 64'h0000_0000_0160_0000;
    localparam [63:0] ATTN_META_BASE = 64'h0000_0000_0168_0000;
    localparam [63:0] ATTN_SCORE_OUT_BASE = 64'h0000_0000_0170_0000;
    localparam [63:0] SOFTMAX_SCORE_BASE = 64'h0000_0000_0180_0000;
    localparam [63:0] SOFTMAX_OUT_BASE = 64'h0000_0000_0190_0000;
    localparam [63:0] ATTN_VALUE_PROB_BASE = 64'h0000_0000_01a0_0000;
    localparam [63:0] ATTN_VALUE_V_BASE = 64'h0000_0000_01b0_0000;
    localparam [63:0] ATTN_VALUE_OUT_BASE = 64'h0000_0000_01c0_0000;
    localparam [63:0] ATTN_COMPOSE_SCORE_BASE = 64'h0000_0000_01d0_0000;
    localparam [63:0] ATTN_COMPOSE_VALUE_BASE = 64'h0000_0000_01e0_0000;
    localparam [63:0] ATTN_COMPOSE_OUT_BASE = 64'h0000_0000_01f0_0000;
    localparam [63:0] OPROJ_ACT_BASE = 64'h0000_0000_0200_0000;
    localparam [63:0] OPROJ_WEIGHT_BASE = 64'h0000_0000_0210_0000;
    localparam [63:0] OPROJ_META_BASE = 64'h0000_0000_0220_0000;
    localparam [63:0] OPROJ_OUT_BASE = 64'h0000_0000_0230_0000;
    localparam [63:0] RESIDUAL_LHS_BASE = 64'h0000_0000_0240_0000;
    localparam [63:0] RESIDUAL_RHS_BASE = 64'h0000_0000_0250_0000;
    localparam [63:0] RESIDUAL_OUT_BASE = 64'h0000_0000_0260_0000;
    localparam [63:0] POST_RMS_ACT_BASE = 64'h0000_0000_0270_0000;
    localparam [63:0] POST_RMS_GAIN_BASE = 64'h0000_0000_0280_0000;
    localparam [63:0] POST_RMS_OUT_BASE = 64'h0000_0000_0290_0000;
    localparam [63:0] MLP_PROJ_ACT_BASE = 64'h0000_0000_0400_0000;
    localparam [63:0] MLP_PROJ_WEIGHT_BASE = 64'h0000_0000_0500_0000;
    localparam [63:0] MLP_PROJ_META_BASE = 64'h0000_0000_0a00_0000;
    localparam [63:0] MLP_PROJ_OUT_BASE = 64'h0000_0000_0b00_0000;
    localparam [63:0] MLP_UP_PROJ_ACT_BASE = 64'h0000_0000_0c00_0000;
    localparam [63:0] MLP_UP_PROJ_WEIGHT_BASE = 64'h0000_0000_0d00_0000;
    localparam [63:0] MLP_UP_PROJ_META_BASE = 64'h0000_0000_1200_0000;
    localparam [63:0] MLP_UP_PROJ_OUT_BASE = 64'h0000_0000_1300_0000;
    localparam [63:0] SILU_GATE_BASE = 64'h0000_0000_1400_0000;
    localparam [63:0] SILU_UP_BASE = 64'h0000_0000_1410_0000;
    localparam [63:0] SILU_META_BASE = 64'h0000_0000_1420_0000;
    localparam [63:0] SILU_OUT_BASE = SILU_UP_BASE + 64'd4864;
    localparam [63:0] MLP_DOWN_PROJ_ACT_BASE = 64'h0000_0000_1440_0000;
    localparam [63:0] MLP_DOWN_PROJ_WEIGHT_BASE = 64'h0000_0000_1500_0000;
    localparam [63:0] MLP_DOWN_PROJ_META_BASE = 64'h0000_0000_1a00_0000;
    localparam [63:0] MLP_DOWN_PROJ_OUT_BASE = 64'h0000_0000_1b00_0000;
    localparam [63:0] MLP_RESIDUAL_DOWN_BASE = 64'h0000_0000_1c00_0000;
    localparam [63:0] MLP_RESIDUAL_STREAM_BASE = 64'h0000_0000_1d00_0000;
    localparam [63:0] MLP_RESIDUAL_OUT_BASE = 64'h0000_0000_1e00_0000;
    localparam integer KV_DATA_BEATS = 8;
    localparam integer KV_TOTAL_BEATS = 17;
    localparam integer KV_BYTES_PER_TOKEN = 272;
    localparam integer ATTN_SCORE_BEATS = 4;
    localparam integer OBSERVED_MAX_WRITES = 1024;
    localparam [3:0] ATTN_COMPOSE_CORE_VALUE_ACCUM_STATE = 4'd8;

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
    reg [7:0] sram_req_ready;
    wire [7:0] sram_write;
    wire [8*12-1:0] sram_addr;
    wire [8*128-1:0] sram_wdata;
    wire [8*16-1:0] sram_wstrb;
    reg [8*128-1:0] sram_rdata;
    reg [7:0] sram_rvalid;
    wire busy;
    wire cmd_done_valid;
    reg cmd_done_ready;
    wire [15:0] cmd_done_tag;
    wire cmd_done_error;
    wire [47:0] cmd_done_sumsq;
    wire [31:0] cmd_done_inv;
    wire cmd_done_saturation;

    integer failures;
    integer current_case;
    integer current_proj_case;
    integer current_proj_m;
    integer current_proj_n;
    integer current_rope_case;
    integer observed_count;
    integer cycle_count;
    integer pending_read;
    integer read_delay;
    integer pending_error;
    integer pending_write_response;
    integer write_response_delay;
    integer next_read_delay;
    integer next_write_response_delay;
    integer case_index;
    integer layer_index;
    integer command_start_cycle;
    integer last_command_cycles;
    integer max_command_cycles;
    integer total_success_cycles;
    integer success_runs;
    integer descriptor_error_cases;
    integer memory_error_cases;
    integer response_protocol_cases;
    integer watchdog_error_cases;
    integer reset_busy_cases;
    integer expected_beats_per_row;
    integer expected_flat;
    integer qproj_cases_seen;
    integer kproj_cases_seen;
    integer vproj_cases_seen;
    integer rope_cases_seen;
    integer rope_q_cases_seen;
    integer rope_k_cases_seen;
    integer current_rope_beats;
    integer current_rope_table_stride;
    integer rope_table_offset;
    integer rope_table_beat;
    integer current_kv_case;
    integer current_kv_sequence;
    integer kv_write_cases_seen;
    integer current_attn_score_case;
    integer current_attn_metadata_mode;
    integer attn_score_cases_seen;
    integer attn_score_metadata_error_cases;
    integer current_softmax_case;
    integer softmax_cases_seen;
    integer current_attn_value_case;
    integer attn_value_cases_seen;
    integer current_attn_compose_case;
    integer attn_compose_cases_seen;
    integer oproj_cases_seen;
    integer mlp_gate_proj_cases_seen;
    integer mlp_up_proj_cases_seen;
    integer mlp_down_proj_cases_seen;
    integer mlp_residual_cases_seen;
    integer current_silu_case;
    integer silu_gate_cases_seen;
    integer silu_gate_read_count;
    integer silu_up_read_count;
    integer silu_write_req_count;
    integer proj_act_read_count;
    integer proj_weight_read_count;
    integer proj_meta_read_count;
    integer proj_write_req_count;
    integer current_residual_case;
    integer current_mlp_residual_case;
    integer current_post_rms_case;
    integer residual_cases_seen;
    integer post_rms_cases_seen;
    integer smoke_opcode;
    integer qproj_stride_only_mode;
    integer mlp_up_only_mode;
    integer mlp_down_only_mode;
    integer mlp_residual_only_mode;
    integer silu_only_mode;
    integer layer_sweep_only_mode;
    integer final_rmsnorm_only_mode;
    integer lm_head_only_mode;
    integer attn_compose_only_mode;
    integer attn_score_only_mode;
    integer rope_only_mode;
    integer layer_sweep_cases_seen;
    integer layer_sweep_high_layer;
    integer oproj_writes_seen;
    integer oproj_total_cycles;
    integer oproj_max_cycles;
    integer oproj_phase_cycle;
    integer oproj_command_accept_cycle;
    integer oproj_completion_accept_cycle;
    reg [63:0] current_proj_dst_addr;
    reg smoke_mode;
    reg oproj_active;
    reg [63:0] oproj_signature;
    reg [63:0] pending_addr;
    reg [63:0] pending_write_addr;
    reg [63:0] silu_max_gate_read_addr;
    reg [63:0] silu_max_up_read_addr;
    reg [63:0] proj_min_weight_read_addr;
    reg [63:0] proj_max_weight_read_addr;
    reg [7:0] pending_tag;
    reg [7:0] pending_write_tag;
    reg pending_write_error;
    reg [127:0] observed_output [0:OBSERVED_MAX_WRITES-1];
    reg [63:0] observed_write_addr [0:OBSERVED_MAX_WRITES-1];
    reg [15:0] observed_write_strb [0:OBSERVED_MAX_WRITES-1];
    reg inject_read_error;
    reg [7:0] inject_read_error_tag;
    reg inject_write_error;
    reg inject_read_wrong_tag;
    reg inject_write_wrong_tag;
    reg force_mem_req_stall;
    reg attn_retime_trace_mode;

    `include "../generated/rmsnorm_vectors.svh"
    `include "../generated/projection_vectors.svh"
    `include "../generated/rope_vectors.svh"
    `include "../generated/attention_score_vectors.svh"
    `include "../generated/softmax_vectors.svh"
    `include "../generated/attention_value_vectors.svh"
    `include "../generated/attention_compose_vectors.svh"
    `include "../generated/residual_vectors.svh"
    `include "../generated/mlp_residual_vectors.svh"
    `include "../generated/post_attention_rmsnorm_vectors.svh"
    `include "../generated/silu_gate_shell_vectors.svh"

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
        .sram_req_valid_o(sram_req_valid),
        .sram_req_ready_i(sram_req_ready),
        .sram_write_o(sram_write),
        .sram_addr_o(sram_addr),
        .sram_wdata_o(sram_wdata),
        .sram_wstrb_o(sram_wstrb),
        .sram_rdata_i(sram_rdata),
        .sram_rvalid_i(sram_rvalid),
        .busy_o(busy),
        .cmd_done_valid_o(cmd_done_valid),
        .cmd_done_ready_i(cmd_done_ready),
        .cmd_done_tag_o(cmd_done_tag),
        .cmd_done_error_o(cmd_done_error),
        .cmd_done_sumsq_o(cmd_done_sumsq),
        .cmd_done_inv_rms_q30_o(cmd_done_inv),
        .cmd_done_saturation_seen_o(cmd_done_saturation)
    );

    initial begin
        clk = 1'b0;
        forever #5 clk = ~clk;
    end

    always @(negedge clk) begin
        if (attn_retime_trace_mode && rst_n &&
            ((dut.state_q == 7'd124) ||
             (dut.state_q == 7'd68) ||
             (dut.state_q == 7'd121) ||
             (dut.state_q == 7'd122) ||
             (dut.state_q == 7'd125) ||
             (dut.state_q == 7'd123) ||
             mem_wvalid || cmd_done_valid)) begin
            $display(
                "ATTN_RETIME_TRACE cycle=%0d state=%0d token=%0d rounded=%h max=%h max_neg=%0d center_delta=%h center_score=%h mem_req=%0d mem_write=%0d mem_addr=%h mem_wvalid=%0d mem_wdata=%h mem_wstrb=%h done=%0d done_error=%0d done_sat=%0d",
                cycle_count, dut.state_q, dut.attn_token_idx_q,
                dut.attn_rounded_abs_q, dut.attn_score_max_magnitude_q,
                dut.attn_score_max_negative_q, dut.attn_center_delta_q,
                dut.attn_center_score_q, mem_req_valid, mem_req_write,
                mem_req_addr, mem_wvalid, mem_wdata, mem_wstrb,
                cmd_done_valid, cmd_done_error, cmd_done_saturation
            );
        end
    end

    function [127:0] make_kv_word;
        input integer selected_case;
        input integer stream_id;
        input integer beat;
        integer lane;
        reg [127:0] value;
        reg [7:0] byte_value;
        begin
            value = 128'd0;
            for (lane = 0; lane < LANES; lane = lane + 1) begin
                byte_value = (selected_case * 8'd37) + (stream_id * 8'd83) + (beat * 8'd16) + lane[7:0];
                value[lane*8 +: 8] = byte_value;
            end
            make_kv_word = value;
        end
    endfunction

    function [127:0] lookup_proj_activation_data;
        input [63:0] addr;
        input [63:0] base_addr;
        integer physical_beat;
        integer proj_row;
        integer row_beat;
        integer flat;
        begin
            physical_beat = (addr - base_addr) >> 4;
            proj_row = physical_beat / proj_case_input_beats[current_proj_case];
            row_beat = physical_beat % proj_case_input_beats[current_proj_case];
            flat = current_proj_case*PROJ_MAX_ROWS*PROJ_MAX_INPUT_BEATS +
                   proj_row*PROJ_MAX_INPUT_BEATS + row_beat;
            lookup_proj_activation_data = proj_input_beats[flat];
        end
    endfunction

    function [127:0] lookup_read_data;
        input [63:0] addr;
        integer beat;
        reg [255:0] gain_word;
        begin
            lookup_read_data = 128'd0;
            if ((addr >= ACT_BASE) && (addr < (ACT_BASE + TEST_BEATS*16))) begin
                beat = (addr - ACT_BASE) >> 4;
                lookup_read_data = test_input_beats[current_case*TEST_BEATS + beat];
            end else if ((addr >= GAIN_BASE) && (addr < (GAIN_BASE + TEST_BEATS*32))) begin
                beat = (addr - GAIN_BASE) >> 5;
                gain_word = test_gain_beats[current_case*TEST_BEATS + beat];
                if (addr[4]) begin
                    lookup_read_data = gain_word[255:128];
                end else begin
                    lookup_read_data = gain_word[127:0];
                end
            end else if ((addr >= PROJ_ACT_BASE) &&
                         (addr < (PROJ_ACT_BASE + proj_case_rows[current_proj_case]*proj_case_input_beats[current_proj_case]*16))) begin
                lookup_read_data = lookup_proj_activation_data(addr, PROJ_ACT_BASE);
            end else if ((addr >= PROJ_WEIGHT_BASE) &&
                         (addr < (PROJ_WEIGHT_BASE + proj_case_outputs[current_proj_case]*PROJ_WEIGHT_BEATS_PER_OUTPUT*16))) begin
                beat = (addr - PROJ_WEIGHT_BASE) >> 4;
                lookup_read_data = proj_weight_beats[proj_case_weight_offset[current_proj_case] + beat];
            end else if ((addr >= PROJ_META_BASE) &&
                         (addr < (PROJ_META_BASE + proj_case_outputs[current_proj_case]*16))) begin
                beat = (addr - PROJ_META_BASE) >> 4;
                lookup_read_data = proj_meta_beats[proj_case_meta_offset[current_proj_case] + beat];
            end else if ((addr >= OPROJ_ACT_BASE) &&
                         (addr < (OPROJ_ACT_BASE + proj_case_rows[current_proj_case]*proj_case_input_beats[current_proj_case]*16))) begin
                lookup_read_data = lookup_proj_activation_data(addr, OPROJ_ACT_BASE);
            end else if ((addr >= OPROJ_WEIGHT_BASE) &&
                         (addr < (OPROJ_WEIGHT_BASE + proj_case_outputs[current_proj_case]*PROJ_WEIGHT_BEATS_PER_OUTPUT*16))) begin
                beat = (addr - OPROJ_WEIGHT_BASE) >> 4;
                lookup_read_data = proj_weight_beats[proj_case_weight_offset[current_proj_case] + beat];
            end else if ((addr >= OPROJ_META_BASE) &&
                         (addr < (OPROJ_META_BASE + proj_case_outputs[current_proj_case]*16))) begin
                beat = (addr - OPROJ_META_BASE) >> 4;
                lookup_read_data = proj_meta_beats[proj_case_meta_offset[current_proj_case] + beat];
            end else if ((addr >= MLP_PROJ_ACT_BASE) &&
                         (addr < (MLP_PROJ_ACT_BASE + proj_case_rows[current_proj_case]*proj_case_input_beats[current_proj_case]*16))) begin
                lookup_read_data = lookup_proj_activation_data(addr, MLP_PROJ_ACT_BASE);
            end else if ((addr >= MLP_PROJ_WEIGHT_BASE) &&
                         (addr < (MLP_PROJ_WEIGHT_BASE + proj_case_outputs[current_proj_case]*PROJ_WEIGHT_BEATS_PER_OUTPUT*16))) begin
                beat = (addr - MLP_PROJ_WEIGHT_BASE) >> 4;
                lookup_read_data = proj_weight_beats[proj_case_weight_offset[current_proj_case] + beat];
            end else if ((addr >= MLP_PROJ_META_BASE) &&
                         (addr < (MLP_PROJ_META_BASE + proj_case_outputs[current_proj_case]*16))) begin
                beat = (addr - MLP_PROJ_META_BASE) >> 4;
                lookup_read_data = proj_meta_beats[proj_case_meta_offset[current_proj_case] + beat];
            end else if ((addr >= MLP_UP_PROJ_ACT_BASE) &&
                         (addr < (MLP_UP_PROJ_ACT_BASE + proj_case_rows[current_proj_case]*proj_case_input_beats[current_proj_case]*16))) begin
                lookup_read_data = lookup_proj_activation_data(addr, MLP_UP_PROJ_ACT_BASE);
            end else if ((addr >= MLP_UP_PROJ_WEIGHT_BASE) &&
                         (addr < (MLP_UP_PROJ_WEIGHT_BASE + proj_case_outputs[current_proj_case]*PROJ_WEIGHT_BEATS_PER_OUTPUT*16))) begin
                beat = (addr - MLP_UP_PROJ_WEIGHT_BASE) >> 4;
                lookup_read_data = proj_weight_beats[proj_case_weight_offset[current_proj_case] + beat];
            end else if ((addr >= MLP_UP_PROJ_META_BASE) &&
                         (addr < (MLP_UP_PROJ_META_BASE + proj_case_outputs[current_proj_case]*16))) begin
                beat = (addr - MLP_UP_PROJ_META_BASE) >> 4;
                lookup_read_data = proj_meta_beats[proj_case_meta_offset[current_proj_case] + beat];
            end else if ((addr >= MLP_DOWN_PROJ_ACT_BASE) &&
                         (addr < (MLP_DOWN_PROJ_ACT_BASE + proj_case_rows[current_proj_case]*proj_case_input_beats[current_proj_case]*16))) begin
                lookup_read_data = lookup_proj_activation_data(addr, MLP_DOWN_PROJ_ACT_BASE);
            end else if ((addr >= MLP_DOWN_PROJ_WEIGHT_BASE) &&
                         (addr < (MLP_DOWN_PROJ_WEIGHT_BASE + proj_case_outputs[current_proj_case]*proj_case_weight_beats_per_output[current_proj_case]*16))) begin
                beat = (addr - MLP_DOWN_PROJ_WEIGHT_BASE) >> 4;
                lookup_read_data = proj_weight_beats[proj_case_weight_offset[current_proj_case] + beat];
            end else if ((addr >= MLP_DOWN_PROJ_META_BASE) &&
                         (addr < (MLP_DOWN_PROJ_META_BASE + proj_case_outputs[current_proj_case]*16))) begin
                beat = (addr - MLP_DOWN_PROJ_META_BASE) >> 4;
                lookup_read_data = proj_meta_beats[proj_case_meta_offset[current_proj_case] + beat];
            end else if ((addr >= SILU_GATE_BASE) &&
                         (addr < SILU_GATE_BASE + SILU_MAX_INPUT_BEATS*16)) begin
                beat = (addr - SILU_GATE_BASE) >> 4;
                lookup_read_data =
                    silu_gate_beats[current_silu_case*SILU_MAX_INPUT_BEATS + beat];
            end else if ((addr >= SILU_UP_BASE) &&
                         (addr < SILU_UP_BASE + SILU_MAX_INPUT_BEATS*16)) begin
                beat = (addr - SILU_UP_BASE) >> 4;
                lookup_read_data =
                    silu_up_beats[current_silu_case*SILU_MAX_INPUT_BEATS + beat];
            end else if (addr == SILU_META_BASE) begin
                lookup_read_data = {
                    80'd0,
                    silu_case_zero_point[current_silu_case],
                    2'd0,
                    silu_case_right_shift[current_silu_case],
                    silu_case_multiplier[current_silu_case]
                };
            end else if ((addr >= RESIDUAL_LHS_BASE) &&
                         (addr < (RESIDUAL_LHS_BASE + RESIDUAL_BEATS*16))) begin
                beat = (addr - RESIDUAL_LHS_BASE) >> 4;
                lookup_read_data =
                    residual_lhs_beats[current_residual_case*RESIDUAL_BEATS + beat];
            end else if ((addr >= RESIDUAL_RHS_BASE) &&
                         (addr < (RESIDUAL_RHS_BASE + RESIDUAL_BEATS*16))) begin
                beat = (addr - RESIDUAL_RHS_BASE) >> 4;
                lookup_read_data =
                    residual_rhs_beats[current_residual_case*RESIDUAL_BEATS + beat];
            end else if ((addr >= MLP_RESIDUAL_DOWN_BASE) &&
                         (addr < (MLP_RESIDUAL_DOWN_BASE + MLP_RESIDUAL_BEATS*16))) begin
                beat = (addr - MLP_RESIDUAL_DOWN_BASE) >> 4;
                lookup_read_data =
                    mlp_residual_down_beats[
                        current_mlp_residual_case*MLP_RESIDUAL_BEATS + beat
                    ];
            end else if ((addr >= MLP_RESIDUAL_STREAM_BASE) &&
                         (addr < (MLP_RESIDUAL_STREAM_BASE + MLP_RESIDUAL_BEATS*16))) begin
                beat = (addr - MLP_RESIDUAL_STREAM_BASE) >> 4;
                lookup_read_data =
                    mlp_residual_stream_beats[
                        current_mlp_residual_case*MLP_RESIDUAL_BEATS + beat
                    ];
            end else if ((addr >= POST_RMS_ACT_BASE) &&
                         (addr < (POST_RMS_ACT_BASE + POST_RMS_BEATS*16))) begin
                beat = (addr - POST_RMS_ACT_BASE) >> 4;
                lookup_read_data =
                    post_rms_input_beats[current_post_rms_case*POST_RMS_BEATS + beat];
            end else if ((addr >= POST_RMS_GAIN_BASE) &&
                         (addr < (POST_RMS_GAIN_BASE + POST_RMS_BEATS*32))) begin
                beat = (addr - POST_RMS_GAIN_BASE) >> 5;
                gain_word =
                    post_rms_gain_beats[current_post_rms_case*POST_RMS_BEATS + beat];
                lookup_read_data = addr[4] ? gain_word[255:128] : gain_word[127:0];
            end else if ((addr >= ROPE_ACT_BASE) && (addr < (ROPE_ACT_BASE + current_rope_beats*16))) begin
                beat = (addr - ROPE_ACT_BASE) >> 4;
                lookup_read_data = rope_input_beats[current_rope_case*ROPE_BEATS + beat];
            end else if ((addr >= ROPE_SCALE_BASE) && (addr < (ROPE_SCALE_BASE + current_rope_beats*32))) begin
                beat = (addr - ROPE_SCALE_BASE) >> 4;
                lookup_read_data = rope_scale_beats[current_rope_case*ROPE_BEATS*2 + beat];
            end else if ((addr >= (ROPE_TABLE_BASE + rope_sequence_position[current_rope_case]*current_rope_table_stride)) &&
                         (addr < (ROPE_TABLE_BASE + rope_sequence_position[current_rope_case]*current_rope_table_stride + current_rope_beats*64))) begin
                rope_table_offset = addr - (ROPE_TABLE_BASE + rope_sequence_position[current_rope_case]*current_rope_table_stride);
                rope_table_beat = rope_table_offset >> 6;
                case ((rope_table_offset >> 4) & 3)
                    0: lookup_read_data = rope_cos_beats[current_rope_case*ROPE_BEATS*2 + rope_table_beat*2];
                    1: lookup_read_data = rope_cos_beats[current_rope_case*ROPE_BEATS*2 + rope_table_beat*2 + 1];
                    2: lookup_read_data = rope_sin_beats[current_rope_case*ROPE_BEATS*2 + rope_table_beat*2];
                    default: lookup_read_data = rope_sin_beats[current_rope_case*ROPE_BEATS*2 + rope_table_beat*2 + 1];
                endcase
            end else if ((addr >= KV_K_BASE) && (addr < (KV_K_BASE + KV_DATA_BEATS*16))) begin
                beat = (addr - KV_K_BASE) >> 4;
                lookup_read_data = make_kv_word(current_kv_case, 0, beat);
            end else if ((addr >= KV_V_BASE) && (addr < (KV_V_BASE + KV_DATA_BEATS*16))) begin
                beat = (addr - KV_V_BASE) >> 4;
                lookup_read_data = make_kv_word(current_kv_case, 1, beat);
            end else if ((addr >= KV_META_BASE) && (addr < (KV_META_BASE + 16))) begin
                lookup_read_data = make_kv_word(current_kv_case, 2, 0);
            end else if ((addr >= ATTN_Q_BASE) && (addr < (ATTN_Q_BASE + ATTN_SCORE_BEATS*16))) begin
                beat = (addr - ATTN_Q_BASE) >> 4;
                lookup_read_data = attn_score_q_beats[current_attn_score_case*ATTN_SCORE_BEATS_PER_VECTOR + beat];
            end else if ((addr >= ATTN_K_BASE) && (addr < (ATTN_K_BASE + ATTN_SCORE_CONTEXT_MAX*ATTN_SCORE_BEATS_PER_VECTOR*16))) begin
                beat = (addr - ATTN_K_BASE) >> 4;
                lookup_read_data = attn_score_k_beats[current_attn_score_case*ATTN_SCORE_CONTEXT_MAX*ATTN_SCORE_BEATS_PER_VECTOR + beat];
            end else if (addr == ATTN_META_BASE) begin
                case (current_attn_metadata_mode)
                    1: lookup_read_data = ATTN_SCORE_INVALID_NEGATIVE_MULTIPLIER;
                    2: lookup_read_data = ATTN_SCORE_INVALID_RESERVED_BITS;
                    default: lookup_read_data = {
                        90'd0,
                        attn_score_right_shift[current_attn_score_case],
                        attn_score_multiplier[current_attn_score_case]
                    };
                endcase
            end else if (addr == SOFTMAX_SCORE_BASE) begin
                lookup_read_data = softmax_score_word[current_softmax_case];
            end else if (addr == ATTN_VALUE_PROB_BASE) begin
                lookup_read_data = attn_value_probability_word[current_attn_value_case];
            end else if ((addr >= ATTN_VALUE_V_BASE) &&
                         (addr < (ATTN_VALUE_V_BASE +
                                  ATTN_VALUE_CONTEXT_MAX*ATTN_VALUE_BEATS_PER_VECTOR*16))) begin
                beat = (addr - ATTN_VALUE_V_BASE) >> 4;
                lookup_read_data =
                    attn_value_v_beats[current_attn_value_case*
                                       ATTN_VALUE_CONTEXT_MAX*
                                       ATTN_VALUE_BEATS_PER_VECTOR + beat];
            end else if ((addr >= ATTN_COMPOSE_SCORE_BASE) &&
                         (addr < (ATTN_COMPOSE_SCORE_BASE +
                                  ATTN_COMPOSE_MAX_TILES*16))) begin
                beat = (addr - ATTN_COMPOSE_SCORE_BASE) >> 4;
                lookup_read_data =
                    attn_compose_score_tiles[
                        current_attn_compose_case*
                        ATTN_COMPOSE_MAX_TILES + beat];
            end else if ((addr >= ATTN_COMPOSE_VALUE_BASE) &&
                         (addr < (ATTN_COMPOSE_VALUE_BASE +
                                  ATTN_COMPOSE_MAX_CONTEXT*
                                  ATTN_COMPOSE_BEATS*16))) begin
                beat = (addr - ATTN_COMPOSE_VALUE_BASE) >> 4;
                lookup_read_data =
                    attn_compose_value_beats[
                        current_attn_compose_case*
                        ATTN_COMPOSE_MAX_CONTEXT*
                        ATTN_COMPOSE_BEATS + beat];
            end else begin
                lookup_read_data = 128'hbad0_bad0_bad0_bad0_bad0_bad0_bad0_bad0;
            end
        end
    endfunction

    function [63:0] fold_oproj_word;
        input [63:0] signature;
        input [127:0] word;
        input integer selected_case;
        input integer beat;
        reg [63:0] ordinal;
        begin
            ordinal = selected_case*PROJ_BEATS + beat;
            fold_oproj_word = {signature[62:0], signature[63]} ^
                              word[63:0] ^ word[127:64] ^ ordinal;
        end
    endfunction

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            cycle_count <= 0;
            oproj_phase_cycle <= 0;
            oproj_command_accept_cycle <= 0;
            oproj_completion_accept_cycle <= 0;
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
            pending_write_addr <= 64'd0;
            pending_tag <= 8'd0;
            pending_error <= 0;
            read_delay <= 0;
            pending_write_response <= 0;
            write_response_delay <= 0;
            next_read_delay <= -1;
            next_write_response_delay <= -1;
            observed_count <= 0;
            silu_gate_read_count <= 0;
            silu_up_read_count <= 0;
            silu_write_req_count <= 0;
            proj_act_read_count <= 0;
            proj_weight_read_count <= 0;
            proj_meta_read_count <= 0;
            proj_write_req_count <= 0;
            silu_max_gate_read_addr <= 64'd0;
            silu_max_up_read_addr <= 64'd0;
            proj_min_weight_read_addr <= 64'hffff_ffff_ffff_ffff;
            proj_max_weight_read_addr <= 64'd0;
            inject_read_error <= 1'b0;
            inject_read_error_tag <= 8'd0;
            inject_write_error <= 1'b0;
            inject_read_wrong_tag <= 1'b0;
            inject_write_wrong_tag <= 1'b0;
            pending_write_tag <= 8'd0;
            pending_write_error <= 1'b0;
            force_mem_req_stall <= 1'b0;
        end else begin
            cycle_count <= cycle_count + 1;
            if (oproj_active) begin
                oproj_phase_cycle <= oproj_phase_cycle + 1;
            end
            if (oproj_active && cmd_valid && cmd_ready) begin
                oproj_command_accept_cycle <= cycle_count;
            end
            mem_req_ready <= force_mem_req_stall ? 1'b0 :
                             (oproj_active ?
                              (oproj_phase_cycle[1:0] != 2'b01) :
                              (cycle_count[1:0] != 2'b01));
            mem_wready <= oproj_active ?
                          (oproj_phase_cycle[2:0] != 3'b011) :
                          (cycle_count[2:0] != 3'b011);
            if (cmd_done_valid && (pending_write_response || mem_bvalid)) begin
                $fatal(1, "COMPLETION_BEFORE_WRITE_RESPONSE_RETIRE");
            end
            if (mem_rvalid && mem_rready) begin
                mem_rvalid <= 1'b0;
                mem_rerror <= 1'b0;
            end
            if (!mem_rvalid && pending_read) begin
                if (read_delay != 0) begin
                    read_delay <= read_delay - 1;
                end else begin
                    mem_rvalid <= 1'b1;
                    mem_rdata <= lookup_read_data(pending_addr);
                    mem_rtag <= pending_tag;
                    mem_rerror <= pending_error[0];
                    pending_read <= 0;
                end
            end
            if (mem_req_valid && mem_req_ready && !mem_req_write) begin
                pending_read <= 1;
                pending_addr <= mem_req_addr;
                pending_tag <= inject_read_wrong_tag ? (mem_req_tag ^ 8'h01) : mem_req_tag;
                pending_error <= inject_read_error && (mem_req_tag == inject_read_error_tag);
                if (mem_req_tag == 8'hb1) begin
                    silu_gate_read_count <= silu_gate_read_count + 1;
                    if ((silu_gate_read_count == 0) ||
                        (mem_req_addr > silu_max_gate_read_addr)) begin
                        silu_max_gate_read_addr <= mem_req_addr;
                    end
                end
                if (mem_req_tag == 8'hb2) begin
                    silu_up_read_count <= silu_up_read_count + 1;
                    if ((silu_up_read_count == 0) ||
                        (mem_req_addr > silu_max_up_read_addr)) begin
                        silu_max_up_read_addr <= mem_req_addr;
                    end
                end
                if (mem_req_tag == 8'h40 || mem_req_tag == 8'h41) begin
                    proj_act_read_count <= proj_act_read_count + 1;
                end
                if (mem_req_tag == 8'h42) begin
                    proj_weight_read_count <= proj_weight_read_count + 1;
                    if ((proj_weight_read_count == 0) ||
                        (mem_req_addr < proj_min_weight_read_addr)) begin
                        proj_min_weight_read_addr <= mem_req_addr;
                    end
                    if ((proj_weight_read_count == 0) ||
                        (mem_req_addr > proj_max_weight_read_addr)) begin
                        proj_max_weight_read_addr <= mem_req_addr;
                    end
                end
                if (mem_req_tag == 8'h43) begin
                    proj_meta_read_count <= proj_meta_read_count + 1;
                end
                if (inject_read_wrong_tag) begin
                    inject_read_wrong_tag <= 1'b0;
                end
                if (inject_read_error && (mem_req_tag == inject_read_error_tag)) begin
                    inject_read_error <= 1'b0;
                end
                if (next_read_delay >= 0) begin
                    read_delay <= next_read_delay;
                    next_read_delay <= -1;
                end else begin
                    read_delay <= oproj_active ?
                                  {1'b0, oproj_phase_cycle[0]} :
                                  {1'b0, cycle_count[0]};
                end
            end
            if (mem_req_valid && mem_req_ready && mem_req_write) begin
                pending_write_addr <= mem_req_addr;
                if (mem_req_tag == 8'hb3) begin
                    silu_write_req_count <= silu_write_req_count + 1;
                end
                if (mem_req_tag == 8'h44) begin
                    proj_write_req_count <= proj_write_req_count + 1;
                end
            end
            if (mem_wvalid && mem_wready) begin
                if (observed_count < OBSERVED_MAX_WRITES) begin
                    observed_output[observed_count] <= mem_wdata;
                    observed_write_addr[observed_count] <= pending_write_addr;
                        observed_write_strb[observed_count] <= mem_wstrb;
                end
                observed_count <= observed_count + 1;
                pending_write_response <= 1;
                pending_write_tag <= inject_write_wrong_tag ? (mem_wtag ^ 8'h01) : mem_wtag;
                pending_write_error <= inject_write_error;
                if (next_write_response_delay >= 0) begin
                    write_response_delay <= next_write_response_delay;
                    next_write_response_delay <= -1;
                end else begin
                    write_response_delay <= 0;
                end
                if (inject_write_error) begin
                    inject_write_error <= 1'b0;
                end
                if (inject_write_wrong_tag) begin
                    inject_write_wrong_tag <= 1'b0;
                end
            end
            if (mem_bvalid && mem_bready) begin
                mem_bvalid <= 1'b0;
                mem_berror <= 1'b0;
            end
            if (!mem_bvalid && pending_write_response) begin
                if (write_response_delay != 0) begin
                    write_response_delay <= write_response_delay - 1;
                end else begin
                    mem_bvalid <= 1'b1;
                    mem_btag <= pending_write_tag;
                    mem_berror <= pending_write_error;
                    pending_write_response <= 0;
                end
            end
        end
    end

    task apply_reset;
        integer idx;
        begin
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
            cmd_m = 16'd0;
            cmd_n = 16'd0;
            cmd_k = 16'd0;
            cmd_sequence_position = 16'd0;
            cmd_completion_tag = 16'd0;
            cmd_src0_addr = 64'd0;
            cmd_src1_addr = 64'd0;
            cmd_dst_addr = 64'd0;
            cmd_scale_addr = 64'd0;
            cmd_scratch_addr = 64'd0;
            cmd_done_ready = 1'b0;
            sram_req_ready = 8'hff;
            sram_rdata = {8*128{1'b0}};
            sram_rvalid = 8'd0;
            for (idx = 0; idx < OBSERVED_MAX_WRITES; idx = idx + 1) begin
                observed_output[idx] = 128'd0;
                observed_write_addr[idx] = 64'd0;
                observed_write_strb[idx] = 16'd0;
            end
            repeat (5) @(posedge clk);
            rst_n = 1'b1;
            repeat (2) @(posedge clk);
        end
    endtask

    task send_mlp_residual_cmd_full;
        input integer selected_case;
        input [15:0] n_value;
        input [63:0] down_addr;
        input [63:0] residual_addr;
        input [63:0] dst_addr;
        input [15:0] tag;
        begin
            current_mlp_residual_case = selected_case;
            send_residual_cmd_full(
                selected_case, n_value, down_addr, residual_addr, dst_addr, tag
            );
        end
    endtask

    task send_mlp_residual_cmd_layer_full;
        input integer selected_case;
        input integer selected_layer;
        input [15:0] n_value;
        input [63:0] down_addr;
        input [63:0] residual_addr;
        input [63:0] dst_addr;
        input [15:0] tag;
        begin
            current_mlp_residual_case = selected_case;
            send_residual_cmd_layer_full(
                selected_case, selected_layer, n_value, down_addr, residual_addr,
                dst_addr, tag
            );
        end
    endtask

    task wait_mlp_residual_done_and_compare;
        input integer selected_case;
        input integer expect_error;
        input [15:0] expected_tag;
        integer guard;
        integer beat;
        begin
            guard = 0;
            while (!cmd_done_valid && (guard < 500000)) begin
                guard = guard + 1;
                @(posedge clk);
            end
            if (!cmd_done_valid) begin
                $display("MLP_RESIDUAL_TIMEOUT case=%0d", selected_case);
                failures = failures + 1;
            end else begin
                if (cmd_done_tag !== expected_tag ||
                    cmd_done_error !== (expect_error != 0)) begin
                    $display("MLP_RESIDUAL_COMPLETION_MISMATCH tag=%04x error=%0d",
                             cmd_done_tag, cmd_done_error);
                    failures = failures + 1;
                end
                if (expect_error != 2) begin
                    mlp_residual_cases_seen = mlp_residual_cases_seen + 1;
                    last_command_cycles = cycle_count - command_start_cycle;
                    if (observed_count != MLP_RESIDUAL_BEATS ||
                        cmd_done_saturation !==
                            mlp_residual_expected_saturation[selected_case]) begin
                        $display("MLP_RESIDUAL_STATUS_MISMATCH case=%0d writes=%0d saturation=%0d",
                                 selected_case, observed_count, cmd_done_saturation);
                        failures = failures + 1;
                    end
                    for (beat = 0; beat < MLP_RESIDUAL_BEATS; beat = beat + 1) begin
                        if (observed_write_addr[beat] !==
                                MLP_RESIDUAL_OUT_BASE + beat*16 ||
                            observed_write_strb[beat] !== 16'hffff ||
                            observed_output[beat] !==
                                mlp_residual_expected_beats[
                                    selected_case*MLP_RESIDUAL_BEATS + beat
                                ]) begin
                            $display("MLP_RESIDUAL_OUTPUT_MISMATCH case=%0d beat=%0d addr=%016x strb=%04x",
                                     selected_case, beat, observed_write_addr[beat],
                                     observed_write_strb[beat]);
                            failures = failures + 1;
                        end
                    end
                end
            end
            @(negedge clk);
            cmd_done_ready = 1'b1;
            @(posedge clk);
            @(negedge clk);
            cmd_done_ready = 1'b0;
        end
    endtask

    task run_mlp_residual_cases;
        integer selected_case;
        integer local_descriptor_errors;
        integer local_memory_errors;
        integer local_protocol_cases;
        integer saturation_cases;
        reg [63:0] status_value;
        begin
            local_descriptor_errors = 0;
            local_memory_errors = 0;
            local_protocol_cases = 0;
            saturation_cases = 0;
            csr_write64(ACE2_CSR_ERROR_STATUS, 64'hffff_ffff_ffff_ffff);

            force_mem_req_stall = 1'b1;
            send_mlp_residual_cmd_full(
                0, 16'd896, MLP_RESIDUAL_DOWN_BASE, MLP_RESIDUAL_STREAM_BASE,
                MLP_RESIDUAL_OUT_BASE, 16'h5600
            );
            repeat (7) @(posedge clk);
            force_mem_req_stall = 1'b0;
            wait_mlp_residual_done_and_compare(
                0, mlp_residual_expected_saturation[0], 16'h5600
            );

            send_mlp_residual_cmd_full(
                0, 16'd895, MLP_RESIDUAL_DOWN_BASE, MLP_RESIDUAL_STREAM_BASE,
                MLP_RESIDUAL_OUT_BASE, 16'h5601
            );
            wait_mlp_residual_done_and_compare(0, 2, 16'h5601);
            local_descriptor_errors = local_descriptor_errors + 1;
            descriptor_error_cases = descriptor_error_cases + 1;
            csr_read64(ACE2_CSR_ERROR_STATUS, status_value);
            if (!status_value[ACE2_ERR_DESCRIPTOR]) failures = failures + 1;
            csr_write64(ACE2_CSR_ERROR_STATUS, 64'hffff_ffff_ffff_ffff);

            send_mlp_residual_cmd_full(
                0, 16'd896, MLP_RESIDUAL_DOWN_BASE,
                MLP_RESIDUAL_STREAM_BASE + 64'd1, MLP_RESIDUAL_OUT_BASE, 16'h5602
            );
            wait_mlp_residual_done_and_compare(0, 2, 16'h5602);
            local_descriptor_errors = local_descriptor_errors + 1;
            descriptor_error_cases = descriptor_error_cases + 1;
            csr_read64(ACE2_CSR_ERROR_STATUS, status_value);
            if (!status_value[ACE2_ERR_DESCRIPTOR]) failures = failures + 1;
            csr_write64(ACE2_CSR_ERROR_STATUS, 64'hffff_ffff_ffff_ffff);

            inject_read_error = 1'b1;
            inject_read_error_tag = 8'ha0;
            send_mlp_residual_cmd_full(
                1, 16'd896, MLP_RESIDUAL_DOWN_BASE, MLP_RESIDUAL_STREAM_BASE,
                MLP_RESIDUAL_OUT_BASE, 16'h5603
            );
            wait_mlp_residual_done_and_compare(1, 2, 16'h5603);
            local_memory_errors = local_memory_errors + 1;
            local_protocol_cases = local_protocol_cases + 1;
            memory_error_cases = memory_error_cases + 1;
            response_protocol_cases = response_protocol_cases + 1;
            csr_write64(ACE2_CSR_ERROR_STATUS, 64'hffff_ffff_ffff_ffff);

            inject_read_wrong_tag = 1'b1;
            send_mlp_residual_cmd_full(
                1, 16'd896, MLP_RESIDUAL_DOWN_BASE, MLP_RESIDUAL_STREAM_BASE,
                MLP_RESIDUAL_OUT_BASE, 16'h5604
            );
            wait_mlp_residual_done_and_compare(1, 2, 16'h5604);
            local_memory_errors = local_memory_errors + 1;
            local_protocol_cases = local_protocol_cases + 1;
            memory_error_cases = memory_error_cases + 1;
            response_protocol_cases = response_protocol_cases + 1;
            csr_write64(ACE2_CSR_ERROR_STATUS, 64'hffff_ffff_ffff_ffff);

            inject_write_error = 1'b1;
            send_mlp_residual_cmd_full(
                1, 16'd896, MLP_RESIDUAL_DOWN_BASE, MLP_RESIDUAL_STREAM_BASE,
                MLP_RESIDUAL_OUT_BASE, 16'h5605
            );
            wait_mlp_residual_done_and_compare(1, 2, 16'h5605);
            local_memory_errors = local_memory_errors + 1;
            local_protocol_cases = local_protocol_cases + 1;
            memory_error_cases = memory_error_cases + 1;
            response_protocol_cases = response_protocol_cases + 1;
            csr_write64(ACE2_CSR_ERROR_STATUS, 64'hffff_ffff_ffff_ffff);

            inject_write_wrong_tag = 1'b1;
            send_mlp_residual_cmd_full(
                1, 16'd896, MLP_RESIDUAL_DOWN_BASE, MLP_RESIDUAL_STREAM_BASE,
                MLP_RESIDUAL_OUT_BASE, 16'h5606
            );
            wait_mlp_residual_done_and_compare(1, 2, 16'h5606);
            local_memory_errors = local_memory_errors + 1;
            local_protocol_cases = local_protocol_cases + 1;
            memory_error_cases = memory_error_cases + 1;
            response_protocol_cases = response_protocol_cases + 1;
            csr_write64(ACE2_CSR_ERROR_STATUS, 64'hffff_ffff_ffff_ffff);

            next_read_delay = 7;
            next_write_response_delay = 9;
            send_mlp_residual_cmd_full(
                1, 16'd896, MLP_RESIDUAL_DOWN_BASE, MLP_RESIDUAL_STREAM_BASE,
                MLP_RESIDUAL_OUT_BASE, 16'h5607
            );
            wait_mlp_residual_done_and_compare(
                1, mlp_residual_expected_saturation[1], 16'h5607
            );
            local_protocol_cases = local_protocol_cases + 1;
            response_protocol_cases = response_protocol_cases + 1;
            if (mlp_residual_expected_saturation[1])
                csr_write64(ACE2_CSR_ERROR_STATUS, 64'hffff_ffff_ffff_ffff);

            for (selected_case = 2;
                 selected_case < MLP_RESIDUAL_CASE_COUNT;
                 selected_case = selected_case + 1) begin
                send_mlp_residual_cmd_full(
                    selected_case, 16'd896, MLP_RESIDUAL_DOWN_BASE,
                    MLP_RESIDUAL_STREAM_BASE, MLP_RESIDUAL_OUT_BASE,
                    16'h5610 + selected_case
                );
                wait_mlp_residual_done_and_compare(
                    selected_case,
                    mlp_residual_expected_saturation[selected_case],
                    16'h5610 + selected_case
                );
                if (mlp_residual_expected_saturation[selected_case])
                    csr_write64(ACE2_CSR_ERROR_STATUS, 64'hffff_ffff_ffff_ffff);
            end
            for (selected_case = 0;
                 selected_case < MLP_RESIDUAL_CASE_COUNT;
                 selected_case = selected_case + 1)
                saturation_cases = saturation_cases +
                    mlp_residual_expected_saturation[selected_case];
            $display("ACE2_SHELL_MLP_RESIDUAL_TB_PASS cases=%0d writes=%0d cycles=%0d saturation_cases=%0d descriptor_errors=%0d memory_errors=%0d protocol_cases=%0d",
                     mlp_residual_cases_seen, observed_count, last_command_cycles,
                     saturation_cases, local_descriptor_errors, local_memory_errors,
                     local_protocol_cases);
        end
    endtask

    task csr_write64;
        input [31:0] addr;
        input [63:0] data;
        begin
            @(negedge clk);
            csr_addr = addr;
            csr_wdata = data;
            csr_wstrb = 8'hff;
            csr_write = 1'b1;
            csr_valid = 1'b1;
            while (!csr_ready) @(posedge clk);
            @(posedge clk);
            @(negedge clk);
            csr_valid = 1'b0;
            csr_write = 1'b0;
            csr_wstrb = 8'd0;
            csr_rready = 1'b1;
            while (!csr_rvalid) @(posedge clk);
            @(posedge clk);
            @(negedge clk);
            csr_rready = 1'b0;
        end
    endtask

    task csr_read64;
        input [31:0] addr;
        output [63:0] data;
        begin
            @(negedge clk);
            csr_addr = addr;
            csr_write = 1'b0;
            csr_valid = 1'b1;
            while (!csr_ready) @(posedge clk);
            @(posedge clk);
            @(negedge clk);
            csr_valid = 1'b0;
            while (!csr_rvalid) @(posedge clk);
            data = csr_rdata;
            csr_rready = 1'b1;
            @(posedge clk);
            @(negedge clk);
            csr_rready = 1'b0;
        end
    endtask

    task wait_done_and_compare;
        input integer selected_case;
        input integer expect_error;
        input [15:0] expected_tag;
        integer guard;
        integer beat;
        begin
            guard = 0;
            while (!cmd_done_valid && (guard < 500000)) begin
                guard = guard + 1;
                @(posedge clk);
            end
            if (!cmd_done_valid) begin
                $display("SHELL_TIMEOUT case=%0d", selected_case);
                failures = failures + 1;
            end else begin
                if (cmd_done_tag !== expected_tag) begin
                    $display("SHELL_TAG_MISMATCH got=%0d expected=%0d", cmd_done_tag, expected_tag);
                    failures = failures + 1;
                end
                if (cmd_done_error !== expect_error[0]) begin
                    $display("SHELL_ERROR_MISMATCH got=%0d expected=%0d", cmd_done_error, expect_error);
                    failures = failures + 1;
                end
                if (!expect_error) begin
                    last_command_cycles = cycle_count - command_start_cycle;
                    total_success_cycles = total_success_cycles + last_command_cycles;
                    success_runs = success_runs + 1;
                    if (last_command_cycles > max_command_cycles) begin
                        max_command_cycles = last_command_cycles;
                    end
                    if (observed_count != TEST_BEATS) begin
                        $display("SHELL_WRITE_COUNT_MISMATCH got=%0d expected=%0d", observed_count, TEST_BEATS);
                        failures = failures + 1;
                    end
                    if (cmd_done_sumsq !== test_expected_sumsq[selected_case]) begin
                        $display("SHELL_SUMSQ_MISMATCH got=%0d expected=%0d", cmd_done_sumsq, test_expected_sumsq[selected_case]);
                        failures = failures + 1;
                    end
                    if (cmd_done_inv !== test_expected_inv[selected_case]) begin
                        $display("SHELL_INV_MISMATCH got=%0d expected=%0d", cmd_done_inv, test_expected_inv[selected_case]);
                        failures = failures + 1;
                    end
                    if (cmd_done_saturation !== test_expected_saturation[selected_case]) begin
                        $display("SHELL_SAT_MISMATCH got=%0d expected=%0d", cmd_done_saturation, test_expected_saturation[selected_case]);
                        failures = failures + 1;
                    end
                    for (beat = 0; beat < TEST_BEATS; beat = beat + 1) begin
                        if (observed_output[beat] !== test_expected_beats[selected_case*TEST_BEATS + beat]) begin
                            $display("SHELL_OUTPUT_MISMATCH case=%0d beat=%0d got=%032x expected=%032x",
                                     selected_case, beat, observed_output[beat],
                                     test_expected_beats[selected_case*TEST_BEATS + beat]);
                            failures = failures + 1;
                        end
                    end
                end
            end
            @(negedge clk);
            cmd_done_ready = 1'b1;
            @(posedge clk);
            @(negedge clk);
            cmd_done_ready = 1'b0;
        end
    endtask

    task send_rmsnorm_cmd_full;
        input integer selected_case;
        input integer selected_layer;
        input [7:0] opcode;
        input [15:0] m_value;
        input [15:0] n_value;
        input [15:0] k_value;
        input [63:0] src_addr;
        input [63:0] dst_addr;
        input [63:0] scale_addr;
        input [15:0] tag;
        integer guard;
        integer idx;
        begin
            current_case = selected_case;
            observed_count = 0;
            pending_read = 0;
            mem_rvalid = 1'b0;
            for (idx = 0; idx < OBSERVED_MAX_WRITES; idx = idx + 1) begin
                observed_output[idx] = 128'd0;
                observed_write_addr[idx] = 64'd0;
            end
            @(negedge clk);
            cmd_opcode = opcode;
            cmd_flags = 8'h09;
            cmd_layer_id = selected_layer;
            cmd_m = m_value;
            cmd_n = n_value;
            cmd_k = k_value;
            cmd_sequence_position = 16'd0;
            cmd_completion_tag = tag;
            cmd_src0_addr = src_addr;
            cmd_src1_addr = 64'd0;
            cmd_dst_addr = dst_addr;
            cmd_scale_addr = scale_addr;
            cmd_scratch_addr = 64'd0;
            cmd_valid = 1'b1;
            guard = 0;
            while (!cmd_ready && (guard < 256)) begin
                guard = guard + 1;
                @(posedge clk);
            end
            if (!cmd_ready) begin
                $display("SHELL_CMD_NOT_READY");
                failures = failures + 1;
            end
            @(posedge clk);
            command_start_cycle = cycle_count;
            @(negedge clk);
            cmd_valid = 1'b0;
        end
    endtask

    task send_rmsnorm_cmd;
        input integer selected_case;
        input integer selected_layer;
        input [7:0] opcode;
        input [15:0] n_value;
        input [63:0] src_addr;
        input [15:0] tag;
        begin
            send_rmsnorm_cmd_full(selected_case, selected_layer, opcode, 16'd1,
                                  n_value, 16'd0, src_addr, OUT_BASE, GAIN_BASE,
                                  tag);
        end
    endtask

    task send_post_rms_cmd_layer_full;
        input integer selected_case;
        input integer selected_layer;
        input [15:0] n_value;
        input [63:0] scale_addr;
        input [15:0] tag;
        begin
            current_post_rms_case = selected_case;
            send_rmsnorm_cmd_full(selected_case, selected_layer,
                                  ACE2_OPCODE_RMSNORM, 16'd1, n_value, 16'd0,
                                  POST_RMS_ACT_BASE, POST_RMS_OUT_BASE,
                                  scale_addr, tag);
        end
    endtask

    task send_post_rms_cmd_full;
        input integer selected_case;
        input [15:0] n_value;
        input [63:0] scale_addr;
        input [15:0] tag;
        begin
            send_post_rms_cmd_layer_full(selected_case, 0, n_value, scale_addr, tag);
        end
    endtask

    task wait_post_rms_done_and_compare;
        input integer selected_case;
        input integer expect_error;
        input [15:0] expected_tag;
        integer guard;
        integer beat;
        begin
            guard = 0;
            while (!cmd_done_valid && (guard < 500000)) begin
                guard = guard + 1;
                @(posedge clk);
            end
            if (!cmd_done_valid) begin
                $display("POST_RMS_TIMEOUT case=%0d", selected_case);
                failures = failures + 1;
            end else begin
                if (cmd_done_tag !== expected_tag || cmd_done_error !== (expect_error != 0)) begin
                    $display("POST_RMS_COMPLETION_MISMATCH tag=%04x error=%0d", cmd_done_tag,
                             cmd_done_error);
                    failures = failures + 1;
                end
                if (expect_error != 2) begin
                    post_rms_cases_seen = post_rms_cases_seen + 1;
                    if (observed_count != POST_RMS_BEATS ||
                        cmd_done_sumsq !== post_rms_expected_sumsq[selected_case] ||
                        cmd_done_inv !== post_rms_expected_inv[selected_case] ||
                        cmd_done_saturation !== post_rms_expected_saturation[selected_case]) begin
                        $display("POST_RMS_STATUS_MISMATCH case=%0d writes=%0d", selected_case,
                                 observed_count);
                        failures = failures + 1;
                    end
                    for (beat = 0; beat < POST_RMS_BEATS; beat = beat + 1) begin
                        if (observed_output[beat] !==
                            post_rms_expected_beats[selected_case*POST_RMS_BEATS + beat]) begin
                            $display("POST_RMS_OUTPUT_MISMATCH case=%0d beat=%0d", selected_case,
                                     beat);
                            failures = failures + 1;
                        end
                    end
                end
            end
            @(negedge clk);
            cmd_done_ready = 1'b1;
            @(posedge clk);
            @(negedge clk);
            cmd_done_ready = 1'b0;
        end
    endtask

    task send_residual_cmd_layer_full;
        input integer selected_case;
        input integer selected_layer;
        input [15:0] n_value;
        input [63:0] lhs_addr;
        input [63:0] rhs_addr;
        input [63:0] dst_addr;
        input [15:0] tag;
        integer guard;
        integer idx;
        begin
            current_residual_case = selected_case;
            observed_count = 0;
            pending_read = 0;
            mem_rvalid = 1'b0;
            for (idx = 0; idx < OBSERVED_MAX_WRITES; idx = idx + 1) begin
                observed_output[idx] = 128'd0;
                observed_write_addr[idx] = 64'd0;
            end
            @(negedge clk);
            cmd_opcode = ACE2_OPCODE_RESIDUAL_ADD;
            cmd_flags = 8'h09;
            cmd_layer_id = selected_layer;
            cmd_m = 16'd1;
            cmd_n = n_value;
            cmd_k = 16'd0;
            cmd_sequence_position = 16'd0;
            cmd_completion_tag = tag;
            cmd_src0_addr = lhs_addr;
            cmd_src1_addr = rhs_addr;
            cmd_dst_addr = dst_addr;
            cmd_scale_addr = 64'd0;
            cmd_scratch_addr = 64'd0;
            cmd_valid = 1'b1;
            guard = 0;
            while (!cmd_ready && (guard < 256)) begin
                guard = guard + 1;
                @(posedge clk);
            end
            if (!cmd_ready) begin
                $display("RESIDUAL_CMD_NOT_READY");
                failures = failures + 1;
            end
            @(posedge clk);
            command_start_cycle = cycle_count;
            @(negedge clk);
            cmd_valid = 1'b0;
        end
    endtask

    task send_residual_cmd_full;
        input integer selected_case;
        input [15:0] n_value;
        input [63:0] lhs_addr;
        input [63:0] rhs_addr;
        input [63:0] dst_addr;
        input [15:0] tag;
        begin
            send_residual_cmd_layer_full(selected_case, 0, n_value, lhs_addr, rhs_addr, dst_addr, tag);
        end
    endtask

    task wait_residual_done_and_compare;
        input integer selected_case;
        input integer expect_error;
        input [15:0] expected_tag;
        integer guard;
        integer beat;
        begin
            guard = 0;
            while (!cmd_done_valid && (guard < 500000)) begin
                guard = guard + 1;
                @(posedge clk);
            end
            if (!cmd_done_valid) begin
                $display("RESIDUAL_TIMEOUT case=%0d", selected_case);
                failures = failures + 1;
            end else begin
                if (cmd_done_tag !== expected_tag || cmd_done_error !== (expect_error != 0)) begin
                    $display("RESIDUAL_COMPLETION_MISMATCH tag=%04x error=%0d", cmd_done_tag,
                             cmd_done_error);
                    failures = failures + 1;
                end
                if (expect_error != 2) begin
                    residual_cases_seen = residual_cases_seen + 1;
                    if (observed_count != RESIDUAL_BEATS ||
                        cmd_done_saturation !== residual_expected_saturation[selected_case]) begin
                        $display("RESIDUAL_STATUS_MISMATCH case=%0d writes=%0d saturation=%0d",
                                 selected_case, observed_count, cmd_done_saturation);
                        failures = failures + 1;
                    end
                    for (beat = 0; beat < RESIDUAL_BEATS; beat = beat + 1) begin
                        if (observed_output[beat] !==
                            residual_expected_beats[selected_case*RESIDUAL_BEATS + beat]) begin
                            $display("RESIDUAL_OUTPUT_MISMATCH case=%0d beat=%0d", selected_case,
                                     beat);
                            failures = failures + 1;
                        end
                    end
                end
            end
            @(negedge clk);
            cmd_done_ready = 1'b1;
            @(posedge clk);
            @(negedge clk);
            cmd_done_ready = 1'b0;
        end
    endtask

    task run_vector_family_cases;
        integer vector_case;
        integer reset_guard;
        reg [63:0] status_value;
        begin
            force_mem_req_stall = 1'b1;
            send_post_rms_cmd_full(1, 16'd896, POST_RMS_GAIN_BASE, 16'h5001);
            repeat (7) @(posedge clk);
            force_mem_req_stall = 1'b0;
            wait_post_rms_done_and_compare(1, 0, 16'h5001);

            send_post_rms_cmd_full(0, 16'd896, POST_RMS_GAIN_BASE + 64'd2, 16'h5002);
            wait_post_rms_done_and_compare(0, 2, 16'h5002);
            csr_read64(ACE2_CSR_ERROR_STATUS, status_value);
            if (!status_value[ACE2_ERR_DESCRIPTOR]) failures = failures + 1;
            csr_write64(ACE2_CSR_ERROR_STATUS, 64'hffff_ffff_ffff_ffff);

            inject_read_error = 1'b1;
            inject_read_error_tag = 8'h10;
            send_post_rms_cmd_full(2, 16'd896, POST_RMS_GAIN_BASE, 16'h5003);
            wait_post_rms_done_and_compare(2, 2, 16'h5003);
            csr_read64(ACE2_CSR_ERROR_STATUS, status_value);
            if (!status_value[ACE2_ERR_MEMORY]) failures = failures + 1;
            csr_write64(ACE2_CSR_ERROR_STATUS, 64'hffff_ffff_ffff_ffff);

            next_read_delay = 12;
            send_post_rms_cmd_full(3, 16'd896, POST_RMS_GAIN_BASE, 16'h5004);
            while (!pending_read && !mem_rvalid) @(posedge clk);
            csr_write64(ACE2_CSR_CONTROL, 64'h3);
            reset_guard = 0;
            while ((busy || pending_read || mem_rvalid) && (reset_guard < 64)) begin
                reset_guard = reset_guard + 1;
                @(posedge clk);
            end
            if (busy || pending_read || mem_rvalid) begin
                $display("POST_RMS_RESET_DRAIN_TIMEOUT busy=%0d pending=%0d rvalid=%0d",
                         busy, pending_read, mem_rvalid);
                failures = failures + 1;
            end
            csr_read64(ACE2_CSR_ERROR_STATUS, status_value);
            if (!status_value[ACE2_ERR_RESET_BUSY] || cmd_done_valid) failures = failures + 1;
            csr_write64(ACE2_CSR_ERROR_STATUS, 64'hffff_ffff_ffff_ffff);
            csr_write64(ACE2_CSR_CONTROL, 64'h1);

            force_mem_req_stall = 1'b1;
            send_residual_cmd_full(0, 16'd896, RESIDUAL_LHS_BASE, RESIDUAL_RHS_BASE,
                                   RESIDUAL_OUT_BASE, 16'h5101);
            repeat (7) @(posedge clk);
            force_mem_req_stall = 1'b0;
            wait_residual_done_and_compare(0, 0, 16'h5101);

            send_residual_cmd_full(0, 16'd896, RESIDUAL_LHS_BASE,
                                   RESIDUAL_RHS_BASE + 64'd1, RESIDUAL_OUT_BASE, 16'h5102);
            wait_residual_done_and_compare(0, 2, 16'h5102);
            csr_read64(ACE2_CSR_ERROR_STATUS, status_value);
            if (!status_value[ACE2_ERR_DESCRIPTOR]) failures = failures + 1;
            csr_write64(ACE2_CSR_ERROR_STATUS, 64'hffff_ffff_ffff_ffff);

            inject_read_error = 1'b1;
            inject_read_error_tag = 8'ha1;
            send_residual_cmd_full(1, 16'd896, RESIDUAL_LHS_BASE, RESIDUAL_RHS_BASE,
                                   RESIDUAL_OUT_BASE, 16'h5103);
            wait_residual_done_and_compare(1, 2, 16'h5103);
            csr_read64(ACE2_CSR_ERROR_STATUS, status_value);
            if (!status_value[ACE2_ERR_MEMORY]) failures = failures + 1;
            csr_write64(ACE2_CSR_ERROR_STATUS, 64'hffff_ffff_ffff_ffff);

            next_read_delay = 12;
            send_residual_cmd_full(1, 16'd896, RESIDUAL_LHS_BASE, RESIDUAL_RHS_BASE,
                                   RESIDUAL_OUT_BASE, 16'h5104);
            while (!pending_read && !mem_rvalid) @(posedge clk);
            csr_write64(ACE2_CSR_CONTROL, 64'h3);
            reset_guard = 0;
            while ((busy || pending_read || mem_rvalid) && (reset_guard < 64)) begin
                reset_guard = reset_guard + 1;
                @(posedge clk);
            end
            if (busy || pending_read || mem_rvalid) begin
                $display("RESIDUAL_RESET_DRAIN_TIMEOUT busy=%0d pending=%0d rvalid=%0d",
                         busy, pending_read, mem_rvalid);
                failures = failures + 1;
            end
            csr_read64(ACE2_CSR_ERROR_STATUS, status_value);
            if (!status_value[ACE2_ERR_RESET_BUSY] || cmd_done_valid) failures = failures + 1;
            csr_write64(ACE2_CSR_ERROR_STATUS, 64'hffff_ffff_ffff_ffff);
            csr_write64(ACE2_CSR_CONTROL, 64'h1);

            for (vector_case = 0; vector_case < POST_RMS_CASE_COUNT; vector_case = vector_case + 1) begin
                send_post_rms_cmd_full(vector_case, 16'd896, POST_RMS_GAIN_BASE,
                                       16'h5200 + vector_case);
                wait_post_rms_done_and_compare(
                    vector_case, post_rms_expected_saturation[vector_case],
                    16'h5200 + vector_case);
                if (post_rms_expected_saturation[vector_case])
                    csr_write64(ACE2_CSR_ERROR_STATUS, 64'hffff_ffff_ffff_ffff);
            end
            for (vector_case = 0; vector_case < RESIDUAL_CASE_COUNT; vector_case = vector_case + 1) begin
                send_residual_cmd_full(vector_case, 16'd896, RESIDUAL_LHS_BASE,
                                       RESIDUAL_RHS_BASE, RESIDUAL_OUT_BASE,
                                       16'h5300 + vector_case);
                wait_residual_done_and_compare(
                    vector_case, residual_expected_saturation[vector_case],
                    16'h5300 + vector_case);
                if (residual_expected_saturation[vector_case])
                    csr_write64(ACE2_CSR_ERROR_STATUS, 64'hffff_ffff_ffff_ffff);
            end
        end
    endtask

    task send_qproj_cmd_full;
        input integer selected_case;
        input integer selected_layer;
        input [15:0] m_value;
        input [15:0] n_value;
        input [15:0] k_value;
        input [63:0] src0_addr;
        input [63:0] src1_addr;
        input [63:0] dst_addr;
        input [63:0] scale_addr;
        input [15:0] tag;
        integer guard;
        integer idx;
        begin
            current_proj_case = selected_case;
            current_proj_m = m_value;
            current_proj_n = n_value;
            current_proj_dst_addr = dst_addr;
            observed_count = 0;
            pending_read = 0;
            mem_rvalid = 1'b0;
            for (idx = 0; idx < OBSERVED_MAX_WRITES; idx = idx + 1) begin
                observed_output[idx] = 128'd0;
                observed_write_addr[idx] = 64'd0;
            end
            @(negedge clk);
            cmd_opcode = ACE2_OPCODE_W4A8_PROJ;
            cmd_flags = 8'h09;
            cmd_layer_id = selected_layer;
            cmd_m = m_value;
            cmd_n = n_value;
            cmd_k = k_value;
            cmd_sequence_position = 16'd0;
            cmd_completion_tag = tag;
            cmd_src0_addr = src0_addr;
            cmd_src1_addr = src1_addr;
            cmd_dst_addr = dst_addr;
            cmd_scale_addr = scale_addr;
            cmd_scratch_addr = 64'd0;
            cmd_valid = 1'b1;
            guard = 0;
            while (!cmd_ready && (guard < 256)) begin
                guard = guard + 1;
                @(posedge clk);
            end
            if (!cmd_ready) begin
                $display("SHELL_QPROJ_CMD_NOT_READY");
                failures = failures + 1;
            end
            @(posedge clk);
            command_start_cycle = cycle_count;
            @(negedge clk);
            cmd_valid = 1'b0;
        end
    endtask

    task send_qproj_cmd;
        input integer selected_case;
        input integer selected_layer;
        input [15:0] tag;
        begin
            send_qproj_cmd_full(selected_case, selected_layer, proj_case_rows[selected_case], 16'd896, 16'd896,
                                PROJ_ACT_BASE, PROJ_WEIGHT_BASE, PROJ_OUT_BASE, PROJ_META_BASE, tag);
        end
    endtask

    task send_oproj_cmd;
        input integer selected_case;
        input integer selected_layer;
        input [15:0] tag;
        begin
            oproj_active = 1'b1;
            oproj_phase_cycle = 0;
            send_qproj_cmd_full(selected_case, selected_layer, proj_case_rows[selected_case], 16'd896, 16'd896,
                                OPROJ_ACT_BASE, OPROJ_WEIGHT_BASE, OPROJ_OUT_BASE, OPROJ_META_BASE, tag);
        end
    endtask

    task send_mlp_proj_cmd;
        input integer selected_case;
        input [15:0] tag;
        begin
            send_qproj_cmd_full(selected_case, 0, proj_case_rows[selected_case],
                                proj_case_outputs[selected_case], 16'd896,
                                MLP_PROJ_ACT_BASE, MLP_PROJ_WEIGHT_BASE,
                                MLP_PROJ_OUT_BASE, MLP_PROJ_META_BASE, tag);
        end
    endtask

    task send_mlp_up_proj_cmd;
        input integer selected_case;
        input [15:0] tag;
        begin
            send_qproj_cmd_full(selected_case, 0, proj_case_rows[selected_case],
                                proj_case_outputs[selected_case], 16'd896,
                                MLP_UP_PROJ_ACT_BASE, MLP_UP_PROJ_WEIGHT_BASE,
                                MLP_UP_PROJ_OUT_BASE, MLP_UP_PROJ_META_BASE, tag);
        end
    endtask

    task send_mlp_down_proj_cmd;
        input integer selected_case;
        input [15:0] tag;
        begin
            send_qproj_cmd_full(selected_case, 0, proj_case_rows[selected_case],
                                16'd896, proj_case_k[selected_case],
                                MLP_DOWN_PROJ_ACT_BASE, MLP_DOWN_PROJ_WEIGHT_BASE,
                                MLP_DOWN_PROJ_OUT_BASE, MLP_DOWN_PROJ_META_BASE, tag);
        end
    endtask

    task send_lm_head_cmd;
        input integer selected_case;
        input [15:0] tag;
        begin
            send_qproj_cmd_full(selected_case, 24, 16'd1,
                                16'(ACE2_LM_HEAD_TILE_SIZE), 16'd896,
                                PROJ_ACT_BASE, PROJ_WEIGHT_BASE,
                                PROJ_OUT_BASE, PROJ_META_BASE, tag);
        end
    endtask

    task wait_qproj_done_and_compare;
        input integer selected_case;
        input integer expect_error;
        input [15:0] expected_tag;
        integer guard;
        integer beat;
        integer expected_count;
        reg [63:0] expected_addr;
        begin
            cmd_done_ready = 1'b1;
            guard = 0;
            while (!cmd_done_valid && (guard < 40000000)) begin
                guard = guard + 1;
                @(posedge clk);
            end
            if (!cmd_done_valid) begin
                $display("SHELL_QPROJ_TIMEOUT case=%0d observed=%0d", selected_case, observed_count);
                failures = failures + 1;
            end else begin
                if (cmd_done_tag !== expected_tag) begin
                    $display("SHELL_QPROJ_TAG_MISMATCH got=%0d expected=%0d", cmd_done_tag, expected_tag);
                    failures = failures + 1;
                end
                if (cmd_done_error !== expect_error[0]) begin
                    $display("SHELL_QPROJ_ERROR_MISMATCH got=%0d expected=%0d", cmd_done_error, expect_error);
                    failures = failures + 1;
                end
                if (!expect_error || proj_expected_saturation[selected_case]) begin
                    if (cmd_done_saturation !== proj_expected_saturation[selected_case]) begin
                        $display("SHELL_QPROJ_SAT_MISMATCH got=%0d expected=%0d", cmd_done_saturation, proj_expected_saturation[selected_case]);
                        failures = failures + 1;
                    end
                    if (oproj_active) begin
                        oproj_completion_accept_cycle = cycle_count;
                        last_command_cycles =
                            oproj_completion_accept_cycle -
                            oproj_command_accept_cycle;
                    end else begin
                        last_command_cycles = cycle_count - command_start_cycle;
                    end
                    total_success_cycles = total_success_cycles + last_command_cycles;
                    success_runs = success_runs + 1;
                    if (last_command_cycles > max_command_cycles) begin
                        max_command_cycles = last_command_cycles;
                    end
                    expected_beats_per_row = (current_proj_n + 15) / 16;
                    expected_count = current_proj_m * expected_beats_per_row;
                    if (observed_count != expected_count) begin
                        $display("SHELL_QPROJ_WRITE_COUNT_MISMATCH got=%0d expected=%0d", observed_count, expected_count);
                        failures = failures + 1;
                    end
                    for (beat = 0; beat < expected_count; beat = beat + 1) begin
                        expected_flat = selected_case*PROJ_MAX_ROWS*PROJ_MAX_OUTPUT_BEATS +
                                        (beat / expected_beats_per_row)*PROJ_MAX_OUTPUT_BEATS +
                                        (beat % expected_beats_per_row);
                        expected_addr = current_proj_dst_addr +
                                        (beat / expected_beats_per_row)*current_proj_n +
                                        (beat % expected_beats_per_row)*16;
                        if (observed_write_addr[beat] !== expected_addr) begin
                            $display("SHELL_QPROJ_ADDR_MISMATCH case=%0d beat=%0d got=%016x expected=%016x",
                                     selected_case, beat, observed_write_addr[beat],
                                     expected_addr);
                            failures = failures + 1;
                        end
                        if (observed_output[beat] !== proj_expected_beats[expected_flat]) begin
                            $display("SHELL_QPROJ_OUTPUT_MISMATCH case=%0d beat=%0d got=%032x expected=%032x",
                                     selected_case, beat, observed_output[beat],
                                     proj_expected_beats[expected_flat]);
                            failures = failures + 1;
                        end
                    end
                end
            end
            @(negedge clk);
            cmd_done_ready = 1'b0;
        end
    endtask

    task run_oproj_cases;
        integer beat;
        reg [63:0] vector_signature;
        begin
            for (case_index = 0; case_index < PROJ_CASE_MLP_GATE; case_index = case_index + 1) begin
                send_oproj_cmd(case_index, 0, 16'h3f00 + case_index);
                wait_qproj_done_and_compare(
                    case_index,
                    proj_expected_saturation[case_index],
                    16'h3f00 + case_index
                );
                vector_signature = 64'd0;
                for (beat = 0; beat < observed_count; beat = beat + 1) begin
                    oproj_signature = fold_oproj_word(
                        oproj_signature,
                        observed_output[beat],
                        case_index,
                        beat
                    );
                    vector_signature = fold_oproj_word(
                        vector_signature,
                        observed_output[beat],
                        case_index,
                        beat
                    );
                end
                $display("ACE2_SHELL_VECTOR_RESULT simulator=icarus opcode=01 vector=%0d writes=%0d signature=%016x command_accept_cycle=%0d completion_accept_cycle=%0d cycles=%0d",
                         case_index, observed_count, vector_signature,
                         oproj_command_accept_cycle,
                         oproj_completion_accept_cycle,
                         last_command_cycles);
                oproj_writes_seen = oproj_writes_seen + observed_count;
                oproj_total_cycles = oproj_total_cycles + last_command_cycles;
                if (last_command_cycles > oproj_max_cycles) begin
                    oproj_max_cycles = last_command_cycles;
                end
                oproj_cases_seen = oproj_cases_seen + 1;
                if (proj_expected_saturation[case_index]) begin
                    csr_write64(ACE2_CSR_ERROR_STATUS, 64'hffff_ffff_ffff_ffff);
                end
            end
        end
    endtask

    task report_oproj_result;
        begin
            $display("ACE2_SHELL_OPCODE_RESULT opcode=01 vectors=%0d writes=%0d signature=%016x total_cycles=%0d max_cycles=%0d",
                     oproj_cases_seen, oproj_writes_seen, oproj_signature,
                     oproj_total_cycles, oproj_max_cycles);
        end
    endtask

    task run_mlp_projection_cases;
        begin
            send_mlp_proj_cmd(PROJ_CASE_MLP_GATE, 16'h5400);
            wait_qproj_done_and_compare(
                PROJ_CASE_MLP_GATE,
                proj_expected_saturation[PROJ_CASE_MLP_GATE],
                16'h5400
            );
            mlp_gate_proj_cases_seen = mlp_gate_proj_cases_seen + 1;
            if (proj_expected_saturation[PROJ_CASE_MLP_GATE]) begin
                csr_write64(ACE2_CSR_ERROR_STATUS, 64'hffff_ffff_ffff_ffff);
            end

        end
    endtask

    task run_mlp_up_projection_case;
        begin
            send_mlp_up_proj_cmd(PROJ_CASE_MLP_GATE, 16'h5401);
            wait_qproj_done_and_compare(
                PROJ_CASE_MLP_GATE,
                proj_expected_saturation[PROJ_CASE_MLP_GATE],
                16'h5401
            );
            mlp_up_proj_cases_seen = mlp_up_proj_cases_seen + 1;
            if (proj_expected_saturation[PROJ_CASE_MLP_GATE]) begin
                csr_write64(ACE2_CSR_ERROR_STATUS, 64'hffff_ffff_ffff_ffff);
            end
            $display("ACE2_SHELL_MLP_UP_TB_PASS cases=%0d writes=%0d cycles=%0d",
                     mlp_up_proj_cases_seen, observed_count, last_command_cycles);
        end
    endtask

    task run_mlp_down_projection_case;
        begin
            send_mlp_down_proj_cmd(PROJ_CASE_MLP_DOWN, 16'h5402);
            wait_qproj_done_and_compare(
                PROJ_CASE_MLP_DOWN,
                proj_expected_saturation[PROJ_CASE_MLP_DOWN],
                16'h5402
            );
            mlp_down_proj_cases_seen = mlp_down_proj_cases_seen + 1;
            if (proj_expected_saturation[PROJ_CASE_MLP_DOWN]) begin
                csr_write64(ACE2_CSR_ERROR_STATUS, 64'hffff_ffff_ffff_ffff);
            end
            $display("ACE2_SHELL_MLP_DOWN_TB_PASS cases=%0d writes=%0d cycles=%0d",
                     mlp_down_proj_cases_seen, observed_count, last_command_cycles);
        end
    endtask

    task send_silu_gate_cmd_layer;
        input integer selected_case;
        input integer selected_layer;
        input [15:0] n_value;
        input [15:0] tag;
        integer guard;
        integer idx;
        begin
            current_silu_case = selected_case;
            observed_count = 0;
            silu_gate_read_count = 0;
            silu_up_read_count = 0;
            silu_write_req_count = 0;
            silu_max_gate_read_addr = 64'd0;
            silu_max_up_read_addr = 64'd0;
            pending_read = 0;
            mem_rvalid = 1'b0;
            for (idx = 0; idx < OBSERVED_MAX_WRITES; idx = idx + 1) begin
                observed_output[idx] = 128'd0;
                observed_write_addr[idx] = 64'd0;
                observed_write_strb[idx] = 16'd0;
            end
            @(negedge clk);
            cmd_opcode = ACE2_OPCODE_SILU_GATE;
            cmd_flags = 8'h09;
            cmd_layer_id = selected_layer;
            cmd_m = 16'd1;
            cmd_n = n_value;
            cmd_k = 16'd0;
            cmd_sequence_position = 16'd0;
            cmd_completion_tag = tag;
            cmd_src0_addr = SILU_GATE_BASE;
            cmd_src1_addr = SILU_UP_BASE;
            cmd_dst_addr = SILU_OUT_BASE;
            cmd_scale_addr = SILU_META_BASE;
            cmd_scratch_addr = 64'd0;
            cmd_valid = 1'b1;
            guard = 0;
            while (!cmd_ready && (guard < 256)) begin
                guard = guard + 1;
                @(posedge clk);
            end
            if (!cmd_ready) begin
                $display("SILU_CMD_NOT_READY");
                failures = failures + 1;
            end
            @(posedge clk);
            command_start_cycle = cycle_count;
            @(negedge clk);
            cmd_valid = 1'b0;
        end
    endtask

    task send_silu_gate_cmd;
        input integer selected_case;
        input [15:0] n_value;
        input [15:0] tag;
        begin
            send_silu_gate_cmd_layer(selected_case, 0, n_value, tag);
        end
    endtask

    task wait_silu_gate_done_and_compare;
        input integer selected_case;
        input integer expect_descriptor_error;
        input [15:0] expected_tag;
        integer guard;
        integer beat;
        integer expected_count;
        reg [15:0] expected_last_strb;
        reg [63:0] expected_gate_high_water;
        reg [63:0] expected_up_high_water;
        begin
            cmd_done_ready = 1'b1;
            guard = 0;
            while (!cmd_done_valid && (guard < 1000000)) begin
                guard = guard + 1;
                @(posedge clk);
            end
            if (!cmd_done_valid) begin
                $display("SILU_TIMEOUT case=%0d observed=%0d", selected_case, observed_count);
                failures = failures + 1;
            end else if (expect_descriptor_error) begin
                if (!cmd_done_error) begin
                    $display("SILU_DESCRIPTOR_ERROR_NOT_REPORTED case=%0d", selected_case);
                    failures = failures + 1;
                end
            end else begin
                expected_count = (silu_case_length[selected_case] + 15) / 16;
                expected_gate_high_water = SILU_GATE_BASE + (expected_count - 1)*16;
                expected_up_high_water = SILU_UP_BASE + (expected_count - 1)*16;
                if (cmd_done_tag !== expected_tag ||
                    cmd_done_error !== silu_expected_saturation[selected_case] ||
                    cmd_done_saturation !== silu_expected_saturation[selected_case] ||
                    observed_count != expected_count) begin
                    $display("SILU_STATUS_MISMATCH case=%0d tag=%04x error=%0d sat=%0d writes=%0d expected_writes=%0d",
                             selected_case, cmd_done_tag, cmd_done_error,
                             cmd_done_saturation, observed_count, expected_count);
                    failures = failures + 1;
                end
                if (silu_gate_read_count != expected_count ||
                    silu_up_read_count != expected_count ||
                    silu_write_req_count != expected_count ||
                    silu_max_gate_read_addr !== expected_gate_high_water ||
                    silu_max_up_read_addr !== expected_up_high_water) begin
                    $display("SILU_TRAFFIC_MISMATCH case=%0d gate_reads=%0d up_reads=%0d write_reqs=%0d gate_high=%016x expected_gate_high=%016x up_high=%016x expected_up_high=%016x",
                             selected_case, silu_gate_read_count,
                             silu_up_read_count, silu_write_req_count,
                             silu_max_gate_read_addr, expected_gate_high_water,
                             silu_max_up_read_addr, expected_up_high_water);
                    failures = failures + 1;
                end
                if (silu_max_up_read_addr >= SILU_OUT_BASE) begin
                    $display("SILU_SOURCE_DEST_OVERLAP case=%0d up_high=%016x dst=%016x",
                             selected_case, silu_max_up_read_addr, SILU_OUT_BASE);
                    failures = failures + 1;
                end
                for (beat = 0; beat < expected_count; beat = beat + 1) begin
                    if (observed_write_addr[beat] !== SILU_OUT_BASE + beat*16 ||
                        observed_output[beat] !==
                        silu_expected_beats[selected_case*SILU_MAX_OUTPUT_BEATS + beat]) begin
                        $display("SILU_OUTPUT_MISMATCH case=%0d beat=%0d addr=%016x",
                                 selected_case, beat, observed_write_addr[beat]);
                        failures = failures + 1;
                    end
                end
                expected_last_strb =
                    (silu_case_length[selected_case][3:0] == 4'd0) ?
                    16'hffff :
                    (16'hffff >> (16 - silu_case_length[selected_case][3:0]));
                if (observed_write_strb[expected_count-1] !== expected_last_strb) begin
                    $display("SILU_STROBE_MISMATCH case=%0d got=%04x expected=%04x",
                             selected_case, observed_write_strb[expected_count-1],
                             expected_last_strb);
                    failures = failures + 1;
                end
                if (selected_case == SILU_FULL_CASE_INDEX) begin
                    if (observed_output[0][7:0] !==
                            silu_expected_beats[selected_case*SILU_MAX_OUTPUT_BEATS][7:0] ||
                        observed_output[152][7:0] !==
                            silu_expected_beats[selected_case*SILU_MAX_OUTPUT_BEATS + 152][7:0] ||
                        observed_output[303][127:120] !==
                            silu_expected_beats[selected_case*SILU_MAX_OUTPUT_BEATS + 303][127:120]) begin
                        $display("SILU_FIRST_MIDDLE_LAST_LANE_MISMATCH");
                        failures = failures + 1;
                    end
                    if (silu_max_gate_read_addr !== SILU_GATE_BASE + 64'd4848 ||
                        silu_max_up_read_addr !== SILU_UP_BASE + 64'd4848 ||
                        silu_gate_read_count != 304 || silu_up_read_count != 304 ||
                        silu_write_req_count != 304 || observed_count != 304) begin
                        $display("SILU_FULL_PACKED_CONTRACT_MISMATCH gate_reads=%0d up_reads=%0d writes=%0d observed=%0d gate_high=%016x up_high=%016x",
                                 silu_gate_read_count, silu_up_read_count,
                                 silu_write_req_count, observed_count,
                                 silu_max_gate_read_addr, silu_max_up_read_addr);
                        failures = failures + 1;
                    end
                end
                silu_gate_cases_seen = silu_gate_cases_seen + 1;
                last_command_cycles = cycle_count - command_start_cycle;
                total_success_cycles = total_success_cycles + last_command_cycles;
                success_runs = success_runs + 1;
                if (last_command_cycles > max_command_cycles) begin
                    max_command_cycles = last_command_cycles;
                end
            end
            @(negedge clk);
            cmd_done_ready = 1'b0;
        end
    endtask

    task run_silu_gate_cases;
        reg [63:0] error_status_value;
        reg sticky_numeric_expected;
        reg [5:0] executed_boundary_coverage;
        integer sticky_clean_checks;
        begin
            csr_write64(ACE2_CSR_ERROR_STATUS, 64'hffff_ffff_ffff_ffff);
            csr_read64(ACE2_CSR_ERROR_STATUS, error_status_value);
            if (error_status_value[ACE2_ERR_NUMERIC] !== 1'b0) begin
                $display("SILU_NUMERIC_ERROR_CLEAR_FAILED before=%016x",
                         error_status_value);
                failures = failures + 1;
            end
            sticky_numeric_expected = 1'b0;
            sticky_clean_checks = 0;
            executed_boundary_coverage = 6'd0;
            for (case_index = 0; case_index < SILU_CASE_COUNT; case_index = case_index + 1) begin
                if (case_index == 0) begin
                    force_mem_req_stall = 1'b1;
                end
                send_silu_gate_cmd(case_index, silu_case_length[case_index],
                                   16'h5500 + case_index);
                if (case_index == 0) begin
                    repeat (7) @(posedge clk);
                    force_mem_req_stall = 1'b0;
                end
                wait_silu_gate_done_and_compare(case_index, 0, 16'h5500 + case_index);
                executed_boundary_coverage =
                    executed_boundary_coverage |
                    silu_case_boundary_coverage[case_index];
                sticky_numeric_expected =
                    sticky_numeric_expected |
                    silu_expected_saturation[case_index];
                csr_read64(ACE2_CSR_ERROR_STATUS, error_status_value);
                if (error_status_value[ACE2_ERR_NUMERIC] !==
                    sticky_numeric_expected) begin
                    $display("SILU_NUMERIC_ERROR_STICKY_MISMATCH case=%0d got=%0d expected=%0d",
                             case_index,
                             error_status_value[ACE2_ERR_NUMERIC],
                             sticky_numeric_expected);
                    failures = failures + 1;
                end
                if (sticky_numeric_expected &&
                    !silu_expected_saturation[case_index]) begin
                    sticky_clean_checks = sticky_clean_checks + 1;
                end
            end
            if (executed_boundary_coverage !== SILU_REQUIRED_BOUNDARY_COVERAGE) begin
                $display("SILU_SHELL_BOUNDARY_COVERAGE_MISMATCH got=%02x expected=%02x",
                         executed_boundary_coverage,
                         SILU_REQUIRED_BOUNDARY_COVERAGE);
                failures = failures + 1;
            end
            if (sticky_clean_checks == 0) begin
                $display("SILU_NUMERIC_ERROR_STICKY_CLEAN_CASE_MISSING");
                failures = failures + 1;
            end
            csr_write64(ACE2_CSR_ERROR_STATUS, 64'hffff_ffff_ffff_ffff);
            csr_read64(ACE2_CSR_ERROR_STATUS, error_status_value);
            if (error_status_value[ACE2_ERR_NUMERIC] !== 1'b0) begin
                $display("SILU_NUMERIC_ERROR_FINAL_CLEAR_FAILED after=%016x",
                         error_status_value);
                failures = failures + 1;
            end
            $display("ACE2_SHELL_SILU_GATE_TB_PASS cases=%0d writes=%0d cycles=%0d boundary_mask=%02x sticky_checks=%0d packed_reads_per_source=%0d high_water=+4848",
                     silu_gate_cases_seen, observed_count, last_command_cycles,
                     executed_boundary_coverage, sticky_clean_checks,
                     silu_gate_read_count);
        end
    endtask

    task send_rope_cmd_full_flags;
        input integer selected_case;
        input integer selected_layer;
        input [15:0] n_value;
        input [15:0] sequence_position;
        input [63:0] src0_addr;
        input [63:0] src1_addr;
        input [63:0] dst_addr;
        input [63:0] scale_addr;
        input [15:0] tag;
        input [7:0] basis_rotation_half_degrees;
        integer guard;
        integer idx;
        begin
            current_rope_case = selected_case;
            current_rope_beats = (n_value + 15) / 16;
            current_rope_table_stride = current_rope_beats * 64;
            observed_count = 0;
            pending_read = 0;
            mem_rvalid = 1'b0;
            for (idx = 0; idx < OBSERVED_MAX_WRITES; idx = idx + 1) begin
                observed_output[idx] = 128'd0;
                observed_write_addr[idx] = 64'd0;
            end
            @(negedge clk);
            cmd_opcode = ACE2_OPCODE_ROPE;
            cmd_flags = basis_rotation_half_degrees;
            cmd_layer_id = selected_layer;
            cmd_m = 16'd1;
            cmd_n = n_value;
            cmd_k = 16'd0;
            cmd_sequence_position = sequence_position;
            cmd_completion_tag = tag;
            cmd_src0_addr = src0_addr;
            cmd_src1_addr = src1_addr;
            cmd_dst_addr = dst_addr;
            cmd_scale_addr = scale_addr;
            cmd_scratch_addr = 64'd0;
            cmd_valid = 1'b1;
            guard = 0;
            while (!cmd_ready && (guard < 256)) begin
                guard = guard + 1;
                @(posedge clk);
            end
            if (!cmd_ready) begin
                $display("SHELL_ROPE_CMD_NOT_READY");
                failures = failures + 1;
            end
            @(posedge clk);
            command_start_cycle = cycle_count;
            @(negedge clk);
            cmd_valid = 1'b0;
        end
    endtask

    task send_rope_cmd_full;
        input integer selected_case;
        input integer selected_layer;
        input [15:0] n_value;
        input [15:0] sequence_position;
        input [63:0] src0_addr;
        input [63:0] src1_addr;
        input [63:0] dst_addr;
        input [63:0] scale_addr;
        input [15:0] tag;
        begin
            send_rope_cmd_full_flags(
                selected_case, selected_layer, n_value, sequence_position,
                src0_addr, src1_addr, dst_addr, scale_addr, tag,
                ROPE_BASIS_ROTATION_HALF_DEGREES
            );
        end
    endtask

    task send_rope_cmd;
        input integer selected_case;
        input integer selected_layer;
        input [15:0] tag;
        begin
            send_rope_cmd_full(selected_case, selected_layer, 16'd896, rope_sequence_position[selected_case],
                               ROPE_ACT_BASE, ROPE_TABLE_BASE, ROPE_OUT_BASE, ROPE_SCALE_BASE, tag);
        end
    endtask

    task send_rope_k_cmd;
        input integer selected_case;
        input integer selected_layer;
        input [15:0] tag;
        begin
            send_rope_cmd_full(selected_case, selected_layer, 16'd128, rope_sequence_position[selected_case],
                               ROPE_ACT_BASE, ROPE_TABLE_BASE, ROPE_OUT_BASE, ROPE_SCALE_BASE, tag);
        end
    endtask

    task wait_rope_done_and_compare;
        input integer selected_case;
        input integer expect_error;
        input [15:0] expected_tag;
        integer guard;
        integer beat;
        begin
            guard = 0;
            while (!cmd_done_valid && (guard < 1000000)) begin
                guard = guard + 1;
                @(posedge clk);
            end
            if (!cmd_done_valid) begin
                $display("SHELL_ROPE_TIMEOUT case=%0d observed=%0d", selected_case, observed_count);
                failures = failures + 1;
            end else begin
                if (cmd_done_tag !== expected_tag) begin
                    $display("SHELL_ROPE_TAG_MISMATCH got=%0d expected=%0d", cmd_done_tag, expected_tag);
                    failures = failures + 1;
                end
                if (cmd_done_error !== expect_error[0]) begin
                    $display("SHELL_ROPE_ERROR_MISMATCH got=%0d expected=%0d", cmd_done_error, expect_error);
                    failures = failures + 1;
                end
                if (!expect_error || rope_expected_saturation[selected_case]) begin
                    if (cmd_done_saturation !== rope_expected_saturation[selected_case]) begin
                        $display("SHELL_ROPE_SAT_MISMATCH got=%0d expected=%0d", cmd_done_saturation, rope_expected_saturation[selected_case]);
                        failures = failures + 1;
                    end
                    last_command_cycles = cycle_count - command_start_cycle;
                    total_success_cycles = total_success_cycles + last_command_cycles;
                    success_runs = success_runs + 1;
                    rope_cases_seen = rope_cases_seen + 1;
                    if (current_rope_beats == 8) begin
                        rope_k_cases_seen = rope_k_cases_seen + 1;
                    end else begin
                        rope_q_cases_seen = rope_q_cases_seen + 1;
                    end
                    if (last_command_cycles > max_command_cycles) begin
                        max_command_cycles = last_command_cycles;
                    end
                    if (observed_count != current_rope_beats) begin
                        $display("SHELL_ROPE_WRITE_COUNT_MISMATCH got=%0d expected=%0d", observed_count, current_rope_beats);
                        failures = failures + 1;
                    end
                    for (beat = 0; beat < current_rope_beats; beat = beat + 1) begin
                        if (observed_output[beat] !== rope_expected_beats[selected_case*ROPE_BEATS + beat]) begin
                            $display("SHELL_ROPE_OUTPUT_MISMATCH case=%0d beat=%0d got=%032x expected=%032x",
                                     selected_case, beat, observed_output[beat],
                                     rope_expected_beats[selected_case*ROPE_BEATS + beat]);
                            failures = failures + 1;
                        end
                    end
                end
            end
            @(negedge clk);
            cmd_done_ready = 1'b1;
            @(posedge clk);
            @(negedge clk);
            cmd_done_ready = 1'b0;
        end
    endtask

    task send_kv_write_cmd_full;
        input integer selected_case;
        input integer selected_layer;
        input [15:0] n_value;
        input [15:0] sequence_position;
        input [63:0] src0_addr;
        input [63:0] src1_addr;
        input [63:0] dst_addr;
        input [63:0] scale_addr;
        input [15:0] tag;
        integer guard;
        integer idx;
        begin
            current_kv_case = selected_case;
            current_kv_sequence = sequence_position;
            observed_count = 0;
            pending_read = 0;
            mem_rvalid = 1'b0;
            for (idx = 0; idx < OBSERVED_MAX_WRITES; idx = idx + 1) begin
                observed_output[idx] = 128'd0;
                observed_write_addr[idx] = 64'd0;
            end
            @(negedge clk);
            cmd_opcode = ACE2_OPCODE_KV_WRITE;
            cmd_flags = 8'h09;
            cmd_layer_id = selected_layer;
            cmd_m = 16'd1;
            cmd_n = n_value;
            cmd_k = 16'd0;
            cmd_sequence_position = sequence_position;
            cmd_completion_tag = tag;
            cmd_src0_addr = src0_addr;
            cmd_src1_addr = src1_addr;
            cmd_dst_addr = 64'd0;
            cmd_scale_addr = scale_addr;
            cmd_scratch_addr = dst_addr;
            cmd_valid = 1'b1;
            guard = 0;
            while (!cmd_ready && (guard < 256)) begin
                guard = guard + 1;
                @(posedge clk);
            end
            if (!cmd_ready) begin
                $display("SHELL_KV_WRITE_CMD_NOT_READY");
                failures = failures + 1;
            end
            @(posedge clk);
            command_start_cycle = cycle_count;
            @(negedge clk);
            cmd_valid = 1'b0;
        end
    endtask

    task send_kv_write_cmd;
        input integer selected_case;
        input integer selected_layer;
        input [15:0] sequence_position;
        input [15:0] tag;
        begin
            send_kv_write_cmd_full(selected_case, selected_layer, 16'd128, sequence_position,
                                   KV_K_BASE, KV_V_BASE, KV_CACHE_BASE, KV_META_BASE, tag);
        end
    endtask

    task wait_kv_write_done_and_compare;
        input integer selected_case;
        input integer expect_error;
        input [15:0] expected_tag;
        integer guard;
        integer beat;
        reg [63:0] expected_addr;
        reg [127:0] expected_data;
        begin
            guard = 0;
            while (!cmd_done_valid && (guard < 1000000)) begin
                guard = guard + 1;
                @(posedge clk);
            end
            if (!cmd_done_valid) begin
                $display("SHELL_KV_WRITE_TIMEOUT case=%0d observed=%0d", selected_case, observed_count);
                failures = failures + 1;
            end else begin
                if (cmd_done_tag !== expected_tag) begin
                    $display("SHELL_KV_WRITE_TAG_MISMATCH got=%0d expected=%0d", cmd_done_tag, expected_tag);
                    failures = failures + 1;
                end
                if (cmd_done_error !== expect_error[0]) begin
                    $display("SHELL_KV_WRITE_ERROR_MISMATCH got=%0d expected=%0d", cmd_done_error, expect_error);
                    failures = failures + 1;
                end
                if (!expect_error) begin
                    last_command_cycles = cycle_count - command_start_cycle;
                    total_success_cycles = total_success_cycles + last_command_cycles;
                    success_runs = success_runs + 1;
                    kv_write_cases_seen = kv_write_cases_seen + 1;
                    if (last_command_cycles > max_command_cycles) begin
                        max_command_cycles = last_command_cycles;
                    end
                    if (observed_count != KV_TOTAL_BEATS) begin
                        $display("SHELL_KV_WRITE_COUNT_MISMATCH got=%0d expected=%0d", observed_count, KV_TOTAL_BEATS);
                        failures = failures + 1;
                    end
                    for (beat = 0; beat < KV_TOTAL_BEATS; beat = beat + 1) begin
                        if (beat < KV_DATA_BEATS) begin
                            expected_addr = KV_CACHE_BASE + current_kv_sequence*KV_BYTES_PER_TOKEN + beat*16;
                            expected_data = make_kv_word(selected_case, 0, beat);
                        end else if (beat < (KV_DATA_BEATS * 2)) begin
                            expected_addr = KV_CACHE_BASE + current_kv_sequence*KV_BYTES_PER_TOKEN + 128 + (beat - KV_DATA_BEATS)*16;
                            expected_data = make_kv_word(selected_case, 1, beat - KV_DATA_BEATS);
                        end else begin
                            expected_addr = KV_CACHE_BASE + current_kv_sequence*KV_BYTES_PER_TOKEN + 256;
                            expected_data = make_kv_word(selected_case, 2, 0);
                        end
                        if (observed_write_addr[beat] !== expected_addr) begin
                            $display("SHELL_KV_WRITE_ADDR_MISMATCH beat=%0d got=%016x expected=%016x",
                                     beat, observed_write_addr[beat], expected_addr);
                            failures = failures + 1;
                        end
                        if (observed_output[beat] !== expected_data) begin
                            $display("SHELL_KV_WRITE_DATA_MISMATCH beat=%0d got=%032x expected=%032x",
                                     beat, observed_output[beat], expected_data);
                            failures = failures + 1;
                        end
                    end
                end
            end
            @(negedge clk);
            cmd_done_ready = 1'b1;
            @(posedge clk);
            @(negedge clk);
            cmd_done_ready = 1'b0;
        end
    endtask

    task send_attn_score_cmd_full;
        input integer selected_case;
        input integer selected_layer;
        input [15:0] n_value;
        input [15:0] k_value;
        input [63:0] src0_addr;
        input [63:0] src1_addr;
        input [63:0] dst_addr;
        input [63:0] scale_addr;
        input [15:0] tag;
        integer guard;
        integer idx;
        begin
            current_attn_score_case = selected_case;
            observed_count = 0;
            pending_read = 0;
            mem_rvalid = 1'b0;
            for (idx = 0; idx < OBSERVED_MAX_WRITES; idx = idx + 1) begin
                observed_output[idx] = 128'd0;
                observed_write_addr[idx] = 64'd0;
            end
            @(negedge clk);
            cmd_opcode = ACE2_OPCODE_ATTN_SCORE;
            cmd_flags = 8'h09;
            cmd_layer_id = selected_layer;
            cmd_m = 16'd1;
            cmd_n = n_value;
            cmd_k = k_value;
            cmd_sequence_position = 16'd0;
            cmd_completion_tag = tag;
            cmd_src0_addr = src0_addr;
            cmd_src1_addr = src1_addr;
            cmd_dst_addr = dst_addr;
            cmd_scale_addr = scale_addr;
            cmd_scratch_addr = 64'd0;
            cmd_valid = 1'b1;
            guard = 0;
            while (!cmd_ready && (guard < 256)) begin
                guard = guard + 1;
                @(posedge clk);
            end
            if (!cmd_ready) begin
                $display("SHELL_ATTN_SCORE_CMD_NOT_READY");
                failures = failures + 1;
            end
            @(posedge clk);
            command_start_cycle = cycle_count;
            @(negedge clk);
            cmd_valid = 1'b0;
        end
    endtask

    task send_attn_score_cmd;
        input integer selected_case;
        input integer selected_layer;
        input [15:0] tag;
        begin
            send_attn_score_cmd_full(selected_case, selected_layer,
                                     attn_score_context_count[selected_case], 16'd64,
                                     ATTN_Q_BASE, ATTN_K_BASE, ATTN_SCORE_OUT_BASE,
                                     ATTN_META_BASE, tag);
        end
    endtask

    task wait_attn_score_done_and_compare;
        input integer selected_case;
        input integer expect_error;
        input [15:0] expected_tag;
        integer guard;
        begin
            guard = 0;
            while (!cmd_done_valid && (guard < 1000000)) begin
                guard = guard + 1;
                @(posedge clk);
            end
            if (!cmd_done_valid) begin
                $display("SHELL_ATTN_SCORE_TIMEOUT case=%0d observed=%0d", selected_case, observed_count);
                failures = failures + 1;
            end else begin
                if (cmd_done_tag !== expected_tag) begin
                    $display("SHELL_ATTN_SCORE_TAG_MISMATCH got=%0d expected=%0d", cmd_done_tag, expected_tag);
                    failures = failures + 1;
                end
                if (cmd_done_error !== expect_error[0]) begin
                    $display("SHELL_ATTN_SCORE_ERROR_MISMATCH got=%0d expected=%0d", cmd_done_error, expect_error);
                    failures = failures + 1;
                end
                if (!expect_error || attn_score_expected_saturation[selected_case]) begin
                    if (cmd_done_saturation !== attn_score_expected_saturation[selected_case]) begin
                        $display("SHELL_ATTN_SCORE_SAT_MISMATCH got=%0d expected=%0d",
                                 cmd_done_saturation, attn_score_expected_saturation[selected_case]);
                        failures = failures + 1;
                    end
                    last_command_cycles = cycle_count - command_start_cycle;
                    total_success_cycles = total_success_cycles + last_command_cycles;
                    success_runs = success_runs + 1;
                    attn_score_cases_seen = attn_score_cases_seen + 1;
                    if (last_command_cycles > max_command_cycles) begin
                        max_command_cycles = last_command_cycles;
                    end
                    if (observed_count != 1) begin
                        $display("SHELL_ATTN_SCORE_WRITE_COUNT_MISMATCH got=%0d expected=1", observed_count);
                        failures = failures + 1;
                    end
                    if (observed_write_addr[0] !== ATTN_SCORE_OUT_BASE) begin
                        $display("SHELL_ATTN_SCORE_ADDR_MISMATCH got=%016x expected=%016x",
                                 observed_write_addr[0], ATTN_SCORE_OUT_BASE);
                        failures = failures + 1;
                    end
                    if (observed_output[0] !== attn_score_expected_word[selected_case]) begin
                        $display("SHELL_ATTN_SCORE_OUTPUT_MISMATCH case=%0d got=%032x expected=%032x",
                                 selected_case, observed_output[0], attn_score_expected_word[selected_case]);
                        failures = failures + 1;
                    end
                end
            end
            @(negedge clk);
            cmd_done_ready = 1'b1;
            @(posedge clk);
            @(negedge clk);
            cmd_done_ready = 1'b0;
        end
    endtask

    task run_attn_score_metadata_error_cases;
        integer metadata_mode;
        reg [63:0] status_value;
        begin
            for (metadata_mode = 1; metadata_mode <= 2;
                 metadata_mode = metadata_mode + 1) begin
                current_attn_metadata_mode = metadata_mode;
                send_attn_score_cmd(0, 0, 16'h3af0 + metadata_mode);
                wait_attn_score_done_and_compare(
                    0, 1, 16'h3af0 + metadata_mode
                );
                if (observed_count != 0) begin
                    $display(
                        "ATTN_SCORE_INVALID_METADATA_WROTE_OUTPUT mode=%0d writes=%0d",
                        metadata_mode, observed_count
                    );
                    failures = failures + 1;
                end
                csr_read64(ACE2_CSR_ERROR_STATUS, status_value);
                if (!status_value[ACE2_ERR_NUMERIC]) begin
                    $display(
                        "ATTN_SCORE_INVALID_METADATA_NOT_NUMERIC mode=%0d status=%016x",
                        metadata_mode, status_value
                    );
                    failures = failures + 1;
                end
                csr_write64(
                    ACE2_CSR_ERROR_STATUS, 64'hffff_ffff_ffff_ffff
                );
                attn_score_metadata_error_cases =
                    attn_score_metadata_error_cases + 1;
            end
            current_attn_metadata_mode = 0;
        end
    endtask

    task send_softmax_cmd_full;
        input integer selected_case;
        input integer selected_layer;
        input [15:0] n_value;
        input [15:0] k_value;
        input [63:0] src0_addr;
        input [63:0] dst_addr;
        input [15:0] tag;
        integer guard;
        integer idx;
        begin
            current_softmax_case = selected_case;
            observed_count = 0;
            pending_read = 0;
            mem_rvalid = 1'b0;
            for (idx = 0; idx < OBSERVED_MAX_WRITES; idx = idx + 1) begin
                observed_output[idx] = 128'd0;
                observed_write_addr[idx] = 64'd0;
            end
            @(negedge clk);
            cmd_opcode = ACE2_OPCODE_SOFTMAX;
            cmd_flags = 8'h09;
            cmd_layer_id = selected_layer;
            cmd_m = 16'd1;
            cmd_n = n_value;
            cmd_k = k_value;
            cmd_sequence_position = 16'd0;
            cmd_completion_tag = tag;
            cmd_src0_addr = src0_addr;
            cmd_src1_addr = 64'd0;
            cmd_dst_addr = dst_addr;
            cmd_scale_addr = 64'd0;
            cmd_scratch_addr = 64'd0;
            cmd_valid = 1'b1;
            guard = 0;
            while (!cmd_ready && (guard < 256)) begin
                guard = guard + 1;
                @(posedge clk);
            end
            if (!cmd_ready) begin
                $display("SHELL_SOFTMAX_CMD_NOT_READY");
                failures = failures + 1;
            end
            @(posedge clk);
            command_start_cycle = cycle_count;
            @(negedge clk);
            cmd_valid = 1'b0;
        end
    endtask

    task send_softmax_cmd;
        input integer selected_case;
        input integer selected_layer;
        input [15:0] tag;
        begin
            send_softmax_cmd_full(selected_case, selected_layer,
                                  softmax_context_count[selected_case], 16'd0,
                                  SOFTMAX_SCORE_BASE, SOFTMAX_OUT_BASE, tag);
        end
    endtask

    task wait_softmax_done_and_compare;
        input integer selected_case;
        input integer expect_error;
        input [15:0] expected_tag;
        integer guard;
        begin
            guard = 0;
            while (!cmd_done_valid && (guard < 1000000)) begin
                guard = guard + 1;
                @(posedge clk);
            end
            if (!cmd_done_valid) begin
                $display("SHELL_SOFTMAX_TIMEOUT case=%0d observed=%0d", selected_case, observed_count);
                failures = failures + 1;
            end else begin
                if (cmd_done_tag !== expected_tag) begin
                    $display("SHELL_SOFTMAX_TAG_MISMATCH got=%0d expected=%0d", cmd_done_tag, expected_tag);
                    failures = failures + 1;
                end
                if (cmd_done_error !== expect_error[0]) begin
                    $display("SHELL_SOFTMAX_ERROR_MISMATCH got=%0d expected=%0d", cmd_done_error, expect_error);
                    failures = failures + 1;
                end
                if (!expect_error) begin
                    if (cmd_done_saturation !== 1'b0) begin
                        $display("SHELL_SOFTMAX_UNEXPECTED_SAT");
                        failures = failures + 1;
                    end
                    last_command_cycles = cycle_count - command_start_cycle;
                    total_success_cycles = total_success_cycles + last_command_cycles;
                    success_runs = success_runs + 1;
                    softmax_cases_seen = softmax_cases_seen + 1;
                    if (last_command_cycles > max_command_cycles) begin
                        max_command_cycles = last_command_cycles;
                    end
                    if (observed_count != 1) begin
                        $display("SHELL_SOFTMAX_WRITE_COUNT_MISMATCH got=%0d expected=1", observed_count);
                        failures = failures + 1;
                    end
                    if (observed_write_addr[0] !== SOFTMAX_OUT_BASE) begin
                        $display("SHELL_SOFTMAX_ADDR_MISMATCH got=%016x expected=%016x",
                                 observed_write_addr[0], SOFTMAX_OUT_BASE);
                        failures = failures + 1;
                    end
                    if (observed_output[0] !== softmax_expected_word[selected_case]) begin
                        $display("SHELL_SOFTMAX_OUTPUT_MISMATCH case=%0d got=%032x expected=%032x",
                                 selected_case, observed_output[0], softmax_expected_word[selected_case]);
                        failures = failures + 1;
                    end
                end
            end
            @(negedge clk);
            cmd_done_ready = 1'b1;
            @(posedge clk);
            @(negedge clk);
            cmd_done_ready = 1'b0;
        end
    endtask

    task send_attn_value_cmd_full;
        input integer selected_case;
        input integer selected_layer;
        input [15:0] n_value;
        input [15:0] k_value;
        input [63:0] src0_addr;
        input [63:0] src1_addr;
        input [63:0] dst_addr;
        input [15:0] tag;
        integer guard;
        integer idx;
        begin
            current_attn_value_case = selected_case;
            observed_count = 0;
            pending_read = 0;
            mem_rvalid = 1'b0;
            for (idx = 0; idx < OBSERVED_MAX_WRITES; idx = idx + 1) begin
                observed_output[idx] = 128'd0;
                observed_write_addr[idx] = 64'd0;
            end
            @(negedge clk);
            cmd_opcode = ACE2_OPCODE_ATTN_VALUE;
            cmd_flags = 8'h09;
            cmd_layer_id = selected_layer;
            cmd_m = 16'd1;
            cmd_n = n_value;
            cmd_k = k_value;
            cmd_sequence_position = 16'd0;
            cmd_completion_tag = tag;
            cmd_src0_addr = src0_addr;
            cmd_src1_addr = src1_addr;
            cmd_dst_addr = dst_addr;
            cmd_scale_addr = 64'd0;
            cmd_scratch_addr = 64'd0;
            cmd_valid = 1'b1;
            guard = 0;
            while (!cmd_ready && (guard < 256)) begin
                guard = guard + 1;
                @(posedge clk);
            end
            if (!cmd_ready) begin
                $display("SHELL_ATTN_VALUE_CMD_NOT_READY");
                failures = failures + 1;
            end
            @(posedge clk);
            command_start_cycle = cycle_count;
            @(negedge clk);
            cmd_valid = 1'b0;
        end
    endtask

    task send_attn_value_cmd;
        input integer selected_case;
        input integer selected_layer;
        input [15:0] tag;
        begin
            send_attn_value_cmd_full(
                selected_case,
                selected_layer,
                attn_value_context_count[selected_case],
                16'd64,
                ATTN_VALUE_PROB_BASE,
                ATTN_VALUE_V_BASE,
                ATTN_VALUE_OUT_BASE,
                tag
            );
        end
    endtask

    task wait_attn_value_done_and_compare;
        input integer selected_case;
        input integer expect_error;
        input [15:0] expected_tag;
        integer guard;
        integer beat;
        begin
            guard = 0;
            while (!cmd_done_valid && (guard < 1000000)) begin
                guard = guard + 1;
                @(posedge clk);
            end
            if (!cmd_done_valid) begin
                $display("SHELL_ATTN_VALUE_TIMEOUT case=%0d observed=%0d",
                         selected_case, observed_count);
                failures = failures + 1;
            end else begin
                if (cmd_done_tag !== expected_tag) begin
                    $display("SHELL_ATTN_VALUE_TAG_MISMATCH got=%0d expected=%0d",
                             cmd_done_tag, expected_tag);
                    failures = failures + 1;
                end
                if (cmd_done_error !== expect_error[0]) begin
                    $display("SHELL_ATTN_VALUE_ERROR_MISMATCH got=%0d expected=%0d",
                             cmd_done_error, expect_error);
                    failures = failures + 1;
                end
                if (!expect_error || attn_value_expected_saturation[selected_case]) begin
                    if (cmd_done_saturation !==
                        attn_value_expected_saturation[selected_case]) begin
                        $display("SHELL_ATTN_VALUE_SAT_MISMATCH got=%0d expected=%0d",
                                 cmd_done_saturation,
                                 attn_value_expected_saturation[selected_case]);
                        failures = failures + 1;
                    end
                    last_command_cycles = cycle_count - command_start_cycle;
                    total_success_cycles = total_success_cycles + last_command_cycles;
                    success_runs = success_runs + 1;
                    attn_value_cases_seen = attn_value_cases_seen + 1;
                    if (last_command_cycles > max_command_cycles) begin
                        max_command_cycles = last_command_cycles;
                    end
                    if (observed_count != ATTN_VALUE_BEATS_PER_VECTOR) begin
                        $display("SHELL_ATTN_VALUE_WRITE_COUNT_MISMATCH got=%0d expected=%0d",
                                 observed_count, ATTN_VALUE_BEATS_PER_VECTOR);
                        failures = failures + 1;
                    end
                    for (beat = 0; beat < ATTN_VALUE_BEATS_PER_VECTOR; beat = beat + 1) begin
                        if (observed_write_addr[beat] !==
                            (ATTN_VALUE_OUT_BASE + beat*16)) begin
                            $display("SHELL_ATTN_VALUE_ADDR_MISMATCH beat=%0d got=%016x expected=%016x",
                                     beat, observed_write_addr[beat],
                                     ATTN_VALUE_OUT_BASE + beat*16);
                            failures = failures + 1;
                        end
                        if (observed_output[beat] !==
                            attn_value_expected_beats[
                                selected_case*ATTN_VALUE_BEATS_PER_VECTOR + beat]) begin
                            $display("SHELL_ATTN_VALUE_OUTPUT_MISMATCH case=%0d beat=%0d got=%032x expected=%032x",
                                     selected_case, beat, observed_output[beat],
                                     attn_value_expected_beats[
                                         selected_case*ATTN_VALUE_BEATS_PER_VECTOR + beat]);
                            failures = failures + 1;
                        end
                    end
                end
            end
            @(negedge clk);
            cmd_done_ready = 1'b1;
            @(posedge clk);
            @(negedge clk);
            cmd_done_ready = 1'b0;
        end
    endtask

    task send_attn_compose_command;
        input [7:0] phase;
        input integer selected_tile;
        input integer selected_count;
        input [15:0] tag;
        integer guard;
        integer idx;
        begin
            current_attn_compose_case = 0;
            observed_count = 0;
            pending_read = 0;
            mem_rvalid = 1'b0;
            for (idx = 0; idx < OBSERVED_MAX_WRITES; idx = idx + 1) begin
                observed_output[idx] = 128'd0;
                observed_write_addr[idx] = 64'd0;
            end
            @(negedge clk);
            cmd_opcode = ACE2_OPCODE_ATTN_COMPOSE;
            cmd_flags = phase;
            cmd_layer_id = 8'd0;
            cmd_m = 16'd1;
            cmd_n = selected_count;
            cmd_k = 16'd64;
            cmd_sequence_position = 16'd0;
            cmd_completion_tag = tag;
            cmd_src0_addr = ATTN_COMPOSE_SCORE_BASE + selected_tile*16;
            cmd_src1_addr =
                ATTN_COMPOSE_VALUE_BASE +
                selected_tile*ATTN_COMPOSE_TILE*64;
            cmd_dst_addr = ATTN_COMPOSE_OUT_BASE;
            cmd_scale_addr = 64'd0;
            cmd_scratch_addr = 64'd0;
            cmd_valid = 1'b1;
            guard = 0;
            while (!cmd_ready && guard < 256) begin
                guard = guard + 1;
                @(posedge clk);
            end
            if (!cmd_ready) begin
                $display("SHELL_ATTN_COMPOSE_CMD_NOT_READY phase=%0d", phase);
                failures = failures + 1;
            end
            @(posedge clk);
            command_start_cycle = cycle_count;
            @(negedge clk);
            cmd_valid = 1'b0;
        end
    endtask

    task run_attn_compose_command;
        input [7:0] phase;
        input integer selected_tile;
        input integer selected_count;
        input integer expect_output;
        input [15:0] tag;
        integer guard;
        integer output_beat;
        begin
            send_attn_compose_command(
                phase, selected_tile, selected_count, tag
            );
            guard = 0;
            while (!cmd_done_valid && guard < 1000000) begin
                guard = guard + 1;
                @(posedge clk);
            end
            if (!cmd_done_valid) begin
                $display("SHELL_ATTN_COMPOSE_TIMEOUT phase=%0d", phase);
                failures = failures + 1;
            end else begin
                if (cmd_done_tag !== tag || cmd_done_error) begin
                    $display("SHELL_ATTN_COMPOSE_DONE_MISMATCH phase=%0d tag=%0d error=%0d",
                             phase, cmd_done_tag, cmd_done_error);
                    failures = failures + 1;
                end
                if (expect_output) begin
                    if (observed_count != ATTN_COMPOSE_BEATS) begin
                        $display("SHELL_ATTN_COMPOSE_WRITE_COUNT_MISMATCH got=%0d expected=%0d",
                                 observed_count, ATTN_COMPOSE_BEATS);
                        failures = failures + 1;
                    end
                    for (output_beat = 0;
                         output_beat < ATTN_COMPOSE_BEATS;
                         output_beat = output_beat + 1) begin
                        if (observed_write_addr[output_beat] !==
                            ATTN_COMPOSE_OUT_BASE + output_beat*16) begin
                            $display("SHELL_ATTN_COMPOSE_ADDR_MISMATCH beat=%0d got=%016x",
                                     output_beat,
                                     observed_write_addr[output_beat]);
                            failures = failures + 1;
                        end
                        if (observed_output[output_beat] !==
                            attn_compose_expected_beats[output_beat]) begin
                            $display("SHELL_ATTN_COMPOSE_OUTPUT_MISMATCH beat=%0d got=%032x expected=%032x",
                                     output_beat,
                                     observed_output[output_beat],
                                     attn_compose_expected_beats[output_beat]);
                            failures = failures + 1;
                        end
                    end
                    if (cmd_done_saturation !==
                        attn_compose_expected_saturation[0]) begin
                        $display("SHELL_ATTN_COMPOSE_SATURATION_MISMATCH");
                        failures = failures + 1;
                    end
                    attn_compose_cases_seen =
                        attn_compose_cases_seen + 1;
                end else if (observed_count != 0) begin
                    $display("SHELL_ATTN_COMPOSE_UNEXPECTED_WRITE phase=%0d count=%0d",
                             phase, observed_count);
                    failures = failures + 1;
                end
                last_command_cycles = cycle_count - command_start_cycle;
                total_success_cycles =
                    total_success_cycles + last_command_cycles;
                success_runs = success_runs + 1;
            end
            @(negedge clk);
            cmd_done_ready = 1'b1;
            @(posedge clk);
            @(negedge clk);
            cmd_done_ready = 1'b0;
        end
    endtask

    task run_attn_compose_reset_reuse;
        integer guard;
        integer quiet_cycle;
        integer reset_guard;
        integer failures_before;
        reg [63:0] status_value;
        begin
            failures_before = failures;
            run_attn_compose_command(ACE2_ATTN_COMPOSE_MAX_FIRST,
                                     0, 8, 0, 16'h3e00);
            run_attn_compose_command(ACE2_ATTN_COMPOSE_MAX_MORE,
                                     1, 1, 0, 16'h3e01);
            run_attn_compose_command(ACE2_ATTN_COMPOSE_SUM_FIRST,
                                     0, 8, 0, 16'h3e02);
            run_attn_compose_command(ACE2_ATTN_COMPOSE_SUM_MORE,
                                     1, 1, 0, 16'h3e03);
            send_attn_compose_command(ACE2_ATTN_COMPOSE_VALUE_FIRST,
                                      0, 8, 16'h3e04);

            guard = 0;
            while (!((dut.u_attention_compose_core.state_q ==
                      ATTN_COMPOSE_CORE_VALUE_ACCUM_STATE) &&
                     dut.u_attention_compose_core.accumulator_write_q &&
                     dut.u_attention_compose_core.accumulator_commit_q) &&
                   (guard < 1000000)) begin
                guard = guard + 1;
                @(posedge clk);
            end
            if (guard == 1000000) begin
                $display("SHELL_ATTN_COMPOSE_RESET_ACCUM_TIMEOUT");
                failures = failures + 1;
            end

            fork
                begin
                    wait (dut.soft_reset_req_w);
                    if (dut.u_attention_compose_core.state_q !==
                        ATTN_COMPOSE_CORE_VALUE_ACCUM_STATE) begin
                        $display("SHELL_ATTN_COMPOSE_RESET_WRONG_STATE state=%0d",
                                 dut.u_attention_compose_core.state_q);
                        failures = failures + 1;
                    end
                end
                begin
                    csr_write64(ACE2_CSR_CONTROL, 64'h3);
                end
            join

            reset_guard = 0;
            while ((busy || pending_read || mem_rvalid ||
                    dut.u_attention_compose_core.accumulator_write_q ||
                    dut.u_attention_compose_core.accumulator_commit_q) &&
                   (reset_guard < 64)) begin
                reset_guard = reset_guard + 1;
                @(posedge clk);
            end
            if (busy || pending_read || mem_rvalid ||
                dut.u_attention_compose_core.accumulator_write_q ||
                dut.u_attention_compose_core.accumulator_commit_q) begin
                $display("SHELL_ATTN_COMPOSE_RESET_DRAIN_TIMEOUT busy=%0d pending=%0d rvalid=%0d write=%0d commit=%0d",
                         busy, pending_read, mem_rvalid,
                         dut.u_attention_compose_core.accumulator_write_q,
                         dut.u_attention_compose_core.accumulator_commit_q);
                failures = failures + 1;
            end
            if (cmd_done_valid || (observed_count != 0)) begin
                $display("SHELL_ATTN_COMPOSE_RESET_LEAK completion=%0d writes=%0d",
                         cmd_done_valid, observed_count);
                failures = failures + 1;
            end
            for (quiet_cycle = 0; quiet_cycle < 4;
                 quiet_cycle = quiet_cycle + 1) begin
                @(posedge clk);
                if (mem_req_valid || mem_wvalid || mem_rvalid || mem_bvalid) begin
                    $display("SHELL_ATTN_COMPOSE_RESET_NOT_QUIET cycle=%0d req=%0d w=%0d r=%0d b=%0d",
                             quiet_cycle, mem_req_valid, mem_wvalid,
                             mem_rvalid, mem_bvalid);
                    failures = failures + 1;
                end
            end
            reset_busy_cases = reset_busy_cases + 1;
            csr_read64(ACE2_CSR_ERROR_STATUS, status_value);
            if (!status_value[ACE2_ERR_RESET_BUSY]) begin
                $display("SHELL_ATTN_COMPOSE_RESET_BUSY_NOT_STICKY status=%016x",
                         status_value);
                failures = failures + 1;
            end
            csr_write64(ACE2_CSR_ERROR_STATUS, 64'hffff_ffff_ffff_ffff);
            csr_write64(ACE2_CSR_INTERRUPT_ST, 64'hffff_ffff_ffff_ffff);
            csr_write64(ACE2_CSR_CONTROL, 64'h1);

            run_attn_compose_cross_tile();
            if (failures == failures_before) begin
                $display("ACE2_SHELL_ATTN_COMPOSE_RESET_REUSE_PASS writes=4 reset_busy=1");
            end
        end
    endtask

    task run_attn_compose_cross_tile;
        begin
            run_attn_compose_command(ACE2_ATTN_COMPOSE_MAX_FIRST,
                                     0, 8, 0, 16'h3f00);
            run_attn_compose_command(ACE2_ATTN_COMPOSE_MAX_MORE,
                                     1, 1, 0, 16'h3f01);
            run_attn_compose_command(ACE2_ATTN_COMPOSE_SUM_FIRST,
                                     0, 8, 0, 16'h3f02);
            run_attn_compose_command(ACE2_ATTN_COMPOSE_SUM_MORE,
                                     1, 1, 0, 16'h3f03);
            run_attn_compose_command(ACE2_ATTN_COMPOSE_VALUE_FIRST,
                                     0, 8, 0, 16'h3f04);
            run_attn_compose_command(ACE2_ATTN_COMPOSE_VALUE_LAST,
                                     1, 1, 1, 16'h3f05);
        end
    endtask

    task run_shared_layer_sweep;
        reg [63:0] status_value;
        begin
            layer_sweep_high_layer = 23;
            layer_sweep_cases_seen = 0;
            csr_write64(ACE2_CSR_ERROR_STATUS, 64'hffff_ffff_ffff_ffff);

            send_rmsnorm_cmd(0, layer_sweep_high_layer, ACE2_OPCODE_RMSNORM,
                             16'd896, ACT_BASE, 16'h6000);
            wait_done_and_compare(0, 0, 16'h6000);
            layer_sweep_cases_seen = layer_sweep_cases_seen + 1;

            send_qproj_cmd_full(PROJ_CASE_SATURATION, layer_sweep_high_layer,
                                proj_case_rows[PROJ_CASE_SATURATION],
                                16'd896, 16'd896, PROJ_ACT_BASE,
                                PROJ_WEIGHT_BASE, PROJ_OUT_BASE,
                                PROJ_META_BASE, 16'h6001);
            wait_qproj_done_and_compare(
                PROJ_CASE_SATURATION,
                proj_expected_saturation[PROJ_CASE_SATURATION],
                16'h6001
            );
            layer_sweep_cases_seen = layer_sweep_cases_seen + 1;
            if (proj_expected_saturation[PROJ_CASE_SATURATION])
                csr_write64(ACE2_CSR_ERROR_STATUS, 64'hffff_ffff_ffff_ffff);

            send_qproj_cmd_full(PROJ_CASE_BALANCED, layer_sweep_high_layer,
                                proj_case_rows[PROJ_CASE_BALANCED],
                                16'd128, 16'd896, PROJ_ACT_BASE,
                                PROJ_WEIGHT_BASE, PROJ_OUT_BASE,
                                PROJ_META_BASE, 16'h6002);
            wait_qproj_done_and_compare(PROJ_CASE_BALANCED, 0, 16'h6002);
            layer_sweep_cases_seen = layer_sweep_cases_seen + 1;

            send_rope_cmd(0, layer_sweep_high_layer, 16'h6003);
            wait_rope_done_and_compare(0, rope_expected_saturation[0], 16'h6003);
            layer_sweep_cases_seen = layer_sweep_cases_seen + 1;
            if (rope_expected_saturation[0])
                csr_write64(ACE2_CSR_ERROR_STATUS, 64'hffff_ffff_ffff_ffff);

            send_rope_k_cmd(0, layer_sweep_high_layer, 16'h6004);
            wait_rope_done_and_compare(0, rope_expected_saturation[0], 16'h6004);
            layer_sweep_cases_seen = layer_sweep_cases_seen + 1;
            if (rope_expected_saturation[0])
                csr_write64(ACE2_CSR_ERROR_STATUS, 64'hffff_ffff_ffff_ffff);

            send_kv_write_cmd(0, layer_sweep_high_layer, 16'd3, 16'h6005);
            wait_kv_write_done_and_compare(0, 0, 16'h6005);
            layer_sweep_cases_seen = layer_sweep_cases_seen + 1;

            send_attn_score_cmd(0, layer_sweep_high_layer, 16'h6006);
            wait_attn_score_done_and_compare(
                0, attn_score_expected_saturation[0], 16'h6006
            );
            layer_sweep_cases_seen = layer_sweep_cases_seen + 1;
            if (attn_score_expected_saturation[0])
                csr_write64(ACE2_CSR_ERROR_STATUS, 64'hffff_ffff_ffff_ffff);

            send_softmax_cmd(0, layer_sweep_high_layer, 16'h6007);
            wait_softmax_done_and_compare(0, 0, 16'h6007);
            layer_sweep_cases_seen = layer_sweep_cases_seen + 1;

            send_attn_value_cmd(0, layer_sweep_high_layer, 16'h6008);
            wait_attn_value_done_and_compare(
                0, attn_value_expected_saturation[0], 16'h6008
            );
            layer_sweep_cases_seen = layer_sweep_cases_seen + 1;
            if (attn_value_expected_saturation[0])
                csr_write64(ACE2_CSR_ERROR_STATUS, 64'hffff_ffff_ffff_ffff);

            send_residual_cmd_layer_full(0, layer_sweep_high_layer, 16'd896,
                                         RESIDUAL_LHS_BASE, RESIDUAL_RHS_BASE,
                                         RESIDUAL_OUT_BASE, 16'h6009);
            wait_residual_done_and_compare(0, residual_expected_saturation[0],
                                           16'h6009);
            layer_sweep_cases_seen = layer_sweep_cases_seen + 1;
            if (residual_expected_saturation[0])
                csr_write64(ACE2_CSR_ERROR_STATUS, 64'hffff_ffff_ffff_ffff);

            send_post_rms_cmd_layer_full(0, layer_sweep_high_layer, 16'd896,
                                         POST_RMS_GAIN_BASE, 16'h600a);
            wait_post_rms_done_and_compare(0, post_rms_expected_saturation[0],
                                           16'h600a);
            layer_sweep_cases_seen = layer_sweep_cases_seen + 1;
            if (post_rms_expected_saturation[0])
                csr_write64(ACE2_CSR_ERROR_STATUS, 64'hffff_ffff_ffff_ffff);

            send_silu_gate_cmd_layer(0, layer_sweep_high_layer,
                                     silu_case_length[0], 16'h600b);
            wait_silu_gate_done_and_compare(0, 0, 16'h600b);
            layer_sweep_cases_seen = layer_sweep_cases_seen + 1;
            if (silu_expected_saturation[0])
                csr_write64(ACE2_CSR_ERROR_STATUS, 64'hffff_ffff_ffff_ffff);

            send_mlp_residual_cmd_layer_full(
                0, layer_sweep_high_layer, 16'd896, MLP_RESIDUAL_DOWN_BASE,
                MLP_RESIDUAL_STREAM_BASE, MLP_RESIDUAL_OUT_BASE, 16'h600c
            );
            wait_mlp_residual_done_and_compare(
                0, mlp_residual_expected_saturation[0], 16'h600c
            );
            layer_sweep_cases_seen = layer_sweep_cases_seen + 1;
            if (mlp_residual_expected_saturation[0])
                csr_write64(ACE2_CSR_ERROR_STATUS, 64'hffff_ffff_ffff_ffff);

            csr_read64(ACE2_CSR_ERROR_STATUS, status_value);
            if (status_value != 64'd0) begin
                $display("LAYER_SWEEP_UNCLEARED_ERROR_STATUS status=%016x",
                         status_value);
                failures = failures + 1;
            end
            $display("ACE2_SHELL_LAYER_SWEEP_TB_PASS high_layer=%0d cases=%0d projection_shapes=2 opcode_families=9 cycles=%0d",
                     layer_sweep_high_layer, layer_sweep_cases_seen,
                     total_success_cycles);
        end
    endtask

    task run_final_rmsnorm;
        reg [63:0] status_value;
        begin
            csr_write64(ACE2_CSR_ERROR_STATUS, 64'hffff_ffff_ffff_ffff);

            send_rmsnorm_cmd(0, 24, ACE2_OPCODE_RMSNORM, 16'd896,
                             ACT_BASE, 16'h6100);
            wait_done_and_compare(0, 0, 16'h6100);

            send_rmsnorm_cmd(1, 24, ACE2_OPCODE_RMSNORM, 16'd896,
                             ACT_BASE, 16'h6101);
            wait_done_and_compare(1, 0, 16'h6101);

            send_rmsnorm_cmd(2, 25, ACE2_OPCODE_RMSNORM, 16'd896,
                             ACT_BASE, 16'h6102);
            wait_done_and_compare(2, 1, 16'h6102);
            csr_read64(ACE2_CSR_ERROR_STATUS, status_value);
            if (!status_value[ACE2_ERR_DESCRIPTOR]) begin
                $display("FINAL_RMSNORM_BAD_LAYER_NOT_REJECTED status=%016x",
                         status_value);
                failures = failures + 1;
            end
            csr_write64(ACE2_CSR_ERROR_STATUS, 64'hffff_ffff_ffff_ffff);

            send_rmsnorm_cmd_full(2, 24, ACE2_OPCODE_RMSNORM, 16'd2,
                                  16'd896, 16'd0, ACT_BASE, OUT_BASE,
                                  GAIN_BASE, 16'h6103);
            wait_done_and_compare(2, 1, 16'h6103);
            csr_read64(ACE2_CSR_ERROR_STATUS, status_value);
            if (!status_value[ACE2_ERR_DESCRIPTOR]) begin
                $display("FINAL_RMSNORM_BAD_M_NOT_REJECTED status=%016x",
                         status_value);
                failures = failures + 1;
            end
            csr_write64(ACE2_CSR_ERROR_STATUS, 64'hffff_ffff_ffff_ffff);

            send_rmsnorm_cmd_full(2, 24, ACE2_OPCODE_RMSNORM, 16'd1,
                                  16'd896, 16'd1, ACT_BASE, OUT_BASE,
                                  GAIN_BASE, 16'h6104);
            wait_done_and_compare(2, 1, 16'h6104);
            csr_read64(ACE2_CSR_ERROR_STATUS, status_value);
            if (!status_value[ACE2_ERR_DESCRIPTOR]) begin
                $display("FINAL_RMSNORM_BAD_K_NOT_REJECTED status=%016x",
                         status_value);
                failures = failures + 1;
            end
            csr_write64(ACE2_CSR_ERROR_STATUS, 64'hffff_ffff_ffff_ffff);

            $display("ACE2_SHELL_FINAL_RMSNORM_TB_PASS layer=24 cases=2 rejected_layer=25 rejected_m=2 rejected_k=1 descriptor_errors=3 cycles=%0d",
                     total_success_cycles);
        end
    endtask

    task run_lm_head;
        reg [63:0] status_value;
        begin
            csr_write64(ACE2_CSR_ERROR_STATUS, 64'hffff_ffff_ffff_ffff);

            proj_act_read_count = 0;
            proj_weight_read_count = 0;
            proj_meta_read_count = 0;
            proj_write_req_count = 0;
            proj_min_weight_read_addr = 64'hffff_ffff_ffff_ffff;
            proj_max_weight_read_addr = 64'd0;
            send_lm_head_cmd(PROJ_CASE_BALANCED, 16'h6200);
            wait_qproj_done_and_compare(PROJ_CASE_BALANCED, 0, 16'h6200);
            if (proj_act_read_count != 7168 ||
                proj_weight_read_count != 7168 ||
                proj_meta_read_count != 32 ||
                proj_write_req_count != 2 ||
                proj_min_weight_read_addr !== PROJ_WEIGHT_BASE ||
                proj_max_weight_read_addr !== (PROJ_WEIGHT_BASE + 64'd14320) ||
                proj_max_weight_read_addr >= (PROJ_WEIGHT_BASE + 64'd14336)) begin
                $display("LM_HEAD_ACCESS_CONTRACT_MISMATCH act_reads=%0d weight_reads=%0d meta_reads=%0d writes=%0d min_weight=%016x max_weight=%016x expected_min=%016x expected_max=%016x region_end=%016x",
                         proj_act_read_count, proj_weight_read_count,
                         proj_meta_read_count, proj_write_req_count,
                         proj_min_weight_read_addr, proj_max_weight_read_addr,
                         PROJ_WEIGHT_BASE, PROJ_WEIGHT_BASE + 64'd14320,
                         PROJ_WEIGHT_BASE + 64'd14336);
                failures = failures + 1;
            end

            proj_act_read_count = 0;
            proj_weight_read_count = 0;
            proj_meta_read_count = 0;
            proj_write_req_count = 0;
            proj_min_weight_read_addr = 64'hffff_ffff_ffff_ffff;
            proj_max_weight_read_addr = 64'd0;
            send_lm_head_cmd(PROJ_CASE_SATURATION, 16'h6201);
            wait_qproj_done_and_compare(
                PROJ_CASE_SATURATION,
                proj_expected_saturation[PROJ_CASE_SATURATION],
                16'h6201
            );
            if (proj_act_read_count != 7168 ||
                proj_weight_read_count != 7168 ||
                proj_meta_read_count != 32 ||
                proj_write_req_count != 2 ||
                proj_min_weight_read_addr !== PROJ_WEIGHT_BASE ||
                proj_max_weight_read_addr !== (PROJ_WEIGHT_BASE + 64'd14320) ||
                proj_max_weight_read_addr >= (PROJ_WEIGHT_BASE + 64'd14336)) begin
                $display("LM_HEAD_SATURATION_ACCESS_CONTRACT_MISMATCH act_reads=%0d weight_reads=%0d meta_reads=%0d writes=%0d min_weight=%016x max_weight=%016x expected_min=%016x expected_max=%016x region_end=%016x",
                         proj_act_read_count, proj_weight_read_count,
                         proj_meta_read_count, proj_write_req_count,
                         proj_min_weight_read_addr, proj_max_weight_read_addr,
                         PROJ_WEIGHT_BASE, PROJ_WEIGHT_BASE + 64'd14320,
                         PROJ_WEIGHT_BASE + 64'd14336);
                failures = failures + 1;
            end
            csr_write64(ACE2_CSR_ERROR_STATUS, 64'hffff_ffff_ffff_ffff);

            send_qproj_cmd_full(PROJ_CASE_BALANCED, 23, 16'd1, 16'd32,
                                16'd896, PROJ_ACT_BASE, PROJ_WEIGHT_BASE,
                                PROJ_OUT_BASE, PROJ_META_BASE, 16'h6202);
            wait_qproj_done_and_compare(PROJ_CASE_BALANCED, 1, 16'h6202);
            csr_read64(ACE2_CSR_ERROR_STATUS, status_value);
            if (!status_value[ACE2_ERR_DESCRIPTOR]) begin
                $display("LM_HEAD_BAD_LAYER_NOT_REJECTED status=%016x", status_value);
                failures = failures + 1;
            end
            csr_write64(ACE2_CSR_ERROR_STATUS, 64'hffff_ffff_ffff_ffff);

            send_qproj_cmd_full(PROJ_CASE_BALANCED, 24, 16'd2, 16'd32,
                                16'd896, PROJ_ACT_BASE, PROJ_WEIGHT_BASE,
                                PROJ_OUT_BASE, PROJ_META_BASE, 16'h6203);
            wait_qproj_done_and_compare(PROJ_CASE_BALANCED, 1, 16'h6203);
            csr_read64(ACE2_CSR_ERROR_STATUS, status_value);
            if (!status_value[ACE2_ERR_DESCRIPTOR]) begin
                $display("LM_HEAD_BAD_M_NOT_REJECTED status=%016x", status_value);
                failures = failures + 1;
            end
            csr_write64(ACE2_CSR_ERROR_STATUS, 64'hffff_ffff_ffff_ffff);

            send_qproj_cmd_full(PROJ_CASE_BALANCED, 24, 16'd1, 16'd64,
                                16'd896, PROJ_ACT_BASE, PROJ_WEIGHT_BASE,
                                PROJ_OUT_BASE, PROJ_META_BASE, 16'h6204);
            wait_qproj_done_and_compare(PROJ_CASE_BALANCED, 1, 16'h6204);
            csr_read64(ACE2_CSR_ERROR_STATUS, status_value);
            if (!status_value[ACE2_ERR_DESCRIPTOR]) begin
                $display("LM_HEAD_BAD_N_NOT_REJECTED status=%016x", status_value);
                failures = failures + 1;
            end
            csr_write64(ACE2_CSR_ERROR_STATUS, 64'hffff_ffff_ffff_ffff);

            send_qproj_cmd_full(PROJ_CASE_BALANCED, 24, 16'd1, 16'd32,
                                16'd768, PROJ_ACT_BASE, PROJ_WEIGHT_BASE,
                                PROJ_OUT_BASE, PROJ_META_BASE, 16'h6205);
            wait_qproj_done_and_compare(PROJ_CASE_BALANCED, 1, 16'h6205);
            csr_read64(ACE2_CSR_ERROR_STATUS, status_value);
            if (!status_value[ACE2_ERR_DESCRIPTOR]) begin
                $display("LM_HEAD_BAD_K_NOT_REJECTED status=%016x", status_value);
                failures = failures + 1;
            end
            csr_write64(ACE2_CSR_ERROR_STATUS, 64'hffff_ffff_ffff_ffff);

            $display("ACE2_SHELL_LM_HEAD_TB_PASS layer=24 tile_outputs=32 vocab=151936 tiles=4748 cases=2 read_beats=14368 write_beats=2 weight_span_bytes=14336 weight_high_water_offset=14320 rejected_layer=23 rejected_m=2 rejected_n=64 rejected_k=768 descriptor_errors=4 cycles=%0d",
                     total_success_cycles);
        end
    endtask

    initial begin
        reg [63:0] read_value;
        failures = 0;
        current_case = 0;
        current_proj_case = 0;
        current_proj_m = 1;
        current_proj_n = 896;
        current_proj_dst_addr = PROJ_OUT_BASE;
        current_rope_case = 0;
        current_rope_beats = ROPE_BEATS;
        current_rope_table_stride = ROPE_BEATS * 64;
        current_kv_case = 0;
        current_kv_sequence = 0;
        current_attn_score_case = 0;
        current_attn_metadata_mode = 0;
        current_softmax_case = 0;
        current_attn_value_case = 0;
        current_residual_case = 0;
        current_mlp_residual_case = 0;
        current_post_rms_case = 0;
        qproj_cases_seen = 0;
        kproj_cases_seen = 0;
        vproj_cases_seen = 0;
        command_start_cycle = 0;
        last_command_cycles = 0;
        max_command_cycles = 0;
        total_success_cycles = 0;
        success_runs = 0;
        descriptor_error_cases = 0;
        memory_error_cases = 0;
        response_protocol_cases = 0;
        watchdog_error_cases = 0;
        reset_busy_cases = 0;
        layer_sweep_cases_seen = 0;
        layer_sweep_high_layer = 0;
        rope_cases_seen = 0;
        rope_q_cases_seen = 0;
        rope_k_cases_seen = 0;
        kv_write_cases_seen = 0;
        attn_score_cases_seen = 0;
        attn_score_metadata_error_cases = 0;
        softmax_cases_seen = 0;
        attn_value_cases_seen = 0;
        current_attn_compose_case = 0;
        attn_compose_cases_seen = 0;
        oproj_cases_seen = 0;
        mlp_gate_proj_cases_seen = 0;
        mlp_up_proj_cases_seen = 0;
        mlp_down_proj_cases_seen = 0;
        mlp_residual_cases_seen = 0;
        silu_gate_cases_seen = 0;
        residual_cases_seen = 0;
        post_rms_cases_seen = 0;
        oproj_writes_seen = 0;
        oproj_total_cycles = 0;
        oproj_max_cycles = 0;
        oproj_command_accept_cycle = 0;
        oproj_completion_accept_cycle = 0;
        oproj_signature = 64'd0;
        oproj_phase_cycle = 0;
        oproj_active = 1'b0;
        smoke_opcode = 0;
        qproj_stride_only_mode = $test$plusargs("QPROJ_STRIDE_ONLY");
        mlp_up_only_mode = $test$plusargs("MLP_UP_ONLY");
        mlp_down_only_mode = $test$plusargs("MLP_DOWN_ONLY");
        mlp_residual_only_mode = $test$plusargs("MLP_RESIDUAL_ONLY");
        silu_only_mode = $test$plusargs("SILU_ONLY");
        layer_sweep_only_mode = $test$plusargs("LAYER_SWEEP_ONLY");
        final_rmsnorm_only_mode = $test$plusargs("FINAL_RMSNORM_ONLY");
        lm_head_only_mode = $test$plusargs("LM_HEAD_ONLY");
        attn_compose_only_mode = $test$plusargs("ATTN_COMPOSE_ONLY");
        attn_score_only_mode = $test$plusargs("ATTN_SCORE_ONLY");
        attn_retime_trace_mode = $test$plusargs("ATTN_RETIME_TRACE");
        rope_only_mode = $test$plusargs("ROPE_ONLY");
        smoke_mode = $value$plusargs("SMOKE_OPCODE=%h", smoke_opcode);
        if (smoke_mode && (smoke_opcode != ACE2_OPCODE_W4A8_PROJ) &&
            (smoke_opcode != ACE2_OPCODE_RESIDUAL_ADD)) begin
            $fatal(1, "UNSUPPORTED_SMOKE_OPCODE value=%02x", smoke_opcode);
        end
        apply_reset();

        csr_read64(ACE2_CSR_ID, read_value);
        if (read_value !== 64'h4143453200000001) begin
            $display("CSR_ID_MISMATCH got=%016x", read_value);
            failures = failures + 1;
        end
        csr_write64(ACE2_CSR_CONTROL, 64'h1);

        if (qproj_stride_only_mode) begin
            send_qproj_cmd(PROJ_CASE_BALANCED, 0, 16'h3100);
            wait_qproj_done_and_compare(
                PROJ_CASE_BALANCED,
                proj_expected_saturation[PROJ_CASE_BALANCED],
                16'h3100
            );
            qproj_cases_seen = qproj_cases_seen + 1;
            send_qproj_cmd_full(
                PROJ_CASE_C4_V_BIAS, 0, 16'd1, 16'd128, 16'd896,
                PROJ_ACT_BASE, PROJ_WEIGHT_BASE, PROJ_OUT_BASE,
                PROJ_META_BASE, 16'h3101
            );
            wait_qproj_done_and_compare(
                PROJ_CASE_C4_V_BIAS,
                proj_expected_saturation[PROJ_CASE_C4_V_BIAS],
                16'h3101
            );
            qproj_cases_seen = qproj_cases_seen + 1;
            if (failures != 0) begin
                $display("ACE2_SHELL_QPROJ_STRIDE_TB_FAIL failures=%0d", failures);
                $fatal(1, "ACE2_SHELL_QPROJ_STRIDE_TB_FAIL");
            end
            $display("ACE2_SHELL_QPROJ_STRIDE_TB_PASS cases=%0d rows=%0d input_beats_per_row=%0d writes=%0d cycles=%0d c4_bias_channel=30 c4_output=127",
                     qproj_cases_seen, proj_case_rows[PROJ_CASE_BALANCED],
                     proj_case_input_beats[PROJ_CASE_BALANCED], observed_count,
                     last_command_cycles);
            $finish;
        end

        if (mlp_up_only_mode) begin
            run_mlp_up_projection_case();
            if (failures != 0) begin
                $display("ACE2_SHELL_MLP_UP_TB_FAIL failures=%0d", failures);
                $fatal(1, "ACE2_SHELL_MLP_UP_TB_FAIL");
            end
            $finish;
        end

        if (mlp_down_only_mode) begin
            run_mlp_down_projection_case();
            if (failures != 0) begin
                $display("ACE2_SHELL_MLP_DOWN_TB_FAIL failures=%0d", failures);
                $fatal(1, "ACE2_SHELL_MLP_DOWN_TB_FAIL");
            end
            $finish;
        end

        if (mlp_residual_only_mode) begin
            run_mlp_residual_cases();
            if (failures != 0) begin
                $display("ACE2_SHELL_MLP_RESIDUAL_TB_FAIL failures=%0d", failures);
                $fatal(1, "ACE2_SHELL_MLP_RESIDUAL_TB_FAIL");
            end
            $finish;
        end

        if (silu_only_mode) begin
            run_silu_gate_cases();
            if (failures != 0) begin
                $display("ACE2_SHELL_SILU_GATE_TB_FAIL failures=%0d", failures);
                $fatal(1, "ACE2_SHELL_SILU_GATE_TB_FAIL");
            end
            $finish;
        end

        if (layer_sweep_only_mode) begin
            run_shared_layer_sweep();
            if (failures != 0) begin
                $display("ACE2_SHELL_LAYER_SWEEP_TB_FAIL failures=%0d", failures);
                $fatal(1, "ACE2_SHELL_LAYER_SWEEP_TB_FAIL");
            end
            $finish;
        end

        if (final_rmsnorm_only_mode) begin
            run_final_rmsnorm();
            if (failures != 0) begin
                $display("ACE2_SHELL_FINAL_RMSNORM_TB_FAIL failures=%0d",
                         failures);
                $fatal(1, "ACE2_SHELL_FINAL_RMSNORM_TB_FAIL");
            end
            $finish;
        end

        if (lm_head_only_mode) begin
            run_lm_head();
            if (failures != 0) begin
                $display("ACE2_SHELL_LM_HEAD_TB_FAIL failures=%0d", failures);
                $fatal(1, "ACE2_SHELL_LM_HEAD_TB_FAIL");
            end
            $finish;
        end

        if (attn_compose_only_mode) begin
            run_attn_compose_cross_tile();
            if (failures != 0) begin
                $display("ACE2_SHELL_ATTN_COMPOSE_ONLY_TB_FAIL failures=%0d",
                         failures);
                $fatal(1, "ACE2_SHELL_ATTN_COMPOSE_ONLY_TB_FAIL");
            end
            $display("ACE2_SHELL_ATTN_COMPOSE_ONLY_TB_PASS cases=%0d phases=6 tiles=2 contexts=9",
                     attn_compose_cases_seen);
            $finish;
        end

        if (attn_score_only_mode) begin
            run_attn_score_metadata_error_cases();
            for (case_index = 0; case_index < ATTN_SCORE_CASE_COUNT;
                 case_index = case_index + 1) begin
                send_attn_score_cmd(case_index, 0, 16'h3a00 + case_index);
                wait_attn_score_done_and_compare(
                    case_index,
                    attn_score_expected_saturation[case_index],
                    16'h3a00 + case_index
                );
            end
            if (failures != 0) begin
                $display("ACE2_SHELL_ATTN_SCORE_TB_FAIL failures=%0d", failures);
                $fatal(1, "ACE2_SHELL_ATTN_SCORE_TB_FAIL");
            end
            $display(
                "ACE2_SHELL_ATTN_SCORE_TB_PASS cases=%0d unequal_scale_case=1 invalid_metadata_cases=%0d",
                ATTN_SCORE_CASE_COUNT, attn_score_metadata_error_cases
            );
            $finish;
        end

        if (rope_only_mode) begin
            send_rope_cmd_full_flags(
                0, 0, 16'd896, rope_sequence_position[0],
                ROPE_ACT_BASE, ROPE_TABLE_BASE, ROPE_OUT_BASE,
                ROPE_SCALE_BASE, 16'h34ff, 8'd0
            );
            wait_rope_done_and_compare(0, 1, 16'h34ff);
            descriptor_error_cases = descriptor_error_cases + 1;
            csr_read64(ACE2_CSR_ERROR_STATUS, read_value);
            if (!read_value[ACE2_ERR_DESCRIPTOR]) begin
                $display("ROPE_BASIS_TAG_ERROR_NOT_STICKY status=%016x", read_value);
                failures = failures + 1;
            end
            csr_write64(ACE2_CSR_ERROR_STATUS, 64'hffff_ffff_ffff_ffff);
            for (case_index = 0; case_index < ROPE_CASE_COUNT; case_index = case_index + 1) begin
                send_rope_cmd(case_index, 0, 16'h3500 + case_index);
                wait_rope_done_and_compare(
                    case_index, rope_expected_saturation[case_index],
                    16'h3500 + case_index
                );
                send_rope_k_cmd(case_index, 0, 16'h3600 + case_index);
                wait_rope_done_and_compare(
                    case_index, rope_expected_saturation[case_index],
                    16'h3600 + case_index
                );
            end
            if (failures != 0) begin
                $display("ACE2_SHELL_ROPE_TB_FAIL failures=%0d", failures);
                $fatal(1, "ACE2_SHELL_ROPE_TB_FAIL");
            end
            $display(
                "ACE2_SHELL_ROPE_TB_PASS q_cases=%0d k_cases=%0d basis_half_degrees=%0d invalid_basis_cases=1",
                rope_q_cases_seen, rope_k_cases_seen,
                ROPE_BASIS_ROTATION_HALF_DEGREES
            );
            $finish;
        end

        if (smoke_mode) begin
            if (smoke_opcode == ACE2_OPCODE_RESIDUAL_ADD) begin
                run_vector_family_cases();
            end else begin
                run_oproj_cases();
            end
            if (failures != 0) begin
                $display("ACE2_SHELL_SMOKE_TB_FAIL failures=%0d", failures);
                $fatal(1, "ACE2_SHELL_SMOKE_TB_FAIL");
            end
            if (smoke_opcode == ACE2_OPCODE_W4A8_PROJ) report_oproj_result();
            $display("ACE2_SHELL_SMOKE_TB_PASS opcode=%02x", smoke_opcode);
            $finish;
        end

        send_rmsnorm_cmd(0, 0, 8'h7f, 16'd896, ACT_BASE, 16'h00ee);
        wait_done_and_compare(0, 1, 16'h00ee);
        descriptor_error_cases = descriptor_error_cases + 1;
        csr_read64(ACE2_CSR_ERROR_STATUS, read_value);
        if (!read_value[ACE2_ERR_DESCRIPTOR]) begin
            $display("DESCRIPTOR_ERROR_NOT_STICKY status=%016x", read_value);
            failures = failures + 1;
        end
        csr_write64(ACE2_CSR_ERROR_STATUS, 64'hffff_ffff_ffff_ffff);

        send_rmsnorm_cmd(1, 0, ACE2_OPCODE_RMSNORM, 16'd895, ACT_BASE, 16'h00ef);
        wait_done_and_compare(1, 1, 16'h00ef);
        descriptor_error_cases = descriptor_error_cases + 1;
        csr_read64(ACE2_CSR_ERROR_STATUS, read_value);
        if (!read_value[ACE2_ERR_DESCRIPTOR]) begin
            $display("DESCRIPTOR_ERROR_BAD_N_NOT_STICKY status=%016x", read_value);
            failures = failures + 1;
        end
        csr_write64(ACE2_CSR_ERROR_STATUS, 64'hffff_ffff_ffff_ffff);

        send_rmsnorm_cmd(2, 25, ACE2_OPCODE_RMSNORM, 16'd896, ACT_BASE, 16'h00f0);
        wait_done_and_compare(2, 1, 16'h00f0);
        descriptor_error_cases = descriptor_error_cases + 1;
        csr_read64(ACE2_CSR_ERROR_STATUS, read_value);
        if (!read_value[ACE2_ERR_DESCRIPTOR]) begin
            $display("DESCRIPTOR_ERROR_BAD_LAYER_NOT_STICKY status=%016x", read_value);
            failures = failures + 1;
        end
        csr_write64(ACE2_CSR_ERROR_STATUS, 64'hffff_ffff_ffff_ffff);

        send_rmsnorm_cmd_full(3, 0, ACE2_OPCODE_RMSNORM, 16'd1, 16'd896,
                              16'd0, ACT_BASE + 64'd1, OUT_BASE, GAIN_BASE,
                              16'h00f1);
        wait_done_and_compare(3, 1, 16'h00f1);
        descriptor_error_cases = descriptor_error_cases + 1;
        csr_read64(ACE2_CSR_ERROR_STATUS, read_value);
        if (!read_value[ACE2_ERR_DESCRIPTOR]) begin
            $display("DESCRIPTOR_ERROR_SRC_ALIGN_NOT_STICKY status=%016x", read_value);
            failures = failures + 1;
        end
        csr_write64(ACE2_CSR_ERROR_STATUS, 64'hffff_ffff_ffff_ffff);

        send_rmsnorm_cmd_full(3, 0, ACE2_OPCODE_RMSNORM, 16'd1, 16'd896,
                              16'd0, ACT_BASE, OUT_BASE + 64'd8, GAIN_BASE,
                              16'h00f2);
        wait_done_and_compare(3, 1, 16'h00f2);
        descriptor_error_cases = descriptor_error_cases + 1;
        csr_read64(ACE2_CSR_ERROR_STATUS, read_value);
        if (!read_value[ACE2_ERR_DESCRIPTOR]) begin
            $display("DESCRIPTOR_ERROR_DST_ALIGN_NOT_STICKY status=%016x", read_value);
            failures = failures + 1;
        end
        csr_write64(ACE2_CSR_ERROR_STATUS, 64'hffff_ffff_ffff_ffff);

        send_rmsnorm_cmd_full(3, 0, ACE2_OPCODE_RMSNORM, 16'd1, 16'd896,
                              16'd0, ACT_BASE, OUT_BASE, GAIN_BASE + 64'd2,
                              16'h00f3);
        wait_done_and_compare(3, 1, 16'h00f3);
        descriptor_error_cases = descriptor_error_cases + 1;
        csr_read64(ACE2_CSR_ERROR_STATUS, read_value);
        if (!read_value[ACE2_ERR_DESCRIPTOR]) begin
            $display("DESCRIPTOR_ERROR_SCALE_ALIGN_NOT_STICKY status=%016x", read_value);
            failures = failures + 1;
        end
        csr_write64(ACE2_CSR_ERROR_STATUS, 64'hffff_ffff_ffff_ffff);

        inject_read_error = 1'b1;
        inject_read_error_tag = 8'h10;
        send_rmsnorm_cmd(4, 0, ACE2_OPCODE_RMSNORM, 16'd896, ACT_BASE, 16'h00f4);
        wait_done_and_compare(4, 1, 16'h00f4);
        memory_error_cases = memory_error_cases + 1;
        response_protocol_cases = response_protocol_cases + 1;
        csr_read64(ACE2_CSR_ERROR_STATUS, read_value);
        if (!read_value[ACE2_ERR_MEMORY]) begin
            $display("MEMORY_READ_ERROR_NOT_STICKY status=%016x", read_value);
            failures = failures + 1;
        end
        csr_write64(ACE2_CSR_ERROR_STATUS, 64'hffff_ffff_ffff_ffff);

        inject_write_error = 1'b1;
        send_rmsnorm_cmd(5, 0, ACE2_OPCODE_RMSNORM, 16'd896, ACT_BASE, 16'h00f5);
        wait_done_and_compare(5, 1, 16'h00f5);
        memory_error_cases = memory_error_cases + 1;
        response_protocol_cases = response_protocol_cases + 1;
        csr_read64(ACE2_CSR_ERROR_STATUS, read_value);
        if (!read_value[ACE2_ERR_MEMORY]) begin
            $display("MEMORY_WRITE_ERROR_NOT_STICKY status=%016x", read_value);
            failures = failures + 1;
        end
        csr_write64(ACE2_CSR_ERROR_STATUS, 64'hffff_ffff_ffff_ffff);

        inject_read_wrong_tag = 1'b1;
        send_rmsnorm_cmd(6, 0, ACE2_OPCODE_RMSNORM, 16'd896, ACT_BASE, 16'h00fa);
        wait_done_and_compare(6, 1, 16'h00fa);
        memory_error_cases = memory_error_cases + 1;
        response_protocol_cases = response_protocol_cases + 1;
        csr_read64(ACE2_CSR_ERROR_STATUS, read_value);
        if (!read_value[ACE2_ERR_MEMORY]) begin
            $display("MEMORY_READ_TAG_ERROR_NOT_STICKY status=%016x", read_value);
            failures = failures + 1;
        end
        csr_write64(ACE2_CSR_ERROR_STATUS, 64'hffff_ffff_ffff_ffff);

        inject_write_wrong_tag = 1'b1;
        send_softmax_cmd(0, 0, 16'h00fb);
        wait_softmax_done_and_compare(0, 1, 16'h00fb);
        memory_error_cases = memory_error_cases + 1;
        response_protocol_cases = response_protocol_cases + 1;
        csr_read64(ACE2_CSR_ERROR_STATUS, read_value);
        if (!read_value[ACE2_ERR_MEMORY]) begin
            $display("MEMORY_WRITE_TAG_ERROR_NOT_STICKY status=%016x", read_value);
            failures = failures + 1;
        end
        csr_write64(ACE2_CSR_ERROR_STATUS, 64'hffff_ffff_ffff_ffff);

        next_read_delay = 7;
        next_write_response_delay = 9;
        send_softmax_cmd(0, 0, 16'h00fc);
        wait_softmax_done_and_compare(0, 0, 16'h00fc);
        response_protocol_cases = response_protocol_cases + 1;

        csr_write64(ACE2_CSR_INTERRUPT_EN, 64'd1 << ACE2_ERR_WATCHDOG);
        csr_write64(ACE2_CSR_WATCHDOG_LIMIT, 64'd2);
        csr_write64(ACE2_CSR_CONTROL, 64'h0000_0000_0000_000d);
        force_mem_req_stall = 1'b1;
        send_rmsnorm_cmd(6, 0, ACE2_OPCODE_RMSNORM, 16'd896, ACT_BASE, 16'h00f6);
        wait_done_and_compare(6, 1, 16'h00f6);
        force_mem_req_stall = 1'b0;
        watchdog_error_cases = watchdog_error_cases + 1;
        csr_read64(ACE2_CSR_ERROR_STATUS, read_value);
        if (!read_value[ACE2_ERR_WATCHDOG]) begin
            $display("WATCHDOG_ERROR_NOT_STICKY status=%016x", read_value);
            failures = failures + 1;
        end
        csr_read64(ACE2_CSR_STATUS, read_value);
        if (!read_value[2]) begin
            $display("STRICT_ERROR_DID_NOT_HALT status=%016x", read_value);
            failures = failures + 1;
        end
        if (!irq) begin
            $display("WATCHDOG_IRQ_NOT_ASSERTED");
            failures = failures + 1;
        end
        @(negedge clk);
        cmd_opcode = ACE2_OPCODE_RMSNORM;
        cmd_layer_id = 8'd0;
        cmd_m = 16'd1;
        cmd_n = 16'd896;
        cmd_k = 16'd0;
        cmd_sequence_position = 16'd0;
        cmd_src0_addr = ACT_BASE;
        cmd_src1_addr = 64'd0;
        cmd_dst_addr = OUT_BASE;
        cmd_scale_addr = GAIN_BASE;
        cmd_scratch_addr = 64'd0;
        cmd_completion_tag = 16'h00f7;
        cmd_valid = 1'b1;
        @(posedge clk);
        if (cmd_ready) begin
            $display("STRICT_ERROR_ACCEPTED_NEW_COMMAND");
            failures = failures + 1;
        end
        @(negedge clk);
        cmd_valid = 1'b0;
        csr_write64(ACE2_CSR_INTERRUPT_ST, 64'hffff_ffff_ffff_ffff);
        csr_write64(ACE2_CSR_ERROR_STATUS, 64'hffff_ffff_ffff_ffff);
        csr_write64(ACE2_CSR_WATCHDOG_LIMIT, 64'd0);
        csr_write64(ACE2_CSR_CONTROL, 64'h1);

        send_rmsnorm_cmd(7, 0, ACE2_OPCODE_RMSNORM, 16'd896, ACT_BASE, 16'h00f8);
        while (!busy) @(posedge clk);
        csr_write64(ACE2_CSR_CONTROL, 64'h3);
        pending_read = 0;
        mem_rvalid = 1'b0;
        mem_rerror = 1'b0;
        repeat (4) @(posedge clk);
        if (cmd_done_valid) begin
            $display("SOFT_RESET_LEFT_COMPLETION_VALID");
            failures = failures + 1;
        end
        reset_busy_cases = reset_busy_cases + 1;
        csr_read64(ACE2_CSR_ERROR_STATUS, read_value);
        if (!read_value[ACE2_ERR_RESET_BUSY]) begin
            $display("RESET_BUSY_ERROR_NOT_STICKY status=%016x", read_value);
            failures = failures + 1;
        end
        csr_write64(ACE2_CSR_ERROR_STATUS, 64'hffff_ffff_ffff_ffff);
        csr_write64(ACE2_CSR_INTERRUPT_ST, 64'hffff_ffff_ffff_ffff);
        csr_write64(ACE2_CSR_CONTROL, 64'h1);

        csr_write64(ACE2_CSR_ERROR_STATUS, 64'hffff_ffff_ffff_ffff);

        for (layer_index = 0; layer_index <= 24; layer_index = layer_index + 1) begin
            case_index = layer_index % TEST_COUNT;
            send_rmsnorm_cmd(case_index, layer_index, ACE2_OPCODE_RMSNORM, 16'd896, ACT_BASE, 16'h1000 + layer_index);
            wait_done_and_compare(
                case_index,
                test_expected_saturation[case_index],
                16'h1000 + layer_index
            );
            if (test_expected_saturation[case_index])
                csr_write64(ACE2_CSR_ERROR_STATUS, 64'hffff_ffff_ffff_ffff);
        end

        send_rmsnorm_cmd(4, 7, ACE2_OPCODE_RMSNORM, 16'd896, ACT_BASE, 16'h2007);
        wait_done_and_compare(4, test_expected_saturation[4], 16'h2007);
        if (test_expected_saturation[4])
            csr_write64(ACE2_CSR_ERROR_STATUS, 64'hffff_ffff_ffff_ffff);
        send_rmsnorm_cmd(5, 7, ACE2_OPCODE_RMSNORM, 16'd896, ACT_BASE, 16'h2008);
        wait_done_and_compare(5, test_expected_saturation[5], 16'h2008);
        if (test_expected_saturation[5])
            csr_write64(ACE2_CSR_ERROR_STATUS, 64'hffff_ffff_ffff_ffff);

        send_qproj_cmd_full(0, 0, 16'd2, 16'd896, 16'd768, PROJ_ACT_BASE, PROJ_WEIGHT_BASE, PROJ_OUT_BASE, PROJ_META_BASE, 16'h3000);
        wait_qproj_done_and_compare(0, 1, 16'h3000);
        descriptor_error_cases = descriptor_error_cases + 1;
        csr_read64(ACE2_CSR_ERROR_STATUS, read_value);
        if (!read_value[ACE2_ERR_DESCRIPTOR]) begin
            $display("QPROJ_DESCRIPTOR_ERROR_BAD_K_NOT_STICKY status=%016x", read_value);
            failures = failures + 1;
        end
        csr_write64(ACE2_CSR_ERROR_STATUS, 64'hffff_ffff_ffff_ffff);

        send_qproj_cmd_full(0, 0, 16'd2, 16'd256, 16'd896, PROJ_ACT_BASE, PROJ_WEIGHT_BASE, PROJ_OUT_BASE, PROJ_META_BASE, 16'h3001);
        wait_qproj_done_and_compare(0, 1, 16'h3001);
        descriptor_error_cases = descriptor_error_cases + 1;
        csr_read64(ACE2_CSR_ERROR_STATUS, read_value);
        if (!read_value[ACE2_ERR_DESCRIPTOR]) begin
            $display("QPROJ_DESCRIPTOR_ERROR_BAD_N_NOT_STICKY status=%016x", read_value);
            failures = failures + 1;
        end
        csr_write64(ACE2_CSR_ERROR_STATUS, 64'hffff_ffff_ffff_ffff);

        send_qproj_cmd(0, 0, 16'h3100);
        wait_qproj_done_and_compare(0, 0, 16'h3100);
        qproj_cases_seen = qproj_cases_seen + 1;

        send_qproj_cmd(1, 0, 16'h3101);
        wait_qproj_done_and_compare(1, proj_expected_saturation[1], 16'h3101);
        qproj_cases_seen = qproj_cases_seen + 1;
        if (proj_expected_saturation[1]) begin
            csr_write64(ACE2_CSR_ERROR_STATUS, 64'hffff_ffff_ffff_ffff);
        end

        send_qproj_cmd_full(0, 0, proj_case_rows[0], 16'd128, 16'd896,
                            PROJ_ACT_BASE, PROJ_WEIGHT_BASE, PROJ_OUT_BASE, PROJ_META_BASE, 16'h3200);
        wait_qproj_done_and_compare(0, 0, 16'h3200);
        kproj_cases_seen = kproj_cases_seen + 1;

        send_qproj_cmd_full(0, 0, proj_case_rows[0], 16'd128, 16'd896,
                            PROJ_ACT_BASE, PROJ_WEIGHT_BASE, PROJ_OUT_BASE, PROJ_META_BASE, 16'h3300);
        wait_qproj_done_and_compare(0, 0, 16'h3300);
        vproj_cases_seen = vproj_cases_seen + 1;

        send_rope_cmd_full(0, 0, 16'd64, rope_sequence_position[0],
                           ROPE_ACT_BASE, ROPE_TABLE_BASE, ROPE_OUT_BASE, ROPE_SCALE_BASE, 16'h3400);
        wait_rope_done_and_compare(0, 1, 16'h3400);
        descriptor_error_cases = descriptor_error_cases + 1;
        csr_read64(ACE2_CSR_ERROR_STATUS, read_value);
        if (!read_value[ACE2_ERR_DESCRIPTOR]) begin
            $display("ROPE_DESCRIPTOR_ERROR_BAD_N_NOT_STICKY status=%016x", read_value);
            failures = failures + 1;
        end
        csr_write64(ACE2_CSR_ERROR_STATUS, 64'hffff_ffff_ffff_ffff);

        send_rope_cmd_full(0, 0, 16'd896, 16'd32768,
                           ROPE_ACT_BASE, ROPE_TABLE_BASE, ROPE_OUT_BASE, ROPE_SCALE_BASE, 16'h3401);
        wait_rope_done_and_compare(0, 1, 16'h3401);
        descriptor_error_cases = descriptor_error_cases + 1;
        csr_read64(ACE2_CSR_ERROR_STATUS, read_value);
        if (!read_value[ACE2_ERR_DESCRIPTOR]) begin
            $display("ROPE_DESCRIPTOR_ERROR_BAD_POSITION_NOT_STICKY status=%016x", read_value);
            failures = failures + 1;
        end
        csr_write64(ACE2_CSR_ERROR_STATUS, 64'hffff_ffff_ffff_ffff);

        for (case_index = 0; case_index < ROPE_CASE_COUNT; case_index = case_index + 1) begin
            send_rope_cmd(case_index, 0, 16'h3500 + case_index);
            wait_rope_done_and_compare(case_index, rope_expected_saturation[case_index], 16'h3500 + case_index);
        end

        for (case_index = 0; case_index < ROPE_CASE_COUNT; case_index = case_index + 1) begin
            send_rope_k_cmd(case_index, 0, 16'h3600 + case_index);
            wait_rope_done_and_compare(case_index, rope_expected_saturation[case_index], 16'h3600 + case_index);
        end

        send_kv_write_cmd_full(0, 0, 16'd64, 16'd3,
                               KV_K_BASE, KV_V_BASE, KV_CACHE_BASE, KV_META_BASE, 16'h3700);
        wait_kv_write_done_and_compare(0, 1, 16'h3700);
        descriptor_error_cases = descriptor_error_cases + 1;
        csr_read64(ACE2_CSR_ERROR_STATUS, read_value);
        if (!read_value[ACE2_ERR_DESCRIPTOR]) begin
            $display("KV_WRITE_DESCRIPTOR_ERROR_BAD_N_NOT_STICKY status=%016x", read_value);
            failures = failures + 1;
        end
        csr_write64(ACE2_CSR_ERROR_STATUS, 64'hffff_ffff_ffff_ffff);

        send_kv_write_cmd_full(0, 0, 16'd128, 16'd32768,
                               KV_K_BASE, KV_V_BASE, KV_CACHE_BASE, KV_META_BASE, 16'h3701);
        wait_kv_write_done_and_compare(0, 1, 16'h3701);
        descriptor_error_cases = descriptor_error_cases + 1;
        csr_read64(ACE2_CSR_ERROR_STATUS, read_value);
        if (!read_value[ACE2_ERR_DESCRIPTOR]) begin
            $display("KV_WRITE_DESCRIPTOR_ERROR_BAD_POSITION_NOT_STICKY status=%016x", read_value);
            failures = failures + 1;
        end
        csr_write64(ACE2_CSR_ERROR_STATUS, 64'hffff_ffff_ffff_ffff);

        send_kv_write_cmd_full(0, 0, 16'd128, 16'd3,
                               KV_K_BASE + 64'd1, KV_V_BASE, KV_CACHE_BASE, KV_META_BASE, 16'h3702);
        wait_kv_write_done_and_compare(0, 1, 16'h3702);
        descriptor_error_cases = descriptor_error_cases + 1;
        csr_read64(ACE2_CSR_ERROR_STATUS, read_value);
        if (!read_value[ACE2_ERR_DESCRIPTOR]) begin
            $display("KV_WRITE_DESCRIPTOR_ERROR_SRC_ALIGN_NOT_STICKY status=%016x", read_value);
            failures = failures + 1;
        end
        csr_write64(ACE2_CSR_ERROR_STATUS, 64'hffff_ffff_ffff_ffff);

        send_kv_write_cmd_full(0, 0, 16'd128, 16'd3,
                               KV_K_BASE, KV_V_BASE, KV_CACHE_BASE + 64'd1, KV_META_BASE, 16'h3703);
        wait_kv_write_done_and_compare(0, 1, 16'h3703);
        descriptor_error_cases = descriptor_error_cases + 1;
        csr_read64(ACE2_CSR_ERROR_STATUS, read_value);
        if (!read_value[ACE2_ERR_DESCRIPTOR]) begin
            $display("KV_WRITE_DESCRIPTOR_ERROR_SCRATCH_ALIGN_NOT_STICKY status=%016x", read_value);
            failures = failures + 1;
        end
        csr_write64(ACE2_CSR_ERROR_STATUS, 64'hffff_ffff_ffff_ffff);

        send_kv_write_cmd_full(0, 0, 16'd128, 16'd3,
                               KV_K_BASE, KV_V_BASE, KV_CACHE_BASE, KV_META_BASE + 64'd1, 16'h3704);
        wait_kv_write_done_and_compare(0, 1, 16'h3704);
        descriptor_error_cases = descriptor_error_cases + 1;
        csr_read64(ACE2_CSR_ERROR_STATUS, read_value);
        if (!read_value[ACE2_ERR_DESCRIPTOR]) begin
            $display("KV_WRITE_DESCRIPTOR_ERROR_SCALE_ALIGN_NOT_STICKY status=%016x", read_value);
            failures = failures + 1;
        end
        csr_write64(ACE2_CSR_ERROR_STATUS, 64'hffff_ffff_ffff_ffff);

        for (case_index = 0; case_index < 3; case_index = case_index + 1) begin
            send_kv_write_cmd(case_index, 0, 16'(case_index * 257 + 3), 16'h3800 + case_index);
            wait_kv_write_done_and_compare(case_index, 0, 16'h3800 + case_index);
        end

        send_attn_score_cmd_full(0, 0, 16'd0, 16'd64,
                                 ATTN_Q_BASE, ATTN_K_BASE, ATTN_SCORE_OUT_BASE,
                                 ATTN_META_BASE, 16'h3900);
        wait_attn_score_done_and_compare(0, 1, 16'h3900);
        descriptor_error_cases = descriptor_error_cases + 1;
        csr_read64(ACE2_CSR_ERROR_STATUS, read_value);
        if (!read_value[ACE2_ERR_DESCRIPTOR]) begin
            $display("ATTN_SCORE_DESCRIPTOR_ERROR_BAD_N_NOT_STICKY status=%016x", read_value);
            failures = failures + 1;
        end
        csr_write64(ACE2_CSR_ERROR_STATUS, 64'hffff_ffff_ffff_ffff);

        send_attn_score_cmd_full(0, 0, 16'd1, 16'd32,
                                 ATTN_Q_BASE, ATTN_K_BASE, ATTN_SCORE_OUT_BASE,
                                 ATTN_META_BASE, 16'h3901);
        wait_attn_score_done_and_compare(0, 1, 16'h3901);
        descriptor_error_cases = descriptor_error_cases + 1;
        csr_read64(ACE2_CSR_ERROR_STATUS, read_value);
        if (!read_value[ACE2_ERR_DESCRIPTOR]) begin
            $display("ATTN_SCORE_DESCRIPTOR_ERROR_BAD_K_NOT_STICKY status=%016x", read_value);
            failures = failures + 1;
        end
        csr_write64(ACE2_CSR_ERROR_STATUS, 64'hffff_ffff_ffff_ffff);

        send_attn_score_cmd_full(0, 0, 16'd1, 16'd64,
                                 ATTN_Q_BASE + 64'd1, ATTN_K_BASE, ATTN_SCORE_OUT_BASE,
                                 ATTN_META_BASE, 16'h3902);
        wait_attn_score_done_and_compare(0, 1, 16'h3902);
        descriptor_error_cases = descriptor_error_cases + 1;
        csr_read64(ACE2_CSR_ERROR_STATUS, read_value);
        if (!read_value[ACE2_ERR_DESCRIPTOR]) begin
            $display("ATTN_SCORE_DESCRIPTOR_ERROR_SRC_ALIGN_NOT_STICKY status=%016x", read_value);
            failures = failures + 1;
        end
        csr_write64(ACE2_CSR_ERROR_STATUS, 64'hffff_ffff_ffff_ffff);

        send_attn_score_cmd_full(0, 0, 16'd1, 16'd64,
                                 ATTN_Q_BASE, ATTN_K_BASE, ATTN_SCORE_OUT_BASE,
                                 ATTN_META_BASE + 64'd1, 16'h3903);
        wait_attn_score_done_and_compare(0, 1, 16'h3903);
        descriptor_error_cases = descriptor_error_cases + 1;
        csr_read64(ACE2_CSR_ERROR_STATUS, read_value);
        if (!read_value[ACE2_ERR_DESCRIPTOR]) begin
            $display("ATTN_SCORE_DESCRIPTOR_ERROR_SCALE_ALIGN_NOT_STICKY status=%016x", read_value);
            failures = failures + 1;
        end
        csr_write64(ACE2_CSR_ERROR_STATUS, 64'hffff_ffff_ffff_ffff);

        run_attn_score_metadata_error_cases();
        for (case_index = 0; case_index < ATTN_SCORE_CASE_COUNT; case_index = case_index + 1) begin
            send_attn_score_cmd(case_index, 0, 16'h3a00 + case_index);
            wait_attn_score_done_and_compare(case_index, attn_score_expected_saturation[case_index], 16'h3a00 + case_index);
        end

        send_softmax_cmd_full(0, 0, 16'd0, 16'd0,
                              SOFTMAX_SCORE_BASE, SOFTMAX_OUT_BASE, 16'h3b00);
        wait_softmax_done_and_compare(0, 1, 16'h3b00);
        descriptor_error_cases = descriptor_error_cases + 1;
        csr_read64(ACE2_CSR_ERROR_STATUS, read_value);
        if (!read_value[ACE2_ERR_DESCRIPTOR]) begin
            $display("SOFTMAX_DESCRIPTOR_ERROR_BAD_N_ZERO_NOT_STICKY status=%016x", read_value);
            failures = failures + 1;
        end
        csr_write64(ACE2_CSR_ERROR_STATUS, 64'hffff_ffff_ffff_ffff);

        send_softmax_cmd_full(0, 0, 16'd9, 16'd0,
                              SOFTMAX_SCORE_BASE, SOFTMAX_OUT_BASE, 16'h3b01);
        wait_softmax_done_and_compare(0, 1, 16'h3b01);
        descriptor_error_cases = descriptor_error_cases + 1;
        csr_read64(ACE2_CSR_ERROR_STATUS, read_value);
        if (!read_value[ACE2_ERR_DESCRIPTOR]) begin
            $display("SOFTMAX_DESCRIPTOR_ERROR_BAD_N_WIDE_NOT_STICKY status=%016x", read_value);
            failures = failures + 1;
        end
        csr_write64(ACE2_CSR_ERROR_STATUS, 64'hffff_ffff_ffff_ffff);

        send_softmax_cmd_full(0, 0, softmax_context_count[0], 16'd64,
                              SOFTMAX_SCORE_BASE, SOFTMAX_OUT_BASE, 16'h3b02);
        wait_softmax_done_and_compare(0, 1, 16'h3b02);
        descriptor_error_cases = descriptor_error_cases + 1;
        csr_read64(ACE2_CSR_ERROR_STATUS, read_value);
        if (!read_value[ACE2_ERR_DESCRIPTOR]) begin
            $display("SOFTMAX_DESCRIPTOR_ERROR_BAD_K_NOT_STICKY status=%016x", read_value);
            failures = failures + 1;
        end
        csr_write64(ACE2_CSR_ERROR_STATUS, 64'hffff_ffff_ffff_ffff);

        send_softmax_cmd_full(0, 0, softmax_context_count[0], 16'd0,
                              SOFTMAX_SCORE_BASE + 64'd1, SOFTMAX_OUT_BASE, 16'h3b03);
        wait_softmax_done_and_compare(0, 1, 16'h3b03);
        descriptor_error_cases = descriptor_error_cases + 1;
        csr_read64(ACE2_CSR_ERROR_STATUS, read_value);
        if (!read_value[ACE2_ERR_DESCRIPTOR]) begin
            $display("SOFTMAX_DESCRIPTOR_ERROR_SRC_ALIGN_NOT_STICKY status=%016x", read_value);
            failures = failures + 1;
        end
        csr_write64(ACE2_CSR_ERROR_STATUS, 64'hffff_ffff_ffff_ffff);

        for (case_index = 0; case_index < SOFTMAX_CASE_COUNT; case_index = case_index + 1) begin
            send_softmax_cmd(case_index, 0, 16'h3c00 + case_index);
            wait_softmax_done_and_compare(case_index, 0, 16'h3c00 + case_index);
        end

        send_attn_value_cmd_full(0, 0, 16'd0, 16'd64,
                                 ATTN_VALUE_PROB_BASE, ATTN_VALUE_V_BASE,
                                 ATTN_VALUE_OUT_BASE, 16'h3d00);
        wait_attn_value_done_and_compare(0, 1, 16'h3d00);
        descriptor_error_cases = descriptor_error_cases + 1;
        csr_read64(ACE2_CSR_ERROR_STATUS, read_value);
        if (!read_value[ACE2_ERR_DESCRIPTOR]) begin
            $display("ATTN_VALUE_DESCRIPTOR_ERROR_BAD_N_NOT_STICKY status=%016x",
                     read_value);
            failures = failures + 1;
        end
        csr_write64(ACE2_CSR_ERROR_STATUS, 64'hffff_ffff_ffff_ffff);

        send_attn_value_cmd_full(0, 0, attn_value_context_count[0], 16'd32,
                                 ATTN_VALUE_PROB_BASE, ATTN_VALUE_V_BASE,
                                 ATTN_VALUE_OUT_BASE, 16'h3d01);
        wait_attn_value_done_and_compare(0, 1, 16'h3d01);
        descriptor_error_cases = descriptor_error_cases + 1;
        csr_read64(ACE2_CSR_ERROR_STATUS, read_value);
        if (!read_value[ACE2_ERR_DESCRIPTOR]) begin
            $display("ATTN_VALUE_DESCRIPTOR_ERROR_BAD_K_NOT_STICKY status=%016x",
                     read_value);
            failures = failures + 1;
        end
        csr_write64(ACE2_CSR_ERROR_STATUS, 64'hffff_ffff_ffff_ffff);

        send_attn_value_cmd_full(0, 0, attn_value_context_count[0], 16'd64,
                                 ATTN_VALUE_PROB_BASE + 64'd1, ATTN_VALUE_V_BASE,
                                 ATTN_VALUE_OUT_BASE, 16'h3d02);
        wait_attn_value_done_and_compare(0, 1, 16'h3d02);
        descriptor_error_cases = descriptor_error_cases + 1;
        csr_read64(ACE2_CSR_ERROR_STATUS, read_value);
        if (!read_value[ACE2_ERR_DESCRIPTOR]) begin
            $display("ATTN_VALUE_DESCRIPTOR_ERROR_SRC_ALIGN_NOT_STICKY status=%016x",
                     read_value);
            failures = failures + 1;
        end
        csr_write64(ACE2_CSR_ERROR_STATUS, 64'hffff_ffff_ffff_ffff);

        for (case_index = 0; case_index < ATTN_VALUE_CASE_COUNT; case_index = case_index + 1) begin
            send_attn_value_cmd(case_index, 0, 16'h3e00 + case_index);
            wait_attn_value_done_and_compare(
                case_index,
                attn_value_expected_saturation[case_index],
                16'h3e00 + case_index
            );
            if (attn_value_expected_saturation[case_index]) begin
                csr_write64(ACE2_CSR_ERROR_STATUS, 64'hffff_ffff_ffff_ffff);
            end
        end
        run_attn_compose_reset_reuse();

        run_oproj_cases();
        run_vector_family_cases();
        run_mlp_projection_cases();
        send_silu_gate_cmd(0, 16'd0, 16'h54f0);
        wait_silu_gate_done_and_compare(0, 1, 16'h54f0);
        descriptor_error_cases = descriptor_error_cases + 1;
        csr_write64(ACE2_CSR_ERROR_STATUS, 64'hffff_ffff_ffff_ffff);
        send_silu_gate_cmd(0, 16'd4865, 16'h54f1);
        wait_silu_gate_done_and_compare(0, 1, 16'h54f1);
        descriptor_error_cases = descriptor_error_cases + 1;
        csr_write64(ACE2_CSR_ERROR_STATUS, 64'hffff_ffff_ffff_ffff);
        run_silu_gate_cases();
        send_qproj_cmd_full(PROJ_CASE_MLP_DOWN, 0, proj_case_rows[PROJ_CASE_MLP_DOWN],
                            16'd128, proj_case_k[PROJ_CASE_MLP_DOWN],
                            MLP_DOWN_PROJ_ACT_BASE, MLP_DOWN_PROJ_WEIGHT_BASE,
                            MLP_DOWN_PROJ_OUT_BASE, MLP_DOWN_PROJ_META_BASE, 16'h54f2);
        wait_qproj_done_and_compare(PROJ_CASE_MLP_DOWN, 1, 16'h54f2);
        descriptor_error_cases = descriptor_error_cases + 1;
        csr_write64(ACE2_CSR_ERROR_STATUS, 64'hffff_ffff_ffff_ffff);
        run_mlp_down_projection_case();
        run_mlp_residual_cases();
        run_lm_head();
        descriptor_error_cases = descriptor_error_cases + 4;

        if (sram_req_valid !== 8'd0 || sram_write !== 8'd0) begin
            $display("SRAM_INTERFACE_UNEXPECTED_ACTIVITY req=%02x wr=%02x", sram_req_valid, sram_write);
            failures = failures + 1;
        end

        if (failures != 0) begin
            $display("ACE2_SHELL_TB_FAIL failures=%0d", failures);
            $fatal(1, "ACE2_SHELL_TB_FAIL");
        end
        report_oproj_result();
        $display("ACE2_SHELL_CYCLES success_runs=%0d max_cycles=%0d total_cycles=%0d last_cycles=%0d",
                 success_runs, max_command_cycles, total_success_cycles, last_command_cycles);
        $display("ACE2_SHELL_TB_PASS layers=24 repeated_layer=7 generated_cases=%0d qproj_cases=%0d kproj_cases=%0d vproj_cases=%0d oproj_cases=%0d mlp_gate_proj_cases=%0d mlp_up_proj_cases=%0d mlp_down_proj_cases=%0d mlp_residual_cases=%0d silu_gate_cases=%0d rope_q_cases=%0d rope_k_cases=%0d kv_write_cases=%0d attn_score_cases=%0d attn_score_metadata_errors=%0d softmax_cases=%0d attn_value_cases=%0d attn_compose_cases=%0d residual_cases=%0d post_rms_cases=%0d descriptor_errors=%0d memory_errors=%0d watchdog_errors=%0d reset_busy=%0d response_protocol_cases=%0d",
                 TEST_COUNT, qproj_cases_seen, kproj_cases_seen, vproj_cases_seen, oproj_cases_seen, mlp_gate_proj_cases_seen, mlp_up_proj_cases_seen, mlp_down_proj_cases_seen, mlp_residual_cases_seen, silu_gate_cases_seen, rope_q_cases_seen, rope_k_cases_seen, kv_write_cases_seen, attn_score_cases_seen, attn_score_metadata_error_cases, softmax_cases_seen, attn_value_cases_seen, attn_compose_cases_seen, residual_cases_seen, post_rms_cases_seen, descriptor_error_cases, memory_error_cases, watchdog_error_cases, reset_busy_cases, response_protocol_cases);
        $finish;
    end
endmodule

`default_nettype wire
