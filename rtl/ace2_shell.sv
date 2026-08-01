`default_nettype none

import ace2_pkg::ACE2_CSR_CAPABILITIES;
import ace2_pkg::ACE2_CSR_CONTROL;
import ace2_pkg::ACE2_CSR_DESC_BASE;
import ace2_pkg::ACE2_CSR_DESC_HEAD;
import ace2_pkg::ACE2_CSR_DESC_TAIL;
import ace2_pkg::ACE2_CSR_DOORBELL;
import ace2_pkg::ACE2_CSR_ERROR_STATUS;
import ace2_pkg::ACE2_CSR_ID;
import ace2_pkg::ACE2_CSR_INTERRUPT_EN;
import ace2_pkg::ACE2_CSR_INTERRUPT_ST;
import ace2_pkg::ACE2_CSR_PERF_BYTE;
import ace2_pkg::ACE2_CSR_PERF_CYCLE;
import ace2_pkg::ACE2_CSR_PERF_STALL;
import ace2_pkg::ACE2_CSR_PERF_TOKEN;
import ace2_pkg::ACE2_CSR_STATUS;
import ace2_pkg::ACE2_CSR_VERSION;
import ace2_pkg::ACE2_CSR_WATCHDOG_LIMIT;
import ace2_pkg::ACE2_ERR_DESCRIPTOR;
import ace2_pkg::ACE2_ERR_MEMORY;
import ace2_pkg::ACE2_ERR_NUMERIC;
import ace2_pkg::ACE2_ERR_RESET_BUSY;
import ace2_pkg::ACE2_ERR_RESERVED;
import ace2_pkg::ACE2_ERR_WATCHDOG;
import ace2_pkg::ACE2_HIDDEN_SIZE;
import ace2_pkg::ACE2_LM_HEAD_TILE_SIZE;
import ace2_pkg::ACE2_OPCODE_ATTN_SCORE;
import ace2_pkg::ACE2_OPCODE_ATTN_VALUE;
import ace2_pkg::ACE2_OPCODE_ATTN_COMPOSE;
import ace2_pkg::ACE2_OPCODE_SOFTMAX;
import ace2_pkg::ACE2_OPCODE_SILU_GATE;
import ace2_pkg::ACE2_OPCODE_ROPE;
import ace2_pkg::ACE2_OPCODE_KV_WRITE;
import ace2_pkg::ACE2_OPCODE_W4A8_PROJ;
import ace2_pkg::ACE2_OPCODE_RMSNORM;
import ace2_pkg::ACE2_OPCODE_RESIDUAL_ADD;
import ace2_pkg::ACE2_SRAM_ADDR_WIDTH;
import ace2_pkg::ACE2_SRAM_BANKS;
import ace2_pkg::ACE2_VECTOR_LANES;

module ace2_shell #(
    parameter integer HIDDEN_SIZE = ACE2_HIDDEN_SIZE,
    parameter integer LANES = ACE2_VECTOR_LANES,
    parameter integer ACT_WIDTH = 8,
    parameter integer GAIN_WIDTH = 16,
    parameter integer ACC_WIDTH = 48,
    parameter integer INV_RMS_FRAC = 30,
    parameter integer GAIN_FRAC = 8,
    parameter integer PROJ_MAC_LANES = 4,
    parameter integer PROJ_ACC_WIDTH = 32,
    parameter integer PROJ_M_MAX = 16,
    parameter integer SRAM_BANKS = ACE2_SRAM_BANKS,
    parameter integer SRAM_ADDR_WIDTH = ACE2_SRAM_ADDR_WIDTH
) (
    input  wire                              clk_i,
    input  wire                              rst_ni,

    input  wire                              csr_valid_i,
    output wire                              csr_ready_o,
    input  wire                              csr_write_i,
    input  wire [31:0]                       csr_addr_i,
    input  wire [63:0]                       csr_wdata_i,
    input  wire [7:0]                        csr_wstrb_i,
    output reg                               csr_rvalid_o,
    input  wire                              csr_rready_i,
    output reg  [63:0]                       csr_rdata_o,
    output reg                               csr_error_o,
    output wire                              irq_o,

    input  wire                              cmd_valid_i,
    output wire                              cmd_ready_o,
    input  wire [7:0]                        cmd_opcode_i,
    input  wire [7:0]                        cmd_flags_i,
    input  wire [7:0]                        cmd_layer_id_i,
    input  wire [15:0]                       cmd_m_i,
    input  wire [15:0]                       cmd_n_i,
    input  wire [15:0]                       cmd_k_i,
    input  wire [15:0]                       cmd_sequence_position_i,
    input  wire [15:0]                       cmd_completion_tag_i,
    input  wire [63:0]                       cmd_src0_addr_i,
    input  wire [63:0]                       cmd_src1_addr_i,
    input  wire [63:0]                       cmd_dst_addr_i,
    input  wire [63:0]                       cmd_scale_addr_i,
    input  wire [63:0]                       cmd_scratch_addr_i,

    output wire                              mem_req_valid_o,
    input  wire                              mem_req_ready_i,
    output wire                              mem_req_write_o,
    output wire [63:0]                       mem_req_addr_o,
    output wire [15:0]                       mem_req_len_o,
    output wire [7:0]                        mem_req_tag_o,
    output wire                              mem_wvalid_o,
    input  wire                              mem_wready_i,
    output wire [LANES*ACT_WIDTH-1:0]        mem_wdata_o,
    output wire [15:0]                       mem_wstrb_o,
    output wire [7:0]                        mem_wtag_o,
    input  wire                              mem_rvalid_i,
    output wire                              mem_rready_o,
    input  wire [LANES*ACT_WIDTH-1:0]        mem_rdata_i,
    input  wire [7:0]                        mem_rtag_i,
    input  wire                              mem_rerror_i,
    input  wire                              mem_bvalid_i,
    output wire                              mem_bready_o,
    input  wire [7:0]                        mem_btag_i,
    input  wire                              mem_berror_i,

    output wire [SRAM_BANKS-1:0]             sram_req_valid_o,
    input  wire [SRAM_BANKS-1:0]             sram_req_ready_i,
    output wire [SRAM_BANKS-1:0]             sram_write_o,
    output wire [SRAM_BANKS*SRAM_ADDR_WIDTH-1:0] sram_addr_o,
    output wire [SRAM_BANKS*LANES*ACT_WIDTH-1:0] sram_wdata_o,
    output wire [SRAM_BANKS*16-1:0]          sram_wstrb_o,
    input  wire [SRAM_BANKS*LANES*ACT_WIDTH-1:0] sram_rdata_i,
    input  wire [SRAM_BANKS-1:0]             sram_rvalid_i,

    output wire                              busy_o,
    output wire                              cmd_done_valid_o,
    input  wire                              cmd_done_ready_i,
    output wire [15:0]                       cmd_done_tag_o,
    output wire                              cmd_done_error_o,
    output wire [ACC_WIDTH-1:0]              cmd_done_sumsq_o,
    output wire [INV_RMS_FRAC+1:0]           cmd_done_inv_rms_q30_o,
    output wire                              cmd_done_saturation_seen_o
);
    localparam integer BEATS = HIDDEN_SIZE / LANES;
    localparam integer BEAT_INDEX_WIDTH = (BEATS <= 1) ? 1 : $clog2(BEATS + 1);
    localparam integer MLP_INTERMEDIATE_SIZE = 4864;
    localparam integer PROJ_HIDDEN_GROUPS = HIDDEN_SIZE / PROJ_MAC_LANES;
    localparam integer PROJ_MLP_GROUPS = MLP_INTERMEDIATE_SIZE / PROJ_MAC_LANES;
    localparam integer PROJ_GROUPS = PROJ_MLP_GROUPS;
    localparam integer PROJ_GROUPS_PER_STORAGE_BEAT = LANES / PROJ_MAC_LANES;
    localparam integer PROJ_GROUP_STORAGE_SHIFT = (PROJ_GROUPS_PER_STORAGE_BEAT <= 1) ? 0 : $clog2(PROJ_GROUPS_PER_STORAGE_BEAT);
    localparam integer PROJ_GROUP_SELECT_WIDTH = (PROJ_GROUP_STORAGE_SHIFT == 0) ? 1 : PROJ_GROUP_STORAGE_SHIFT;
    localparam integer PROJ_GROUP_INDEX_WIDTH = (PROJ_GROUPS <= 1) ? 1 : $clog2(PROJ_GROUPS + 1);
    localparam integer PROJ_OUTPUT_MAX = (MLP_INTERMEDIATE_SIZE > HIDDEN_SIZE) ? MLP_INTERMEDIATE_SIZE : HIDDEN_SIZE;
    localparam integer PROJ_OUT_INDEX_WIDTH = (PROJ_OUTPUT_MAX <= 1) ? 1 : $clog2(PROJ_OUTPUT_MAX + 1);
    localparam integer PROJ_ROW_INDEX_WIDTH = (PROJ_M_MAX <= 1) ? 1 : $clog2(PROJ_M_MAX + 1);
    localparam integer PROJ_HIDDEN_WEIGHT_BYTES_PER_OUTPUT = (HIDDEN_SIZE / LANES) * 16;
    localparam integer PROJ_MLP_WEIGHT_BYTES_PER_OUTPUT = (MLP_INTERMEDIATE_SIZE / LANES) * 16;
    localparam integer PROJ_WEIGHT_OFFSET_MAX = HIDDEN_SIZE * MLP_INTERMEDIATE_SIZE;
    localparam integer PROJ_WEIGHT_OFFSET_WIDTH = (PROJ_WEIGHT_OFFSET_MAX <= 1) ? 1 : $clog2(PROJ_WEIGHT_OFFSET_MAX + PROJ_MLP_WEIGHT_BYTES_PER_OUTPUT + 1);
    localparam integer PROJ_LINEAR_OFFSET_MAX = PROJ_M_MAX * PROJ_OUTPUT_MAX;
    localparam integer PROJ_LINEAR_OFFSET_WIDTH = (PROJ_LINEAR_OFFSET_MAX <= 1) ? 1 : $clog2(PROJ_LINEAR_OFFSET_MAX + 1);
    localparam integer ROPE_LANES = LANES / 8;
    localparam integer ROPE_SEGMENTS = LANES / ROPE_LANES;
    localparam integer ROPE_SEGMENT_INDEX_WIDTH = (ROPE_SEGMENTS <= 1) ? 1 : $clog2(ROPE_SEGMENTS);
    localparam integer ROPE_AUX_GROUPS_PER_MEM_BEAT = LANES / (2 * ROPE_LANES);
    localparam integer ROPE_AUX_SELECT_WIDTH = (ROPE_AUX_GROUPS_PER_MEM_BEAT <= 1) ? 1 : $clog2(ROPE_AUX_GROUPS_PER_MEM_BEAT);
    localparam integer ATTN_HEAD_DIM = 64;
    localparam integer ATTN_CONTEXT_MAX = 8;
    localparam integer ATTN_MAC_LANES = 1;
    localparam integer SILU_LANES = LANES / 2;
    localparam integer PAYLOAD_BANK_WIDTH = 2 * ACT_WIDTH;
    localparam integer PAYLOAD_BANKS = LANES / 2;
    localparam integer PAYLOAD_BANK_INDEX_WIDTH =
        (PAYLOAD_BANKS <= 1) ? 1 : $clog2(PAYLOAD_BANKS);
    localparam integer SILU_PAYLOAD_BANKS = 64 / PAYLOAD_BANK_WIDTH;
    localparam integer SILU_INPUT_BANKS = 128 / PAYLOAD_BANK_WIDTH;
    localparam integer SILU_INPUT_BEATS_MAX = (MLP_INTERMEDIATE_SIZE + SILU_LANES - 1) / SILU_LANES;
    localparam integer SILU_INPUT_BEAT_WIDTH = $clog2(SILU_INPUT_BEATS_MAX + 1);
    localparam integer ATTN_GROUPS = ATTN_HEAD_DIM / ATTN_MAC_LANES;
    localparam integer ATTN_GROUPS_PER_MEM_BEAT = LANES / ATTN_MAC_LANES;
    localparam integer ATTN_GROUP_STORAGE_SHIFT = (ATTN_GROUPS_PER_MEM_BEAT <= 1) ? 0 : $clog2(ATTN_GROUPS_PER_MEM_BEAT);
    localparam integer ATTN_GROUP_SELECT_WIDTH = (ATTN_GROUP_STORAGE_SHIFT == 0) ? 1 : ATTN_GROUP_STORAGE_SHIFT;
    localparam integer ATTN_GROUP_INDEX_WIDTH = (ATTN_GROUPS <= 1) ? 1 : $clog2(ATTN_GROUPS + 1);
    localparam integer ATTN_TOKEN_INDEX_WIDTH = (ATTN_CONTEXT_MAX <= 1) ? 1 : $clog2(ATTN_CONTEXT_MAX);
    localparam integer CONTROL_BITS = 5;
    localparam integer STATUS_BITS = 6;
    localparam [15:0] ROPE_MAX_SEQUENCE_POSITION = 16'd32767;
    localparam [15:0] ROPE_K_SIZE_16 = 16'd128;
    // Q/K projection weights are fused with a shared +22.5 degree rotation
    // for each RoPE pair. Commands encode the basis in half-degree units.
    localparam [7:0] ROPE_BASIS_ROTATION_HALF_DEGREES = 8'd45;
    localparam [BEAT_INDEX_WIDTH-1:0] ROPE_K_LAST_BEAT = BEAT_INDEX_WIDTH'(8 - 1);
    localparam [BEAT_INDEX_WIDTH-1:0] KV_LAST_DATA_BEAT = BEAT_INDEX_WIDTH'(8 - 1);
    localparam [ROPE_SEGMENT_INDEX_WIDTH-1:0] ROPE_LAST_SEGMENT = ROPE_SEGMENT_INDEX_WIDTH'(ROPE_SEGMENTS - 1);
    localparam [ATTN_GROUP_INDEX_WIDTH-1:0] ATTN_LAST_GROUP = ATTN_GROUP_INDEX_WIDTH'(ATTN_GROUPS - 1);
    localparam [15:0] ATTN_HEAD_DIM_16 = 16'(ATTN_HEAD_DIM);
    localparam [15:0] ATTN_CONTEXT_MAX_16 = 16'(ATTN_CONTEXT_MAX);
    localparam [1:0] KV_PHASE_K = 2'd0;
    localparam [1:0] KV_PHASE_V = 2'd1;
    localparam [1:0] KV_PHASE_META = 2'd2;
    localparam [BEAT_INDEX_WIDTH-1:0] LAST_BEAT = BEAT_INDEX_WIDTH'(BEATS - 1);
    localparam [PROJ_GROUP_INDEX_WIDTH-1:0] PROJ_HIDDEN_LAST_GROUP = PROJ_GROUP_INDEX_WIDTH'(PROJ_HIDDEN_GROUPS - 1);
    localparam [PROJ_GROUP_INDEX_WIDTH-1:0] PROJ_MLP_LAST_GROUP = PROJ_GROUP_INDEX_WIDTH'(PROJ_MLP_GROUPS - 1);
    localparam [3:0] PROJ_LAST_PACK_LANE = 4'd15;
    localparam [15:0] HIDDEN_SIZE_16 = 16'(HIDDEN_SIZE);
    localparam [15:0] LM_HEAD_TILE_SIZE_16 = 16'(ACE2_LM_HEAD_TILE_SIZE);
    localparam [15:0] MLP_INTERMEDIATE_SIZE_16 = 16'(MLP_INTERMEDIATE_SIZE);
    localparam [63:0] ACE2_ID_VALUE = 64'h4143453200000001;
    localparam [63:0] ACE2_VERSION_VALUE = 64'h0000000000000001;
    localparam [63:0] ACE2_CAP_VALUE = 64'h0000000000081091;

    localparam [6:0] ST_IDLE       = 7'd0;
    localparam [6:0] ST_START      = 7'd1;
    localparam [6:0] ST_ACT_REQ    = 7'd2;
    localparam [6:0] ST_ACT_RECV   = 7'd3;
    localparam [6:0] ST_SCALE_ACT_REQ  = 7'd4;
    localparam [6:0] ST_SCALE_ACT_RECV = 7'd5;
    localparam [6:0] ST_GAIN_REQ0  = 7'd6;
    localparam [6:0] ST_GAIN_RECV0 = 7'd7;
    localparam [6:0] ST_GAIN_REQ1  = 7'd8;
    localparam [6:0] ST_GAIN_RECV1 = 7'd9;
    localparam [6:0] ST_CORE_WAIT  = 7'd10;
    localparam [6:0] ST_CORE_FEED  = 7'd11;
    localparam [6:0] ST_WRITE_REQ  = 7'd12;
    localparam [6:0] ST_WRITE_DATA = 7'd13;
    localparam [6:0] ST_WAIT_DONE  = 7'd14;
    localparam [6:0] ST_COMPLETE   = 7'd15;
    localparam [6:0] ST_PROJ_START = 7'd16;
    localparam [6:0] ST_PROJ_ACT0_REQ = 7'd17;
    localparam [6:0] ST_PROJ_ACT0_RECV = 7'd18;
    localparam [6:0] ST_PROJ_ACT1_REQ = 7'd19;
    localparam [6:0] ST_PROJ_ACT1_RECV = 7'd20;
    localparam [6:0] ST_PROJ_WGT_REQ = 7'd21;
    localparam [6:0] ST_PROJ_WGT_RECV = 7'd22;
    localparam [6:0] ST_PROJ_FEED = 7'd23;
    localparam [6:0] ST_PROJ_META_REQ = 7'd24;
    localparam [6:0] ST_PROJ_META_RECV = 7'd25;
    localparam [6:0] ST_PROJ_WAIT_OUT = 7'd26;
    localparam [6:0] ST_PROJ_WRITE_REQ = 7'd27;
    localparam [6:0] ST_PROJ_WRITE_DATA = 7'd28;
    localparam [6:0] ST_CMD_DISPATCH = 7'd29;
    localparam [6:0] ST_PROJ_META_FEED = 7'd30;
    localparam [6:0] ST_ROPE_START = 7'd31;
    localparam [6:0] ST_ROPE_ACT_REQ = 7'd32;
    localparam [6:0] ST_ROPE_ACT_RECV = 7'd33;
    localparam [6:0] ST_ROPE_PAIR_REQ = 7'd34;
    localparam [6:0] ST_ROPE_PAIR_RECV = 7'd35;
    localparam [6:0] ST_ROPE_SCALE0_REQ = 7'd36;
    localparam [6:0] ST_ROPE_SCALE0_RECV = 7'd37;
    localparam [6:0] ST_ROPE_TABLE_ADDR = 7'd38;
    localparam [6:0] ST_ROPE_SCALE_ADDR = 7'd39;
    localparam [6:0] ST_ROPE_PAIR_SCALE0_REQ = 7'd40;
    localparam [6:0] ST_ROPE_PAIR_SCALE0_RECV = 7'd41;
    localparam [6:0] ST_ROPE_PAIR_SCALE_ADDR = 7'd42;
    localparam [6:0] ST_ROPE_COS0_REQ = 7'd44;
    localparam [6:0] ST_ROPE_COS0_RECV = 7'd45;
    localparam [6:0] ST_ROPE_SIN0_REQ = 7'd48;
    localparam [6:0] ST_ROPE_SIN0_RECV = 7'd49;
    localparam [6:0] ST_ROPE_FEED = 7'd52;
    localparam [6:0] ST_ROPE_WAIT_OUT = 7'd53;
    localparam [6:0] ST_ROPE_WRITE_REQ = 7'd54;
    localparam [6:0] ST_ROPE_WRITE_DATA = 7'd55;
    localparam [6:0] ST_KV_READ_REQ = 7'd56;
    localparam [6:0] ST_KV_READ_RECV = 7'd57;
    localparam [6:0] ST_KV_WRITE_REQ = 7'd58;
    localparam [6:0] ST_KV_WRITE_DATA = 7'd59;
    localparam [6:0] ST_KV_PREP_LOW = 7'd60;
    localparam [6:0] ST_KV_PREP_HIGH = 7'd61;
    localparam [6:0] ST_ATTN_START = 7'd62;
    localparam [6:0] ST_ATTN_Q_REQ = 7'd63;
    localparam [6:0] ST_ATTN_Q_RECV = 7'd64;
    localparam [6:0] ST_ATTN_K_REQ = 7'd65;
    localparam [6:0] ST_ATTN_K_RECV = 7'd66;
    localparam [6:0] ST_ATTN_FEED = 7'd67;
    localparam [6:0] ST_ATTN_WAIT_OUT = 7'd68;
    localparam [6:0] ST_ATTN_WRITE_REQ = 7'd69;
    localparam [6:0] ST_ATTN_WRITE_DATA = 7'd70;
    localparam [6:0] ST_SOFTMAX_SCORE_REQ = 7'd71;
    localparam [6:0] ST_SOFTMAX_SCORE_RECV = 7'd72;
    localparam [6:0] ST_SOFTMAX_START = 7'd73;
    localparam [6:0] ST_SOFTMAX_WAIT_OUT = 7'd74;
    localparam [6:0] ST_SOFTMAX_WRITE_REQ = 7'd75;
    localparam [6:0] ST_SOFTMAX_WRITE_DATA = 7'd76;
    localparam [6:0] ST_AV_PROB_REQ = 7'd77;
    localparam [6:0] ST_AV_PROB_RECV = 7'd78;
    localparam [6:0] ST_AV_START = 7'd79;
    localparam [6:0] ST_AV_V_REQ = 7'd80;
    localparam [6:0] ST_AV_V_RECV = 7'd81;
    localparam [6:0] ST_AV_PRODUCT = 7'd82;
    localparam [6:0] ST_AV_ACCUM = 7'd83;
    localparam [6:0] ST_AV_ROUND = 7'd84;
    localparam [6:0] ST_AV_WRITE_REQ = 7'd85;
    localparam [6:0] ST_AV_WRITE_DATA = 7'd86;
    localparam [6:0] ST_WRITE_RESP = 7'd87;
    localparam [6:0] ST_WRITE_RETIRE = 7'd88;
    localparam [6:0] ST_RES_SRC0_REQ = 7'd89;
    localparam [6:0] ST_RES_SRC0_RECV = 7'd90;
    localparam [6:0] ST_RES_SRC1_REQ = 7'd91;
    localparam [6:0] ST_RES_SRC1_RECV = 7'd92;
    localparam [6:0] ST_RES_WRITE_REQ = 7'd93;
    localparam [6:0] ST_RES_WRITE_DATA = 7'd94;
    localparam [6:0] ST_SILU_META_REQ = 7'd95;
    localparam [6:0] ST_SILU_META_RECV = 7'd96;
    localparam [6:0] ST_SILU_START = 7'd97;
    localparam [6:0] ST_SILU_GATE_REQ = 7'd98;
    localparam [6:0] ST_SILU_GATE_RECV = 7'd99;
    localparam [6:0] ST_SILU_UP_REQ = 7'd100;
    localparam [6:0] ST_SILU_UP_RECV = 7'd101;
    localparam [6:0] ST_SILU_FEED = 7'd102;
    localparam [6:0] ST_SILU_WAIT = 7'd103;
    localparam [6:0] ST_SILU_WRITE_REQ = 7'd104;
    localparam [6:0] ST_SILU_WRITE_DATA = 7'd105;
    localparam [6:0] ST_COMPOSE_SCORE_REQ = 7'd106;
    localparam [6:0] ST_COMPOSE_SCORE_RECV = 7'd107;
    localparam [6:0] ST_COMPOSE_START = 7'd108;
    localparam [6:0] ST_COMPOSE_WAIT = 7'd109;
    localparam [6:0] ST_COMPOSE_VALUE_REQ = 7'd110;
    localparam [6:0] ST_COMPOSE_VALUE_RECV = 7'd111;
    localparam [6:0] ST_COMPOSE_VALUE_FEED = 7'd112;
    localparam [6:0] ST_COMPOSE_OUT_REQ = 7'd113;
    localparam [6:0] ST_COMPOSE_OUT_DATA = 7'd114;
    localparam [6:0] ST_ATTN_META_REQ = 7'd115;
    localparam [6:0] ST_ATTN_META_RECV = 7'd116;
    localparam [6:0] ST_ATTN_SCALE = 7'd117;
    localparam [6:0] ST_ATTN_SHIFT_INIT = 7'd118;
    localparam [6:0] ST_ATTN_SHIFT = 7'd119;
    localparam [6:0] ST_ATTN_MUL = 7'd120;
    localparam [6:0] ST_ATTN_CENTER = 7'd121;
    localparam [6:0] ST_ATTN_CENTER_CALC = 7'd122;
    localparam [6:0] ST_ATTN_CENTER_STORE = 7'd123;
    localparam [6:0] ST_ATTN_ROUND = 7'd124;
    localparam [6:0] ST_ATTN_CENTER_QUANT = 7'd125;
    localparam [3:0] OP_KIND_RMSNORM = 4'd0;
    localparam [3:0] OP_KIND_PROJ    = 4'd1;
    localparam [3:0] OP_KIND_ROPE    = 4'd2;
    localparam [3:0] OP_KIND_KV      = 4'd3;
    localparam [3:0] OP_KIND_ATTN    = 4'd4;
    localparam [3:0] OP_KIND_SOFTMAX = 4'd5;
    localparam [3:0] OP_KIND_ATTN_VALUE = 4'd6;
    localparam [3:0] OP_KIND_RESIDUAL = 4'd7;
    localparam [3:0] OP_KIND_SILU = 4'd8;
    localparam [3:0] OP_KIND_ATTN_COMPOSE = 4'd9;

    reg [6:0] state_low_q;
    reg silu_active_q;
    reg [6:0] state_d;
    wire [6:0] state_q = state_low_q;
    wire [6:0] prefix_state_q;
    reg [CONTROL_BITS-1:0] control_q;
    reg [STATUS_BITS-1:0] interrupt_enable_q;
    reg [STATUS_BITS-1:0] interrupt_status_q;
    reg [STATUS_BITS-1:0] error_status_q;
    reg [63:0] desc_base_q;
    reg [63:0] desc_head_q;
    reg [63:0] desc_tail_q;
    reg desc_tail_inc_q;
    reg [63:0] watchdog_limit_q;
    reg [63:0] watchdog_count_q;
    reg watchdog_armed_q;
    reg watchdog_fire_q;
    reg [15:0] cmd_completion_tag_buf_q;
    reg [7:0] cmd_opcode_buf_q;
    reg [7:0] cmd_flags_buf_q;
    reg [7:0] cmd_layer_id_buf_q;
    reg [15:0] cmd_m_buf_q;
    reg [15:0] cmd_n_buf_q;
    reg [15:0] cmd_k_buf_q;
    reg [15:0] cmd_sequence_position_buf_q;
    reg [63:0] cmd_src0_addr_buf_q;
    reg [63:0] cmd_src1_addr_buf_q;
    reg [63:0] cmd_dst_addr_buf_q;
    reg [63:0] cmd_scale_addr_buf_q;
    reg [63:0] cmd_scratch_addr_buf_q;
    reg [63:0] perf_cycle_q;
    reg [63:0] perf_byte_q;
    reg [63:0] perf_token_q;
    reg [63:0] perf_stall_q;
    reg perf_cycle_event_q;
    reg perf_byte_event_q;
    reg perf_stall_event_q;
    reg csr_req_valid_q;
    reg csr_req_write_q;
    reg csr_req_high_zero_q;
    reg [7:0] csr_req_addr_low_q;
    reg [63:0] csr_req_wdata_q;
    reg [7:0] csr_req_wstrb_q;
    reg [BEAT_INDEX_WIDTH-1:0] beat_idx_q;
    reg [BEAT_INDEX_WIDTH-1:0] out_idx_q;
    reg [15:0] completion_tag_q;
    reg [3:0] op_kind_q;
    reg [7:0] flags_q;
    reg [15:0] m_q;
    reg [15:0] n_q;
    reg [15:0] k_q;
    reg proj_mlp_shape_q;
    reg [15:0] sequence_position_q;
    reg [63:0] src0_addr_q;
    reg [63:0] src1_addr_q;
    reg [63:0] dst_addr_q;
    reg [63:0] scale_addr_q;
    reg [63:0] scratch_addr_q;
    reg [1:0] kv_phase_q;
    reg [31:0] kv_dst_low_q;
    reg kv_dst_carry_q;
    reg [63:0] kv_src_addr_q;
    reg [63:0] kv_dst_addr_q;
    reg kv_dst_load_base_q;
    reg kv_dst_advance_q;
    reg [63:0] mem_req_addr_q;
    reg [63:0] prefix_mem_req_addr_q;
    reg [63:0] proj_mem_req_addr_q;
    reg [63:0] silu_mem_req_addr_q;
    reg mem_rready_q;
    reg read_outstanding_q;
    reg write_data_pending_q;
    reg write_response_outstanding_q;
    reg reset_drain_q;
    reg [LANES*ACT_WIDTH-1:0] reset_write_data_q;
    reg [7:0] reset_write_tag_q;
    reg [15:0] reset_write_strb_q;
    reg descriptor_valid_q;
    wire [LANES*ACT_WIDTH-1:0] shared_payload_q;
    reg [LANES*ACT_WIDTH-1:0] gain_low_q;
    reg [LANES*ACT_WIDTH-1:0] gain_high_q;
    reg [PROJ_ROW_INDEX_WIDTH-1:0] proj_row_idx_q;
    reg [PROJ_OUT_INDEX_WIDTH-1:0] proj_out_idx_q;
    reg [PROJ_GROUP_INDEX_WIDTH-1:0] proj_group_idx_q;
    reg [3:0] proj_pack_lane_q;
    reg [PROJ_MAC_LANES*ACT_WIDTH-1:0] proj_act_low_q;
    reg [PROJ_MAC_LANES*4-1:0] proj_weight_q;
    reg [PROJ_WEIGHT_OFFSET_WIDTH-1:0] proj_weight_offset_q;
    reg [PROJ_LINEAR_OFFSET_WIDTH-1:0] proj_meta_offset_q;
    reg [PROJ_LINEAR_OFFSET_WIDTH-1:0] proj_out_write_offset_q;
    reg [31:0] proj_multiplier_q;
    reg [5:0] proj_right_shift_q;
    reg signed [ACT_WIDTH-1:0] proj_output_zero_point_q;
    reg proj_start_valid_q;
    reg proj_meta_valid_q;
    reg proj_wait_out_active_q;
    reg proj_write_data_active_q;
    reg proj_done_saturation_q;
    reg [ROPE_LANES*ACT_WIDTH-1:0] rope_act_q;
    reg [ROPE_LANES*ACT_WIDTH-1:0] rope_pair_act_q;
    reg [ROPE_LANES*16-1:0] rope_scale_q;
    reg [ROPE_LANES*16-1:0] rope_pair_scale_q;
    reg [ROPE_LANES*16-1:0] rope_cos_q;
    reg [ROPE_LANES*16-1:0] rope_sin_q;
    reg proj_wgt_recv_q;
    reg proj_meta_recv_q;
    reg proj_act_recv_q;
    reg gain_low_recv_q;
    reg gain_high_recv_q;
    reg rope_act_recv_q;
    reg rope_pair_recv_q;
    reg rope_scale_recv_q;
    reg rope_pair_scale_recv_q;
    reg rope_cos_recv_q;
    reg rope_sin_recv_q;
    reg rope_start_valid_q;
    reg rope_done_saturation_q;
    reg [ROPE_SEGMENT_INDEX_WIDTH-1:0] rope_segment_q;
    reg [ATTN_TOKEN_INDEX_WIDTH-1:0] attn_token_idx_q;
    reg [ATTN_GROUP_INDEX_WIDTH-1:0] attn_group_idx_q;
    reg [ATTN_MAC_LANES*ACT_WIDTH-1:0] attn_q_data_q;
    reg signed [31:0] attn_product_q;
    wire signed [31:0] attn_acc_q;
    reg signed [31:0] attn_score_multiplier_q;
    reg [5:0] attn_score_right_shift_q;
    reg attn_score_metadata_invalid_q;
    reg [63:0] attn_scaled_product_q;
    reg [63:0] attn_shift_value_q;
    reg [5:0] attn_shift_remaining_q;
    reg attn_shift_sticky_q;
    reg attn_shift_guard_q;
    reg [64:0] attn_rounded_abs_q;
    reg attn_score_negative_q;
    reg [63:0] attn_mul_multiplicand_q;
    reg [31:0] attn_mul_multiplier_q;
    reg [63:0] attn_mul_accum_q;
    reg [5:0] attn_mul_count_q;
    reg [1:0] attn_mul_chunk_q;
    reg attn_mul_carry_q;
    reg attn_mul_negative_q;
    reg [64:0] attn_score_magnitude_q [0:ATTN_CONTEXT_MAX-1];
    reg attn_score_negative_row_q [0:ATTN_CONTEXT_MAX-1];
    reg [64:0] attn_score_max_magnitude_q;
    reg attn_score_max_negative_q;
    reg [ATTN_TOKEN_INDEX_WIDTH-1:0] attn_center_idx_q;
    reg [ATTN_TOKEN_INDEX_WIDTH-1:0] attn_center_write_idx_q;
    reg [64:0] attn_center_magnitude_q;
    reg attn_center_negative_q;
    reg [65:0] attn_center_delta_q;
    reg [15:0] attn_center_score_q;
    reg attn_center_saturation_q;
    reg attn_done_saturation_q;
    reg attn_q_recv_q;
    reg [1:0] attn_value_nibble_q;
    reg softmax_start_valid_q;
    reg softmax_out_ready_q;
    reg softmax_clear_q;
    reg compose_start_valid_q;
    reg [127:0] compose_value_data_q;
    reg [SILU_INPUT_BEAT_WIDTH-1:0] silu_input_beat_q;
    reg silu_output_half_q;
    reg silu_write_final_q;
    reg [127:0] silu_gate_data_q;
    wire [127:0] silu_up_data_q;
    reg signed [31:0] silu_multiplier_q;
    reg [5:0] silu_right_shift_q;
    reg signed [7:0] silu_zero_point_q;
    reg silu_start_valid_q;
    reg silu_beat_valid_q;
    reg [27:0] rope_table_position_base_q;
    reg [31:0] rope_table_base_low_q;
    reg [31:0] rope_table_base_high_q;
    reg rope_table_base_carry_q;
    reg [31:0] rope_table_beat_low_q;
    reg [31:0] rope_table_beat_high_q;
    reg rope_table_beat_carry_q;
    reg [31:0] rope_table_half_low_q;
    reg rope_table_half_carry_q;
    reg [31:0] rope_scale_addr_low_q;
    reg rope_scale_addr_carry_q;
    (* keep = "true" *) reg cmd_load_decode_q;
    (* keep = "true" *) reg cmd_load_shape_q;
    (* keep = "true" *) reg cmd_load_src0_q;
    (* keep = "true" *) reg cmd_load_src1_q;
    (* keep = "true" *) reg cmd_load_dst_q;
    (* keep = "true" *) reg cmd_load_aux_q;
    (* keep = "true" *) reg cmd_dispatch_q;
    reg cmd_decode_busy_q;
    reg cmd_ready_q;
    reg done_valid_q;
    reg done_error_q;
    reg done_saturation_q;
    reg core_clear_q;
    reg core_start_valid_q;
    reg core_in_valid_q;
    reg act_feed_last_q;
    reg forward_progress_q;
    reg read_fault_q;
    reg write_fault_q;
    reg accepted_read_transition_q;
    reg [7:0] write_response_tag_q;
    reg [6:0] write_resume_state_q;
    reg write_completes_descriptor_q;

    wire control_enable_w = control_q[0];
    wire irq_global_enable_w = control_q[2];
    wire strict_errors_w = control_q[3];
    wire perf_clear_w = control_q[4];
    wire soft_reset_req_w = control_q[1];
    wire [6:0] state_commit_w =
        soft_reset_req_w ? ST_IDLE :
        (response_fault_pending_w || watchdog_fire_w) ? ST_COMPLETE :
        (((state_q == ST_IDLE) && cmd_dispatch_q) ? ST_CMD_DISPATCH : state_d);
    wire shell_busy_w = (state_q != ST_IDLE) || done_valid_q || reset_drain_q;
    wire halted_on_error_w = strict_errors_w && (error_status_q != {STATUS_BITS{1'b0}});
    wire csr_fire_w = csr_valid_i && csr_ready_o;
    wire cmd_fire_w = cmd_valid_i && cmd_ready_q;
    wire accepted_req_w = mem_req_valid_o && mem_req_ready_i;
    wire accepted_read_req_w = accepted_req_w && !mem_req_write_o;
    wire accepted_write_req_w = accepted_req_w && mem_req_write_o;
    wire accepted_read_w = mem_rvalid_i && mem_rready_q;
    wire accepted_write_w = mem_wvalid_o && mem_wready_i;
    wire accepted_b_w = mem_bvalid_i && mem_bready_o;
    wire reset_drain_remaining_w =
        (read_outstanding_q && !accepted_read_w) ||
        (write_data_pending_q && !accepted_write_w) ||
        ((write_response_outstanding_q || (reset_drain_q && accepted_write_w)) &&
         !accepted_b_w);
    wire core_write_accepted_w = (state_q == ST_WRITE_DATA) && core_out_valid_w && mem_wready_i;
    wire proj_write_accepted_w = (state_q == ST_PROJ_WRITE_DATA) && proj_write_data_active_q && mem_wready_i;
    wire rope_write_accepted_w = (state_q == ST_ROPE_WRITE_DATA) && mem_wready_i;
    wire response_fault_pending_w = read_fault_q || write_fault_q;
    wire [7:0] expected_read_tag_w =
        (state_q == ST_ACT_RECV) ? 8'h10 :
        (state_q == ST_SCALE_ACT_RECV) ? 8'h11 :
        (state_q == ST_GAIN_RECV0) ? 8'h20 :
        (state_q == ST_GAIN_RECV1) ? 8'h21 :
        (state_q == ST_PROJ_ACT0_RECV) ? 8'h40 :
        (state_q == ST_PROJ_ACT1_RECV) ? 8'h41 :
        (state_q == ST_PROJ_WGT_RECV) ? 8'h42 :
        (state_q == ST_PROJ_META_RECV) ? 8'h43 :
        (state_q == ST_ROPE_ACT_RECV) ? 8'h50 :
        (state_q == ST_ROPE_PAIR_RECV) ? 8'h51 :
        (state_q == ST_ROPE_SCALE0_RECV) ? 8'h52 :
        (state_q == ST_ROPE_PAIR_SCALE0_RECV) ? 8'h54 :
        (state_q == ST_ROPE_COS0_RECV) ? 8'h56 :
        (state_q == ST_ROPE_SIN0_RECV) ? 8'h58 :
        (state_q == ST_KV_READ_RECV) ? (8'h60 + {6'd0, kv_phase_q}) :
        (state_q == ST_ATTN_META_RECV) ? 8'h76 :
        (state_q == ST_ATTN_Q_RECV) ? 8'h70 :
        (state_q == ST_ATTN_K_RECV) ? 8'h72 :
        (state_q == ST_SOFTMAX_SCORE_RECV) ? 8'h80 :
        (state_q == ST_AV_PROB_RECV) ? 8'h90 :
        (state_q == ST_AV_V_RECV) ? 8'h92 :
        (state_q == ST_COMPOSE_SCORE_RECV) ? 8'hc0 :
        (state_q == ST_COMPOSE_VALUE_RECV) ? 8'hc1 :
        (state_q == ST_RES_SRC0_RECV) ? 8'ha0 :
        (state_q == ST_RES_SRC1_RECV) ? 8'ha1 :
        (state_q == ST_SILU_META_RECV) ? 8'hb0 :
        (state_q == ST_SILU_GATE_RECV) ? 8'hb1 :
        (state_q == ST_SILU_UP_RECV) ? 8'hb2 : 8'd0;
    wire [SILU_INPUT_BEAT_WIDTH-1:0] silu_input_beats_w =
        SILU_INPUT_BEAT_WIDTH'((n_q + 16'd7) >> 3);
    wire [SILU_INPUT_BEAT_WIDTH-1:0] silu_last_input_beat_w =
        silu_input_beats_w - SILU_INPUT_BEAT_WIDTH'(1);
    wire silu_final_input_w = (silu_input_beat_q == silu_last_input_beat_w);
    wire [3:0] silu_lane_count_w =
        silu_final_input_w && (n_q[2:0] != 3'd0) ? {1'b0, n_q[2:0]} : 4'd8;
    wire [14:0] silu_input_offset_w =
        {{(15-SILU_INPUT_BEAT_WIDTH-4){1'b0}}, silu_input_beat_q, 4'd0};
    wire [14:0] silu_next_input_offset_w = silu_input_offset_w + 15'd16;
    wire [14:0] silu_output_offset_w =
        {{(15-SILU_INPUT_BEAT_WIDTH-3){1'b0}},
         silu_input_beat_q[SILU_INPUT_BEAT_WIDTH-1:1], 4'd0};
    wire read_response_fault_w = mem_rerror_i || (mem_rtag_i != expected_read_tag_w);
    wire write_response_fault_w = mem_berror_i || (mem_btag_i != write_response_tag_q);
    wire mem_stall_w = ((mem_req_valid_o && !mem_req_ready_i) ||
                        (mem_wvalid_o && !mem_wready_i) ||
                        (mem_rvalid_i && !mem_rready_o) ||
                        ((state_q == ST_WRITE_RESP) && !mem_bvalid_i));
    wire [63:0] beat_idx_ext_w = {{(64-BEAT_INDEX_WIDTH){1'b0}}, beat_idx_q};
    wire [63:0] out_idx_ext_w = {{(64-BEAT_INDEX_WIDTH){1'b0}}, out_idx_q};
    wire [BEAT_INDEX_WIDTH-1:0] rope_pair_beat_idx_w = beat_idx_q ^ BEAT_INDEX_WIDTH'(2);
    wire [63:0] rope_pair_beat_idx_ext_w = {{(64-BEAT_INDEX_WIDTH){1'b0}}, rope_pair_beat_idx_w};
    wire [18:0] sequence_times7_w = {sequence_position_q, 3'b000} - {3'd0, sequence_position_q};
    wire [27:0] rope_table_position_base_w = (n_q == ROPE_K_SIZE_16) ? {3'd0, sequence_position_q, 9'd0} :
                                                                         {sequence_times7_w, 9'd0};
    wire [31:0] rope_table_position_offset_w = {4'd0, rope_table_position_base_q};
    wire [BEAT_INDEX_WIDTH-1:0] rope_last_beat_w = (n_q == ROPE_K_SIZE_16) ? ROPE_K_LAST_BEAT : LAST_BEAT;
    wire [31:0] rope_table_base_low_sum_w = src1_addr_q[31:0] + rope_table_position_offset_w;
    wire rope_table_base_low_carry_w = (rope_table_base_low_sum_w < src1_addr_q[31:0]);
    wire [31:0] rope_table_base_high_sum_w = src1_addr_q[63:32] + {31'd0, rope_table_base_carry_q};
    wire [31:0] rope_table_beat_offset_w = {{(32-BEAT_INDEX_WIDTH-6){1'b0}}, beat_idx_q, 6'd0};
    wire [31:0] rope_table_beat_low_sum_w = rope_table_base_low_q + rope_table_beat_offset_w;
    wire rope_table_beat_low_carry_w = (rope_table_beat_low_sum_w < rope_table_base_low_q);
    wire [31:0] rope_table_beat_high_sum_w = rope_table_base_high_q + {31'd0, rope_table_beat_carry_q};
    wire [32:0] rope_table_half_low_sum_w = {1'b0, rope_table_beat_low_q} + {28'd0, rope_segment_q[ROPE_SEGMENT_INDEX_WIDTH-1], 4'd0};
    wire [31:0] rope_table_half_high_sum_w = rope_table_beat_high_q + {31'd0, rope_table_half_carry_q};
    wire [63:0] rope_table_half_addr_w = {rope_table_half_high_sum_w, rope_table_half_low_q};
    wire [11:0] rope_scale_offset_w = {1'b0, beat_idx_q, 5'd0} + {7'd0, rope_segment_q[ROPE_SEGMENT_INDEX_WIDTH-1], 4'd0};
    wire [11:0] rope_pair_scale_offset_w = {1'b0, rope_pair_beat_idx_w, 5'd0} + {7'd0, rope_segment_q[ROPE_SEGMENT_INDEX_WIDTH-1], 4'd0};
    wire [32:0] rope_scale_low_sum_w = {1'b0, scale_addr_q[31:0]} + {21'd0, rope_scale_offset_w};
    wire [32:0] rope_pair_scale_low_sum_w = {1'b0, scale_addr_q[31:0]} + {21'd0, rope_pair_scale_offset_w};
    wire [31:0] rope_scale_addr_high_sum_w = scale_addr_q[63:32] + {31'd0, rope_scale_addr_carry_q};
    wire [63:0] rope_scale_addr_w = {rope_scale_addr_high_sum_w, rope_scale_addr_low_q};
    wire rope_second_half_w = beat_idx_q[1];
    wire [31:0] kv_sequence_offset_w = {8'd0, sequence_position_q, 8'd0} +
                                       {12'd0, sequence_position_q, 4'd0};
    wire [32:0] kv_dst_low_sum_w = {1'b0, scratch_addr_q[31:0]} + {1'b0, kv_sequence_offset_w};
    wire [31:0] kv_dst_high_sum_w = scratch_addr_q[63:32] + {31'd0, kv_dst_carry_q};
    wire [BEAT_INDEX_WIDTH-1:0] kv_last_beat_w = (kv_phase_q == KV_PHASE_META) ? {BEAT_INDEX_WIDTH{1'b0}} : KV_LAST_DATA_BEAT;
    wire kv_final_write_w = (kv_phase_q == KV_PHASE_META) && (beat_idx_q == {BEAT_INDEX_WIDTH{1'b0}});
    wire [ATTN_TOKEN_INDEX_WIDTH-1:0] attn_last_token_w = (n_q == ATTN_CONTEXT_MAX_16) ?
                                                          ATTN_TOKEN_INDEX_WIDTH'(ATTN_CONTEXT_MAX - 1) :
                                                          (n_q[ATTN_TOKEN_INDEX_WIDTH-1:0] - {{(ATTN_TOKEN_INDEX_WIDTH-1){1'b0}}, 1'b1});
    wire [63:0] attn_token_idx_ext_w = {{(64-ATTN_TOKEN_INDEX_WIDTH){1'b0}}, attn_token_idx_q};
    wire [63:0] attn_group_idx_ext_w = {{(64-ATTN_GROUP_INDEX_WIDTH){1'b0}}, attn_group_idx_q};
    wire [63:0] attn_next_group_idx_ext_w = attn_group_idx_ext_w + 64'd1;
    wire [63:0] attn_group_storage_idx_ext_w = attn_group_idx_ext_w >> ATTN_GROUP_STORAGE_SHIFT;
    wire [63:0] attn_next_group_storage_idx_ext_w = attn_next_group_idx_ext_w >> ATTN_GROUP_STORAGE_SHIFT;
    wire [3:0] attn_group_lane_offset_w = 4'(attn_group_idx_q[ATTN_GROUP_SELECT_WIDTH-1:0] * ATTN_MAC_LANES);
    wire [63:0] attn_k_token_base_w = attn_token_idx_ext_w << 6;
    wire [63:0] attn_q_addr_w = src0_addr_q + (attn_group_storage_idx_ext_w << 4);
    wire [63:0] attn_next_q_addr_w = src0_addr_q + (attn_next_group_storage_idx_ext_w << 4);
    wire [63:0] attn_k_addr_w = src1_addr_q + attn_k_token_base_w + (attn_group_storage_idx_ext_w << 4);
    wire [63:0] attn_value_write_addr_w = dst_addr_q + (attn_group_storage_idx_ext_w << 4);
    wire attn_value_last_pack_lane_w = (attn_group_idx_q[3:0] == 4'hf);
    wire attn_value_final_output_w = (attn_group_idx_q == ATTN_LAST_GROUP);
    wire [63:0] proj_row_idx_ext_w = {{(64-PROJ_ROW_INDEX_WIDTH){1'b0}}, proj_row_idx_q};
    wire [63:0] proj_group_idx_ext_w = {{(64-PROJ_GROUP_INDEX_WIDTH){1'b0}}, proj_group_idx_q};
    wire [63:0] proj_next_group_idx_ext_w = proj_group_idx_ext_w + 64'd1;
    wire [63:0] proj_group_storage_idx_ext_w = proj_group_idx_ext_w >> PROJ_GROUP_STORAGE_SHIFT;
    wire [63:0] proj_next_group_storage_idx_ext_w = proj_next_group_idx_ext_w >> PROJ_GROUP_STORAGE_SHIFT;
    wire [3:0] proj_group_lane_offset_w = 4'(proj_group_idx_q[PROJ_GROUP_SELECT_WIDTH-1:0] * PROJ_MAC_LANES);
    wire [63:0] proj_hidden_row_base_w = (proj_row_idx_ext_w << 10) - (proj_row_idx_ext_w << 7);
    wire [63:0] proj_mlp_row_base_w =
        (proj_row_idx_ext_w << 12) + (proj_row_idx_ext_w << 9) + (proj_row_idx_ext_w << 8);
    wire [63:0] proj_row_base_w =
        proj_mlp_shape_q ? proj_mlp_row_base_w : proj_hidden_row_base_w;
    wire [PROJ_GROUP_INDEX_WIDTH-1:0] proj_last_group_w =
        proj_mlp_shape_q ? PROJ_MLP_LAST_GROUP : PROJ_HIDDEN_LAST_GROUP;
    wire [12:0] proj_weight_addr_stride_w =
        proj_mlp_shape_q ?
        13'(PROJ_MLP_WEIGHT_BYTES_PER_OUTPUT) :
        13'(PROJ_HIDDEN_WEIGHT_BYTES_PER_OUTPUT);
    wire [PROJ_OUT_INDEX_WIDTH-1:0] proj_last_out_w = n_q[PROJ_OUT_INDEX_WIDTH-1:0] - {{(PROJ_OUT_INDEX_WIDTH-1){1'b0}}, 1'b1};
    wire [PROJ_ROW_INDEX_WIDTH-1:0] proj_last_row_w = m_q[PROJ_ROW_INDEX_WIDTH-1:0] - {{(PROJ_ROW_INDEX_WIDTH-1){1'b0}}, 1'b1};
    wire proj_final_output_w = (proj_out_idx_q == proj_last_out_w) && (proj_row_idx_q == proj_last_row_w);
    wire [PROJ_WEIGHT_OFFSET_WIDTH-1:0] proj_group_byte_offset_w =
        PROJ_WEIGHT_OFFSET_WIDTH'(proj_group_storage_idx_ext_w << 4);
    wire [PROJ_WEIGHT_OFFSET_WIDTH-1:0] proj_weight_request_offset_w =
        proj_weight_offset_q + proj_group_byte_offset_w;
    wire [PROJ_WEIGHT_OFFSET_WIDTH-1:0] proj_meta_request_offset_w =
        PROJ_WEIGHT_OFFSET_WIDTH'(proj_meta_offset_q);
    wire [PROJ_WEIGHT_OFFSET_WIDTH-1:0] proj_out_request_offset_w =
        PROJ_WEIGHT_OFFSET_WIDTH'(proj_out_write_offset_q);
    wire watchdog_active_w = (state_q != ST_IDLE) && (state_q != ST_WAIT_DONE) && !done_valid_q && (watchdog_limit_q != 64'd0);
    wire forward_progress_w = csr_fire_w ||
                              accepted_read_w ||
                              accepted_write_w ||
                              accepted_b_w ||
                              (state_q == ST_WAIT_DONE);
    wire watchdog_last_tick_w = watchdog_armed_q && (watchdog_count_q[63:1] == 63'd0);
    wire watchdog_fire_w = watchdog_fire_q;
    wire [LANES*GAIN_WIDTH-1:0] gain_beat_w = {gain_high_q, gain_low_q};
    wire rms_descriptor_valid_w = (cmd_opcode_buf_q == ACE2_OPCODE_RMSNORM) &&
                                  (cmd_m_buf_q == 16'd1) &&
                                  (cmd_n_buf_q == HIDDEN_SIZE_16) &&
                                  (cmd_k_buf_q == 16'd0) &&
                                  (cmd_layer_id_buf_q <= 8'd24) &&
                                  (cmd_src0_addr_buf_q[3:0] == 4'd0) &&
                                  (cmd_dst_addr_buf_q[3:0] == 4'd0) &&
                                  (cmd_scale_addr_buf_q[3:0] == 4'd0);
    wire proj_hidden_k_shape_w =
        (cmd_k_buf_q == HIDDEN_SIZE_16) &&
        ((cmd_n_buf_q == HIDDEN_SIZE_16) ||
         (cmd_n_buf_q == 16'd128) ||
         (cmd_n_buf_q == MLP_INTERMEDIATE_SIZE_16));
    wire proj_mlp_down_shape_w =
        (cmd_k_buf_q == MLP_INTERMEDIATE_SIZE_16) &&
        (cmd_n_buf_q == HIDDEN_SIZE_16);
    wire proj_lm_head_shape_w =
        (cmd_m_buf_q == 16'd1) &&
        (cmd_n_buf_q == LM_HEAD_TILE_SIZE_16) &&
        (cmd_k_buf_q == HIDDEN_SIZE_16) &&
        (cmd_layer_id_buf_q == 8'd24);
    wire proj_descriptor_valid_w = (cmd_opcode_buf_q == ACE2_OPCODE_W4A8_PROJ) &&
                                   (cmd_m_buf_q != 16'd0) &&
                                   (cmd_m_buf_q <= 16'(PROJ_M_MAX)) &&
                                   (((proj_hidden_k_shape_w || proj_mlp_down_shape_w) &&
                                     (cmd_layer_id_buf_q < 8'd24)) ||
                                    proj_lm_head_shape_w) &&
                                   (cmd_src0_addr_buf_q[3:0] == 4'd0) &&
                                   (cmd_src1_addr_buf_q[3:0] == 4'd0) &&
                                   (cmd_dst_addr_buf_q[3:0] == 4'd0) &&
                                   (cmd_scale_addr_buf_q[3:0] == 4'd0);
    wire rope_descriptor_valid_w = (cmd_opcode_buf_q == ACE2_OPCODE_ROPE) &&
                                   (cmd_flags_buf_q == ROPE_BASIS_ROTATION_HALF_DEGREES) &&
                                   (cmd_m_buf_q == 16'd1) &&
                                   ((cmd_n_buf_q == HIDDEN_SIZE_16) || (cmd_n_buf_q == ROPE_K_SIZE_16)) &&
                                   (cmd_k_buf_q == 16'd0) &&
                                   (cmd_sequence_position_buf_q <= ROPE_MAX_SEQUENCE_POSITION) &&
                                   (cmd_layer_id_buf_q < 8'd24) &&
                                   (cmd_src0_addr_buf_q[3:0] == 4'd0) &&
                                   (cmd_src1_addr_buf_q[3:0] == 4'd0) &&
                                   (cmd_scale_addr_buf_q[3:0] == 4'd0) &&
                                   (cmd_scratch_addr_buf_q[3:0] == 4'd0);
    wire kv_write_descriptor_valid_w = (cmd_opcode_buf_q == ACE2_OPCODE_KV_WRITE) &&
                                       (cmd_m_buf_q == 16'd1) &&
                                       (cmd_n_buf_q == ROPE_K_SIZE_16) &&
                                       (cmd_k_buf_q == 16'd0) &&
                                       (cmd_sequence_position_buf_q <= ROPE_MAX_SEQUENCE_POSITION) &&
                                       (cmd_layer_id_buf_q < 8'd24) &&
                                       (cmd_src0_addr_buf_q[3:0] == 4'd0) &&
                                       (cmd_src1_addr_buf_q[3:0] == 4'd0) &&
                                       (cmd_scratch_addr_buf_q[3:0] == 4'd0) &&
                                       (cmd_scale_addr_buf_q[3:0] == 4'd0);
    wire attn_score_descriptor_valid_w = (cmd_opcode_buf_q == ACE2_OPCODE_ATTN_SCORE) &&
                                        (cmd_m_buf_q == 16'd1) &&
                                        (cmd_n_buf_q != 16'd0) &&
                                        (cmd_n_buf_q <= ATTN_CONTEXT_MAX_16) &&
                                        (cmd_k_buf_q == ATTN_HEAD_DIM_16) &&
                                        (cmd_layer_id_buf_q < 8'd24) &&
                                        (cmd_src0_addr_buf_q[3:0] == 4'd0) &&
                                        (cmd_src1_addr_buf_q[3:0] == 4'd0) &&
                                        (cmd_dst_addr_buf_q[3:0] == 4'd0) &&
                                        (cmd_scale_addr_buf_q[3:0] == 4'd0);
    wire softmax_descriptor_valid_w = (cmd_opcode_buf_q == ACE2_OPCODE_SOFTMAX) &&
                                     (cmd_m_buf_q == 16'd1) &&
                                     (cmd_n_buf_q != 16'd0) &&
                                     (cmd_n_buf_q <= ATTN_CONTEXT_MAX_16) &&
                                     (cmd_k_buf_q == 16'd0) &&
                                     (cmd_layer_id_buf_q < 8'd24) &&
                                     (cmd_src0_addr_buf_q[3:0] == 4'd0) &&
                                     (cmd_dst_addr_buf_q[3:0] == 4'd0);
    wire attn_value_descriptor_valid_w = (cmd_opcode_buf_q == ACE2_OPCODE_ATTN_VALUE) &&
                                         (cmd_m_buf_q == 16'd1) &&
                                         (cmd_n_buf_q != 16'd0) &&
                                         (cmd_n_buf_q <= ATTN_CONTEXT_MAX_16) &&
                                         (cmd_k_buf_q == ATTN_HEAD_DIM_16) &&
                                         (cmd_layer_id_buf_q < 8'd24) &&
                                         (cmd_src0_addr_buf_q[3:0] == 4'd0) &&
                                         (cmd_src1_addr_buf_q[3:0] == 4'd0) &&
                                         (cmd_dst_addr_buf_q[3:0] == 4'd0);
    wire [7:0] compose_command_w =
        (state_q == ST_IDLE) ? cmd_flags_buf_q : flags_q;
    wire [15:0] compose_tile_count_w =
        (state_q == ST_IDLE) ? cmd_n_buf_q : n_q;
    wire compose_command_allowed_w;
    wire compose_descriptor_valid_w =
        (cmd_opcode_buf_q == ACE2_OPCODE_ATTN_COMPOSE) &&
        (cmd_m_buf_q == 16'd1) &&
        (cmd_n_buf_q != 16'd0) &&
        (cmd_n_buf_q <= ATTN_CONTEXT_MAX_16) &&
        (cmd_k_buf_q == ATTN_HEAD_DIM_16) &&
        (cmd_layer_id_buf_q < 8'd24) &&
        (cmd_src0_addr_buf_q[3:0] == 4'd0) &&
        (cmd_src1_addr_buf_q[3:0] == 4'd0) &&
        (cmd_dst_addr_buf_q[3:0] == 4'd0) &&
        compose_command_allowed_w;
    wire residual_descriptor_valid_w = (cmd_opcode_buf_q == ACE2_OPCODE_RESIDUAL_ADD) &&
                                       (cmd_m_buf_q == 16'd1) &&
                                       (cmd_n_buf_q == HIDDEN_SIZE_16) &&
                                       (cmd_k_buf_q == 16'd0) &&
                                       (cmd_layer_id_buf_q < 8'd24) &&
                                       (cmd_src0_addr_buf_q[3:0] == 4'd0) &&
                                       (cmd_src1_addr_buf_q[3:0] == 4'd0) &&
                                       (cmd_dst_addr_buf_q[3:0] == 4'd0);
    wire silu_descriptor_valid_w = (cmd_opcode_buf_q == ACE2_OPCODE_SILU_GATE) &&
                                   (cmd_m_buf_q == 16'd1) &&
                                   (cmd_n_buf_q != 16'd0) &&
                                   (cmd_n_buf_q <= MLP_INTERMEDIATE_SIZE_16) &&
                                   (cmd_k_buf_q == 16'd0) &&
                                   (cmd_layer_id_buf_q < 8'd24) &&
                                   (cmd_src0_addr_buf_q[3:0] == 4'd0) &&
                                   (cmd_src1_addr_buf_q[3:0] == 4'd0) &&
                                   (cmd_dst_addr_buf_q[3:0] == 4'd0) &&
                                   (cmd_scale_addr_buf_q[3:0] == 4'd0);
    wire descriptor_valid_w = rms_descriptor_valid_w ||
                              proj_descriptor_valid_w ||
                              rope_descriptor_valid_w ||
                              kv_write_descriptor_valid_w ||
                              attn_score_descriptor_valid_w ||
                              softmax_descriptor_valid_w ||
                              attn_value_descriptor_valid_w ||
                              compose_descriptor_valid_w ||
                              residual_descriptor_valid_w ||
                              silu_descriptor_valid_w;
    wire [3:0] cmd_op_kind_i_w = (cmd_opcode_i == ACE2_OPCODE_RMSNORM) ? OP_KIND_RMSNORM :
                                 (cmd_opcode_i == ACE2_OPCODE_W4A8_PROJ) ? OP_KIND_PROJ :
                                 (cmd_opcode_i == ACE2_OPCODE_ROPE) ? OP_KIND_ROPE :
                                 (cmd_opcode_i == ACE2_OPCODE_ATTN_SCORE) ? OP_KIND_ATTN :
                                 (cmd_opcode_i == ACE2_OPCODE_SOFTMAX) ? OP_KIND_SOFTMAX :
                                 (cmd_opcode_i == ACE2_OPCODE_ATTN_VALUE) ? OP_KIND_ATTN_VALUE :
                                 (cmd_opcode_i == ACE2_OPCODE_ATTN_COMPOSE) ? OP_KIND_ATTN_COMPOSE :
                                 (cmd_opcode_i == ACE2_OPCODE_RESIDUAL_ADD) ? OP_KIND_RESIDUAL :
                                 (cmd_opcode_i == ACE2_OPCODE_SILU_GATE) ? OP_KIND_SILU :
                                 OP_KIND_KV;

    wire core_start_ready_w;
    wire core_start_valid_w = core_start_valid_q && !response_fault_pending_w;
    wire core_in_ready_w;
    wire core_in_valid_w = core_in_valid_q;
    wire core_scale_act_ready_w;
    (* keep = "true" *) wire core_feed_valid_w;
    assign core_feed_valid_w = (state_q == ST_CORE_FEED) && !response_fault_pending_w;
    wire core_scale_act_valid_w = core_feed_valid_w;
    wire core_gain_ready_w;
    wire core_gain_valid_w = core_feed_valid_w;
    wire core_out_valid_w;
    wire core_out_ready_w = (state_q == ST_WRITE_DATA) && mem_wready_i;
    wire [LANES*ACT_WIDTH-1:0] core_out_data_w;
    wire core_done_valid_w;
    wire core_done_ready_w = (state_q == ST_WAIT_DONE) && !done_valid_q;
    wire [ACC_WIDTH-1:0] core_sumsq_w;
    wire [INV_RMS_FRAC+1:0] core_inv_w;
    wire core_saturation_w;
    wire proj_start_ready_w;
    wire proj_pair_ready_w;
    wire proj_pair_valid_w = (state_q == ST_PROJ_FEED) && !response_fault_pending_w;
    wire [PROJ_MAC_LANES*ACT_WIDTH-1:0] proj_act_pair_w = proj_act_low_q;
    wire proj_meta_ready_w;
    wire proj_meta_valid_w = proj_meta_valid_q && !response_fault_pending_w;
    wire proj_out_valid_w;
    wire proj_out_ready_w = proj_wait_out_active_q;
    wire [ACT_WIDTH-1:0] proj_out_data_w;
    wire signed [PROJ_ACC_WIDTH-1:0] proj_acc_w;
    wire proj_saturation_w;
    wire signed [31:0] proj_meta_multiplier_w = proj_multiplier_q;
    wire [5:0] proj_meta_right_shift_w = proj_right_shift_q;
    wire signed [ACT_WIDTH-1:0] proj_meta_zero_point_w = proj_output_zero_point_q;
    wire [PROJ_MAC_LANES*ACT_WIDTH-1:0] proj_selected_act_w = mem_rdata_i[proj_group_lane_offset_w*ACT_WIDTH +: PROJ_MAC_LANES*ACT_WIDTH];
    wire [PROJ_MAC_LANES*4-1:0] proj_selected_weight_w = mem_rdata_i[proj_group_lane_offset_w*4 +: PROJ_MAC_LANES*4];
    wire [ROPE_AUX_SELECT_WIDTH-1:0] rope_aux_select_w = rope_segment_q[ROPE_AUX_SELECT_WIDTH-1:0];
    wire [ROPE_LANES*ACT_WIDTH-1:0] rope_selected_act_w = mem_rdata_i[(rope_segment_q*ROPE_LANES*ACT_WIDTH) +: ROPE_LANES*ACT_WIDTH];
    wire [ROPE_LANES*16-1:0] rope_selected_aux_w = mem_rdata_i[(rope_aux_select_w*ROPE_LANES*16) +: ROPE_LANES*16];
    wire rope_start_ready_w;
    wire rope_beat_ready_w;
    wire rope_beat_valid_w = (state_q == ST_ROPE_FEED) && !response_fault_pending_w;
    wire rope_out_valid_w;
    wire rope_out_ready_w = (state_q == ST_ROPE_WAIT_OUT);
    wire [ROPE_LANES*ACT_WIDTH-1:0] rope_out_data_w;
    wire rope_saturation_w;
    wire softmax_start_ready_w;
    wire softmax_out_valid_w;
    wire softmax_out_ready_w = softmax_out_ready_q;
    wire [ATTN_CONTEXT_MAX*16-1:0] softmax_out_data_w;
    wire softmax_saturation_w;
    wire compose_start_ready_w;
    wire compose_value_ready_w;
    wire compose_out_valid_w;
    wire [127:0] compose_out_data_w;
    /* verilator lint_off UNUSED */
    wire compose_out_last_unused_w;
    /* verilator lint_on UNUSED */
    wire compose_command_done_w;
    wire compose_saturation_w;
    wire [15:0] compose_context_count_w;
    wire compose_start_valid_w =
        compose_start_valid_q && !response_fault_pending_w;
    wire compose_value_valid_w =
        (state_q == ST_COMPOSE_VALUE_FEED) && !response_fault_pending_w;
    wire compose_out_ready_w =
        (state_q == ST_COMPOSE_OUT_DATA) && mem_wready_i &&
        !response_fault_pending_w;
    wire compose_value_final_w =
        (attn_token_idx_q == attn_last_token_w) &&
        (beat_idx_q[1:0] == 2'd3);
    wire [63:0] compose_value_addr_w =
        src1_addr_q + (attn_token_idx_ext_w << 6) +
        ({{62{1'b0}}, beat_idx_q[1:0]} << 4);
    wire [63:0] compose_out_addr_w =
        dst_addr_q + ({{62{1'b0}}, beat_idx_q[1:0]} << 4);
    wire compose_out_last_shell_w = (beat_idx_q[1:0] == 2'd3);
    wire silu_start_ready_w;
    wire silu_beat_ready_w;
    wire silu_out_valid_w;
    wire silu_out_ready_w = (state_q == ST_SILU_WAIT);
    wire [63:0] silu_out_data_w;
    wire silu_saturation_w;
    wire [15:0] attn_score_w;
    wire attn_saturation_w;
    wire [ATTN_MAC_LANES*ACT_WIDTH-1:0] attn_selected_data_w = mem_rdata_i[attn_group_lane_offset_w*ACT_WIDTH +: ATTN_MAC_LANES*ACT_WIDTH];
    wire [15:0] attn_value_probability_w = gain_low_q[attn_token_idx_q*16 +: 16];
    wire [3:0] attn_value_probability_nibble_w =
        attn_value_probability_w[attn_value_nibble_q*4 +: 4];
    wire signed [ACT_WIDTH-1:0] attn_mac_lhs_w =
        (op_kind_q == OP_KIND_ATTN_VALUE) ?
        $signed({4'd0, attn_value_probability_nibble_w}) :
        $signed(attn_q_data_q);
    wire signed [ACT_WIDTH-1:0] attn_mac_rhs_w =
        (op_kind_q == OP_KIND_ATTN_VALUE) ?
        $signed(attn_q_data_q) :
        $signed(attn_selected_data_w);
    wire signed [31:0] attn_product_w = attn_mac_lhs_w * attn_mac_rhs_w;
    wire signed [31:0] attn_value_shifted_product_w =
        attn_product_q <<< {attn_value_nibble_q, 2'b00};
    wire attn_acc_zero_w =
        (state_q == ST_CMD_DISPATCH) ||
        (state_q == ST_ATTN_START) ||
        ((state_q == ST_ATTN_WAIT_OUT) &&
         (attn_token_idx_q != attn_last_token_w)) ||
        (state_q == ST_AV_START);
    wire attn_acc_score_add_w = (state_q == ST_ATTN_FEED);
    wire attn_acc_value_add_w = (state_q == ST_AV_ACCUM);
    wire [31:0] attn_acc_magnitude_w =
        attn_acc_q[31] ? (~attn_acc_q + 32'd1) : attn_acc_q;
    wire [31:0] attn_multiplier_magnitude_w =
        attn_score_multiplier_q[31] ?
        (~attn_score_multiplier_q + 32'd1) :
        attn_score_multiplier_q;
    wire [15:0] attn_mul_accum_chunk_w =
        (attn_mul_chunk_q == 2'd0) ? attn_mul_accum_q[15:0] :
        (attn_mul_chunk_q == 2'd1) ? attn_mul_accum_q[31:16] :
        (attn_mul_chunk_q == 2'd2) ? attn_mul_accum_q[47:32] :
                                     attn_mul_accum_q[63:48];
    wire [15:0] attn_mul_multiplicand_chunk_w =
        (attn_mul_chunk_q == 2'd0) ? attn_mul_multiplicand_q[15:0] :
        (attn_mul_chunk_q == 2'd1) ? attn_mul_multiplicand_q[31:16] :
        (attn_mul_chunk_q == 2'd2) ? attn_mul_multiplicand_q[47:32] :
                                     attn_mul_multiplicand_q[63:48];
    wire [15:0] attn_mul_addend_chunk_w =
        attn_mul_multiplier_q[0] ? attn_mul_multiplicand_chunk_w : 16'd0;
    wire [16:0] attn_mul_chunk_sum_w =
        {1'b0, attn_mul_accum_chunk_w} +
        {1'b0, attn_mul_addend_chunk_w} +
        {{16{1'b0}}, attn_mul_carry_q};
    wire attn_round_increment_w =
        attn_shift_guard_q &&
        (attn_shift_sticky_q || attn_shift_value_q[0]);
    wire [64:0] attn_rounded_abs_w =
        {1'b0, attn_shift_value_q} + {{64{1'b0}}, attn_round_increment_w};
    wire attn_score_gt_max_w =
        (!attn_score_negative_q && attn_score_max_negative_q) ||
        ((attn_score_negative_q == attn_score_max_negative_q) &&
         (attn_score_negative_q ?
          (attn_rounded_abs_q < attn_score_max_magnitude_q) :
          (attn_rounded_abs_q > attn_score_max_magnitude_q)));
    wire attn_center_fetch_negative_w =
        attn_score_negative_row_q[attn_center_idx_q];
    wire [64:0] attn_center_fetch_magnitude_w =
        attn_score_magnitude_q[attn_center_idx_q];
    wire [65:0] attn_center_delta_w =
        !attn_score_max_negative_q ?
        (attn_center_negative_q ?
         ({1'b0, attn_score_max_magnitude_q} +
          {1'b0, attn_center_magnitude_q}) :
         ({1'b0, attn_score_max_magnitude_q} -
          {1'b0, attn_center_magnitude_q})) :
        ({1'b0, attn_center_magnitude_q} -
         {1'b0, attn_score_max_magnitude_q});
    wire attn_output_saturate_w = attn_center_delta_q > 66'd32768;
    assign attn_score_w =
        attn_output_saturate_w ?
        16'h8000 :
        ((attn_center_delta_q == 66'd0) ?
         16'h0000 : (~attn_center_delta_q[15:0] + 16'd1));
    assign attn_saturation_w = attn_output_saturate_w;
    wire attn_value_acc_negative_w = attn_acc_q[31];
    wire [31:0] attn_value_acc_abs_w =
        attn_value_acc_negative_w ? (~attn_acc_q + 32'd1) : attn_acc_q;
    wire [16:0] attn_value_rounded_base_w = attn_value_acc_abs_w[31:15];
    wire [14:0] attn_value_rounded_remainder_w = attn_value_acc_abs_w[14:0];
    wire attn_value_round_increment_w =
        (attn_value_rounded_remainder_w > 15'd16384) ||
        ((attn_value_rounded_remainder_w == 15'd16384) && attn_value_rounded_base_w[0]);
    wire [17:0] attn_value_rounded_abs_w =
        {1'b0, attn_value_rounded_base_w} + {17'd0, attn_value_round_increment_w};
    wire attn_value_positive_saturate_w =
        !attn_value_acc_negative_w && (attn_value_rounded_abs_w > 18'd127);
    wire attn_value_negative_saturate_w =
        attn_value_acc_negative_w && (attn_value_rounded_abs_w > 18'd128);
    wire attn_value_saturation_w =
        attn_value_positive_saturate_w || attn_value_negative_saturate_w;
    wire [7:0] attn_value_output_w =
        attn_value_saturation_w ? (attn_value_acc_negative_w ? 8'h80 : 8'h7f) :
        (attn_value_acc_negative_w ?
         (~attn_value_rounded_abs_w[7:0] + 8'd1) :
         attn_value_rounded_abs_w[7:0]);
    wire residual_write_accepted_w = (state_q == ST_RES_WRITE_DATA) && mem_wready_i;
    wire silu_write_accepted_w = (state_q == ST_SILU_WRITE_DATA) && mem_wready_i;
    wire [LANES-1:0] residual_saturation_lane_w;
    wire [LANES*ACT_WIDTH-1:0] residual_out_data_w;
    genvar residual_lane;
    generate
        for (residual_lane = 0; residual_lane < LANES; residual_lane = residual_lane + 1) begin : gen_residual_add
            wire signed [ACT_WIDTH:0] residual_sum_w =
                $signed(shared_payload_q[residual_lane*ACT_WIDTH +: ACT_WIDTH]) +
                $signed(gain_low_q[residual_lane*ACT_WIDTH +: ACT_WIDTH]);
            wire residual_positive_sat_w = !residual_sum_w[ACT_WIDTH] &&
                                           residual_sum_w[ACT_WIDTH-1];
            wire residual_negative_sat_w = residual_sum_w[ACT_WIDTH] &&
                                           !residual_sum_w[ACT_WIDTH-1];
            assign residual_saturation_lane_w[residual_lane] =
                residual_positive_sat_w || residual_negative_sat_w;
            assign residual_out_data_w[residual_lane*ACT_WIDTH +: ACT_WIDTH] =
                residual_positive_sat_w ? 8'h7f :
                residual_negative_sat_w ? 8'h80 : residual_sum_w[ACT_WIDTH-1:0];
        end
    endgenerate
    wire residual_saturation_w = |residual_saturation_lane_w;
    wire core_numeric_saturation_w =
        (state_q == ST_WAIT_DONE) && core_done_valid_w && core_saturation_w;
    wire proj_numeric_saturation_w =
        (state_q == ST_PROJ_WAIT_OUT) && proj_wait_out_active_q &&
        proj_out_valid_w && proj_saturation_w;
    wire softmax_numeric_saturation_w =
        (state_q == ST_SOFTMAX_WAIT_OUT) && softmax_out_valid_w &&
        softmax_saturation_w;
    wire attn_numeric_saturation_w =
        ((state_q == ST_ATTN_CENTER_STORE) && attn_center_saturation_q) ||
        ((state_q == ST_AV_ROUND) && attn_value_saturation_w);
    wire silu_numeric_saturation_w =
        (state_q == ST_SILU_WAIT) && silu_out_valid_w && silu_saturation_w;
    wire kv_write_accepted_w = (state_q == ST_KV_WRITE_DATA) && mem_wready_i;
    wire [LANES*ACT_WIDTH-1:0] mem_write_data_w = (state_q == ST_WRITE_DATA) ? core_out_data_w :
                                                  (state_q == ST_PROJ_WRITE_DATA) ? shared_payload_q :
                                                  (state_q == ST_ROPE_WRITE_DATA) ? shared_payload_q :
                                                  (state_q == ST_ATTN_WRITE_DATA) ? shared_payload_q :
                                                  (state_q == ST_SOFTMAX_WRITE_DATA) ? shared_payload_q :
                                                  (state_q == ST_AV_WRITE_DATA) ? shared_payload_q :
                                                  (state_q == ST_RES_WRITE_DATA) ? residual_out_data_w :
                                                  (state_q == ST_SILU_WRITE_DATA) ? shared_payload_q :
                                                  (state_q == ST_COMPOSE_OUT_DATA) ? compose_out_data_w :
                                                  (state_q == ST_KV_WRITE_DATA) ? shared_payload_q :
                                                  {LANES*ACT_WIDTH{1'b0}};
    wire [7:0] mem_write_tag_w = (state_q == ST_WRITE_DATA) ? 8'h30 :
                                 (state_q == ST_PROJ_WRITE_DATA) ? 8'h44 :
                                 (state_q == ST_ROPE_WRITE_DATA) ? 8'h5a :
                                 (state_q == ST_ATTN_WRITE_DATA) ? 8'h74 :
                                 (state_q == ST_SOFTMAX_WRITE_DATA) ? 8'h84 :
                                 (state_q == ST_AV_WRITE_DATA) ? 8'h94 :
                                 (state_q == ST_RES_WRITE_DATA) ? 8'ha2 :
                                 (state_q == ST_SILU_WRITE_DATA) ? 8'hb3 :
                                 (state_q == ST_COMPOSE_OUT_DATA) ? 8'hc2 :
                                 (state_q == ST_KV_WRITE_DATA) ? (8'h68 + {6'd0, kv_phase_q}) : 8'd0;
    wire [6:0] write_resume_state_w =
        (state_q == ST_WRITE_DATA) ?
            ((out_idx_q == LAST_BEAT) ? ST_WAIT_DONE : ST_SCALE_ACT_REQ) :
        (state_q == ST_PROJ_WRITE_DATA) ?
            (proj_final_output_w ? ST_COMPLETE : ST_PROJ_START) :
        (state_q == ST_ROPE_WRITE_DATA) ?
            ((beat_idx_q == rope_last_beat_w) ? ST_COMPLETE : ST_ROPE_START) :
        (state_q == ST_KV_WRITE_DATA) ?
            (kv_final_write_w ? ST_COMPLETE : ST_KV_READ_REQ) :
        (state_q == ST_AV_WRITE_DATA) ?
            (attn_value_final_output_w ? ST_COMPLETE : ST_AV_START) :
        (state_q == ST_RES_WRITE_DATA) ?
            ((beat_idx_q == LAST_BEAT) ? ST_COMPLETE : ST_RES_SRC0_REQ) :
        (state_q == ST_SILU_WRITE_DATA) ?
            (silu_write_final_q ? ST_COMPLETE : ST_SILU_GATE_REQ) :
        (state_q == ST_COMPOSE_OUT_DATA) ?
            (compose_out_last_shell_w ? ST_COMPLETE : ST_COMPOSE_WAIT) :
        ((state_q == ST_ATTN_WRITE_DATA) || (state_q == ST_SOFTMAX_WRITE_DATA)) ?
            ST_COMPLETE : ST_IDLE;
    wire write_completes_descriptor_w = (write_resume_state_w == ST_COMPLETE);
    wire write_retire_success_w = (state_q == ST_WRITE_RETIRE) && !write_fault_q;
    wire compose_command_retire_w =
        (state_q == ST_COMPOSE_WAIT) && compose_command_done_w &&
        !done_valid_q;

    function automatic [63:0] add_small64;
        input [63:0] base;
        input [11:0] offset;
        reg [32:0] low_sum;
        begin
            low_sum = {1'b0, base[31:0]} + {20'd0, offset};
            add_small64 = {base[63:32] + {31'd0, low_sum[32]}, low_sum[31:0]};
        end
    endfunction

    function automatic [63:0] add_proj_offset64;
        input [63:0] base;
        input [PROJ_WEIGHT_OFFSET_WIDTH-1:0] offset;
        reg [32:0] low_sum;
        begin
            low_sum = {1'b0, base[31:0]} + {{(33-PROJ_WEIGHT_OFFSET_WIDTH){1'b0}}, offset};
            add_proj_offset64 = {base[63:32] + {31'd0, low_sum[32]}, low_sum[31:0]};
        end
    endfunction

    function automatic [63:0] add_silu_offset64;
        input [63:0] base;
        input [14:0] offset;
        reg [16:0] segment0_sum;
        reg [16:0] segment1_inc;
        reg [16:0] segment2_inc;
        reg [16:0] segment3_inc;
        reg carry1;
        reg carry2;
        reg carry3;
        begin
            // Bound carry propagation on the register-to-register address path.
            segment0_sum = {1'b0, base[15:0]} + {2'd0, offset};
            segment1_inc = {1'b0, base[31:16]} + 17'd1;
            segment2_inc = {1'b0, base[47:32]} + 17'd1;
            segment3_inc = {1'b0, base[63:48]} + 17'd1;
            carry1 = segment0_sum[16];
            carry2 = carry1 && segment1_inc[16];
            carry3 = carry2 && segment2_inc[16];
            add_silu_offset64[15:0] = segment0_sum[15:0];
            add_silu_offset64[31:16] = carry1 ? segment1_inc[15:0] : base[31:16];
            add_silu_offset64[47:32] = carry2 ? segment2_inc[15:0] : base[47:32];
            add_silu_offset64[63:48] = carry3 ? segment3_inc[15:0] : base[63:48];
        end
    endfunction

    assign csr_ready_o = !csr_req_valid_q && (!csr_rvalid_o || csr_rready_i);
    assign irq_o = irq_global_enable_w && ((interrupt_status_q & interrupt_enable_q) != {STATUS_BITS{1'b0}});
    assign cmd_ready_o = cmd_ready_q;
    assign busy_o = shell_busy_w;
    assign cmd_done_valid_o = done_valid_q;
    assign cmd_done_tag_o = completion_tag_q;
    assign cmd_done_error_o = done_error_q;
    assign cmd_done_sumsq_o = core_sumsq_w;
    assign cmd_done_inv_rms_q30_o = core_inv_w;
    assign cmd_done_saturation_seen_o = done_saturation_q;

    assign mem_req_valid_o = !soft_reset_req_w && !reset_drain_q &&
                             !response_fault_pending_w && ((state_q == ST_ACT_REQ) ||
                             (state_q == ST_SCALE_ACT_REQ) ||
                             (state_q == ST_GAIN_REQ0) ||
                             (state_q == ST_GAIN_REQ1) ||
                            ((state_q == ST_WRITE_REQ) && core_out_valid_w) ||
                            (state_q == ST_PROJ_ACT0_REQ) ||
                            (state_q == ST_PROJ_ACT1_REQ) ||
                            (state_q == ST_PROJ_WGT_REQ) ||
                            (state_q == ST_PROJ_META_REQ) ||
                            (state_q == ST_PROJ_WRITE_REQ) ||
                            (state_q == ST_ROPE_ACT_REQ) ||
                            (state_q == ST_ROPE_PAIR_REQ) ||
                            (state_q == ST_ROPE_SCALE0_REQ) ||
                            (state_q == ST_ROPE_PAIR_SCALE0_REQ) ||
                            (state_q == ST_ROPE_COS0_REQ) ||
                            (state_q == ST_ROPE_SIN0_REQ) ||
                            (state_q == ST_ROPE_WRITE_REQ) ||
                            (state_q == ST_ATTN_META_REQ) ||
                            (state_q == ST_ATTN_Q_REQ) ||
                            (state_q == ST_ATTN_K_REQ) ||
                            (state_q == ST_ATTN_WRITE_REQ) ||
                            (state_q == ST_SOFTMAX_SCORE_REQ) ||
                            (state_q == ST_SOFTMAX_WRITE_REQ) ||
                            (state_q == ST_AV_PROB_REQ) ||
                            (state_q == ST_AV_V_REQ) ||
                            (state_q == ST_AV_WRITE_REQ) ||
                            (state_q == ST_RES_SRC0_REQ) ||
                            (state_q == ST_RES_SRC1_REQ) ||
                            (state_q == ST_RES_WRITE_REQ) ||
                            (state_q == ST_SILU_META_REQ) ||
                            (state_q == ST_SILU_GATE_REQ) ||
                            (state_q == ST_SILU_UP_REQ) ||
                            (state_q == ST_SILU_WRITE_REQ) ||
                            (state_q == ST_KV_READ_REQ) ||
                            (state_q == ST_KV_WRITE_REQ) ||
                            (state_q == ST_COMPOSE_SCORE_REQ) ||
                            (state_q == ST_COMPOSE_VALUE_REQ) ||
                            (state_q == ST_COMPOSE_OUT_REQ));
    assign mem_req_write_o = (state_q == ST_WRITE_REQ) || (state_q == ST_PROJ_WRITE_REQ) || (state_q == ST_ROPE_WRITE_REQ) || (state_q == ST_ATTN_WRITE_REQ) || (state_q == ST_SOFTMAX_WRITE_REQ) || (state_q == ST_AV_WRITE_REQ) || (state_q == ST_RES_WRITE_REQ) || (state_q == ST_SILU_WRITE_REQ) || (state_q == ST_KV_WRITE_REQ) || (state_q == ST_COMPOSE_OUT_REQ);
    wire proj_mem_req_addr_selected_w =
        (state_q == ST_PROJ_ACT0_REQ) ||
        (state_q == ST_PROJ_ACT1_REQ) ||
        (state_q == ST_PROJ_WGT_REQ) ||
        (state_q == ST_PROJ_META_REQ) ||
        (state_q == ST_PROJ_WRITE_REQ);
    assign mem_req_addr_o = silu_active_q ? silu_mem_req_addr_q :
                            proj_mem_req_addr_selected_w ? proj_mem_req_addr_q :
                            prefix_mem_req_addr_q;
    assign mem_req_len_o = 16'd1;
    assign mem_req_tag_o = (state_q == ST_ACT_REQ)   ? 8'h10 :
                          (state_q == ST_SCALE_ACT_REQ) ? 8'h11 :
                          (state_q == ST_GAIN_REQ0) ? 8'h20 :
                          (state_q == ST_GAIN_REQ1) ? 8'h21 :
                          (state_q == ST_PROJ_ACT0_REQ) ? 8'h40 :
                          (state_q == ST_PROJ_ACT1_REQ) ? 8'h41 :
                          (state_q == ST_PROJ_WGT_REQ) ? 8'h42 :
                          (state_q == ST_PROJ_META_REQ) ? 8'h43 :
                          (state_q == ST_PROJ_WRITE_REQ) ? 8'h44 :
                          (state_q == ST_ROPE_ACT_REQ) ? 8'h50 :
                          (state_q == ST_ROPE_PAIR_REQ) ? 8'h51 :
                          (state_q == ST_ROPE_SCALE0_REQ) ? 8'h52 :
                          (state_q == ST_ROPE_PAIR_SCALE0_REQ) ? 8'h54 :
                          (state_q == ST_ROPE_COS0_REQ) ? 8'h56 :
                          (state_q == ST_ROPE_SIN0_REQ) ? 8'h58 :
                          (state_q == ST_ROPE_WRITE_REQ) ? 8'h5a :
                          (state_q == ST_ATTN_META_REQ) ? 8'h76 :
                          (state_q == ST_ATTN_Q_REQ) ? 8'h70 :
                          (state_q == ST_ATTN_K_REQ) ? 8'h72 :
                          (state_q == ST_ATTN_WRITE_REQ) ? 8'h74 :
                          (state_q == ST_SOFTMAX_SCORE_REQ) ? 8'h80 :
                          (state_q == ST_SOFTMAX_WRITE_REQ) ? 8'h84 :
                          (state_q == ST_AV_PROB_REQ) ? 8'h90 :
                          (state_q == ST_AV_V_REQ) ? 8'h92 :
                          (state_q == ST_AV_WRITE_REQ) ? 8'h94 :
                          (state_q == ST_RES_SRC0_REQ) ? 8'ha0 :
                          (state_q == ST_RES_SRC1_REQ) ? 8'ha1 :
                          (state_q == ST_RES_WRITE_REQ) ? 8'ha2 :
                          (state_q == ST_SILU_META_REQ) ? 8'hb0 :
                          (state_q == ST_SILU_GATE_REQ) ? 8'hb1 :
                          (state_q == ST_SILU_UP_REQ) ? 8'hb2 :
                          (state_q == ST_SILU_WRITE_REQ) ? 8'hb3 :
                          (state_q == ST_COMPOSE_SCORE_REQ) ? 8'hc0 :
                          (state_q == ST_COMPOSE_VALUE_REQ) ? 8'hc1 :
                          (state_q == ST_COMPOSE_OUT_REQ) ? 8'hc2 :
                          (state_q == ST_KV_READ_REQ) ? (8'h60 + {6'd0, kv_phase_q}) :
                          (state_q == ST_KV_WRITE_REQ) ? (8'h68 + {6'd0, kv_phase_q}) : 8'h30;
    assign mem_wvalid_o = (reset_drain_q && write_data_pending_q) ||
                          (!soft_reset_req_w && !reset_drain_q &&
                           !response_fault_pending_w &&
                           (((state_q == ST_WRITE_DATA) && core_out_valid_w) ||
                            (state_q == ST_PROJ_WRITE_DATA) ||
                            (state_q == ST_ROPE_WRITE_DATA) ||
                            (state_q == ST_ATTN_WRITE_DATA) ||
                            (state_q == ST_SOFTMAX_WRITE_DATA) ||
                            (state_q == ST_AV_WRITE_DATA) ||
                            (state_q == ST_RES_WRITE_DATA) ||
                            (state_q == ST_SILU_WRITE_DATA) ||
                            (state_q == ST_COMPOSE_OUT_DATA) ||
                            (state_q == ST_KV_WRITE_DATA)));
    assign mem_wdata_o = reset_drain_q ? reset_write_data_q : mem_write_data_w;
    assign mem_wstrb_o = reset_drain_q ? reset_write_strb_q :
        (((state_q == ST_SILU_WRITE_DATA) && silu_write_final_q &&
         (n_q[3:0] != 4'd0)) ?
        (16'hffff >> (5'd16 - {1'b0, n_q[3:0]})) : 16'hffff);
    assign mem_wtag_o = reset_drain_q ? reset_write_tag_q : mem_write_tag_w;
    wire mem_rready_next_w = (reset_drain_q && read_outstanding_q) ||
                             ((state_q == ST_ACT_RECV) && !core_in_valid_q && core_in_ready_w) ||
                             ((state_q == ST_SCALE_ACT_RECV) && core_scale_act_ready_w) ||
                             (state_q == ST_GAIN_RECV0) ||
                             ((state_q == ST_GAIN_RECV1) && core_gain_ready_w) ||
                             (state_q == ST_PROJ_ACT0_RECV) ||
                             (state_q == ST_PROJ_ACT1_RECV) ||
                             (state_q == ST_PROJ_WGT_RECV) ||
                             (state_q == ST_PROJ_META_RECV) ||
                             (state_q == ST_ROPE_ACT_RECV) ||
                             (state_q == ST_ROPE_PAIR_RECV) ||
                             (state_q == ST_ROPE_SCALE0_RECV) ||
                             (state_q == ST_ROPE_PAIR_SCALE0_RECV) ||
                             (state_q == ST_ROPE_COS0_RECV) ||
                             (state_q == ST_ROPE_SIN0_RECV) ||
                             (state_q == ST_ATTN_META_RECV) ||
                             (state_q == ST_ATTN_Q_RECV) ||
                             (state_q == ST_ATTN_K_RECV) ||
                             (state_q == ST_SOFTMAX_SCORE_RECV) ||
                             (state_q == ST_AV_PROB_RECV) ||
                             (state_q == ST_AV_V_RECV) ||
                             (state_q == ST_RES_SRC0_RECV) ||
                             (state_q == ST_RES_SRC1_RECV) ||
                             (state_q == ST_SILU_META_RECV) ||
                             (state_q == ST_SILU_GATE_RECV) ||
                             (state_q == ST_SILU_UP_RECV) ||
                             (state_q == ST_COMPOSE_SCORE_RECV) ||
                             (state_q == ST_COMPOSE_VALUE_RECV) ||
                             (state_q == ST_KV_READ_RECV);
    assign mem_rready_o = mem_rready_q;
    assign mem_bready_o = (state_q == ST_WRITE_RESP) ||
                          (reset_drain_q && write_response_outstanding_q);

    assign sram_req_valid_o = {SRAM_BANKS{1'b0}};
    assign sram_write_o = {SRAM_BANKS{1'b0}};
    assign sram_addr_o = {SRAM_BANKS*SRAM_ADDR_WIDTH{1'b0}};
    assign sram_wdata_o = {SRAM_BANKS*LANES*ACT_WIDTH{1'b0}};
    assign sram_wstrb_o = {SRAM_BANKS*16{1'b0}};

    wire unused_inputs_w = ^{sram_req_ready_i, sram_rdata_i, sram_rvalid_i, n_q, k_q, sequence_position_q, proj_acc_w, proj_multiplier_q, proj_right_shift_q, proj_output_zero_point_q, compose_context_count_w};

    ace2_rmsnorm_core #(
        .HIDDEN_SIZE(HIDDEN_SIZE),
        .LANES(LANES),
        .ACT_WIDTH(ACT_WIDTH),
        .GAIN_WIDTH(GAIN_WIDTH),
        .ACC_WIDTH(ACC_WIDTH),
        .INV_RMS_FRAC(INV_RMS_FRAC),
        .GAIN_FRAC(GAIN_FRAC)
    ) u_rmsnorm_core (
        .clk_i(clk_i),
        .rst_ni(rst_ni),
        .clear_i(core_clear_q),
        .start_valid_i(core_start_valid_w),
        .start_ready_o(core_start_ready_w),
        .in_valid_i(core_in_valid_w),
        .in_ready_o(core_in_ready_w),
        .in_data_i(shared_payload_q),
        .gain_valid_i(core_gain_valid_w),
        .gain_ready_o(core_gain_ready_w),
        .gain_data_i(gain_beat_w),
        .scale_act_valid_i(core_scale_act_valid_w),
        .scale_act_ready_o(core_scale_act_ready_w),
        .scale_act_data_i(shared_payload_q),
        .out_valid_o(core_out_valid_w),
        .out_ready_i(core_out_ready_w),
        .out_data_o(core_out_data_w),
        .done_valid_o(core_done_valid_w),
        .done_ready_i(core_done_ready_w),
        .sumsq_o(core_sumsq_w),
        .inv_rms_q30_o(core_inv_w),
        .saturation_seen_o(core_saturation_w)
    );

    ace2_w4a8_proj_core #(
        .K_SIZE(MLP_INTERMEDIATE_SIZE),
        .MAC_LANES(PROJ_MAC_LANES),
        .ACT_WIDTH(ACT_WIDTH),
        .WGT_WIDTH(4),
        .ACC_WIDTH(PROJ_ACC_WIDTH)
    ) u_w4a8_proj_core (
        .clk_i(clk_i),
        .rst_ni(rst_ni),
        .clear_i(core_clear_q),
        .start_valid_i(proj_start_valid_q),
        .start_ready_o(proj_start_ready_w),
        .last_group_i(proj_last_group_w),
        .pair_valid_i(proj_pair_valid_w),
        .pair_ready_o(proj_pair_ready_w),
        .act_data_i(proj_act_pair_w),
        .weight_data_i(proj_weight_q),
        .meta_valid_i(proj_meta_valid_w),
        .meta_ready_o(proj_meta_ready_w),
        .multiplier_i(proj_meta_multiplier_w),
        .right_shift_i(proj_meta_right_shift_w),
        .output_zero_point_i(proj_meta_zero_point_w),
        .out_valid_o(proj_out_valid_w),
        .out_ready_i(proj_out_ready_w),
        .out_data_o(proj_out_data_w),
        .acc_o(proj_acc_w),
        .saturation_seen_o(proj_saturation_w)
    );

    ace2_rope_core #(
        .LANES(ROPE_LANES),
        .ACT_WIDTH(ACT_WIDTH),
        .SCALE_WIDTH(16),
        .TRIG_WIDTH(16)
    ) u_rope_core (
        .clk_i(clk_i),
        .rst_ni(rst_ni),
        .clear_i(core_clear_q),
        .start_valid_i(rope_start_valid_q),
        .start_ready_o(rope_start_ready_w),
        .beat_valid_i(rope_beat_valid_w),
        .beat_ready_o(rope_beat_ready_w),
        .act_data_i(rope_act_q),
        .pair_data_i(rope_pair_act_q),
        .scale_data_i(rope_scale_q),
        .pair_scale_data_i(rope_pair_scale_q),
        .cos_data_i(rope_cos_q),
        .sin_data_i(rope_sin_q),
        .second_half_i(rope_second_half_w),
        .out_valid_o(rope_out_valid_w),
        .out_ready_i(rope_out_ready_w),
        .out_data_o(rope_out_data_w),
        .saturation_seen_o(rope_saturation_w)
    );

    ace2_softmax_core #(
        .CONTEXT_MAX(ATTN_CONTEXT_MAX),
        .SCORE_WIDTH(16),
        .PROB_WIDTH(16)
    ) u_softmax_core (
        .clk_i(clk_i),
        .rst_ni(rst_ni),
        .clear_i(softmax_clear_q),
        .start_valid_i(softmax_start_valid_q),
        .start_ready_o(softmax_start_ready_w),
        .context_count_i(n_q),
        .score_data_i(shared_payload_q),
        .out_valid_o(softmax_out_valid_w),
        .out_ready_i(softmax_out_ready_w),
        .prob_data_o(softmax_out_data_w),
        .saturation_seen_o(softmax_saturation_w)
    );

    ace2_attention_compose_core u_attention_compose_core (
        .clk_i(clk_i),
        .rst_ni(rst_ni),
        .clear_i(soft_reset_req_w),
        .command_i(compose_command_w),
        .tile_count_i(compose_tile_count_w),
        .command_allowed_o(compose_command_allowed_w),
        .start_valid_i(compose_start_valid_w),
        .start_authorized_i(descriptor_valid_q),
        .start_ready_o(compose_start_ready_w),
        .score_data_i(shared_payload_q),
        .value_ready_o(compose_value_ready_w),
        .value_valid_i(compose_value_valid_w),
        .value_data_i(compose_value_data_q),
        .out_valid_o(compose_out_valid_w),
        .out_ready_i(compose_out_ready_w),
        .out_data_o(compose_out_data_w),
        .out_last_o(compose_out_last_unused_w),
        .command_done_o(compose_command_done_w),
        .saturation_seen_o(compose_saturation_w),
        .context_count_o(compose_context_count_w)
    );

    ace2_silu_gate_core u_silu_gate_core (
        .clk_i(clk_i),
        .rst_ni(rst_ni),
        .clear_i(core_clear_q),
        .start_valid_i(silu_start_valid_q),
        .start_ready_o(silu_start_ready_w),
        .multiplier_i(silu_multiplier_q),
        .right_shift_i(silu_right_shift_q),
        .output_zero_point_i(silu_zero_point_q),
        .beat_valid_i(silu_beat_valid_q),
        .beat_ready_o(silu_beat_ready_w),
        .lane_count_i(silu_lane_count_w),
        .gate_data_i(silu_gate_data_q),
        .up_data_i(silu_up_data_q),
        .out_valid_o(silu_out_valid_w),
        .out_ready_i(silu_out_ready_w),
        .out_data_o(silu_out_data_w),
        .saturation_seen_o(silu_saturation_w)
    );

    ace2_attention_accumulator u_attention_accumulator (
        .clk_i(clk_i),
        .rst_ni(rst_ni),
        .soft_reset_i(soft_reset_req_w),
        .response_fault_i(response_fault_pending_w),
        .watchdog_fire_i(watchdog_fire_w),
        .zero_i(attn_acc_zero_w),
        .score_add_i(attn_acc_score_add_w),
        .value_add_i(attn_acc_value_add_w),
        .score_addend_i(attn_product_q),
        .value_addend_i(attn_value_shifted_product_w),
        .acc_o(attn_acc_q)
    );

    ace2_state_shadow u_prefix_state_shadow (
        .clk_i(clk_i),
        .rst_ni(rst_ni),
        .state_i(state_commit_w),
        .state_o(prefix_state_q)
    );

    function [63:0] apply_wstrb64;
        input [63:0] old_value;
        input [63:0] new_value;
        input [7:0] strobe;
        integer byte_idx;
        begin
            apply_wstrb64 = old_value;
            for (byte_idx = 0; byte_idx < 8; byte_idx = byte_idx + 1) begin
                if (strobe[byte_idx]) begin
                    apply_wstrb64[byte_idx*8 +: 8] = new_value[byte_idx*8 +: 8];
                end
            end
        end
    endfunction

    function [CONTROL_BITS-1:0] apply_wstrb_control;
        input [CONTROL_BITS-1:0] old_value;
        input [CONTROL_BITS-1:0] new_value;
        input strobe;
        begin
            apply_wstrb_control = old_value;
            if (strobe) begin
                apply_wstrb_control = new_value[CONTROL_BITS-1:0];
            end
        end
    endfunction

    function [STATUS_BITS-1:0] apply_wstrb_status;
        input [STATUS_BITS-1:0] old_value;
        input [STATUS_BITS-1:0] new_value;
        input strobe;
        begin
            apply_wstrb_status = old_value;
            if (strobe) begin
                apply_wstrb_status = new_value[STATUS_BITS-1:0];
            end
        end
    endfunction

    function [63:0] csr_read_value;
        input [7:0] addr_low;
        input high_zero;
        reg [63:0] status_value;
        reg [63:0] control_value;
        reg [63:0] error_value;
        reg [63:0] interrupt_enable_value;
        reg [63:0] interrupt_status_value;
        begin
            status_value = 64'd0;
            status_value[0] = !shell_busy_w;
            status_value[1] = shell_busy_w;
            status_value[2] = halted_on_error_w;
            status_value[3] = !cmd_valid_i;
            status_value[4] = watchdog_active_w;
            control_value = {{(64-CONTROL_BITS){1'b0}}, control_q};
            error_value = {{(64-STATUS_BITS){1'b0}}, error_status_q};
            interrupt_enable_value = {{(64-STATUS_BITS){1'b0}}, interrupt_enable_q};
            interrupt_status_value = {{(64-STATUS_BITS){1'b0}}, interrupt_status_q};
            if (!high_zero) begin
                csr_read_value = 64'd0;
            end else begin
                case (addr_low)
                    ACE2_CSR_ID[7:0]:             csr_read_value = ACE2_ID_VALUE;
                    ACE2_CSR_VERSION[7:0]:        csr_read_value = ACE2_VERSION_VALUE;
                    ACE2_CSR_CAPABILITIES[7:0]:   csr_read_value = ACE2_CAP_VALUE;
                    ACE2_CSR_CONTROL[7:0]:        csr_read_value = control_value;
                    ACE2_CSR_STATUS[7:0]:         csr_read_value = status_value;
                    ACE2_CSR_ERROR_STATUS[7:0]:   csr_read_value = error_value;
                    ACE2_CSR_INTERRUPT_EN[7:0]:   csr_read_value = interrupt_enable_value;
                    ACE2_CSR_INTERRUPT_ST[7:0]:   csr_read_value = interrupt_status_value;
                    ACE2_CSR_DESC_BASE[7:0]:      csr_read_value = desc_base_q;
                    ACE2_CSR_DESC_HEAD[7:0]:      csr_read_value = desc_head_q;
                    ACE2_CSR_DESC_TAIL[7:0]:      csr_read_value = desc_tail_q;
                    ACE2_CSR_WATCHDOG_LIMIT[7:0]: csr_read_value = watchdog_limit_q;
                    ACE2_CSR_PERF_CYCLE[7:0]:     csr_read_value = perf_cycle_q;
                    ACE2_CSR_PERF_BYTE[7:0]:      csr_read_value = perf_byte_q;
                    ACE2_CSR_PERF_TOKEN[7:0]:     csr_read_value = perf_token_q;
                    ACE2_CSR_PERF_STALL[7:0]:     csr_read_value = perf_stall_q;
                    default:                      csr_read_value = 64'd0;
                endcase
            end
        end
    endfunction

    function csr_known_addr;
        input [7:0] addr_low;
        input high_zero;
        begin
            if (!high_zero) begin
                csr_known_addr = 1'b0;
            end else begin
                case (addr_low)
                    ACE2_CSR_ID[7:0],
                    ACE2_CSR_VERSION[7:0],
                    ACE2_CSR_CAPABILITIES[7:0],
                    ACE2_CSR_CONTROL[7:0],
                    ACE2_CSR_STATUS[7:0],
                    ACE2_CSR_ERROR_STATUS[7:0],
                    ACE2_CSR_INTERRUPT_EN[7:0],
                    ACE2_CSR_INTERRUPT_ST[7:0],
                    ACE2_CSR_DESC_BASE[7:0],
                    ACE2_CSR_DESC_HEAD[7:0],
                    ACE2_CSR_DESC_TAIL[7:0],
                    ACE2_CSR_DOORBELL[7:0],
                    ACE2_CSR_WATCHDOG_LIMIT[7:0],
                    ACE2_CSR_PERF_CYCLE[7:0],
                    ACE2_CSR_PERF_BYTE[7:0],
                    ACE2_CSR_PERF_TOKEN[7:0],
                    ACE2_CSR_PERF_STALL[7:0]: csr_known_addr = 1'b1;
                    default:                  csr_known_addr = 1'b0;
                endcase
            end
        end
    endfunction

    always @* begin
        state_d = state_q;
        case (state_q)
            ST_IDLE: begin
            end
            ST_CMD_DISPATCH: begin
                if (descriptor_valid_q) begin
                    if (op_kind_q == OP_KIND_RMSNORM) begin
                        state_d = ST_START;
                    end else if (op_kind_q == OP_KIND_ROPE) begin
                        state_d = ST_ROPE_START;
                    end else if (op_kind_q == OP_KIND_PROJ) begin
                        state_d = ST_PROJ_START;
                    end else if (op_kind_q == OP_KIND_ATTN) begin
                        state_d = ST_ATTN_META_REQ;
                    end else if (op_kind_q == OP_KIND_SOFTMAX) begin
                        state_d = ST_SOFTMAX_SCORE_REQ;
                    end else if (op_kind_q == OP_KIND_ATTN_VALUE) begin
                        state_d = ST_AV_PROB_REQ;
                    end else if (op_kind_q == OP_KIND_ATTN_COMPOSE) begin
                        state_d = ST_COMPOSE_SCORE_REQ;
                    end else if (op_kind_q == OP_KIND_RESIDUAL) begin
                        state_d = ST_RES_SRC0_REQ;
                    end else if (op_kind_q == OP_KIND_SILU) begin
                        state_d = ST_SILU_META_REQ;
                    end else begin
                        state_d = ST_KV_PREP_LOW;
                    end
                end else begin
                    state_d = ST_COMPLETE;
                end
            end
            ST_START: begin
                if (core_start_valid_q && core_start_ready_w) begin
                    state_d = ST_ACT_REQ;
                end
            end
            ST_ACT_REQ: begin
                if (mem_req_ready_i) begin
                    state_d = ST_ACT_RECV;
                end
            end
            ST_ACT_RECV: begin
                if (core_in_valid_q && core_in_ready_w) begin
                    if (act_feed_last_q) begin
                        state_d = ST_SCALE_ACT_REQ;
                    end else begin
                        state_d = ST_ACT_REQ;
                    end
                end
            end
            ST_SCALE_ACT_REQ: begin
                if (mem_req_ready_i) begin
                    state_d = ST_SCALE_ACT_RECV;
                end
            end
            ST_SCALE_ACT_RECV: begin
                if (accepted_read_transition_q) begin
                    state_d = ST_GAIN_REQ0;
                end
            end
            ST_GAIN_REQ0: begin
                if (mem_req_ready_i) begin
                    state_d = ST_GAIN_RECV0;
                end
            end
            ST_GAIN_RECV0: begin
                if (accepted_read_transition_q) begin
                    state_d = ST_GAIN_REQ1;
                end
            end
            ST_GAIN_REQ1: begin
                if (mem_req_ready_i) begin
                    state_d = ST_GAIN_RECV1;
                end
            end
            ST_GAIN_RECV1: begin
                if (accepted_read_transition_q) begin
                    state_d = ST_CORE_WAIT;
                end
            end
            ST_CORE_WAIT: begin
                state_d = ST_CORE_FEED;
            end
            ST_CORE_FEED: begin
                if (core_feed_valid_w && core_scale_act_ready_w && core_gain_ready_w) begin
                    state_d = ST_WRITE_REQ;
                end
            end
            ST_WRITE_REQ: begin
                if (core_out_valid_w && mem_req_ready_i) begin
                    state_d = ST_WRITE_DATA;
                end
            end
            ST_WRITE_DATA: begin
                if (core_write_accepted_w) begin
                    state_d = ST_WRITE_RESP;
                end
            end
            ST_WAIT_DONE: begin
                if (core_done_valid_w && !done_valid_q) begin
                    state_d = ST_COMPLETE;
                end
            end
            ST_COMPLETE: begin
                if (done_valid_q && cmd_done_ready_i) begin
                    state_d = ST_IDLE;
                end
            end
            ST_PROJ_START: begin
                if (proj_start_valid_q && proj_start_ready_w) begin
                    state_d = ST_PROJ_ACT0_REQ;
                end
            end
            ST_PROJ_ACT0_REQ: begin
                if (mem_req_ready_i) begin
                    state_d = ST_PROJ_ACT0_RECV;
                end
            end
            ST_PROJ_ACT0_RECV: begin
                if (accepted_read_transition_q) begin
                    state_d = ST_PROJ_WGT_REQ;
                end
            end
            ST_PROJ_ACT1_REQ: begin
                if (mem_req_ready_i) begin
                    state_d = ST_PROJ_ACT1_RECV;
                end
            end
            ST_PROJ_ACT1_RECV: begin
                if (accepted_read_transition_q) begin
                    state_d = ST_PROJ_WGT_REQ;
                end
            end
            ST_PROJ_WGT_REQ: begin
                if (mem_req_ready_i) begin
                    state_d = ST_PROJ_WGT_RECV;
                end
            end
            ST_PROJ_WGT_RECV: begin
                if (accepted_read_transition_q) begin
                    state_d = ST_PROJ_FEED;
                end
            end
            ST_PROJ_FEED: begin
                if (proj_pair_valid_w && proj_pair_ready_w) begin
                    if (proj_group_idx_q == proj_last_group_w) begin
                        state_d = ST_PROJ_META_REQ;
                    end else begin
                        state_d = ST_PROJ_ACT0_REQ;
                    end
                end
            end
            ST_PROJ_META_REQ: begin
                if (mem_req_ready_i) begin
                    state_d = ST_PROJ_META_RECV;
                end
            end
            ST_PROJ_META_RECV: begin
                if (accepted_read_w) begin
                    state_d = ST_PROJ_META_FEED;
                end
            end
            ST_PROJ_META_FEED: begin
                if (proj_meta_valid_w && proj_meta_ready_w) begin
                    state_d = ST_PROJ_WAIT_OUT;
                end
            end
            ST_PROJ_WAIT_OUT: begin
                if (proj_out_valid_w) begin
                    if (proj_pack_lane_q == PROJ_LAST_PACK_LANE) begin
                        state_d = ST_PROJ_WRITE_REQ;
                    end else begin
                        state_d = ST_PROJ_START;
                    end
                end
            end
            ST_PROJ_WRITE_REQ: begin
                if (mem_req_ready_i) begin
                    state_d = ST_PROJ_WRITE_DATA;
                end
            end
            ST_PROJ_WRITE_DATA: begin
                if (proj_write_accepted_w) begin
                    state_d = ST_WRITE_RESP;
                end
            end
            ST_ROPE_START: begin
                if (rope_start_valid_q && rope_start_ready_w) begin
                    state_d = ST_ROPE_ACT_REQ;
                end
            end
            ST_ROPE_ACT_REQ: begin
                if (mem_req_ready_i) begin
                    state_d = ST_ROPE_ACT_RECV;
                end
            end
            ST_ROPE_ACT_RECV: begin
                if (accepted_read_transition_q) begin
                    state_d = ST_ROPE_PAIR_REQ;
                end
            end
            ST_ROPE_PAIR_REQ: begin
                if (mem_req_ready_i) begin
                    state_d = ST_ROPE_PAIR_RECV;
                end
            end
            ST_ROPE_PAIR_RECV: begin
                if (accepted_read_transition_q) begin
                    state_d = ST_ROPE_SCALE_ADDR;
                end
            end
            ST_ROPE_SCALE_ADDR: begin
                state_d = ST_ROPE_SCALE0_REQ;
            end
            ST_ROPE_SCALE0_REQ: begin
                if (mem_req_ready_i) begin
                    state_d = ST_ROPE_SCALE0_RECV;
                end
            end
            ST_ROPE_SCALE0_RECV: begin
                if (accepted_read_transition_q) begin
                    state_d = ST_ROPE_PAIR_SCALE_ADDR;
                end
            end
            ST_ROPE_PAIR_SCALE_ADDR: begin
                state_d = ST_ROPE_PAIR_SCALE0_REQ;
            end
            ST_ROPE_PAIR_SCALE0_REQ: begin
                if (mem_req_ready_i) begin
                    state_d = ST_ROPE_PAIR_SCALE0_RECV;
                end
            end
            ST_ROPE_PAIR_SCALE0_RECV: begin
                if (accepted_read_transition_q) begin
                    state_d = ST_ROPE_TABLE_ADDR;
                end
            end
            ST_ROPE_TABLE_ADDR: begin
                state_d = ST_ROPE_COS0_REQ;
            end
            ST_ROPE_COS0_REQ: begin
                if (mem_req_ready_i) begin
                    state_d = ST_ROPE_COS0_RECV;
                end
            end
            ST_ROPE_COS0_RECV: begin
                if (accepted_read_transition_q) begin
                    state_d = ST_ROPE_SIN0_REQ;
                end
            end
            ST_ROPE_SIN0_REQ: begin
                if (mem_req_ready_i) begin
                    state_d = ST_ROPE_SIN0_RECV;
                end
            end
            ST_ROPE_SIN0_RECV: begin
                if (accepted_read_transition_q) begin
                    state_d = ST_ROPE_FEED;
                end
            end
            ST_ROPE_FEED: begin
                if (rope_beat_valid_w && rope_beat_ready_w) begin
                    state_d = ST_ROPE_WAIT_OUT;
                end
            end
            ST_ROPE_WAIT_OUT: begin
                if (rope_out_valid_w) begin
                    state_d = (rope_segment_q == ROPE_LAST_SEGMENT) ? ST_ROPE_WRITE_REQ : ST_ROPE_START;
                end
            end
            ST_ROPE_WRITE_REQ: begin
                if (mem_req_ready_i) begin
                    state_d = ST_ROPE_WRITE_DATA;
                end
            end
            ST_ROPE_WRITE_DATA: begin
                if (rope_write_accepted_w) begin
                    state_d = ST_WRITE_RESP;
                end
            end
            ST_KV_PREP_LOW: begin
                state_d = ST_KV_PREP_HIGH;
            end
            ST_KV_PREP_HIGH: begin
                state_d = ST_KV_READ_REQ;
            end
            ST_KV_READ_REQ: begin
                if (mem_req_ready_i) begin
                    state_d = ST_KV_READ_RECV;
                end
            end
            ST_KV_READ_RECV: begin
                if (accepted_read_transition_q) begin
                    state_d = ST_KV_WRITE_REQ;
                end
            end
            ST_KV_WRITE_REQ: begin
                if (mem_req_ready_i) begin
                    state_d = ST_KV_WRITE_DATA;
                end
            end
            ST_KV_WRITE_DATA: begin
                if (kv_write_accepted_w) begin
                    state_d = ST_WRITE_RESP;
                end
            end
            ST_ATTN_META_REQ: begin
                if (mem_req_ready_i) begin
                    state_d = ST_ATTN_META_RECV;
                end
            end
            ST_ATTN_META_RECV: begin
                if (accepted_read_transition_q) begin
                    state_d = attn_score_metadata_invalid_q ?
                              ST_COMPLETE : ST_ATTN_START;
                end
            end
            ST_ATTN_START: begin
                state_d = ST_ATTN_Q_REQ;
            end
            ST_ATTN_Q_REQ: begin
                if (mem_req_ready_i) begin
                    state_d = ST_ATTN_Q_RECV;
                end
            end
            ST_ATTN_Q_RECV: begin
                if (accepted_read_transition_q) begin
                    state_d = ST_ATTN_K_REQ;
                end
            end
            ST_ATTN_K_REQ: begin
                if (mem_req_ready_i) begin
                    state_d = ST_ATTN_K_RECV;
                end
            end
            ST_ATTN_K_RECV: begin
                if (accepted_read_transition_q) begin
                    state_d = ST_ATTN_FEED;
                end
            end
            ST_ATTN_FEED: begin
                state_d =
                    (attn_group_idx_q == ATTN_LAST_GROUP) ?
                    ST_ATTN_SCALE :
                    ST_ATTN_Q_REQ;
            end
            ST_ATTN_SCALE: begin
                state_d = ST_ATTN_MUL;
            end
            ST_ATTN_MUL: begin
                state_d =
                    ((attn_mul_count_q == 6'd31) &&
                     (attn_mul_chunk_q == 2'd3)) ?
                    ST_ATTN_SHIFT_INIT :
                    ST_ATTN_MUL;
            end
            ST_ATTN_SHIFT_INIT: begin
                state_d =
                    (attn_score_right_shift_q == 6'd0) ?
                    ST_ATTN_ROUND :
                    ST_ATTN_SHIFT;
            end
            ST_ATTN_SHIFT: begin
                state_d =
                    (attn_shift_remaining_q == 6'd1) ?
                    ST_ATTN_ROUND :
                    ST_ATTN_SHIFT;
            end
            ST_ATTN_ROUND: begin
                state_d = ST_ATTN_WAIT_OUT;
            end
            ST_ATTN_WAIT_OUT: begin
                state_d = (attn_token_idx_q == attn_last_token_w) ?
                          ST_ATTN_CENTER : ST_ATTN_START;
            end
            ST_ATTN_CENTER: begin
                state_d = ST_ATTN_CENTER_CALC;
            end
            ST_ATTN_CENTER_CALC: begin
                state_d = ST_ATTN_CENTER_QUANT;
            end
            ST_ATTN_CENTER_QUANT: begin
                state_d = ST_ATTN_CENTER_STORE;
            end
            ST_ATTN_CENTER_STORE: begin
                state_d = (attn_center_idx_q == attn_last_token_w) ?
                          ST_ATTN_WRITE_REQ : ST_ATTN_CENTER;
            end
            ST_ATTN_WRITE_REQ: begin
                if (mem_req_ready_i) begin
                    state_d = ST_ATTN_WRITE_DATA;
                end
            end
            ST_ATTN_WRITE_DATA: begin
                if (mem_wready_i) begin
                    state_d = ST_WRITE_RESP;
                end
            end
            ST_SOFTMAX_SCORE_REQ: begin
                if (mem_req_ready_i) begin
                    state_d = ST_SOFTMAX_SCORE_RECV;
                end
            end
            ST_SOFTMAX_SCORE_RECV: begin
                if (accepted_read_transition_q) begin
                    state_d = ST_SOFTMAX_START;
                end
            end
            ST_SOFTMAX_START: begin
                if (softmax_start_valid_q && softmax_start_ready_w) begin
                    state_d = ST_SOFTMAX_WAIT_OUT;
                end
            end
            ST_SOFTMAX_WAIT_OUT: begin
                if (softmax_out_valid_w) begin
                    state_d = ST_SOFTMAX_WRITE_REQ;
                end
            end
            ST_SOFTMAX_WRITE_REQ: begin
                if (mem_req_ready_i) begin
                    state_d = ST_SOFTMAX_WRITE_DATA;
                end
            end
            ST_SOFTMAX_WRITE_DATA: begin
                if (mem_wready_i) begin
                    state_d = ST_WRITE_RESP;
                end
            end
            ST_AV_PROB_REQ: begin
                if (mem_req_ready_i) begin
                    state_d = ST_AV_PROB_RECV;
                end
            end
            ST_AV_PROB_RECV: begin
                if (accepted_read_transition_q) begin
                    state_d = ST_AV_START;
                end
            end
            ST_AV_START: begin
                state_d = ST_AV_V_REQ;
            end
            ST_AV_V_REQ: begin
                if (mem_req_ready_i) begin
                    state_d = ST_AV_V_RECV;
                end
            end
            ST_AV_V_RECV: begin
                if (accepted_read_transition_q) begin
                    state_d = ST_AV_PRODUCT;
                end
            end
            ST_AV_PRODUCT: begin
                state_d = ST_AV_ACCUM;
            end
            ST_AV_ACCUM: begin
                if (attn_value_nibble_q != 2'd3) begin
                    state_d = ST_AV_PRODUCT;
                end else if (attn_token_idx_q != attn_last_token_w) begin
                    state_d = ST_AV_V_REQ;
                end else begin
                    state_d = ST_AV_ROUND;
                end
            end
            ST_AV_ROUND: begin
                state_d = attn_value_last_pack_lane_w ? ST_AV_WRITE_REQ : ST_AV_START;
            end
            ST_AV_WRITE_REQ: begin
                if (mem_req_ready_i) begin
                    state_d = ST_AV_WRITE_DATA;
                end
            end
            ST_AV_WRITE_DATA: begin
                if (mem_wready_i) begin
                    state_d = ST_WRITE_RESP;
                end
            end
            ST_RES_SRC0_REQ: begin
                if (mem_req_ready_i) begin
                    state_d = ST_RES_SRC0_RECV;
                end
            end
            ST_RES_SRC0_RECV: begin
                if (accepted_read_transition_q) begin
                    state_d = ST_RES_SRC1_REQ;
                end
            end
            ST_RES_SRC1_REQ: begin
                if (mem_req_ready_i) begin
                    state_d = ST_RES_SRC1_RECV;
                end
            end
            ST_RES_SRC1_RECV: begin
                if (accepted_read_transition_q) begin
                    state_d = ST_RES_WRITE_REQ;
                end
            end
            ST_RES_WRITE_REQ: begin
                if (mem_req_ready_i) begin
                    state_d = ST_RES_WRITE_DATA;
                end
            end
            ST_RES_WRITE_DATA: begin
                if (mem_wready_i) begin
                    state_d = ST_WRITE_RESP;
                end
            end
            ST_SILU_META_REQ: begin
                if (mem_req_ready_i) begin
                    state_d = ST_SILU_META_RECV;
                end
            end
            ST_SILU_META_RECV: begin
                if (accepted_read_transition_q) begin
                    state_d = ST_SILU_START;
                end
            end
            ST_SILU_START: begin
                if (silu_start_valid_q && silu_start_ready_w) begin
                    state_d = ST_SILU_GATE_REQ;
                end
            end
            ST_SILU_GATE_REQ: begin
                if (mem_req_ready_i) begin
                    state_d = ST_SILU_GATE_RECV;
                end
            end
            ST_SILU_GATE_RECV: begin
                if (accepted_read_transition_q) begin
                    state_d = ST_SILU_UP_REQ;
                end
            end
            ST_SILU_UP_REQ: begin
                if (mem_req_ready_i) begin
                    state_d = ST_SILU_UP_RECV;
                end
            end
            ST_SILU_UP_RECV: begin
                if (accepted_read_transition_q) begin
                    state_d = ST_SILU_FEED;
                end
            end
            ST_SILU_FEED: begin
                if (silu_beat_valid_q && silu_beat_ready_w) begin
                    state_d = ST_SILU_WAIT;
                end
            end
            ST_SILU_WAIT: begin
                if (silu_out_valid_w) begin
                    state_d = (silu_output_half_q || silu_final_input_w) ?
                              ST_SILU_WRITE_REQ : ST_SILU_GATE_REQ;
                end
            end
            ST_SILU_WRITE_REQ: begin
                if (mem_req_ready_i) begin
                    state_d = ST_SILU_WRITE_DATA;
                end
            end
            ST_SILU_WRITE_DATA: begin
                if (mem_wready_i) begin
                    state_d = ST_WRITE_RESP;
                end
            end
            ST_COMPOSE_SCORE_REQ: begin
                if (mem_req_ready_i) begin
                    state_d = ST_COMPOSE_SCORE_RECV;
                end
            end
            ST_COMPOSE_SCORE_RECV: begin
                if (accepted_read_transition_q) begin
                    state_d = ST_COMPOSE_START;
                end
            end
            ST_COMPOSE_START: begin
                if (compose_start_valid_w && compose_start_ready_w) begin
                    state_d = ST_COMPOSE_WAIT;
                end
            end
            ST_COMPOSE_WAIT: begin
                if (compose_out_valid_w) begin
                    state_d = ST_COMPOSE_OUT_REQ;
                end else if (compose_value_ready_w) begin
                    state_d = ST_COMPOSE_VALUE_REQ;
                end else if (compose_command_done_w) begin
                    state_d = ST_COMPLETE;
                end
            end
            ST_COMPOSE_VALUE_REQ: begin
                if (mem_req_ready_i) begin
                    state_d = ST_COMPOSE_VALUE_RECV;
                end
            end
            ST_COMPOSE_VALUE_RECV: begin
                if (accepted_read_transition_q) begin
                    state_d = ST_COMPOSE_VALUE_FEED;
                end
            end
            ST_COMPOSE_VALUE_FEED: begin
                if (compose_value_valid_w && compose_value_ready_w) begin
                    state_d = compose_value_final_w ?
                              ST_COMPOSE_WAIT : ST_COMPOSE_VALUE_REQ;
                end
            end
            ST_COMPOSE_OUT_REQ: begin
                if (mem_req_ready_i) begin
                    state_d = ST_COMPOSE_OUT_DATA;
                end
            end
            ST_COMPOSE_OUT_DATA: begin
                if (mem_wready_i) begin
                    state_d = ST_WRITE_RESP;
                end
            end
            ST_WRITE_RESP: begin
                if (accepted_b_w) begin
                    state_d = ST_WRITE_RETIRE;
                end
            end
            ST_WRITE_RETIRE: begin
                state_d = write_resume_state_q;
            end
            default: begin
                state_d = ST_IDLE;
            end
        endcase
    end

    always @(posedge clk_i) begin
        if (proj_act_recv_q) begin
            proj_act_low_q <= proj_selected_act_w;
        end
        if (gain_low_recv_q) begin
            gain_low_q <= mem_rdata_i;
        end
        if (gain_high_recv_q) begin
            gain_high_q <= mem_rdata_i;
        end
        if (proj_wgt_recv_q) begin
            proj_weight_q <= proj_selected_weight_w;
        end
        if (proj_meta_recv_q) begin
            proj_multiplier_q <= mem_rdata_i[31:0];
            proj_right_shift_q <= mem_rdata_i[37:32];
            proj_output_zero_point_q <= mem_rdata_i[47:40];
        end
        if (rope_act_recv_q) begin
            rope_act_q <= rope_selected_act_w;
        end
        if (rope_pair_recv_q) begin
            rope_pair_act_q <= rope_selected_act_w;
        end
        if (rope_scale_recv_q) begin
            rope_scale_q <= rope_selected_aux_w;
        end
        if (rope_pair_scale_recv_q) begin
            rope_pair_scale_q <= rope_selected_aux_w;
        end
        if (rope_cos_recv_q) begin
            rope_cos_q <= rope_selected_aux_w;
        end
        if (rope_sin_recv_q) begin
            rope_sin_q <= rope_selected_aux_w;
        end
        if (attn_q_recv_q) begin
            attn_q_data_q <= attn_selected_data_w;
        end
        if ((state_q == ST_SILU_META_RECV) && accepted_read_w) begin
            silu_multiplier_q <= mem_rdata_i[31:0];
            silu_right_shift_q <= mem_rdata_i[37:32];
            silu_zero_point_q <= mem_rdata_i[47:40];
        end
        if ((state_q == ST_SILU_GATE_RECV) && accepted_read_w) begin
            silu_gate_data_q <= mem_rdata_i;
        end
    end

    // Prearmed bank-local enables keep the SiLU-up state decode off 128 data
    // enable pins while still capturing the accepted response edge.
    genvar silu_input_bank;
    generate
        for (silu_input_bank = 0; silu_input_bank < SILU_INPUT_BANKS;
             silu_input_bank = silu_input_bank + 1) begin : gen_silu_up_input_bank
            wire [6:0] silu_up_state_q;
            reg silu_up_recv_q;
            reg [PAYLOAD_BANK_WIDTH-1:0] silu_up_data_bank_q;

            assign silu_up_data_q[
                silu_input_bank*PAYLOAD_BANK_WIDTH +: PAYLOAD_BANK_WIDTH
            ] = silu_up_data_bank_q;

            ace2_state_shadow u_silu_up_state_shadow (
                .clk_i(clk_i),
                .rst_ni(rst_ni),
                .state_i(state_commit_w),
                .state_o(silu_up_state_q)
            );

            always @(posedge clk_i or negedge rst_ni) begin
                if (!rst_ni) begin
                    silu_up_recv_q <= 1'b0;
                end else begin
                    silu_up_recv_q <=
                                      (silu_up_state_q == ST_SILU_UP_RECV) &&
                                      !accepted_read_w && !soft_reset_req_w &&
                                      !watchdog_fire_w &&
                                      !response_fault_pending_w;
                end
            end

            always @(posedge clk_i) begin
                if (silu_up_recv_q) begin
                    silu_up_data_bank_q <= mem_rdata_i[
                        silu_input_bank*PAYLOAD_BANK_WIDTH +:
                        PAYLOAD_BANK_WIDTH
                    ];
                end
            end
        end
    endgenerate

    // Bank-local control replicas keep full-payload updates off one high-fanout
    // decode while retaining the original nonblocking-assignment priority.
    genvar payload_bank;
    generate
        for (payload_bank = 0; payload_bank < PAYLOAD_BANKS;
             payload_bank = payload_bank + 1) begin : gen_shared_payload_bank
            wire [6:0] payload_state_q;
            reg payload_recv_q;
            reg [PAYLOAD_BANK_WIDTH-1:0] payload_data_q;
            wire payload_bank_selected_by_proj_w =
                proj_pack_lane_q[3:1] ==
                PAYLOAD_BANK_INDEX_WIDTH'(payload_bank);
            wire payload_bank_selected_by_rope_w =
                rope_segment_q == ROPE_SEGMENT_INDEX_WIDTH'(payload_bank);
            wire payload_bank_selected_by_attn_w =
                attn_center_write_idx_q ==
                ATTN_TOKEN_INDEX_WIDTH'(payload_bank);
            wire payload_bank_selected_by_av_w =
                attn_group_idx_q[3:1] ==
                PAYLOAD_BANK_INDEX_WIDTH'(payload_bank);
            assign shared_payload_q[
                payload_bank*PAYLOAD_BANK_WIDTH +: PAYLOAD_BANK_WIDTH
            ] = payload_data_q;

            ace2_state_shadow u_payload_state_shadow (
                .clk_i(clk_i),
                .rst_ni(rst_ni),
                .state_i(state_commit_w),
                .state_o(payload_state_q)
            );

            always @(posedge clk_i or negedge rst_ni) begin
                if (!rst_ni) begin
                    payload_recv_q <= 1'b0;
                end else begin
                    payload_recv_q <=
                        ((payload_state_q == ST_ACT_RECV) ||
                         (payload_state_q == ST_SCALE_ACT_RECV) ||
                         (payload_state_q == ST_SOFTMAX_SCORE_RECV) ||
                         (payload_state_q == ST_COMPOSE_SCORE_RECV) ||
                         (payload_state_q == ST_KV_READ_RECV) ||
                         (payload_state_q == ST_RES_SRC0_RECV)) &&
                        !accepted_read_w && !soft_reset_req_w &&
                        !watchdog_fire_w && !response_fault_pending_w;
                end
            end

            always @(posedge clk_i) begin
                if (payload_recv_q) begin
                    payload_data_q <= mem_rdata_i[
                        payload_bank*PAYLOAD_BANK_WIDTH +: PAYLOAD_BANK_WIDTH
                    ];
                end
                if ((payload_state_q == ST_CMD_DISPATCH) ||
                    ((payload_state_q == ST_PROJ_WRITE_DATA) &&
                     proj_write_data_active_q && mem_wready_i)) begin
                    payload_data_q <= {PAYLOAD_BANK_WIDTH{1'b0}};
                end
                if (proj_wait_out_active_q && proj_out_valid_w &&
                    payload_bank_selected_by_proj_w) begin
                    payload_data_q[
                        proj_pack_lane_q[0]*ACT_WIDTH +: ACT_WIDTH
                    ] <= proj_out_data_w;
                end
                if ((payload_state_q == ST_ROPE_WAIT_OUT) &&
                    rope_out_valid_w && payload_bank_selected_by_rope_w) begin
                    payload_data_q <= rope_out_data_w;
                end
                if ((payload_state_q == ST_ATTN_CENTER_STORE) &&
                    payload_bank_selected_by_attn_w) begin
                    payload_data_q <= attn_center_score_q;
                end
                if ((payload_state_q == ST_SOFTMAX_WAIT_OUT) &&
                    softmax_out_valid_w) begin
                    payload_data_q <= softmax_out_data_w[
                        payload_bank*PAYLOAD_BANK_WIDTH +: PAYLOAD_BANK_WIDTH
                    ];
                end
                if ((payload_state_q == ST_AV_ROUND) &&
                    payload_bank_selected_by_av_w) begin
                    payload_data_q[
                        attn_group_idx_q[0]*ACT_WIDTH +: ACT_WIDTH
                    ] <= attn_value_output_w;
                end
                if ((payload_state_q == ST_AV_WRITE_DATA) && mem_wready_i) begin
                    payload_data_q <= {PAYLOAD_BANK_WIDTH{1'b0}};
                end
                if ((payload_state_q == ST_SILU_WAIT) && silu_out_valid_w &&
                    (silu_output_half_q ==
                     (payload_bank >= SILU_PAYLOAD_BANKS))) begin
                    payload_data_q <= silu_out_data_w[
                        (payload_bank % SILU_PAYLOAD_BANKS)*
                        PAYLOAD_BANK_WIDTH +: PAYLOAD_BANK_WIDTH
                    ];
                end
                if ((payload_state_q == ST_SILU_WRITE_DATA) && mem_wready_i) begin
                    payload_data_q <= {PAYLOAD_BANK_WIDTH{1'b0}};
                end
            end
        end
    endgenerate

    always @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            accepted_read_transition_q <= 1'b0;
        end else if (soft_reset_req_w || response_fault_pending_w ||
                     watchdog_fire_w) begin
            accepted_read_transition_q <= 1'b0;
        end else begin
            accepted_read_transition_q <= accepted_read_w;
        end
    end

    // Reset reaches the product register through the idle state instead of the
    // high-fanout CSR soft-reset net; every attention dispatch clears it first.
    always @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            attn_product_q <= 32'sd0;
        end else if (watchdog_fire_w) begin
            attn_product_q <= 32'sd0;
        end else if (!response_fault_pending_w) begin
            case (state_q)
                ST_IDLE,
                ST_CMD_DISPATCH,
                ST_ATTN_START,
                ST_AV_START:
                    attn_product_q <= 32'sd0;
                ST_ATTN_K_RECV: begin
                    if (accepted_read_w) begin
                        attn_product_q <= attn_product_w;
                    end
                end
                ST_ATTN_WAIT_OUT: begin
                    if (attn_token_idx_q != attn_last_token_w) begin
                        attn_product_q <= 32'sd0;
                    end
                end
                ST_AV_PRODUCT:
                    attn_product_q <= attn_product_w;
                default: begin
                end
            endcase
        end
    end

    // Keep score requantization state off the high-fanout CSR soft-reset path.
    // A soft reset commits ST_IDLE before another descriptor can be accepted.
    always @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            attn_score_multiplier_q <= 32'sd0;
            attn_score_right_shift_q <= 6'd0;
            attn_score_metadata_invalid_q <= 1'b0;
            attn_scaled_product_q <= 64'sd0;
            attn_shift_value_q <= 64'd0;
            attn_shift_remaining_q <= 6'd0;
            attn_shift_sticky_q <= 1'b0;
            attn_shift_guard_q <= 1'b0;
            attn_rounded_abs_q <= 65'd0;
            attn_score_negative_q <= 1'b0;
            attn_mul_multiplicand_q <= 64'd0;
            attn_mul_multiplier_q <= 32'd0;
            attn_mul_accum_q <= 64'd0;
            attn_mul_count_q <= 6'd0;
            attn_mul_chunk_q <= 2'd0;
            attn_mul_carry_q <= 1'b0;
            attn_mul_negative_q <= 1'b0;
            attn_score_max_magnitude_q <= 65'd0;
            attn_score_max_negative_q <= 1'b0;
            attn_center_idx_q <= {ATTN_TOKEN_INDEX_WIDTH{1'b0}};
            attn_center_write_idx_q <= {ATTN_TOKEN_INDEX_WIDTH{1'b0}};
            attn_center_magnitude_q <= 65'd0;
            attn_center_negative_q <= 1'b0;
            attn_center_delta_q <= 66'd0;
            attn_center_score_q <= 16'd0;
            attn_center_saturation_q <= 1'b0;
        end else if (watchdog_fire_w) begin
            attn_score_multiplier_q <= 32'sd0;
            attn_score_right_shift_q <= 6'd0;
            attn_score_metadata_invalid_q <= 1'b0;
            attn_scaled_product_q <= 64'sd0;
            attn_shift_value_q <= 64'd0;
            attn_shift_remaining_q <= 6'd0;
            attn_shift_sticky_q <= 1'b0;
            attn_shift_guard_q <= 1'b0;
            attn_rounded_abs_q <= 65'd0;
            attn_score_negative_q <= 1'b0;
            attn_mul_multiplicand_q <= 64'd0;
            attn_mul_multiplier_q <= 32'd0;
            attn_mul_accum_q <= 64'd0;
            attn_mul_count_q <= 6'd0;
            attn_mul_chunk_q <= 2'd0;
            attn_mul_carry_q <= 1'b0;
            attn_mul_negative_q <= 1'b0;
            attn_score_max_magnitude_q <= 65'd0;
            attn_score_max_negative_q <= 1'b0;
            attn_center_idx_q <= {ATTN_TOKEN_INDEX_WIDTH{1'b0}};
            attn_center_write_idx_q <= {ATTN_TOKEN_INDEX_WIDTH{1'b0}};
            attn_center_magnitude_q <= 65'd0;
            attn_center_negative_q <= 1'b0;
            attn_center_delta_q <= 66'd0;
            attn_center_score_q <= 16'd0;
            attn_center_saturation_q <= 1'b0;
        end else if (!response_fault_pending_w) begin
            case (state_q)
                ST_IDLE: begin
                    attn_score_multiplier_q <= 32'sd0;
                    attn_score_right_shift_q <= 6'd0;
                    attn_score_metadata_invalid_q <= 1'b0;
                    attn_scaled_product_q <= 64'sd0;
                    attn_shift_value_q <= 64'd0;
                    attn_shift_remaining_q <= 6'd0;
                    attn_shift_sticky_q <= 1'b0;
                    attn_shift_guard_q <= 1'b0;
                    attn_rounded_abs_q <= 65'd0;
                    attn_score_negative_q <= 1'b0;
                    attn_mul_multiplicand_q <= 64'd0;
                    attn_mul_multiplier_q <= 32'd0;
                    attn_mul_accum_q <= 64'd0;
                    attn_mul_count_q <= 6'd0;
                    attn_mul_chunk_q <= 2'd0;
                    attn_mul_carry_q <= 1'b0;
                    attn_mul_negative_q <= 1'b0;
                    attn_score_max_magnitude_q <= 65'd0;
                    attn_score_max_negative_q <= 1'b0;
                    attn_center_idx_q <= {ATTN_TOKEN_INDEX_WIDTH{1'b0}};
                    attn_center_write_idx_q <=
                        {ATTN_TOKEN_INDEX_WIDTH{1'b0}};
                    attn_center_magnitude_q <= 65'd0;
                    attn_center_negative_q <= 1'b0;
                    attn_center_delta_q <= 66'd0;
                    attn_center_score_q <= 16'd0;
                    attn_center_saturation_q <= 1'b0;
                end
                ST_ATTN_META_RECV: begin
                    if (accepted_read_w) begin
                        attn_score_multiplier_q <= mem_rdata_i[31:0];
                        attn_score_right_shift_q <= mem_rdata_i[37:32];
                        attn_score_metadata_invalid_q <=
                            mem_rdata_i[31] || (|mem_rdata_i[127:38]);
                    end
                end
                ST_ATTN_SCALE: begin
                    attn_mul_multiplicand_q <=
                        {32'd0, attn_acc_magnitude_w};
                    attn_mul_multiplier_q <= attn_multiplier_magnitude_w;
                    attn_mul_accum_q <= 64'd0;
                    attn_mul_count_q <= 6'd0;
                    attn_mul_chunk_q <= 2'd0;
                    attn_mul_carry_q <= 1'b0;
                    attn_mul_negative_q <=
                        attn_acc_q[31] ^ attn_score_multiplier_q[31];
                end
                ST_ATTN_MUL: begin
                    case (attn_mul_chunk_q)
                        2'd0:
                            attn_mul_accum_q[15:0] <=
                                attn_mul_chunk_sum_w[15:0];
                        2'd1:
                            attn_mul_accum_q[31:16] <=
                                attn_mul_chunk_sum_w[15:0];
                        2'd2:
                            attn_mul_accum_q[47:32] <=
                                attn_mul_chunk_sum_w[15:0];
                        default:
                            attn_mul_accum_q[63:48] <=
                                attn_mul_chunk_sum_w[15:0];
                    endcase
                    if (attn_mul_chunk_q == 2'd3) begin
                        attn_mul_chunk_q <= 2'd0;
                        attn_mul_carry_q <= 1'b0;
                        attn_mul_multiplicand_q <=
                            attn_mul_multiplicand_q << 1;
                        attn_mul_multiplier_q <=
                            attn_mul_multiplier_q >> 1;
                        attn_mul_count_q <= attn_mul_count_q + 6'd1;
                        if (attn_mul_count_q == 6'd31) begin
                            attn_scaled_product_q <= {
                                attn_mul_chunk_sum_w[15:0],
                                attn_mul_accum_q[47:0]
                            };
                        end
                    end else begin
                        attn_mul_chunk_q <= attn_mul_chunk_q + 2'd1;
                        attn_mul_carry_q <= attn_mul_chunk_sum_w[16];
                    end
                end
                ST_ATTN_SHIFT_INIT: begin
                    attn_score_negative_q <= attn_mul_negative_q;
                    attn_shift_value_q <= attn_scaled_product_q;
                    attn_shift_remaining_q <= attn_score_right_shift_q;
                    attn_shift_sticky_q <= 1'b0;
                    attn_shift_guard_q <= 1'b0;
                end
                ST_ATTN_SHIFT: begin
                    attn_shift_value_q <= attn_shift_value_q >> 1;
                    attn_shift_remaining_q <=
                        attn_shift_remaining_q - 6'd1;
                    if (attn_shift_remaining_q == 6'd1) begin
                        attn_shift_guard_q <= attn_shift_value_q[0];
                    end else begin
                        attn_shift_sticky_q <=
                            attn_shift_sticky_q | attn_shift_value_q[0];
                    end
                end
                    ST_ATTN_ROUND: begin
                        attn_rounded_abs_q <= attn_rounded_abs_w;
                    end
                    ST_ATTN_WAIT_OUT: begin
                        attn_score_magnitude_q[attn_token_idx_q] <=
                            attn_rounded_abs_q;
                        attn_score_negative_row_q[attn_token_idx_q] <=
                            attn_score_negative_q;
                        if ((attn_token_idx_q == {ATTN_TOKEN_INDEX_WIDTH{1'b0}}) ||
                            attn_score_gt_max_w) begin
                            attn_score_max_magnitude_q <= attn_rounded_abs_q;
                            attn_score_max_negative_q <= attn_score_negative_q;
                        end
                        if (attn_token_idx_q == attn_last_token_w) begin
                            attn_center_idx_q <= {ATTN_TOKEN_INDEX_WIDTH{1'b0}};
                        end
                    end
                    ST_ATTN_CENTER: begin
                        attn_center_write_idx_q <= attn_center_idx_q;
                        attn_center_magnitude_q <=
                            attn_center_fetch_magnitude_w;
                        attn_center_negative_q <=
                            attn_center_fetch_negative_w;
                    end
                    ST_ATTN_CENTER_CALC: begin
                        attn_center_delta_q <= attn_center_delta_w;
                    end
                    ST_ATTN_CENTER_QUANT: begin
                        attn_center_score_q <= attn_score_w;
                        attn_center_saturation_q <= attn_saturation_w;
                    end
                    ST_ATTN_CENTER_STORE: begin
                        if (attn_center_idx_q != attn_last_token_w) begin
                            attn_center_idx_q <= attn_center_idx_q +
                                {{(ATTN_TOKEN_INDEX_WIDTH-1){1'b0}}, 1'b1};
                        end
                    end
                default: begin
                end
            endcase
        end
    end

    always @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            silu_input_beat_q <= {SILU_INPUT_BEAT_WIDTH{1'b0}};
        end else begin
            case (state_q)
                ST_IDLE,
                ST_CMD_DISPATCH:
                    silu_input_beat_q <= {SILU_INPUT_BEAT_WIDTH{1'b0}};
                ST_SILU_WAIT: begin
                    if (silu_out_valid_w && !silu_final_input_w) begin
                        silu_input_beat_q <= silu_input_beat_q +
                            SILU_INPUT_BEAT_WIDTH'(1);
                    end
                end
                default: begin
                end
            endcase
        end
    end

    always @(posedge clk_i) begin
        case (state_q)
            ST_ROPE_START: begin
                rope_table_position_base_q <= rope_table_position_base_w;
            end
            ST_ROPE_ACT_REQ: begin
                rope_table_base_low_q <= rope_table_base_low_sum_w;
                rope_table_base_carry_q <= rope_table_base_low_carry_w;
            end
            ST_ROPE_ACT_RECV: begin
                rope_table_base_high_q <= rope_table_base_high_sum_w;
            end
            ST_ROPE_PAIR_RECV: begin
                rope_table_beat_low_q <= rope_table_beat_low_sum_w;
                rope_table_beat_carry_q <= rope_table_beat_low_carry_w;
                rope_scale_addr_low_q <= rope_scale_low_sum_w[31:0];
                rope_scale_addr_carry_q <= rope_scale_low_sum_w[32];
            end
            ST_ROPE_SCALE0_RECV: begin
                rope_scale_addr_low_q <= rope_pair_scale_low_sum_w[31:0];
                rope_scale_addr_carry_q <= rope_pair_scale_low_sum_w[32];
            end
            ST_ROPE_PAIR_SCALE0_RECV: begin
                rope_table_beat_high_q <= rope_table_beat_high_sum_w;
                rope_table_half_low_q <= rope_table_half_low_sum_w[31:0];
                rope_table_half_carry_q <= rope_table_half_low_sum_w[32];
            end
            default: begin
            end
        endcase
    end

    always @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            prefix_mem_req_addr_q <= 64'd0;
        end else if (soft_reset_req_w) begin
            prefix_mem_req_addr_q <= 64'd0;
        end else if (!response_fault_pending_w && !watchdog_fire_w) begin
            case (prefix_state_q)
                ST_CMD_DISPATCH: begin
                    if ((op_kind_q == OP_KIND_ATTN)) begin
                        prefix_mem_req_addr_q <= scale_addr_q;
                    end else if ((op_kind_q == OP_KIND_SOFTMAX) ||
                        (op_kind_q == OP_KIND_ATTN_VALUE) ||
                        (op_kind_q == OP_KIND_ATTN_COMPOSE) ||
                        (op_kind_q == OP_KIND_RESIDUAL)) begin
                        prefix_mem_req_addr_q <= src0_addr_q;
                    end
                end
                ST_START: begin
                    if (core_start_valid_q && core_start_ready_w) begin
                        prefix_mem_req_addr_q <= src0_addr_q;
                    end
                end
                ST_ACT_RECV: begin
                    if (accepted_read_w) begin
                        if (beat_idx_q == LAST_BEAT) begin
                            prefix_mem_req_addr_q <= src0_addr_q;
                        end else begin
                            prefix_mem_req_addr_q <=
                                src0_addr_q + ((beat_idx_ext_w + 64'd1) << 4);
                        end
                    end
                end
                ST_GAIN_RECV0:
                    prefix_mem_req_addr_q <=
                        scale_addr_q + (beat_idx_ext_w << 5) + 64'd16;
                ST_SCALE_ACT_RECV:
                    prefix_mem_req_addr_q <= scale_addr_q + (beat_idx_ext_w << 5);
                ST_GAIN_RECV1:
                    prefix_mem_req_addr_q <= dst_addr_q + (out_idx_ext_w << 4);
                ST_WRITE_DATA: begin
                    if (core_write_accepted_w && (out_idx_q != LAST_BEAT)) begin
                        prefix_mem_req_addr_q <=
                            src0_addr_q + ((beat_idx_ext_w + 64'd1) << 4);
                    end
                end
                ST_ROPE_START: begin
                    if (rope_start_valid_q && rope_start_ready_w) begin
                        prefix_mem_req_addr_q <= src0_addr_q + (beat_idx_ext_w << 4);
                    end
                end
                ST_ROPE_ACT_RECV:
                    prefix_mem_req_addr_q <=
                        src0_addr_q + (rope_pair_beat_idx_ext_w << 4);
                ST_ROPE_SCALE_ADDR,
                ST_ROPE_PAIR_SCALE_ADDR:
                    prefix_mem_req_addr_q <= rope_scale_addr_w;
                ST_ROPE_TABLE_ADDR:
                    prefix_mem_req_addr_q <= rope_table_half_addr_w;
                ST_ROPE_COS0_RECV:
                    prefix_mem_req_addr_q <=
                        add_small64(rope_table_half_addr_w, 12'd32);
                ST_ROPE_WAIT_OUT: begin
                    if (rope_out_valid_w && (rope_segment_q == ROPE_LAST_SEGMENT)) begin
                        prefix_mem_req_addr_q <= dst_addr_q + (beat_idx_ext_w << 4);
                    end
                end
                ST_KV_PREP_HIGH:
                    prefix_mem_req_addr_q <= src0_addr_q;
                ST_KV_READ_RECV: begin
                    if (accepted_read_w) begin
                        prefix_mem_req_addr_q <= kv_dst_addr_q;
                    end
                end
                ST_KV_WRITE_DATA: begin
                    if (kv_write_accepted_w) begin
                        if (!kv_final_write_w && (beat_idx_q == kv_last_beat_w)) begin
                            prefix_mem_req_addr_q <=
                                (kv_phase_q == KV_PHASE_K) ? src1_addr_q : scale_addr_q;
                        end else if (!kv_final_write_w) begin
                            prefix_mem_req_addr_q <=
                                add_small64(kv_src_addr_q, 12'd16);
                        end
                    end
                end
                ST_ATTN_META_RECV:
                    prefix_mem_req_addr_q <= src0_addr_q;
                ST_ATTN_START:
                    prefix_mem_req_addr_q <= attn_q_addr_w;
                ST_ATTN_Q_RECV:
                    prefix_mem_req_addr_q <= attn_k_addr_w;
                ST_ATTN_FEED: begin
                    if (attn_group_idx_q != ATTN_LAST_GROUP) begin
                        prefix_mem_req_addr_q <= attn_next_q_addr_w;
                    end
                end
                ST_ATTN_WAIT_OUT: begin
                    prefix_mem_req_addr_q <=
                        (attn_token_idx_q == attn_last_token_w) ?
                        dst_addr_q : src0_addr_q;
                end
                ST_SOFTMAX_SCORE_REQ:
                    prefix_mem_req_addr_q <= src0_addr_q;
                ST_SOFTMAX_START: begin
                    if (softmax_start_valid_q && softmax_start_ready_w) begin
                        prefix_mem_req_addr_q <= dst_addr_q;
                    end
                end
                ST_AV_START:
                    prefix_mem_req_addr_q <=
                        src1_addr_q + (attn_group_storage_idx_ext_w << 4);
                ST_AV_ACCUM: begin
                    if ((attn_value_nibble_q == 2'd3) &&
                        (attn_token_idx_q != attn_last_token_w)) begin
                        prefix_mem_req_addr_q <=
                            add_small64(prefix_mem_req_addr_q, 12'd64);
                    end
                end
                ST_AV_ROUND: begin
                    if (attn_value_last_pack_lane_w) begin
                        prefix_mem_req_addr_q <= attn_value_write_addr_w;
                    end
                end
                ST_COMPOSE_WAIT: begin
                    if (compose_out_valid_w) begin
                        prefix_mem_req_addr_q <= compose_out_addr_w;
                    end else if (compose_value_ready_w) begin
                        prefix_mem_req_addr_q <= compose_value_addr_w;
                    end
                end
                ST_COMPOSE_VALUE_FEED: begin
                    if (compose_value_valid_w && compose_value_ready_w &&
                        !compose_value_final_w) begin
                        prefix_mem_req_addr_q <=
                            add_small64(prefix_mem_req_addr_q, 12'd16);
                    end
                end
                ST_COMPOSE_OUT_DATA: begin
                    if (mem_wready_i && !compose_out_last_shell_w) begin
                        prefix_mem_req_addr_q <=
                            add_small64(prefix_mem_req_addr_q, 12'd16);
                    end
                end
                ST_RES_SRC0_RECV: begin
                    if (accepted_read_w) begin
                        prefix_mem_req_addr_q <=
                            src1_addr_q + (beat_idx_ext_w << 4);
                    end
                end
                ST_RES_SRC1_RECV: begin
                    if (accepted_read_w) begin
                        prefix_mem_req_addr_q <= dst_addr_q + (beat_idx_ext_w << 4);
                    end
                end
                ST_RES_WRITE_DATA: begin
                    if (residual_write_accepted_w && (beat_idx_q != LAST_BEAT)) begin
                        prefix_mem_req_addr_q <=
                            src0_addr_q + ((beat_idx_ext_w + 64'd1) << 4);
                    end
                end
                default: begin
                end
            endcase
        end
    end

    always @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            proj_mem_req_addr_q <= 64'd0;
        end else if (soft_reset_req_w) begin
            proj_mem_req_addr_q <= 64'd0;
        end else if (!response_fault_pending_w && !watchdog_fire_w) begin
            case (prefix_state_q)
                ST_PROJ_START: begin
                    if (proj_start_valid_q && proj_start_ready_w) begin
                        proj_mem_req_addr_q <=
                            src0_addr_q + proj_row_base_w +
                            (proj_group_storage_idx_ext_w << 4);
                    end
                end
                ST_PROJ_ACT0_RECV,
                ST_PROJ_ACT1_RECV:
                    proj_mem_req_addr_q <=
                        add_proj_offset64(src1_addr_q, proj_weight_request_offset_w);
                ST_PROJ_FEED: begin
                    if (proj_pair_valid_w && proj_pair_ready_w) begin
                        if (proj_group_idx_q == proj_last_group_w) begin
                            proj_mem_req_addr_q <=
                                add_proj_offset64(scale_addr_q, proj_meta_request_offset_w);
                        end else begin
                            proj_mem_req_addr_q <=
                                src0_addr_q + proj_row_base_w +
                                (proj_next_group_storage_idx_ext_w << 4);
                        end
                    end
                end
                default: begin
                end
            endcase
            if (proj_wait_out_active_q && proj_out_valid_w &&
                (proj_pack_lane_q == PROJ_LAST_PACK_LANE)) begin
                proj_mem_req_addr_q <=
                    add_proj_offset64(dst_addr_q, proj_out_request_offset_w);
            end
        end
    end

    always @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            kv_phase_q <= KV_PHASE_K;
            kv_dst_low_q <= 32'd0;
            kv_dst_carry_q <= 1'b0;
            kv_src_addr_q <= 64'd0;
            kv_dst_addr_q <= 64'd0;
            kv_dst_load_base_q <= 1'b0;
            kv_dst_advance_q <= 1'b0;
        end else begin
            kv_dst_load_base_q <= (state_q == ST_KV_PREP_HIGH);
            kv_dst_advance_q <= (state_q == ST_KV_WRITE_DATA) && kv_write_accepted_w && !kv_final_write_w;
            if (kv_dst_load_base_q) begin
                kv_dst_addr_q <= {kv_dst_high_sum_w, kv_dst_low_q};
            end else if (kv_dst_advance_q) begin
                kv_dst_addr_q <= add_small64(kv_dst_addr_q, 12'd16);
            end
            case (state_q)
                ST_CMD_DISPATCH: begin
                    kv_phase_q <= KV_PHASE_K;
                end
                ST_KV_PREP_LOW: begin
                    kv_dst_low_q <= kv_dst_low_sum_w[31:0];
                    kv_dst_carry_q <= kv_dst_low_sum_w[32];
                end
                ST_KV_PREP_HIGH: begin
                    kv_src_addr_q <= src0_addr_q;
                end
                ST_KV_WRITE_DATA: begin
                    if (kv_write_accepted_w) begin
                        if (!kv_final_write_w && (beat_idx_q == kv_last_beat_w)) begin
                            if (kv_phase_q == KV_PHASE_K) begin
                                kv_phase_q <= KV_PHASE_V;
                                kv_src_addr_q <= src1_addr_q;
                            end else begin
                                kv_phase_q <= KV_PHASE_META;
                                kv_src_addr_q <= scale_addr_q;
                            end
                        end else if (!kv_final_write_w) begin
                            kv_src_addr_q <= add_small64(kv_src_addr_q, 12'd16);
                        end
                    end
                end
                default: begin
                end
            endcase
        end
    end

    always @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            perf_token_q <= 64'd0;
        end else if (perf_clear_w) begin
            perf_token_q <= 64'd0;
        end else if (write_retire_success_w && write_completes_descriptor_q) begin
            if (op_kind_q == OP_KIND_PROJ) begin
                perf_token_q <= {perf_token_q[63:32], perf_token_q[31:0] + {{16{1'b0}}, m_q}};
            end else begin
                perf_token_q <= {perf_token_q[63:32], perf_token_q[31:0] + 32'd1};
            end
        end else if (((state_q == ST_WAIT_DONE) && core_done_valid_w &&
                      !done_valid_q) || compose_command_retire_w) begin
            perf_token_q <= {perf_token_q[63:32], perf_token_q[31:0] + 32'd1};
        end
    end

    always @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            desc_tail_q <= 64'd0;
            desc_tail_inc_q <= 1'b0;
        end else begin
            if (desc_tail_inc_q) begin
                desc_tail_q <= desc_tail_q + 64'd1;
            end
            desc_tail_inc_q <=
                (write_retire_success_w && write_completes_descriptor_q) ||
                ((state_q == ST_WAIT_DONE) && core_done_valid_w &&
                 !done_valid_q) || compose_command_retire_w;
        end
    end

    always @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            state_low_q <= ST_IDLE;
            silu_active_q <= 1'b0;
            control_q <= {CONTROL_BITS{1'b0}};
            interrupt_enable_q <= {STATUS_BITS{1'b0}};
            interrupt_status_q <= {STATUS_BITS{1'b0}};
            error_status_q <= {STATUS_BITS{1'b0}};
            desc_base_q <= 64'd0;
            desc_head_q <= 64'd0;
            watchdog_limit_q <= 64'd0;
            watchdog_count_q <= 64'd0;
            watchdog_armed_q <= 1'b0;
            watchdog_fire_q <= 1'b0;
            perf_cycle_q <= 64'd0;
            perf_byte_q <= 64'd0;
            perf_stall_q <= 64'd0;
            perf_cycle_event_q <= 1'b0;
            perf_byte_event_q <= 1'b0;
            perf_stall_event_q <= 1'b0;
            csr_req_valid_q <= 1'b0;
            csr_req_write_q <= 1'b0;
            csr_req_high_zero_q <= 1'b0;
            csr_req_addr_low_q <= 8'd0;
            csr_req_wdata_q <= 64'd0;
            csr_req_wstrb_q <= 8'd0;
            cmd_completion_tag_buf_q <= 16'd0;
            cmd_opcode_buf_q <= 8'd0;
            cmd_flags_buf_q <= 8'd0;
            cmd_layer_id_buf_q <= 8'd0;
            cmd_m_buf_q <= 16'd0;
            cmd_n_buf_q <= 16'd0;
            cmd_k_buf_q <= 16'd0;
            cmd_sequence_position_buf_q <= 16'd0;
            cmd_src0_addr_buf_q <= 64'd0;
            cmd_src1_addr_buf_q <= 64'd0;
            cmd_dst_addr_buf_q <= 64'd0;
            cmd_scale_addr_buf_q <= 64'd0;
            cmd_scratch_addr_buf_q <= 64'd0;
            beat_idx_q <= {BEAT_INDEX_WIDTH{1'b0}};
            out_idx_q <= {BEAT_INDEX_WIDTH{1'b0}};
            completion_tag_q <= 16'd0;
            op_kind_q <= OP_KIND_RMSNORM;
            flags_q <= 8'd0;
            m_q <= 16'd0;
            n_q <= 16'd0;
            k_q <= 16'd0;
            proj_mlp_shape_q <= 1'b0;
            sequence_position_q <= 16'd0;
            src0_addr_q <= 64'd0;
            src1_addr_q <= 64'd0;
            dst_addr_q <= 64'd0;
            scale_addr_q <= 64'd0;
            scratch_addr_q <= 64'd0;
            mem_req_addr_q <= 64'd0;
            silu_mem_req_addr_q <= 64'd0;
            mem_rready_q <= 1'b0;
            read_outstanding_q <= 1'b0;
            write_data_pending_q <= 1'b0;
            write_response_outstanding_q <= 1'b0;
            reset_drain_q <= 1'b0;
            reset_write_data_q <= {LANES*ACT_WIDTH{1'b0}};
            reset_write_tag_q <= 8'd0;
            reset_write_strb_q <= 16'd0;
            descriptor_valid_q <= 1'b0;
            proj_row_idx_q <= {PROJ_ROW_INDEX_WIDTH{1'b0}};
            proj_out_idx_q <= {PROJ_OUT_INDEX_WIDTH{1'b0}};
            proj_group_idx_q <= {PROJ_GROUP_INDEX_WIDTH{1'b0}};
            proj_pack_lane_q <= 4'd0;
            proj_start_valid_q <= 1'b0;
            proj_meta_valid_q <= 1'b0;
            proj_wait_out_active_q <= 1'b0;
            proj_write_data_active_q <= 1'b0;
            proj_done_saturation_q <= 1'b0;
            rope_start_valid_q <= 1'b0;
            rope_done_saturation_q <= 1'b0;
            rope_segment_q <= {ROPE_SEGMENT_INDEX_WIDTH{1'b0}};
            attn_done_saturation_q <= 1'b0;
            attn_q_recv_q <= 1'b0;
            attn_value_nibble_q <= 2'd0;
            softmax_start_valid_q <= 1'b0;
            softmax_out_ready_q <= 1'b0;
            softmax_clear_q <= 1'b0;
            compose_start_valid_q <= 1'b0;
            compose_value_data_q <= 128'd0;
            silu_output_half_q <= 1'b0;
            silu_write_final_q <= 1'b0;
            silu_start_valid_q <= 1'b0;
            silu_beat_valid_q <= 1'b0;
            proj_wgt_recv_q <= 1'b0;
            proj_meta_recv_q <= 1'b0;
            proj_act_recv_q <= 1'b0;
            gain_low_recv_q <= 1'b0;
            gain_high_recv_q <= 1'b0;
            rope_act_recv_q <= 1'b0;
            rope_pair_recv_q <= 1'b0;
            rope_scale_recv_q <= 1'b0;
            rope_pair_scale_recv_q <= 1'b0;
            rope_cos_recv_q <= 1'b0;
            rope_sin_recv_q <= 1'b0;
            cmd_load_decode_q <= 1'b0;
            cmd_load_shape_q <= 1'b0;
            cmd_load_src0_q <= 1'b0;
            cmd_load_src1_q <= 1'b0;
            cmd_load_dst_q <= 1'b0;
            cmd_load_aux_q <= 1'b0;
            cmd_dispatch_q <= 1'b0;
            cmd_decode_busy_q <= 1'b0;
            cmd_ready_q <= 1'b0;
            done_valid_q <= 1'b0;
            done_error_q <= 1'b0;
            done_saturation_q <= 1'b0;
            core_clear_q <= 1'b0;
            core_start_valid_q <= 1'b0;
            core_in_valid_q <= 1'b0;
            act_feed_last_q <= 1'b0;
            forward_progress_q <= 1'b0;
            read_fault_q <= 1'b0;
            write_fault_q <= 1'b0;
            write_response_tag_q <= 8'd0;
            write_resume_state_q <= ST_IDLE;
            write_completes_descriptor_q <= 1'b0;
            csr_rvalid_o <= 1'b0;
            csr_rdata_o <= 64'd0;
            csr_error_o <= 1'b0;
        end else begin
            state_low_q <= state_commit_w;
            if (silu_active_q) begin
                if ((state_q == ST_SILU_WRITE_DATA) && mem_wready_i) begin
                    silu_active_q <= 1'b0;
                end
            end else if ((state_q == ST_CMD_DISPATCH) &&
                         descriptor_valid_q && (op_kind_q == OP_KIND_SILU)) begin
                silu_active_q <= 1'b1;
            end else if ((state_q == ST_WRITE_RETIRE) &&
                         (write_resume_state_q == ST_SILU_GATE_REQ)) begin
                silu_active_q <= 1'b1;
            end
            core_clear_q <= 1'b0;
            mem_rready_q <= mem_rready_next_w;
            read_fault_q <= accepted_read_w && !reset_drain_q && read_response_fault_w;
            write_fault_q <= accepted_b_w && !reset_drain_q && write_response_fault_w;
            if (accepted_read_req_w) begin
                read_outstanding_q <= 1'b1;
            end else if (accepted_read_w) begin
                read_outstanding_q <= 1'b0;
            end
            if (accepted_write_req_w) begin
                write_data_pending_q <= 1'b1;
            end else if (accepted_write_w) begin
                write_data_pending_q <= 1'b0;
            end
            if (accepted_write_w) begin
                write_response_outstanding_q <= 1'b1;
            end else if (accepted_b_w) begin
                write_response_outstanding_q <= 1'b0;
            end
            if (reset_drain_q && !reset_drain_remaining_w) begin
                reset_drain_q <= 1'b0;
            end
            if (accepted_write_w) begin
                write_response_tag_q <= mem_wtag_o;
                write_resume_state_q <= write_resume_state_w;
                write_completes_descriptor_q <= write_completes_descriptor_w;
            end
            // Break equivalent load enables so synthesis keeps command-field fanout split.
            cmd_load_decode_q <= cmd_fire_w;
            cmd_load_shape_q <= cmd_fire_w && !cmd_m_i[15];
            cmd_load_src0_q <= cmd_fire_w && !cmd_src0_addr_i[0];
            cmd_load_src1_q <= cmd_fire_w && !cmd_src1_addr_i[0];
            cmd_load_dst_q <= cmd_fire_w && !cmd_dst_addr_i[0];
            cmd_load_aux_q <= cmd_fire_w && !cmd_scale_addr_i[0];
            cmd_dispatch_q <= cmd_load_decode_q;
            if (cmd_fire_w) begin
                cmd_decode_busy_q <= 1'b1;
            end else if ((state_q == ST_IDLE) && cmd_dispatch_q) begin
                cmd_decode_busy_q <= 1'b0;
            end
            cmd_completion_tag_buf_q <= cmd_completion_tag_i;
            cmd_opcode_buf_q <= cmd_opcode_i;
            cmd_flags_buf_q <= cmd_flags_i;
            cmd_layer_id_buf_q <= cmd_layer_id_i;
            cmd_m_buf_q <= cmd_m_i;
            cmd_n_buf_q <= cmd_n_i;
            cmd_k_buf_q <= cmd_k_i;
            cmd_sequence_position_buf_q <= cmd_sequence_position_i;
            cmd_src0_addr_buf_q <= cmd_src0_addr_i;
            cmd_src1_addr_buf_q <= cmd_src1_addr_i;
            cmd_dst_addr_buf_q <= cmd_dst_addr_i;
            cmd_scale_addr_buf_q <= cmd_scale_addr_i;
            cmd_scratch_addr_buf_q <= cmd_scratch_addr_i;
            if (cmd_load_decode_q) begin
                completion_tag_q <= cmd_completion_tag_buf_q;
                flags_q <= cmd_flags_buf_q;
                descriptor_valid_q <= descriptor_valid_w;
            end
            if (cmd_fire_w) begin
                op_kind_q <= cmd_op_kind_i_w;
            end
            if (cmd_load_shape_q) begin
                m_q <= cmd_m_buf_q;
                n_q <= cmd_n_buf_q;
                k_q <= cmd_k_buf_q;
                proj_mlp_shape_q <=
                    (cmd_k_buf_q == MLP_INTERMEDIATE_SIZE_16);
                sequence_position_q <= cmd_sequence_position_buf_q;
            end
            if (cmd_load_src0_q) begin
                src0_addr_q <= cmd_src0_addr_buf_q;
            end
            if (cmd_load_src1_q) begin
                src1_addr_q <= cmd_src1_addr_buf_q;
            end
            if (cmd_load_dst_q) begin
                dst_addr_q <= cmd_dst_addr_buf_q;
            end
            if (cmd_load_aux_q) begin
                scale_addr_q <= cmd_scale_addr_buf_q;
                scratch_addr_q <= cmd_scratch_addr_buf_q;
            end
            proj_wgt_recv_q <= (state_q == ST_PROJ_WGT_RECV) && !accepted_read_w && !soft_reset_req_w && !watchdog_fire_w && !response_fault_pending_w;
            proj_meta_recv_q <= (state_q == ST_PROJ_META_RECV) && !accepted_read_w && !soft_reset_req_w && !watchdog_fire_w && !response_fault_pending_w;
            proj_act_recv_q <= ((state_q == ST_PROJ_ACT0_RECV) ||
                                (state_q == ST_PROJ_ACT1_RECV)) &&
                               !accepted_read_w && !soft_reset_req_w &&
                               !watchdog_fire_w && !response_fault_pending_w;
            gain_low_recv_q <= ((state_q == ST_GAIN_RECV0) ||
                                (state_q == ST_AV_PROB_RECV) ||
                                (state_q == ST_RES_SRC1_RECV)) &&
                               !accepted_read_w && !soft_reset_req_w &&
                               !watchdog_fire_w && !response_fault_pending_w;
            gain_high_recv_q <= (state_q == ST_GAIN_RECV1) &&
                                !accepted_read_w && !soft_reset_req_w &&
                                !watchdog_fire_w && !response_fault_pending_w;
            rope_act_recv_q <= (state_q == ST_ROPE_ACT_RECV) && !accepted_read_w && !soft_reset_req_w && !watchdog_fire_w && !response_fault_pending_w;
            rope_pair_recv_q <= (state_q == ST_ROPE_PAIR_RECV) && !accepted_read_w && !soft_reset_req_w && !watchdog_fire_w && !response_fault_pending_w;
            rope_scale_recv_q <= (state_q == ST_ROPE_SCALE0_RECV) && !accepted_read_w && !soft_reset_req_w && !watchdog_fire_w && !response_fault_pending_w;
            rope_pair_scale_recv_q <= (state_q == ST_ROPE_PAIR_SCALE0_RECV) && !accepted_read_w && !soft_reset_req_w && !watchdog_fire_w && !response_fault_pending_w;
            rope_cos_recv_q <= (state_q == ST_ROPE_COS0_RECV) && !accepted_read_w && !soft_reset_req_w && !watchdog_fire_w && !response_fault_pending_w;
            rope_sin_recv_q <= (state_q == ST_ROPE_SIN0_RECV) && !accepted_read_w && !soft_reset_req_w && !watchdog_fire_w && !response_fault_pending_w;
            attn_q_recv_q <= ((state_q == ST_ATTN_Q_RECV) || (state_q == ST_AV_V_RECV)) &&
                             !accepted_read_w && !soft_reset_req_w && !watchdog_fire_w &&
                             !response_fault_pending_w;
            if ((state_q == ST_COMPOSE_VALUE_RECV) && accepted_read_w) begin
                compose_value_data_q <= mem_rdata_i;
            end
            core_start_valid_q <= (state_q == ST_START) && !soft_reset_req_w && !response_fault_pending_w;
            proj_start_valid_q <= (state_q == ST_PROJ_START) && !soft_reset_req_w && !response_fault_pending_w;
            proj_meta_valid_q <= (((state_q == ST_PROJ_META_RECV) && accepted_read_w) ||
                                  (proj_meta_valid_q && !proj_meta_ready_w)) &&
                                 !soft_reset_req_w && !watchdog_fire_w && !response_fault_pending_w;
            proj_wait_out_active_q <= (((state_q == ST_PROJ_META_FEED) && proj_meta_valid_q && proj_meta_ready_w) ||
                                       (proj_wait_out_active_q && !proj_out_valid_w)) &&
                                      !soft_reset_req_w && !watchdog_fire_w && !response_fault_pending_w;
            if ((state_q == ST_PROJ_WRITE_REQ) && mem_req_ready_i) begin
                proj_write_data_active_q <= 1'b1;
            end else if (proj_write_accepted_w) begin
                proj_write_data_active_q <= 1'b0;
            end
            rope_start_valid_q <= (state_q == ST_ROPE_START) && !soft_reset_req_w && !response_fault_pending_w;
            softmax_start_valid_q <= (state_q == ST_SOFTMAX_START) && !soft_reset_req_w && !response_fault_pending_w;
            softmax_out_ready_q <= (state_q == ST_SOFTMAX_WAIT_OUT) &&
                                   !soft_reset_req_w && !response_fault_pending_w;
            softmax_clear_q <= core_clear_q;
            compose_start_valid_q <= (state_q == ST_COMPOSE_START) &&
                                     !soft_reset_req_w &&
                                     !response_fault_pending_w;
            silu_start_valid_q <= (state_q == ST_SILU_START) &&
                                  !soft_reset_req_w && !response_fault_pending_w;
            silu_beat_valid_q <= (state_q == ST_SILU_FEED) &&
                                 !soft_reset_req_w && !response_fault_pending_w;
            forward_progress_q <= forward_progress_w;
            if (core_in_valid_q && core_in_ready_w) begin
                core_in_valid_q <= 1'b0;
                act_feed_last_q <= 1'b0;
            end
            cmd_ready_q <= (state_q == ST_IDLE) && !done_valid_q && !reset_drain_q &&
                           control_enable_w && !halted_on_error_w && !cmd_fire_w &&
                           !cmd_decode_busy_q;
            control_q[1] <= 1'b0;
            csr_error_o <= 1'b0;
            perf_cycle_event_q <= control_enable_w;
            perf_byte_event_q <= accepted_read_w || accepted_write_w;
            perf_stall_event_q <= mem_stall_w;
            if (perf_cycle_event_q) begin
                perf_cycle_q <= perf_cycle_q + 64'd1;
            end
            if (perf_stall_event_q) begin
                perf_stall_q <= perf_stall_q + 64'd1;
            end
            watchdog_fire_q <= 1'b0;
            if (!watchdog_active_w || forward_progress_q) begin
                watchdog_count_q <= watchdog_limit_q;
                watchdog_armed_q <= watchdog_active_w && (watchdog_limit_q != 64'd0);
            end else if (!watchdog_armed_q) begin
                watchdog_count_q <= watchdog_limit_q;
                watchdog_armed_q <= 1'b1;
            end else if (watchdog_last_tick_w) begin
                watchdog_count_q <= 64'd0;
                watchdog_armed_q <= 1'b0;
                watchdog_fire_q <= 1'b1;
            end else begin
                watchdog_count_q <= watchdog_count_q - 64'd1;
            end
            if (perf_byte_event_q) begin
                perf_byte_q <= perf_byte_q + 64'd16;
            end
            if (core_numeric_saturation_w) begin
                error_status_q[ACE2_ERR_NUMERIC] <= 1'b1;
            end
            if (core_numeric_saturation_w || proj_numeric_saturation_w ||
                attn_numeric_saturation_w || softmax_numeric_saturation_w ||
                silu_numeric_saturation_w ||
                (residual_write_accepted_w && residual_saturation_w)) begin
                done_saturation_q <= 1'b1;
            end

            if (csr_rvalid_o && csr_rready_i) begin
                csr_rvalid_o <= 1'b0;
            end
            if (csr_fire_w) begin
                csr_req_valid_q <= 1'b1;
                csr_req_write_q <= csr_write_i;
                csr_req_high_zero_q <= (csr_addr_i[31:8] == 24'd0);
                csr_req_addr_low_q <= csr_addr_i[7:0];
                csr_req_wdata_q <= csr_wdata_i;
                csr_req_wstrb_q <= csr_wstrb_i;
            end
            if (csr_req_valid_q && (!csr_rvalid_o || csr_rready_i)) begin
                csr_req_valid_q <= 1'b0;
                csr_rvalid_o <= 1'b1;
                csr_rdata_o <= csr_read_value(csr_req_addr_low_q, csr_req_high_zero_q);
                csr_error_o <= !csr_known_addr(csr_req_addr_low_q, csr_req_high_zero_q);
                if (!csr_known_addr(csr_req_addr_low_q, csr_req_high_zero_q) && strict_errors_w) begin
                    error_status_q[ACE2_ERR_RESERVED] <= 1'b1;
                    interrupt_status_q[ACE2_ERR_DESCRIPTOR] <= 1'b1;
                end
                if (csr_req_write_q && csr_req_high_zero_q) begin
                    case (csr_req_addr_low_q)
                        ACE2_CSR_CONTROL[7:0]: begin
                            control_q <= apply_wstrb_control(control_q, csr_req_wdata_q[CONTROL_BITS-1:0], csr_req_wstrb_q[0]);
                        end
                        ACE2_CSR_ERROR_STATUS[7:0]: begin
                            error_status_q <= error_status_q & ~apply_wstrb_status({STATUS_BITS{1'b0}}, csr_req_wdata_q[STATUS_BITS-1:0], csr_req_wstrb_q[0]);
                        end
                        ACE2_CSR_INTERRUPT_EN[7:0]: begin
                            interrupt_enable_q <= apply_wstrb_status(interrupt_enable_q, csr_req_wdata_q[STATUS_BITS-1:0], csr_req_wstrb_q[0]);
                        end
                        ACE2_CSR_INTERRUPT_ST[7:0]: begin
                            interrupt_status_q <= interrupt_status_q & ~apply_wstrb_status({STATUS_BITS{1'b0}}, csr_req_wdata_q[STATUS_BITS-1:0], csr_req_wstrb_q[0]);
                        end
                        ACE2_CSR_DESC_BASE[7:0]: begin
                            desc_base_q <= apply_wstrb64(desc_base_q, csr_req_wdata_q, csr_req_wstrb_q);
                        end
                        ACE2_CSR_DESC_HEAD[7:0]: begin
                            desc_head_q <= apply_wstrb64(desc_head_q, csr_req_wdata_q, csr_req_wstrb_q);
                        end
                        ACE2_CSR_DOORBELL[7:0]: begin
                            desc_head_q <= desc_head_q + 64'd1;
                        end
                        ACE2_CSR_WATCHDOG_LIMIT[7:0]: begin
                            watchdog_limit_q <= apply_wstrb64(watchdog_limit_q, csr_req_wdata_q, csr_req_wstrb_q);
                        end
                        default: begin
                        end
                    endcase
                end
            end

            if (perf_clear_w) begin
                perf_cycle_q <= 64'd0;
                perf_byte_q <= 64'd0;
                perf_stall_q <= 64'd0;
                perf_cycle_event_q <= 1'b0;
                perf_byte_event_q <= 1'b0;
                perf_stall_event_q <= 1'b0;
                control_q[4] <= 1'b0;
            end

            if (soft_reset_req_w) begin
                silu_active_q <= 1'b0;
                cmd_load_decode_q <= 1'b0;
                cmd_load_shape_q <= 1'b0;
                cmd_load_src0_q <= 1'b0;
                cmd_load_src1_q <= 1'b0;
                cmd_load_dst_q <= 1'b0;
                cmd_load_aux_q <= 1'b0;
                cmd_dispatch_q <= 1'b0;
                cmd_decode_busy_q <= 1'b0;
                cmd_ready_q <= 1'b0;
                interrupt_status_q <= {STATUS_BITS{1'b0}};
                error_status_q <= {STATUS_BITS{1'b0}};
                if (shell_busy_w) begin
                    interrupt_status_q[ACE2_ERR_RESET_BUSY] <= 1'b1;
                    error_status_q[ACE2_ERR_RESET_BUSY] <= 1'b1;
                end
                done_valid_q <= 1'b0;
                done_error_q <= 1'b0;
                beat_idx_q <= {BEAT_INDEX_WIDTH{1'b0}};
                out_idx_q <= {BEAT_INDEX_WIDTH{1'b0}};
                proj_row_idx_q <= {PROJ_ROW_INDEX_WIDTH{1'b0}};
                proj_out_idx_q <= {PROJ_OUT_INDEX_WIDTH{1'b0}};
                proj_group_idx_q <= {PROJ_GROUP_INDEX_WIDTH{1'b0}};
                proj_pack_lane_q <= 4'd0;
                proj_start_valid_q <= 1'b0;
                proj_meta_valid_q <= 1'b0;
                proj_wait_out_active_q <= 1'b0;
                proj_write_data_active_q <= 1'b0;
                proj_done_saturation_q <= 1'b0;
                rope_start_valid_q <= 1'b0;
                rope_done_saturation_q <= 1'b0;
                rope_segment_q <= {ROPE_SEGMENT_INDEX_WIDTH{1'b0}};
                attn_token_idx_q <= {ATTN_TOKEN_INDEX_WIDTH{1'b0}};
                attn_group_idx_q <= {ATTN_GROUP_INDEX_WIDTH{1'b0}};
                attn_done_saturation_q <= 1'b0;
                attn_q_recv_q <= 1'b0;
                attn_value_nibble_q <= 2'd0;
                softmax_start_valid_q <= 1'b0;
                compose_start_valid_q <= 1'b0;
                proj_wgt_recv_q <= 1'b0;
                proj_meta_recv_q <= 1'b0;
                proj_act_recv_q <= 1'b0;
                gain_low_recv_q <= 1'b0;
                gain_high_recv_q <= 1'b0;
                rope_act_recv_q <= 1'b0;
                rope_pair_recv_q <= 1'b0;
                rope_scale_recv_q <= 1'b0;
                rope_pair_scale_recv_q <= 1'b0;
                rope_cos_recv_q <= 1'b0;
                rope_sin_recv_q <= 1'b0;
                done_saturation_q <= 1'b0;
                mem_req_addr_q <= 64'd0;
                silu_mem_req_addr_q <= 64'd0;
                reset_drain_q <= reset_drain_remaining_w;
                if (write_data_pending_q) begin
                    reset_write_data_q <= mem_write_data_w;
                    reset_write_tag_q <= mem_write_tag_w;
                    reset_write_strb_q <= mem_wstrb_o;
                end
                mem_rready_q <= read_outstanding_q && !accepted_read_w;
                watchdog_count_q <= 64'd0;
                watchdog_armed_q <= 1'b0;
                watchdog_fire_q <= 1'b0;
                core_clear_q <= 1'b1;
                core_in_valid_q <= 1'b0;
                act_feed_last_q <= 1'b0;
                proj_start_valid_q <= 1'b0;
                proj_wait_out_active_q <= 1'b0;
                proj_write_data_active_q <= 1'b0;
                rope_start_valid_q <= 1'b0;
                attn_done_saturation_q <= 1'b0;
                attn_q_recv_q <= 1'b0;
                attn_value_nibble_q <= 2'd0;
                softmax_start_valid_q <= 1'b0;
                cmd_load_decode_q <= 1'b0;
                cmd_load_shape_q <= 1'b0;
                cmd_load_src0_q <= 1'b0;
                cmd_load_src1_q <= 1'b0;
                cmd_load_dst_q <= 1'b0;
                cmd_load_aux_q <= 1'b0;
                cmd_dispatch_q <= 1'b0;
                cmd_decode_busy_q <= 1'b0;
                mem_rready_q <= 1'b0;
                forward_progress_q <= 1'b0;
                read_fault_q <= 1'b0;
                write_fault_q <= 1'b0;
                write_response_tag_q <= 8'd0;
                write_resume_state_q <= ST_IDLE;
                write_completes_descriptor_q <= 1'b0;
            end else if (response_fault_pending_w) begin
                silu_active_q <= 1'b0;
                done_valid_q <= 1'b1;
                done_error_q <= 1'b1;
                error_status_q[ACE2_ERR_MEMORY] <= 1'b1;
                interrupt_status_q[ACE2_ERR_MEMORY] <= 1'b1;
                core_clear_q <= 1'b1;
                core_in_valid_q <= 1'b0;
                act_feed_last_q <= 1'b0;
                proj_start_valid_q <= 1'b0;
                proj_meta_valid_q <= 1'b0;
                proj_wait_out_active_q <= 1'b0;
                proj_write_data_active_q <= 1'b0;
                proj_act_recv_q <= 1'b0;
                gain_low_recv_q <= 1'b0;
                gain_high_recv_q <= 1'b0;
                rope_start_valid_q <= 1'b0;
                attn_q_recv_q <= 1'b0;
                softmax_start_valid_q <= 1'b0;
                forward_progress_q <= 1'b0;
            end else if (watchdog_fire_w) begin
                silu_active_q <= 1'b0;
                done_valid_q <= 1'b1;
                done_error_q <= 1'b1;
                error_status_q[ACE2_ERR_WATCHDOG] <= 1'b1;
                interrupt_status_q[ACE2_ERR_WATCHDOG] <= 1'b1;
                watchdog_count_q <= 64'd0;
                watchdog_armed_q <= 1'b0;
                watchdog_fire_q <= 1'b0;
                core_clear_q <= 1'b1;
                core_in_valid_q <= 1'b0;
                act_feed_last_q <= 1'b0;
                proj_start_valid_q <= 1'b0;
                proj_meta_valid_q <= 1'b0;
                proj_write_data_active_q <= 1'b0;
                rope_start_valid_q <= 1'b0;
                attn_q_recv_q <= 1'b0;
                softmax_start_valid_q <= 1'b0;
                cmd_load_decode_q <= 1'b0;
                cmd_load_shape_q <= 1'b0;
                cmd_load_src0_q <= 1'b0;
                cmd_load_src1_q <= 1'b0;
                cmd_load_dst_q <= 1'b0;
                cmd_load_aux_q <= 1'b0;
                cmd_dispatch_q <= 1'b0;
                cmd_decode_busy_q <= 1'b0;
                forward_progress_q <= 1'b0;
            end else begin
                if (silu_active_q) begin
                    mem_req_addr_q <= 64'd0;
                end
                case (state_q)
                    ST_IDLE: begin
                    end
                    ST_CMD_DISPATCH: begin
                        beat_idx_q <= {BEAT_INDEX_WIDTH{1'b0}};
                        out_idx_q <= {BEAT_INDEX_WIDTH{1'b0}};
                        silu_output_half_q <= 1'b0;
                        silu_write_final_q <= 1'b0;
                        proj_row_idx_q <= {PROJ_ROW_INDEX_WIDTH{1'b0}};
                        proj_out_idx_q <= {PROJ_OUT_INDEX_WIDTH{1'b0}};
                        proj_group_idx_q <= {PROJ_GROUP_INDEX_WIDTH{1'b0}};
                        proj_pack_lane_q <= 4'd0;
                        proj_done_saturation_q <= 1'b0;
                        rope_done_saturation_q <= 1'b0;
                        rope_segment_q <= {ROPE_SEGMENT_INDEX_WIDTH{1'b0}};
                        attn_token_idx_q <= {ATTN_TOKEN_INDEX_WIDTH{1'b0}};
                        attn_group_idx_q <= {ATTN_GROUP_INDEX_WIDTH{1'b0}};
                        attn_done_saturation_q <= 1'b0;
                        attn_value_nibble_q <= 2'd0;
                        done_saturation_q <= 1'b0;
                        done_valid_q <= 1'b0;
                        done_error_q <= 1'b0;
                        if (op_kind_q == OP_KIND_ATTN) begin
                            mem_req_addr_q <= scale_addr_q;
                        end else if ((op_kind_q == OP_KIND_SOFTMAX) ||
                            (op_kind_q == OP_KIND_ATTN_VALUE) ||
                            (op_kind_q == OP_KIND_ATTN_COMPOSE) ||
                            (op_kind_q == OP_KIND_RESIDUAL)) begin
                            mem_req_addr_q <= src0_addr_q;
                        end else if (op_kind_q == OP_KIND_SILU) begin
                            silu_mem_req_addr_q <= scale_addr_q;
                        end
                    end
                    ST_START: begin
                        if (core_start_valid_q && core_start_ready_w) begin
                            mem_req_addr_q <= src0_addr_q;
                        end
                    end
                    ST_ACT_RECV: begin
                        if (accepted_read_w) begin
                            core_in_valid_q <= 1'b1;
                            act_feed_last_q <= (beat_idx_q == LAST_BEAT);
                            if (beat_idx_q == LAST_BEAT) begin
                                beat_idx_q <= {BEAT_INDEX_WIDTH{1'b0}};
                                mem_req_addr_q <= src0_addr_q;
                            end else begin
                                beat_idx_q <= beat_idx_q + {{(BEAT_INDEX_WIDTH-1){1'b0}}, 1'b1};
                                mem_req_addr_q <= src0_addr_q + ((beat_idx_ext_w + 64'd1) << 4);
                            end
                        end
                    end
                    ST_GAIN_RECV0: begin
                        mem_req_addr_q <= scale_addr_q + (beat_idx_ext_w << 5) + 64'd16;
                    end
                    ST_SCALE_ACT_RECV: begin
                        mem_req_addr_q <= scale_addr_q + (beat_idx_ext_w << 5);
                    end
                    ST_GAIN_RECV1: begin
                        mem_req_addr_q <= dst_addr_q + (out_idx_ext_w << 4);
                    end
                    ST_WRITE_DATA: begin
                        if (core_write_accepted_w) begin
                            if (out_idx_q == LAST_BEAT) begin
                                out_idx_q <= {BEAT_INDEX_WIDTH{1'b0}};
                            end else begin
                                out_idx_q <= out_idx_q + {{(BEAT_INDEX_WIDTH-1){1'b0}}, 1'b1};
                                beat_idx_q <= beat_idx_q + {{(BEAT_INDEX_WIDTH-1){1'b0}}, 1'b1};
                                mem_req_addr_q <= src0_addr_q + ((beat_idx_ext_w + 64'd1) << 4);
                            end
                        end
                    end
                    ST_PROJ_START: begin
                        if (proj_start_valid_q && proj_start_ready_w) begin
                            proj_group_idx_q <= {PROJ_GROUP_INDEX_WIDTH{1'b0}};
                            mem_req_addr_q <= src0_addr_q + proj_row_base_w + (proj_group_storage_idx_ext_w << 4);
                        end
                    end
                    ST_PROJ_ACT0_RECV: begin
                        mem_req_addr_q <= add_proj_offset64(src1_addr_q, proj_weight_request_offset_w);
                    end
                    ST_PROJ_ACT1_RECV: begin
                        mem_req_addr_q <= add_proj_offset64(src1_addr_q, proj_weight_request_offset_w);
                    end
                    ST_PROJ_WGT_RECV: begin
                    end
                    ST_PROJ_FEED: begin
                        if (proj_pair_valid_w && proj_pair_ready_w) begin
                            if (proj_group_idx_q == proj_last_group_w) begin
                                proj_group_idx_q <= {PROJ_GROUP_INDEX_WIDTH{1'b0}};
                                mem_req_addr_q <= add_proj_offset64(scale_addr_q, proj_meta_request_offset_w);
                            end else begin
                                proj_group_idx_q <= proj_group_idx_q + {{(PROJ_GROUP_INDEX_WIDTH-1){1'b0}}, 1'b1};
                                mem_req_addr_q <= src0_addr_q + proj_row_base_w + (proj_next_group_storage_idx_ext_w << 4);
                            end
                        end
                    end
                    ST_PROJ_META_RECV: begin
                    end
                    ST_PROJ_WAIT_OUT: begin
                    end
                    ST_PROJ_WRITE_DATA: begin
                    end
                    ST_ROPE_START: begin
                        if (rope_start_valid_q && rope_start_ready_w) begin
                            mem_req_addr_q <= src0_addr_q + (beat_idx_ext_w << 4);
                        end
                    end
                    ST_ROPE_ACT_RECV: begin
                        mem_req_addr_q <= src0_addr_q + (rope_pair_beat_idx_ext_w << 4);
                    end
                    ST_ROPE_PAIR_RECV: begin
                    end
                    ST_ROPE_SCALE_ADDR: begin
                        mem_req_addr_q <= rope_scale_addr_w;
                    end
                    ST_ROPE_SCALE0_RECV: begin
                    end
                    ST_ROPE_PAIR_SCALE_ADDR: begin
                        mem_req_addr_q <= rope_scale_addr_w;
                    end
                    ST_ROPE_PAIR_SCALE0_RECV: begin
                    end
                    ST_ROPE_TABLE_ADDR: begin
                        mem_req_addr_q <= rope_table_half_addr_w;
                    end
                    ST_ROPE_COS0_RECV: begin
                        mem_req_addr_q <= add_small64(rope_table_half_addr_w, 12'd32);
                    end
                    ST_ROPE_SIN0_RECV: begin
                    end
                    ST_ROPE_WAIT_OUT: begin
                        if (rope_out_valid_w) begin
                            rope_done_saturation_q <= rope_done_saturation_q | rope_saturation_w;
                            done_saturation_q <= done_saturation_q | rope_saturation_w;
                            if (rope_segment_q == ROPE_LAST_SEGMENT) begin
                                mem_req_addr_q <= dst_addr_q + (beat_idx_ext_w << 4);
                            end else begin
                                rope_segment_q <= rope_segment_q + {{(ROPE_SEGMENT_INDEX_WIDTH-1){1'b0}}, 1'b1};
                            end
                        end
                    end
                    ST_ROPE_WRITE_DATA: begin
                        if (rope_write_accepted_w) begin
                            if (beat_idx_q != rope_last_beat_w) begin
                                beat_idx_q <= beat_idx_q + {{(BEAT_INDEX_WIDTH-1){1'b0}}, 1'b1};
                                rope_segment_q <= {ROPE_SEGMENT_INDEX_WIDTH{1'b0}};
                            end
                        end
                    end
                    ST_KV_PREP_HIGH: begin
                        mem_req_addr_q <= src0_addr_q;
                    end
                    ST_KV_READ_RECV: begin
                        if (accepted_read_w) begin
                            mem_req_addr_q <= kv_dst_addr_q;
                        end
                    end
                    ST_KV_WRITE_DATA: begin
                        if (kv_write_accepted_w) begin
                            if (!kv_final_write_w && (beat_idx_q == kv_last_beat_w)) begin
                                beat_idx_q <= {BEAT_INDEX_WIDTH{1'b0}};
                                if (kv_phase_q == KV_PHASE_K) begin
                                    mem_req_addr_q <= src1_addr_q;
                                end else begin
                                    mem_req_addr_q <= scale_addr_q;
                                end
                            end else if (!kv_final_write_w) begin
                                beat_idx_q <= beat_idx_q + {{(BEAT_INDEX_WIDTH-1){1'b0}}, 1'b1};
                                mem_req_addr_q <= add_small64(kv_src_addr_q, 12'd16);
                            end
                        end
                    end
                    ST_ATTN_META_RECV: begin
                        if (accepted_read_w) begin
                            mem_req_addr_q <= src0_addr_q;
                        end
                    end
                    ST_ATTN_START: begin
                        mem_req_addr_q <= attn_q_addr_w;
                    end
                    ST_ATTN_Q_RECV: begin
                        mem_req_addr_q <= attn_k_addr_w;
                    end
                    ST_ATTN_K_RECV: begin
                        if (accepted_read_w) begin
                        end
                    end
                    ST_ATTN_FEED: begin
                        if (attn_group_idx_q == ATTN_LAST_GROUP) begin
                            attn_group_idx_q <= {ATTN_GROUP_INDEX_WIDTH{1'b0}};
                        end else begin
                            attn_group_idx_q <= attn_group_idx_q + {{(ATTN_GROUP_INDEX_WIDTH-1){1'b0}}, 1'b1};
                            mem_req_addr_q <= attn_next_q_addr_w;
                        end
                    end
                    ST_ATTN_WAIT_OUT: begin
                        if (attn_token_idx_q == attn_last_token_w) begin
                            mem_req_addr_q <= dst_addr_q;
                        end else begin
                            attn_token_idx_q <= attn_token_idx_q + {{(ATTN_TOKEN_INDEX_WIDTH-1){1'b0}}, 1'b1};
                            attn_group_idx_q <= {ATTN_GROUP_INDEX_WIDTH{1'b0}};
                            mem_req_addr_q <= src0_addr_q;
                        end
                    end
                    ST_ATTN_CENTER_STORE: begin
                        attn_done_saturation_q <=
                            attn_done_saturation_q |
                            attn_center_saturation_q;
                        done_saturation_q <=
                            done_saturation_q | attn_center_saturation_q;
                    end
                    ST_ATTN_WRITE_DATA: begin
                    end
                    ST_SOFTMAX_SCORE_REQ: begin
                        mem_req_addr_q <= src0_addr_q;
                    end
                    ST_SOFTMAX_SCORE_RECV: begin
                    end
                    ST_SOFTMAX_START: begin
                        if (softmax_start_valid_q && softmax_start_ready_w) begin
                            mem_req_addr_q <= dst_addr_q;
                        end
                    end
                    ST_SOFTMAX_WAIT_OUT: begin
                        if (softmax_out_valid_w) begin
                            done_saturation_q <= done_saturation_q | softmax_saturation_w;
                        end
                    end
                    ST_SOFTMAX_WRITE_DATA: begin
                    end
                    ST_COMPOSE_WAIT: begin
                        if (compose_command_done_w && !done_valid_q) begin
                            done_valid_q <= 1'b1;
                            done_error_q <= 1'b0;
                            interrupt_status_q[0] <= 1'b1;
                        end
                    end
                    ST_COMPOSE_VALUE_FEED: begin
                        if (compose_value_valid_w && compose_value_ready_w) begin
                            if (beat_idx_q[1:0] == 2'd3) begin
                                beat_idx_q <= {BEAT_INDEX_WIDTH{1'b0}};
                                if (attn_token_idx_q == attn_last_token_w) begin
                                    attn_token_idx_q <=
                                        {ATTN_TOKEN_INDEX_WIDTH{1'b0}};
                                end else begin
                                    attn_token_idx_q <= attn_token_idx_q +
                                        ATTN_TOKEN_INDEX_WIDTH'(1);
                                end
                            end else begin
                                beat_idx_q <= beat_idx_q +
                                    BEAT_INDEX_WIDTH'(1);
                            end
                        end
                    end
                    ST_COMPOSE_OUT_DATA: begin
                        if (mem_wready_i) begin
                            if (!compose_out_last_shell_w) begin
                                beat_idx_q <= beat_idx_q +
                                    BEAT_INDEX_WIDTH'(1);
                            end
                        end
                    end
                    ST_AV_PROB_RECV: begin
                    end
                    ST_AV_START: begin
                        attn_token_idx_q <= {ATTN_TOKEN_INDEX_WIDTH{1'b0}};
                        attn_value_nibble_q <= 2'd0;
                        mem_req_addr_q <= src1_addr_q + (attn_group_storage_idx_ext_w << 4);
                    end
                    ST_AV_V_RECV: begin
                    end
                    ST_AV_PRODUCT: begin
                    end
                    ST_AV_ACCUM: begin
                        if (attn_value_nibble_q == 2'd3) begin
                            attn_value_nibble_q <= 2'd0;
                            if (attn_token_idx_q != attn_last_token_w) begin
                                attn_token_idx_q <= attn_token_idx_q +
                                    {{(ATTN_TOKEN_INDEX_WIDTH-1){1'b0}}, 1'b1};
                                mem_req_addr_q <= add_small64(mem_req_addr_q, 12'd64);
                            end
                        end else begin
                            attn_value_nibble_q <= attn_value_nibble_q + 2'd1;
                        end
                    end
                    ST_AV_ROUND: begin
                        attn_done_saturation_q <=
                            attn_done_saturation_q | attn_value_saturation_w;
                        done_saturation_q <= done_saturation_q | attn_value_saturation_w;
                        if (attn_value_last_pack_lane_w) begin
                            mem_req_addr_q <= attn_value_write_addr_w;
                        end else begin
                            attn_group_idx_q <= attn_group_idx_q +
                                {{(ATTN_GROUP_INDEX_WIDTH-1){1'b0}}, 1'b1};
                        end
                    end
                    ST_AV_WRITE_DATA: begin
                        if (mem_wready_i) begin
                            if (!attn_value_final_output_w) begin
                                attn_group_idx_q <= attn_group_idx_q +
                                    {{(ATTN_GROUP_INDEX_WIDTH-1){1'b0}}, 1'b1};
                            end
                        end
                    end
                    ST_RES_SRC0_RECV: begin
                        if (accepted_read_w) begin
                            mem_req_addr_q <= src1_addr_q + (beat_idx_ext_w << 4);
                        end
                    end
                    ST_RES_SRC1_RECV: begin
                        if (accepted_read_w) begin
                            mem_req_addr_q <= dst_addr_q + (beat_idx_ext_w << 4);
                        end
                    end
                    ST_RES_WRITE_DATA: begin
                        if (residual_write_accepted_w && (beat_idx_q != LAST_BEAT)) begin
                            beat_idx_q <= beat_idx_q + {{(BEAT_INDEX_WIDTH-1){1'b0}}, 1'b1};
                            mem_req_addr_q <= src0_addr_q + ((beat_idx_ext_w + 64'd1) << 4);
                        end
                    end
                    ST_SILU_START: begin
                            if (silu_start_valid_q && silu_start_ready_w) begin
                                silu_mem_req_addr_q <= src0_addr_q;
                            end
                        end
                        ST_SILU_GATE_RECV: begin
                            if (accepted_read_w) begin
                                silu_mem_req_addr_q <=
                                    add_silu_offset64(src1_addr_q, silu_input_offset_w);
                            end
                        end
                        ST_SILU_WAIT: begin
                            if (silu_out_valid_w) begin
                                done_saturation_q <= done_saturation_q | silu_saturation_w;
                                if (silu_output_half_q) begin
                                    silu_output_half_q <= 1'b0;
                                    silu_write_final_q <= silu_final_input_w;
                                    silu_mem_req_addr_q <=
                                        add_silu_offset64(dst_addr_q, silu_output_offset_w);
                                end else if (silu_final_input_w) begin
                                    silu_write_final_q <= 1'b1;
                                    silu_mem_req_addr_q <=
                                        add_silu_offset64(dst_addr_q, silu_output_offset_w);
                                end else begin
                                    silu_output_half_q <= 1'b1;
                                    silu_mem_req_addr_q <=
                                        add_silu_offset64(src0_addr_q, silu_next_input_offset_w);
                                end
                            end
                        end
                        ST_SILU_WRITE_DATA: begin
                            if (silu_write_accepted_w && !silu_write_final_q) begin
                                silu_mem_req_addr_q <=
                                    add_silu_offset64(src0_addr_q, silu_input_offset_w);
                            end
                        end
                    ST_WRITE_RESP: begin
                    end
                    ST_WRITE_RETIRE: begin
                        if (write_completes_descriptor_q) begin
                            done_valid_q <= 1'b1;
                            done_error_q <= done_error_q | done_saturation_q |
                                            ((op_kind_q ==
                                              OP_KIND_ATTN_COMPOSE) &&
                                             compose_saturation_w);
                            if (op_kind_q == OP_KIND_ATTN_COMPOSE) begin
                                done_saturation_q <=
                                    done_saturation_q |
                                    compose_saturation_w;
                            end
                            interrupt_status_q[0] <= 1'b1;
                            if (done_saturation_q ||
                                ((op_kind_q == OP_KIND_ATTN_COMPOSE) &&
                                 compose_saturation_w)) begin
                                error_status_q[ACE2_ERR_NUMERIC] <= 1'b1;
                                interrupt_status_q[ACE2_ERR_NUMERIC] <= 1'b1;
                            end
                        end
                    end
                    ST_WAIT_DONE: begin
                        if (core_done_valid_w && !done_valid_q) begin
                            done_valid_q <= 1'b1;
                            done_error_q <= done_error_q | core_saturation_w;
                            done_saturation_q <= core_saturation_w;
                            interrupt_status_q[0] <= 1'b1;
                            if (core_saturation_w) begin
                                error_status_q[ACE2_ERR_NUMERIC] <= 1'b1;
                                interrupt_status_q[ACE2_ERR_NUMERIC] <= 1'b1;
                            end
                        end
                    end
                    ST_COMPLETE: begin
                        if (!done_valid_q) begin
                            done_valid_q <= 1'b1;
                            done_error_q <= 1'b1;
                            if (attn_score_metadata_invalid_q) begin
                                error_status_q[ACE2_ERR_NUMERIC] <= 1'b1;
                                interrupt_status_q[ACE2_ERR_NUMERIC] <= 1'b1;
                            end else begin
                                error_status_q[ACE2_ERR_DESCRIPTOR] <= 1'b1;
                                interrupt_status_q[ACE2_ERR_DESCRIPTOR] <= 1'b1;
                            end
                        end else if (cmd_done_ready_i) begin
                            done_valid_q <= 1'b0;
                            done_error_q <= 1'b0;
                        end
                    end
                    default: begin
                    end
                endcase
                if (proj_write_accepted_w &&
                    !((proj_out_idx_q == proj_last_out_w) &&
                      (proj_row_idx_q == (m_q[PROJ_ROW_INDEX_WIDTH-1:0] -
                                          {{(PROJ_ROW_INDEX_WIDTH-1){1'b0}}, 1'b1})))) begin
                        if (proj_out_idx_q == proj_last_out_w) begin
                            proj_row_idx_q <= proj_row_idx_q + {{(PROJ_ROW_INDEX_WIDTH-1){1'b0}}, 1'b1};
                            proj_out_idx_q <= {PROJ_OUT_INDEX_WIDTH{1'b0}};
                        end else begin
                            proj_out_idx_q <= proj_out_idx_q + {{(PROJ_OUT_INDEX_WIDTH-1){1'b0}}, 1'b1};
                        end
                        proj_group_idx_q <= {PROJ_GROUP_INDEX_WIDTH{1'b0}};
                        proj_pack_lane_q <= 4'd0;
                end
                if (proj_wait_out_active_q && proj_out_valid_w) begin
                    proj_done_saturation_q <= proj_done_saturation_q | proj_saturation_w;
                    done_saturation_q <= done_saturation_q | proj_saturation_w;
                    if (proj_pack_lane_q == PROJ_LAST_PACK_LANE) begin
                        proj_pack_lane_q <= 4'd0;
                        mem_req_addr_q <= add_proj_offset64(dst_addr_q, proj_out_request_offset_w);
                    end else begin
                        proj_pack_lane_q <= proj_pack_lane_q + 4'd1;
                        proj_out_idx_q <= proj_out_idx_q + {{(PROJ_OUT_INDEX_WIDTH-1){1'b0}}, 1'b1};
                    end
                end
            end
        end
    end

    always @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            proj_weight_offset_q <= {PROJ_WEIGHT_OFFSET_WIDTH{1'b0}};
            proj_meta_offset_q <= {PROJ_LINEAR_OFFSET_WIDTH{1'b0}};
            proj_out_write_offset_q <= {PROJ_LINEAR_OFFSET_WIDTH{1'b0}};
        end else begin
            if ((state_q == ST_CMD_DISPATCH) && (op_kind_q == OP_KIND_PROJ)) begin
                proj_weight_offset_q <= {PROJ_WEIGHT_OFFSET_WIDTH{1'b0}};
                proj_meta_offset_q <= {PROJ_LINEAR_OFFSET_WIDTH{1'b0}};
                proj_out_write_offset_q <= {PROJ_LINEAR_OFFSET_WIDTH{1'b0}};
            end
            if (proj_write_accepted_w && !proj_final_output_w) begin
                if (proj_out_idx_q == proj_last_out_w) begin
                    proj_weight_offset_q <= {PROJ_WEIGHT_OFFSET_WIDTH{1'b0}};
                    proj_meta_offset_q <= {PROJ_LINEAR_OFFSET_WIDTH{1'b0}};
                end else begin
                    proj_weight_offset_q <= proj_weight_offset_q +
                                            PROJ_WEIGHT_OFFSET_WIDTH'(proj_weight_addr_stride_w);
                    proj_meta_offset_q <= proj_meta_offset_q +
                                          PROJ_LINEAR_OFFSET_WIDTH'(16);
                end
                proj_out_write_offset_q <= proj_out_write_offset_q +
                                           PROJ_LINEAR_OFFSET_WIDTH'(16);
            end
            if (proj_wait_out_active_q && proj_out_valid_w &&
                (proj_pack_lane_q != PROJ_LAST_PACK_LANE)) begin
                proj_weight_offset_q <= proj_weight_offset_q +
                                        PROJ_WEIGHT_OFFSET_WIDTH'(proj_weight_addr_stride_w);
                proj_meta_offset_q <= proj_meta_offset_q +
                                      PROJ_LINEAR_OFFSET_WIDTH'(16);
            end
        end
    end

    wire unused_sink_w = unused_inputs_w;
endmodule

module ace2_attention_accumulator (
    input  wire               clk_i,
    input  wire               rst_ni,
    input  wire               soft_reset_i,
    input  wire               response_fault_i,
    input  wire               watchdog_fire_i,
    input  wire               zero_i,
    input  wire               score_add_i,
    input  wire               value_add_i,
    input  wire signed [31:0] score_addend_i,
    input  wire signed [31:0] value_addend_i,
    output reg  signed [31:0] acc_o
);
    always @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            acc_o <= 32'sd0;
        end else if (soft_reset_i) begin
            acc_o <= 32'sd0;
        end else if (response_fault_i) begin
            acc_o <= acc_o;
        end else if (watchdog_fire_i || zero_i) begin
            acc_o <= 32'sd0;
        end else if (score_add_i) begin
            acc_o <= acc_o + score_addend_i;
        end else if (value_add_i) begin
            acc_o <= acc_o + value_addend_i;
        end
    end
endmodule

module ace2_state_shadow (
    input  wire       clk_i,
    input  wire       rst_ni,
    input  wire [6:0] state_i,
    output reg  [6:0] state_o
);
    always @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            state_o <= 7'd0;
        end else begin
            state_o <= state_i;
        end
    end
endmodule

`default_nettype wire
