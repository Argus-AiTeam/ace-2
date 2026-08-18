`default_nettype none

package ace2_pkg;
    localparam integer ACE2_ADDR_WIDTH = 64;
    localparam integer ACE2_CSR_ADDR_WIDTH = 32;
    localparam integer ACE2_CSR_DATA_WIDTH = 64;
    localparam integer ACE2_CSR_STRB_WIDTH = 8;
    localparam integer ACE2_MEM_DATA_WIDTH = 128;
    localparam integer ACE2_MEM_STRB_WIDTH = 16;
    localparam integer ACE2_MEM_TAG_WIDTH = 8;
    localparam integer ACE2_SRAM_BANKS = 8;
    localparam integer ACE2_SRAM_ADDR_WIDTH = 12;
    localparam integer ACE2_HIDDEN_SIZE = 896;
    localparam integer ACE2_VOCAB_SIZE = 151936;
    localparam integer ACE2_LM_HEAD_TILE_SIZE = 32;
    localparam integer ACE2_LM_HEAD_TILE_COUNT = ACE2_VOCAB_SIZE / ACE2_LM_HEAD_TILE_SIZE;
    localparam integer ACE2_VECTOR_LANES = 16;

    localparam [7:0] ACE2_OPCODE_W4A8_PROJ = 8'h01;
    localparam [7:0] ACE2_OPCODE_RMSNORM   = 8'h02;
    localparam [7:0] ACE2_OPCODE_ROPE      = 8'h03;
    localparam [7:0] ACE2_OPCODE_ATTN_SCORE = 8'h04;
    localparam [7:0] ACE2_OPCODE_SOFTMAX   = 8'h05;
    localparam [7:0] ACE2_OPCODE_ATTN_VALUE = 8'h06;
    localparam [7:0] ACE2_OPCODE_SILU_GATE = 8'h07;
    localparam [7:0] ACE2_OPCODE_RESIDUAL_ADD = 8'h08;
    localparam [7:0] ACE2_OPCODE_ATTN_COMPOSE = 8'h09;
    localparam [7:0] ACE2_OPCODE_KV_WRITE  = 8'h0a;
    localparam [7:0] ACE2_OPCODE_FUSED_QKV = 8'h0b;

    localparam [7:0] ACE2_ATTN_COMPOSE_MAX_FIRST = 8'h00;
    localparam [7:0] ACE2_ATTN_COMPOSE_MAX_MORE = 8'h01;
    localparam [7:0] ACE2_ATTN_COMPOSE_SUM_FIRST = 8'h02;
    localparam [7:0] ACE2_ATTN_COMPOSE_SUM_MORE = 8'h03;
    localparam [7:0] ACE2_ATTN_COMPOSE_VALUE_FIRST = 8'h04;
    localparam [7:0] ACE2_ATTN_COMPOSE_VALUE_MORE = 8'h05;
    localparam [7:0] ACE2_ATTN_COMPOSE_VALUE_LAST = 8'h06;

    localparam [31:0] ACE2_CSR_ID              = 32'h000;
    localparam [31:0] ACE2_CSR_VERSION         = 32'h008;
    localparam [31:0] ACE2_CSR_CAPABILITIES    = 32'h010;
    localparam [31:0] ACE2_CSR_CONTROL         = 32'h018;
    localparam [31:0] ACE2_CSR_STATUS          = 32'h020;
    localparam [31:0] ACE2_CSR_ERROR_STATUS    = 32'h028;
    localparam [31:0] ACE2_CSR_INTERRUPT_EN    = 32'h030;
    localparam [31:0] ACE2_CSR_INTERRUPT_ST    = 32'h038;
    localparam [31:0] ACE2_CSR_DESC_BASE       = 32'h040;
    localparam [31:0] ACE2_CSR_DESC_HEAD       = 32'h048;
    localparam [31:0] ACE2_CSR_DESC_TAIL       = 32'h050;
    localparam [31:0] ACE2_CSR_DOORBELL        = 32'h058;
    localparam [31:0] ACE2_CSR_WATCHDOG_LIMIT  = 32'h060;
    localparam [31:0] ACE2_CSR_PERF_CYCLE      = 32'h068;
    localparam [31:0] ACE2_CSR_PERF_BYTE       = 32'h070;
    localparam [31:0] ACE2_CSR_PERF_TOKEN      = 32'h078;
    localparam [31:0] ACE2_CSR_PERF_STALL      = 32'h080;

    localparam [2:0] ACE2_ERR_DESCRIPTOR = 3'd0;
    localparam [2:0] ACE2_ERR_MEMORY     = 3'd1;
    localparam [2:0] ACE2_ERR_NUMERIC    = 3'd2;
    localparam [2:0] ACE2_ERR_WATCHDOG   = 3'd3;
    localparam [2:0] ACE2_ERR_RESET_BUSY = 3'd4;
    localparam [2:0] ACE2_ERR_RESERVED   = 3'd5;

    typedef struct packed {
        logic [7:0]  opcode;
        logic [7:0]  flags;
        logic [7:0]  layer_id;
        logic [15:0] n;
        logic [15:0] completion_tag;
        logic [63:0] src0_addr;
        logic [63:0] dst_addr;
        logic [63:0] scale_addr;
    } ace2_cmd_rmsnorm_t;
endpackage

`default_nettype wire
