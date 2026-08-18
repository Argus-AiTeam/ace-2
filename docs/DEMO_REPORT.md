# ACE-2 Transformer RTL Demo Report

## Result

`ACE2_LOCAL_RTL_DEMO_PASS`

| Metric | Value |
|---|---:|
| Certified RTL files | 23 |
| RMSNorm cases | 16 |
| Beats per case | 56 |
| Hidden size | 896 |
| Cases exercising saturation | 4 |
| Verilator lint log lines | 16 |
| Local challenge ID | `82554a2396e56aa8f142c7c8abbbc92b` |
| Generated at | `2026-08-18T10:48:59.885128+00:00` |
| Source commit | `d42a233f66f8d78fe7e89d1f0e8d946303498371` |
| Local platform | `Linux 6.8.0-1064-azure x86_64` |
| Fresh challenge output SHA-256 | `01331f1fd67a32ad7cb645fa250726fd1f48637276a780117804169099c1da8b` |
| Transformer operator groups | 5 |
| Selected shell integration modes | 6 |
| Operator-suite runtime | 58.58 seconds |
| Random operator seed | `demo-seed-20260818` |
| Random Python-oracle groups | 5 |
| Layer-0 operator rows run by fast demo | 9/18 |
| Complete shell regression | PASS in this workspace |
| Default shell proof | PASS |
| Dedicated MLP-up proof | PASS |

## Evidence chain

1. All 23 certified RTL hashes matched.
2. The complete shell passed Verilator lint with no fatal error.
3. The independent generator reproduced the packaged vectors byte-for-byte.
4. This machine generated challenge `82554a2396e56aa8f142c7c8abbbc92b` after the demo started.
5. Icarus recompiled the RTL and passed 16 cases x 56 beats,
   including the fresh challenge case.
6. A deliberately corrupted expected beat was rejected by the same checker.
7. The run produced `build/demo_challenge/rmsnorm-waveform.vcd`.
8. Five independent Transformer core groups passed both packaged edge cases and
   fresh random cases generated from seed `demo-seed-20260818` by the checked-in
   bit-accurate Python references.
9. Six selected `ace2_shell` integration modes passed their packaged
   bit-accurate oracle vectors.
10. Extended coverage is marked complete only when both the default shell log
    contains `ACE2_SHELL_TB_PASS` and the dedicated MLP-up log contains
    `ACE2_SHELL_MLP_UP_TB_PASS`.

![Fresh local RMSNorm challenge waveform](demo_challenge/rmsnorm-waveform.svg)

## Local toolchain

| Tool | Version observed by this run |
|---|---|
| Python | `Python 3.13.5` |
| Verilator | `Verilator 4.038 2020-07-11 rev v4.036-114-g0cd4a57ad` |
| Icarus Verilog | `Icarus Verilog version 11.0 (stable) ()` |
| VVP | `Icarus Verilog runtime version 11.0 (stable) ()` |

## RMSNorm workload identities

| Case | Sum of squares | Output scale | Output SHA-256 |
|---|---:|---:|---|
| `all_zero` | 0 | 0.03125000 | `d54f02b97ed5bc2a...` |
| `all_one` | 896 | 0.02343750 | `2a3abe9478b9e7a6...` |
| `alternating_extremes` | 14,565,824 | 0.01562500 | `04815572ac575082...` |
| `ramp` | 4,889,152 | 0.03906250 | `fc80012a92fec2c7...` |
| `half_lsb_ties` | 896 | 0.06250000 | `9251f48d21bde7f6...` |
| `sparse_signed_saturation` | 32,513 | 0.00781250 | `d5d93924318a6a18...` |
| `seeded_random_0` | 4,867,787 | 0.01562500 | `84937fb81bc696e5...` |
| `seeded_random_1` | 4,976,562 | 0.02343750 | `3146db69f4c1c693...` |
| `seeded_random_2` | 4,767,924 | 0.03125000 | `5b2c4008e082a6c5...` |
| `seeded_random_3` | 4,866,435 | 0.03906250 | `bd597d4f07cacb72...` |
| `seeded_random_4` | 4,945,431 | 0.01562500 | `3ec86d6e984ea064...` |
| `seeded_random_5` | 4,828,132 | 0.02343750 | `9adc126258206a08...` |
| `seeded_random_6` | 5,070,925 | 0.03125000 | `3f2d0fe999d54fdd...` |
| `seeded_random_7` | 4,679,885 | 0.03906250 | `01f0d6b16f226542...` |
| `gain_scale_floor_dominates` | 71,675 | 0.00976592 | `11d9b6e774af27fc...` |
| `local_challenge_55cbd4e79ea0` | 4,925,154 | 0.03125000 | `01331f1fd67a32ad...` |

## Transformer operator coverage

### Random Python-oracle challenge

| Operator | Cases compiled into RTL testbench | Oracle JSON SHA-256 |
|---|---:|---|
| attention_compose | 4 | `f24799b7d0e2f28267b89fd9bdef46989249762f6cdacc6fd8c1cdf1c1b61997` |
| attention_score | 5 | `c4efb37f7fefee7bb89266721e17a6e5a7c3219b6ea369a2175b4942e1283679` |
| rope | 6 | `18fb802f6d11b8bce47498af315773608634e9a5d1f4f993b259a0ed3b56cb63` |
| silu_gate | 9 | `6ca56b5ca030b90bd69cc8de345639a3a0a45f8c96fa66f6d6703c17bc70008d` |
| softmax | 6 | `1aaab3dd616c77d608ec3f1de197c65c41d895636c4251595332844d6c6171ed` |

| # | Layer-0 operator | Single demo | This workspace | Evidence path |
|---:|---|---|---|---|
| 01 | Input RMSNorm | `make demo-input-rmsnorm` | PASS now | Fresh RMSNorm challenge |
| 02 | Q projection | `make demo-q-proj` | PASS via extended | Complete shell regression |
| 03 | K projection | `make demo-k-proj` | PASS via extended | Complete shell regression |
| 04 | V projection | `make demo-v-proj` | PASS via extended | Complete shell regression |
| 05 | RoPE Q | `make demo-rope-q` | PASS now | RoPE core |
| 06 | RoPE K | `make demo-rope-k` | PASS now | RoPE core |
| 07 | KV write | `make demo-kv-write` | PASS via extended | Complete shell regression |
| 08 | Attention score | `make demo-attention-score` | PASS now | Attention score core + shell |
| 09 | Softmax | `make demo-softmax` | PASS now | Softmax core |
| 10 | Attention value | `make demo-attention-value` | PASS via extended | Complete shell regression |
| 11 | O projection | `make demo-o-proj` | PASS via extended | Complete shell regression |
| 12 | Attention residual add | `make demo-attention-residual` | PASS now | Residual shell |
| 13 | Post-attention RMSNorm | `make demo-post-attention-rmsnorm` | PASS now | Residual/post-norm shell |
| 14 | MLP gate projection | `make demo-mlp-gate` | PASS via extended | Complete shell regression |
| 15 | MLP up projection | `make demo-mlp-up` | PASS via extended | Complete shell regression |
| 16 | SiLU gate | `make demo-silu` | PASS now | SiLU core |
| 17 | MLP down projection | `make demo-mlp-down` | PASS via extended | Complete shell regression |
| 18 | MLP residual add | `make demo-mlp-residual` | PASS now | MLP residual shell |

### Executed test processes

| Test | Execution path | Runtime | Result marker |
|---|---|---:|---|
| RoPE | independent core | 3.00 s | `ACE2_ROPE_TB_PASS cases=5 beats_per_case=56` |
| Attention score | independent core | 0.01 s | `ACE2_ATTN_SCORE_TB_PASS cases=4 context_max=8` |
| Softmax | independent core | 0.01 s | `ACE2_SOFTMAX_TB_PASS cases=5 context_max=8` |
| Attention compose | independent core | 0.03 s | `ACE2_ATTN_COMPOSE_TB_PASS cases=3 contexts=9,16,17 context_max=32768 tile=8` |
| SiLU gate | independent core | 1.87 s | `ACE2_SILU_GATE_TB_PASS cases=8 boundary_mask=3f` |
| Attention score shell | shell integration | 2.33 s | `ACE2_SHELL_ATTN_SCORE_TB_PASS cases=4 unequal_scale_case=1 invalid_metadata_cases=2` |
| Attention compose shell | shell integration | 1.49 s | `ACE2_SHELL_ATTN_COMPOSE_ONLY_TB_PASS cases=1 phases=6 tiles=2 contexts=9` |
| Residual and post-norm shell | shell integration | 27.15 s | `ACE2_SHELL_SMOKE_TB_PASS opcode=00000008` |
| MLP residual shell | shell integration | 1.72 s | `ACE2_SHELL_MLP_RESIDUAL_TB_PASS cases=4 writes=56 cycles=897 saturation_cases=3 descriptor_errors=2 memory_errors=4 protocol_cases=5` |
| Final RMSNorm shell | shell integration | 8.47 s | `ACE2_SHELL_FINAL_RMSNORM_TB_PASS layer=24 cases=2 rejected_layer=25 rejected_m=2 rejected_k=1 descriptor_errors=3 cycles=126877` |
| LM head shell | shell integration | 12.52 s | `ACE2_SHELL_LM_HEAD_TB_PASS layer=24 tile_outputs=32 vocab=151936 tiles=4748 cases=2 read_beats=14368 write_beats=2 weight_span_bytes=14336 weight_high_water_offset=14320 rejected_layer=23 rejected_m=2 rejected_n=64 rejected_k=768 descriptor_errors=4 cycles=181274` |

## Timing-closure progression

| Tree | Setup slack | Decision |
|---|---:|---|
| Initial full tree | -0.1484 ns | NO-GO |
| Shell fanout repair | -0.5275 ns | NO-GO |
| RMSNorm enable repair | -0.1741 ns | NO-GO |
| Final-sum preload split | +0.6966 ns | PASS |

## Certification boundary

This run directly proves, on the user's machine, certified RTL identity,
complete-shell lint, fresh RMSNorm oracle/RTL agreement, failure detection, and
waveform generation. Alpha 2's 24-layer/two-token and mapped SKY130 results are
historical certification claims and are not rerun by this demo. The demo does
not claim arbitrary chat, FPGA execution, routed signoff, tapeout, or silicon.
