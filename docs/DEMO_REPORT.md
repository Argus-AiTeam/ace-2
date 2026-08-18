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
| Local challenge ID | `59c35ebf73601302158309473ffc00dd` |
| Generated at | `2026-08-18T10:11:16.298813+00:00` |
| Source commit | `8fb312cf9d5134dc10f92e11fa6145dfec07338b` |
| Local platform | `Linux 6.8.0-1064-azure x86_64` |
| Fresh challenge output SHA-256 | `212454244b24ca361394b7939c10181009b74247901fa53adb7fb678f9de7097` |
| Transformer operator groups | 5 |
| Selected shell integration modes | 6 |
| Operator-suite runtime | 58.92 seconds |
| Layer-0 operator rows run by fast demo | 9/18 |
| Complete shell regression | Run make demo-extended |

## Evidence chain

1. All 23 certified RTL hashes matched.
2. The complete shell passed Verilator lint with no fatal error.
3. The independent generator reproduced the packaged vectors byte-for-byte.
4. This machine generated challenge `59c35ebf73601302158309473ffc00dd` after the demo started.
5. Icarus recompiled the RTL and passed 16 cases x 56 beats,
   including the fresh challenge case.
6. A deliberately corrupted expected beat was rejected by the same checker.
7. The run produced `build/demo_challenge/rmsnorm-waveform.vcd`.
8. Six independent Transformer core groups and five selected `ace2_shell`
   integration modes passed against packaged oracle vectors.

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
| `local_challenge_84d7277a700e` | 5,080,736 | 0.03125000 | `212454244b24ca36...` |

## Transformer operator coverage

| # | Layer-0 operator | Fast demo | Evidence path |
|---:|---|---|---|
| 01 | Input RMSNorm | PASS now | Fresh RMSNorm challenge |
| 02 | Q projection | demo-extended | Complete shell regression |
| 03 | K projection | demo-extended | Complete shell regression |
| 04 | V projection | demo-extended | Complete shell regression |
| 05 | RoPE Q | PASS now | RoPE core |
| 06 | RoPE K | PASS now | RoPE core |
| 07 | KV write | demo-extended | Complete shell regression |
| 08 | Attention score | PASS now | Attention score core + shell |
| 09 | Softmax | PASS now | Softmax core |
| 10 | Attention value | demo-extended | Complete shell regression |
| 11 | O projection | demo-extended | Complete shell regression |
| 12 | Attention residual add | PASS now | Residual shell |
| 13 | Post-attention RMSNorm | PASS now | Residual/post-norm shell |
| 14 | MLP gate projection | demo-extended | Complete shell regression |
| 15 | MLP up projection | demo-extended | Complete shell regression |
| 16 | SiLU gate | PASS now | SiLU core |
| 17 | MLP down projection | demo-extended | Complete shell regression |
| 18 | MLP residual add | PASS now | MLP residual shell |

### Executed test processes

| Test | Execution path | Runtime | Result marker |
|---|---|---:|---|
| RoPE | independent core | 3.02 s | `ACE2_ROPE_TB_PASS cases=5 beats_per_case=56` |
| Attention score | independent core | 0.01 s | `ACE2_ATTN_SCORE_TB_PASS cases=4 context_max=8` |
| Softmax | independent core | 0.01 s | `ACE2_SOFTMAX_TB_PASS cases=5 context_max=8` |
| Attention compose | independent core | 0.03 s | `ACE2_ATTN_COMPOSE_TB_PASS cases=3 contexts=9,16,17 context_max=32768 tile=8` |
| SiLU gate | independent core | 1.92 s | `ACE2_SILU_GATE_TB_PASS cases=8 boundary_mask=3f` |
| Attention score shell | shell integration | 2.55 s | `ACE2_SHELL_ATTN_SCORE_TB_PASS cases=4 unequal_scale_case=1 invalid_metadata_cases=2` |
| Attention compose shell | shell integration | 1.48 s | `ACE2_SHELL_ATTN_COMPOSE_ONLY_TB_PASS cases=1 phases=6 tiles=2 contexts=9` |
| Residual and post-norm shell | shell integration | 27.22 s | `ACE2_SHELL_SMOKE_TB_PASS opcode=00000008` |
| MLP residual shell | shell integration | 1.72 s | `ACE2_SHELL_MLP_RESIDUAL_TB_PASS cases=4 writes=56 cycles=897 saturation_cases=3 descriptor_errors=2 memory_errors=4 protocol_cases=5` |
| Final RMSNorm shell | shell integration | 8.54 s | `ACE2_SHELL_FINAL_RMSNORM_TB_PASS layer=24 cases=2 rejected_layer=25 rejected_m=2 rejected_k=1 descriptor_errors=3 cycles=126877` |
| LM head shell | shell integration | 12.42 s | `ACE2_SHELL_LM_HEAD_TB_PASS layer=24 tile_outputs=32 vocab=151936 tiles=4748 cases=2 read_beats=14368 write_beats=2 weight_span_bytes=14336 weight_high_water_offset=14320 rejected_layer=23 rejected_m=2 rejected_n=64 rejected_k=768 descriptor_errors=4 cycles=181274` |

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
