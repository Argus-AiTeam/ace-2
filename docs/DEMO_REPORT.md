# ACE-2 Alpha 2 Demo Report

## Result

`ACE2_ALPHA2_DEMO_PASS`

| Metric | Value |
|---|---:|
| Certified RTL files | 23 |
| RMSNorm cases | 15 |
| Beats per case | 56 |
| Hidden size | 896 |
| Cases exercising saturation | 4 |
| Verilator lint log lines | 16 |

## Evidence chain

1. All 23 certified RTL hashes matched.
2. The complete shell passed Verilator lint with no fatal error.
3. The independent generator reproduced the packaged vectors byte-for-byte.
4. Icarus RTL simulation passed 15 cases x 56 beats.

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

## Timing-closure progression

| Tree | Setup slack | Decision |
|---|---:|---|
| Initial full tree | -0.1484 ns | NO-GO |
| Shell fanout repair | -0.5275 ns | NO-GO |
| RMSNorm enable repair | -0.1741 ns | NO-GO |
| Final-sum preload split | +0.6966 ns | PASS |

## Certification boundary

Alpha 2 certifies one frozen pre-tokenized input through 24 layers and two
generated tokens, plus mapped SKY130 synthesis/OpenSTA at 100 MHz. This fast
demo verifies release identity and a focused arithmetic discriminator; it does
not replay the sealed full-model run or claim arbitrary chat, FPGA, routed
signoff, tapeout, or silicon.
