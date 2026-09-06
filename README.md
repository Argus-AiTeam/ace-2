<div align="center">

# Argus Compute Engine 2 (ACE-2)

### Evidence-first Qwen2.5-0.5B W4A8 accelerator engineering

[English](README.md) | [简体中文](README.zh-CN.md)

[![Release](https://img.shields.io/github/v/release/aHappend/ace-2?include_prereleases&label=release)](https://github.com/aHappend/ace-2/releases)
[![License](https://img.shields.io/github/license/aHappend/ace-2)](LICENSE)
[![RTL](https://img.shields.io/badge/RTL-SystemVerilog-5C4EE5)](rtl/)
[![Target](https://img.shields.io/badge/SKY130-100%20MHz-18A999)](docs/PPA_SUMMARY.md)
[![Built by](https://img.shields.io/badge/built_by-Argus_AI_Team-7C3AED)](https://github.com/Argus-AiTeam)
[![Claim boundary](https://img.shields.io/badge/claims-evidence_bound-0F766E)](KNOWN_LIMITATIONS.md)

**ACE means Argus Compute Engine. ACE-2 was designed, implemented, tested,
reviewed, and iterated primarily by
[Argus](https://argusbot.cn/) under human-owned objectives and release
authority.**

</div>

![ACE-2 certified Alpha 2 baseline](docs/ace2-alpha2-overview.svg)

## September 6, 2026: selected engineering highlights

**[Read the selected milestone highlights](docs/results/ACE2_PROGRESS_20260906.md)**
and [machine-readable highlights](evidence/public/ace2-progress-20260906/highlights.json).
The existing public source baseline is unchanged; this is **not a runnable
release of the newer local implementation or a comprehensive quality report**.

V74 completed **840 RTL layer invocations across 35 causal positions**, with
persistent K/V, host feedback and two token selections. This is bounded local
simulation coverage, not dialogue-quality certification or independent
whole-model reference agreement. Regression 0006 added an evidence audit and
fresh shell smoke, not another 840-layer replay.

An offline S16 residual profile reduced **ordinary-channel merge MAE by
99.30% on one held fixture**. This metric concerns the merge only, not
normalized-output fidelity or chat quality; runtime/RTL integration remains
unimplemented. Earlier broader summaries remain in public Git history;
the current presentation is selective, not an assertion that all evaluations
passed.

**Current scope is local simulation/software diagnosis. Hardware Stage 2,
FPGA, synthesis, PPA and U280 work is cancelled.** The Alpha 2/3 results and
productization plan below are preserved historical records, not current
authorization or results for these newer paths. See
[publication boundaries](docs/results/PUBLICATION_BOUNDARY.md).

> **Alpha 3 scope:** a public productization-progress snapshot built on the
> unchanged Alpha 2 certified RTL baseline. It documents the post-Alpha-2 BF16
> model-quality program and the exact gates that still block arbitrary-text
> W4A8 chat and U280 deployment. Alpha 3 does not claim a new certified model,
> general chat, FPGA execution, routed signoff, or silicon.

## Alpha 3 at a glance

| Area | Alpha 3 status |
|---|---|
| Certified RTL baseline | **Preserved unchanged from Alpha 2** |
| Layer-0 fixed-point operators | **18 / 18 exact PASS** |
| Full runtime commands | **13,914 / 13,914 PASS** |
| Demonstrated model path | **24 layers, two generated tokens** |
| SKY130 mapped result | **62,283 cells, 0.614082704 mm2** |
| Timing | **100 MHz PASS, +0.6966 ns setup slack** |
| BF16 successor | **S6 sealed at probe-gate NO-GO; official dev was not accessed** |
| Execution admission | **V8 recovery package Fresh-L2 accepted; external root still required** |
| Recorded generation diagnostic | **Fixed-input two-token execution completed; capability evidence only** |
| Arbitrary-text W4A8 chat | **Not yet accepted** |
| Alveo U280 deployment | **Not started; external tool/board access required** |

The machine-readable identities, model revision, image hash, schedule hash,
and exact Alpha 2 certification boundary remain summarized in
[CERTIFICATION.md](CERTIFICATION.md). See
[Alpha 3 productization progress](docs/ALPHA3_PROGRESS.md) for the new work and
its explicit non-claims.

The latest public-safe productization result does not advance the certified
RTL baseline. The V8 host-trust recovery package passed 58 verifier checks with
zero reported issues and received Fresh-L2 acceptance for content SHA-256
`07663099352edfad32eb39919ad9475f1f887328ebb549bdb9cae1c48f5ccad1`.
Its status is `BUILD_READY_EXTERNAL_ROOT_REQUIRED`: it has not been installed,
no privileged execution occurred, and Stage 1 is not complete. See
[Host-trust recovery status](docs/HOST_TRUST_RECOVERY.md).

## Why ACE-2 is an Argus result

ACE-2 is part of the wider body of work published by the
[Argus AI Team](https://github.com/Argus-AiTeam). Argus carried out the
iterative engineering loop: architecture decomposition, RTL and oracle
implementation, deterministic test generation, long-running verification,
failure localization, evidence binding, reviewer handoffs, and fail-closed
rollback decisions. Human control remained at the mission, budget,
authorization, credential, and publication boundaries.

This attribution is not a substitute for evidence. The repository keeps
accepted results, rejected candidates, reproducible demos, and explicit
non-claims separate. See [Argus design provenance](ARGUS_PROVENANCE.md).

An independently reviewed, fixed-input generation record is available in
[the public two-token diagnostic evidence bundle](evidence/public/fixed-hi-two-token-diagnostic-v1/).
It completed 175,855 Verilated commands and two token selections. This
demonstrates recorded token generation, not useful language quality or
arbitrary-text chat.

## What ACE-2 contains

```mermaid
flowchart LR
    H[Host command stream] --> D[Descriptor + DMA shell]
    D --> N[RMSNorm]
    N --> Q[W4A8 Q / K / V / O projections]
    Q --> R[RoPE + attention score]
    R --> S[Softmax + value composition]
    S --> M[MLP gate / up / SiLU / down]
    M --> A[Residual + KV state]
    A --> L[Final RMSNorm + LM head]
    L --> T[Token IDs]
```

The release includes the certified RTL, deterministic fixed-point references,
generated test vectors, Verilator/Icarus harnesses, image/runtime utilities,
and release-local SKY130 flow scripts. Model weights, proprietary PDK data,
private benchmarks, build products, and sealed internal run packets are not
distributed.

### Current RTL organization

ACE-2 implements one reusable Transformer-layer engine rather than physically
replicating all 24 model layers. The host selects a layer, supplies its weights
and descriptors, invokes the operators in order, and feeds the resulting hidden
state into the next layer.

```mermaid
flowchart TB
    HOST[Host runtime and model package] --> IFACE[128-bit command/data interface]
    IFACE --> SHELL

    subgraph SHELL[ace2_shell]
        CTRL[Command decoder<br/>descriptor, completion, error control]
        MEM[Banked SRAM, DMA and KV state]

        subgraph PROJ[Shared W4A8 projection path]
            MAC[Four MAC lanes]
            PUSE[Q / K / V / O<br/>Gate / Up / Down]
        end

        subgraph VEC[Vector and special-function cores]
            NORM[RMSNorm]
            ROPE[RoPE]
            SM[Softmax]
            SILU[SiLU / SwiGLU]
            RES[Residual and requantization]
        end

        subgraph ATTN[Attention and state]
            KV[KV cache read/write]
            SCORE[Attention score]
            VALUE[Attention value/compose]
        end

        CTRL --> NORM --> PROJ --> ROPE --> KV --> SCORE --> SM --> VALUE
        VALUE --> PROJ --> RES --> NORM --> PROJ --> SILU --> PROJ --> RES
        MEM <--> PROJ
        MEM <--> ATTN
        MEM <--> VEC
    end

    SHELL --> NEXT[Layer output / next-layer input]
```

The current design is command-driven and resource-shared:

- the same layer engine is reused across all model layers and token positions;
- seven major projections share one W4A8 MAC path;
- RMSNorm, RoPE, Softmax and SiLU are separate reusable cores;
- KV state persists across token steps;
- operators execute in sequence rather than as a fully autonomous layer
  pipeline;
- fused opcode `0x0b` executes Q, K and V as one ordered descriptor while
  reusing the activation tile; it does not add three independent projection
  engines;
- larger Qwen and other decoder-only model shapes still require the planned
  parameterized model/hardware contract.

This organization keeps area controlled and makes the individual cores
reusable, while leaving clear optimization opportunities in MAC parallelism,
operator fusion and Prefill/Decode scheduling.

### Fused QKV dataflow

The shell can cache one 56-beat, 896-byte activation tile and reuse it across
the ordered Q, K and V projection phases. The legacy three-descriptor path
remains available. On the frozen Qwen2.5-0.5B-shaped RTL benchmark:

| Metric | Legacy Q/K/V | Fused QKV |
|---|---:|---:|
| Commands | 3 | 1 |
| Activation reads | 64,512 | 56 |
| Total reads | 97,920 | 33,464 |
| Simulator cycles | 1,044,326 | 805,011 |

All 1,152 output bytes match the same fixed-point oracle in both modes.
Backpressure, reset/restart, corrupted read tags and legacy compatibility are
also checked. This is a bounded projection result, not a full-model chat or
whole-shell timing claim.

```sh
make fused-qkv-freeze
make fused-qkv-check
```

### Immutable PPA preflight

`flow/immutable_ppa/` provides a non-consuming preflight for future
base/candidate SKY130 comparisons. It freezes the exact 12 shell parameters,
64 public ports, fused-QKV contract, RTL/SDC/flow hashes, container digest and
absolute Yosys/OpenSTA/library paths. The preflight validates and renders
commands but deliberately cannot run synthesis or STA.

```sh
python3 flow/immutable_ppa/benchmark_interface.py --repo "$PWD"
python3 -m unittest \
  flow/immutable_ppa/test_benchmark_interface.py \
  flow/immutable_ppa/test_immutable_ppa.py
```

Official comparison namespaces use exclusive creation and reject overwrite or
retry. This package itself contains no PPA, timing-closure or FPGA claim.

### Parameterized Qwen2.5 model contracts

ACE-2 now includes executable model/hardware descriptors for Qwen2.5 0.5B,
1.5B, 3B and 7B. The shared schema validates model dimensions, GQA geometry,
precision choices, memory-layout requirements and estimated weight/KV capacity.

```sh
make model-hardware-contract-check
```

| Model | Contract scope | Estimated packed weights | Maximum weight + KV estimate |
|---|---|---:|---:|
| Qwen2.5-0.5B | Existing package/runtime preflight | 526.7 MB | 740.6 MB |
| Qwen2.5-1.5B | Structural only | 1.25 GB | 3.19 GB |
| Qwen2.5-3B | Structural only | 2.18 GB | 2.81 GB |
| Qwen2.5-7B | Structural only | 4.65 GB | 8.53 GB |

The larger-model descriptors establish machine-checked structural contracts;
they do **not** claim that 1.5B, 3B or 7B has executed in RTL. The maximum
capacity figures use each model's declared maximum context and therefore are
planning bounds rather than measured board allocation.

### Mixed-precision planning

The same validation command also generates deterministic precision plans for
all four model contracts:

| Policy | Intended use | Current status |
|---|---|---|
| `w4a8` | W4 projections with A8 activation/KV paths | Current RTL format |
| `w8a8` | Higher-precision projection candidate | Structural candidate; no RTL execution claim |
| `mixed_w4a8_a16_bf16` | W4 projections with A16/BF16-sensitive operator classes | Structural candidate; no RTL execution claim |

Each plan records per-operator precision, estimated weight/KV capacity,
maximum-context decode traffic, descriptor hashes and explicit hardware-support
status. The validator fails closed on schema/type mismatches and includes a
signed-int4 ties-to-even packing reference. W8A8 and mixed BF16 support remain
software/hardware co-design plans until corresponding RTL is implemented and
verified.

## Open IP Library

The [ACE-2 Open IP Library](IP_LIBRARY.md) organizes the canonical RTL into
nine reusable packages with machine-readable manifests. It distinguishes
standalone cores (`rmsnorm`, `rope`, `softmax`, projection, and SiLU/SwiGLU),
standalone attention cores with shared shell integration, the shell-owned KV
write path, and MLP/Transformer-layer integration bundles.

```sh
make ip-list
make ip-validate
make ip-demo IP=rmsnorm
make ip-softmax
```

Package results are emitted under `build/ip_library/`. The existing 18
operator demos prove the listed ACE-2 paths, but not every operator name is a
separate standalone core. See each manifest for canonical sources,
Qwen2.5-0.5B parameters, interfaces, dependencies, proof mapping, and
limitations. This packaging does not claim arbitrary Transformer support,
full-model chat completion, or FPGA deployment.

## Run the visual demo

Install Python 3, GNU Make, Verilator, and Icarus Verilog, then run:

```sh
make demo
```

The demo does not replay the billion-cycle full-model certification. It runs a
fast, public-safe, machine-local evidence chain:

1. verifies every certified RTL file hash;
2. checks the open-source toolchain;
3. lints the complete accelerator shell;
4. regenerates deterministic RMSNorm vectors with the independent oracle;
5. simulates 15 RTL cases x 56 beats against expected results;
6. generates a fresh unpredictable local challenge and recompiles the RTL;
7. emits a VCD waveform for the challenge run;
8. proves the checker rejects an intentionally corrupted expected result;
9. generates fresh seeded random questions for five Transformer core groups,
   computes bit-accurate Python answers, and compares them with RTL output;
10. runs six selected `ace2_shell` integration modes;
11. displays all 18 certified Layer-0 operators, distinguishing fast-demo
    execution from slow extended-shell coverage;
12. produces a standalone visual evidence dashboard with the local challenge,
   tool versions, source commit, logs, and output hashes.

Expected final marker:

```text
ACE2_LOCAL_RTL_DEMO_PASS
```

Open the generated dashboard:

```text
build/DEMO_REPORT.html
```

**[View a sample Alpha 2 evidence report](docs/DEMO_REPORT.md)** without
installing the simulation toolchain.

See [DEMO.md](DEMO.md) for the complete walkthrough and raw artifact map.

To run the complete public shell regression, including the slower projection,
KV-write, and attention-value paths:

```sh
make demo-extended
```

Replay a reported random challenge with `make demo SEED=<seed>`.

Inspect one Layer-0 operator at a time:

```sh
make demo-operators              # list all 18 names
make demo-softmax
make demo-mlp-up                 # slow: full 896 x 4864 projection
make demo-operator OP=kv-write   # equivalent generic form
```

Each command writes a focused log and `result.json` under
`build/single_operator/<operator>/`. RoPE Q/K and residual/post-norm have
separate commands but transparently share their paired shell proof path.

| Operator | Command | Operator | Command |
|---|---|---|---|
| Input RMSNorm | `make demo-input-rmsnorm` | Q projection | `make demo-q-proj` |
| K projection | `make demo-k-proj` | V projection | `make demo-v-proj` |
| RoPE Q | `make demo-rope-q` | RoPE K | `make demo-rope-k` |
| KV write | `make demo-kv-write` | Attention score | `make demo-attention-score` |
| Softmax | `make demo-softmax` | Attention value | `make demo-attention-value` |
| O projection | `make demo-o-proj` | Attention residual | `make demo-attention-residual` |
| Post-attention RMSNorm | `make demo-post-attention-rmsnorm` | MLP gate | `make demo-mlp-gate` |
| MLP up | `make demo-mlp-up` | SiLU | `make demo-silu` |
| MLP down | `make demo-mlp-down` | MLP residual | `make demo-mlp-residual` |

The dashboard marks all 18 rows PASS only after the default shell log contains
`ACE2_SHELL_TB_PASS` and the dedicated MLP-up replay contains
`ACE2_SHELL_MLP_UP_TB_PASS`. Neither command replays the sealed full-model
schedule or claims FPGA execution.

## Engineering progression

ACE-2 reached timing closure through measured, tree-specific iterations rather
than by hiding failed candidates:

| RTL frontier | Setup slack | Result |
|---|---:|---|
| Initial complete runtime tree | -0.1484 ns | NO-GO |
| Low-fanout shell control repair | -0.5275 ns | NO-GO |
| RMSNorm capture-enable repair | -0.1741 ns | NO-GO |
| RMSNorm final-sum preload split | **+0.6966 ns** | **100 MHz PASS** |

The final split introduces `ST_MEAN_PRELOAD`, separating the 48-bit final
sum-of-squares carry from dividend loading. The exact final tree is bound by
[CERTIFIED_RTL.sha256](CERTIFIED_RTL.sha256).

## What is proven, and what is not

**Proven and carried forward unchanged from Alpha 2**

- all 18 Layer-0 operator boundaries;
- 13,914-command, 24-layer, two-token RTL execution;
- exact model/image/schedule identities;
- mapped SKY130 100 MHz and 2.0 mm2 area-cap compliance;
- independent Fresh Reviewer certification.

**Not yet claimed**

- arbitrary natural-language conversation or unrestricted generation;
- stable tokenizer, host, or deployment API;
- FPGA emulation, bitstream, or board execution;
- routed timing, power signoff, DRC/LVS, GDS, tapeout, or silicon.

See [KNOWN_LIMITATIONS.md](KNOWN_LIMITATIONS.md) for the full list.

## Historical Alpha 3 productization path

This preserved plan is superseded by the September 6 scope above; its
hardware stages are not current work.

- **Current gate:** an independent external-root channel must authenticate and
  invoke the exact accepted V8 recovery package. The current account cannot
  self-establish that trust root.
- **Next:** design and independently review a new BF16 successor after S6
  failed closed at probe lock. S6 may not be retried, resumed, or rescored.
- **Stage 1:** arbitrary-text prefill, tokenizer/host integration, KV reuse,
  readable multi-token decoding, quantized-reference/RTL agreement, and a
  one-command accelerator-facing chat demo.
- **Stage 2:** AMD/Xilinx Alveo U280 PCIe/XRT + HBM2 integration, build
  evidence, and board execution when the external toolchain and hardware are
  genuinely available.
- **Later:** board validation expansion and physical-design signoff.

Productization work is not part of the certified baseline until it receives
reproducible evidence and an independent Fresh Reviewer verdict.

## Repository map

```text
rtl/                  Certified synthesizable RTL
constraints/          Release-local timing constraints
flow/                 SKY130 synthesis/STA scripts
verification/         Deterministic vectors, tests, and runtime harnesses
tools/                Fixed-point references and image/runtime utilities
docs/                 Architecture, PPA, and traceability summaries
CERTIFIED_RTL.sha256  Exact certified RTL manifest
CERTIFICATION.md      Evidence identities and claim boundary
CHANGELOG.md          Version history
```

## Versions

- [`v0.3.0-alpha.1`](../../releases/tag/v0.3.0-alpha.1) — **ACE-2 Alpha 3**,
  productization progress with the certified Alpha 2 baseline preserved.
- [`v0.2.0-alpha.1`](../../releases/tag/v0.2.0-alpha.1) — **ACE-2 Alpha 2**,
  certified two-token RTL snapshot.
- [`v0.1.0-alpha.1`](../../releases/tag/v0.1.0-alpha.1) — **ACE-2 Alpha 1**,
  accepted prefix through `layer_0.v_proj`.

Tags preserve previous snapshots; `main` describes the latest version.

## License

Licensed under the [Apache License 2.0](LICENSE). The license applies to ACE-2
source, tools, and documentation in this repository, including preserved
historical versions, unless a file explicitly states otherwise.
