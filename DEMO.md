# ACE-2 Alpha 2 Visual Demo

## Purpose

`make demo` gives reviewers a fast, reproducible path from certified source
identity to real RTL output across the Transformer data path. Every invocation
creates machine-local RMSNorm and multi-operator challenges, computes expected
answers with checked-in bit-accurate Python references, recompiles the RTL,
runs independent operator cores and selected `ace2_shell` integration modes,
emits a waveform, and proves that the checker rejects a deliberately corrupted
expectation.

## Run

```sh
make demo
```

The command performs:

1. `sha256sum -c CERTIFIED_RTL.sha256`;
2. Python, Verilator, Icarus, and VVP availability checks;
3. complete `ace2_shell` Verilator lint;
4. deterministic regeneration of 15 RMSNorm oracle cases;
5. byte-for-byte comparison with packaged JSON/SystemVerilog vectors;
6. Icarus RTL simulation of 15 cases x 56 beats;
7. generation of a fresh random challenge and a second RTL compilation/run;
8. VCD waveform generation for the local challenge;
9. a negative-control run with one intentionally corrupted expected beat,
   which must fail;
10. fresh seeded random Python-oracle cases for RoPE, attention score, softmax,
    attention compose, and SiLU, compiled into temporary testbench vectors;
10. independent RTL tests for RoPE, attention score, softmax, attention
    compose, and SiLU;
11. selected shell integration runs for attention score/compose, MLP residual,
    attention residual/post-norm, final RMSNorm, and LM head;
12. generation of `build/DEMO_REPORT.md` and
   `build/DEMO_REPORT.html`.

Expected terminal markers:

```text
ACE2_CERTIFIED_RTL_HASH_PASS
ACE2_ENV_CHECK_PASS
ACE2_RTL_LINT_PASS
ACE2_RMSNORM_ORACLE_PASS
ACE2_RMSNORM_TB_PASS cases=15 beats_per_case=56
ACE2_RMSNORM_RTL_SIM_PASS
ACE2_LOCAL_CHALLENGE_CREATED <random-id>
ACE2_RMSNORM_TB_PASS cases=16 beats_per_case=56
ACE2_LOCAL_CHALLENGE_RTL_PASS
ACE2_NEGATIVE_CONTROL_PASS expected_corruption_was_rejected
ACE2_TRANSFORMER_OPERATOR_SUITE_PASS core_groups=5 shell_modes=6
ACE2_DEMO_REPORT_WRITTEN build/DEMO_REPORT.md
ACE2_VISUAL_EVIDENCE_WRITTEN build/DEMO_REPORT.html
ACE2_LOCAL_RTL_DEMO_PASS
```

## Dashboard

Open `build/DEMO_REPORT.html` in any browser. It is self-contained and
includes:

- the host-to-token architecture flow;
- a Transformer operator matrix with per-test runtime and exact PASS markers;
- all 18 certified Layer-0 operator names, with honest fast-versus-extended
  execution status;
- certification metric cards;
- the 15 generated RMSNorm workloads;
- the fresh challenge ID, local tool versions, source commit, and output hash;
- a VCD waveform at `build/demo_challenge/rmsnorm-waveform.vcd`;
- a browser-viewable waveform preview at
  `build/demo_challenge/rmsnorm-waveform.svg`;
- evidence that the negative control was rejected;
- exact SHA-256 identity prefixes;
- the four-step timing-closure progression;
- links between claims and release-local evidence;
- a prominent statement of what Alpha 2 does not prove.

The Markdown companion is suitable for CI logs and text-only environments.

## Complete slow shell regression

```sh
make demo-extended
```

The extended target runs the default full-shell schedule and then a dedicated
`+MLP_UP_ONLY` replay. Both `ACE2_SHELL_TB_PASS` and
`ACE2_SHELL_MLP_UP_TB_PASS` are required before the report marks all 18
Layer-0 operator rows PASS.

To replay exactly the same random operator questions:

```sh
make demo SEED=<seed-shown-in-the-report>
```

## Single-operator demos

Run `make demo-operators` to list all 18 public names. Every name has a direct
target and a generic equivalent:

```sh
make demo-attention-score
make demo-operator OP=attention-score
```

Focused evidence is written to `build/single_operator/<operator>/sim.log` and
`result.json`. Paired hardware paths are not misrepresented as isolated:
`rope-q`/`rope-k` share the RoPE shell run, while `attention-residual` and
`post-attention-rmsnorm` share the vector-family shell run.

This first runs the fast demo, then executes the complete public
`ace2_shell_tb.sv` regression. It may take substantially longer than the fast
demo under Icarus. The projection family, KV write, and attention value rows
are only marked PASS by the generated dashboard after this full regression
emits `ACE2_SHELL_TB_PASS`. It still does not replay the sealed full-model
command schedule or claim FPGA execution.

## Optional schematic

With Yosys and Graphviz installed:

```sh
make visuals
```

This adds a browser-viewable synthesized projection-core schematic. It is an
open-source structural view, not a proprietary FPGA or ASIC layout.

## Presentation sequence

1. Start with the metric cards in the generated dashboard.
2. Show `CERTIFIED_RTL.sha256` and rerun `make demo`.
3. Explain why a fast RMSNorm discriminator complements, but does not replace,
   the sealed full-model run.
4. Walk through the timing progression from three NO-GO trees to +0.6966 ns.
5. End with `KNOWN_LIMITATIONS.md` and the local-chat/U280 roadmap.

## Honest interpretation

`ACE2_LOCAL_RTL_DEMO_PASS` means the packaged RTL is hash-identical to the
certified source, the release-local RMSNorm oracle/RTL discriminator passes, a
fresh local challenge passes after recompilation, the listed Transformer core
and shell-integration tests pass, and a corrupted expectation is rejected. It
does not independently rerun or expand the sealed 24-layer/two-token
certification.
