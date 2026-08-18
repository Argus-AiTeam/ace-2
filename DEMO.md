# ACE-2 Alpha 2 Visual Demo

## Purpose

`make demo` gives reviewers a fast, reproducible path from certified source
identity to real RTL output. Every invocation creates a new machine-local
challenge after the command starts, recompiles the RTL, emits a VCD waveform,
and proves that the checker rejects a deliberately corrupted expectation. It
intentionally avoids replaying the sealed 1,240,410,384-cycle full-model run.

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
10. generation of `build/DEMO_REPORT.md` and
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
ACE2_DEMO_REPORT_WRITTEN build/DEMO_REPORT.md
ACE2_VISUAL_EVIDENCE_WRITTEN build/DEMO_REPORT.html
ACE2_LOCAL_RTL_DEMO_PASS
```

## Dashboard

Open `build/DEMO_REPORT.html` in any browser. It is self-contained and
includes:

- the host-to-token architecture flow;
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
fresh local challenge passes after recompilation, and a corrupted expectation
is rejected. It does not independently rerun or expand the full two-token
certification.
