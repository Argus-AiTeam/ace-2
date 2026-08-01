# ACE-2 Alpha Demonstration

## Goal

The Alpha demo shows a real, reproducible RTL engineering workflow designed by
[Argus](https://argusbot.cn/). It demonstrates the structural
accelerator shell and the numerically accepted projection prefix without
claiming complete Qwen inference.

## Run the demo

```sh
make demo
```

The command performs:

1. tool availability checks;
2. Verilator lint of the structural `ace2_shell`;
3. deterministic regeneration of W4A8 projection vectors with the independent
   Python fixed-point oracle;
4. byte-for-byte comparison against packaged vectors;
5. Icarus compilation and simulation of the projection RTL;
6. RTL output checks against oracle-generated expected values.
7. generation of a human-readable evidence summary at
   `build/DEMO_REPORT.md`.

Expected terminal markers:

```text
ACE2_ENV_CHECK_PASS
ACE2_RTL_LINT_PASS
ACE2_ACCEPTED_PREFIX_ORACLE_PASS
ACE2_W4A8_PROJ_TB_PASS
ACE2_ACCEPTED_PREFIX_RTL_SIM_PASS
ACE2_DEMO_REPORT_WRITTEN build/DEMO_REPORT.md
ACE2_ALPHA_DEMO_PASS
```

Failures are not suppressed. Missing tools produce installation guidance.
Generated logs and binaries are written only under `build/`.

The final terminal summary explains how many oracle workloads and selected RTL
outputs were exercised, what each check proves, and where the accepted
end-to-end boundary stops. `build/DEMO_REPORT.md` preserves the same explanation
with a workload table and direct links to each raw evidence artifact.

## Suggested five-minute presentation

1. **Problem:** explain that modern LLM accelerators need both structural RTL
   coverage and end-to-end numerical correctness.
2. **Argus design process:** show
   [ARGUS_PROVENANCE.md](ARGUS_PROVENANCE.md) and explain that Argus generated,
   tested, reviewed, and rejected candidates under human-defined gates.
3. **Architecture:** show `docs/ARCHITECTURE.md` and the shell/control dataflow.
4. **Reproducible execution:** run `make demo` live.
5. **Oracle agreement:** point out that deterministic projection vectors are
   independently regenerated before RTL simulation.
6. **Engineering honesty:** show [STATUS.md](STATUS.md); distinguish 434-item
   structural coverage from the accepted numerical frontier through
   `layer_0.v_proj`.
7. **Roadmap:** explain that Beta repairs the RoPE numerical chain while reusing
   the Alpha RTL and verification base.

## Optional synthesis

```sh
make synth
```

This runs technology-independent Yosys synthesis of the accepted W4A8
projection core. The full shell uses SystemVerilog package constructs that are
not portable across all stock Yosys builds, so release-local full-shell
synthesis is not claimed. No PDK is used or distributed.

## Allowed claims

- “ACE-2 is an Argus-designed experimental accelerator Alpha.”
- “The package includes synthesizable RTL and a structural 434-item flow.”
- “The accepted contiguous numerical frontier reaches `layer_0.v_proj`.”
- “The public demo reproduces accepted-prefix oracle and RTL agreement.”
- “Beta numerical repair continues from the Alpha engineering base.”

## Disallowed claims

- complete or usable Qwen inference;
- accepted numerical support for all 434 items;
- production readiness;
- silicon, tapeout, signoff, FPGA, or PDK-qualified results;
- accepted PPA for the current rejected numerical candidate.
