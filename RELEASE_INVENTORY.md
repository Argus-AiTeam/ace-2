# Release Inventory and Rationale

This package was assembled by strict whitelist. No source-tree directory was
copied recursively.

## Included

- `rtl/ace2_pkg.sv`, `rtl/ace2_shell.sv`: public package constants and the
  descriptor/control/DMA structural shell.
- `rtl/ace2_rmsnorm_core.sv`, `rtl/ace2_w4a8_proj_core.sv`,
  `rtl/ace2_rope_core.sv`, `rtl/ace2_dynamic_rope_head_core.sv`,
  `rtl/ace2_fixed_q7_rope_score_core.sv`,
  `rtl/ace2_relative_rope_score_fusion_core.sv`,
  `rtl/ace2_attention_score_core.sv`, `rtl/ace2_softmax_core.sv`,
  `rtl/ace2_attention_compose_core.sv`, and
  `rtl/ace2_silu_gate_core.sv`: synthesizable operator dependencies required
  to lint and elaborate the structural shell. Their inclusion is not a claim
  of end-to-end numerical acceptance.
- `rtl/generated/ace2_silu_lut.svh`: generated constant table required by the
  included SiLU RTL.
- `tools/ace2_projection_reference.py`: independent fixed-point projection
  oracle.
- `tools/ace2_rmsnorm_reference.py`: oracle dependency used to construct the
  RMSNorm-to-projection vector case.
- `tools/gen_projection_vectors.py`: deterministic public synthetic-vector
  generator.
- `verification/generated/projection_vectors.{json,svh}`: packaged synthetic
  accepted-prefix vectors.
- `verification/tb/ace2_w4a8_proj_tb.sv`: projection RTL testbench.
- release-authored build, documentation, policy, scan, and manifest files.

## Deliberately excluded

Version-control metadata; hidden state; agent/session/daemon files; caches;
virtual environments; build outputs; quarantine; raw/internal logs; credentials;
keys; tokens; model weights; datasets; raw/private benchmarks; hidden evaluator
material; candidate evidence trees; proprietary or restricted PDK/IP/toolchain
files; and machine-specific configuration.

Standalone rejected candidate RTL that is not needed by the accepted shell is
also excluded. Its status is documented without distributing private evidence.
