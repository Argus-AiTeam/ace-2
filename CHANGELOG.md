# Changelog

## v0.2.0-alpha.1 - 2026-08-04

ACE-2 Alpha 2:

- advanced Layer-0 acceptance from the early projection prefix to 18/18
  fixed-point operators;
- added the complete 24-layer, two-token Qwen2.5-0.5B W4A8 command path;
- completed 13,914/13,914 RTL runtime commands with token IDs `[0, 0]`;
- fixed the SiLU packed-int8 adapter and dense-W4 LM-head row stride;
- reduced shell/FSM and RMSNorm timing cones, including the
  `ST_MEAN_PRELOAD` split;
- certified the 23-file RTL tree;
- achieved mapped SKY130/OpenSTA 100 MHz PASS at 62,283 cells and
  0.614082704 mm2 non-SRAM area;
- retained explicit non-claims for general chat, FPGA, physical signoff,
  tapeout, and silicon.

## v0.1.0-alpha.1 - 2026-08-01

ACE-2 Alpha 1:

- published the structural accelerator shell and deterministic projection
  demonstration;
- accepted the contiguous numerical prefix through `layer_0.v_proj`;
- identified `layer_0.rope_q` as the first unsupported composed operator;
- made no full-model inference claim.
