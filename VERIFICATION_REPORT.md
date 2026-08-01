# Release Verification Report

Date: 2026-08-01  
Package: ACE-2 `v0.1.0-alpha`

## Verified locally

- `make demo`: passed.
  - Environment check passed.
  - Full structural shell Verilator lint passed with four non-fatal warnings:
    one colocated module filename warning and three intentionally unused-bit
    warnings.
  - Accepted-prefix projection oracle regeneration matched the packaged vectors
    byte for byte.
  - Icarus projection simulation passed 5 cases and 21 checked outputs.
- `make synth`: passed for technology-independent Yosys synthesis of
  `ace2_w4a8_proj_core`; Yosys reported zero design-check problems.
- Sensitive-reference scan: no absolute home paths, agent/runtime paths,
  credential assignments, private-key headers, raw private benchmark paths, or
  quarantine paths found.
- Disallowed artifact scan: clean after removing generated `build/` and Python
  cache directories.

## Not verified or claimed

- Full-shell technology mapping, place-and-route, STA, or power.
- Full-model numerical correctness or usable Qwen inference.
- End-to-end support at or after `layer_0.rope_q`.
- The rejected projection-shadow staged-attention candidate.

## Release blocker

Public redistribution remains blocked because no clear project-level license
was present. See `LICENSE_PENDING.md`.
