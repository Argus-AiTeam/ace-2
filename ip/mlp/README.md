# Qwen2.5 MLP Integration Bundle

> Generated from [`ip/catalog.json`](../catalog.json). Edit the catalog and run
> `python3 scripts/generate_ip_package_docs.py` to keep metadata synchronized.

**Classification:** `integration_bundle`  
**Maturity:** verified operator and integration paths; not a single isolated MLP core

Shared-shell composition of gate/up/down W4A8 projections, SiLU/SwiGLU, and residual update.

## Canonical RTL (referenced, not copied)

- `rtl/ace2_w4a8_proj_core.sv`
- `rtl/ace2_silu_gate_core.sv`
- `rtl/ace2_shell.sv`

## Shared dependencies / integration paths

- `rtl/generated/ace2_silu_lut.svh`
- `verification/tb/ace2_shell_tb.sv`

## Qwen2.5-0.5B-compatible parameters

- `hidden_size`: `896`
- `intermediate_size`: `4864`
- `activation_width`: `8`
- `weight_width`: `4`
- `vector_lanes`: `16`

## Interfaces

- ace2_shell command descriptor interface
- 128-bit tagged memory interface
- completion/status path

## Verification

```sh
make ip-demo IP=mlp
```

Proof type: **multi-command shared shell integration bundle**.
Runs every public MLP operator proof, including the slow full 896 x 4864 MLP-up projection.
Results are written under `build/ip_library/mlp/`; PASS is only
reported after every mapped underlying proof passes.

## Known limitations

- No separately instantiable ace2_mlp_core module exists.
- The MLP-up proof is slow under Icarus.
- Weights, scales, memory layout, and command generation are external.

## License

This package references canonical repository sources and inherits
[Apache-2.0](../../LICENSE).
