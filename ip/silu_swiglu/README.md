# SiLU/SwiGLU Datapath

> Generated from [`ip/catalog.json`](../catalog.json). Edit the catalog and run
> `python3 scripts/generate_ip_package_docs.py` to keep metadata synchronized.

**Classification:** `standalone_core`  
**Maturity:** verified RTL core with focused shell proof

Reusable fixed-point SiLU lookup, gate/up multiplication, requantization, and saturation core.

## Canonical RTL (referenced, not copied)

- `rtl/ace2_silu_gate_core.sv`

## Shared dependencies / integration paths

- `rtl/generated/ace2_silu_lut.svh`
- `rtl/ace2_shell.sv`
- `verification/tb/ace2_shell_tb.sv`

## Qwen2.5-0.5B-compatible parameters

- `input_lane_width`: `16`
- `input_lanes_per_beat`: `8`
- `output_width`: `8`
- `shell_intermediate_size`: `4864`

## Interfaces

- start/config handshake
- gate and up packed beat ready/valid
- lane count
- INT8 packed output ready/valid
- saturation status

## Verification

```sh
make ip-demo IP=silu_swiglu
```

Proof type: **focused shared shell proof of standalone core**.
Runs the +SILU_ONLY shell mode and checks ACE2_SHELL_SILU_GATE_TB_PASS.
Results are written under `build/ip_library/silu_swiglu/`; PASS is only
reported after every mapped underlying proof passes.

## Known limitations

- Gate and up projections are separate projection-core/shell operations.
- The LUT and quantization formats are ACE-2-specific rather than arbitrary activation formats.

## License

This package references canonical repository sources and inherits
[Apache-2.0](../../LICENSE).
