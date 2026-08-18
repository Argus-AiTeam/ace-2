# W4A8 Projection Core

> Generated from [`ip/catalog.json`](../catalog.json). Edit the catalog and run
> `python3 scripts/generate_ip_package_docs.py` to keep metadata synchronized.

**Classification:** `standalone_core`  
**Maturity:** verified RTL core; Q/K/V may use legacy descriptors or the shell's ordered fused-QKV descriptor, while O, MLP gate/up/down, and LM-head roles remain separately scheduled

Reusable signed INT8 activation x signed INT4 weight projection MAC, requantization, bias, saturation, and overflow reporting.

## Canonical RTL (referenced, not copied)

- `rtl/ace2_w4a8_proj_core.sv`

## Shared dependencies / integration paths

- `rtl/ace2_shell.sv`
- `verification/tb/ace2_shell_tb.sv`

## Qwen2.5-0.5B-compatible parameters

- `K_SIZE`: `[896, 4864]`
- `MAC_LANES`: `4`
- `ACT_WIDTH`: `8`
- `WGT_WIDTH`: `4`
- `ACC_WIDTH`: `32`

## Interfaces

- start ready/valid
- packed activation/weight pair ready/valid
- requantization metadata ready/valid
- INT8 output ready/valid
- accumulator/overflow/saturation status

## Verification

```sh
make ip-demo IP=w4a8_projection
```

Proof type: **shared shell proof of canonical standalone core**.
The focused Q-projection stride mode instantiates the canonical projection core through ace2_shell; other projection roles share this core but have separate operator demos.
Results are written under `build/ip_library/w4a8_projection/`; PASS is only
reported after every mapped underlying proof passes.

## Known limitations

- The core computes one output accumulation at a time; matrix traversal, SRAM addressing, weights, scales, activation-tile reuse, and role scheduling belong to ace2_shell.
- Fused QKV reuses this shared core in three ordered phases; it is not three physically parallel projection cores.
- The focused package command proves Q projection, not every projection role.

## License

This package references canonical repository sources and inherits
[Apache-2.0](../../LICENSE).
