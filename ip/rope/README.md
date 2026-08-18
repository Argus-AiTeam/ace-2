# RoPE Core

> Generated from [`ip/catalog.json`](../catalog.json). Edit the catalog and run
> `python3 scripts/generate_ip_package_docs.py` to keep metadata synchronized.

**Classification:** `standalone_core`  
**Maturity:** verified RTL core with a paired Q/K shared shell proof

Reusable fixed-point rotary position embedding datapath for paired Q/K activation slices.

## Canonical RTL (referenced, not copied)

- `rtl/ace2_rope_core.sv`

## Shared dependencies / integration paths

- `rtl/ace2_shell.sv`
- `verification/tb/ace2_shell_tb.sv`

## Qwen2.5-0.5B-compatible parameters

- `LANES_core_default`: `16`
- `LANES_shell`: `2`
- `ACT_WIDTH`: `8`
- `SCALE_WIDTH`: `16`
- `TRIG_WIDTH`: `16`
- `head_dim`: `64`

## Interfaces

- start ready/valid
- paired activation beat ready/valid
- per-lane scales
- cos/sin inputs
- first/second-half selector
- output ready/valid and saturation status

## Verification

```sh
make ip-demo IP=rope
```

Proof type: **paired shared shell proof**.
The +ROPE_ONLY proof executes both Q and K roles; rope-q and rope-k are names for the same paired run.
Results are written under `build/ip_library/rope/`; PASS is only
reported after every mapped underlying proof passes.

## Known limitations

- Position/scaling data is supplied externally.
- The package does not provide a tokenizer, sequence scheduler, or arbitrary RoPE variant adapter.

## License

This package references canonical repository sources and inherits
[Apache-2.0](../../LICENSE).
