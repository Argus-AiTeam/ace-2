# Attention Datapaths

> Generated from [`ip/catalog.json`](../catalog.json). Edit the catalog and run
> `python3 scripts/generate_ip_package_docs.py` to keep metadata synchronized.

**Classification:** `standalone_cores_with_shared_shell`  
**Maturity:** standalone score/compose RTL cores with focused shell proofs

Reusable score and tiled value-composition cores plus their verified shell integration paths.

## Canonical RTL (referenced, not copied)

- `rtl/ace2_attention_score_core.sv`
- `rtl/ace2_attention_compose_core.sv`

## Shared dependencies / integration paths

- `rtl/ace2_pkg.sv`
- `rtl/ace2_shell.sv`
- `verification/tb/ace2_shell_tb.sv`

## Qwen2.5-0.5B-compatible parameters

- `HEAD_DIM`: `64`
- `score_ACT_WIDTH`: `8`
- `score_ACC_WIDTH`: `32`
- `score_OUTPUT_WIDTH`: `16`
- `compose_TILE_MAX`: `8`
- `compose_CONTEXT_MAX`: `32768`
- `shell_attention_context_max`: `8`

## Interfaces

- score start and Q/K pair ready/valid
- scaled score output ready/valid
- compose command authorization
- score tile input
- value beat ready/valid
- 128-bit composed output stream

## Verification

```sh
make ip-demo IP=attention
```

Proof type: **two shared shell proofs of canonical cores and integration**.
Runs focused score and value paths. Softmax and KV write are separate packages and commands.
Results are written under `build/ip_library/attention/`; PASS is only
reported after every mapped underlying proof passes.

## Known limitations

- This is not a complete multi-head attention subsystem by itself.
- Masking, head scheduling, memory traversal, softmax, and KV policy are shell/integration responsibilities.

## License

This package references canonical repository sources and inherits
[Apache-2.0](../../LICENSE).
