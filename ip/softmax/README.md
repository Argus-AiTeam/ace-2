# Softmax Core

> Generated from [`ip/catalog.json`](../catalog.json). Edit the catalog and run
> `python3 scripts/generate_ip_package_docs.py` to keep metadata synchronized.

**Classification:** `standalone_core`  
**Maturity:** verified RTL core integrated and focused through ace2_shell

Reusable fixed-point bounded-context softmax with Q15 probability outputs.

## Canonical RTL (referenced, not copied)

- `rtl/ace2_softmax_core.sv`

## Shared dependencies / integration paths

- `rtl/ace2_shell.sv`
- `verification/tb/ace2_shell_tb.sv`

## Qwen2.5-0.5B-compatible parameters

- `CONTEXT_MAX_default`: `8`
- `SCORE_WIDTH`: `16`
- `PROB_WIDTH`: `16`
- `probability_format`: `Q15`

## Interfaces

- start ready/valid
- context count
- packed score vector input
- packed probability vector output ready/valid
- saturation status

## Verification

```sh
make ip-demo IP=softmax
```

Proof type: **focused shared shell proof of standalone core**.
Runs the +SOFTMAX_ONLY shell mode against the canonical softmax core.
Results are written under `build/ip_library/softmax/`; PASS is only
reported after every mapped underlying proof passes.

## Known limitations

- The ACE-2 shell instantiates CONTEXT_MAX=8.
- The LUT/rounding behavior is fixed-point ACE-2 behavior, not a general floating-point softmax implementation.

## License

This package references canonical repository sources and inherits
[Apache-2.0](../../LICENSE).
