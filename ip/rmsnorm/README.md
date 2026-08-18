# RMSNorm Core

> Generated from [`ip/catalog.json`](../catalog.json). Edit the catalog and run
> `python3 scripts/generate_ip_package_docs.py` to keep metadata synchronized.

**Classification:** `standalone_core`  
**Maturity:** independently simulated RTL core with deterministic oracle vectors; the broader demo also provides a fresh challenge path

Reusable streaming fixed-point RMSNorm for the ACE-2 896-element hidden vector.

## Canonical RTL (referenced, not copied)

- `rtl/ace2_rmsnorm_core.sv`

## Shared dependencies / integration paths

- `verification/tb/ace2_rmsnorm_tb.sv`
- `verification/generated/rmsnorm_vectors.svh`

## Qwen2.5-0.5B-compatible parameters

- `HIDDEN_SIZE`: `896`
- `LANES`: `16`
- `ACT_WIDTH`: `8`
- `GAIN_WIDTH`: `16`
- `ACC_WIDTH`: `48`
- `INV_RMS_FRAC`: `30`
- `GAIN_FRAC`: `8`

## Interfaces

- start ready/valid
- activation stream ready/valid
- gain stream ready/valid
- scale-activation stream ready/valid
- output stream ready/valid
- done handshake and numerical status

## Verification

```sh
make ip-demo IP=rmsnorm
```

Proof type: **isolated core testbench**.
Runs ace2_rmsnorm_tb.sv against checked-in independent oracle vectors.
Results are written under `build/ip_library/rmsnorm/`; PASS is only
reported after every mapped underlying proof passes.

## Known limitations

- The public proof is parameterized around the Qwen2.5-0.5B hidden size and fixed-point formats; other dimensions/formats are not certified.

## License

This package references canonical repository sources and inherits
[Apache-2.0](../../LICENSE).
