# Qwen2.5-0.5B Transformer Layer Integration Bundle

> Generated from [`ip/catalog.json`](../catalog.json). Edit the catalog and run
> `python3 scripts/generate_ip_package_docs.py` to keep metadata synchronized.

**Classification:** `integration_bundle`  
**Maturity:** selected reusable cores plus shared shell integration; not a single drop-in layer core

Discoverable bundle of the canonical ACE-2 shell and cores used for the demonstrated Qwen2.5-0.5B W4A8 layer path.

## Canonical RTL (referenced, not copied)

- `rtl/ace2_pkg.sv`
- `rtl/ace2_shell.sv`
- `rtl/ace2_rmsnorm_core.sv`
- `rtl/ace2_w4a8_proj_core.sv`
- `rtl/ace2_rope_core.sv`
- `rtl/ace2_attention_score_core.sv`
- `rtl/ace2_softmax_core.sv`
- `rtl/ace2_attention_compose_core.sv`
- `rtl/ace2_silu_gate_core.sv`

## Shared dependencies / integration paths

- `rtl/generated`
- `verification/tb/ace2_shell_tb.sv`

## Qwen2.5-0.5B-compatible parameters

- `model`: `Qwen2.5-0.5B`
- `hidden_size`: `896`
- `intermediate_size`: `4864`
- `head_dim`: `64`
- `layers_in_certified_schedule`: `24`
- `activation_width`: `8`
- `weight_width`: `4`
- `shell_attention_context_max`: `8`

## Interfaces

- ace2_shell CSR interface
- command descriptor interface
- 128-bit tagged memory interface
- interrupt/completion/status interface

## Verification

```sh
make ip-demo IP=qwen25_transformer_layer
```

Proof type: **selected shared shell integration proofs**.
The package demo checks residual/post-norm and MLP-residual integration slices. It is not a full-layer or full-model replay; use existing extended/certification evidence for those separate scopes.
Results are written under `build/ip_library/qwen25_transformer_layer/`; PASS is only
reported after every mapped underlying proof passes.

## Known limitations

- There is no single ace2_transformer_layer module.
- The package demo is a selected integration proof, not full-model chat completion.
- Host runtime, weights, tokenizer, unrestricted generation, and FPGA deployment are outside this package.

## License

This package references canonical repository sources and inherits
[Apache-2.0](../../LICENSE).
