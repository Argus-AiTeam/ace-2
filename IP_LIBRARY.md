# ACE-2 Open IP Library

[中文](IP_LIBRARY.zh-CN.md) | [Machine-readable catalog](ip/catalog.json)

The Open IP Library packages the existing, verified ACE-2 Qwen2.5-0.5B W4A8
RTL into discoverable reuse units. It references the canonical files in
`rtl/`; it does not fork or copy the certified implementations.

## Catalog

| Package | Reuse boundary | Public proof |
|---|---|---|
| [`w4a8_projection`](ip/w4a8_projection/) | Standalone projection core; shell schedules matrix roles | Q-projection shell mode |
| [`rmsnorm`](ip/rmsnorm/) | Standalone streaming core | Isolated RTL testbench |
| [`rope`](ip/rope/) | Standalone core | Paired Q/K shell mode |
| [`kv_cache`](ip/kv_cache/) | Shared shell write path, not an isolated cache core | KV-write shell mode |
| [`attention`](ip/attention/) | Standalone score/compose cores plus shell integration | Score and value modes |
| [`softmax`](ip/softmax/) | Standalone bounded-context core | Softmax shell mode |
| [`silu_swiglu`](ip/silu_swiglu/) | Standalone activation/gating core | SiLU shell mode |
| [`mlp`](ip/mlp/) | Integration bundle, not one standalone core | Gate/up/SiLU/down/residual modes |
| [`qwen25_transformer_layer`](ip/qwen25_transformer_layer/) | Shell integration bundle, not one drop-in layer | Selected residual/norm integration modes |

The 18 existing operator demos establish support across the ACE-2 data path,
but several names share a core or shell proof. Package manifests therefore
label each item as a standalone core, a shared shell path, or an integration
bundle.

## Discover and run

```sh
make ip-list
make ip-validate
make ip-demo IP=rmsnorm
make ip-softmax                 # shortcut
make ip-demo-all                # includes the slow MLP-up proof
```

Results are written to `build/ip_library/<package>/result.json`. A package is
reported PASS only when every mapped `run_single_operator_demo.py` proof
returns successfully and its required RTL marker is present.

## Integration example

Consume canonical sources directly. For a projection-core integration:

```systemverilog
ace2_w4a8_proj_core #(
    .K_SIZE(896), .MAC_LANES(4), .ACT_WIDTH(8),
    .WGT_WIDTH(4), .ACC_WIDTH(32)
) u_q_projection (/* connect the documented ready/valid ports */);
```

Compile from the repository root with `rtl/ace2_w4a8_proj_core.sv`; do not copy
the source into an `ip/` package. Matrix traversal, memory addressing, weight
and scale delivery, and Q/K/V/O/MLP role scheduling remain integration
responsibilities implemented by `ace2_shell`. The shell now also exposes
fused opcode `0x0b`, which caches one activation tile and executes ordered
Q/K/V phases through the shared projection engine. It is an integration bundle,
not a claim of three physically parallel projection cores.

## Scope and non-claims

This library is compatible with the demonstrated Qwen2.5-0.5B W4A8 fixed-point
path (hidden size 896, intermediate size 4864, head dimension 64). It does not
claim arbitrary Transformer support, a stable full-model inference API,
arbitrary-text chat completion, model weights, or FPGA deployment.

## Contributing

Public contributions are welcome for portable wrappers, independent oracles,
redistribution-safe vectors, documentation, and integrations that preserve
the evidence boundary. Add a package only when its canonical sources,
interfaces, dependencies, proof mapping, limitations, and Apache-2.0
inheritance are explicit. See [CONTRIBUTING.md](CONTRIBUTING.md).
