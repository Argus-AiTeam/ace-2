# KV Cache Path

> Generated from [`ip/catalog.json`](../catalog.json). Edit the catalog and run
> `python3 scripts/generate_ip_package_docs.py` to keep metadata synchronized.

**Classification:** `shared_shell_path`  
**Maturity:** verified integration path; not an isolated standalone cache controller core

Descriptor, memory, and state-machine path for verified KV writes in the ACE-2 shell.

## Canonical RTL (referenced, not copied)

- `rtl/ace2_shell.sv`
- `rtl/ace2_pkg.sv`

## Shared dependencies / integration paths

- `verification/tb/ace2_shell_tb.sv`

## Qwen2.5-0.5B-compatible parameters

- `hidden_size`: `896`
- `head_dim`: `64`
- `memory_data_width`: `128`
- `shell_attention_context_max`: `8`

## Interfaces

- ace2_shell command descriptor interface
- 128-bit tagged memory request/response interface
- completion/status path

## Verification

```sh
make ip-demo IP=kv_cache
```

Proof type: **shared shell path**.
Runs the +KV_WRITE_ONLY ace2_shell proof; there is no claim of a separately instantiable KV-cache core.
Results are written under `build/ip_library/kv_cache/`; PASS is only
reported after every mapped underlying proof passes.

## Known limitations

- KV storage allocation, host policy, long-context management, and cache reuse APIs are not standalone public IP.
- The shell proof covers the demonstrated bounded context path, not arbitrary context lengths.

## License

This package references canonical repository sources and inherits
[Apache-2.0](../../LICENSE).
