# Argus Compute Engine 2 (ACE-2)

**Latest snapshot: `v0.2.0-alpha.1` (ACE-2 Alpha 2)**

ACE-2 is an experimental Qwen2.5-0.5B W4A8 accelerator designed and
iterated by [Argus](https://argusbot.cn/) under human-defined objectives and
review gates.

Alpha 2 advances the accepted scope from an early Layer-0 prefix to:

- all 18 Layer-0 operators passing their exact fixed-point checks;
- a complete 24-layer, two-token Qwen command schedule;
- 13,914/13,914 RTL runtime commands passing;
- generated token IDs `[0, 0]` with `first_failure = null`;
- a certified 23-file RTL tree;
- mapped SKY130 synthesis/OpenSTA at 100 MHz with 62,283 cells,
  0.614082704 mm2 non-SRAM area, and +0.6966 ns detailed setup slack.

The exact certification boundary is documented in
[CERTIFICATION.md](CERTIFICATION.md). This is not yet a general chat product:
arbitrary-text prefill, unrestricted generation, host deployment, FPGA board
execution, routed timing, power signoff, DRC/LVS, tapeout, and silicon are not
claimed.

## Reproduce the release-local checks

Required for the basic public-safe checks:

- Python 3
- GNU Make
- Verilator
- Icarus Verilog

Verify the certified RTL snapshot:

```sh
sha256sum -c CERTIFIED_RTL.sha256
```

Run the deterministic projection demonstration:

```sh
make demo
```

Expected marker:

```text
ACE2_ALPHA2_DEMO_PASS
```

The full two-token run additionally requires the pinned Qwen model revision,
the accepted packed-W4 image, and the runtime package described in
[CERTIFICATION.md](CERTIFICATION.md). Model weights, private benchmark inputs,
raw PDK data, build outputs, and local Argus state are intentionally excluded.

## Repository layout

```text
rtl/                  Certified synthesizable RTL
constraints/          Release-local timing constraints
flow/                 SKY130 synthesis/STA scripts
verification/         Deterministic vectors, tests, and runtime harnesses
tools/                Fixed-point references and image/runtime utilities
docs/                 Architecture and traceability documentation
CERTIFIED_RTL.sha256  Exact certified RTL file manifest
CERTIFICATION.md      Alpha 2 evidence and claim boundary
CHANGELOG.md          Version history
STATUS.md             Current support summary
KNOWN_LIMITATIONS.md  Explicit non-claims
```

## Version history

- [`v0.2.0-alpha.1`](../../releases/tag/v0.2.0-alpha.1): Alpha 2 certified
  two-token RTL snapshot.
- [`v0.1.0-alpha.1`](../../releases/tag/v0.1.0-alpha.1): Alpha 1 accepted
  prefix through `layer_0.v_proj`.

Git tags preserve previous source snapshots. `main` documents the latest
snapshot; old source trees are not duplicated in directories.

## License

Licensed under the [Apache License 2.0](LICENSE). The license applies to ACE-2
source, tools, and documentation in this repository, including preserved
historical versions, unless a file explicitly states otherwise.
