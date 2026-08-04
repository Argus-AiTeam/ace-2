# Development History to the ACE-2 Alpha

> **Historical Alpha 1 snapshot.** This document intentionally ends at the
> first release. Alpha 2 changes are summarized in `CHANGELOG.md`.

This is a curated, public-safe engineering history of **Argus Compute Engine 2
(ACE-2)** from mission definition through the `v0.1.0-alpha` package cut. It is
not a raw agent transcript. Private prompts, credentials, machine paths,
datasets, model weights, internal benchmark material, and runtime state are
deliberately excluded.

This document ends at the Alpha baseline. Post-Alpha Beta repair experiments
are intentionally outside its scope.

## 1. Mission and acceptance contract

The project began as a Qwen2.5-0.5B-oriented W4A8 accelerator research mission.
Argus separated the work into independently reviewable layers:

```text
model-flow contract
  -> architecture and memory model
  -> shell, control, and DMA interfaces
  -> fixed-point operator RTL
  -> independent software references
  -> deterministic RTL verification
  -> end-to-end numerical acceptance
  -> synthesis and PPA evidence
```

From the beginning, source-code presence was not treated as sufficient proof of
usable inference. Structural coverage, local block correctness, and composed
model quality were tracked as different claims.

## 2. Structural execution framework

Argus constructed a descriptor-driven accelerator shell representing the
complete 434-item model flow. The engineering base included:

- command, status, error, and completion handling;
- DMA-style input, weight, metadata, and output interfaces;
- reusable fixed-point compute cores;
- layer/operator dispatch and downstream control infrastructure;
- RMSNorm, W4A8 projection, RoPE research paths, attention, softmax, SiLU, and
  composition blocks;
- traceability between model-flow items and RTL structure.

This stage established broad implementation infrastructure. It did not claim
that every represented operator was numerically accepted end to end.

## 3. Independent numerical references

The verification strategy was built around independent software references
rather than self-comparison against RTL:

- signed W4A8 projection arithmetic;
- per-output multiplier, shift, zero-point, rounding, and saturation behavior;
- RMSNorm-to-projection scale compatibility;
- deterministic synthetic inputs, weights, metadata, and expected outputs;
- SHA-256 identities for generated evidence.

The release retains the projection and RMSNorm reference subset used by the
public demo. Five packaged oracle workloads represent balanced arithmetic,
saturation boundaries, MLP gate and down projections, and the
RMSNorm-to-projection consumer contract.

## 4. RTL verification and implementation hardening

Argus added lint, simulation, directed semantic cases, and generic synthesis
entrypoints. The accepted-prefix testbench checks selected outputs across the
five oracle workloads and explicitly exercises rounding ties, shift-zero,
extreme-shift, positive, and negative arithmetic.

The complete structural shell is linted separately from accepted-prefix
numerical simulation. This prevents a clean parser/elaboration result from
being misreported as full-model correctness.

## 5. Full-model evidence changed the support claim

Local operator checks initially made the broad RTL framework look more complete
than it was numerically. Full-model evaluation then exposed the first material
end-to-end divergence at the layer-0 RoPE boundary.

Argus responded by reducing the accepted contiguous frontier to:

```text
layer_0.input_rmsnorm
layer_0.q_proj
layer_0.k_proj
layer_0.v_proj
```

The first unsupported operator became `layer_0.rope_q`. Downstream RTL was
preserved because it remained useful structural and research infrastructure,
but it was no longer described as accepted composed-model support.

## 6. Bounded numerical repair research

Before the Alpha cut, several RoPE/attention research paths were implemented or
retained for bounded investigation, including dynamic head scaling, fixed-Q7
score handling, relative-RoPE score fusion, and staged attention composition.

The final pre-Alpha projection-shadow staged-attention candidate passed its
standalone software reference, generated-vector checks, RTL simulation, and
lint. It nevertheless failed the two-dataset full-model discriminator and
paired smoke. Argus sealed it as a bounded no-go rather than promoting a local
PASS into an end-to-end capability claim.

This result established the central Alpha lesson:

> Standalone RTL correctness is necessary, but it is not evidence of usable
> model inference when an upstream numerical contract is wrong.

## 7. Preserved implementation and PPA context

The failed numerical candidate did not erase the rest of the design. The
shell, control system, DMA interfaces, operator library, verification tools,
and downstream RTL remained available for subsequent repair.

The preserved historical aggregate recorded:

- 62,199 cells;
- 0.6108746272 mm² reported non-SRAM area;
- +0.1502 ns setup slack at 100 MHz.

These are historical contextual values, not new measurements produced by the
release package. They are not tapeout, routed-GDS, signoff, FPGA, or silicon
claims.

## 8. Alpha package preparation

The Alpha was assembled through a whitelist rather than by recursively copying
the working tree. The package includes only the source, references, synthetic
vectors, testbench, documentation, and release tooling needed to reproduce its
bounded claims.

It deliberately excludes:

- model weights and datasets;
- private benchmark inputs and raw internal evidence;
- credentials, agent state, prompts, logs, and machine configuration;
- proprietary PDK, IP, and toolchain material;
- rejected standalone candidate RTL not required by the structural shell.

The release workflow added:

- `make demo` for environment, shell lint, oracle regeneration, RTL simulation,
  and evidence reporting;
- `make visuals` for an English HTML dashboard, a Yosys-derived SVG netlist
  schematic, and a standard VCD simulation waveform;
- `make synth` for technology-independent projection-core synthesis;
- `MANIFEST.sha256` for release-file integrity.

## 9. Alpha baseline

The `v0.1.0-alpha` baseline therefore makes a deliberately narrow but
reproducible claim:

| Claim | Alpha position |
|---|---|
| Complete 434-item model flow represented structurally | Yes |
| Shell/control/DMA and downstream RTL preserved | Yes |
| Accepted contiguous numerical frontier through `layer_0.v_proj` | Yes |
| Full end-to-end numerical support | No |
| Usable Qwen inference | Not claimed |
| First unresolved operator | `layer_0.rope_q` |
| Public reproducible demo | Accepted-prefix projection subset |

The Alpha is not the end of the numerical repair effort. It is the auditable
engineering baseline from which Beta development continues without restarting
the accelerator from zero.

## Evidence map

| History statement | Release evidence |
|---|---|
| Structural shell and operator library exist | `rtl/`, `docs/ARCHITECTURE.md` |
| Model-flow claim is structural, not numerical | `STATUS.md`, `KNOWN_LIMITATIONS.md` |
| Accepted projection prefix is reproducible | `tools/`, `verification/`, `make demo` |
| Visual evidence can be regenerated | `make visuals`, `DEMO.md` |
| Historical PPA values are contextual only | `docs/PPA_SUMMARY.md` |
| Included and excluded material is explicit | `RELEASE_INVENTORY.md` |
| Argus designed and iterated the system | `ARGUS_PROVENANCE.md` |
| Release file integrity is deterministic | `MANIFEST.sha256` |
