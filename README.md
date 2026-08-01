# Argus Compute Engine 2 (ACE-2) Experimental Alpha

**Version: `v0.1.0-alpha`**

ACE stands for **Argus Compute Engine**. ACE-2 is an engineering and research
Alpha for a Qwen2.5-0.5B-oriented W4A8
accelerator. The architecture, RTL, verification flow, numerical experiments,
and evidence-driven revisions were designed and iterated by
[Argus](https://argusbot.cn/) under human-defined objectives,
approval gates, and immutable engineering targets.

Argus is an autonomous research and engineering agent system. Public Argus
activity and research pages are available at:

- [Argus official website](https://argusbot.cn/)
- [Argus public results](https://argusbot.cn/results.html)
- [Argus public research](https://argusbot.cn/research.html)

See [ARGUS_PROVENANCE.md](ARGUS_PROVENANCE.md) for how Argus planned, built,
tested, rejected, and revised ACE-2.

## What this Alpha is

This package is a self-contained, redistribution-oriented engineering snapshot.
It includes:

- synthesizable SystemVerilog for the accelerator shell and principal compute
  blocks;
- a structural execution framework covering the 434-item model flow;
- command/control, DMA-facing, arithmetic, RoPE, attention, softmax, and
  activation building blocks;
- deterministic public projection vectors;
- an independent Python fixed-point oracle;
- RTL testbench, lint, simulation, and generic synthesis entrypoints;
- architecture, traceability, PPA-context, status, and limitation documents;
- a deterministic SHA-256 release manifest.

It contains no model weights, datasets, private benchmark inputs, PDK files,
proprietary IP, toolchain binaries, credentials, or Argus runtime/session state.

## Critical engineering status

| Area | Alpha status |
|---|---|
| Structural RTL/framework coverage | Full 434-item execution flow |
| Accepted contiguous numerical frontier | Through `layer_0.v_proj` |
| First unsupported end-to-end operator | `layer_0.rope_q` |
| Full-model numerical quality | **Not accepted** |
| Usable Qwen inference | **Not claimed** |
| Latest projection-shadow candidate | Bounded no-go |
| Alpha purpose | RTL architecture, workflow, and accepted-prefix demonstration |

The complete structural shell must not be interpreted as complete numerical
support. Downstream blocks exist and can be tested independently, but the first
end-to-end numerical break invalidates the composed model result after the
accepted prefix. The Beta effort repairs that numerical chain while reusing the
Alpha shell, control system, downstream RTL, and verification infrastructure.

Read [STATUS.md](STATUS.md) and
[KNOWN_LIMITATIONS.md](KNOWN_LIMITATIONS.md) before citing results.

## Five-minute demonstration

Required tools:

- Python 3
- GNU Make
- Verilator
- Icarus Verilog (`iverilog` and `vvp`)

Run:

```sh
make demo
```

The demo:

1. checks the public tool environment;
2. lints the structural accelerator shell;
3. regenerates accepted-prefix W4A8 projection vectors with an independent
   Python oracle;
4. compares generated vectors byte-for-byte;
5. compiles and simulates the projection RTL;
6. checks RTL results against oracle-generated expected values.
7. writes a standalone, English visual evidence dashboard at
   `build/DEMO_REPORT.html`, plus `build/DEMO_REPORT.md`, explaining the
   exercised workloads, machine-produced evidence, exact hashes, and honest
   support boundary.

Expected final marker:

```text
ACE2_ALPHA_DEMO_PASS
```

Optional technology-independent synthesis:

```sh
make synth
```

For browser-viewable synthesized netlist graphics and a standard simulation
waveform:

```sh
make visuals
```

This creates `build/ace2_w4a8_proj_schematic.svg` and
`build/projection-waveform.vcd`. The VCD is compatible with GTKWave, Vivado,
and other standard waveform viewers; no proprietary tool is required to
generate it.

See [DEMO.md](DEMO.md) for expected output, presentation guidance, and honest
claim boundaries.

## Repository layout

```text
rtl/                    Synthesizable accelerator RTL
verification/           Public deterministic vectors and RTL testbench
tools/                  Independent numerical references and vector generator
docs/                   Architecture, traceability, and PPA context
scripts/                Deterministic release-manifest generator
Makefile                Demo, lint, simulation, synthesis, and manifest targets
STATUS.md               Current accepted engineering state
KNOWN_LIMITATIONS.md     Explicit non-claims and unresolved numerical boundary
ARGUS_PROVENANCE.md      Argus design and evidence-driven iteration process
VERIFICATION_REPORT.md  Release-local checks and scan results
RELEASE_INVENTORY.md    Included/excluded content rationale
```

## Reproducibility and evidence policy

ACE-2 follows an evidence-first workflow:

- structural presence is not treated as numerical acceptance;
- generated vectors are deterministic and independently reproduced;
- failures remain visible rather than being rewritten into passing evidence;
- accepted support advances only as a contiguous end-to-end frontier;
- rejected candidates are retained as bounded negative evidence;
- release files are covered by `MANIFEST.sha256`.

Regenerate and verify the manifest after intentional release changes:

```sh
make clean
make demo
make clean
make manifest
sha256sum -c MANIFEST.sha256
```

## Alpha-to-Beta development

`v0.1.0-alpha` is the public engineering baseline. Beta development should:

1. preserve the Alpha tag and reproducible demo;
2. repair the numerical chain beginning at `layer_0.rope_q`;
3. require independent oracle, RTL, and two-dataset quality evidence;
4. rerun downstream verification and PPA after numerical acceptance;
5. avoid claiming complete inference until the full model quality gate passes.

## Licensing blocker

No clear ACE-2 project-level license was present in the source snapshot. The
Argus software repository is MIT-licensed, but that does not automatically
license ACE-2. Public redistribution is blocked until the ACE-2 owner selects
and approves a project license. See [LICENSE_PENDING.md](LICENSE_PENDING.md).
