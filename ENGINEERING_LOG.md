# ACE-2 Engineering Log

This is the end-to-end, public-safe engineering log for **Argus Compute Engine 2
(ACE-2)**. It starts with the original mission definition, follows the
construction of the Alpha engineering base, records the discovery of the first
end-to-end numerical failure, and continues through the current Beta repair
candidate.

The ordering below is reconstructed from retained contracts, source,
verification artifacts, and reviewed decisions. It is intentionally more
detailed than a release overview, but it is not a raw private transcript.

## Step 01 — Define the mission

**Work performed**

- selected Qwen2.5-0.5B as the model profile;
- selected W4A8 as the principal arithmetic format;
- defined a research accelerator rather than a software-only model port;
- separated structural implementation, local numerical correctness, composed
  model quality, PPA, and release claims.

**Outcome**

The project began with an explicit rule: RTL source existing in a repository
would not, by itself, count as usable inference support.

## Step 02 — Freeze engineering constraints

**Work performed**

- established a 2.0 mm² non-SRAM area cap;
- established a 100 MHz frequency floor;
- defined a contiguous support frontier instead of allowing isolated
  downstream blocks to imply full support;
- required independent evidence before advancing accepted capability.

**Outcome**

Every later candidate could be rejected without changing the target or
silently relaxing the quality gate.

## Step 03 — Map the model flow

**Work performed**

- decomposed the model into a 434-item structural execution flow;
- represented layer-0 and repeated-layer operator sequencing;
- identified projection, normalization, RoPE, attention, softmax, activation,
  residual, and final-output responsibilities;
- created traceability between model-flow entries and accelerator structure.

**Outcome**

The design had a complete structural map before numerical acceptance was
claimed.

## Step 04 — Define the accelerator shell

**Work performed**

- specified descriptor-driven command execution;
- defined control, status, completion, and error behavior;
- defined DMA-style activation, weight, metadata, and output movement;
- defined operator dispatch and shared-resource control;
- created the top-level shell and package-level formats.

**Produced**

- `rtl/ace2_pkg.sv`
- `rtl/ace2_shell.sv`
- architecture and traceability documentation

**Outcome**

The project gained a reusable execution framework that did not depend on any
single RoPE repair candidate.

## Step 05 — Implement RMSNorm support

**Work performed**

- implemented fixed-point RMSNorm behavior;
- created an independent software reference;
- defined output scaling compatible with the projection consumer;
- included RMSNorm output in a projection test workload.

**Produced**

- `rtl/ace2_rmsnorm_core.sv`
- `tools/ace2_rmsnorm_reference.py`

**Outcome**

RMSNorm-to-projection scale compatibility became directly testable rather than
an undocumented assumption.

## Step 06 — Implement W4A8 projection

**Work performed**

- implemented signed int8 activation by signed int4 weight accumulation;
- implemented per-output multiplier, right shift, zero point, rounding, and
  saturation behavior;
- supported hidden-width, MLP gate, and MLP down-projection shapes;
- created an independent scalar/tensor reference.

**Produced**

- `rtl/ace2_w4a8_proj_core.sv`
- `tools/ace2_projection_reference.py`

**Outcome**

Projection became the strongest reproducible accepted-prefix block and later
formed the public Alpha demo.

## Step 07 — Implement downstream operator infrastructure

**Work performed**

- implemented and integrated RoPE research paths;
- implemented attention-score arithmetic;
- implemented softmax;
- implemented attention-value composition;
- implemented SiLU gating and its generated lookup table;
- connected the blocks to the structural shell.

**Produced**

- `rtl/ace2_rope_core.sv`
- `rtl/ace2_dynamic_rope_head_core.sv`
- `rtl/ace2_fixed_q7_rope_score_core.sv`
- `rtl/ace2_relative_rope_score_fusion_core.sv`
- `rtl/ace2_attention_score_core.sv`
- `rtl/ace2_softmax_core.sv`
- `rtl/ace2_attention_compose_core.sv`
- `rtl/ace2_silu_gate_core.sv`

**Outcome**

The downstream RTL library existed and could be linted or tested locally. It
was not yet evidence that the composed model was numerically correct.

## Step 08 — Build deterministic vector generation

**Work performed**

- created deterministic synthetic activation and weight patterns;
- covered balanced signed arithmetic and intentional saturation;
- covered 896-wide projection, 4,864-wide MLP reduction, and RMSNorm consumer
  scaling;
- generated both machine-readable JSON and SystemVerilog include files;
- attached SHA-256 identities to inputs, weights, and outputs.

**Produced**

- `tools/gen_projection_vectors.py`
- `verification/generated/projection_vectors.json`
- `verification/generated/projection_vectors.svh`

**Outcome**

The same input could be regenerated independently and consumed by both software
and RTL.

## Step 09 — Build directed RTL simulation

**Work performed**

- created the projection RTL testbench;
- checked selected outputs across five oracle workloads;
- added directed rounding-tie, shift-zero, shift-63, positive, and negative
  arithmetic cases;
- failed on mismatch, timeout, missing saturation, or accumulator error.

**Produced**

- `verification/tb/ace2_w4a8_proj_tb.sv`

**Measured public-demo scope**

- 5 oracle workloads;
- 8,480 declared row/output positions in the vector package;
- 21 selected RTL outputs;
- 6 directed arithmetic cases.

**Outcome**

The accepted projection prefix had an independent, deterministic RTL
discriminator.

## Step 10 — Add structural lint and synthesis checks

**Work performed**

- linted the complete structural shell with Verilator;
- simulated the accepted projection prefix with Icarus Verilog;
- added technology-independent Yosys synthesis for the projection core;
- kept structural lint separate from numerical acceptance.

**Outcome**

A shell parsing successfully could no longer be confused with the full model
producing correct values.

## Step 11 — Establish historical PPA context

**Work performed**

- preserved the accepted aggregate implementation context;
- recorded cell count, non-SRAM area, and timing at the frequency floor;
- separated historical PPA context from candidate PPA.

**Preserved aggregate**

- 62,199 cells;
- 0.6108746272 mm² non-SRAM area;
- +0.1502 ns setup slack at 100 MHz.

**Outcome**

Rejected numerical candidates could not inherit an accepted PPA claim.

## Step 12 — Integrate a full-model fixed-point discriminator

**Work performed**

- connected local fixed-point operators into a model-level numerical path;
- captured intermediate values at projection, score, softmax, and
  attention-value boundaries;
- compared candidate and baseline behavior on frozen WikiText-2 and C4 inputs;
- added paired model-quality smoke thresholds.

**Outcome**

The verification scope expanded from “does this RTL match its local oracle?” to
“does the composed model remain numerically usable?”

## Step 13 — Discover the first authoritative divergence

**Finding**

Full-model evidence showed that the first material unsupported operator was
`layer_0.rope_q`.

**Action**

- stopped claiming complete numerical support for the 434-item flow;
- preserved the downstream source and verification infrastructure;
- rolled the accepted contiguous frontier back to:

```text
layer_0.input_rmsnorm
layer_0.q_proj
layer_0.k_proj
layer_0.v_proj
```

**Outcome**

The project retained substantial implementation work while making the support
claim narrower and more accurate.

## Step 14 — Investigate early RoPE/attention repair paths

**Work performed**

- evaluated dynamic head scaling;
- evaluated fixed-Q7 score handling;
- evaluated relative-RoPE score fusion;
- retained local RTL where useful for research and shell elaboration;
- rejected paths that failed bounded numerical or review gates.

**Outcome**

These investigations established that changing only a local RoPE arithmetic
expression was insufficient; projection scale and attention score range had to
be considered together.

## Step 15 — Build projection-shadow staged attention

**Contract**

`layer0_projection_shadow_staged_attention_v1`

**Work performed**

- froze and reviewed a projection-shadow architecture;
- implemented its software reference and RTL;
- generated vectors and passed standalone simulation;
- passed full-model self-test and warning-free lint;
- ran two-dataset focused capture and paired smoke.

**Measured result**

- centered-score zero fraction: 90.41% on WikiText-2;
- centered-score zero fraction: 91.80% on C4;
- score relative L2 regressed to 0.968717 and 0.983148;
- paired ratios 13685.620430 and 4622.743280 both failed their thresholds.

**Decision**

Bounded no-go. Q6.9 score materialization collapsed score diversity.

## Step 16 — Build tile-max signed Q6.17 delta

**Contract**

`layer0_tile_max_delta_attention_v1`

**Work performed**

- retained wide score numerators;
- computed an exact maximum inside each 64-key tile;
- subtracted the tile maximum before narrowing;
- stored signed Q6.17 local deltas;
- implemented hierarchical softmax composition;
- completed architecture/environment review, software reference, RTL,
  generated vectors, simulation, lint, and a two-dataset focused gate.

**Measured result**

- centered-score zero fraction fell to 1.55% and 2.13%;
- softmax and attention-value relative L2 improved strongly;
- score relative L2 nevertheless regressed from 0.257266 to 0.494172 on
  WikiText-2 and from 0.268586 to 0.471271 on C4.

**Decision**

Bounded no-go. Fixed Q6.17 clipped far-negative centered scores. Paired smoke
was correctly not run because the focused score gate failed.

## Step 17 — Build tile block-floating score

**Contract**

`layer0_tile_bfp_score_attention_v1`

**Work performed**

- replaced fixed Q6.17 storage with a signed 24-bit block-floating score;
- preserved per-tile dynamic range;
- completed two architecture review rounds and hash-only environment review;
- implemented independent references, vectors, RTL, simulation, and lint;
- ran the focused two-dataset discriminator;
- ran the one authorized paired smoke after the focused gate passed.

**Measured focused result**

- zero fractions remained low at 1.55% and 2.13%;
- score relative L2 improved to 0.061939 and 0.066248;
- attention-value relative L2 improved to 0.336848 and 0.346583.

**Measured paired-smoke result**

- WikiText-2: 13685.620430, required `< 13549.939050`;
- C4: 4622.743280, required `< 4477.990517`.

**Decision**

Bounded no-go. This candidate solved the dominant score-storage error and
passed the focused gate, but still missed both final model-quality thresholds.

## Step 18 — Select shared native-accumulator tagged attention

**Contract**

`shared_native_accumulator_tagged_attention_v1`

**Work performed**

- stopped retuning tile-BFP after its authorization was consumed;
- selected a structurally distinct all-layer mechanism;
- removed intermediate requantized Q/K score storage from the contract;
- defined native accumulator values with explicit integer-domain tags;
- defined one shared engine across all 24 transformer layers;
- quantified cycles, bandwidth, SRAM, utilization, 612 mux-bit positions, and
  736 state bits;
- passed independent architecture review;
- completed hash-only environment review without repeating the 27 raw probes.

**Current state**

Environment review is complete and the Manager has advanced to RTL. Two
synthesizable standalone candidate modules now implement tagged native-record
RoPE handling and bounded 64-lane tagged Q20.44 score accumulation. Generated
JSON/SystemVerilog vectors, independent Python tests, Icarus simulation,
candidate-top Verilator lint, and elaboration pass.

The standalone focused RTL candidate is undergoing independent RTL checklist
review. The two-dataset quality discriminator, paired smoke, shell admission,
candidate PPA, and full-shell regression have not run.

## Step 19 — Prepare the experimental Alpha package

**Work performed**

- assembled files through a strict whitelist;
- excluded weights, datasets, private benchmarks, credentials, runtime state,
  internal logs, proprietary PDK/IP, and machine configuration;
- documented the 434-item structural flow and the narrower numerical frontier;
- added Argus attribution and the official `https://argusbot.cn/` link;
- left licensing explicitly pending rather than borrowing the Argus software
  license.

**Outcome**

The package became suitable for private release review without exposing
restricted project material.

## Step 20 — Build the reproducible Alpha demo

**Work performed**

- added `make demo` for tool checks, full-shell lint, oracle regeneration,
  byte comparison, RTL simulation, and evidence reporting;
- generated a pure-English HTML evidence dashboard;
- generated a Yosys-derived synthesized-netlist SVG;
- generated a standard VCD waveform compatible with GTKWave and Vivado;
- generated a SHA-256 release manifest.

**Outcome**

The Alpha no longer ends with an unexplained PASS marker. A reviewer can see
what ran, how much data was exercised, which artifacts prove it, and where the
support boundary stops.

## Step 21 — Create the private release repository

**Work performed**

- isolated GitHub authentication under the project directory;
- created `aHappend/ace-2-alpha`;
- kept the repository private;
- pushed the reproducible Alpha package and subsequent documentation updates;
- preserved the rule that no public release occurs without explicit operator
  instruction and a selected project license.

## Current project position

| Area | Current position |
|---|---|
| Structural model-flow representation | 434 items |
| Accepted numerical frontier | Through `layer_0.v_proj` |
| First unsupported operator | `layer_0.rope_q` |
| Rejected bounded Beta candidates | Projection-shadow, tile-max Q6.17, tile-BFP |
| Active candidate | Shared native-accumulator tagged attention |
| Active candidate stage | Standalone focused RTL passes; independent RTL review in progress |
| Alpha repository | Private |
| Public release | Not authorized |

For complete per-candidate metric tables and gate decisions, see
[BETA_ENGINEERING_LOG.md](BETA_ENGINEERING_LOG.md).
