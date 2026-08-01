# ACE-2 Beta Engineering Log

This is the detailed, public-safe candidate ledger for the numerical repair
work that continues from the ACE-2 Alpha baseline. It records what was built,
which gates ran, the measured result, why a candidate was accepted or rejected,
and what was deliberately not run.

The ledger is evidence-driven, not a raw agent transcript. Private datasets,
model weights, runtime state, machine paths, and internal prompts are excluded.
Candidate source and evidence hashes are retained where useful for identity.

## Fixed acceptance boundary

The following boundary remains unchanged until a candidate passes every
required numerical gate:

| Item | Frozen value |
|---|---|
| Accepted contiguous frontier | Through `layer_0.v_proj` |
| First unsupported operator | `layer_0.rope_q` |
| Non-SRAM area cap | 2.0 mm² |
| Frequency floor | 100 MHz |
| Historical accepted aggregate | 62,199 cells, 0.6108746272 mm², +0.1502 ns at 100 MHz |

A candidate may pass its software reference, standalone RTL, lint, and focused
quality checks without becoming accepted capability. The frontier advances only
after the complete candidate-specific gate sequence passes.

## Candidate progression at a glance

| Candidate | RTL/reference | Focused two-dataset gate | Paired smoke | Decision |
|---|---|---|---|---|
| Projection-shadow staged attention | PASS | FAIL | FAIL | Bounded no-go |
| Tile-max signed Q6.17 delta | PASS | FAIL | Not run by policy | Bounded no-go |
| Tile block-floating score | PASS | PASS | FAIL | Bounded no-go |
| Shared native-accumulator tagged attention | Not started | Pending | Pending | Environment certified; Manager RTL advance pending |

## 1. Projection-shadow staged attention

**Contract:** `layer0_projection_shadow_staged_attention_v1`  
**Sealed:** 2026-08-01 10:25 UTC  
**Candidate identity:** `a6bf3259765c3b18...`

### Mechanism

The candidate introduced projection-shadow handling and staged attention while
materializing the score path in fixed Q6.9. The objective was to preserve the
existing accelerator shell while repairing the layer-0 RoPE-to-attention
numerical chain.

### Work completed

- architecture contract and bounded implementation authorization;
- independent software reference;
- generated deterministic vectors;
- standalone RTL implementation and simulation;
- full-model fixed-point self-test;
- warning-free projection and score lint;
- two-dataset focused capture;
- one frozen 128-token paired smoke.

The scalar reference, tensor path, generated vectors, and standalone RTL agreed.
This established implementation fidelity, but not model quality.

### Focused result

| Metric | WikiText-2 baseline | WikiText-2 candidate | C4 baseline | C4 candidate |
|---|---:|---:|---:|---:|
| Score relative L2 | 0.257266 | 0.968717 | 0.268586 | 0.983148 |
| Centered-score zero fraction | 1.73% | **90.41%** | 2.18% | **91.80%** |
| Attention-value relative L2 | 0.633926 | 0.788201 | 0.614382 | 0.755054 |

The Q6.9 materialization collapsed score diversity on both datasets. More than
90% of centered scores became zero, and score plus attention-value error
regressed.

### Paired-smoke result

| Dataset | Required ratio | Candidate ratio | Result |
|---|---:|---:|---|
| WikiText-2 | `< 13549.939050` | 13685.620430 | FAIL |
| C4-en-512 | `< 4477.990517` | 4622.743280 | FAIL |

### Decision

**Bounded no-go.** The candidate passed local implementation checks but failed
both the focused discriminator and paired smoke. The accepted frontier did not
move. Full-shell regression, candidate PPA, official validation, prototype, and
signoff were not run.

### Engineering lesson

A single fixed Q6.9 score domain could not preserve the per-channel projection
range. The next candidate had to center scores before narrowing.

## 2. Tile-max signed Q6.17 delta

**Contract:** `layer0_tile_max_delta_attention_v1`  
**Sealed:** 2026-08-01 11:25 UTC  
**Candidate identity:** `1917c460199d219a...`

### Mechanism

The candidate kept the score numerator wide, divided keys into 64-entry tiles,
computed the exact maximum inside each tile, subtracted that maximum before
narrowing, stored a signed Q6.17 local delta, and merged tiles through a
hierarchical softmax path.

### Work completed

- architecture and memory/control contract;
- independent architecture review and hash-only environment qualification;
- scalar and tensor software references;
- generated RTL vectors;
- bounded score/attention RTL implementation;
- standalone RTL simulation;
- warning-free score and softmax lint;
- full-model self-test;
- two-dataset focused discriminator.

### Focused result

| Metric | WikiText-2 baseline | WikiText-2 candidate | C4 baseline | C4 candidate |
|---|---:|---:|---:|---:|
| Centered-score zero fraction | — | **1.55%** | — | **2.13%** |
| Score relative L2 | 0.257266 | **0.494172** | 0.268586 | **0.471271** |
| Softmax relative L2 | 0.574513 | 0.190303 | 0.676291 | 0.206284 |
| Attention-value relative L2 | 0.633926 | 0.336848 | 0.614382 | 0.346583 |

The architecture removed the greater-than-90% zero collapse and substantially
improved softmax and attention-value error. However, far-negative centered
scores clipped in the fixed Q6.17 representation, so score relative L2
regressed on both datasets.

### Paired-smoke result

**Not run.** The frozen policy required strict score-relative-L2 improvement in
the focused discriminator before paired smoke. Because that condition failed
on both datasets, running paired smoke, full-shell regression, or PPA was
forbidden.

### Decision

**Bounded no-go.** The candidate solved the zero-collapse symptom but not the
score-range representation problem. The accepted frontier remained through
`layer_0.v_proj`.

### Engineering lesson

Centering before narrowing was correct, but a fixed fractional format was still
too narrow for the complete negative score range. The successor needed a
per-tile dynamic exponent rather than a fixed Q format.

## 3. Tile block-floating score

**Contract:** `layer0_tile_bfp_score_attention_v1`  
**Sealed:** 2026-08-01 12:06 UTC  
**Candidate identity:** `66e26c60e789ffb8...`

### Mechanism

The candidate retained wide tile extrema and replaced the fixed Q6.17 delta
with a signed 24-bit block-floating score representation. Each tile carried the
dynamic scale needed to preserve centered-score range before hierarchical
softmax composition.

### Work completed

- architecture and quantified compute/memory/control model;
- two independent architecture review rounds;
- hash-only environment qualification without repeating raw probes;
- independent scalar/tensor reference;
- generated RTL vectors;
- bounded RTL implementation and standalone Icarus simulation;
- full-model self-test and warning-free lint;
- two-dataset focused discriminator;
- one frozen paired smoke.

### Focused result

| Metric | WikiText-2 baseline | WikiText-2 candidate | C4 baseline | C4 candidate |
|---|---:|---:|---:|---:|
| Centered-score zero fraction | — | **1.55%** | — | **2.13%** |
| Score relative L2 | 0.257266 | **0.061939** | 0.268586 | **0.066248** |
| Attention-value relative L2 | 0.633926 | **0.336848** | 0.614382 | **0.346583** |

The block-floating representation preserved the centered-score range and
strictly improved score and attention-value relative L2 on both datasets. The
focused discriminator passed.

### Paired-smoke result

| Dataset | Required ratio | Candidate ratio | Margin above threshold | Result |
|---|---:|---:|---:|---|
| WikiText-2 | `< 13549.939050` | 13685.620430 | +135.681380 | FAIL |
| C4-en-512 | `< 4477.990517` | 4622.743280 | +144.752762 | FAIL |

Despite the large focused improvement, the final frozen model-quality ratios
were still slightly worse than the immediate predecessor thresholds.

### Decision

**Bounded no-go.** The candidate was the first of this sequence to pass the
focused gate, but it failed both mandatory paired-smoke thresholds. The accepted
frontier did not move. No candidate PPA, full-shell regression, official
validation, prototype, or signoff followed.

### Engineering lesson

The remaining error was no longer dominated by centered-score storage. The next
architecture had to remove intermediate Q/K requantization more fundamentally
instead of retuning tile-BFP parameters.

## 4. Shared native-accumulator tagged attention

**Contract:** `shared_native_accumulator_tagged_attention_v1`  
**Status at log update:** environment independently certified; Manager RTL
advance pending  
**Proposal identity:** `b654898590d6d213...`

### Mechanism

This successor removes the intermediate requantized Q/K score contract. It
keeps native accumulator values with explicit integer-domain tags and uses one
shared attention engine across all 24 transformer layers.

The frozen architecture defines:

- exact integer formats and domain tags;
- command, result, error, and reset behavior;
- all-layer scheduling and shared-engine reuse;
- cycle, bandwidth, SRAM, and utilization estimates;
- 612 mux-bit positions and 736 state-bit sharing proxies.

### Work completed

- structurally distinct architecture contract;
- standing implementation authorization consumed;
- architecture binder and contract checks;
- independent architecture stage-closing review;
- hash-only environment compatibility review using the existing 27 raw
  environment artifacts without rerunning probes.

### Work not yet completed

- candidate RTL;
- independent reference and generated vectors;
- standalone RTL simulation and lint;
- focused two-dataset discriminator;
- paired smoke;
- candidate PPA or full-shell regression.

### Current decision

**In progress, not accepted capability.** The independent environment review is
complete. Only the Manager may advance the stage to RTL. The historical
frontier and all Alpha claims remain unchanged until the bounded implementation
passes its numerical gates.

## Decision policy used by every candidate

```text
architecture review
  -> environment compatibility
  -> bounded software reference and RTL
  -> generated-vector simulation and lint
  -> two-dataset focused discriminator
  -> one paired smoke, only when the focused gate permits it
  -> full-shell/PPA work only after numerical acceptance
```

Rejected candidates remain visible as negative engineering evidence. They are
not silently rewritten, retuned after authorization is consumed, or counted as
accepted RTL capability.
