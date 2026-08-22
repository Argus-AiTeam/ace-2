# W4A8 Carrier Diagnostic Findings

Status: Stage 1 software diagnosis in progress

This note publishes the smallest useful summary of four sealed, read-only
counterfactual experiments on the Qwen2.5-0.5B W4A8 software path. It does not
claim that readable multi-token W4A8 generation, RTL agreement, or Stage 1
certification is complete.

The machine-readable values and source report hashes are recorded in
`evidence/verification/w4a8-diagnosis-20260822/SUMMARY.json`. Large tensor
captures and run-local helpers are intentionally excluded from Git.

## Confirmed findings

### Per-channel RMSNorm is not the primary defect

Replacing the relevant RMSNorm behavior reduced the layer-2 up-projection
component-A NRMSE by 18.77% on the common Scale32 grid and 16.48% with the
physical per-channel float oracle. Both are below the frozen 50% gate.

Classification: `RMSNORM_NOT_PRIMARY`

### The layer-2 incoming residual is the dominant carrier

Replacing only the incoming residual with its BF16 oracle reduced downstream
component-A NRMSE by 52.60%. Replacing only the attention branch made the
metric 15.08% worse. Replacing both branches under the production merge reduced
NRMSE by 81.56%; the physical sum-then-single-quantize upper bound reduced it
by 83.42%.

Classification: `UPSTREAM_RESIDUAL_PRIMARY`

### A local source-domain merge is insufficient

A bit-exact source-domain Scale32 fusion at the layer-2 attention residual join
improved downstream component-A NRMSE by only 0.19%. Source-code round trips,
the independent reference, and the runtime attention fusion all matched
bit-exactly, with zero destination saturation.

Classification: `SOURCE_DOMAIN_FUSION_INSUFFICIENT`

This rules out a local residual-join rewrite as the primary repair.

### The defect is mixed inside layer 1

The corrected layer-1 experiment injected every counterfactual carrier at the
real layer-2 input and executed a distinct layer-2 attention, residual join,
RMSNorm, gate projection, and up projection for each lane. It did not reuse a
baseline attention output.

- BF16 layer-1 post-attention residual with W4 MLP-down: 29.90% reduction.
- W4 layer-1 post-attention residual with BF16 MLP-down: 19.03% reduction.
- Both BF16 branches under the production merge: 50.77% reduction.
- Physical sum-then-single-quantize: 51.38% reduction.
- Destination saturation: 0% in every lane.

Classification: `MIXED_LAYER1_DEFECT`

The smallest current repair surface is therefore the pair of layer-1
post-attention and MLP-down carrier interfaces. The individual contributions
must not be treated as additive.

## Engineering implications

1. Do not spend further runs searching RMSNorm scales or changing only the
   layer-2 residual merge.
2. Decompose the layer-1 post-attention carrier into incoming residual,
   attention/o-projection, and merge quantization.
3. Decompose layer-1 MLP-down into input carrier, W4 weight/accumulator, and
   output conversion/clamp.
4. Propagate every controlled candidate through the real downstream layer-2
   path before accepting a repair.
5. Add production code only after a hardware-plausible repair crosses the
   frozen 50% gate and preserves focused regressions.

## Non-claims

- No production W4A8 repair is published by this note.
- BF16 remains an oracle, not a fallback.
- Stage 1 is not complete.
- RTL-backed chat and Stage 2 U280 deployment are not authorized by these
  results.
