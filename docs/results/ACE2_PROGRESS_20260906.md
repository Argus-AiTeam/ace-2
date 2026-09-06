# ACE-2 progress: bounded RTL evidence and unresolved numerical quality

This September 6, 2026 update publishes **sanitized evidence summaries only**.
It does not change the existing public source baseline, ship the newer local
implementation, or provide a self-contained runnable release. Current work
is local simulation/software diagnosis; hardware Stage 2, FPGA, synthesis,
PPA and U280 work is cancelled. Older public hardware results remain
historical results for their own bound source, not claims for these runs.

**Bottom line:** bounded RTL numerical agreement is real but is not useful
chat quality. The current exact LoRA176 Stage-1 software path failed all six
fixed short examples; three incremental diagnostic contrasts did not repair
the representative answer. A wider residual profile improves one local error
metric but remains unimplemented in the runtime/RTL and unvalidated for chat.

The [machine-readable provenance](../../evidence/public/ace2-progress-20260906/provenance.json)
records original artifact names/statuses, typed model/checkpoint/source
identities, original-byte SHA256 hashes, and the selected results below.
[Publication boundaries](PUBLICATION_BOUNDARY.md) explain what is not shipped.

## 1. V74: original two-token RTL evidence, not a chatbot-quality result

Original `reports/ace2-stage1-chat-attempt-0002` recorded `PASS` for
`Grüße,世界`: 34 prompt tokens, 35 causal execution positions, **840 layer
invocations**, and tokens `[15663, 25055]`, decoded exactly as ` aside crystal`.
Recorded integer-boundary mismatches were zero, with persistent K/V and
token-0 feedback into the next position. The text is syntactically decodable,
**not a useful answer**.

**Oracle erratum:** local expected integer boundaries were derived before
their corresponding comparisons, but RTL-observed K/V replaced host cache
rows before downstream positions. Corrected layer-23 V also feeds the
current position after its local comparison. Historical
`host_reference_consumed_rtl_as_input=false` / comparison-only wording is
therefore not valid for the whole causal chain. Neither the original JSON
nor its seal/status was rewritten. No end-to-end independent-cache
certification is claimed.

The original V74 procedure depended on the preserved aborted attempt 0001,
its exact partial frontier/source manifest, a fresh fixed attempt 0002, and
at least **64,800 seconds** of timeout. A fresh V73 command is not equivalent
V74 reproduction. This update provides neither a replay command guaranteed
to work from public checkout nor permission to overwrite immutable attempts.

## 2. Regression 0006: historical evidence audit plus a fresh shell smoke

`reports/rtl-chat-independent-oracle-regression-0006/result.json` recorded
`PASS`: an audit of the historical 43,056-member V74 seal, 840 retained layer
records, 1,680 K/V files, both head records and tokenizer decode, plus a
**fresh Icarus shell projection-opcode-01 smoke**.

It did **not** rerun 840 layers, regenerate the six later answers, or freshly
replay every full-vocabulary logit. Transient V74 vectors/binaries had been
pruned. Its historical “independent oracle” wording is subject to the erratum
above. A shell smoke is not a new full-model chat run, synthesis/PPA result,
or hardware deployment.

## 3. Current LoRA176 software comparison: six fixed failures

The completed `current-lora176-software-quality-20260906-0950` diagnostic
compared the same BF16 parent and LoRA checkpoint-176: **BF16 6/6 versus
current Stage-1 W4A8-plus-rank1 0/6**. Separate parent semantic reading of all
twelve prompt/answer pairs agreed with the deterministic screens. These are
six frozen examples, **not general benchmark accuracy**. The older August
G32 0/6 involved a different model/configuration.

See [all twelve exact answers and limits](CURRENT_LORA176_SOFTWARE_QUALITY_20260906.md).
Both arms had identical per-case input IDs, greedy/EOS policy and caps
`[10, 7, 10, 6, 7, 12]` under the unchanged 43-token context bound.
Every BF16 answer reached EOS. Every Stage-1 answer was already irrelevant
or noncompliant before context truncation; no unobserved continuation is
scored.

These are pure-software diagnostic continuations, not six new RTL replies.
The sampler uses its own software-computed K/V, with independent per-case
copies of a software prefix. **1,152** matches to prior RTL K/V hashes cover
only **24 shared-prefix positions x 24 layers x K/V**, not answer trajectories.
The sampler bypasses the public four-token wrapper and post-argmax guards:
the factual answer would abort the public product at generated index 1 on a
nonbreaking-space-only token.

This path has per-row W4/A8 main projections, static-first-token QKV,
Q9_ONE RoPE, **int8-factor rank1**, floating scale-selection shadow and
zero/omitted parent QKV biases. It is not a pure quantization-only ablation,
AWQ W4A16 validation, or the newer dynamic-head Scale32 `QUALITY_CONFIG`.

## 4. 1021: first divergence and three negative incremental contrasts

The completed `current-lora176-software-diagnosis-20260906-1021` report is
`COMPLETED_DIAGNOSIS_NO_VALIDATED_QUALITY_REPAIR`. For `7+5=? Answer only.`,
the first boundary exceeding its descriptive threshold was position 0,
layer 0 Q projection: relative L2 **1.0241**, RMSE **7.8473**, versus parent
Q-bias RMS **7.8541**. Omitted biases explain a major shadow mismatch.
This is not proof Q alone causes the answer failure: with one causal key,
Q/K do not change softmax; V/residual/normalization also matter.

Head controls on the same Stage-1 hidden state still chose token 67426
(`symbols`), with no saturated logits and one top selection; controls on
BF16 hidden state chose correct first token 16 (`1`). This implicates earlier
hidden degradation rather than merely final-head clipping or tie handling.
Float-head controls were diagnostics, not accepted W4A8/RTL responses.

| Incremental contrast | Control and one added change | Exact diagnostic answer (JSON string) | Result |
| --- | --- | --- | --- |
| 1 | Current Stage-1 + restored parent QKV biases | `" in \n 7. 7"` | Failed |
| 2 | Contrast 1 + wide score centering before signed16 clipping | `"'?'?'?'\n'?'A"` | Failed |
| 3 | Contrast 2 + expanded SiLU range/Q6.9 output and compensated multiplier | `" \\/'0,000"` | Failed |

All three used the same seven-token arithmetic cap and were truncated.
These are **incremental contrasts**, not three independent one-change tests
against the original baseline. None met the frozen exact-12-and-EOS
criterion; no six-case candidate confirmation or validated quality repair
followed, and no candidate was adopted.

## 5. 1100: exact local requantization and residual loss

`current-lora176-layer2-requantization-20260906-1100` examines **position 0,
layer 2 of the bias-restored plus centered-score candidate, before the
SiLU contrast**. It is not a statement about all original-baseline inputs.
Holding W4 inputs/weights/accumulators and the incoming residual fixed:

- **453/896** outputs were already zero in the signed32 accumulator;
  requantization newly erased **424** nonzero outputs, yielding **877** A8
  down-output zeros.
- **848** nonzero incoming residual contributions were lost during alignment
  to the coarse final grid. Same-grid arithmetic fusion did not recover them.
- All 424 new zeros also followed exact ideal-grid rounding, with zero
  down-output saturation; the evidence does not attribute them to an
  erroneous multiplier or saturation.

One tighter scalar-scale policy reduced ordinary-channel merge MAE by only
**9.727%**, below its frozen 25% criterion, and worsened outlier normalization
RMSE. It was rejected; the status remains
`COMPLETE_EXACT_BOUNDARY_ATTRIBUTION_NO_REPAIR_PROMOTED`.

## 6. 1128: wider residual feasibility, not adopted architecture or quality

`current-lora176-layer2-wide-residual-20260906-1128` tests a **new S16 residual
merge / S16 RMSNorm input / S8 normalized output** profile at that same held
boundary. Across the same **887 ordinary channels**, merge MAE fell from
**0.30670038 to 0.00215219**, a **99.30% local reduction**. The original
`PASS_LOCAL_COMPONENT_FEASIBILITY` is limited to this component/profile.

Important failures/limits remain: **839/887 ordinary normalized outputs are
still zero**, there is **one S8 normalization saturation**, and the 453
upstream zero accumulators cannot be reconstructed. Held-case S16 merge
saturation was zero; this does not remove normalized-output loss.

Scale selection was offline. Runtime Scale32 encoding, reduction/descriptor
semantics, integrated packing and RTL are **unimplemented**; offline
signed16 packing/range checks do not establish runtime support. This is not
the current strict-A8 path, a full-model quality repair, or production
adoption. No model layers or generation were executed for this held-array
component experiment.

## 7. 1200: normalized S8 grid attribution, no next candidate

The parent-reviewed completed
`current-lora176-normalized-a8-loss-20260906-1200` report records
`COMPLETED_FIXED_GRID_ATTRIBUTION`, with findings status
`COMPLETED_FIXED_GRID_ATTRIBUTION_NO_CANDIDATE`. It uses the same held
position-0/layer-2 corrected control and offline S16 profile described above,
not original-baseline inputs generally or the current strict-A8 product.

**All 896 outputs exactly match** original-raw-gain BF16/epsilon normalization
of the **same S16 input**, followed by the unchanged S8 grid. The **839
ordinary zeros** partition into **28 preexisting real-reference zeros + 3
S16-merge-induced zeros + 808 nonzero targets erased by the fixed grid + 0
additional zeros from combined integer RMS/gain approximation**.
Replacing the whole approximate normalization with this ideal same-input
reference recovers no additional outputs on that grid. This does not mean
each approximation has zero numerical error; individual arithmetic changes
were not separately swept. One outlier clips in both ideal and actual output.

For **this held normalized target only**, avoiding positive post-rounding
clipping requires scale **S > 39/170 (0.22941176...)**, while preserving
**every** nonzero requires **S < 183/131072 (0.00139618...)**. These
single-S8-scale conditions are incompatible. This is **not a proof W4A8 is
impossible**, nor proof that every erased channel is semantically necessary
or that preserving all of them would repair dialogue.

There was no next candidate, alternative scale/calibration sweep, generation,
runtime/RTL integration or quality promotion. The local path remains
unchanged. Original report SHA256:
`fb682bf784ad0c6b7a53d860ae5d3615cd44412abfbfadf8fc5a101286ac351c`;
the complete original-file identities are in the linked public provenance.

## Evidence and release limits

The underlying completed reports retain their original names, hashes and
statuses, including negative results and historically overbroad oracle
wording with the explicit erratum. Source/model hashes do not imply the
execution source is included in this public baseline.

No weights, tensor values, raw logs, private filesystem locations, broad
private source snapshot, or private Git ancestry is exported. Original
archives remain external; checksums of this derived package do not validate
archive availability or replay completeness. The final normalized-A8-loss
diagnosis is included following parent-review confirmation; no later
experiment or unreviewed candidate is implied.
