# ACE-2: selected engineering highlights

This September 6, 2026 presentation highlights **bounded engineering
milestones**, not a comprehensive diagnostic report or model-quality
assessment. It is a documentation/evidence-summary update: existing public
source is unchanged, and the newer local implementation is not shipped as
a self-contained runnable release.

**September 7 follow-up:** [Four-token full-chain independent numerical
agreement](ACE2_FOUR_TOKEN_INDEPENDENT_ORACLE_20260907.md) covers a separately
checked exact trace. The historical V74 claims below retain their original scope.

## Bounded RTL host, operator and persistent-KV coverage

The original V74 `reports/ace2-stage1-chat-attempt-0002` recorded:

| Completed coverage | Bound |
| --- | --- |
| Prompt handling | 34 tokens from a UTF-8 prompt |
| Causal execution | 35 positions across 24 layers |
| RTL layer invocations | 840 |
| Full-vocabulary head selections | Two generated-token selections |
| Host/KV integration | Persistent K/V and first-token feedback into the next position |
| Recorded integer-boundary comparisons | Zero local mismatches |

This is **actual historical local RTL simulation**, not a new replay,
FPGA/board execution or hardware latency result. It establishes the listed
execution/coverage boundaries, **not useful dialogue quality or general
chatbot certification**. No selected text excerpt is presented as a
complete answer.

Local integer expectations were derived before their corresponding
comparisons. RTL-observed K/V was then fed into host state before downstream
positions; corrected layer-23 V also feeds its local position. Accordingly,
this is **not an RTL-independent whole-model reference trajectory**.
Historical comparison-only oracle wording must be read with that
qualification; original receipts and statuses were not rewritten.

The execution's numerical path uses per-row W4/A8 main projections,
static-first-token QKV, Q9_ONE RoPE, int8-factor rank1 correction, a floating
scale-selection shadow and zero/omitted parent QKV biases. It is not an
all-int4 or pure-quantization ablation, nor the newer dynamic-head Scale32
`QUALITY_CONFIG` path.

## Evidence maintenance plus a fresh shell smoke

Regression `rtl-chat-independent-oracle-regression-0006` audited the
historical V74 seal and retained layer/KV/head records, then completed a
**fresh Icarus shell projection-opcode-01 smoke**. The audit covered
43,056 sealed members, 840 retained layer records and 1,680 K/V files.

The distinction matters: the fresh execution was a bounded shell smoke,
**not another 840-layer replay**, fresh full-vocabulary-logit replay or new
full-model answer trajectory. These results do not expand the public
baseline's source-specific certification.

## One offline S16 fixture: 99.30% lower ordinary merge error

A separate held-boundary software experiment compared an A8 residual merge
with a **new offline signed16 residual-merge profile**, holding W4/A8 inputs,
signed32 accumulators and the incoming residual fixed.

| Local measurement | Same-fixture A8 merge | Offline S16 merge |
| --- | ---: | ---: |
| Ordinary channels | 887 | 887 |
| Merge mean absolute error | 0.30670038 | 0.00215219 |

This is a **99.30% reduction in ordinary-channel merge MAE on one fixture**,
measured against the same held real-valued merge reference. The fixture is
position 0, layer 2 of a bias-restored, score-centered software control,
before a separate SiLU-format contrast; it is not the unchanged production
baseline over arbitrary inputs.

The profile aligns the existing accumulators and residual to one offline
S16 scale, uses signed rounding/checked products and a signed16 merge, and
defines S16 RMSNorm input with S8 normalized output. **The highlighted
metric concerns the merge only**, not normalized-output fidelity,
end-to-end quality or W4A8 chat improvement. Runtime Scale32 metadata,
integrated packing/descriptor semantics and RTL support are unimplemented;
offline packing checks are not hardware implementation or adoption.

## Publication and evidence boundaries

[Machine-readable highlights](../../evidence/public/ace2-progress-20260906/highlights.json)
retain selected original artifact identities and metric definitions.
Original-byte hashes are not hashes of these derived summaries.
[Checksums](../../evidence/public/ace2-progress-20260906/SHA256SUMS) cover
the listed current public presentation files only.

This selected presentation does not assert that unreported evaluations
passed or that engineering issues are absent. Broader material published
earlier remains in public Git history; the current scope change is a normal
follow-up commit, not a history rewrite. Original reports remain unchanged.

Current scope is local simulation/software research. Hardware Stage 2,
FPGA, synthesis, PPA and U280 work is cancelled. See the
[publication boundary](PUBLICATION_BOUNDARY.md) for source/reproduction limits.
