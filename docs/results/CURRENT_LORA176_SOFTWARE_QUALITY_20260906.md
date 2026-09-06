# Current LoRA176: six-case negative software-quality result

On six frozen short prompts, the exact current Stage-1 **W4A8-plus-rank1**
software path passed **0/6**, versus **6/6** for the same-LoRA176 BF16 control.
The deterministic screens and separate parent reading of all twelve pairs
agreed. This is failure on these examples, **not representative accuracy or
a claim that every possible input fails**. The older August G32 result is a
different model/configuration.

## All recorded prompt/answer pairs

Each text cell is a JSON string containing the full recorded decoded text,
not a repaired answer. `\u00a0` exposes nonbreaking whitespace, `\n` a newline,
and `\ufffd` a replacement character. Both arms used the same per-case cap.

| Prompt | Same-LoRA176 BF16 answer | Current Stage-1 W4A8-plus-rank1 answer | Cap |
| --- | --- | --- | --- |
| `"Capital of France?"` | `"The capital of France is Paris."` | `"体制改革\u00a0bsiteptscolm').\"]bool.Comparatoriciesoodles"` | 10 |
| `"7+5=? Answer only."` | `"12"` | `"symbolsallah/by.navigateByUrl TauToDevice\ufffd"` | 7 |
| `"Reply exactly: BLUE"` | `"BLUE"` | `"illionerateachtjamMinMaxynn Bannon出行型号 \ufffd"` | 10 |
| `"Name: Ada. Return name only."` | `"Ada"` | `" URLWithString翰antdendereco túbenh"` | 6 |
| `"French for hello? One word."` | `"Bonjour!"` | `".getElementsByNamebenhcontribiloguePJ.*/\negra"` | 7 |
| `"Hello!"` | `"Hello! How can I help you today?"` | `"为基础mêmeolumes写的(history闭环 nhuliascalarppardfections.isNotBlank"` | 12 |

All BF16 answers reached EOS; all Stage-1 answers were context-truncated
under the same unchanged **43-token context bound**, already irrelevant or
noncompliant before truncation. Unobserved longer continuations are not scored.
The control uses the same parent/checkpoint, tokenizer/template, paired input
IDs and greedy/EOS policy, but this is **not quantization-only causality**:
Stage-1 includes int8-factor rank1, floating scale-selection shadow and omitted
parent QKV biases. It is not the newer dynamic-head Scale32 `QUALITY_CONFIG`
path, pure all-int4 inference or AWQ W4A16 validation.

The sampler bypassed the public four-token wrapper and post-argmax
readability/domain guards. Its factual continuation would abort the real
product at generated index **1** (token 4102, whitespace-only). These are
software diagnostics, not completed public-product replies or six fresh RTL
answers.

Software-computed K/V and independent per-case copies of the software prefix
fed subsequent positions. The **1,152** prior RTL K/V hash matches cover only
24 shared-prefix positions x 24 layers x K/V, not answer trajectories. This
fresh software lineage does not repair the historical V74 whole-chain oracle
claim; see [the erratum and full progress summary](ACE2_PROGRESS_20260906.md).

## Original-byte provenance

Original archive:
`reports/verification/current-lora176-software-quality-20260906-0950/`.
Original `result.json` SHA256:
`db18eb36181c3f421636b455bebcf9362c0058598010ad46f50d149e71f6c472`;
original `REPORT_SHA256SUMS` SHA256:
`5ef3e61f316d9cac24acf3e6b1509e17e6115b9f9fa6e5ee78c780a4e321ac37`.

Separate parent
`reports/verification/current-lora176-software-quality-review-20260906-0950/assessment.json`
SHA256: `52e8d0ea8a038924f519f419e26cc2354c38d7ca033ed592850d53a3ee001806`.
The original diagnostic's pending-parent-review fields were not rewritten:
the separate assessment records completion. It is a semantic reading, not
an independent inference rerun, full source audit or RTL acceptance.

The [machine-readable provenance](../../evidence/public/ace2-progress-20260906/provenance.json)
also binds scope, verification and lineage reports. All original hashes
refer to original external bytes, **not this derived Markdown or sanitized
receipts**. No inference was rerun for publication. This authorized
documentation update remains subject to the [evidence-only release boundary](PUBLICATION_BOUNDARY.md).
