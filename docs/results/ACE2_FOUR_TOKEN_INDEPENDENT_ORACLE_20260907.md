# ACE-2: four-token full-chain independent numerical agreement

**September 7, 2026 — accepted bounded local-simulation milestone.**
A host-only reference independently reconstructed an entire four-token
generation and matched the retained computer-local RTL execution, with
**zero integer-byte or selected-token mismatches**.

## What was checked

The exact user input was `Grüße,世界`, expanded to 34 tokens by the official
chat template. The sealed execution processed 37 positions: 34 prefill
positions plus three generated-token feedback positions, producing four
greedy selections.

| Verified coverage | Result |
| --- | ---: |
| Causal positions / layers per position | 37 / 24 |
| Retained RTL layer records | 888 |
| Retained K/V append files | 1,776 |
| Full-domain output-head steps | 4 × 151,936 output rows |
| Integer-byte / selected-token mismatches | 0 / 0 |

The independently propagated reference covered persistent K/V, quantized
hidden states and RMSNorm, full-domain logits, and greedy decisions.
Retained K/V append bytes were compared directly; complete cache prefixes
were reconstructed and checked against their sealed hashes. Pruned
layer-output, RMSNorm and logit payloads were checked by exact byte count
and SHA-256 against sealed execution records, not by claiming those raw
payloads remain available.

## Why the independence result matters

The reference completed the **whole sequence before reading RTL result
values**. It maintained its own K/V caches and token feedback; the layer-23
V correction used a frozen host integer reference rather than RTL-observed
correction values. RTL results were comparison-only inputs after reference
completion.

This closes the causal-state/reference-feedback erratum **for this exact
prompt and sealed four-token trace only**. It does not retroactively
recertify the separate two-token V74 evidence in the
[September 6 highlights](ACE2_PROGRESS_20260906.md).

Independence here means independently propagated state and control flow.
The oracle reuses accepted fixed-point arithmetic primitive definitions and
the same model/adapter inputs: it is **not an implementation-diverse second
specification of every primitive**, nor independent fidelity validation
against BF16 weights.

## Precision and publication scope

The checked profile uses Qwen2.5-0.5B-Instruct with ACE-2 LoRA checkpoint-176,
per-row **W4A8 main projections**, and a frozen **int8-factor rank-1 layer-23
V correction**. It retains static-first-token QKV metadata, Q9_ONE RoPE,
a floating scale-selection shadow and zero/omitted parent QKV biases.
It is not official AWQ W4A16, all-INT4, or the newer dynamic-head Scale32
profile; official-tokenizer provenance does not change that precision.

This is numerical agreement for bounded generation, **not semantic
dialogue-quality certification or broad arbitrary-prompt support**. No
multi-prompt-suite success, hardware, synthesis, PPA, FPGA, board, bitstream
or deployment result is claimed.

The completed host-only comparison reused a sealed, previously compiled
and simulated RTL trace; it launched no new RTL simulation. Publication
itself reran neither simulation nor the host oracle. These are newly
authored aggregate facts, not a new runtime/source release or turnkey
reproduction package. Matching private helpers, dependencies and complete
archives are not bundled.

[Machine-readable summary and original evidence hashes](ACE2_FOUR_TOKEN_INDEPENDENT_ORACLE_20260907.json)
record the exact scope. Original scientific evidence and acceptance
decisions remain unchanged.
