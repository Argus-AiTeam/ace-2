# Official RTL arbitrary-text generation attempt 0002

This compact publication records an immutable Icarus Verilog execution of the
Qwen2.5-0.5B-Instruct ACE-2 W4A8 path. The frozen 34-token prompt produced four
tokens:

```text
[26614, 43895, 92464, 76521]
TI tariffs(ang ArgumentOutOfRangeException
```

The output is printable but semantically incoherent and is not a valid answer
to the prompt. This attempt therefore advances executable RTL coverage but does
not complete Stage 1 or establish model quality.

## What the evidence establishes

- The official process started exactly once and the attempt is sealed.
- 37 token positions crossed all 24 transformer layers: 888 layer executions.
- All 7,992 recorded component mismatch counters are zero.
- Every layer record reports exact W4A8 RTL/integer-reference agreement,
  including KV append and residual/projection boundaries.
- Four full-vocabulary LM-head steps each evaluated 151,936 outputs.
- Every LM-head step has zero integer mismatches and exact selected-logit and
  selected-token agreement.
- The generated tokens were fed back exactly at positions 34, 35, and 36.
- The run recorded 7,112 simulations, 13,703,170,650 simulator cycles, 14,224
  observed child processes, and no missing Icarus children.
- Total wall time was 50,875.10838294102 seconds (14:07:55.108); peak
  process-tree RSS was 5,493,408 KiB.
- No software transformer or logits fallback selected the published tokens.
- A Fresh-L2 reviewer independently accepted the bounded execution and
  RTL/reference evidence.

This proves that the bound RTL faithfully implements the bound W4A8 integer
reference for this execution. It does **not** prove that the W4A8 quantization
preserves BF16 model quality, that the design meets synthesis/PPA targets, or
that it has run on an FPGA.

## Why the raw evidence is not in Git

The sealed raw namespace is 87,775,958,624 bytes (about 82 GiB), with 85,368
files and 6,281 directories. Almost all space is generated Icarus material:
each of 37 positions is about 2.2 GiB, and each position contains 24 compiled
layer simulations. Individual compiled `residual.vvp` files are about 75.6 MB.

GitHub source repositories are not an appropriate transport for this generated
binary corpus. This directory instead publishes:

- hashes and sizes of the terminal raw artifacts;
- all 66 source bindings used by the official run;
- the four full-vocabulary LM-head summaries;
- the sealed start and status markers; and
- a verifier for this compact package and an optional local raw archive.

The raw archive remains immutable and its complete 85,367-entry `SHA256SUMS`
file verified successfully. Its own SHA-256 is recorded in
`PUBLICATION_MANIFEST.json`.

## Reproduce the checks

Verify this compact package:

```bash
python evidence/publication/rtl-arbitrary-text-generation-v1/attempt-0002/verify.py
```

If the raw archive and source tree are available locally:

```bash
python evidence/publication/rtl-arbitrary-text-generation-v1/attempt-0002/verify.py \
  --raw-root /path/to/attempt-0002 \
  --project-root /path/to/ace-2
```

Add `--full-raw` to verify every member listed by the raw archive's
`SHA256SUMS`; this reads roughly 82 GiB and can take several minutes.
