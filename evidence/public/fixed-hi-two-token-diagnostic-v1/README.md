# Fixed `Hi` Two-Token Verilated Diagnostic

This public evidence bundle records a completed fixed-input ACE-2 W4A8
diagnostic. It is separate from `make demo` and is not replayed by that command.

## Recorded result

| Field | Recorded value |
|---|---|
| User input | Fixed text `Hi` |
| Chat-template input length | 14 tokens |
| Simulator | Verilator |
| Commands completed | 175,855 / 175,855 |
| Simulator cycles | 14,244,639,094 |
| Runtime exit code | 0 |
| First failure | `null` |
| Generated token IDs | `[529, 529]` |
| Static tokenizer decode | `ertert` |
| Independent disposition | `ACCEPT_FIXED_DIAGNOSTIC_ONLY` |

The output is not useful language. This result is published only to show that
the recorded fixed-input RTL schedule completed and produced two token IDs. It
is not evidence of meaningful chat quality.

## Included records

- [`runtime_summary.json`](runtime_summary.json) — exact terminal runtime
  summary copied from the sealed local diagnostic;
- [`runtime_progress.json`](runtime_progress.json) — exact terminal progress
  record with model, image, schedule, and embedding hashes;
- [`review_public.json`](review_public.json) — public-safe Fresh-L2 disposition;
- [`SOURCE_BINDINGS.json`](SOURCE_BINDINGS.json) — hashes binding the public
  records and the larger local evidence reviewed at acceptance time.

The exact copied runtime records retain their original SHA-256 identities:

```text
366419e2c358f475fccd0dc3039e4ba7233cce121340a513731eb06eca639c5d  runtime_summary.json
644d72f89d25275c8f51494941c3a0ac8aba4f18772f1fd670678978d4eb8fab  runtime_progress.json
```

## Claim boundary

This bundle does **not** claim:

- arbitrary-text or arbitrary-prompt generation;
- meaningful, fluent, or high-quality output;
- interactive latency or useful performance;
- product or Stage 1 completion;
- FPGA, U280, or other hardware execution;
- permission to replay sealed exactly-once evidence.

Model weights, packed images, command journals, runtime packages, private
review records, local paths, and authorization artifacts are intentionally not
published.
