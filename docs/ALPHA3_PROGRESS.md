# ACE-2 Alpha 3 Productization Progress

Alpha 3 is a public-safe progress release. It preserves the exact Alpha 2
certified RTL baseline and documents the work required to turn that bounded
two-token demonstration into a useful accelerator-backed chat system.

## Preserved certified result

The accepted baseline remains:

- 18/18 Layer-0 fixed-point operator boundaries;
- 13,914/13,914 runtime commands;
- all 24 transformer layers and two generated tokens;
- a 23-file synthesizable RTL tree;
- mapped SKY130 100 MHz timing closure;
- 62,283 mapped cells and 0.614082704 mm2 non-SRAM area.

No exactly-once certification or PPA action was replayed for Alpha 3.

## Model-quality work after Alpha 2

The product path uses Qwen2.5-0.5B-Instruct as the authenticated source model.
Several bounded BF16 successor attempts were rejected by frozen quality gates.
The current S6 successor was redesigned around:

- category-balanced training data;
- independent per-category gradients;
- deterministic conflict projection;
- post-combine clipping;
- probe-locked checkpoint selection with no fallback;
- frozen dev, per-category, safety, retention, and holdout gates.

The S6 package passed its independent package review. Exactly one lifecycle was
then explicitly authorized and executed. It failed closed at probe lock:

- epoch 1: 8/28 hard passes;
- epoch 2: 7/28 hard passes;
- epoch 3: 7/28 hard passes;
- only arithmetic and context-memory category minima passed;
- each epoch recorded four critical safety failures;
- no checkpoint was selected;
- official dev, retention, and holdout were not accessed.

The terminal taxonomy is `PROBE_QUALITY_GATE_FAILURE`. S6 is sealed and may not
be retried, resumed, rescored, or converted into a downstream quality claim.
Alpha 3 includes only these aggregate facts, not checkpoints, model weights,
row-level probe outputs, backend logs, or authority records.

## Development update: 2026-08-18

Post-Alpha-3 work strengthened the provenance and execution-admission boundary
needed before any protected Stage 1 action. The activation attempt failed
closed because the available host account retained root-equivalent local
service access, so it could not provide a trust root independent of the agent
being evaluated.

After several rejected intermediate recovery lineages, V8 closed the
package-self-authentication defect by requiring the first trusted action and
the accepted package identity to arrive through a channel outside the package
being authenticated. Fresh-L2 review accepted V8 with:

- 58 verifier checks completed;
- zero reported issues;
- content SHA-256
  `07663099352edfad32eb39919ad9475f1f887328ebb549bdb9cae1c48f5ccad1`;
- status `BUILD_READY_EXTERNAL_ROOT_REQUIRED`;
- no installed-tree verification, privileged execution, or Stage 1
  completion.

The resulting public status is:

- the Alpha 2 certified RTL and all terminal model evidence remain unchanged;
- no protected Stage 1 execution was accepted or replayed;
- arbitrary-text chat, new RTL acceptance, XRT integration, and U280 execution
  remain unauthorized and unclaimed;
- the accepted V8 recovery package is build-ready for an independent external
  root operator;
- the current account cannot authenticate, install, or execute V8 as its own
  trust authority;
- the sole remaining recovery prerequisite is an external channel that
  independently supplies the exact accepted package identity and root
  invocation.

This update intentionally excludes local paths, migration payloads, claims,
receipts, runtime logs, credentials, protected evidence, and private package
contents. See [`HOST_TRUST_RECOVERY.md`](HOST_TRUST_RECOVERY.md) for the
public-safe trust boundary.

## Ordered remaining gates

1. Establish and independently verify the execution-admission trust root.
2. A structurally justified new BF16 successor must be frozen and independently
   reviewed; S6 itself cannot continue.
3. That successor must pass its frozen probe selector before official dev.
4. It must then pass official dev, all category minima, zero critical safety
   failures, retention, and exactly-once holdout.
5. A Fresh Reviewer must accept the complete BF16 result.
6. Only then may W4A8 arbitrary-text prefill, tokenizer/host integration, KV
   reuse, readable multi-token decoding, and quantized-reference/RTL agreement
   become the active product gate.
7. U280 work follows the accepted local chat system and requires a build-ready
   PCIe/XRT/HBM2 package plus external Vitis/Vivado/XRT and board access.

## Explicit non-claims

Alpha 3 does not claim:

- a qualified S6 model or permission to replay S6;
- an established independent host trust root or execution authority;
- arbitrary-text or multi-turn chat;
- a general W4A8 quality result;
- a new RTL certification beyond Alpha 2;
- FPGA emulation, bitstream generation, or U280 board execution;
- routed signoff, tapeout, or silicon.

The release is useful as a reproducible certified RTL baseline and an honest
record of the productization gates still in progress.
