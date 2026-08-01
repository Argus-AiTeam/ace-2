# ACE-2 Alpha Status

ACE is short for **Argus Compute Engine**; ACE-2 is its second-generation
experimental design.

Release label: **`v0.1.0-alpha`**

ACE-2 was designed and iterated by
[Argus](https://argusbot.cn/), with human ownership of the
mission, acceptance targets, authorization gates, and release decisions.

| Area | Alpha status |
|---|---|
| RTL/control/DMA framework | Structural coverage for the 434-item flow |
| Verification infrastructure | Public subset demonstrates W4A8 projection |
| Accepted numerical frontier | Contiguous through `layer_0.v_proj` |
| First unsupported operator | `layer_0.rope_q` |
| Full-model numerical quality | Not accepted |
| Qwen inference usability | Not claimed |
| Synthesis/PPA | Generic synthesis entrypoint; historical aggregate context |
| Intended audience | RTL, accelerator, verification, and agent-engineering researchers |

## Why structural coverage and accepted support differ

The repository contains downstream attention, softmax, activation, control, and
shell RTL. Those blocks can be linted, synthesized, and tested independently.
However, full-model benchmarking exposed an end-to-end numerical failure at the
RoPE boundary. Once an upstream value is wrong, downstream execution can remain
structurally valid while the composed model result is invalid.

For this reason the accepted contiguous frontier stops at `layer_0.v_proj`.
This is an evidence boundary, not a statement that all downstream source files
are absent.

## Current negative evidence

Historical full-model smoke ratios were far above the immutable 1.05x target.
The latest `layer0_projection_shadow_staged_attention_v1` candidate passed its
standalone reference, RTL simulation, and lint checks but failed the two-dataset
quality discriminator and paired smoke. It is a bounded no-go and is not part
of the accepted shell.

## What can be demonstrated honestly

- reproducible environment checks;
- structural shell lint;
- deterministic accepted-prefix oracle generation;
- W4A8 projection RTL simulation against independent expected values;
- generic synthesis of the accepted projection core;
- architecture and verification methodology;
- Argus evidence-driven candidate rejection and revision.

## Beta direction

Beta reuses the Alpha engineering base and repairs the numerical chain beginning
at RoPE. It does not restart the accelerator from zero. Complete numerical
support requires refreshed end-to-end verification after the first unsupported
operator is repaired.
