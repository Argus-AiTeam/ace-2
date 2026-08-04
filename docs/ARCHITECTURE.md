# ACE-2 Alpha 2 Architecture Summary

Argus Compute Engine 2 (ACE-2) is a descriptor-driven W4A8 accelerator RTL
for Qwen2.5-0.5B. Its shell coordinates DMA-style memory traffic, fixed-point
operator cores, residual paths, KV state, and LM-head execution.

The certified source tree contains 23 synthesizable RTL files covering:

- RMSNorm and residual addition;
- W4A8 Q/K/V/O, MLP, down-projection, and LM-head projection paths;
- RoPE variants and attention score/value composition;
- softmax and SiLU gating;
- command, status, memory, and completion control.

Alpha 2 executes the full 24-layer, two-token command schedule for one frozen
pre-tokenized input. All 18 Layer-0 operator boundaries pass exact fixed-point
checks, and the composed runtime completes 13,914/13,914 commands.

This architecture snapshot does not yet implement a supported arbitrary-text
host/chat interface or unrestricted generation. Those are post-Alpha-2
productization tasks and are not part of the certified tree.
