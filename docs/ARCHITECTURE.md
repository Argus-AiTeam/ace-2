# Public Architecture Summary

ACE-2 is a W4A8 accelerator research RTL with a descriptor-driven shell,
control/status handling, DMA-style memory interfaces, fixed-point operator
cores, and a framework representing the complete 434-item Qwen2.5-0.5B flow.

Included structural operator blocks cover RMSNorm, W4A8 projection, RoPE
research paths, attention score, softmax, attention composition, SiLU gating,
and shell integration.

The existence or reachability of an operator block does not establish numerical
acceptance. The accepted contiguous numerical frontier is:

```text
layer_0.input_rmsnorm
layer_0.q_proj
layer_0.k_proj
layer_0.v_proj
```

The next operator, `layer_0.rope_q`, is unsupported end to end. Downstream
blocks remain structural/research infrastructure only.
