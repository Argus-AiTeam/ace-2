# Known Limitations

1. Numerical acceptance stops after `layer_0.v_proj`; `layer_0.rope_q` and all
   downstream end-to-end behavior are unsupported.
2. The package does not provide usable Qwen inference or accepted full-model
   quality.
3. Structural 434-item coverage describes command/control and RTL framework
   reachability, not bit-accurate end-to-end model correctness.
4. The projection-shadow staged-attention candidate is experimental and a
   bounded no-go.
5. The demo uses deterministic synthetic vectors, not model weights or private
   datasets.
6. Generic synthesis is not place-and-route, signoff, tapeout, FPGA, or silicon
   evidence.
7. Historical aggregate PPA is informational and is not reproduced by the
   default demo.
8. Public redistribution is blocked pending project-level license resolution.
