# Known Limitations

ACE-2 Alpha 2 is certified only for the demonstrated 24-layer, two-token
Qwen2.5-0.5B W4A8 command integration and mapped SKY130 synthesis/OpenSTA.

Not yet demonstrated:

- arbitrary natural-language prompt prefill;
- general multi-token or conversational generation;
- a stable tokenizer/host/deployment API;
- FPGA emulation, bitstream generation, or board execution;
- routed timing and congestion closure;
- power signoff;
- DRC/LVS or GDS/tapeout;
- silicon validation;
- performance beyond the recorded RTL simulation and mapped timing scope.

The distributed repository excludes model weights, private benchmark inputs,
raw proprietary PDK data, build products, local session state, and the sealed
exactly-once evidence archive.
