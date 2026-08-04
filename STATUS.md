# ACE-2 Alpha 2 Status

Release label: **`v0.2.0-alpha.1`**

| Area | Status |
|---|---|
| Layer-0 fixed-point operators | 18/18 exact PASS |
| Full model scope | Qwen2.5-0.5B W4A8, 24 layers, two tokens |
| RTL runtime | 13,914/13,914 commands PASS |
| Generated token IDs | `[0, 0]` |
| Runtime first failure | `null` |
| Simulator cycles | 1,240,410,384 |
| Certified RTL tree | `bf12e2c83b4d569b27bbbc14835ed8d36c39ec4e8820725cc7ad054fd7ffb4f6` |
| Mapped cells | 62,283 |
| Non-SRAM area | 0.614082704 mm2 |
| Detailed setup slack | +0.6966 ns at 100 MHz |
| OpenSTA WNS / TNS | 0.00 ns / 0.00 ns |
| Formal product certificate | `CERTIFIED` for the demonstrated scope |

The certified scope is a frozen pre-tokenized input and two generated tokens.
It proves the composed accelerator command path and mapped SKY130 timing/area
target for that scope. It does not prove arbitrary-text chat, long generation,
FPGA execution, routed timing, power, physical verification, or silicon.
