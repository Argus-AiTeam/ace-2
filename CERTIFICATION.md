# ACE-2 Alpha 2 Certification Summary

This document is a distribution-safe summary of the internal machine-readable
certificate `ace2-qwen2p5-0p5b-w4a8-final-product-certificate-v1`.

## Functional authority

- Model: Qwen2.5-0.5B W4A8
- Model revision: `060db6499f32faf8b98477b0a26969ef7d8b9987`
- Raw model SHA-256:
  `88c142557820ccad55bb59756bfcfcf891de9cc6202816bd346445188a0ed342`
- Packed image SHA-256:
  `e24e0365e9fad5df2efe3e40df12e3f89f951f37c83449cb40e7d18fb614eafb`
- Command schedule SHA-256:
  `838b2c019a6028a92ffef8b9cc087cdcb616f33f60a20c6b24cb33aed37bb002`
- Runtime: 13,914/13,914 commands PASS
- Generated token IDs: `[0, 0]`
- First failure: `null`
- Simulator cycles: 1,240,410,384

## RTL authority

- Certified RTL tree SHA-256:
  `bf12e2c83b4d569b27bbbc14835ed8d36c39ec4e8820725cc7ad054fd7ffb4f6`
- Exact source hashes: [CERTIFIED_RTL.sha256](CERTIFIED_RTL.sha256)

## Mapped SKY130 authority

- Standard-cell library: SKY130 HD, TT, 25 C, 1.80 V
- Clock period: 10.000 ns
- Frequency target: 100 MHz
- Cells: 62,283
- Non-SRAM area: 0.614082704 mm2
- Detailed setup slack: +0.6966 ns
- WNS / TNS: 0.00 ns / 0.00 ns

## Review verdict

Fresh review issued `FORMAL_PRODUCT_CERTIFIED` for the demonstrated two-token
integration and mapped synthesis/OpenSTA scope. The source certificate and
sealed raw evidence remain in the controlled project archive; this repository
does not include local paths, private runtime state, model weights, proprietary
PDK files, or consumed exactly-once execution packets.

## Non-claims

This certification does not claim arbitrary-text chat, unrestricted token
generation, FPGA board operation, routed timing, power signoff, DRC/LVS,
GDS/tapeout, silicon validation, or external deployment interfaces.
