# ACE-2 Alpha 3 Status

Release label: **`v0.3.0-alpha.1`**

## Certified baseline

The Alpha 2 certification is preserved byte-for-byte as the accepted public
baseline:

| Area | Certified status |
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

## Productization status

| Area | Status at Alpha 3 cut |
|---|---|
| Authenticated source model | Qwen2.5-0.5B-Instruct revision frozen |
| BF16 successor package | S6 accepted by independent review |
| BF16 execution | Sole S6 lifecycle terminal; replay/resume forbidden |
| BF16 quality conclusion | `PROBE_QUALITY_GATE_FAILURE` |
| S6 probe result | 8/28 hard passes at epoch 1; 7/28 at epochs 2 and 3 |
| Official dev / retention / holdout | Not accessed |
| Arbitrary-text W4A8 chat | Blocked on a future accepted BF16 successor |
| Quantized-reference/RTL chat agreement | Not yet demonstrated |
| Alveo U280 package | Not yet built |
| Vitis/Vivado/XRT and U280 access | Not available in the release environment |

The certified scope remains a frozen pre-tokenized input and two generated
tokens. Alpha 3 does not extend that certification. See
[`docs/ALPHA3_PROGRESS.md`](docs/ALPHA3_PROGRESS.md).
