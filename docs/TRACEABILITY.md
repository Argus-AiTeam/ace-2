# ACE-2 Alpha 2 Traceability

| Claim | Release evidence |
|---|---|
| Certified synthesizable RTL tree exists | `rtl/` plus `CERTIFIED_RTL.sha256` |
| Layer-0 support is complete | 18/18 exact operator checks recorded in `CERTIFICATION.md` |
| Full two-token command path completes | 13,914/13,914 runtime commands, `first_failure = null` |
| Runtime identity is frozen | Model, image, and schedule hashes in `CERTIFICATION.md` |
| 100 MHz mapped timing passes | SKY130 metrics and exact RTL tree hash in `CERTIFICATION.md` |
| Release-local arithmetic is reproducible | `make demo` regenerates and simulates RMSNorm vectors |
| Arbitrary-text chat is complete | **No; post-Alpha-2 work** |
| FPGA deployment is complete | **No; post-Alpha-2 work** |
| Routed/tapeout/silicon signoff exists | **No** |

Raw model weights, private benchmarks, proprietary PDK data, and sealed
exactly-once evidence are retained outside this source snapshot.
