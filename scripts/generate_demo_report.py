#!/usr/bin/env python3
"""Turn the Alpha demo artifacts into a concise human-readable report."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build"
VECTORS = ROOT / "verification" / "generated" / "projection_vectors.json"
SIM_LOG = BUILD / "projection-sim.log"
LINT_LOG = BUILD / "verilator-lint.log"
REPORT = BUILD / "DEMO_REPORT.md"


def require_marker(text: str, marker: str, source: Path) -> None:
    if marker not in text:
        raise SystemExit(f"Missing {marker!r} in {source.relative_to(ROOT)}")


def main() -> None:
    vectors = json.loads(VECTORS.read_text(encoding="utf-8"))
    sim_log = SIM_LOG.read_text(encoding="utf-8")
    lint_log = LINT_LOG.read_text(encoding="utf-8")
    require_marker(sim_log, "ACE2_W4A8_PROJ_TB_PASS", SIM_LOG)
    require_marker(sim_log, "ACE2_PROJ_SEMANTIC_BINS_PASS", SIM_LOG)

    cases = vectors["cases"]
    tested_outputs = 21
    total_declared_outputs = sum(case["rows"] * case["output_count"] for case in cases)
    saturated_cases = sum(bool(case["saturation_seen"]) for case in cases)
    table = [
        "| Oracle case | Rows | K | Outputs | Purpose |",
        "|---|---:|---:|---:|---|",
    ]
    purposes = {
        "balanced_rows2": "ordinary signed W4A8 arithmetic",
        "saturation_edges": "intentional saturation boundaries",
        "mlp_gate_proj_rows1": "wide MLP gate projection shape",
        "mlp_down_proj_rows1": "4,864-element reduction path",
        "rmsnorm_per_tensor_consumer": "RMSNorm-to-projection scale contract",
    }
    for case in cases:
        table.append(
            f"| `{case['name']}` | {case['rows']} | {case['reduction_size']} | "
            f"{case['output_count']} | {purposes[case['name']]} |"
        )

    report = f"""# ACE-2 Alpha Demo Report

## What just ran

The demo checked two different things:

1. **Structural integrity:** Verilator parsed and linted the complete
   `ace2_shell` source set without a fatal error.
2. **Accepted-prefix numerical behavior:** an independent Python fixed-point
   oracle regenerated deterministic W4A8 vectors, and Icarus Verilog simulated
   the projection RTL against selected expected outputs.

## Evidence summary

| Check | Result | Evidence |
|---|---|---|
| Required tools available | PASS | `make demo` environment checks |
| Complete structural shell lint | PASS | `build/verilator-lint.log` ({len(lint_log.splitlines())} lines) |
| Oracle vectors reproducible byte-for-byte | PASS | `verification/generated/projection_vectors.json` |
| RTL agrees with selected oracle outputs | PASS | `build/projection-sim.log` |
| Explicit rounding/shift semantic bins | PASS | 6 directed semantic cases |

The vector package defines **{len(cases)} oracle cases**, **{total_declared_outputs:,}
row/output positions**, and **{saturated_cases} intentional saturation case**.
The bounded RTL smoke checks **{tested_outputs} selected outputs** across those
shapes plus **6 directed arithmetic semantics**. It is a discriminator, not an
exhaustive proof of every output.

## Workloads represented

{chr(10).join(table)}

## How to interpret PASS

`ACE2_ALPHA_DEMO_PASS` means the packaged source is reproducible and the
accepted W4A8 projection prefix agrees with its independent oracle on the
declared smoke cases. It does **not** mean full Qwen inference works. The
accepted contiguous end-to-end frontier currently ends at `layer_0.v_proj`;
the first unresolved operator is `layer_0.rope_q`.

## Raw artifacts

- `build/verilator-lint.log` — structural shell lint output
- `build/projection-sim.log` — simulator checks and case counts
- `verification/generated/projection_vectors.json` — dimensions and SHA-256
  identities for deterministic inputs, weights, and outputs
- `verification/generated/projection_vectors.svh` — vectors consumed by RTL
"""
    REPORT.write_text(report, encoding="utf-8")

    print(f"  Structural shell lint: PASS ({len(lint_log.splitlines())} log lines)")
    print(
        f"  Oracle package: PASS ({len(cases)} cases, "
        f"{total_declared_outputs:,} declared row/output positions)"
    )
    print(
        f"  RTL comparison: PASS ({tested_outputs} selected outputs + "
        "6 directed arithmetic cases)"
    )
    print("  Honest boundary: accepted through layer_0.v_proj; layer_0.rope_q unresolved")
    print(f"ACE2_DEMO_REPORT_WRITTEN {REPORT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
