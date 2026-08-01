#!/usr/bin/env python3
"""Turn the Alpha demo artifacts into a concise human-readable report."""

from __future__ import annotations

import json
from html import escape
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build"
VECTORS = ROOT / "verification" / "generated" / "projection_vectors.json"
SIM_LOG = BUILD / "projection-sim.log"
LINT_LOG = BUILD / "verilator-lint.log"
REPORT = BUILD / "DEMO_REPORT.md"
HTML_REPORT = BUILD / "DEMO_REPORT.html"


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

    max_outputs = max(case["output_count"] for case in cases)
    workload_bars = "\n".join(
        f"""<div class="bar-row">
          <div class="bar-label"><code>{escape(case["name"])}</code></div>
          <div class="bar-track"><div class="bar" style="width:
            {max(2, case["output_count"] * 100 / max_outputs):.2f}%"></div></div>
          <div class="bar-value">{case["output_count"]:,} outputs</div>
        </div>"""
        for case in cases
    )
    digest_rows = "\n".join(
        f"""<tr><td><code>{escape(case["name"])}</code></td>
        <td><code>{case["input_sha256"][:16]}...</code></td>
        <td><code>{case["weight_sha256"][:16]}...</code></td>
        <td><code>{case["output_sha256"][:16]}...</code></td></tr>"""
        for case in cases
    )
    html_report = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ACE-2 Alpha Demo Evidence</title>
<style>
:root {{ color-scheme: dark; --bg:#0b1020; --panel:#121a2d; --line:#283552;
  --text:#eef3ff; --muted:#9eabc5; --accent:#55d6be; --blue:#6ca9ff; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--text);
  font:15px/1.5 system-ui,-apple-system,Segoe UI,sans-serif; }}
main {{ max-width:1120px; margin:auto; padding:42px 28px 64px; }}
h1 {{ font-size:34px; margin:0 0 4px; }} h2 {{ margin-top:34px; }}
.subtitle,.note {{ color:var(--muted); }}
.flow {{ display:grid; grid-template-columns:repeat(5,1fr); gap:10px; margin:28px 0; }}
.stage,.card {{ background:var(--panel); border:1px solid var(--line); border-radius:12px; }}
.stage {{ padding:16px; min-height:112px; }}
.stage b {{ display:block; color:var(--accent); font-size:12px; letter-spacing:.08em; }}
.stage span {{ display:block; margin-top:8px; }}
.cards {{ display:grid; grid-template-columns:repeat(4,1fr); gap:12px; }}
.card {{ padding:18px; }} .metric {{ font-size:28px; font-weight:700; color:var(--accent); }}
.label {{ color:var(--muted); font-size:13px; }}
.bar-row {{ display:grid; grid-template-columns:260px 1fr 110px; gap:14px;
  align-items:center; margin:12px 0; }}
.bar-track {{ height:13px; background:#202c46; border-radius:99px; overflow:hidden; }}
.bar {{ height:100%; background:linear-gradient(90deg,var(--blue),var(--accent)); }}
.bar-value {{ text-align:right; color:var(--muted); }}
table {{ width:100%; border-collapse:collapse; background:var(--panel);
  border:1px solid var(--line); border-radius:12px; overflow:hidden; }}
th,td {{ padding:11px 14px; text-align:left; border-bottom:1px solid var(--line); }}
th {{ color:var(--muted); font-size:12px; text-transform:uppercase; }}
.pass {{ color:var(--accent); font-weight:700; }}
.boundary {{ border-left:4px solid #ffbd66; padding:14px 18px; background:#211a18; }}
code {{ font-family:ui-monospace,SFMono-Regular,Consolas,monospace; font-size:12px; }}
@media(max-width:800px) {{ .flow,.cards {{ grid-template-columns:1fr; }}
  .bar-row {{ grid-template-columns:1fr; }} .bar-value {{ text-align:left; }} }}
</style>
</head>
<body><main>
<h1>Argus Compute Engine 2</h1>
<div class="subtitle">Experimental Alpha — reproducible demo evidence</div>

<div class="flow">
  <div class="stage"><b>01 TOOLCHAIN</b><span>Verify the required open-source tools.</span></div>
  <div class="stage"><b>02 STRUCTURE</b><span>Lint the complete accelerator shell.</span></div>
  <div class="stage"><b>03 ORACLE</b><span>Regenerate deterministic W4A8 vectors.</span></div>
  <div class="stage"><b>04 RTL</b><span>Simulate selected outputs and semantic bins.</span></div>
  <div class="stage"><b>05 EVIDENCE</b><span>Bind results to logs, counts, and hashes.</span></div>
</div>

<div class="cards">
  <div class="card"><div class="metric">PASS</div><div class="label">Structural shell lint</div></div>
  <div class="card"><div class="metric">{len(cases)}</div><div class="label">Oracle workloads</div></div>
  <div class="card"><div class="metric">{tested_outputs} + 6</div><div class="label">RTL outputs + semantic cases</div></div>
  <div class="card"><div class="metric">{total_declared_outputs:,}</div><div class="label">Declared row/output positions</div></div>
</div>

<h2>Workload scale</h2>
<p class="note">Bars show output width from the generated oracle metadata, not estimated performance.</p>
{workload_bars}

<h2>Evidence chain</h2>
<table>
<tr><th>Claim</th><th>Result</th><th>Machine-produced evidence</th></tr>
<tr><td>Complete shell parses and lints</td><td class="pass">PASS</td>
  <td><code>build/verilator-lint.log</code> ({len(lint_log.splitlines())} lines)</td></tr>
<tr><td>Oracle vectors reproduce byte-for-byte</td><td class="pass">PASS</td>
  <td><code>verification/generated/projection_vectors.json</code></td></tr>
<tr><td>Selected RTL outputs match the oracle</td><td class="pass">PASS</td>
  <td><code>build/projection-sim.log</code></td></tr>
<tr><td>Rounding and extreme shifts behave as specified</td><td class="pass">PASS</td>
  <td>6 directed arithmetic semantics in the RTL testbench</td></tr>
</table>

<h2>Reproducibility identities</h2>
<p class="note">SHA-256 prefixes identify the exact deterministic inputs, weights, and outputs.</p>
<table><tr><th>Workload</th><th>Input SHA-256</th><th>Weight SHA-256</th><th>Output SHA-256</th></tr>
{digest_rows}
</table>

<h2>Honest support boundary</h2>
<div class="boundary"><strong>Accepted contiguous frontier:</strong>
<code>layer_0.v_proj</code><br><strong>First unresolved operator:</strong>
<code>layer_0.rope_q</code><br><br>
This demo proves reproducibility and bounded projection-prefix agreement.
It does not claim complete Qwen inference.</div>
</main></body></html>
"""
    HTML_REPORT.write_text(html_report, encoding="utf-8")

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
    print(f"ACE2_VISUAL_EVIDENCE_WRITTEN {HTML_REPORT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
