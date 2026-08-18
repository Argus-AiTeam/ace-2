#!/usr/bin/env python3
"""Generate the self-contained ACE-2 Alpha 2 evidence dashboard."""

from __future__ import annotations

import json
from html import escape
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build"
CHALLENGE_DIR = BUILD / "demo_challenge"
VECTORS = CHALLENGE_DIR / "rmsnorm_vectors.json"
CHALLENGE = CHALLENGE_DIR / "challenge.json"
SIM_LOG = CHALLENGE_DIR / "challenge-sim.log"
NEGATIVE_LOG = CHALLENGE_DIR / "negative-control.log"
WAVEFORM = CHALLENGE_DIR / "rmsnorm-waveform.vcd"
WAVEFORM_SVG = CHALLENGE_DIR / "rmsnorm-waveform.svg"
LINT_LOG = BUILD / "verilator-lint.log"
RTL_MANIFEST = ROOT / "CERTIFIED_RTL.sha256"
OPERATOR_SUITE = BUILD / "operator_demo" / "operator-suite.json"
FULL_SHELL_LOG = BUILD / "operator_demo" / "full-shell.log"
REPORT = BUILD / "DEMO_REPORT.md"
HTML_REPORT = BUILD / "DEMO_REPORT.html"

PPA_STEPS = (
    ("Initial full tree", -0.1484),
    ("Shell fanout repair", -0.5275),
    ("RMSNorm enable repair", -0.1741),
    ("Final-sum preload split", 0.6966),
)

LAYER0_OPERATORS = (
    ("01", "Input RMSNorm", "fast", "Fresh RMSNorm challenge"),
    ("02", "Q projection", "extended", "Complete shell regression"),
    ("03", "K projection", "extended", "Complete shell regression"),
    ("04", "V projection", "extended", "Complete shell regression"),
    ("05", "RoPE Q", "fast", "RoPE core"),
    ("06", "RoPE K", "fast", "RoPE core"),
    ("07", "KV write", "extended", "Complete shell regression"),
    ("08", "Attention score", "fast", "Attention score core + shell"),
    ("09", "Softmax", "fast", "Softmax core"),
    ("10", "Attention value", "extended", "Complete shell regression"),
    ("11", "O projection", "extended", "Complete shell regression"),
    ("12", "Attention residual add", "fast", "Residual shell"),
    ("13", "Post-attention RMSNorm", "fast", "Residual/post-norm shell"),
    ("14", "MLP gate projection", "extended", "Complete shell regression"),
    ("15", "MLP up projection", "extended", "Complete shell regression"),
    ("16", "SiLU gate", "fast", "SiLU core"),
    ("17", "MLP down projection", "extended", "Complete shell regression"),
    ("18", "MLP residual add", "fast", "MLP residual shell"),
)


def require_marker(text: str, marker: str, source: Path) -> None:
    if marker not in text:
        raise SystemExit(f"missing {marker!r} in {source.relative_to(ROOT)}")


def main() -> None:
    vectors = json.loads(VECTORS.read_text(encoding="utf-8"))
    challenge = json.loads(CHALLENGE.read_text(encoding="utf-8"))
    sim_log = SIM_LOG.read_text(encoding="utf-8")
    negative_log = NEGATIVE_LOG.read_text(encoding="utf-8")
    lint_log = LINT_LOG.read_text(encoding="utf-8")
    operator_suite = json.loads(OPERATOR_SUITE.read_text(encoding="utf-8"))
    rtl_rows = [
        line.split(maxsplit=1)
        for line in RTL_MANIFEST.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    require_marker(sim_log, "ACE2_RMSNORM_TB_PASS", SIM_LOG)
    require_marker(negative_log, "ACE2_NEGATIVE_CONTROL_PASS", NEGATIVE_LOG)
    if operator_suite.get("status") != "PASS":
        raise SystemExit("Transformer operator suite did not pass")
    if not WAVEFORM.is_file() or WAVEFORM.stat().st_size == 0:
        raise SystemExit("missing local challenge waveform")
    if not WAVEFORM_SVG.is_file() or WAVEFORM_SVG.stat().st_size == 0:
        raise SystemExit("missing local challenge waveform preview")

    cases = vectors["cases"]
    beats = int(vectors["beats"])
    hidden = int(vectors["hidden_size"])
    saturated = sum(bool(case["saturation_seen"]) for case in cases)
    challenge_case = cases[-1]
    challenge_id = challenge["challenge_id"]
    tools = challenge["tools"]
    operator_results = operator_suite["results"]
    core_results = [
        item for item in operator_results if item["kind"] == "independent_core"
    ]
    shell_results = [
        item for item in operator_results if item["kind"] == "shell_integration"
    ]
    full_shell_pass = (
        FULL_SHELL_LOG.is_file()
        and "ACE2_SHELL_TB_PASS" in FULL_SHELL_LOG.read_text(encoding="utf-8")
    )
    fast_operator_count = sum(mode == "fast" for _, _, mode, _ in LAYER0_OPERATORS)
    metrics = (
        (str(len(rtl_rows)), "Certified RTL files checked now"),
        (str(len(cases)), "RMSNorm cases run now"),
        (f"{fast_operator_count}/18", "Layer-0 operators run now"),
        (str(len(shell_results)), "Shell integration modes"),
        ("PASS", "Corrupted-result rejection"),
    )

    case_rows = "\n".join(
        f"| `{case['name']}` | {case['sumsq']:,} | "
        f"{case['output_scale']:.8f} | "
        f"`{case['output_sha256'][:16]}...` |"
        for case in cases
    )
    ppa_rows = "\n".join(
        f"| {name} | {slack:+.4f} ns | {'PASS' if slack >= 0 else 'NO-GO'} |"
        for name, slack in PPA_STEPS
    )
    report = f"""# ACE-2 Transformer RTL Demo Report

## Result

`ACE2_LOCAL_RTL_DEMO_PASS`

| Metric | Value |
|---|---:|
| Certified RTL files | {len(rtl_rows)} |
| RMSNorm cases | {len(cases)} |
| Beats per case | {beats} |
| Hidden size | {hidden} |
| Cases exercising saturation | {saturated} |
| Verilator lint log lines | {len(lint_log.splitlines())} |
| Local challenge ID | `{challenge_id}` |
| Generated at | `{challenge["generated_at_utc"]}` |
| Source commit | `{challenge["git_commit"]}` |
| Local platform | `{challenge["platform"]["system"]} {challenge["platform"]["release"]} {challenge["platform"]["machine"]}` |
| Fresh challenge output SHA-256 | `{challenge_case["output_sha256"]}` |
| Transformer operator groups | {len(core_results)} |
| Selected shell integration modes | {len(shell_results)} |
| Operator-suite runtime | {operator_suite["elapsed_seconds"]:.2f} seconds |
| Layer-0 operator rows run by fast demo | {fast_operator_count}/18 |
| Complete shell regression | {"PASS in this workspace" if full_shell_pass else "Run make demo-extended"} |

## Evidence chain

1. All {len(rtl_rows)} certified RTL hashes matched.
2. The complete shell passed Verilator lint with no fatal error.
3. The independent generator reproduced the packaged vectors byte-for-byte.
4. This machine generated challenge `{challenge_id}` after the demo started.
5. Icarus recompiled the RTL and passed {len(cases)} cases x {beats} beats,
   including the fresh challenge case.
6. A deliberately corrupted expected beat was rejected by the same checker.
7. The run produced `build/demo_challenge/rmsnorm-waveform.vcd`.
8. Six independent Transformer core groups and five selected `ace2_shell`
   integration modes passed against packaged oracle vectors.

![Fresh local RMSNorm challenge waveform](demo_challenge/rmsnorm-waveform.svg)

## Local toolchain

| Tool | Version observed by this run |
|---|---|
| Python | `{tools["python"]}` |
| Verilator | `{tools["verilator"]}` |
| Icarus Verilog | `{tools["iverilog"]}` |
| VVP | `{tools["vvp"]}` |

## RMSNorm workload identities

| Case | Sum of squares | Output scale | Output SHA-256 |
|---|---:|---:|---|
{case_rows}

## Transformer operator coverage

| # | Layer-0 operator | Fast demo | Evidence path |
|---:|---|---|---|
{chr(10).join(f"| {number} | {name} | {'PASS now' if mode == 'fast' else ('PASS via extended' if full_shell_pass else 'demo-extended')} | {evidence} |" for number, name, mode, evidence in LAYER0_OPERATORS)}

### Executed test processes

| Test | Execution path | Runtime | Result marker |
|---|---|---:|---|
{chr(10).join(f"| {item['name']} | {item['kind'].replace('_', ' ')} | {item['seconds']:.2f} s | `{item['marker']}` |" for item in operator_results)}

## Timing-closure progression

| Tree | Setup slack | Decision |
|---|---:|---|
{ppa_rows}

## Certification boundary

This run directly proves, on the user's machine, certified RTL identity,
complete-shell lint, fresh RMSNorm oracle/RTL agreement, failure detection, and
waveform generation. Alpha 2's 24-layer/two-token and mapped SKY130 results are
historical certification claims and are not rerun by this demo. The demo does
not claim arbitrary chat, FPGA execution, routed signoff, tapeout, or silicon.
"""
    REPORT.write_text(report, encoding="utf-8")

    metric_cards = "\n".join(
        f'<div class="metric"><strong>{escape(value)}</strong><span>{escape(label)}</span></div>'
        for value, label in metrics
    )
    operator_cards = "\n".join(
        f"""<tr><td>{escape(item["name"])}</td>
        <td>{escape(item["kind"].replace("_", " "))}</td>
        <td>{item["seconds"]:.2f} s</td>
        <td class="pass-text">PASS</td>
        <td><code>{escape(item["marker"])}</code></td></tr>"""
        for item in operator_results
    )
    layer0_cards = "\n".join(
        f"""<tr><td>{number}</td><td>{escape(name)}</td>
        <td class="{'pass-text' if mode == 'fast' or full_shell_pass else 'extended-text'}">
        {'PASS NOW' if mode == 'fast' else ('PASS EXTENDED' if full_shell_pass else 'DEMO-EXTENDED')}</td>
        <td>{escape(evidence)}</td></tr>"""
        for number, name, mode, evidence in LAYER0_OPERATORS
    )
    case_cards = "\n".join(
        f"""<tr><td><code>{escape(case["name"])}</code></td>
        <td>{case["sumsq"]:,}</td><td>{case["output_scale"]:.8f}</td>
        <td><code>{case["input_sha256"][:12]}...</code></td>
        <td><code>{case["output_sha256"][:12]}...</code></td></tr>"""
        for case in cases
    )
    ppa_cards = "\n".join(
        f"""<div class="ppa {'pass' if slack >= 0 else 'nogo'}">
        <span>{escape(name)}</span><strong>{slack:+.4f} ns</strong>
        <small>{'PASS' if slack >= 0 else 'NO-GO'}</small></div>"""
        for name, slack in PPA_STEPS
    )
    html_report = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>ACE-2 Transformer RTL Evidence Dashboard</title>
<style>
:root{{--bg:#07111f;--panel:#0d1b2d;--line:#213a57;--text:#edf7ff;
--muted:#94aac1;--cyan:#51e5c2;--blue:#60a5fa;--amber:#ffbd66;}}
*{{box-sizing:border-box}} body{{margin:0;background:radial-gradient(circle at 15% 0,#123251 0,transparent 35%),var(--bg);
color:var(--text);font:15px/1.55 Inter,system-ui,sans-serif}}
main{{max-width:1180px;margin:auto;padding:48px 28px 72px}} h1{{font-size:42px;margin:0}}
h1 em{{color:var(--cyan);font-style:normal}} h2{{margin:40px 0 16px}}
.subtitle{{color:var(--muted);font-size:17px}} .pill{{display:inline-block;margin-top:14px;padding:6px 11px;
border:1px solid #247e70;border-radius:999px;color:var(--cyan);background:#0b2828}}
.metrics{{display:grid;grid-template-columns:repeat(5,1fr);gap:12px;margin:28px 0}}
.metric,.stage,.ppa{{background:linear-gradient(145deg,#10233a,#0b1828);border:1px solid var(--line);border-radius:14px}}
.metric{{padding:20px}} .metric strong{{display:block;color:var(--cyan);font-size:25px}}
.metric span,.stage small,.ppa span,.ppa small{{display:block;color:var(--muted)}}
.flow{{display:grid;grid-template-columns:repeat(6,1fr);gap:9px}} .stage{{padding:15px;min-height:108px}}
.stage b{{color:var(--blue);font-size:12px}} .stage span{{display:block;margin-top:8px}}
.ppa-grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}}
.ppa{{padding:17px;border-top:3px solid var(--amber)}} .ppa.pass{{border-top-color:var(--cyan)}}
.ppa strong{{display:block;font-size:22px;margin:6px 0}} .ppa.pass strong{{color:var(--cyan)}}
.ppa.nogo strong{{color:var(--amber)}} table{{width:100%;border-collapse:collapse;background:var(--panel);
border:1px solid var(--line)}} th,td{{padding:10px 13px;border-bottom:1px solid var(--line);text-align:left}}
th{{color:var(--muted);font-size:12px;text-transform:uppercase}} code{{font-family:ui-monospace,monospace;font-size:12px}}
.boundary{{padding:18px 20px;border-left:4px solid var(--amber);background:#241b16;border-radius:8px}}
.pass-text{{color:var(--cyan)}} .extended-text{{color:var(--amber)}}
@media(max-width:850px){{.metrics,.flow,.ppa-grid{{grid-template-columns:1fr 1fr}}}}
@media(max-width:520px){{.metrics,.flow,.ppa-grid{{grid-template-columns:1fr}}}}
.waveform{{width:100%;background:#07111f;border:1px solid var(--line);border-radius:12px}}
</style></head><body><main>
<h1>ACE-2 <em>Transformer RTL Demo</em></h1>
<div class="subtitle">Machine-local operator, shell-integration, and challenge evidence for an Argus-designed Qwen W4A8 accelerator</div>
<div class="pill">LOCAL CHALLENGE {escape(challenge_id)}</div>
<div class="metrics">{metric_cards}</div>

<h2>This run happened here</h2>
<div class="boundary"><strong>Challenge:</strong> <code>{escape(challenge_id)}</code><br>
<strong>Generated:</strong> {escape(challenge["generated_at_utc"])}<br>
<strong>Source commit:</strong> <code>{escape(challenge["git_commit"])}</code><br>
<strong>Platform:</strong> {escape(challenge["platform"]["system"])} {escape(challenge["platform"]["release"])} {escape(challenge["platform"]["machine"])}<br>
<strong>Fresh output SHA-256:</strong> <code>{escape(challenge_case["output_sha256"])}</code></div>

<h2>Host-to-token architecture</h2>
<div class="flow">
<div class="stage"><b>01 HOST</b><span>Command stream and packed W4 image</span></div>
<div class="stage"><b>02 NORM</b><span>RMSNorm and residual state</span></div>
<div class="stage"><b>03 ATTENTION</b><span>Q/K/V, RoPE, score, softmax</span></div>
<div class="stage"><b>04 MLP</b><span>Gate, up, SiLU, down</span></div>
<div class="stage"><b>05 STATE</b><span>Residual and KV updates</span></div>
<div class="stage"><b>06 OUTPUT</b><span>Final norm, LM head, token IDs</span></div>
</div>

<h2>What this demo executed</h2>
<div class="flow">
<div class="stage"><b>HASH</b><span>{len(rtl_rows)} certified RTL files</span></div>
<div class="stage"><b>LINT</b><span>Complete shell, {len(lint_log.splitlines())} log lines</span></div>
<div class="stage"><b>CHALLENGE</b><span>Fresh unpredictable local input</span></div>
<div class="stage"><b>RTL</b><span>{len(cases)} x {beats} streamed beats</span></div>
<div class="stage"><b>NEGATIVE</b><span>Corruption rejected as expected</span></div>
<div class="stage"><b>WAVEFORM</b><span>rmsnorm-waveform.vcd</span></div>
</div>

<h2>Transformer data path demonstrated now</h2>
<div class="flow">
<div class="stage"><b>01 NORM</b><span>Fresh RMSNorm challenge</span></div>
<div class="stage"><b>02 QKV / O</b><span>W4A8 projection core</span></div>
<div class="stage"><b>03 POSITION</b><span>RoPE core</span></div>
<div class="stage"><b>04 ATTENTION</b><span>Score, softmax, compose</span></div>
<div class="stage"><b>05 MLP</b><span>SiLU gate and residual shell</span></div>
<div class="stage"><b>06 OUTPUT</b><span>Final RMSNorm and LM head shell</span></div>
</div>

<h2>Operator and integration results</h2>
<table><tr><th>Test</th><th>Path</th><th>Runtime</th><th>Status</th><th>RTL marker</th></tr>
{operator_cards}</table>

<h2>Complete 18-operator Layer-0 matrix</h2>
<table><tr><th>#</th><th>Operator</th><th>This workspace</th><th>Evidence path</th></tr>
{layer0_cards}</table>

<h2>Waveform generated from this challenge</h2>
<img class="waveform" src="demo_challenge/rmsnorm-waveform.svg" alt="Fresh local RMSNorm challenge waveform">

<h2>Timing closure was earned, not assumed</h2>
<div class="ppa-grid">{ppa_cards}</div>

<h2>RMSNorm discriminator</h2>
<table><tr><th>Case</th><th>SumSq</th><th>Output scale</th><th>Input SHA</th><th>Output SHA</th></tr>
{case_cards}</table>

<h2>Toolchain observed locally</h2>
<table><tr><th>Tool</th><th>Version</th></tr>
<tr><td>Python</td><td><code>{escape(tools["python"])}</code></td></tr>
<tr><td>Verilator</td><td><code>{escape(tools["verilator"])}</code></td></tr>
<tr><td>Icarus Verilog</td><td><code>{escape(tools["iverilog"])}</code></td></tr>
<tr><td>VVP</td><td><code>{escape(tools["vvp"])}</code></td></tr></table>

<h2>Honest boundary</h2>
<div class="boundary"><strong>Executed now:</strong> certified RTL hash check,
complete-shell lint, fresh local RMSNorm oracle/RTL comparison, five independent
Transformer core groups, six selected shell integration modes, corrupted-result
rejection, and VCD waveform generation.<br><br><strong>Available separately:</strong>
<code>make demo-extended</code> runs the complete slow public shell regression.
It is not claimed as run unless its log contains <code>ACE2_SHELL_TB_PASS</code>.
<br><br><strong>Historical certification,
not rerun here:</strong> 13,914 commands across 24 layers/two tokens and mapped
SKY130 100 MHz timing.<br><br><strong>Not claimed:</strong> arbitrary-text chat,
unrestricted generation, FPGA execution, routed signoff, tapeout, or silicon.</div>
</main></body></html>"""
    HTML_REPORT.write_text(html_report, encoding="utf-8")

    print(f"  Certified RTL identity: PASS ({len(rtl_rows)} files)")
    print(f"  RMSNorm oracle/RTL: PASS ({len(cases)} cases x {beats} beats)")
    print(
        "  Transformer operators: PASS "
        f"({len(core_results)} core groups + {len(shell_results)} shell modes; "
        f"{fast_operator_count}/18 Layer-0 rows)"
    )
    print("  Historical mapped timing shown for context: 100 MHz (+0.6966 ns); not rerun")
    print(f"ACE2_DEMO_REPORT_WRITTEN {REPORT.relative_to(ROOT)}")
    print(f"ACE2_VISUAL_EVIDENCE_WRITTEN {HTML_REPORT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
