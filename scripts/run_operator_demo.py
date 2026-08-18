#!/usr/bin/env python3
"""Run the fast ACE-2 Transformer operator and shell-integration demo."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build" / "operator_demo"
RTL = ROOT / "rtl"
TB = ROOT / "verification" / "tb"
RANDOM_CHALLENGE = BUILD / "random_challenge"
RANDOM_TB = RANDOM_CHALLENGE / "verification" / "tb"

CORE_TESTS = (
    ("RoPE", "ace2_rope_core.sv", "ace2_rope_tb.sv", "ACE2_ROPE_TB_PASS"),
    ("Attention score", "ace2_attention_score_core.sv", "ace2_attention_score_tb.sv", "ACE2_ATTN_SCORE_TB_PASS"),
    ("Softmax", "ace2_softmax_core.sv", "ace2_softmax_tb.sv", "ACE2_SOFTMAX_TB_PASS"),
    ("Attention compose", "ace2_attention_compose_core.sv", "ace2_attention_compose_tb.sv", "ACE2_ATTN_COMPOSE_TB_PASS"),
    ("SiLU gate", "ace2_silu_gate_core.sv", "ace2_silu_gate_tb.sv", "ACE2_SILU_GATE_TB_PASS"),
)

SHELL_MODES = (
    ("Attention score shell", "+ATTN_SCORE_ONLY", "ACE2_SHELL_ATTN_SCORE_TB_PASS"),
    ("Attention compose shell", "+ATTN_COMPOSE_ONLY", "ACE2_SHELL_ATTN_COMPOSE_ONLY_TB_PASS"),
    ("Residual and post-norm shell", "+SMOKE_OPCODE=08", "ACE2_SHELL_SMOKE_TB_PASS"),
    ("MLP residual shell", "+MLP_RESIDUAL_ONLY", "ACE2_SHELL_MLP_RESIDUAL_TB_PASS"),
    ("Final RMSNorm shell", "+FINAL_RMSNORM_ONLY", "ACE2_SHELL_FINAL_RMSNORM_TB_PASS"),
    ("LM head shell", "+LM_HEAD_ONLY", "ACE2_SHELL_LM_HEAD_TB_PASS"),
)


def run(command: list[str], log_path: Path) -> tuple[str, float]:
    started = time.monotonic()
    result = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    elapsed = time.monotonic() - started
    log_path.write_text(result.stdout, encoding="utf-8")
    if result.returncode:
        raise SystemExit(
            f"{' '.join(command)} failed with exit {result.returncode}; see "
            f"{log_path.relative_to(ROOT)}"
        )
    return result.stdout, elapsed


def require_marker(output: str, marker: str, log_path: Path) -> str:
    for line in output.splitlines():
        if marker in line:
            return line.strip()
    raise SystemExit(f"missing {marker!r} in {log_path.relative_to(ROOT)}")


def compile_core(core: str, testbench: str, output: Path) -> None:
    compile_log = output.with_suffix(".compile.log")
    run(
        [
            "iverilog",
            "-g2012",
            "-Irtl",
            "-Irtl/generated",
            "-Iverification/tb",
            "-o",
            str(output),
            str(RTL / core),
            str(RANDOM_TB / testbench),
        ],
        compile_log,
    )


def compile_shell(output: Path) -> None:
    sources = [RTL / "ace2_pkg.sv"]
    sources.extend(
        path
        for path in sorted(RTL.glob("ace2_*_core.sv"))
        if path.name != "ace2_pkg.sv"
    )
    sources.extend((RTL / "ace2_shell.sv", TB / "ace2_shell_tb.sv"))
    run(
        [
            "iverilog",
            "-g2012",
            "-Irtl",
            "-Irtl/generated",
            "-Iverification/tb",
            "-o",
            str(output),
            *(str(path) for path in sources),
        ],
        output.with_suffix(".compile.log"),
    )


def main() -> None:
    BUILD.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, object]] = []
    seed = os.environ.get("SEED")
    challenge_command = [sys.executable, str(ROOT / "scripts" / "create_operator_challenge.py")]
    if seed:
        challenge_command.extend(("--seed", seed))
    challenge_output, _ = run(
        challenge_command,
        BUILD / "random-challenge.log",
    )
    challenge = json.loads(
        (RANDOM_CHALLENGE / "challenge.json").read_text(encoding="utf-8")
    )
    print(require_marker(
        challenge_output,
        "ACE2_RANDOM_OPERATOR_CHALLENGE_CREATED",
        BUILD / "random-challenge.log",
    ))

    for index, (name, core, testbench, marker) in enumerate(CORE_TESTS, start=1):
        binary = BUILD / f"core-{index}.vvp"
        log = BUILD / f"core-{index}.log"
        compile_core(core, testbench, binary)
        output, elapsed = run(["vvp", str(binary)], log)
        marker_line = require_marker(output, marker, log)
        results.append(
            {
                "name": name,
                "kind": "independent_core",
                "status": "PASS",
                "seconds": round(elapsed, 3),
                "marker": marker_line,
                "log": str(log.relative_to(ROOT)),
                "oracle": "bit-accurate Python reference plus random seeded case",
            }
        )
        print(f"ACE2_OPERATOR_PASS kind=core name={name!r} seconds={elapsed:.2f}")

    shell_binary = BUILD / "ace2_shell_tb.vvp"
    compile_shell(shell_binary)
    for index, (name, plusarg, marker) in enumerate(SHELL_MODES, start=1):
        log = BUILD / f"shell-{index}.log"
        output, elapsed = run(["vvp", str(shell_binary), plusarg], log)
        marker_line = require_marker(output, marker, log)
        results.append(
            {
                "name": name,
                "kind": "shell_integration",
                "status": "PASS",
                "seconds": round(elapsed, 3),
                "marker": marker_line,
                "log": str(log.relative_to(ROOT)),
            }
        )
        print(f"ACE2_OPERATOR_PASS kind=shell name={name!r} seconds={elapsed:.2f}")

    summary = {
        "schema_version": 2,
        "status": "PASS",
        "random_seed": challenge["seed"],
        "random_challenge": str(
            (RANDOM_CHALLENGE / "challenge.json").relative_to(ROOT)
        ),
        "random_oracles": challenge["operators"],
        "operator_groups": len(CORE_TESTS),
        "shell_modes": len(SHELL_MODES),
        "elapsed_seconds": round(sum(float(item["seconds"]) for item in results), 3),
        "results": results,
        "boundary": (
            "Independent core tests included fresh seeded random Python-oracle cases; "
            "selected ace2_shell integration modes used packaged bit-accurate vectors. "
            "Projection-family, KV-write, and attention-value coverage remains in "
            "make demo-extended. This fast suite is not the sealed 24-layer/two-token replay."
        ),
    }
    summary_path = BUILD / "operator-suite.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        "ACE2_TRANSFORMER_OPERATOR_SUITE_PASS "
        f"core_groups={len(CORE_TESTS)} shell_modes={len(SHELL_MODES)}"
    )


if __name__ == "__main__":
    main()
