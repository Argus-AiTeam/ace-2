#!/usr/bin/env python3
"""Run one named Layer-0 operator proof and write a focused result."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build" / "single_operator"
RTL = ROOT / "rtl"
TB = ROOT / "verification" / "tb"

SHELL_SOURCES = [
    RTL / "ace2_pkg.sv",
    *[
        path
        for path in sorted(RTL.glob("ace2_*_core.sv"))
        if path.name != "ace2_pkg.sv"
    ],
    RTL / "ace2_shell.sv",
    TB / "ace2_shell_tb.sv",
]

OPERATORS = {
    "input-rmsnorm": ("core", "ace2_rmsnorm_tb.sv", "ACE2_RMSNORM_TB_PASS", []),
    "q-proj": ("shell", "", "ACE2_SHELL_QPROJ_STRIDE_TB_PASS", ["+QPROJ_STRIDE_ONLY"]),
    "k-proj": ("shell", "", "ACE2_SHELL_KPROJ_TB_PASS", ["+KPROJ_ONLY"]),
    "v-proj": ("shell", "", "ACE2_SHELL_VPROJ_TB_PASS", ["+VPROJ_ONLY"]),
    "rope-q": ("shell", "", "ACE2_SHELL_ROPE_TB_PASS", ["+ROPE_ONLY"]),
    "rope-k": ("shell", "", "ACE2_SHELL_ROPE_TB_PASS", ["+ROPE_ONLY"]),
    "kv-write": ("shell", "", "ACE2_SHELL_KV_WRITE_TB_PASS", ["+KV_WRITE_ONLY"]),
    "attention-score": ("shell", "", "ACE2_SHELL_ATTN_SCORE_TB_PASS", ["+ATTN_SCORE_ONLY"]),
    "softmax": ("shell", "", "ACE2_SHELL_SOFTMAX_TB_PASS", ["+SOFTMAX_ONLY"]),
    "attention-value": ("shell", "", "ACE2_SHELL_ATTN_VALUE_TB_PASS", ["+ATTN_VALUE_ONLY"]),
    "o-proj": ("shell", "", "ACE2_SHELL_SMOKE_TB_PASS", ["+SMOKE_OPCODE=01"]),
    "attention-residual": ("shell", "", "ACE2_SHELL_SMOKE_TB_PASS", ["+SMOKE_OPCODE=08"]),
    "post-attention-rmsnorm": ("shell", "", "ACE2_SHELL_SMOKE_TB_PASS", ["+SMOKE_OPCODE=08"]),
    "mlp-gate": ("shell", "", "ACE2_SHELL_MLP_GATE_TB_PASS", ["+MLP_GATE_ONLY"]),
    "mlp-up": ("shell", "", "ACE2_SHELL_MLP_UP_TB_PASS", ["+MLP_UP_ONLY"]),
    "silu": ("shell", "", "ACE2_SHELL_SILU_GATE_TB_PASS", ["+SILU_ONLY"]),
    "mlp-down": ("shell", "", "ACE2_SHELL_MLP_DOWN_TB_PASS", ["+MLP_DOWN_ONLY"]),
    "mlp-residual": ("shell", "", "ACE2_SHELL_MLP_RESIDUAL_TB_PASS", ["+MLP_RESIDUAL_ONLY"]),
}

SHARED_PATHS = {
    "rope-q": "The RoPE shell proof executes both Q and K roles.",
    "rope-k": "The RoPE shell proof executes both Q and K roles.",
    "attention-residual": "The vector-family proof also executes post-attention RMSNorm.",
    "post-attention-rmsnorm": "The vector-family proof also executes attention residual add.",
}


def command_result(command: list[str], log: Path) -> tuple[str, float]:
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
    log.write_text(result.stdout, encoding="utf-8")
    if result.returncode:
        raise SystemExit(
            f"operator command failed with exit {result.returncode}; see "
            f"{log.relative_to(ROOT)}"
        )
    return result.stdout, elapsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("operator", choices=sorted(OPERATORS))
    args = parser.parse_args()
    kind, testbench, marker, plusargs = OPERATORS[args.operator]
    operator_dir = BUILD / args.operator
    operator_dir.mkdir(parents=True, exist_ok=True)
    binary = operator_dir / "sim.vvp"

    if kind == "shell":
        compile_command = [
            "iverilog", "-g2012", "-Irtl", "-Irtl/generated",
            "-Iverification/tb", "-o", str(binary),
            *(str(path) for path in SHELL_SOURCES),
        ]
    else:
        compile_command = [
            "iverilog", "-g2012", "-Irtl", "-Irtl/generated",
            "-Iverification/tb", "-o", str(binary),
            *(str(path) for path in SHELL_SOURCES[:-1]),
            str(TB / testbench),
        ]
    command_result(compile_command, operator_dir / "compile.log")
    output, elapsed = command_result(
        ["vvp", str(binary), *plusargs],
        operator_dir / "sim.log",
    )
    marker_line = next(
        (line.strip() for line in output.splitlines() if marker in line),
        "",
    )
    if not marker_line:
        raise SystemExit(f"missing {marker!r} in {operator_dir / 'sim.log'}")

    result = {
        "schema_version": 1,
        "operator": args.operator,
        "status": "PASS",
        "seconds": round(elapsed, 3),
        "marker": marker_line,
        "shared_path_note": SHARED_PATHS.get(args.operator),
        "log": str((operator_dir / "sim.log").relative_to(ROOT)),
    }
    (operator_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"ACE2_SINGLE_OPERATOR_PASS operator={args.operator} seconds={elapsed:.2f}")
    print(marker_line)
    if result["shared_path_note"]:
        print(f"NOTE: {result['shared_path_note']}")
    print(f"Result: {(operator_dir / 'result.json').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
