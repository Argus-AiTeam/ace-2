#!/usr/bin/env python3
"""Freeze and execute the bounded fused-QKV RTL benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "benchmark/fused_qkv_v1/contract.json"
TB = ROOT / "verification/tb/ace2_fused_qkv_benchmark_tb.sv"
TOOL = Path(__file__).resolve()
LIVE_SHELL = ROOT / "rtl/ace2_shell.sv"
BUILD_ROOT = ROOT / "build/fused-qkv-v1"
EVIDENCE_ROOT = ROOT / "evidence/verification/fused-qkv-v1"
K_SIZE = 896
OUTPUT_COUNTS = (896, 128, 128)
TOTAL_OUTPUTS = sum(OUTPUT_COUNTS)
RTL_SOURCES = [
    "rtl/ace2_pkg.sv",
    "rtl/ace2_rmsnorm_core.sv",
    "rtl/ace2_dynamic_scale32_core.sv",
    "rtl/ace2_w4a8_proj_core.sv",
    "rtl/ace2_rope_core.sv",
    "rtl/ace2_dynamic_rope_head_core.sv",
    "rtl/ace2_fixed_q7_rope_score_core.sv",
    "rtl/ace2_relative_rope_score_fusion_core.sv",
    "rtl/ace2_attention_score_core.sv",
    "rtl/ace2_softmax_core.sv",
    "rtl/ace2_attention_compose_core.sv",
    "rtl/ace2_silu_gate_core.sv",
]
PASS_RE = re.compile(
    r"ACE2_FUSED_QKV_PASS mode=\s*(?P<mode>\w+)"
    r"\s+commands=(?P<commands>\d+)"
    r"\s+outputs_checked=(?P<outputs_checked>\d+)"
    r"\s+simulator_cycles=(?P<simulator_cycles>\d+)"
    r"\s+memory_read_requests=(?P<memory_read_requests>\d+)"
    r"\s+activation_read_requests=(?P<activation_read_requests>\d+)"
    r"\s+weight_read_requests=(?P<weight_read_requests>\d+)"
    r"\s+metadata_read_requests=(?P<metadata_read_requests>\d+)"
    r"\s+memory_write_requests=(?P<memory_write_requests>\d+)"
    r"\s+backpressure_stalls=(?P<backpressure_stalls>\d+)"
)


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def write_atomic(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    return {"bytes": path.stat().st_size, "sha256": sha256_file(path)}


def tool_version(argv: list[str]) -> str:
    completed = subprocess.run(
        argv, cwd=ROOT, check=False, text=True, capture_output=True
    )
    raw = (completed.stdout + completed.stderr).strip()
    if completed.returncode != 0 or not raw:
        raise RuntimeError(f"tool version failed: {' '.join(argv)}")
    return raw.splitlines()[0]


def to_sint(value: int, width: int) -> int:
    value &= (1 << width) - 1
    return value - (1 << width) if value & (1 << (width - 1)) else value


def round_shift_even(value: int, shift: int) -> int:
    sign = -1 if value < 0 else 1
    magnitude = abs(value)
    base = magnitude >> shift
    remainder = magnitude & ((1 << shift) - 1)
    half = 1 << (shift - 1)
    if remainder > half or (remainder == half and (base & 1)):
        base += 1
    return sign * base


def activation(element: int) -> int:
    return ((element * 13 + 7) % 255) - 127


def weight(projection: int, output: int, element: int) -> int:
    return ((projection * 11 + output * 5 + element * 3 + 1) % 15) - 7


def reference_outputs() -> tuple[bytes, list[int]]:
    values = bytearray()
    diversity: list[int] = []
    activations = [activation(element) for element in range(K_SIZE)]
    for projection, output_count in enumerate(OUTPUT_COUNTS):
        projection_values = bytearray()
        for output in range(output_count):
            accumulator = sum(
                act * weight(projection, output, element)
                for element, act in enumerate(activations)
            )
            accumulator += output * 31 + projection * 257 - 1984
            multiplier = to_sint(
                1_000_003 + projection * 4_099 + output * 97, 32
            )
            scaled = round_shift_even(accumulator * multiplier, 25)
            projection_values.append(max(-128, min(127, scaled)) & 0xFF)
        values.extend(projection_values)
        diversity.append(len(set(projection_values)))
    return bytes(values), diversity


def source_records() -> dict[str, dict[str, Any]]:
    records = {path: file_record(ROOT / path) for path in RTL_SOURCES}
    records["rtl/ace2_shell.sv"] = file_record(LIVE_SHELL)
    records["rtl/ace2_pkg.sv"] = file_record(ROOT / "rtl/ace2_pkg.sv")
    records["constraints/ace2_rmsnorm_core.sdc"] = file_record(
        ROOT / "constraints/ace2_rmsnorm_core.sdc"
    )
    records[str(TB.relative_to(ROOT))] = file_record(TB)
    records[str(TOOL.relative_to(ROOT))] = file_record(TOOL)
    return records


def compile_benchmark(shell: Path, output: Path) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    binary = output / "ace2_fused_qkv_benchmark.vvp"
    sources = [str(ROOT / path) for path in RTL_SOURCES]
    sources.extend([str(shell), str(TB)])
    completed = subprocess.run(
        [
            "iverilog",
            "-g2012",
            "-Wall",
            "-Irtl",
            "-s",
            "ace2_fused_qkv_benchmark_tb",
            "-o",
            str(binary),
            *sources,
        ],
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=True,
    )
    (output / "compile.stdout.log").write_text(completed.stdout, encoding="utf-8")
    (output / "compile.stderr.log").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(
            f"fused QKV benchmark compile failed:\n"
            f"{completed.stdout}\n{completed.stderr}"
        )
    return binary


def freeze() -> None:
    if CONTRACT.exists():
        raise RuntimeError("fused QKV benchmark contract is already frozen")
    freeze_output = BUILD_ROOT / "contract-compile"
    if freeze_output.exists():
        shutil.rmtree(freeze_output)
    binary = compile_benchmark(LIVE_SHELL, freeze_output)
    expected, diversity = reference_outputs()
    contract = {
        "schema_version": 1,
        "name": "fused-qkv-v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "fixed_external_benchmark": False,
        "protected_or_sealed_inputs_used": False,
        "official_full_chat_invoked": False,
        "public_rtl_contract": {
            "top_module": "ace2_shell",
            "parameters": "unchanged defaults",
            "ports_frozen_by": file_record(LIVE_SHELL),
            "clock": "clk_i, 10 ns period",
            "reset": "rst_ni active-low asynchronous; five initial rising edges",
            "command_protocol": (
                "legacy opcode 0x01 remains one projection per descriptor; "
                "fused opcode 0x0b is one ordered Q/K/V descriptor"
            ),
            "memory_protocol": (
                "one-beat abstract streaming requests, deterministic ready "
                "stalls, and one-cycle read responses"
            ),
            "compile_probe": file_record(binary),
        },
        "workload": {
            "operator": "Qwen2.5-0.5B-shaped Q/K/V projection",
            "legacy_opcode": 1,
            "fused_opcode": 11,
            "flags": 0,
            "m": 1,
            "q_n": OUTPUT_COUNTS[0],
            "k_n": OUTPUT_COUNTS[1],
            "v_n": OUTPUT_COUNTS[2],
            "k": K_SIZE,
            "legacy_host_commands": 3,
            "fused_host_commands": 1,
            "outputs_checked": TOTAL_OUTPUTS,
            "activation_formula": "((element*13+7)%255)-127",
            "weight_formula": (
                "((projection*11+output*5+element*3+1)%15)-7"
            ),
            "bias_formula": "output*31+projection*257-1984",
            "multiplier_formula": (
                "1000003+projection*4099+output*97"
            ),
            "right_shift": 25,
            "output_zero_point": 0,
            "expected_output": {
                "bytes": len(expected),
                "sha256": sha256_bytes(expected),
                "distinct_values_q_k_v": diversity,
            },
        },
        "score_policy": {
            "correctness": (
                "all 1152 Q/K/V RTL bytes exactly equal the Python "
                "fixed-point oracle in legacy and fused modes"
            ),
            "shared_reads": (
                "fused activation reads equal 56 and are fewer than legacy"
            ),
            "control": "fused commands equal 1 and legacy commands equal 3",
            "performance": (
                "fused simulator cycles or measured wall time is lower than "
                "three legacy projections"
            ),
            "regressions": (
                "deterministic request/write backpressure, mid-command reset "
                "recovery, read-tag corruption, and legacy commands all pass"
            ),
        },
        "tools": {
            "python": sys.version.split()[0],
            "iverilog": tool_version(["iverilog", "-V"]),
            "vvp": tool_version(["vvp", "-V"]),
            "yosys": tool_version(["yosys", "-V"]),
        },
        "evaluator": {
            str(TB.relative_to(ROOT)): file_record(TB),
            str(TOOL.relative_to(ROOT)): file_record(TOOL),
        },
        "pre_edit_sources_and_constraint_hashes": source_records(),
        "limitations": [
            (
                "This deterministic Qwen-shaped QKV workload is not a full "
                "model or chat attempt."
            ),
            (
                "The abstract streaming-memory boundary does not model a "
                "physical DRAM controller."
            ),
            (
                "Area and timing require separate mission-local SKY130 "
                "synthesis/OpenSTA evidence."
            ),
        ],
    }
    write_atomic(CONTRACT, canonical_bytes(contract))
    print(
        "ACE2_FUSED_QKV_FREEZE_PASS "
        f"interface_sha256={contract['public_rtl_contract']['ports_frozen_by']['sha256']} "
        f"expected_sha256={contract['workload']['expected_output']['sha256']}"
    )


def read_contract() -> dict[str, Any]:
    if not CONTRACT.is_file():
        raise RuntimeError("missing frozen fused QKV contract")
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    for path, expected in contract["evaluator"].items():
        if file_record(ROOT / path) != expected:
            raise RuntimeError(f"frozen evaluator changed: {path}")
    expected, diversity = reference_outputs()
    if {
        "bytes": len(expected),
        "sha256": sha256_bytes(expected),
        "distinct_values_q_k_v": diversity,
    } != contract["workload"]["expected_output"]:
        raise RuntimeError("frozen QKV oracle changed")
    return contract


def execute_case(
    binary: Path,
    expected_path: Path,
    output: Path,
    name: str,
    plusargs: list[str],
) -> tuple[subprocess.CompletedProcess[str], float]:
    start = time.perf_counter()
    completed = subprocess.run(
        [
            "vvp",
            str(binary),
            f"+EXPECTED={expected_path.relative_to(ROOT)}",
            *plusargs,
        ],
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=True,
    )
    wall_seconds = time.perf_counter() - start
    (output / f"{name}.stdout.log").write_text(
        completed.stdout, encoding="utf-8"
    )
    (output / f"{name}.stderr.log").write_text(
        completed.stderr, encoding="utf-8"
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"{name} failed:\n{completed.stdout}\n{completed.stderr}"
        )
    return completed, wall_seconds


def parse_metrics(stdout: str, expected_mode: str) -> dict[str, int]:
    match = PASS_RE.search(stdout)
    if match is None or match.group("mode") != expected_mode:
        raise RuntimeError(f"missing {expected_mode} benchmark PASS marker")
    return {
        key: int(value)
        for key, value in match.groupdict().items()
        if key != "mode"
    }


def run(attempt: str) -> None:
    contract = read_contract()
    output = EVIDENCE_ROOT / attempt
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"attempt output is not fresh: {output}")
    output.mkdir(parents=True, exist_ok=True)
    expected, diversity = reference_outputs()
    expected_path = output / "expected.hex"
    expected_path.write_text(
        "".join(f"{value:02x}\n" for value in expected), encoding="ascii"
    )
    binary = compile_benchmark(LIVE_SHELL, output)

    legacy_run, legacy_wall = execute_case(
        binary, expected_path, output, "legacy", ["+LEGACY"]
    )
    fused_run, fused_wall = execute_case(
        binary, expected_path, output, "fused", ["+FUSED"]
    )
    reset_run, reset_wall = execute_case(
        binary, expected_path, output, "reset", ["+FUSED", "+RESET"]
    )
    corruption_run, corruption_wall = execute_case(
        binary, expected_path, output, "corruption", ["+FUSED", "+CORRUPT"]
    )
    if "ACE2_FUSED_QKV_RESET_PASS" not in reset_run.stdout:
        raise RuntimeError("missing reset recovery PASS marker")
    if "ACE2_FUSED_QKV_CORRUPTION_PASS" not in corruption_run.stdout:
        raise RuntimeError("missing corruption PASS marker")

    legacy = parse_metrics(legacy_run.stdout, "legacy")
    fused = parse_metrics(fused_run.stdout, "fused")
    status = (
        "PASS"
        if legacy["commands"] == 3
        and fused["commands"] == 1
        and legacy["outputs_checked"] == fused["outputs_checked"] == TOTAL_OUTPUTS
        and fused["activation_read_requests"] == K_SIZE // 16
        and fused["activation_read_requests"]
        < legacy["activation_read_requests"]
        and fused["memory_write_requests"] == legacy["memory_write_requests"]
        and (
            fused["simulator_cycles"] < legacy["simulator_cycles"]
            or fused_wall < legacy_wall
        )
        and legacy["backpressure_stalls"] > 0
        and fused["backpressure_stalls"] > 0
        else "FAIL"
    )
    result = {
        "schema_version": 1,
        "status": status,
        "contract": file_record(CONTRACT),
        "attempt": attempt,
        "equivalence": {
            "rtl_python_oracle": "PASS_BIT_EXACT",
            "legacy_fused_outputs": "PASS_SAME_ORACLE",
            "bytes_compared_per_mode": TOTAL_OUTPUTS,
            "distinct_values_q_k_v": diversity,
        },
        "legacy": {**legacy, "wall_seconds": legacy_wall},
        "fused": {**fused, "wall_seconds": fused_wall},
        "improvement": {
            "activation_read_reduction_percent": (
                1.0
                - fused["activation_read_requests"]
                / legacy["activation_read_requests"]
            )
            * 100.0,
            "total_read_reduction_percent": (
                1.0
                - fused["memory_read_requests"]
                / legacy["memory_read_requests"]
            )
            * 100.0,
            "command_reduction_percent": (
                1.0 - fused["commands"] / legacy["commands"]
            )
            * 100.0,
            "cycle_reduction_percent": (
                1.0
                - fused["simulator_cycles"] / legacy["simulator_cycles"]
            )
            * 100.0,
            "wall_time_reduction_percent": (
                1.0 - fused_wall / legacy_wall
            )
            * 100.0,
        },
        "focused_regressions": {
            "legacy_qkv": "PASS_BIT_EXACT",
            "fused_qkv": "PASS_BIT_EXACT",
            "backpressure": "PASS",
            "mid_command_reset": "PASS",
            "read_tag_corruption": "PASS",
            "wall_seconds": {
                "reset": reset_wall,
                "corruption": corruption_wall,
            },
        },
        "sources_and_constraint_hashes": source_records(),
        "limitations": contract["limitations"]
        + [
            (
                "Full-shell timing is unresolved until the separate "
                "mission-local SKY130/OpenSTA run."
            ),
            "Independent Reviewer acceptance is pending.",
        ],
    }
    write_atomic(output / "result.json", canonical_bytes(result))
    print(
        f"ACE2_FUSED_QKV_BENCHMARK_{status} "
        f"legacy_reads={legacy['memory_read_requests']} "
        f"fused_reads={fused['memory_read_requests']} "
        f"legacy_activation_reads={legacy['activation_read_requests']} "
        f"fused_activation_reads={fused['activation_read_requests']} "
        f"legacy_commands={legacy['commands']} fused_commands={fused['commands']} "
        f"legacy_cycles={legacy['simulator_cycles']} "
        f"fused_cycles={fused['simulator_cycles']}"
    )
    if status != "PASS":
        raise RuntimeError("fused QKV candidate did not meet frozen score policy")


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="action", required=True)
    subparsers.add_parser("freeze")
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--attempt", required=True)
    args = parser.parse_args()
    if args.action == "freeze":
        freeze()
    else:
        run(args.attempt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
