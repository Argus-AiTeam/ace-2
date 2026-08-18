#!/usr/bin/env python3
"""Run the accepted two-token Qwen command schedule against the live shell RTL."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import struct
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from huggingface_hub.constants import HF_HUB_CACHE

from tools.ace2_absolute_rope_online_attention_reference import (
    absolute_coefficients_q15,
)
from tools.model_hardware_contract import runtime_preflight


ROOT = Path(__file__).resolve().parents[1]
MISSION_ID = "run-full-qwen-command-schedule-runtime-v1"
REVISION = "060db6499f32faf8b98477b0a26969ef7d8b9987"
IMAGE_DIR = ROOT / "evidence/verification/build-full-qwen-packed-w4-metadata-image-v2"
IMAGE = IMAGE_DIR / "full_model_image.bin"
IMAGE_CONTRACT = IMAGE_DIR / "image_contract_v2.json"
IMAGE_AUDIT = IMAGE_DIR / "independent_postbuild_audit.json"
SCHEDULE_DIR = ROOT / "evidence/verification/rtl-full-qwen-autoregressive-integration-v1"
SCHEDULE = SCHEDULE_DIR / "command_schedule.json"
SCHEDULE_VALIDATION = SCHEDULE_DIR / "command_schedule_validation.json"
MODEL = (
    Path(HF_HUB_CACHE)
    / "models--Qwen--Qwen2.5-0.5B"
    / "snapshots"
    / REVISION
    / "model.safetensors"
)
DEFAULT_BINARY = ROOT / "build/verilator_full_qwen_runtime/Vace2_shell_runtime_harness"
DEFAULT_PACKAGE = ROOT / "build/full_qwen_runtime/command_schedule.bin"
DEFAULT_OUTPUT = ROOT / f"evidence/verification/{MISSION_ID}/full"

EXPECTED_IMAGE_BYTES = 254_421_520
EXPECTED_IMAGE_SHA256 = "e24e0365e9fad5df2efe3e40df12e3f89f951f37c83449cb40e7d18fb614eafb"
EXPECTED_IMAGE_CONTRACT_SHA256 = "32b0d2279fa42159f684ccc49916ed177ca99299031392eee3fbcc23baf28d48"
EXPECTED_SCHEDULE_BYTES = 7_530_678
EXPECTED_SCHEDULE_SHA256 = "838b2c019a6028a92ffef8b9cc087cdcb616f33f60a20c6b24cb33aed37bb002"
EXPECTED_MODEL_BYTES = 988_097_824
EXPECTED_MODEL_SHA256 = "88c142557820ccad55bb59756bfcfcf891de9cc6202816bd346445188a0ed342"
EXPECTED_SHELL_SHA256 = "3bb8caab4f06e6be9b170b5b3d91cb89b237715132e52507cd60f0514c61ab30"
EXPECTED_COMMANDS = 13_914
EXPECTED_TOKEN_COUNTS = {"0": 6117, "1": 7797}
MODEL_CONTRACT_ID = "qwen2.5-0.5b"
MODEL_MAX_SEQUENCE_POSITIONS = 32_768
MODEL_KV_BYTES_PER_TOKEN_PER_LAYER = 272

OPERATORS = [
    "input_rmsnorm",
    "q_proj",
    "k_proj",
    "v_proj",
    "rope_q",
    "rope_k",
    "kv_write",
    "attention_score",
    "softmax",
    "attention_value",
    "attention_compose",
    "o_proj",
    "attention_residual_add",
    "post_attention_rmsnorm",
    "mlp_gate_proj",
    "mlp_up_proj",
    "silu_gate",
    "mlp_down_proj",
    "mlp_residual_add",
    "final_rmsnorm",
    "lm_head_tile",
]
OPERATOR_IDS = {name: index for index, name in enumerate(OPERATORS)}
HEADER = struct.Struct("<8sIIIIIIQ32s32s32s256s")
COMMAND = struct.Struct("<IHBBBBHHHHHhhiQQQQQ")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path, expected_bytes: int | None = None, expected_sha256: str | None = None) -> dict[str, Any]:
    require(path.is_file(), f"missing required input: {path}")
    size = path.stat().st_size
    digest = sha256_file(path)
    if expected_bytes is not None:
        require(size == expected_bytes, f"byte count changed: {path}")
    if expected_sha256 is not None:
        require(digest == expected_sha256, f"SHA-256 changed: {path}")
    return {"bytes": size, "sha256": digest}


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
    directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def embedding_tensor_offset(path: Path) -> tuple[int, list[int]]:
    with path.open("rb") as handle:
        header_length_raw = handle.read(8)
        require(len(header_length_raw) == 8, "safetensors header length is truncated")
        header_length = int.from_bytes(header_length_raw, "little")
        header_raw = handle.read(header_length)
    require(len(header_raw) == header_length, "safetensors header is truncated")
    header = json.loads(header_raw)
    record = header.get("model.embed_tokens.weight")
    require(isinstance(record, dict), "embedding tensor is absent from raw safetensors")
    require(record.get("dtype") == "BF16", "embedding tensor dtype changed")
    require(record.get("shape") == [151936, 896], "embedding tensor shape changed")
    offsets = record.get("data_offsets")
    require(
        isinstance(offsets, list)
        and len(offsets) == 2
        and offsets[1] - offsets[0] == 151936 * 896 * 2,
        "embedding tensor offsets changed",
    )
    return 8 + header_length + int(offsets[0]), list(map(int, record["shape"]))


def rope_runtime_records(positions: tuple[int, int] = (0, 1)) -> bytes:
    payload = bytearray()
    for position in positions:
        cosine, sine = absolute_coefficients_q15(position)
        require(len(cosine) == 32 and len(sine) == 32, "RoPE coefficient geometry changed")
        payload.extend(struct.pack("<32h32h", *cosine, *sine))
    require(len(payload) == 256, "two-position RoPE runtime preload changed")
    return bytes(payload)


def model_hardware_contract_preflight(
    embedding_shape: list[int],
) -> dict[str, Any]:
    return runtime_preflight(
        model_id=MODEL_CONTRACT_ID,
        embedding_shape=embedding_shape,
        max_sequence_positions=MODEL_MAX_SEQUENCE_POSITIONS,
        kv_bytes_per_token_per_layer=MODEL_KV_BYTES_PER_TOKEN_PER_LAYER,
    )


def validate_inputs(
    image: Path,
    image_contract: Path,
    image_audit: Path,
    schedule_path: Path,
    schedule_validation_path: Path,
    model: Path,
) -> tuple[dict[str, Any], dict[str, Any], int, list[int]]:
    records = {
        "sealed_image": file_record(image, EXPECTED_IMAGE_BYTES, EXPECTED_IMAGE_SHA256),
        "image_contract": file_record(
            image_contract, expected_sha256=EXPECTED_IMAGE_CONTRACT_SHA256
        ),
        "independent_postbuild_audit": file_record(image_audit),
        "accepted_schedule": file_record(
            schedule_path, EXPECTED_SCHEDULE_BYTES, EXPECTED_SCHEDULE_SHA256
        ),
        "schedule_validation": file_record(schedule_validation_path),
        "raw_safetensors": file_record(model, EXPECTED_MODEL_BYTES, EXPECTED_MODEL_SHA256),
        "live_shell": file_record(
            ROOT / "rtl/ace2_shell.sv", expected_sha256=EXPECTED_SHELL_SHA256
        ),
    }
    audit = read_json(image_audit)
    require(
        audit["checks"]["artifact_identity"]["status"] == "PASS"
        and audit["checks"]["region_map"]["status"] == "PASS"
        and audit["checks"]["schedule_coverage"]["status"] == "PASS",
        "accepted post-build audit no longer reports identity/region/schedule PASS",
    )
    audited_image = audit["checks"]["artifact_identity"]["artifacts"]["full_model_image.bin"]
    require(audited_image == records["sealed_image"], "audit no longer binds the sealed image")
    contract = read_json(image_contract)
    require(
        contract["contract"]["full_image"]["bytes"] == EXPECTED_IMAGE_BYTES
        and contract["contract"]["schema_version"] == 2,
        "image contract geometry changed",
    )
    schedule = read_json(schedule_path)
    validation = read_json(schedule_validation_path)
    require(schedule["command_count"] == EXPECTED_COMMANDS, "accepted command count changed")
    require(schedule["command_count_per_token"] == EXPECTED_TOKEN_COUNTS, "token command counts changed")
    require(len(schedule["commands"]) == EXPECTED_COMMANDS, "command array length changed")
    require(
        validation == {
            "first_command": "input_rmsnorm",
            "kv_positions_per_layer": [0, 1],
            "last_command": "lm_head_tile",
            "layers": 24,
            "lm_head_tiles_per_token": 4748,
            "status": "PASS",
            "validated_commands": EXPECTED_COMMANDS,
        },
        "accepted schedule validation changed",
    )
    for ordinal, command in enumerate(schedule["commands"]):
        require(command["ordinal"] == ordinal, f"schedule ordinal changed at {ordinal}")
        require(command["completion_tag"] == (ordinal & 0xFFFF), f"completion tag changed at {ordinal}")
        require(command["operator"] in OPERATOR_IDS, f"unknown operator at {ordinal}")
    embedding_offset, embedding_shape = embedding_tensor_offset(model)
    records["model_hardware_contract"] = model_hardware_contract_preflight(
        embedding_shape
    )
    return schedule, records, embedding_offset, embedding_shape


def build_package(
    schedule: dict[str, Any],
    output: Path,
    embedding_offset: int,
    embedding_shape: list[int],
) -> dict[str, Any]:
    host_steps = schedule["host_steps"]
    require(host_steps[0]["operation"] == "embedding_row_preload", "seed embedding host step changed")
    seed_token = int(host_steps[0]["token_id"])
    rope_records = rope_runtime_records()
    raw = bytearray(
        HEADER.pack(
            b"ACE2RT1\0",
            1,
            COMMAND.size,
            len(schedule["commands"]),
            seed_token,
            int(embedding_shape[0]),
            int(embedding_shape[1]),
            embedding_offset,
            bytes.fromhex(EXPECTED_SCHEDULE_SHA256),
            bytes.fromhex(EXPECTED_IMAGE_SHA256),
            bytes.fromhex(EXPECTED_MODEL_SHA256),
            rope_records,
        )
    )
    for ordinal, command in enumerate(schedule["commands"]):
        query_head = -1 if command.get("query_head") is None else int(command["query_head"])
        context_token = -1 if command.get("context_token") is None else int(command["context_token"])
        vocab_tile = -1 if command.get("vocab_tile") is None else int(command["vocab_tile"])
        raw.extend(
            COMMAND.pack(
                ordinal,
                int(command["token_step"]),
                int(command["layer_id"]),
                OPERATOR_IDS[command["operator"]],
                int(command["opcode"]),
                int(command["flags"]),
                int(command["m"]),
                int(command["n"]),
                int(command["k"]),
                int(command["sequence_position"]),
                int(command["completion_tag"]),
                query_head,
                context_token,
                vocab_tile,
                int(command["src0_addr"]),
                int(command["src1_addr"]),
                int(command["dst_addr"]),
                int(command["scale_addr"]),
                int(command["scratch_addr"]),
            )
        )
    expected_bytes = HEADER.size + EXPECTED_COMMANDS * COMMAND.size
    require(len(raw) == expected_bytes, "runtime package byte count changed")
    write_atomic(output, bytes(raw))
    return {
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "record_bytes": COMMAND.size,
        "commands": EXPECTED_COMMANDS,
        "seed_token_id": seed_token,
        "embedding_tensor_absolute_offset": embedding_offset,
        "rope_positions": [0, 1],
        "rope_preload_sha256": hashlib.sha256(rope_records).hexdigest(),
    }


def source_records() -> dict[str, Any]:
    paths = [
        Path(__file__),
        ROOT / "tools/model_hardware_contract.py",
        ROOT / "verification/verilator/ace2_shell_runtime_harness.sv",
        ROOT / "verification/verilator/ace2_shell_runtime_main.cpp",
    ]
    return {str(path.relative_to(ROOT)): file_record(path) for path in paths}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, default=IMAGE)
    parser.add_argument("--image-contract", type=Path, default=IMAGE_CONTRACT)
    parser.add_argument("--image-audit", type=Path, default=IMAGE_AUDIT)
    parser.add_argument("--schedule", type=Path, default=SCHEDULE)
    parser.add_argument("--schedule-validation", type=Path, default=SCHEDULE_VALIDATION)
    parser.add_argument("--model-safetensors", type=Path, default=MODEL)
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--package", type=Path, default=DEFAULT_PACKAGE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--stop-after", type=int, default=0)
    parser.add_argument("--timeout-cycles", type=int, default=100_000_000)
    parser.add_argument(
        "--generated-at-utc",
        help="Explicit YYYY-MM-DDTHH:MM:SSZ provenance time",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()

    schedule, inputs, embedding_offset, embedding_shape = validate_inputs(
        args.image.resolve(),
        args.image_contract.resolve(),
        args.image_audit.resolve(),
        args.schedule.resolve(),
        args.schedule_validation.resolve(),
        args.model_safetensors.resolve(),
    )
    package_record = build_package(
        schedule, args.package.resolve(), embedding_offset, embedding_shape
    )
    if args.prepare_only:
        print(
            "ACE2_RUNTIME_PREPARE_PASS "
            f"commands={package_record['commands']} package_sha256={package_record['sha256']}"
        )
        return 0

    binary = args.binary.resolve()
    require(binary.is_file(), f"missing Verilator runtime binary: {binary}")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    provenance_path = output / "provenance.json"
    failure_path = output / "first_failure.json"
    if args.resume and failure_path.is_file():
        archive_dir = output / "recovered_runtime_failures"
        archive_dir.mkdir(parents=True, exist_ok=True)
        archive_index = 0
        while (archive_dir / f"attempt-{archive_index:04d}.json").exists():
            archive_index += 1
        os.replace(failure_path, archive_dir / f"attempt-{archive_index:04d}.json")
    generated_at_utc = args.generated_at_utc or utc_now()
    require(
        re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", generated_at_utc)
        is not None,
        "--generated-at-utc must use YYYY-MM-DDTHH:MM:SSZ",
    )
    provenance: dict[str, Any] = {
        "schema_version": 1,
        "mission_id": MISSION_ID,
        "generated_at_utc": generated_at_utc,
        "classification": "raw_safetensors_embedding_preload_plus_sealed_image_rtl_runtime",
        "inputs": {
            "evidence/verification/build-full-qwen-packed-w4-metadata-image-v2/full_model_image.bin": inputs["sealed_image"],
            "evidence/verification/build-full-qwen-packed-w4-metadata-image-v2/image_contract_v2.json": inputs["image_contract"],
            "evidence/verification/build-full-qwen-packed-w4-metadata-image-v2/independent_postbuild_audit.json": inputs["independent_postbuild_audit"],
            "evidence/verification/rtl-full-qwen-autoregressive-integration-v1/command_schedule.json": inputs["accepted_schedule"],
            "evidence/verification/rtl-full-qwen-autoregressive-integration-v1/command_schedule_validation.json": inputs["schedule_validation"],
            f"hf-cache://Qwen/Qwen2.5-0.5B@{REVISION}/model.safetensors": inputs["raw_safetensors"],
            "rtl/ace2_shell.sv": inputs["live_shell"],
        },
        "runtime_sources": source_records(),
        "runtime_package": package_record,
        "model_hardware_contract": inputs["model_hardware_contract"],
        "execution": {
            "requested_stop_after": args.stop_after or EXPECTED_COMMANDS,
            "resume": args.resume,
            "timeout_cycles_per_command": args.timeout_cycles,
            "continuous_full_schedule": (args.stop_after == 0 or args.stop_after == EXPECTED_COMMANDS),
            "binary_label": "build/verilator_full_qwen_runtime/Vace2_shell_runtime_harness",
        },
        "tool_versions": {
            "python": sys.version.split()[0],
            "verilator": subprocess.run(
                ["verilator", "--version"], check=True, text=True, capture_output=True
            ).stdout.strip(),
        },
        "scope_guards": {
            "model_constructed": False,
            "model_called": False,
            "recalibration": False,
            "image_regenerated": False,
            "schedule_changed": False,
            "synthesizable_rtl_changed": False,
            "synthesis_opensta_ppa": False,
        },
        "status": "RUNNING",
    }
    write_atomic(provenance_path, canonical_bytes(provenance))

    command = [
        str(binary),
        "--package",
        str(args.package.resolve()),
        "--image",
        str(args.image.resolve()),
        "--model",
        str(args.model_safetensors.resolve()),
        "--output",
        str(output),
        "--stop-after",
        str(args.stop_after),
        "--timeout-cycles",
        str(args.timeout_cycles),
    ]
    if args.resume:
        command.append("--resume")
    completed = subprocess.run(command, check=False)
    provenance["completed_at_utc"] = args.generated_at_utc or utc_now()
    provenance["runtime_exit_code"] = completed.returncode
    summary_path = output / "summary.json"
    if summary_path.is_file():
        provenance["summary"] = {
            **file_record(summary_path),
            "status": read_json(summary_path).get("status"),
        }
    if failure_path.is_file():
        provenance["first_failure"] = {
            **file_record(failure_path),
            "category": read_json(failure_path).get("category"),
        }
    provenance["status"] = "PASS" if completed.returncode == 0 else "STOPPED"
    write_atomic(provenance_path, canonical_bytes(provenance))
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
