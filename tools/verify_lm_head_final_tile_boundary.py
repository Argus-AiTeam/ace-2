#!/usr/bin/env python3
"""Mechanically audit the final dense-W4 LM-head tile boundary."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MISSION_ID = "repair-full-qwen-final-lm-head-tile-boundary-v1"
REVISION = "060db6499f32faf8b98477b0a26969ef7d8b9987"
EXPECTED_PRECHANGE_TREE = "3312e4f634f363bda495ce98633a4f618eeef5fb224d952ecc72d3052a30b46b"
EXPECTED_PRECHANGE_SHELL = "65fe2890e8a1e8b5cfade04f9ce7240e98a27f39416d05f8557e1a56e82f73e0"
EXPECTED_IMAGE = "e24e0365e9fad5df2efe3e40df12e3f89f951f37c83449cb40e7d18fb614eafb"
EXPECTED_SCHEDULE = "838b2c019a6028a92ffef8b9cc087cdcb616f33f60a20c6b24cb33aed37bb002"
EXPECTED_MODEL = "88c142557820ccad55bb59756bfcfcf891de9cc6202816bd346445188a0ed342"
EXPECTED_JOURNAL = "ff9ce2fa371ae11ef244786b41790c4ca73ab57d39ebf1fe46c0f214b39a67ca"
VOCAB = 151_936
HIDDEN = 896
TILE_OUTPUTS = 32
MAC_LANES = 4
MEM_BEAT_BYTES = 16
PACKED_BYTES_PER_OUTPUT = HIDDEN // 2
TILE_BYTES = TILE_OUTPUTS * PACKED_BYTES_PER_OUTPUT
TILES = VOCAB // TILE_OUTPUTS
LM_HEAD_BASE = 4_473_880_576
PACKED_REGION_END = 4_541_947_904
SELECTED_TILES = (0, TILES // 2, TILES - 3, TILES - 2, TILES - 1)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def file_record(path: Path, expected: str | None = None) -> dict[str, Any]:
    require(path.is_file(), f"missing artifact: {path}")
    digest = sha256_file(path)
    if expected is not None:
        require(digest == expected, f"SHA-256 changed: {path}")
    return {"bytes": path.stat().st_size, "sha256": digest}


def read_safetensors_header(path: Path) -> tuple[dict[str, Any], int]:
    with path.open("rb") as handle:
        header_length_raw = handle.read(8)
        require(len(header_length_raw) == 8, "safetensors header length is truncated")
        header_length = int.from_bytes(header_length_raw, "little")
        header_raw = handle.read(header_length)
    require(len(header_raw) == header_length, "safetensors header is truncated")
    return json.loads(header_raw), 8 + header_length


def raw_tensor_contract(model: Path) -> dict[str, Any]:
    header, data_base = read_safetensors_header(model)
    tensor = header.get("model.embed_tokens.weight")
    require(isinstance(tensor, dict), "tied embedding tensor is absent")
    require(tensor.get("dtype") == "BF16", "tied tensor dtype changed")
    require(tensor.get("shape") == [VOCAB, HIDDEN], "tied tensor shape changed")
    offsets = tensor.get("data_offsets")
    require(isinstance(offsets, list) and len(offsets) == 2, "tied tensor offsets changed")
    tensor_bytes = int(offsets[1]) - int(offsets[0])
    require(tensor_bytes == VOCAB * HIDDEN * 2, "raw tied tensor byte count changed")
    raw_last_tile_bytes = TILE_OUTPUTS * HIDDEN * 2
    last_tile_offset = data_base + int(offsets[1]) - raw_last_tile_bytes
    with model.open("rb") as handle:
        handle.seek(last_tile_offset)
        last_tile = handle.read(raw_last_tile_bytes)
    require(len(last_tile) == raw_last_tile_bytes, "raw tied tensor last tile is truncated")
    return {
        "tensor_key": "model.embed_tokens.weight",
        "dtype": "BF16",
        "shape": [VOCAB, HIDDEN],
        "tensor_bytes": tensor_bytes,
        "raw_last_tile_bytes": raw_last_tile_bytes,
        "raw_last_tile_sha256": sha256_bytes(last_tile),
        "derived_packed_w4_bytes": VOCAB * HIDDEN // 2,
        "derived_packed_last_tile_bytes": TILE_BYTES,
    }


def image_contract(image: Path, manifest_path: Path, contract_path: Path) -> dict[str, Any]:
    manifest = read_json(manifest_path)
    contract = read_json(contract_path)["contract"]
    lm_head = next(item for item in manifest["linear_tensors"] if item["name"] == "lm_head")
    packed = lm_head["packed_w4"]
    require(packed["memory_addr"] == LM_HEAD_BASE, "LM-head packed base changed")
    require(packed["bytes"] == VOCAB * HIDDEN // 2, "LM-head packed byte count changed")
    require(packed["memory_addr"] + packed["bytes"] == PACKED_REGION_END, "LM-head end changed")
    packed_region = next(item for item in contract["full_image"]["regions"] if item["name"] == "packed_w4")
    require(packed_region["memory_end"] == PACKED_REGION_END, "packed-W4 region end changed")
    require(packed["file_offset"] + packed["bytes"] == packed_region["bytes"], "LM-head is not terminal in packed-W4 file region")

    selected: dict[str, Any] = {}
    with image.open("rb") as handle:
        for tile in SELECTED_TILES:
            offset = packed["file_offset"] + tile * TILE_BYTES
            handle.seek(offset)
            raw = handle.read(TILE_BYTES)
            require(len(raw) == TILE_BYTES, f"image tile {tile} is truncated")
            selected[str(tile)] = {
                "file_offset": offset,
                "memory_base": LM_HEAD_BASE + tile * TILE_BYTES,
                "bytes": len(raw),
                "sha256": sha256_bytes(raw),
                "terminal_16_bytes_hex": raw[-16:].hex(),
            }
    require(selected[str(TILES - 1)]["memory_base"] + TILE_BYTES == PACKED_REGION_END, "last image tile does not end at region boundary")
    return {
        "manifest_lm_head": lm_head,
        "packed_region": packed_region,
        "selected_tiles": selected,
        "actual_image_bytes": image.stat().st_size,
    }


def schedule_contract(schedule_path: Path) -> dict[str, Any]:
    schedule = read_json(schedule_path)
    selected: dict[str, Any] = {}
    for token_step in (0, 1):
        commands = [
            command for command in schedule["commands"]
            if command["operator"] == "lm_head_tile" and command["token_step"] == token_step
        ]
        require(len(commands) == TILES, f"token {token_step} LM-head tile count changed")
        for tile, command in enumerate(commands):
            expected_base = LM_HEAD_BASE + tile * TILE_BYTES
            require(command["vocab_tile"] == tile, f"token {token_step} tile index changed at {tile}")
            require(command["n"] == TILE_OUTPUTS and command["k"] == HIDDEN, f"token {token_step} tile shape changed at {tile}")
            require(command["src1_addr"] == expected_base, f"token {token_step} tile base changed at {tile}")
            require(expected_base + TILE_BYTES <= PACKED_REGION_END, f"token {token_step} tile exceeds packed region at {tile}")
            if tile in SELECTED_TILES:
                selected[f"token{token_step}_tile{tile}"] = {
                    "ordinal": command["ordinal"],
                    "src1_addr": command["src1_addr"],
                    "scale_addr": command["scale_addr"],
                    "tile_end": command["src1_addr"] + TILE_BYTES,
                }
    return {
        "tile_count_per_token": TILES,
        "tile_stride_bytes": TILE_BYTES,
        "selected_tiles": selected,
        "ordinal_6116": selected[f"token0_tile{TILES - 1}"],
    }


def rtl_tree(prechange_shell: Path, live_shell: Path) -> dict[str, Any]:
    prechange_text = prechange_shell.read_text(encoding="utf-8")
    live_text = live_shell.read_text(encoding="utf-8")
    require(sha256_file(prechange_shell) == EXPECTED_PRECHANGE_SHELL, "prechange shell snapshot changed")
    require("PROJ_GROUPS_PER_STORAGE_BEAT = LANES / PROJ_MAC_LANES" in prechange_text, "prechange shared storage geometry is absent")
    require("PROJ_HIDDEN_WEIGHT_BYTES_PER_OUTPUT = (HIDDEN_SIZE / LANES) * 16" in prechange_text, "prechange padded row stride is absent")
    require("PROJ_WGT_GROUPS_PER_STORAGE_BEAT = (LANES * ACT_WIDTH) / (PROJ_MAC_LANES * 4)" in live_text, "live dense-W4 beat geometry is absent")
    require("PROJ_HIDDEN_WEIGHT_BYTES_PER_OUTPUT = HIDDEN_SIZE / 2" in live_text, "live dense-W4 row stride is absent")
    require("proj_wgt_group_storage_idx_ext_w = proj_group_idx_ext_w >> PROJ_WGT_GROUP_STORAGE_SHIFT" in live_text, "live weight group indexing is absent")

    rows = []
    prechange_rows = []
    for path in sorted((ROOT / "rtl").glob("**/*.sv")):
        relative = path.relative_to(ROOT).as_posix()
        live_digest = sha256_file(path)
        rows.append(f"{live_digest}  {relative}\n")
        old_digest = EXPECTED_PRECHANGE_SHELL if relative == "rtl/ace2_shell.sv" else live_digest
        prechange_rows.append(f"{old_digest}  {relative}\n")
    current_tree = sha256_bytes("".join(rows).encode())
    reconstructed_prechange_tree = sha256_bytes("".join(prechange_rows).encode())
    require(reconstructed_prechange_tree == EXPECTED_PRECHANGE_TREE, "prechange RTL tree reconstruction changed")

    groups = HIDDEN // MAC_LANES
    old_groups_per_beat = 16 // MAC_LANES
    new_groups_per_beat = 32 // MAC_LANES
    old_stride = HIDDEN
    new_stride = HIDDEN // 2
    old_high_water = (TILE_OUTPUTS - 1) * old_stride + ((groups - 1) // old_groups_per_beat) * MEM_BEAT_BYTES
    new_high_water = (TILE_OUTPUTS - 1) * new_stride + ((groups - 1) // new_groups_per_beat) * MEM_BEAT_BYTES
    first_end_access = next(
        (out_index, group_index)
        for out_index in range(TILE_OUTPUTS)
        for group_index in range(groups)
        if out_index * old_stride + (group_index // old_groups_per_beat) * MEM_BEAT_BYTES >= TILE_BYTES
    )
    require(first_end_access == (16, 0), "prechange first region-end access derivation changed")
    require(new_high_water == TILE_BYTES - MEM_BEAT_BYTES, "live high-water is not the last in-region beat")
    read_beats = TILE_OUTPUTS * (groups + groups + 1)
    write_beats = TILE_OUTPUTS // MEM_BEAT_BYTES
    return {
        "prechange_shell": file_record(prechange_shell, EXPECTED_PRECHANGE_SHELL),
        "live_shell": file_record(live_shell),
        "prechange_tree_sha256": reconstructed_prechange_tree,
        "live_tree_sha256": current_tree,
        "changed_rtl_files": ["rtl/ace2_shell.sv"],
        "prechange": {
            "weight_groups_per_128b_beat": old_groups_per_beat,
            "weight_row_stride_bytes": old_stride,
            "high_water_offset": old_high_water,
            "first_region_end_access": {
                "output_index": first_end_access[0],
                "group_index": first_end_access[1],
                "offset": TILE_BYTES,
                "address": PACKED_REGION_END,
            },
        },
        "corrected": {
            "weight_groups_per_128b_beat": new_groups_per_beat,
            "weight_row_stride_bytes": new_stride,
            "high_water_offset": new_high_water,
            "high_water_address_for_tile_4747": LM_HEAD_BASE + (TILES - 1) * TILE_BYTES + new_high_water,
            "tile_end_address": PACKED_REGION_END,
            "read_beats": read_beats,
            "write_beats": write_beats,
            "access_at_tile_end": False,
        },
    }


def runtime_contract(
    runtime_dir: Path,
    source_journal: Path,
    binary: Path,
    package: Path,
) -> dict[str, Any]:
    journal = runtime_dir / "progress.journal"
    commands_path = runtime_dir / "commands.jsonl"
    progress_path = runtime_dir / "progress.json"
    summary_path = runtime_dir / "summary.json"
    stdout_path = runtime_dir / "runtime.stdout.log"
    require(not (runtime_dir / "first_failure.json").exists(), "runtime emitted first_failure.json")
    source_prefix = source_journal.read_bytes()
    with journal.open("rb") as handle:
        require(handle.read(len(source_prefix)) == source_prefix, "continued journal does not preserve the ordinal-6116 source prefix")

    records = [json.loads(line) for line in commands_path.read_text(encoding="utf-8").splitlines()]
    require(len(records) == 13_914, "completed command JSONL length changed")
    require(all(record["ordinal"] == ordinal for ordinal, record in enumerate(records)), "completed command ordinals are discontinuous")
    require(not any(record["completion_error"] for record in records), "runtime contains a completion error")
    selected_ordinals = (6116, 9166, 11540, 13911, 13912, 13913)
    selected: dict[str, Any] = {}
    for ordinal in selected_ordinals:
        record = records[ordinal]
        require(record["operator"] == "lm_head_tile", f"selected ordinal {ordinal} is not LM head")
        require(record["read_beats"] == 14_368, f"selected ordinal {ordinal} read count changed")
        require(record["write_beats"] == 2, f"selected ordinal {ordinal} write count changed")
        selected[str(ordinal)] = record
    require(records[6116]["generated_token_after"] == 0, "token-0 argmax did not complete at ordinal 6116")
    require(records[9166]["token_step"] == 1, "second-token LM head did not begin at ordinal 9166")
    require(records[13913]["generated_token_after"] == 0, "token-1 argmax did not complete at ordinal 13913")

    progress = read_json(progress_path)
    summary = read_json(summary_path)
    require(progress["status"] == "PASS" and progress["next_ordinal"] == 13_914, "runtime progress is incomplete")
    require(summary["status"] == "PASS" and summary["commands_completed"] == 13_914, "runtime summary is incomplete")
    require(progress["generated_token_ids"] == [0, 0], "runtime feedback tokens changed")
    require(summary["generated_token_ids"] == [0, 0], "runtime summary tokens changed")
    require(progress["resume_count"] == 1 and progress["resume_warmup_cycles"] == 0, "ordinal-6116 resume semantics changed")
    require("ACE2_RUNTIME_PASS commands=13914" in stdout_path.read_text(encoding="utf-8"), "runtime PASS marker is absent")
    return {
        "status": "PASS",
        "source_journal": file_record(source_journal, EXPECTED_JOURNAL),
        "source_journal_preserved_as_exact_prefix": True,
        "binary": file_record(binary),
        "runtime_package": file_record(package),
        "artifacts": {
            "progress_journal": file_record(journal),
            "commands_jsonl": file_record(commands_path),
            "progress_json": file_record(progress_path),
            "summary_json": file_record(summary_path),
            "stdout_log": file_record(stdout_path),
        },
        "selected_lm_head_commands": selected,
        "commands_completed": summary["commands_completed"],
        "simulator_cycles": summary["simulator_cycles"],
        "generated_token_ids": summary["generated_token_ids"],
        "first_failure_present": False,
        "scope_guards": {
            "model_constructed": False,
            "model_called": False,
            "recalibration": False,
            "image_changed": False,
            "schedule_changed": False,
            "ppa_executed": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--image", type=Path, default=ROOT / "evidence/verification/build-full-qwen-packed-w4-metadata-image-v2/full_model_image.bin")
    parser.add_argument("--manifest", type=Path, default=ROOT / "evidence/verification/build-full-qwen-packed-w4-metadata-image-v2/manifest.json")
    parser.add_argument("--image-contract", type=Path, default=ROOT / "evidence/verification/build-full-qwen-packed-w4-metadata-image-v2/image_contract_v2.json")
    parser.add_argument("--schedule", type=Path, default=ROOT / "evidence/verification/rtl-full-qwen-autoregressive-integration-v1/command_schedule.json")
    parser.add_argument("--journal", type=Path, default=ROOT / "evidence/verification/rtl-silu-packed-int8-width-adapter-repair-v1/runtime-continuation/progress.journal")
    parser.add_argument("--prechange-shell", type=Path, default=ROOT / "evidence/verification/repair-full-qwen-final-lm-head-tile-boundary-v1/prechange/ace2_shell.sv")
    parser.add_argument("--live-shell", type=Path, default=ROOT / "rtl/ace2_shell.sv")
    parser.add_argument("--output", type=Path, default=ROOT / "evidence/verification/repair-full-qwen-final-lm-head-tile-boundary-v1/contract_audit.json")
    parser.add_argument("--runtime-dir", type=Path)
    parser.add_argument("--binary", type=Path, default=ROOT / "build/verilator_full_qwen_runtime/Vace2_shell_runtime_harness")
    parser.add_argument("--package", type=Path, default=ROOT / "build/full_qwen_runtime/command_schedule.bin")
    parser.add_argument("--runtime-report", type=Path, default=ROOT / "evidence/verification/repair-full-qwen-final-lm-head-tile-boundary-v1/runtime_validation.json")
    args = parser.parse_args()

    artifacts = {
        "sealed_image": file_record(args.image, EXPECTED_IMAGE),
        "accepted_schedule": file_record(args.schedule, EXPECTED_SCHEDULE),
        "raw_safetensors": file_record(args.model, EXPECTED_MODEL),
        "continuation_journal_through_6115": file_record(args.journal, EXPECTED_JOURNAL),
    }
    report = {
        "schema_version": 1,
        "mission_id": MISSION_ID,
        "status": "PASS",
        "classification": "RTL_EXTRA_READ_DENSE_W4_ADDRESSING",
        "artifacts": artifacts,
        "raw_safetensors_contract": raw_tensor_contract(args.model),
        "image_contract": image_contract(args.image, args.manifest, args.image_contract),
        "schedule_contract": schedule_contract(args.schedule),
        "rtl_contract": rtl_tree(args.prechange_shell, args.live_shell),
        "scope_guards": {
            "model_constructed": False,
            "model_called": False,
            "recalibration": False,
            "image_changed": False,
            "schedule_changed": False,
            "ppa_executed": False,
        },
    }
    write_json(args.output, report)
    print(
        "ACE2_LM_HEAD_BOUNDARY_AUDIT_PASS "
        f"classification={report['classification']} "
        f"tile_bytes={TILE_BYTES} "
        f"read_beats={report['rtl_contract']['corrected']['read_beats']} "
        f"write_beats={report['rtl_contract']['corrected']['write_beats']} "
        f"high_water={report['rtl_contract']['corrected']['high_water_address_for_tile_4747']} "
        f"region_end={PACKED_REGION_END} "
        f"rtl_tree={report['rtl_contract']['live_tree_sha256']}"
    )
    if args.runtime_dir is not None:
        runtime = runtime_contract(
            args.runtime_dir.resolve(),
            args.journal.resolve(),
            args.binary.resolve(),
            args.package.resolve(),
        )
        write_json(args.runtime_report, runtime)
        print(
            "ACE2_LM_HEAD_RUNTIME_VALIDATION_PASS "
            f"commands={runtime['commands_completed']} "
            f"cycles={runtime['simulator_cycles']} "
            f"tokens={','.join(map(str, runtime['generated_token_ids']))} "
            f"journal_sha256={runtime['artifacts']['progress_journal']['sha256']}"
        )


if __name__ == "__main__":
    main()
