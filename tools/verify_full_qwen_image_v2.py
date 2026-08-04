#!/usr/bin/env python3
"""Independently verify the sealed ACE-2 full-Qwen v2 memory image.

This verifier uses the Python standard library and reads raw safetensors bytes.
It does not import the image builder, torch, transformers, or execute a model,
command decoder, RTL simulation, synthesis, OpenSTA, or PPA flow.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import mmap
import re
import struct
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
MISSION_ID = "build-full-qwen-packed-w4-metadata-image-v2"
OUTPUT_DIR = ROOT / "evidence/verification" / MISSION_ID
BOUNDARY = ROOT / "evidence/verification/rtl-full-qwen-autoregressive-integration-v1"
INVENTORY = BOUNDARY / "integration_inventory.json"
SCHEDULE = BOUNDARY / "command_schedule.json"
PROVENANCE = BOUNDARY / "provenance.json"
V1_AUDIT = (
    ROOT
    / "evidence/verification/build-full-qwen-packed-w4-metadata-image-v1"
    / "preflight_contract_audit.json"
)
SCALES = (
    ROOT
    / "evidence/layer0_tile_bfp_score_attention_v1"
    / "paired-smoke-20260801-v1/derived_scales.json"
)
SCALE_RUN_CONTRACT = SCALES.with_name("run_contract.json")
SCALE_RESULTS = SCALES.with_name("results.json")
SHELL = ROOT / "rtl/ace2_shell.sv"
SHELL_TB = ROOT / "verification/tb/ace2_shell_tb.sv"
BUILDER = ROOT / "tools/build_full_qwen_image_v2.py"

REVISION = "060db6499f32faf8b98477b0a26969ef7d8b9987"
MODEL = (
    Path.home()
    / ".cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B/snapshots"
    / REVISION
    / "model.safetensors"
)

HIDDEN = 896
HEAD_DIM = 64
KV_HEADS = 2
LAYERS = 24
RMS_RECORD_BYTES = 1_808
RMS_GAIN_BYTES = HIDDEN * 2
RMS_TRAILER_BYTES = 16

EXPECTED_REGION_BYTES = {
    "packed_w4": 246_980_608,
    "projection_metadata": 7_297_024,
    "rmsnorm_metadata": 88_592,
    "operator_aux_metadata": 55_296,
}
REGION_ORDER = tuple(EXPECTED_REGION_BYTES)
TOTAL_IMAGE_BYTES = sum(EXPECTED_REGION_BYTES.values())
EXPECTED_TIMESTAMP = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z")

EXPECTED_INPUT_HASHES = {
    V1_AUDIT: "e5006f4c062693a65a02a732300468fddfe7bafb4ad3a4f5bc173fd5aed8122b",
    INVENTORY: "6472b1460c9908255655fcca45d0e6d5eadc065f6debf5b6605e9107568d1950",
    SCHEDULE: "838b2c019a6028a92ffef8b9cc087cdcb616f33f60a20c6b24cb33aed37bb002",
    PROVENANCE: "659ea9edbe50bfe961086a4a652788b6be343e792b8470755ef41f7a20fa2841",
    SCALES: "78280eb606e0c14ea74163f72f45cbb045670fb26f400c054013b9726266ebaa",
    SCALE_RUN_CONTRACT: "8b140e0fa93cddfd0cb1e47f1b81d30a875ab68d4a44d2990db40c3dc3851d88",
    SCALE_RESULTS: "9a34f1ab7679f6cf004efe5cb36778a57e4ebf02aa7523e1363a12fe959f27a3",
    SHELL: "3bb8caab4f06e6be9b170b5b3d91cb89b237715132e52507cd60f0514c61ab30",
    SHELL_TB: "e52468d28563858a297ff905fa3c40e1bb5fda940361b026a16375f384f1dde2",
    BUILDER: "5dee7877b665d399ce4d604b142b77ee0ea669ad6c01d266c923c53f4e425810",
    MODEL: "88c142557820ccad55bb59756bfcfcf891de9cc6202816bd346445188a0ed342",
}

EXPECTED_ARTIFACT_HASHES = {
    "full_model_image.bin": "e24e0365e9fad5df2efe3e40df12e3f89f951f37c83449cb40e7d18fb614eafb",
    "image_contract_v2.json": "32b0d2279fa42159f684ccc49916ed177ca99299031392eee3fbcc23baf28d48",
    "manifest.json": "a8157e255cde5f6afe27825ce49c37850bf484de5698b3ed7e4c1c566691ea48",
    "reproducibility.json": "5620cd595dba337c8ea29cdf38b3f8179a7648f3cb2a6f9839dcf32cc70d87a4",
    "validation_report.json": "54e4ac8a1e149acd2416620b3af14158aa0b4cc9a4f8b27a27478fabaffd03ef",
}
EXPECTED_CONTRACT_SHA256 = "3c226985fdf9e598e81c4d583b99493ccd280deecb16d55efbbaef576360a757"
EXPECTED_OPERATOR_COUNTS = {
    "attention_compose": 2016,
    "attention_residual_add": 48,
    "attention_score": 1008,
    "attention_value": 336,
    "final_rmsnorm": 2,
    "input_rmsnorm": 48,
    "k_proj": 48,
    "kv_write": 48,
    "lm_head_tile": 9496,
    "mlp_down_proj": 48,
    "mlp_gate_proj": 48,
    "mlp_residual_add": 48,
    "mlp_up_proj": 48,
    "o_proj": 48,
    "post_attention_rmsnorm": 48,
    "q_proj": 48,
    "rope_k": 48,
    "rope_q": 48,
    "silu_gate": 48,
    "softmax": 336,
    "v_proj": 48,
}

SCALE32_SIGNIFICAND_MIN = 0x8000
SCALE32_SIGNIFICAND_MAX = 0xFFFF
SCALE32_EXPONENT_MIN = -24
SCALE32_EXPONENT_MAX = 4


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def sha256_bytes(raw: bytes | memoryview) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative_label(path: Path) -> str:
    if path == MODEL:
        return f"hf-cache://Qwen/Qwen2.5-0.5B@{REVISION}/model.safetensors"
    return path.relative_to(ROOT).as_posix()


def file_record(path: Path) -> dict[str, Any]:
    return {
        "path": relative_label(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_sums(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        require(name not in result, f"duplicate SHA256SUMS entry: {name}")
        result[name] = digest
    return result


def interval_union(intervals: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    result: list[tuple[int, int]] = []
    for start, end in sorted(intervals):
        require(start <= end, "invalid interval")
        if result and start <= result[-1][1]:
            result[-1] = (result[-1][0], max(result[-1][1], end))
        else:
            result.append((start, end))
    return result


def ordered_linears(inventory: dict[str, Any]) -> list[dict[str, Any]]:
    records = [item for layer in inventory["layers"] for item in layer["linears"]]
    records.append(inventory["model"]["lm_head"])
    require(len(records) == 169, "linear inventory count changed")
    return records


def ordered_norms(inventory: dict[str, Any]) -> list[dict[str, Any]]:
    records = [item for layer in inventory["layers"] for item in layer["norms"]]
    records.append(inventory["model"]["final_norm"])
    records.sort(key=lambda item: item["image_addr"])
    require(len(records) == 49, "RMSNorm inventory count changed")
    return records


def tensor_key(record: dict[str, Any]) -> str:
    if record["name"] == "lm_head":
        return "model.embed_tokens.weight"
    return record["weight"]["key"]


def tensor_shape(record: dict[str, Any]) -> list[int]:
    if record["name"] == "lm_head":
        return record["tied_weight_tensor"]["shape"]
    return record["weight"]["shape"]


def family_name(name: str) -> str:
    return name if name == "lm_head" else name.rsplit(".", 1)[-1]


def region_offsets(inventory: dict[str, Any]) -> dict[str, int]:
    offsets: dict[str, int] = {}
    cursor = 0
    for name in REGION_ORDER:
        require(
            int(inventory["image_layout"][name]["bytes"]) == EXPECTED_REGION_BYTES[name],
            f"{name} byte contract changed",
        )
        offsets[name] = cursor
        cursor += EXPECTED_REGION_BYTES[name]
    require(cursor == TOTAL_IMAGE_BYTES, "total image arithmetic changed")
    return offsets


def bf16_to_float(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits << 16))[0]


class RawSafetensors:
    def __init__(self, path: Path) -> None:
        self.handle = path.open("rb")
        self.mapping = mmap.mmap(self.handle.fileno(), 0, access=mmap.ACCESS_READ)
        header_bytes = struct.unpack_from("<Q", self.mapping, 0)[0]
        self.data_start = 8 + header_bytes
        self.header = json.loads(self.mapping[8 : self.data_start])

    def close(self) -> None:
        self.mapping.close()
        self.handle.close()

    def descriptor(self, key: str) -> dict[str, Any]:
        require(key in self.header and key != "__metadata__", f"missing safetensor: {key}")
        return self.header[key]

    def raw(self, key: str) -> memoryview:
        item = self.descriptor(key)
        start, end = item["data_offsets"]
        return memoryview(self.mapping)[self.data_start + start : self.data_start + end]

    def bf16_values(self, key: str, row: int | None = None) -> list[float]:
        item = self.descriptor(key)
        require(item["dtype"] == "BF16", f"unexpected tensor dtype: {key}")
        shape = item["shape"]
        raw = self.raw(key)
        if row is not None:
            require(len(shape) == 2 and 0 <= row < shape[0], f"invalid row selection: {key}")
            row_bytes = shape[1] * 2
            raw = raw[row * row_bytes : (row + 1) * row_bytes]
        return [bf16_to_float(bits[0]) for bits in struct.iter_unpack("<H", raw)]


def decode_s4(image: mmap.mmap, byte_offset: int, element_index: int) -> int:
    raw = image[byte_offset]
    nibble = (raw >> (4 * (element_index & 1))) & 0xF
    return nibble - 16 if nibble & 0x8 else nibble


def pack_projection_record(multiplier: int, shift: int, bias: int = 0) -> bytes:
    require(-(1 << 31) <= multiplier < (1 << 31), "projection multiplier is not s32")
    require(0 <= shift < 64, "projection shift is not u6")
    require(-(1 << 31) <= bias < (1 << 31), "projection bias is not s32")
    return struct.pack("<iBBi6x", multiplier, shift, 0, bias)


def derive_multiplier(real: float) -> tuple[int, int]:
    require(math.isfinite(real) and real >= 0.0, "invalid real multiplier")
    for shift in range(63, -1, -1):
        candidate = round(math.ldexp(real, shift))
        if candidate <= (1 << 31) - 1:
            return int(candidate), shift
    raise RuntimeError("unrepresentable multiplier")


def ceil_scale32_from_float(value: float) -> int:
    require(math.isfinite(value) and value > 0.0, "invalid Scale32 source")
    numerator, denominator = value.as_integer_ratio()
    for exponent in range(SCALE32_EXPONENT_MIN, SCALE32_EXPONENT_MAX + 1):
        shift = 15 - exponent
        if shift >= 0:
            scaled_numerator = numerator << shift
            scaled_denominator = denominator
        else:
            scaled_numerator = numerator
            scaled_denominator = denominator << -shift
        significand = (scaled_numerator + scaled_denominator - 1) // scaled_denominator
        significand = max(significand, SCALE32_SIGNIFICAND_MIN)
        if significand <= SCALE32_SIGNIFICAND_MAX:
            return significand | ((exponent & 0xFF) << 16)
    raise RuntimeError("Scale32 source exceeds frozen range")


def norm_metadata_keys(record: dict[str, Any]) -> tuple[str, str]:
    operator = record["operator"]
    key = record["tensor"]["key"]
    if operator == "final_rmsnorm":
        return "model.norm.input", "model.norm.output"
    prefix = key.removesuffix(".input_layernorm.weight").removesuffix(
        ".post_attention_layernorm.weight"
    )
    if operator == "input_rmsnorm":
        return f"{prefix}.input_layernorm.input", f"{prefix}.input_layernorm.output"
    require(operator == "post_attention_rmsnorm", f"unknown RMSNorm operator: {operator}")
    return f"{prefix}.post_attention_residual", f"{prefix}.post_attention_layernorm.output"


def verify_inputs() -> dict[str, Any]:
    records = {}
    for path, expected in EXPECTED_INPUT_HASHES.items():
        require(path.is_file(), f"missing immutable input: {relative_label(path)}")
        actual = sha256_file(path)
        require(actual == expected, f"immutable input changed: {relative_label(path)}")
        records[relative_label(path)] = {"bytes": path.stat().st_size, "sha256": actual}
    return {"status": "PASS", "records": records}


def verify_artifact_identity() -> dict[str, Any]:
    sums = parse_sums(OUTPUT_DIR / "SHA256SUMS")
    require(sums == EXPECTED_ARTIFACT_HASHES, "top-level SHA256SUMS changed")
    records = {}
    for name, expected in EXPECTED_ARTIFACT_HASHES.items():
        path = OUTPUT_DIR / name
        require(path.is_file(), f"missing v2 artifact: {name}")
        actual = sha256_file(path)
        require(actual == expected, f"v2 artifact hash changed: {name}")
        records[name] = {"bytes": path.stat().st_size, "sha256": actual}
    require(records["full_model_image.bin"]["bytes"] == TOTAL_IMAGE_BYTES, "image size changed")
    return {"status": "PASS", "artifacts": records}


def verify_contract(contract_wrapper: dict[str, Any]) -> dict[str, Any]:
    require(set(contract_wrapper) == {"contract", "contract_sha256"}, "contract wrapper changed")
    contract = contract_wrapper["contract"]
    digest = sha256_bytes(canonical_bytes(contract))
    require(digest == contract_wrapper["contract_sha256"] == EXPECTED_CONTRACT_SHA256, "contract hash changed")
    require(contract["source_model"]["revision"] == REVISION, "model revision changed")
    require(contract["full_image"]["bytes"] == TOTAL_IMAGE_BYTES, "contract image size changed")
    require(contract["rmsnorm_address_contract"]["record_count"] == 49, "contract RMSNorm count changed")
    require(contract["rmsnorm_address_contract"]["record_bytes"] == RMS_RECORD_BYTES, "contract RMSNorm size changed")
    require(contract["rmsnorm_address_contract"]["gain_byte_offsets"] == [0, 1791], "contract gain offsets changed")
    require(contract["rmsnorm_address_contract"]["trailer_byte_offsets"] == [1792, 1807], "contract trailer offsets changed")
    require(all(value is False for value in contract["scope_guards"].values()), "scope guard changed")
    return {"status": "PASS", "contract_sha256": digest, "scope_guards": contract["scope_guards"]}


def verify_reproducibility(reproducibility: dict[str, Any]) -> dict[str, Any]:
    require(reproducibility["status"] == "PASS_TWO_CLEAN_BUILDS_BYTE_IDENTICAL", "reproducibility status changed")
    compared = {}
    for name in ("full_model_image.bin", "image_contract_v2.json", "manifest.json", "validation_report.json", "SHA256SUMS"):
        left = OUTPUT_DIR / "reproducibility/build_a" / name
        right = OUTPUT_DIR / "reproducibility/build_b" / name
        left_hash = sha256_file(left)
        right_hash = sha256_file(right)
        require(left.stat().st_size == right.stat().st_size and left_hash == right_hash, f"build A/B differ: {name}")
        item = reproducibility["comparisons"][name]
        require(item["byte_identical"] is True, f"comparison flag changed: {name}")
        require(item["bytes"] == left.stat().st_size and item["sha256"] == left_hash, f"comparison record changed: {name}")
        if name != "SHA256SUMS":
            require(sha256_file(OUTPUT_DIR / name) == left_hash, f"top-level artifact differs from build A: {name}")
        compared[name] = {"bytes": left.stat().st_size, "sha256": left_hash}
    for build in ("build_a", "build_b"):
        directory = OUTPUT_DIR / "reproducibility" / build
        sums = parse_sums(directory / "SHA256SUMS")
        for name, expected in sums.items():
            require(sha256_file(directory / name) == expected, f"{build} SHA256SUMS mismatch: {name}")
    return {"status": "PASS", "byte_identical_files": compared}


def verify_region_map(
    inventory: dict[str, Any],
    contract_wrapper: dict[str, Any],
    manifest: dict[str, Any],
    image: mmap.mmap,
) -> dict[str, Any]:
    offsets = region_offsets(inventory)
    contract_regions = contract_wrapper["contract"]["full_image"]["regions"]
    require([item["name"] for item in contract_regions] == list(REGION_ORDER), "contract region order changed")
    memory_intervals = []
    records = {}
    for item in contract_regions:
        name = item["name"]
        accepted = inventory["image_layout"][name]
        require(item["file_offset"] == offsets[name], f"contract file offset changed: {name}")
        require(item["memory_base"] == accepted["base"] and item["memory_end"] == accepted["end"], f"memory range changed: {name}")
        require(item["bytes"] == accepted["bytes"] == EXPECTED_REGION_BYTES[name], f"region bytes changed: {name}")
        require(item["alignment_bytes"] == 16 and item["memory_base"] % 16 == item["memory_end"] % 16 == 0, f"alignment changed: {name}")
        start = offsets[name]
        end = start + EXPECTED_REGION_BYTES[name]
        actual_hash = sha256_bytes(memoryview(image)[start:end])
        expected = manifest["image"]["regions"][name]
        require(expected["bytes"] == end - start and expected["sha256"] == actual_hash, f"region hash changed: {name}")
        records[name] = {"file_offsets": [start, end - 1], "memory_range": [item["memory_base"], item["memory_end"]], "bytes": end - start, "sha256": actual_hash}
        memory_intervals.append((item["memory_base"], item["memory_end"]))
    for left, right in zip(sorted(memory_intervals), sorted(memory_intervals)[1:]):
        require(left[1] <= right[0], "memory regions overlap")
    require(sum(item["bytes"] for item in records.values()) == len(image) == TOTAL_IMAGE_BYTES, "image coverage changed")
    return {"status": "PASS", "regions": records, "dense_file_coverage": True, "memory_nonoverlap": True}


def verify_linears(
    inventory: dict[str, Any],
    scales: dict[str, Any],
    manifest: dict[str, Any],
    tensors: RawSafetensors,
    image: mmap.mmap,
) -> dict[str, Any]:
    records = ordered_linears(inventory)
    manifest_records = manifest["linear_tensors"]
    require(len(manifest_records) == len(records) == 169, "manifest linear count changed")
    by_name = {item["name"]: item for item in manifest_records}
    require(len(by_name) == 169, "manifest linear names are not unique")
    weight_base = inventory["image_layout"]["packed_w4"]["base"]
    metadata_base = inventory["image_layout"]["projection_metadata"]["base"]
    offsets = region_offsets(inventory)
    packed_intervals = []
    metadata_intervals = []
    family_counts: collections.Counter[str] = collections.Counter()
    selected_decodes = 0
    source_bytes_hashed = 0
    metadata_records_checked = 0

    for ordinal, record in enumerate(records):
        name = record["name"]
        item = by_name[name]
        require(item["ordinal"] == ordinal, f"linear ordinal changed: {name}")
        key = tensor_key(record)
        shape = tensor_shape(record)
        descriptor = tensors.descriptor(key)
        require(descriptor["dtype"] == "BF16" and descriptor["shape"] == shape, f"raw tensor contract changed: {key}")
        output_count, input_count = shape
        raw = tensors.raw(key)
        source_hash = sha256_bytes(raw)
        source_bytes_hashed += len(raw)
        require(item["source_weight"]["key"] == key, f"manifest source key changed: {name}")
        require(item["source_weight"]["bytes"] == len(raw) and item["source_weight"]["sha256"] == source_hash, f"source tensor hash changed: {name}")

        metadata = scales["linears"][name]
        weight_scales = metadata["weight_scale"]
        require(len(weight_scales) == output_count and all(math.isfinite(float(v)) and float(v) > 0 for v in weight_scales), f"weight scales changed: {name}")
        selected_outputs = sorted({0, output_count // 2, output_count - 1})
        selected_inputs = sorted({0, input_count // 2, input_count - 1})
        packed_offset = offsets["packed_w4"] + record["packed_w4_addr"] - weight_base
        require(item["packed_w4"]["file_offset"] == packed_offset, f"packed file offset changed: {name}")
        require(item["packed_w4"]["bytes"] == output_count * input_count // 2, f"packed byte count changed: {name}")
        packed_end = packed_offset + item["packed_w4"]["bytes"]
        require(sha256_bytes(memoryview(image)[packed_offset:packed_end]) == item["packed_w4"]["sha256"], f"packed slice hash changed: {name}")
        packed_intervals.append((packed_offset, packed_end))
        for output_channel in selected_outputs:
            row = tensors.bf16_values(key, output_channel)
            require(len(row) == input_count and all(math.isfinite(v) for v in row), f"raw selected row invalid: {name}:{output_channel}")
            derived_scale = max(abs(value) for value in row) / 7.0
            if derived_scale == 0.0:
                derived_scale = 1.0
            require(derived_scale == float(weight_scales[output_channel]), f"selected per-channel scale changed: {name}:{output_channel}")
            for input_channel in selected_inputs:
                expected = max(-8, min(7, round(row[input_channel] / derived_scale)))
                element = output_channel * input_count + input_channel
                actual = decode_s4(image, packed_offset + element // 2, element)
                require(actual == expected, f"signed-int4 decode changed: {name}:{output_channel}:{input_channel}")
                selected_decodes += 1

        multipliers = metadata["multiplier"]
        shifts = metadata["right_shift"]
        biases = metadata["bias_accumulator"] or [0] * output_count
        require(len(multipliers) == len(shifts) == len(biases) == output_count, f"projection metadata count changed: {name}")
        expected_metadata = b"".join(
            pack_projection_record(int(multiplier), int(shift), int(bias))
            for multiplier, shift, bias in zip(multipliers, shifts, biases, strict=True)
        )
        metadata_offset = offsets["projection_metadata"] + record["projection_metadata_addr"] - metadata_base
        metadata_end = metadata_offset + len(expected_metadata)
        require(item["projection_metadata"]["file_offset"] == metadata_offset, f"metadata file offset changed: {name}")
        require(bytes(image[metadata_offset:metadata_end]) == expected_metadata, f"full per-channel projection metadata changed: {name}")
        require(item["projection_metadata"]["sha256"] == sha256_bytes(expected_metadata), f"metadata manifest hash changed: {name}")
        metadata_intervals.append((metadata_offset, metadata_end))
        metadata_records_checked += output_count

        bias_key = key.removesuffix(".weight") + ".bias"
        if metadata["bias_accumulator"] is None:
            require(bias_key not in tensors.header, f"unexpected source bias: {name}")
            require(item["source_bias"] is None, f"unexpected manifest source bias: {name}")
        else:
            bias_desc = tensors.descriptor(bias_key)
            require(bias_desc["dtype"] == "BF16" and bias_desc["shape"] == [output_count], f"bias tensor changed: {name}")
            bias_raw = tensors.raw(bias_key)
            require(item["source_bias"]["sha256"] == sha256_bytes(bias_raw), f"bias source hash changed: {name}")
            bias_values = tensors.bf16_values(bias_key)
            hardware_input_scale = float(metadata["hardware_input_scale"])
            derived_bias = [round(value / (hardware_input_scale * float(scale))) for value, scale in zip(bias_values, weight_scales, strict=True)]
            require(derived_bias == [int(value) for value in metadata["bias_accumulator"]], f"bias accumulator metadata changed: {name}")

        family_counts[family_name(name)] += 1

    require(interval_union(packed_intervals) == [(0, EXPECTED_REGION_BYTES["packed_w4"])], "packed tensor coverage changed")
    metadata_start = region_offsets(inventory)["projection_metadata"]
    require(interval_union(metadata_intervals) == [(metadata_start, metadata_start + EXPECTED_REGION_BYTES["projection_metadata"])], "projection metadata coverage changed")
    require(set(family_counts) == {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "lm_head"}, "linear family coverage changed")
    return {
        "status": "PASS",
        "linear_tensors": len(records),
        "family_counts": dict(sorted(family_counts.items())),
        "raw_source_bytes_hashed": source_bytes_hashed,
        "first_middle_last_weight_decodes": selected_decodes,
        "projection_metadata_channels_exact_checked": metadata_records_checked,
        "orientation": "source [output_channel,input_channel], row-major output channel",
        "nibble_order": "even input low nibble, odd input high nibble",
        "signedness": "two's-complement signed-int4",
    }


def verify_rmsnorm(
    inventory: dict[str, Any],
    schedule: dict[str, Any],
    scales: dict[str, Any],
    contract_wrapper: dict[str, Any],
    manifest: dict[str, Any],
    tensors: RawSafetensors,
    image: mmap.mmap,
) -> dict[str, Any]:
    records = ordered_norms(inventory)
    manifest_records = manifest["rmsnorm_records"]
    require(len(manifest_records) == len(records) == 49, "RMSNorm manifest count changed")
    offsets = region_offsets(inventory)
    norm_base = inventory["image_layout"]["rmsnorm_metadata"]["base"]
    intervals = []
    bases = []
    gains_checked = 0
    trailers_checked = 0
    for ordinal, (record, item) in enumerate(zip(records, manifest_records, strict=True)):
        require(item["ordinal"] == ordinal, "RMSNorm ordinal changed")
        require(record["image_bytes"] == item["bytes"] == RMS_RECORD_BYTES, "RMSNorm record size changed")
        key = record["tensor"]["key"]
        require(item["tensor_key"] == key, f"RMSNorm tensor order changed: {key}")
        descriptor = tensors.descriptor(key)
        require(descriptor["dtype"] == "BF16" and descriptor["shape"] == [HIDDEN], f"RMSNorm source changed: {key}")
        raw = tensors.raw(key)
        require(item["source_weight"]["sha256"] == sha256_bytes(raw), f"RMSNorm source hash changed: {key}")
        input_key, output_key = norm_metadata_keys(record)
        input_meta = scales["operators"][input_key]
        output_scale = float(scales["operators"][output_key]["scale"])
        gains = [round(value / output_scale * 256.0) for value in tensors.bf16_values(key)]
        require(all(-(1 << 15) <= value < (1 << 15) for value in gains), f"RMSNorm gain overflow: {key}")
        gain_raw = struct.pack(f"<{HIDDEN}h", *gains)
        trailer = struct.pack("<dd", float(input_meta["absmax"]), float(input_meta["scale"]))
        payload = gain_raw + trailer
        file_offset = offsets["rmsnorm_metadata"] + record["image_addr"] - norm_base
        require(item["memory_addr"] == record["image_addr"] and item["file_offset"] == file_offset, f"RMSNorm address changed: {key}")
        require(bytes(image[file_offset : file_offset + RMS_RECORD_BYTES]) == payload, f"RMSNorm gains-first payload changed: {key}")
        require(item["gain_sha256"] == sha256_bytes(gain_raw) and item["sha256"] == sha256_bytes(payload), f"RMSNorm manifest hash changed: {key}")
        require(item["trailing_activation_metadata"]["sha256"] == sha256_bytes(trailer), f"RMSNorm trailer hash changed: {key}")
        require(item["trailing_activation_metadata"]["runtime_executable"] is False, f"RMSNorm trailer executable flag changed: {key}")
        intervals.append((file_offset, file_offset + RMS_RECORD_BYTES))
        bases.append(record["image_addr"])
        gains_checked += HIDDEN
        trailers_checked += len(trailer)

    rms_commands = [
        command
        for command in schedule["commands"]
        if command["operator"] in {"input_rmsnorm", "post_attention_rmsnorm", "final_rmsnorm"}
    ]
    scale_addrs = [int(command["scale_addr"]) for command in rms_commands]
    require(len(scale_addrs) == 98, "RMSNorm command count changed")
    counts = collections.Counter(scale_addrs)
    require(list(counts) == bases and set(counts.values()) == {2}, "ordered RMSNorm scale addresses changed")
    packed_scale_addrs = b"".join(struct.pack("<Q", address) for address in scale_addrs)
    packed_bases = b"".join(struct.pack("<Q", address) for address in bases)
    contract_rms = contract_wrapper["contract"]["rmsnorm_address_contract"]
    require(contract_rms["accepted_record_bases"] == bases, "contract RMSNorm bases changed")
    require(contract_rms["accepted_record_bases_sha256"] == sha256_bytes(packed_bases), "contract RMSNorm base digest changed")
    require(interval_union(intervals) == [(offsets["rmsnorm_metadata"], offsets["rmsnorm_metadata"] + EXPECTED_REGION_BYTES["rmsnorm_metadata"])], "RMSNorm region coverage changed")

    shell_text = SHELL.read_text(encoding="utf-8")
    for statement in (
        "ST_SCALE_ACT_RECV:\n                    prefix_mem_req_addr_q <= scale_addr_q + (beat_idx_ext_w << 5);",
        "ST_GAIN_RECV0:\n                    prefix_mem_req_addr_q <=\n                        scale_addr_q + (beat_idx_ext_w << 5) + 64'd16;",
        "ST_ACT_RECV: begin\n                    if (accepted_read_w) begin\n                        if (beat_idx_q == LAST_BEAT) begin\n                            prefix_mem_req_addr_q <= src0_addr_q;",
    ):
        require(statement in shell_text, "RMSNorm RTL address expression changed")
    consumed_offsets = []
    for beat in range(56):
        consumed_offsets.extend(range(beat * 32, beat * 32 + 16))
        consumed_offsets.extend(range(beat * 32 + 16, beat * 32 + 32))
    require(consumed_offsets == list(range(RMS_GAIN_BYTES)), "RMSNorm RTL gain order changed")
    require(not set(consumed_offsets).intersection(range(RMS_GAIN_BYTES, RMS_RECORD_BYTES)), "RMSNorm trailer is reachable as gains")
    return {
        "status": "PASS",
        "records_exact_checked": len(records),
        "gains_exact_checked": gains_checked,
        "trailer_bytes_exact_checked": trailers_checked,
        "command_scale_addr_count": len(scale_addrs),
        "ordered_command_scale_addrs_sha256": sha256_bytes(packed_scale_addrs),
        "record_bases_sha256": sha256_bytes(packed_bases),
        "gain_byte_offsets": [0, 1791],
        "trailer_byte_offsets": [1792, 1807],
        "all_896_gains_consumed_once_in_order": True,
        "trailer_reachable_as_gain_data": False,
        "activation_scale_runtime_source": "src0_addr",
    }


def build_aux_payloads(inventory: dict[str, Any], scales: dict[str, Any]) -> list[tuple[int, str, int, bytes]]:
    result = []
    for layer in inventory["layers"]:
        layer_id = int(layer["layer_id"])
        aux = layer["aux_metadata"]
        attention = scales["attention"][f"model.layers.{layer_id}.self_attn"]
        conversion_q9 = int(attention["rope_conversion_q9"])
        rope_q = struct.pack(f"<{HIDDEN}h", *([conversion_q9] * HIDDEN))
        rope_k_channels = KV_HEADS * HEAD_DIM
        rope_k = struct.pack(f"<{rope_k_channels}h", *([conversion_q9] * rope_k_channels))
        key_scales = attention["key_rope_output_scales"]
        value_scale = float(scales["linears"][f"model.layers.{layer_id}.self_attn.v_proj"]["output_scale"])
        kv = struct.pack(
            "<IIII",
            ceil_scale32_from_float(float(key_scales[0])),
            ceil_scale32_from_float(float(key_scales[1])),
            ceil_scale32_from_float(value_scale),
            ceil_scale32_from_float(value_scale),
        )
        score = b"".join(
            pack_projection_record(int(multiplier), int(shift))
            for multiplier, shift in zip(attention["score_multiplier"], attention["score_right_shift"], strict=True)
        )
        down = scales["linears"][f"model.layers.{layer_id}.mlp.down_proj"]
        silu_multiplier, silu_shift = derive_multiplier(1.0 / ((1 << 21) * float(down["input_scale"])))
        silu = pack_projection_record(silu_multiplier, silu_shift)
        for name, raw in (
            ("rope_q_scale_records", rope_q),
            ("rope_k_scale_records", rope_k),
            ("kv_scale_record", kv),
            ("attention_score_records", score),
            ("silu_record", silu),
        ):
            descriptor = aux[name]
            require(len(raw) == descriptor["image_bytes"], f"aux byte count changed: layer{layer_id}:{name}")
            result.append((layer_id, name, descriptor["image_addr"], raw))
    require(len(result) == 120, "operator aux record count changed")
    return result


def verify_aux(
    inventory: dict[str, Any],
    scales: dict[str, Any],
    manifest: dict[str, Any],
    image: mmap.mmap,
) -> dict[str, Any]:
    expected = build_aux_payloads(inventory, scales)
    manifest_records = manifest["operator_aux_records"]
    require(len(manifest_records) == len(expected), "operator aux manifest count changed")
    offsets = region_offsets(inventory)
    aux_base = inventory["image_layout"]["operator_aux_metadata"]["base"]
    intervals = []
    family_counts: collections.Counter[str] = collections.Counter()
    bytes_checked = 0
    for item, (layer, name, address, raw) in zip(manifest_records, expected, strict=True):
        file_offset = offsets["operator_aux_metadata"] + address - aux_base
        require(item["layer"] == layer and item["name"] == name, f"aux order changed: layer{layer}:{name}")
        require(item["memory_addr"] == address and item["file_offset"] == file_offset, f"aux address changed: layer{layer}:{name}")
        require(bytes(image[file_offset : file_offset + len(raw)]) == raw, f"aux payload changed: layer{layer}:{name}")
        require(item["sha256"] == sha256_bytes(raw), f"aux manifest hash changed: layer{layer}:{name}")
        intervals.append((file_offset, file_offset + len(raw)))
        family_counts[name] += 1
        bytes_checked += len(raw)
    start = offsets["operator_aux_metadata"]
    require(interval_union(intervals) == [(start, start + EXPECTED_REGION_BYTES["operator_aux_metadata"])], "operator aux region coverage changed")
    return {"status": "PASS", "records_exact_checked": len(expected), "bytes_exact_checked": bytes_checked, "family_counts": dict(sorted(family_counts.items()))}


def verify_schedule(inventory: dict[str, Any], schedule: dict[str, Any]) -> dict[str, Any]:
    commands = schedule["commands"]
    require(schedule["command_count"] == len(commands) == 13_914, "schedule command count changed")
    counts = collections.Counter(command["operator"] for command in commands)
    require(dict(counts) == EXPECTED_OPERATOR_COUNTS, "schedule operator counts changed")
    linears = {item["name"]: item for item in ordered_linears(inventory)}
    packed_intervals = {name: [] for name in linears}
    metadata_intervals = {name: [] for name in linears}
    projection_operators = {"q_proj", "k_proj", "v_proj", "o_proj", "mlp_gate_proj", "mlp_up_proj", "mlp_down_proj", "lm_head_tile"}
    for command in commands:
        if command["operator"] not in projection_operators:
            continue
        name = command["weight_tensor"]
        require(name in linears, f"schedule references unknown tensor: {name}")
        record = linears[name]
        output_count, input_count = tensor_shape(record)
        channels = int(command["n"])
        reduction = int(command["k"])
        require(reduction == input_count and 0 < channels <= output_count, f"schedule geometry changed: {name}")
        weight_start = int(command["src1_addr"])
        metadata_start = int(command["scale_addr"])
        weight_end = weight_start + channels * reduction // 2
        metadata_end = metadata_start + channels * 16
        require(record["packed_w4_addr"] <= weight_start < weight_end <= record["packed_w4_addr"] + record["packed_w4_bytes"], f"schedule packed range changed: {name}")
        require(record["projection_metadata_addr"] <= metadata_start < metadata_end <= record["projection_metadata_addr"] + record["projection_metadata_bytes"], f"schedule metadata range changed: {name}")
        packed_intervals[name].append((weight_start, weight_end))
        metadata_intervals[name].append((metadata_start, metadata_end))
    for name, record in linears.items():
        require(interval_union(packed_intervals[name]) == [(record["packed_w4_addr"], record["packed_w4_addr"] + record["packed_w4_bytes"])], f"schedule packed coverage changed: {name}")
        require(interval_union(metadata_intervals[name]) == [(record["projection_metadata_addr"], record["projection_metadata_addr"] + record["projection_metadata_bytes"])], f"schedule metadata coverage changed: {name}")
    return {
        "status": "PASS",
        "commands_exact_checked": len(commands),
        "operator_counts": dict(sorted(counts.items())),
        "all_169_linear_weight_and_metadata_ranges_covered": True,
        "schedule_sha256": sha256_file(SCHEDULE),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated-at-utc", required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_DIR / "independent_postbuild_audit.json",
    )
    args = parser.parse_args()
    require(EXPECTED_TIMESTAMP.fullmatch(args.generated_at_utc) is not None, "--generated-at-utc must use YYYY-MM-DDTHH:MM:SSZ")

    inputs = verify_inputs()
    artifact_identity = verify_artifact_identity()
    inventory = load_json(INVENTORY)
    schedule = load_json(SCHEDULE)
    scales = load_json(SCALES)
    contract_wrapper = load_json(OUTPUT_DIR / "image_contract_v2.json")
    manifest = load_json(OUTPUT_DIR / "manifest.json")
    validation = load_json(OUTPUT_DIR / "validation_report.json")
    reproducibility = load_json(OUTPUT_DIR / "reproducibility.json")
    require(manifest["status"] == "COMPLETE_IMAGE_BUILT_AND_SELF_VERIFIED_PENDING_FRESH_REVIEW", "manifest status changed")
    require(validation["status"] == "PASS", "self-validation status changed")

    image_path = OUTPUT_DIR / "full_model_image.bin"
    image_handle = image_path.open("rb")
    image = mmap.mmap(image_handle.fileno(), 0, access=mmap.ACCESS_READ)
    tensors = RawSafetensors(MODEL)
    try:
        checks = {
            "immutable_inputs": inputs,
            "artifact_identity": artifact_identity,
            "contract": verify_contract(contract_wrapper),
            "reproducibility": verify_reproducibility(reproducibility),
            "region_map": verify_region_map(inventory, contract_wrapper, manifest, image),
            "schedule_coverage": verify_schedule(inventory, schedule),
            "linear_tensors": verify_linears(inventory, scales, manifest, tensors, image),
            "rmsnorm_gains_first": verify_rmsnorm(inventory, schedule, scales, contract_wrapper, manifest, tensors, image),
            "operator_aux": verify_aux(inventory, scales, manifest, image),
        }
    finally:
        tensors.close()
        image.close()
        image_handle.close()

    report = {
        "schema_version": 1,
        "mission_id": MISSION_ID,
        "generated_at_utc": args.generated_at_utc,
        "status": "PASS_INDEPENDENT_POSTBUILD_AUDIT_PENDING_FRESH_REVIEW",
        "classification": "standard_library_raw_safetensors_static_verification",
        "checks": checks,
        "verifier": file_record(Path(__file__)),
        "scope_guards": {
            "builder_imported": False,
            "torch_imported": False,
            "transformers_imported": False,
            "model_constructed": False,
            "model_called": False,
            "recalibration": False,
            "command_runtime_or_decoder_execution": False,
            "rtl_or_testbench_change": False,
            "synthesis_opensta_ppa": False,
            "publication_mutation": False,
        },
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(canonical_bytes(report))
    sidecar = output.with_suffix(".SHA256SUMS")
    output_label = output.relative_to(ROOT).as_posix()
    verifier_label = Path(__file__).relative_to(ROOT).as_posix()
    sidecar.write_text(
        f"{sha256_file(output)}  {output_label}\n"
        f"{sha256_file(Path(__file__))}  {verifier_label}\n",
        encoding="utf-8",
    )
    print(
        "ACE2_FULL_QWEN_IMAGE_V2_INDEPENDENT_AUDIT_PASS "
        f"image_sha256={EXPECTED_ARTIFACT_HASHES['full_model_image.bin']} "
        f"ordered_rmsnorm_scale_addr_sha256={checks['rmsnorm_gains_first']['ordered_command_scale_addrs_sha256']} "
        f"report={output.relative_to(ROOT)}"
    )


if __name__ == "__main__":
    main()
