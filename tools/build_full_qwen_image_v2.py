#!/usr/bin/env python3
"""Build and verify the corrected ACE-2 full-Qwen memory image.

This is a raw-safetensors packaging flow.  It does not construct or call a
model, execute the command schedule, or invoke RTL/PPA tools.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import struct
import sys
from pathlib import Path
from typing import Any, Iterable

import torch
from huggingface_hub.constants import HF_HUB_CACHE
from safetensors import safe_open


ROOT = Path(__file__).resolve().parents[1]
MISSION_ID = "build-full-qwen-packed-w4-metadata-image-v2"
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
RMS_REFERENCE = ROOT / "tools/ace2_rmsnorm_reference.py"

REVISION = "060db6499f32faf8b98477b0a26969ef7d8b9987"
MODEL = (
    Path(HF_HUB_CACHE)
    / "models--Qwen--Qwen2.5-0.5B"
    / "snapshots"
    / REVISION
    / "model.safetensors"
)

HIDDEN = 896
HEAD_DIM = 64
KV_HEADS = 2
LAYERS = 24
RECORD_BYTES = 1_808
GAIN_BYTES = HIDDEN * 2
TRAILER_BYTES = 16

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
    RMS_REFERENCE: "400cd5c4858f08b78283bd1bc18fe8bfc6bec719819e9296fc2ccc5d62a79c8a",
    MODEL: "88c142557820ccad55bb59756bfcfcf891de9cc6202816bd346445188a0ed342",
}

EXPECTED_REGION_BYTES = {
    "packed_w4": 246_980_608,
    "projection_metadata": 7_297_024,
    "rmsnorm_metadata": 88_592,
    "operator_aux_metadata": 55_296,
}
REGION_ORDER = tuple(EXPECTED_REGION_BYTES)
TOTAL_IMAGE_BYTES = sum(EXPECTED_REGION_BYTES.values())
EXPECTED_TIMESTAMP = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z")

SCALE32_SIGNIFICAND_MIN = 0x8000
SCALE32_SIGNIFICAND_MAX = 0xFFFF
SCALE32_EXPONENT_MIN = -24
SCALE32_EXPONENT_MAX = 4


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_bytes(canonical_bytes(value))


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


def raw_tensor_bytes(value: torch.Tensor) -> bytes:
    tensor = value.detach().contiguous().cpu()
    if tensor.dtype == torch.bfloat16:
        return tensor.view(torch.uint16).numpy().tobytes(order="C")
    return tensor.numpy().tobytes(order="C")


def packed_s4(value: torch.Tensor) -> bytes:
    flat = value.detach().reshape(-1).to(torch.int16).cpu()
    require(flat.numel() % 2 == 0, "signed-int4 payload has odd element count")
    packed = torch.bitwise_or(
        flat[0::2] & 0xF,
        torch.bitwise_left_shift(flat[1::2] & 0xF, 4),
    )
    return packed.to(torch.uint8).numpy().tobytes(order="C")


def derive_multiplier(real: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    value = real.detach().to(torch.float64).cpu()
    require(bool(torch.all(torch.isfinite(value))), "non-finite real multiplier")
    require(not bool(torch.any(value < 0)), "negative real multiplier")
    multiplier = torch.zeros_like(value, dtype=torch.int64)
    right_shift = torch.full_like(value, -1, dtype=torch.int64)
    for shift in range(63, -1, -1):
        candidate = torch.round(value * math.ldexp(1.0, shift))
        select = (right_shift < 0) & (candidate <= (1 << 31) - 1)
        multiplier = torch.where(select, candidate.to(torch.int64), multiplier)
        right_shift = torch.where(
            select,
            torch.full_like(right_shift, shift),
            right_shift,
        )
    require(not bool(torch.any(right_shift < 0)), "unrepresentable multiplier")
    return multiplier, right_shift


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
    raise OverflowError("Scale32 source exceeds frozen range")


def validate_scale32(record: int) -> None:
    require(0 <= record <= 0xFFFFFFFF, "Scale32 record is not u32")
    require((record >> 24) == 0, "Scale32 reserved byte is nonzero")
    significand = record & 0xFFFF
    exponent_u8 = (record >> 16) & 0xFF
    exponent = exponent_u8 - 256 if exponent_u8 & 0x80 else exponent_u8
    require(
        SCALE32_SIGNIFICAND_MIN <= significand <= SCALE32_SIGNIFICAND_MAX,
        "Scale32 significand is not normalized",
    )
    require(
        SCALE32_EXPONENT_MIN <= exponent <= SCALE32_EXPONENT_MAX,
        "Scale32 exponent is outside frozen range",
    )


def pack_projection_record(multiplier: int, shift: int, bias: int) -> bytes:
    require(-(1 << 31) <= multiplier < (1 << 31), "projection multiplier is not s32")
    require(0 <= shift < 64, "projection shift is not u6")
    require(-(1 << 31) <= bias < (1 << 31), "projection bias is not s32")
    record = bytearray(16)
    struct.pack_into("<i", record, 0, multiplier)
    record[4] = shift
    record[5] = 0
    struct.pack_into("<i", record, 6, bias)
    return bytes(record)


def unpack_projection_record(raw: bytes) -> dict[str, int]:
    require(len(raw) == 16, "projection record is not 16 bytes")
    require((raw[4] & 0xC0) == 0, "projection shift reserved bits are nonzero")
    require(raw[5] == 0, "projection output zero point is not zero")
    require(raw[10:] == b"\x00" * 6, "projection high reserved bytes are nonzero")
    return {
        "multiplier": struct.unpack_from("<i", raw, 0)[0],
        "right_shift": raw[4] & 0x3F,
        "output_zero_point": struct.unpack_from("<b", raw, 5)[0],
        "bias_accumulator": struct.unpack_from("<i", raw, 6)[0],
    }


def decode_s4(raw: bytes, element_index: int) -> int:
    byte = raw[element_index // 2]
    nibble = (byte >> (4 * (element_index & 1))) & 0xF
    return nibble - 16 if nibble & 0x8 else nibble


def interval_union(intervals: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    result: list[tuple[int, int]] = []
    for start, end in sorted(intervals):
        require(start <= end, "invalid interval")
        if result and start <= result[-1][1]:
            result[-1] = (result[-1][0], max(result[-1][1], end))
        else:
            result.append((start, end))
    return result


def region_file_offsets(inventory: dict[str, Any]) -> dict[str, int]:
    result: dict[str, int] = {}
    cursor = 0
    for name in REGION_ORDER:
        require(
            inventory["image_layout"][name]["bytes"] == EXPECTED_REGION_BYTES[name],
            f"accepted {name} byte count changed",
        )
        result[name] = cursor
        cursor += EXPECTED_REGION_BYTES[name]
    require(cursor == TOTAL_IMAGE_BYTES, "image total arithmetic changed")
    return result


def ordered_linears(inventory: dict[str, Any]) -> list[dict[str, Any]]:
    records = [item for layer in inventory["layers"] for item in layer["linears"]]
    records.append(inventory["model"]["lm_head"])
    require(len(records) == LAYERS * 7 + 1, "linear inventory count changed")
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
    if name == "lm_head":
        return name
    return name.rsplit(".", 1)[-1]


def output_scale_vector(metadata: dict[str, Any], output_count: int) -> torch.Tensor:
    scalar = metadata.get("output_scale")
    if scalar is not None:
        require(math.isfinite(float(scalar)) and float(scalar) > 0, "invalid output scale")
        return torch.full((output_count,), float(scalar), dtype=torch.float64)
    heads = metadata.get("output_head_scales")
    require(isinstance(heads, list) and heads, "missing per-head output scales")
    require(output_count % len(heads) == 0, "per-head output scale geometry changed")
    values = torch.tensor(heads, dtype=torch.float64)
    require(bool(torch.all(torch.isfinite(values))) and not bool(torch.any(values <= 0)), "invalid per-head scale")
    return values.repeat_interleave(output_count // len(heads))


def validate_inputs() -> dict[str, dict[str, Any]]:
    require("transformers" not in sys.modules, "transformers was imported")
    records: dict[str, dict[str, Any]] = {}
    for path, expected in EXPECTED_INPUT_HASHES.items():
        require(path.is_file(), f"missing immutable input: {relative_label(path)}")
        actual = sha256_file(path)
        require(actual == expected, f"immutable input hash changed: {relative_label(path)}")
        records[relative_label(path)] = {
            "bytes": path.stat().st_size,
            "sha256": actual,
        }
    return records


def contract_body(
    generated_at_utc: str,
    inventory: dict[str, Any],
    input_records: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    file_offsets = region_file_offsets(inventory)
    regions = []
    for name in REGION_ORDER:
        item = inventory["image_layout"][name]
        regions.append(
            {
                "name": name,
                "file_offset": file_offsets[name],
                "memory_base": item["base"],
                "memory_end": item["end"],
                "bytes": item["bytes"],
                "alignment_bytes": 16,
            }
        )
    norm_bases = [item["image_addr"] for item in ordered_norms(inventory)]
    return {
        "schema_version": 2,
        "mission_id": MISSION_ID,
        "generated_at_utc": generated_at_utc,
        "operator_authorization": {
            "authorization_id": "operator-rmsnorm-image-layout-v2",
            "decision": "gains-first RMSNorm records with preserved 16-byte trailing metadata at existing addresses",
            "schedule_address_change_authorized": False,
            "rtl_change_authorized": False,
        },
        "immutable_v1_boundary": {
            "status": "preserved_byte_for_byte",
            "blocked_audit": input_records[relative_label(V1_AUDIT)],
            "accepted_inventory": input_records[relative_label(INVENTORY)],
            "accepted_schedule": input_records[relative_label(SCHEDULE)],
            "accepted_boundary_provenance": input_records[relative_label(PROVENANCE)],
        },
        "source_model": {
            "repository": "Qwen/Qwen2.5-0.5B",
            "revision": REVISION,
            "safetensors": input_records[relative_label(MODEL)],
            "access": "safe_open raw tensors only; no model construction or call",
        },
        "frozen_scale_provenance": {
            "derived_scales": input_records[relative_label(SCALES)],
            "historical_run_contract": input_records[relative_label(SCALE_RUN_CONTRACT)],
            "historical_results": input_records[relative_label(SCALE_RESULTS)],
            "recalibrated_here": False,
        },
        "full_image": {
            "bytes": TOTAL_IMAGE_BYTES,
            "regions": regions,
            "file_layout": "dense concatenation in declared region order; memory addresses remain the accepted sparse bases",
        },
        "encodings": {
            "packed_w4": "row-major output-channel then input-channel; two's-complement signed-int4; even element in low nibble, odd element in high nibble",
            "projection_metadata": "16-byte little-endian record: multiplier s32 bits31:0, right_shift u6 bits37:32, reserved bits39:38 zero, output_zero_point s8 bits47:40 zero, bias_accumulator s32 bits79:48, bits127:80 zero",
            "rmsnorm_v2": "896 little-endian signed-int16 Q7.8 gains at bytes0..1791 followed by little-endian binary64 activation absmax and scale at bytes1792..1807",
            "rope_scale": "little-endian signed-int16 Q9 conversion, one value per channel",
            "kv_scale": "four little-endian Scale32 u32 records ordered K-head0,K-head1,V-head0,V-head1",
            "attention_score": "16-byte record with multiplier s32 bits31:0, right_shift u6 bits37:32, all remaining bits zero",
            "silu": "16-byte record with multiplier s32 bits31:0, right_shift u6 bits37:32, output zero point s8 bits47:40 zero, all remaining bits zero",
        },
        "rmsnorm_address_contract": {
            "record_count": 49,
            "record_bytes": RECORD_BYTES,
            "accepted_record_bases": norm_bases,
            "accepted_record_bases_sha256": sha256_bytes(
                b"".join(struct.pack("<Q", value) for value in norm_bases)
            ),
            "gain_byte_offsets": [0, GAIN_BYTES - 1],
            "trailer_byte_offsets": [GAIN_BYTES, RECORD_BYTES - 1],
            "schedule_scale_addresses_preserved": True,
            "activation_scale_runtime_source": "src0_addr, not cmd_scale_addr",
        },
        "scope_guards": {
            "model_constructed": False,
            "model_called": False,
            "recalibration": False,
            "command_runtime_or_decoder_execution": False,
            "rtl_or_testbench_change": False,
            "synthesis_opensta_ppa": False,
            "publication_mutation": False,
        },
        "builder_source": file_record(Path(__file__)),
    }


class ImageWriter:
    def __init__(self, path: Path, inventory: dict[str, Any]) -> None:
        self.path = path
        self.handle = path.open("wb")
        self.inventory = inventory
        self.file_offsets = region_file_offsets(inventory)
        self.full_hash = hashlib.sha256()
        self.region_hashes = {name: hashlib.sha256() for name in REGION_ORDER}
        self.region_counts = {name: 0 for name in REGION_ORDER}
        self.active_region_index = 0

    def write(self, region: str, raw: bytes) -> None:
        require(region == REGION_ORDER[self.active_region_index], "region write order changed")
        expected_offset = self.file_offsets[region] + self.region_counts[region]
        require(self.handle.tell() == expected_offset, f"{region} file offset changed")
        self.handle.write(raw)
        self.full_hash.update(raw)
        self.region_hashes[region].update(raw)
        self.region_counts[region] += len(raw)
        require(
            self.region_counts[region] <= EXPECTED_REGION_BYTES[region],
            f"{region} overflow",
        )

    def finish_region(self, region: str) -> None:
        require(region == REGION_ORDER[self.active_region_index], "region finish order changed")
        require(
            self.region_counts[region] == EXPECTED_REGION_BYTES[region],
            f"{region} byte count mismatch",
        )
        self.active_region_index += 1

    def close(self) -> dict[str, Any]:
        require(self.active_region_index == len(REGION_ORDER), "not all image regions finished")
        self.handle.flush()
        os.fsync(self.handle.fileno())
        self.handle.close()
        require(self.path.stat().st_size == TOTAL_IMAGE_BYTES, "full image byte count mismatch")
        return {
            "bytes": TOTAL_IMAGE_BYTES,
            "sha256": self.full_hash.hexdigest(),
            "regions": {
                name: {
                    "bytes": self.region_counts[name],
                    "sha256": self.region_hashes[name].hexdigest(),
                }
                for name in REGION_ORDER
            },
        }


def build_linear_payloads(
    writer: ImageWriter,
    weights: Any,
    inventory: dict[str, Any],
    scales: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    decode_expectations: dict[str, dict[str, Any]] = {}
    keys = set(weights.keys())
    linear_records = ordered_linears(inventory)

    for ordinal, record in enumerate(linear_records):
        name = record["name"]
        key = tensor_key(record)
        shape = tensor_shape(record)
        require(key in keys, f"missing raw tensor: {key}")
        source = weights.get_slice(key)
        require(list(source.get_shape()) == shape, f"tensor shape changed: {key}")
        require(str(source.get_dtype()) == "BF16", f"tensor dtype changed: {key}")
        output_count, input_count = shape
        require(input_count % 2 == 0, f"odd projection reduction size: {name}")
        metadata = scales["linears"].get(name)
        require(isinstance(metadata, dict), f"missing frozen linear metadata: {name}")
        frozen_weight_scale = torch.tensor(metadata["weight_scale"], dtype=torch.float64)
        require(frozen_weight_scale.shape == (output_count,), f"weight scale count changed: {name}")
        require(bool(torch.all(torch.isfinite(frozen_weight_scale))), f"non-finite weight scale: {name}")
        require(not bool(torch.any(frozen_weight_scale <= 0)), f"non-positive weight scale: {name}")

        output_scale = output_scale_vector(metadata, output_count)
        hardware_input_scale = float(metadata["hardware_input_scale"])
        require(
            math.isfinite(hardware_input_scale) and hardware_input_scale > 0,
            f"invalid hardware input scale: {name}",
        )
        derived_multiplier, derived_shift = derive_multiplier(
            hardware_input_scale * frozen_weight_scale / output_scale
        )
        frozen_multiplier = torch.tensor(metadata["multiplier"], dtype=torch.int64)
        frozen_shift = torch.tensor(metadata["right_shift"], dtype=torch.int64)
        require(torch.equal(derived_multiplier, frozen_multiplier), f"multiplier metadata differs: {name}")
        require(torch.equal(derived_shift, frozen_shift), f"right-shift metadata differs: {name}")

        bias_key = key.removesuffix(".weight") + ".bias"
        frozen_bias_values = metadata.get("bias_accumulator")
        source_bias_record: dict[str, Any] | None = None
        if frozen_bias_values is None:
            require(bias_key not in keys, f"unexpected source bias without metadata: {name}")
            frozen_bias = torch.zeros(output_count, dtype=torch.int64)
        else:
            require(bias_key in keys, f"missing source bias: {name}")
            bias = weights.get_tensor(bias_key).contiguous()
            require(tuple(bias.shape) == (output_count,), f"bias shape changed: {name}")
            require(bias.dtype == torch.bfloat16, f"bias dtype changed: {name}")
            derived_bias = torch.round(
                bias.to(torch.float64) / (hardware_input_scale * frozen_weight_scale)
            ).to(torch.int64)
            frozen_bias = torch.tensor(frozen_bias_values, dtype=torch.int64)
            require(torch.equal(derived_bias.cpu(), frozen_bias), f"bias metadata differs: {name}")
            source_bias_record = {
                "key": bias_key,
                "bytes": bias.numel() * 2,
                "sha256": sha256_bytes(raw_tensor_bytes(bias)),
            }

        packed_expected_bytes = output_count * input_count // 2
        require(record["packed_w4_bytes"] == packed_expected_bytes, f"packed byte contract changed: {name}")
        weight_base = inventory["image_layout"]["packed_w4"]["base"]
        expected_file_offset = writer.file_offsets["packed_w4"] + record["packed_w4_addr"] - weight_base
        require(writer.handle.tell() == expected_file_offset, f"packed address map changed: {name}")

        source_hash = hashlib.sha256()
        qweight_hash = hashlib.sha256()
        packed_hash = hashlib.sha256()
        selected_outputs = sorted({0, output_count // 2, output_count - 1})
        selected_inputs = sorted({0, input_count // 2, input_count - 1})
        selected_values: dict[tuple[int, int], int] = {}
        chunk_rows = 256
        packed_bytes_written = 0
        for start in range(0, output_count, chunk_rows):
            stop = min(output_count, start + chunk_rows)
            raw_weight = source[start:stop].contiguous()
            require(raw_weight.dtype == torch.bfloat16, f"raw chunk dtype changed: {name}")
            raw_bytes = raw_tensor_bytes(raw_weight)
            source_hash.update(raw_bytes)
            weight = raw_weight.to(torch.float64)
            derived_scale = weight.abs().amax(dim=1) / 7.0
            derived_scale = torch.where(
                derived_scale > 0,
                derived_scale,
                torch.ones_like(derived_scale),
            ).cpu()
            require(
                torch.equal(derived_scale, frozen_weight_scale[start:stop]),
                f"per-channel weight scale differs: {name}[{start}:{stop}]",
            )
            qweight = torch.round(weight / derived_scale[:, None]).clamp(-8, 7).to(torch.int8).cpu()
            qraw = raw_tensor_bytes(qweight)
            qweight_hash.update(qraw)
            packed = packed_s4(qweight)
            packed_hash.update(packed)
            writer.write("packed_w4", packed)
            packed_bytes_written += len(packed)
            for out_index in selected_outputs:
                if start <= out_index < stop:
                    local = out_index - start
                    for in_index in selected_inputs:
                        selected_values[(out_index, in_index)] = int(qweight[local, in_index])

        require(packed_bytes_written == packed_expected_bytes, f"packed write length differs: {name}")
        require(qweight_hash.hexdigest() == metadata["qweight_sha256"], f"qweight hash differs: {name}")
        require(len(selected_values) == len(selected_outputs) * len(selected_inputs), f"decode witnesses incomplete: {name}")
        decode_expectations[name] = {
            "packed_file_offset": expected_file_offset,
            "input_count": input_count,
            "output_channels": selected_outputs,
            "input_channels": selected_inputs,
            "values": [
                {
                    "output_channel": out_index,
                    "input_channel": in_index,
                    "expected_s4": selected_values[(out_index, in_index)],
                }
                for out_index in selected_outputs
                for in_index in selected_inputs
            ],
        }
        records.append(
            {
                "ordinal": ordinal,
                "name": name,
                "family": family_name(name),
                "source_weight": {
                    "key": key,
                    "dtype": "BF16",
                    "shape": shape,
                    "bytes": output_count * input_count * 2,
                    "sha256": source_hash.hexdigest(),
                },
                "source_bias": source_bias_record,
                "packed_w4": {
                    "memory_addr": record["packed_w4_addr"],
                    "file_offset": expected_file_offset,
                    "bytes": packed_bytes_written,
                    "sha256": packed_hash.hexdigest(),
                    "qweight_s8_sha256": qweight_hash.hexdigest(),
                },
                "metadata_source_sha256": sha256_bytes(canonical_bytes(metadata)),
                "orientation": "source shape [output_channel,input_channel], row-major output channel",
                "nibble_order": "input channel even low nibble, odd high nibble",
            }
        )

    writer.finish_region("packed_w4")

    record_by_name = {item["name"]: item for item in records}
    for record in linear_records:
        name = record["name"]
        metadata = scales["linears"][name]
        output_count = tensor_shape(record)[0]
        multipliers = metadata["multiplier"]
        shifts = metadata["right_shift"]
        biases = metadata["bias_accumulator"] or [0] * output_count
        require(
            len(multipliers) == len(shifts) == len(biases) == output_count,
            f"projection metadata count differs: {name}",
        )
        meta_base = inventory["image_layout"]["projection_metadata"]["base"]
        expected_file_offset = (
            writer.file_offsets["projection_metadata"]
            + record["projection_metadata_addr"]
            - meta_base
        )
        require(writer.handle.tell() == expected_file_offset, f"metadata address map changed: {name}")
        raw = bytearray(output_count * 16)
        selected = sorted({0, output_count // 2, output_count - 1})
        expected_records = []
        for channel, (multiplier, shift, bias) in enumerate(
            zip(multipliers, shifts, biases, strict=True)
        ):
            packed = pack_projection_record(int(multiplier), int(shift), int(bias))
            raw[channel * 16 : (channel + 1) * 16] = packed
            if channel in selected:
                expected_records.append(
                    {
                        "channel": channel,
                        "multiplier": int(multiplier),
                        "right_shift": int(shift),
                        "output_zero_point": 0,
                        "bias_accumulator": int(bias),
                    }
                )
        require(len(raw) == record["projection_metadata_bytes"], f"metadata byte count differs: {name}")
        writer.write("projection_metadata", bytes(raw))
        record_by_name[name]["projection_metadata"] = {
            "memory_addr": record["projection_metadata_addr"],
            "file_offset": expected_file_offset,
            "bytes": len(raw),
            "sha256": sha256_bytes(bytes(raw)),
            "decode_channels": expected_records,
        }
        decode_expectations[name]["metadata_file_offset"] = expected_file_offset
        decode_expectations[name]["metadata_records"] = expected_records

    writer.finish_region("projection_metadata")
    return records, decode_expectations


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
    require(operator == "post_attention_rmsnorm", f"unknown norm operator: {operator}")
    return f"{prefix}.post_attention_residual", f"{prefix}.post_attention_layernorm.output"


def build_norm_payloads(
    writer: ImageWriter,
    weights: Any,
    inventory: dict[str, Any],
    scales: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    expectations: list[dict[str, Any]] = []
    norm_base = inventory["image_layout"]["rmsnorm_metadata"]["base"]
    for ordinal, record in enumerate(ordered_norms(inventory)):
        require(record["image_bytes"] == RECORD_BYTES, "RMSNorm record size changed")
        key = record["tensor"]["key"]
        tensor = weights.get_tensor(key).contiguous()
        require(tuple(tensor.shape) == (HIDDEN,), f"RMSNorm shape changed: {key}")
        require(tensor.dtype == torch.bfloat16, f"RMSNorm dtype changed: {key}")
        input_key, output_key = norm_metadata_keys(record)
        input_meta = scales["operators"].get(input_key)
        output_meta = scales["operators"].get(output_key)
        require(isinstance(input_meta, dict), f"missing RMSNorm input metadata: {input_key}")
        require(isinstance(output_meta, dict), f"missing RMSNorm output metadata: {output_key}")
        output_scale = float(output_meta["scale"])
        require(math.isfinite(output_scale) and output_scale > 0, f"invalid RMSNorm output scale: {key}")
        gains = torch.round(tensor.to(torch.float64) / output_scale * 256.0).to(torch.int64)
        require(
            not bool(torch.any(gains < -(1 << 15))) and not bool(torch.any(gains > (1 << 15) - 1)),
            f"RMSNorm Q7.8 gain overflow: {key}",
        )
        gains_list = gains.tolist()
        gain_raw = b"".join(struct.pack("<h", int(value)) for value in gains_list)
        require(len(gain_raw) == GAIN_BYTES, "RMSNorm gain bytes changed")
        activation_absmax = float(input_meta["absmax"])
        activation_scale = float(input_meta["scale"])
        require(
            math.isfinite(activation_absmax)
            and activation_absmax > 0
            and math.isfinite(activation_scale)
            and activation_scale > 0,
            f"invalid RMSNorm activation metadata: {key}",
        )
        trailer = struct.pack("<dd", activation_absmax, activation_scale)
        require(len(trailer) == TRAILER_BYTES, "RMSNorm trailer bytes changed")
        payload = gain_raw + trailer
        require(len(payload) == RECORD_BYTES, "RMSNorm payload bytes changed")
        expected_file_offset = (
            writer.file_offsets["rmsnorm_metadata"] + record["image_addr"] - norm_base
        )
        require(writer.handle.tell() == expected_file_offset, f"RMSNorm address map changed: {key}")
        writer.write("rmsnorm_metadata", payload)
        selected = [0, HIDDEN // 2, HIDDEN - 1]
        records.append(
            {
                "ordinal": ordinal,
                "operator": record["operator"],
                "tensor_key": key,
                "source_weight": {
                    "bytes": tensor.numel() * 2,
                    "sha256": sha256_bytes(raw_tensor_bytes(tensor)),
                },
                "memory_addr": record["image_addr"],
                "file_offset": expected_file_offset,
                "bytes": RECORD_BYTES,
                "sha256": sha256_bytes(payload),
                "gain_bytes": GAIN_BYTES,
                "gain_sha256": sha256_bytes(gain_raw),
                "gain_decode": [
                    {"channel": channel, "signed_q7_8": int(gains_list[channel])}
                    for channel in selected
                ],
                "output_scale": output_scale,
                "trailing_activation_metadata": {
                    "byte_offsets": [GAIN_BYTES, RECORD_BYTES - 1],
                    "source_key": input_key,
                    "absmax_f64": activation_absmax,
                    "scale_f64": activation_scale,
                    "sha256": sha256_bytes(trailer),
                    "runtime_executable": False,
                },
            }
        )
        expectations.append(
            {
                "tensor_key": key,
                "file_offset": expected_file_offset,
                "gains": gains_list,
                "trailer": trailer,
            }
        )
    writer.finish_region("rmsnorm_metadata")
    return records, expectations


def build_aux_payloads(
    writer: ImageWriter,
    inventory: dict[str, Any],
    scales: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    expectations: list[dict[str, Any]] = []
    aux_base = inventory["image_layout"]["operator_aux_metadata"]["base"]
    for layer in inventory["layers"]:
        layer_id = int(layer["layer_id"])
        attention_name = f"model.layers.{layer_id}.self_attn"
        attention = scales["attention"][attention_name]
        aux = layer["aux_metadata"]
        conversion_q9 = int(attention["rope_conversion_q9"])
        require(-(1 << 15) <= conversion_q9 < (1 << 15), "RoPE conversion does not fit s16")
        down_meta = scales["linears"][f"model.layers.{layer_id}.mlp.down_proj"]
        silu_real = torch.tensor(
            [1.0 / ((1 << 21) * float(down_meta["input_scale"]))],
            dtype=torch.float64,
        )
        silu_multiplier, silu_shift = derive_multiplier(silu_real)

        payloads: list[tuple[str, bytes, dict[str, Any]]] = []
        rope_q = struct.pack(f"<{HIDDEN}h", *([conversion_q9] * HIDDEN))
        payloads.append(
            (
                "rope_q_scale_records",
                rope_q,
                {"conversion_q9": conversion_q9, "channels": HIDDEN},
            )
        )
        rope_k_channels = KV_HEADS * HEAD_DIM
        rope_k = struct.pack(f"<{rope_k_channels}h", *([conversion_q9] * rope_k_channels))
        payloads.append(
            (
                "rope_k_scale_records",
                rope_k,
                {"conversion_q9": conversion_q9, "channels": rope_k_channels},
            )
        )
        key_scales = attention["key_rope_output_scales"]
        require(len(key_scales) == KV_HEADS, "KV key scale count changed")
        value_scale = float(scales["linears"][f"model.layers.{layer_id}.self_attn.v_proj"]["output_scale"])
        kv_records = [
            ceil_scale32_from_float(float(key_scales[0])),
            ceil_scale32_from_float(float(key_scales[1])),
            ceil_scale32_from_float(value_scale),
            ceil_scale32_from_float(value_scale),
        ]
        for item in kv_records:
            validate_scale32(item)
        kv_raw = struct.pack("<IIII", *kv_records)
        payloads.append(("kv_scale_record", kv_raw, {"scale32_u32": kv_records}))
        score_multiplier = attention["score_multiplier"]
        score_shift = attention["score_right_shift"]
        require(len(score_multiplier) == len(score_shift) == 14, "attention score head count changed")
        score_raw = b"".join(
            pack_projection_record(int(multiplier), int(shift), 0)
            for multiplier, shift in zip(score_multiplier, score_shift, strict=True)
        )
        payloads.append(
            (
                "attention_score_records",
                score_raw,
                {
                    "head_count": 14,
                    "first_middle_last": [
                        {
                            "head": head,
                            "multiplier": int(score_multiplier[head]),
                            "right_shift": int(score_shift[head]),
                        }
                        for head in (0, 7, 13)
                    ],
                },
            )
        )
        silu_raw = pack_projection_record(
            int(silu_multiplier.item()),
            int(silu_shift.item()),
            0,
        )
        payloads.append(
            (
                "silu_record",
                silu_raw,
                {
                    "multiplier": int(silu_multiplier.item()),
                    "right_shift": int(silu_shift.item()),
                    "output_zero_point": 0,
                    "source_down_projection_input_scale": float(down_meta["input_scale"]),
                },
            )
        )

        for name, raw, decoded in payloads:
            descriptor = aux[name]
            require(len(raw) == descriptor["image_bytes"], f"aux byte count differs: layer{layer_id}:{name}")
            expected_file_offset = (
                writer.file_offsets["operator_aux_metadata"]
                + descriptor["image_addr"]
                - aux_base
            )
            require(writer.handle.tell() == expected_file_offset, f"aux address map changed: layer{layer_id}:{name}")
            writer.write("operator_aux_metadata", raw)
            record = {
                "layer": layer_id,
                "name": name,
                "memory_addr": descriptor["image_addr"],
                "file_offset": expected_file_offset,
                "bytes": len(raw),
                "sha256": sha256_bytes(raw),
                "decoded": decoded,
            }
            records.append(record)
            expectations.append(record | {"raw": raw})
    writer.finish_region("operator_aux_metadata")
    return records, expectations


def validate_schedule_coverage(
    inventory: dict[str, Any],
    schedule: dict[str, Any],
) -> dict[str, Any]:
    commands = schedule["commands"]
    require(schedule["command_count"] == len(commands) == 13_914, "schedule command count changed")
    linears = {item["name"]: item for item in ordered_linears(inventory)}
    norms = {item["image_addr"]: item for item in ordered_norms(inventory)}
    aux_by_addr = {
        item["image_addr"]: (layer["layer_id"], name, item)
        for layer in inventory["layers"]
        for name, item in layer["aux_metadata"].items()
    }
    projection_intervals: dict[str, list[tuple[int, int]]] = {name: [] for name in linears}
    metadata_intervals: dict[str, list[tuple[int, int]]] = {name: [] for name in linears}
    norm_references: dict[int, int] = {addr: 0 for addr in norms}
    aux_references: dict[tuple[int, str], int] = {
        (layer, name): 0 for layer, name, _item in aux_by_addr.values()
    }

    projection_operators = {
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "mlp_gate_proj",
        "mlp_up_proj",
        "mlp_down_proj",
        "lm_head_tile",
    }
    norm_operators = {"input_rmsnorm", "post_attention_rmsnorm", "final_rmsnorm"}
    for command in commands:
        operator = command["operator"]
        if operator in projection_operators:
            name = command["weight_tensor"]
            require(name in linears, f"schedule references unknown tensor: {name}")
            record = linears[name]
            channels = int(command["n"])
            reduction = int(command["k"])
            require(reduction == tensor_shape(record)[1], f"schedule reduction differs: {name}")
            weight_start = int(command["src1_addr"])
            meta_start = int(command["scale_addr"])
            weight_bytes = channels * reduction // 2
            meta_bytes = channels * 16
            require(
                record["packed_w4_addr"] <= weight_start
                and weight_start + weight_bytes <= record["packed_w4_addr"] + record["packed_w4_bytes"],
                f"schedule weight range outside tensor: {name}",
            )
            require(
                record["projection_metadata_addr"] <= meta_start
                and meta_start + meta_bytes <= record["projection_metadata_addr"] + record["projection_metadata_bytes"],
                f"schedule metadata range outside tensor: {name}",
            )
            projection_intervals[name].append((weight_start, weight_start + weight_bytes))
            metadata_intervals[name].append((meta_start, meta_start + meta_bytes))
        elif operator in norm_operators:
            address = int(command["scale_addr"])
            require(address in norms, "RMSNorm schedule address changed")
            norm_references[address] += 1
        elif operator == "rope_q":
            address = int(command["scale_addr"])
            require(address in aux_by_addr and aux_by_addr[address][1] == "rope_q_scale_records", "RoPE-Q metadata address changed")
            layer, name, _item = aux_by_addr[address]
            aux_references[(layer, name)] += 1
        elif operator == "rope_k":
            address = int(command["scale_addr"])
            require(address in aux_by_addr and aux_by_addr[address][1] == "rope_k_scale_records", "RoPE-K metadata address changed")
            layer, name, _item = aux_by_addr[address]
            aux_references[(layer, name)] += 1
        elif operator == "kv_write":
            address = int(command["scale_addr"])
            require(address in aux_by_addr and aux_by_addr[address][1] == "kv_scale_record", "KV metadata address changed")
            layer, name, _item = aux_by_addr[address]
            aux_references[(layer, name)] += 1
        elif operator == "attention_score":
            address = int(command["scale_addr"])
            containing = [
                (layer, name, item)
                for layer, name, item in aux_by_addr.values()
                if name == "attention_score_records"
                and item["image_addr"] <= address < item["image_addr"] + item["image_bytes"]
            ]
            require(len(containing) == 1, "attention score metadata address changed")
            layer, name, item = containing[0]
            require((address - item["image_addr"]) % 16 == 0, "attention score record is unaligned")
            aux_references[(layer, name)] += 1
        elif operator == "silu_gate":
            address = int(command["scale_addr"])
            require(address in aux_by_addr and aux_by_addr[address][1] == "silu_record", "SiLU metadata address changed")
            layer, name, _item = aux_by_addr[address]
            aux_references[(layer, name)] += 1

    for name, record in linears.items():
        require(
            interval_union(projection_intervals[name])
            == [(record["packed_w4_addr"], record["packed_w4_addr"] + record["packed_w4_bytes"])],
            f"schedule does not cover all packed bytes: {name}",
        )
        require(
            interval_union(metadata_intervals[name])
            == [
                (
                    record["projection_metadata_addr"],
                    record["projection_metadata_addr"] + record["projection_metadata_bytes"],
                )
            ],
            f"schedule does not cover all metadata bytes: {name}",
        )
    require(set(norm_references.values()) == {2}, "each RMSNorm record must be referenced once per token")
    require(len(norm_references) == 49 and sum(norm_references.values()) == 98, "RMSNorm schedule coverage changed")
    expected_aux_counts = {
        "rope_q_scale_records": 2,
        "rope_k_scale_records": 2,
        "kv_scale_record": 2,
        "attention_score_records": 42,
        "silu_record": 2,
    }
    for (layer, name), count in aux_references.items():
        require(count == expected_aux_counts[name], f"aux schedule reference count changed: layer{layer}:{name}")
    return {
        "status": "PASS",
        "command_count": len(commands),
        "linear_tensor_count": len(linears),
        "rmsnorm_command_count": sum(norm_references.values()),
        "rmsnorm_unique_scale_addresses": len(norm_references),
        "operator_aux_record_count": len(aux_references),
        "all_linear_weight_and_metadata_ranges_covered": True,
        "all_rmsnorm_and_aux_records_referenced": True,
    }


def validate_rmsnorm_access(
    inventory: dict[str, Any],
    schedule: dict[str, Any],
) -> dict[str, Any]:
    shell_text = SHELL.read_text(encoding="utf-8")
    required_statements = [
        "ST_SCALE_ACT_RECV:\n                    prefix_mem_req_addr_q <= scale_addr_q + (beat_idx_ext_w << 5);",
        "ST_GAIN_RECV0:\n                    prefix_mem_req_addr_q <=\n                        scale_addr_q + (beat_idx_ext_w << 5) + 64'd16;",
        "ST_ACT_RECV: begin\n                    if (accepted_read_w) begin\n                        if (beat_idx_q == LAST_BEAT) begin\n                            prefix_mem_req_addr_q <= src0_addr_q;",
    ]
    for statement in required_statements:
        require(statement in shell_text, "RMSNorm executable address statement changed")
    norm_bases = {item["image_addr"] for item in ordered_norms(inventory)}
    rms_commands = [
        item
        for item in schedule["commands"]
        if item["operator"] in {"input_rmsnorm", "post_attention_rmsnorm", "final_rmsnorm"}
    ]
    require(len(rms_commands) == 98, "RMSNorm command count changed")
    require({item["scale_addr"] for item in rms_commands} == norm_bases, "RMSNorm command bases changed")
    consumed_offsets: list[int] = []
    for beat in range(56):
        consumed_offsets.extend(range(beat * 32, beat * 32 + 16))
        consumed_offsets.extend(range(beat * 32 + 16, beat * 32 + 32))
    require(consumed_offsets == list(range(GAIN_BYTES)), "RMSNorm gain access is not contiguous and ordered")
    require(max(consumed_offsets) == GAIN_BYTES - 1, "RMSNorm gain access endpoint changed")
    require(not set(consumed_offsets).intersection(range(GAIN_BYTES, RECORD_BYTES)), "RMSNorm trailer is executable")
    return {
        "status": "PASS",
        "commands_checked": len(rms_commands),
        "record_bases_checked": len(norm_bases),
        "gain_beats_per_command": 56,
        "gain_bytes_per_command": len(consumed_offsets),
        "consumed_byte_offsets": [0, GAIN_BYTES - 1],
        "trailer_byte_offsets": [GAIN_BYTES, RECORD_BYTES - 1],
        "all_896_gains_consumed_once_in_channel_order": True,
        "trailer_reachable_as_gain_data": False,
        "activation_scale_source": "src0_addr",
    }


def validate_region_map(inventory: dict[str, Any]) -> dict[str, Any]:
    memory_intervals = []
    file_offsets = region_file_offsets(inventory)
    file_intervals = []
    for name in REGION_ORDER:
        item = inventory["image_layout"][name]
        require(item["base"] % 16 == item["end"] % 16 == 0, f"{name} memory alignment changed")
        memory_intervals.append((item["base"], item["end"], name))
        file_intervals.append(
            (file_offsets[name], file_offsets[name] + item["bytes"], name)
        )
    for intervals, label in ((memory_intervals, "memory"), (file_intervals, "file")):
        ordered = sorted(intervals)
        for left, right in zip(ordered, ordered[1:]):
            require(left[1] <= right[0], f"{label} regions overlap: {left[2]} and {right[2]}")
    require(file_intervals[-1][1] == TOTAL_IMAGE_BYTES, "dense image endpoint changed")
    return {
        "status": "PASS",
        "region_count": len(REGION_ORDER),
        "memory_regions_nonoverlapping": True,
        "file_regions_contiguous_nonoverlapping": True,
        "alignment_bytes": 16,
        "total_image_bytes": TOTAL_IMAGE_BYTES,
    }


def verify_image_decodes(
    image_path: Path,
    linear_expectations: dict[str, dict[str, Any]],
    norm_expectations: list[dict[str, Any]],
    aux_expectations: list[dict[str, Any]],
) -> dict[str, Any]:
    require(image_path.stat().st_size == TOTAL_IMAGE_BYTES, "decode target image length changed")
    linear_checks = []
    with image_path.open("rb") as handle:
        for name, expectation in linear_expectations.items():
            input_count = expectation["input_count"]
            for item in expectation["values"]:
                element = item["output_channel"] * input_count + item["input_channel"]
                byte_offset = expectation["packed_file_offset"] + element // 2
                handle.seek(byte_offset)
                actual = decode_s4(handle.read(1), element & 1)
                require(actual == item["expected_s4"], f"packed decode differs: {name}:{item}")
            for item in expectation["metadata_records"]:
                handle.seek(expectation["metadata_file_offset"] + item["channel"] * 16)
                actual = unpack_projection_record(handle.read(16))
                require(actual == {k: item[k] for k in actual}, f"metadata decode differs: {name}:{item['channel']}")
            linear_checks.append(
                {
                    "name": name,
                    "weight_decode_points": len(expectation["values"]),
                    "metadata_decode_channels": [item["channel"] for item in expectation["metadata_records"]],
                }
            )

        norm_checks = []
        for expectation in norm_expectations:
            handle.seek(expectation["file_offset"])
            gain_raw = handle.read(GAIN_BYTES)
            decoded = list(struct.unpack(f"<{HIDDEN}h", gain_raw))
            require(decoded == expectation["gains"], f"RMSNorm full gain decode differs: {expectation['tensor_key']}")
            trailer = handle.read(TRAILER_BYTES)
            require(trailer == expectation["trailer"], f"RMSNorm trailer decode differs: {expectation['tensor_key']}")
            norm_checks.append(
                {
                    "tensor_key": expectation["tensor_key"],
                    "all_gain_channels_decoded": HIDDEN,
                    "trailer_bytes_decoded": TRAILER_BYTES,
                }
            )

        aux_checks = []
        for expectation in aux_expectations:
            handle.seek(expectation["file_offset"])
            raw = handle.read(expectation["bytes"])
            require(raw == expectation["raw"], f"operator aux decode differs: layer{expectation['layer']}:{expectation['name']}")
            if expectation["name"] == "kv_scale_record":
                for item in struct.unpack("<IIII", raw):
                    validate_scale32(item)
            elif expectation["name"] in {"attention_score_records", "silu_record"}:
                for offset in range(0, len(raw), 16):
                    unpack_projection_record(raw[offset : offset + 16])
            aux_checks.append(
                {
                    "layer": expectation["layer"],
                    "name": expectation["name"],
                    "bytes_decoded": len(raw),
                }
            )

    family_counts: dict[str, int] = {}
    for item in linear_checks:
        family_counts[family_name(item["name"])] = family_counts.get(family_name(item["name"]), 0) + 1
    require(set(family_counts) == {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "lm_head"}, "tensor family decode coverage changed")
    return {
        "status": "PASS",
        "linear_tensors_checked": len(linear_checks),
        "linear_family_counts": family_counts,
        "weight_decode_points_per_tensor": 9,
        "metadata_decode_channels_per_tensor": 3,
        "rmsnorm_records_full_decoded": len(norm_checks),
        "rmsnorm_gain_channels_per_record": HIDDEN,
        "operator_aux_records_exact_decoded": len(aux_checks),
    }


def write_sums(directory: Path, names: list[str]) -> None:
    rows = [f"{sha256_file(directory / name)}  {name}" for name in sorted(names)]
    (directory / "SHA256SUMS").write_text("\n".join(rows) + "\n", encoding="utf-8")


def build_once(
    directory: Path,
    generated_at_utc: str,
    inventory: dict[str, Any],
    schedule: dict[str, Any],
    scales: dict[str, Any],
    contract: dict[str, Any],
    contract_sha256: str,
    static_validation: dict[str, Any],
) -> dict[str, Any]:
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True)
    contract_wrapper = {"contract": contract, "contract_sha256": contract_sha256}
    write_json(directory / "image_contract_v2.json", contract_wrapper)
    image_tmp = directory / "full_model_image.bin.tmp"
    writer = ImageWriter(image_tmp, inventory)
    with safe_open(MODEL, framework="pt", device="cpu") as weights:
        linear_records, linear_expectations = build_linear_payloads(
            writer, weights, inventory, scales
        )
        norm_records, norm_expectations = build_norm_payloads(
            writer, weights, inventory, scales
        )
        aux_records, aux_expectations = build_aux_payloads(writer, inventory, scales)
    image_record = writer.close()
    image_path = directory / "full_model_image.bin"
    image_tmp.replace(image_path)
    require(sha256_file(image_path) == image_record["sha256"], "streamed image hash differs from file hash")
    decode_validation = verify_image_decodes(
        image_path,
        linear_expectations,
        norm_expectations,
        aux_expectations,
    )
    validation = {
        "schema_version": 1,
        "mission_id": MISSION_ID,
        "generated_at_utc": generated_at_utc,
        "status": "PASS",
        "contract_sha256": contract_sha256,
        "checks": static_validation | {"image_decode": decode_validation},
        "counts": {
            "linear_tensors": len(linear_records),
            "rmsnorm_records": len(norm_records),
            "operator_aux_records": len(aux_records),
            "total_image_bytes": image_record["bytes"],
        },
        "scope_guards": contract["scope_guards"],
    }
    write_json(directory / "validation_report.json", validation)
    manifest = {
        "schema_version": 1,
        "mission_id": MISSION_ID,
        "generated_at_utc": generated_at_utc,
        "status": "COMPLETE_IMAGE_BUILT_AND_SELF_VERIFIED_PENDING_FRESH_REVIEW",
        "contract": {
            "path": "image_contract_v2.json",
            "bytes": (directory / "image_contract_v2.json").stat().st_size,
            "sha256": sha256_file(directory / "image_contract_v2.json"),
            "contract_sha256": contract_sha256,
        },
        "image": {
            "path": "full_model_image.bin",
            **image_record,
        },
        "validation_report": {
            "path": "validation_report.json",
            "bytes": (directory / "validation_report.json").stat().st_size,
            "sha256": sha256_file(directory / "validation_report.json"),
        },
        "linear_tensors": linear_records,
        "rmsnorm_records": norm_records,
        "operator_aux_records": aux_records,
        "input_hashes": contract["immutable_v1_boundary"]
        | contract["frozen_scale_provenance"]
        | {"source_model": contract["source_model"]["safetensors"]},
        "scope_guards": contract["scope_guards"],
    }
    write_json(directory / "manifest.json", manifest)
    write_sums(
        directory,
        [
            "full_model_image.bin",
            "image_contract_v2.json",
            "manifest.json",
            "validation_report.json",
        ],
    )
    return {
        "image": file_record(image_path),
        "contract": file_record(directory / "image_contract_v2.json"),
        "manifest": file_record(directory / "manifest.json"),
        "validation": file_record(directory / "validation_report.json"),
        "sums": file_record(directory / "SHA256SUMS"),
    }


def files_equal(left: Path, right: Path) -> bool:
    require(left.stat().st_size == right.stat().st_size, f"reproducibility size differs: {left.name}")
    with left.open("rb") as a, right.open("rb") as b:
        while True:
            chunk_a = a.read(1 << 20)
            chunk_b = b.read(1 << 20)
            if chunk_a != chunk_b:
                return False
            if not chunk_a:
                return True


def hardlink_replace(source: Path, destination: Path) -> None:
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    os.link(source, destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "evidence/verification" / MISSION_ID,
    )
    parser.add_argument("--generated-at-utc", required=True)
    args = parser.parse_args()
    require(
        EXPECTED_TIMESTAMP.fullmatch(args.generated_at_utc) is not None,
        "--generated-at-utc must use YYYY-MM-DDTHH:MM:SSZ",
    )
    input_records = validate_inputs()
    inventory = json.loads(INVENTORY.read_text(encoding="utf-8"))
    schedule = json.loads(SCHEDULE.read_text(encoding="utf-8"))
    scales = json.loads(SCALES.read_text(encoding="utf-8"))
    require(scales.get("schema_version") == 2, "frozen scale schema changed")
    require(len(scales.get("linears", {})) == 169, "frozen linear scale inventory changed")
    require(len(scales.get("attention", {})) == 24, "frozen attention scale inventory changed")
    require(len(scales.get("operators", {})) == 122, "frozen operator scale inventory changed")

    static_validation = {
        "region_map": validate_region_map(inventory),
        "schedule_coverage": validate_schedule_coverage(inventory, schedule),
        "rmsnorm_access": validate_rmsnorm_access(inventory, schedule),
        "immutable_inputs": {"status": "PASS", "records": input_records},
    }
    contract = contract_body(args.generated_at_utc, inventory, input_records)
    contract_sha256 = sha256_bytes(canonical_bytes(contract))

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    build_a = output / "reproducibility/build_a"
    build_b = output / "reproducibility/build_b"
    first = build_once(
        build_a,
        args.generated_at_utc,
        inventory,
        schedule,
        scales,
        contract,
        contract_sha256,
        static_validation,
    )
    second = build_once(
        build_b,
        args.generated_at_utc,
        inventory,
        schedule,
        scales,
        contract,
        contract_sha256,
        static_validation,
    )

    compared_names = [
        "full_model_image.bin",
        "image_contract_v2.json",
        "manifest.json",
        "validation_report.json",
        "SHA256SUMS",
    ]
    comparisons = {}
    for name in compared_names:
        left = build_a / name
        right = build_b / name
        require(files_equal(left, right), f"two-build byte comparison failed: {name}")
        comparisons[name] = {
            "bytes": left.stat().st_size,
            "sha256": sha256_file(left),
            "byte_identical": True,
        }

    for name in compared_names:
        if name != "SHA256SUMS":
            hardlink_replace(build_a / name, output / name)
    reproducibility = {
        "schema_version": 1,
        "mission_id": MISSION_ID,
        "generated_at_utc": args.generated_at_utc,
        "status": "PASS_TWO_CLEAN_BUILDS_BYTE_IDENTICAL",
        "build_a": first,
        "build_b": second,
        "comparisons": comparisons,
        "frozen_timestamp_equal": True,
        "frozen_inputs_equal": True,
        "manifest_bytes_equal": True,
        "image_bytes_equal": True,
    }
    write_json(output / "reproducibility.json", reproducibility)
    write_sums(
        output,
        [
            "full_model_image.bin",
            "image_contract_v2.json",
            "manifest.json",
            "reproducibility.json",
            "validation_report.json",
        ],
    )

    validate_inputs()
    require(sha256_file(V1_AUDIT) == EXPECTED_INPUT_HASHES[V1_AUDIT], "v1 audit changed during build")
    require(sha256_file(SHELL) == EXPECTED_INPUT_HASHES[SHELL], "RTL changed during build")
    require(sha256_file(SHELL_TB) == EXPECTED_INPUT_HASHES[SHELL_TB], "testbench changed during build")
    print(
        "ACE2_FULL_QWEN_IMAGE_V2_PASS "
        f"bytes={TOTAL_IMAGE_BYTES} "
        f"sha256={comparisons['full_model_image.bin']['sha256']} "
        f"manifest_sha256={comparisons['manifest.json']['sha256']} "
        f"output={output.relative_to(ROOT)}"
    )


if __name__ == "__main__":
    main()
