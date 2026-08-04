#!/usr/bin/env python3
"""Paired BF16 and contract-faithful fixed-point W4A8 Qwen2.5 reference."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import re
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from datasets import load_dataset
from torch import Tensor, nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from ace2_quality_contracts import (
    SCALE32_ALL_ZERO_RECORD,
    ceil_scale32_from_float,
    ceil_scale32_from_ratio,
    dynamic_rope_output_scale,
    dynamic_score_pair_parameters,
    fixed_q7_score_pair_parameters,
    pack_scale32,
    round_divide_even_signed,
    scale32_ratio,
    unpack_scale32,
    validate_rtl_binding,
)
from ace2_relative_rope_score_reference import relative_coefficients_q15
from ace2_absolute_rope_online_attention_reference import (
    K_SCALE32 as ABSOLUTE_ROPE_K_SCALE32,
    Q_SCALE32 as ABSOLUTE_ROPE_Q_SCALE32,
    OnlineAttentionState as AbsoluteOnlineAttentionState,
    absolute_coefficients_q15,
    exp_table_q31 as absolute_exp_table_q31,
    finalize_online_state as finalize_absolute_online_state,
    exp_q31 as absolute_exp_q31,
    score_raw_to_logit_q12_20,
    update_online_state as update_absolute_online_state,
)
from ace2_down_projection_residual_fusion_hook import (
    ExactScale32AllLayerHook,
    FusionMetadata,
)
from ace2_down_projection_residual_fusion_reference import fuse_lane
from ace2_cross_layer_error_carry_hook import (
    CrossLayerCarryMetadata,
    CrossLayerErrorCarryRuntime,
)


ROOT = Path(__file__).resolve().parents[1]
PROMPT_MANIFEST = ROOT / "benchmark" / "quality" / "PROMPT_MANIFEST.json"
QUALITY_CONFIG = ROOT / "benchmark" / "quality" / "QUALITY_CONFIG.json"
QUALITY_REQUIREMENTS = ROOT / "benchmark" / "quality" / "requirements.txt"
LM_EVAL_TASKS = ROOT / "benchmark" / "quality" / "lm_eval_tasks"
QK_RESIDUAL_METADATA = (
    ROOT / "reference" / "generated" / "qk_residual_scale32_metadata.json"
)
V_RESIDUAL_METADATA = (
    ROOT / "reference" / "generated" / "v_residual_scale32_metadata.json"
)
DPRF_METADATA = (
    ROOT
    / "reference"
    / "generated"
    / "down_projection_residual_fusion_metadata.json"
)
QECR_METADATA = (
    ROOT / "reference" / "generated" / "cross_layer_error_carry_metadata.json"
)
INT32_MAX = (1 << 31) - 1
INT64_MAX = (1 << 63) - 1
RMS_HIDDEN_SIZE = 896
RMS_GAIN_FRAC = 8
RMS_INV_FRAC = 30
ROPE_HEAD_DIM = 64
ROPE_SCALE_FRAC = 9
ROPE_SCALE_Q9_MAX = 32767
ROPE_SAFE_CONVERSION_Q9 = math.floor(
    127 * (1 << ROPE_SCALE_FRAC) / (128 * math.sqrt(2.0))
)
ROPE_UNIT_CONVERSION_Q9 = 1 << ROPE_SCALE_FRAC
ROPE_DIAGNOSTIC_MECHANISMS = {
    "scalar_pair_rotate45_safe_int8": (ROPE_SAFE_CONVERSION_Q9, 8),
    "scalar_pair_rotate_safe_int8": (ROPE_SAFE_CONVERSION_Q9, 8),
    "scalar_safe_int8": (ROPE_SAFE_CONVERSION_Q9, 8),
    "scalar_unit_gain_int8": (ROPE_UNIT_CONVERSION_Q9, 8),
    "scalar_unit_gain_int9": (ROPE_UNIT_CONVERSION_Q9, 9),
    "dynamic_rope_head_scale_v1": (ROPE_UNIT_CONVERSION_Q9, 8),
    "layer0_fixed_q7_rope_score_v1": (ROPE_UNIT_CONVERSION_Q9, 16),
    "layer0_relative_rope_score_fusion_v1": (ROPE_UNIT_CONVERSION_Q9, 8),
    "layer0_absolute_rope_online_attention_v1": (ROPE_UNIT_CONVERSION_Q9, 8),
    "layer0_projection_shadow_staged_attention_v1": (ROPE_UNIT_CONVERSION_Q9, 8),
    "layer0_tile_max_delta_attention_v1": (ROPE_UNIT_CONVERSION_Q9, 8),
    "layer0_tile_bfp_score_attention_v1": (ROPE_UNIT_CONVERSION_Q9, 8),
    "shared_native_accumulator_tagged_attention_v1": (ROPE_UNIT_CONVERSION_Q9, 32),
    "shared_qk_residual_cross_term_baseline_v1": (ROPE_UNIT_CONVERSION_Q9, 8),
    "shared_qk_residual_cross_term_attention_v1": (ROPE_UNIT_CONVERSION_Q9, 8),
    "shared_v_residual_value_correction_baseline_v1": (ROPE_UNIT_CONVERSION_Q9, 8),
    "shared_v_residual_value_correction_attention_v1": (ROPE_UNIT_CONVERSION_Q9, 8),
    "shared_down_projection_residual_fusion_baseline_v1": (ROPE_UNIT_CONVERSION_Q9, 8),
    "shared_down_projection_residual_fusion_v1": (ROPE_UNIT_CONVERSION_Q9, 8),
    "cross_layer_quantization_error_carry_final_output_v1": (
        ROPE_UNIT_CONVERSION_Q9,
        8,
    ),
}
ACTIVE_ROPE_MECHANISM = "dynamic_rope_head_scale_v1"
ACTIVE_ROPE_PAIR_ROTATION_DEGREES = None
ATTENTION_SCORE_FRAC = 9
ATTENTION_HEAD_SCALE_SHIFT = 3
SOFTMAX_PROB_FRAC = 15
TILE_DELTA_FRAC = 17
TILE_DELTA_SENTINEL = -(16 << TILE_DELTA_FRAC)
TAGGED_SCORE_FRAC = 44
TAGGED_EXP_INPUT_FRAC = 26
TAGGED_SCORE_SENTINEL = -(16 << TAGGED_SCORE_FRAC)
SOFTMAX_EXP_STEP = 64
SOFTMAX_EXP_ROUND = 32
SOFTMAX_EXP_LUT = [
    32768,
    28918,
    25520,
    22521,
    19875,
    17539,
    15479,
    13660,
    12055,
    10638,
    9388,
    8285,
    7312,
    6452,
    5694,
    5025,
    4435,
    3914,
    3454,
    3048,
    2690,
    2374,
    2095,
    1849,
    1631,
    1440,
    1271,
    1121,
    990,
    873,
    771,
    680,
    600,
    530,
    467,
    412,
    364,
    321,
    283,
    250,
    221,
    195,
    172,
    152,
    134,
    118,
    104,
    92,
    81,
    72,
    63,
    56,
    49,
    43,
    38,
    34,
    30,
    26,
    23,
    21,
    18,
    16,
    14,
    12,
    11,
]


def rope_linear_contract(
    module_name: str,
    rope_diagnostic_mechanism: str | None,
) -> tuple[bool, bool]:
    """Return (use_per_head_output_scales, emit_scale32) for one linear."""
    is_qk_projection = module_name.endswith(
        (".self_attn.q_proj", ".self_attn.k_proj")
    )
    if not is_qk_projection:
        return False, False
    if rope_diagnostic_mechanism in {
        "layer0_fixed_q7_rope_score_v1",
        "layer0_relative_rope_score_fusion_v1",
        "layer0_absolute_rope_online_attention_v1",
        "layer0_projection_shadow_staged_attention_v1",
        "layer0_tile_max_delta_attention_v1",
        "layer0_tile_bfp_score_attention_v1",
    }:
        is_layer0 = module_name.startswith("model.layers.0.self_attn.")
        return not is_layer0, is_layer0
    if rope_diagnostic_mechanism == "shared_native_accumulator_tagged_attention_v1":
        return False, False
    if rope_diagnostic_mechanism in {
        "shared_qk_residual_cross_term_baseline_v1",
        "shared_qk_residual_cross_term_attention_v1",
        "shared_v_residual_value_correction_baseline_v1",
        "shared_v_residual_value_correction_attention_v1",
        "shared_down_projection_residual_fusion_baseline_v1",
        "shared_down_projection_residual_fusion_v1",
        "cross_layer_quantization_error_carry_final_output_v1",
    }:
        return True, False
    return (
        rope_diagnostic_mechanism is None,
        rope_diagnostic_mechanism == "dynamic_rope_head_scale_v1",
    )


def rope_mechanism_for_layer(
    layer_index: int,
    rope_diagnostic_mechanism: str | None,
) -> str | None:
    """Keep reviewed structural score replacements local to layer 0."""
    if (
        rope_diagnostic_mechanism
        in {
            "layer0_fixed_q7_rope_score_v1",
            "layer0_relative_rope_score_fusion_v1",
            "layer0_absolute_rope_online_attention_v1",
            "layer0_projection_shadow_staged_attention_v1",
            "layer0_tile_max_delta_attention_v1",
            "layer0_tile_bfp_score_attention_v1",
        }
        and layer_index != 0
    ):
        return None
    return rope_diagnostic_mechanism
SILU_LUT = [
    max(-32768, min(32767, round((index / 8.0) / (1.0 + math.exp(-index / 8.0)) * (1 << 12))))
    for index in range(-64, 65)
]
EXPECTED_SOFTWARE = {
    "datasets": "4.8.5",
    "lm_eval": "0.4.9.2",
    "torch": "2.11.0",
    "transformers": "4.57.6",
}


def utc_now() -> str:
    override = os.environ.get("ACE2_EVIDENCE_UTC")
    if override:
        datetime.strptime(override, "%Y-%m-%dT%H:%M:%SZ")
        return override
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def canonical_json_sha256(value: dict[str, Any]) -> str:
    clone = json.loads(json.dumps(value))
    clone.setdefault("integrity", {})["canonical_sha256"] = None
    payload = json.dumps(clone, indent=2, sort_keys=True) + "\n"
    return hashlib.sha256(payload.encode()).hexdigest()


@lru_cache(maxsize=1)
def load_qk_residual_scale32_metadata() -> dict[str, Any]:
    metadata = json.loads(QK_RESIDUAL_METADATA.read_text(encoding="utf-8"))
    if metadata.get("contract_id") != "shared_qk_residual_cross_term_attention_v1":
        raise ValueError("Q/K residual metadata contract differs")
    if metadata.get("model") != {
        "repository": "Qwen/Qwen2.5-0.5B",
        "revision": "060db6499f32faf8b98477b0a26969ef7d8b9987",
        "layers": 24,
        "query_heads_per_layer": 14,
        "kv_heads_per_layer": 2,
    }:
        raise ValueError("Q/K residual metadata model geometry differs")
    expected = metadata.get("integrity", {}).get("canonical_sha256")
    if expected != canonical_json_sha256(metadata):
        raise ValueError("Q/K residual metadata canonical hash differs")
    records = metadata.get("records")
    if not isinstance(records, list) or len(records) != 24 * (14 + 2):
        raise ValueError("Q/K residual metadata must contain 384 ordered rows")
    for source in metadata.get("source_artifacts", []):
        if source["path"].startswith("tools/"):
            continue
        path = ROOT / source["path"]
        if not path.is_file() or sha256_file(path) != source["sha256"]:
            raise ValueError(f"Q/K residual metadata source differs: {source['path']}")
    return metadata


@lru_cache(maxsize=24)
def qk_residual_layer_metadata(layer_index: int) -> dict[str, list[Any]]:
    if not 0 <= layer_index < 24:
        raise ValueError("Q/K residual layer index must be 0..23")
    metadata = load_qk_residual_scale32_metadata()
    selected = [row for row in metadata["records"] if row["layer"] == layer_index]
    expected_order = [
        *(('q_proj', head) for head in range(14)),
        *(('k_proj', head) for head in range(2)),
    ]
    observed_order = [(row["projection"], row["head"]) for row in selected]
    if observed_order != expected_order:
        raise ValueError(f"Q/K residual metadata row order differs for layer {layer_index}")
    result: dict[str, list[Any]] = {}
    for projection, head_count in (("q_proj", 14), ("k_proj", 2)):
        rows = [row for row in selected if row["projection"] == projection]
        if len(rows) != head_count:
            raise ValueError(f"Q/K residual metadata head count differs for {projection}")
        result[f"{projection}_output_absmax"] = [
            float(row["baseline_source_decimal"]) * 127.0 for row in rows
        ]
        result[f"{projection}_baseline_scale32"] = [
            int(row["baseline_scale32"]["packed_u32"]) for row in rows
        ]
        result[f"{projection}_residual_scale32"] = [
            int(row["residual_scale32"]["packed_u32"]) for row in rows
        ]
    return result


def qk_residual_projection_output_absmax(
    module_name: str,
    rope_diagnostic_mechanism: str | None,
) -> list[float] | None:
    if rope_diagnostic_mechanism not in {
        "shared_qk_residual_cross_term_baseline_v1",
        "shared_qk_residual_cross_term_attention_v1",
        "shared_v_residual_value_correction_baseline_v1",
        "shared_v_residual_value_correction_attention_v1",
        "shared_down_projection_residual_fusion_baseline_v1",
        "shared_down_projection_residual_fusion_v1",
        "cross_layer_quantization_error_carry_final_output_v1",
    }:
        return None
    match = re.fullmatch(r"model\.layers\.(\d+)\.self_attn\.(q_proj|k_proj)", module_name)
    if match is None:
        return None
    layer_index = int(match.group(1))
    projection = match.group(2)
    return qk_residual_layer_metadata(layer_index)[f"{projection}_output_absmax"]


@lru_cache(maxsize=1)
def load_v_residual_scale32_metadata() -> dict[str, Any]:
    metadata = json.loads(V_RESIDUAL_METADATA.read_text(encoding="utf-8"))
    if metadata.get("contract_id") != "shared_v_residual_value_correction_attention_v1":
        raise ValueError("V residual metadata contract differs")
    model = metadata.get("model", {})
    expected_model = {
        "repository": "Qwen/Qwen2.5-0.5B",
        "revision": "060db6499f32faf8b98477b0a26969ef7d8b9987",
        "resolved_revision": "060db6499f32faf8b98477b0a26969ef7d8b9987",
        "layers": 24,
        "kv_heads_per_layer": 2,
    }
    if model != expected_model:
        raise ValueError("V residual metadata model geometry differs")
    expected = metadata.get("integrity", {}).get("canonical_sha256")
    if expected != canonical_json_sha256(metadata):
        raise ValueError("V residual metadata canonical hash differs")
    records = metadata.get("records")
    if not isinstance(records, list) or len(records) != 48:
        raise ValueError("V residual metadata must contain 48 ordered rows")
    observed_order = [(row.get("layer"), row.get("kv_head")) for row in records]
    expected_order = [(layer, kv_head) for layer in range(24) for kv_head in range(2)]
    if observed_order != expected_order:
        raise ValueError("V residual metadata row order differs")
    for source in metadata.get("source_artifacts", []):
        path = ROOT / source["path"]
        if not path.is_file() or sha256_file(path) != source["sha256"]:
            raise ValueError(f"V residual metadata source differs: {source['path']}")
    return metadata


@lru_cache(maxsize=24)
def v_residual_layer_metadata(layer_index: int) -> dict[str, Any]:
    if not 0 <= layer_index < 24:
        raise ValueError("V residual layer index must be 0..23")
    rows = [
        row
        for row in load_v_residual_scale32_metadata()["records"]
        if row["layer"] == layer_index
    ]
    if [row["kv_head"] for row in rows] != [0, 1]:
        raise ValueError(f"V residual KV-head order differs for layer {layer_index}")
    baseline_records = [int(row["baseline_v_scale32"]["packed_u32"]) for row in rows]
    residual_records = [int(row["residual_v_scale32"]["packed_u32"]) for row in rows]
    if baseline_records[0] != baseline_records[1]:
        raise ValueError("frozen baseline V Scale32 must be shared by both KV heads")
    if rows[0]["baseline_source_decimal"] != rows[1]["baseline_source_decimal"]:
        raise ValueError("frozen baseline V source scale differs by KV head")
    return {
        "baseline_output_scale": float(rows[0]["baseline_source_decimal"]),
        "baseline_v_scale32": baseline_records,
        "residual_v_scale32": residual_records,
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tensor(value: Tensor) -> str:
    data = value.detach().cpu().contiguous().numpy().tobytes()
    return hashlib.sha256(data).hexdigest()


@lru_cache(maxsize=1)
def load_down_projection_residual_fusion_metadata() -> dict[str, Any]:
    metadata = json.loads(DPRF_METADATA.read_text(encoding="utf-8"))
    if metadata.get("contract_id") != "shared_down_projection_residual_fusion_v1":
        raise ValueError("down-projection residual-fusion metadata contract differs")
    model = metadata.get("model", {})
    if model.get("repository") != "Qwen/Qwen2.5-0.5B":
        raise ValueError("down-projection residual-fusion model repository differs")
    if model.get("revision") != "060db6499f32faf8b98477b0a26969ef7d8b9987":
        raise ValueError("down-projection residual-fusion model revision differs")
    layers = metadata.get("layers")
    if not isinstance(layers, list) or len(layers) != 24:
        raise ValueError("down-projection residual-fusion metadata must contain 24 layers")
    if [layer.get("layer_id") for layer in layers] != list(range(24)):
        raise ValueError("down-projection residual-fusion layer order differs")
    for layer in layers:
        records = layer.get("accumulator_scale32")
        if not isinstance(records, list) or len(records) != RMS_HIDDEN_SIZE:
            raise ValueError("down-projection residual-fusion lane count differs")
        for record in records:
            unpack_scale32(int(record))
        unpack_scale32(int(layer["residual_scale32"]))
        unpack_scale32(int(layer["destination_scale32"]))
    binary = DPRF_METADATA.with_suffix(".bin")
    if sha256_file(binary) != metadata.get("identity", {}).get("immutable_image_sha256"):
        raise ValueError("down-projection residual-fusion metadata image hash differs")
    return metadata


@lru_cache(maxsize=1)
def down_projection_residual_fusion_metadata() -> FusionMetadata:
    layers = load_down_projection_residual_fusion_metadata()["layers"]
    return FusionMetadata.from_sequences(
        [layer["accumulator_scale32"] for layer in layers],
        [layer["residual_scale32"] for layer in layers],
        [layer["destination_scale32"] for layer in layers],
    )


@lru_cache(maxsize=1)
def load_cross_layer_error_carry_metadata() -> dict[str, Any]:
    metadata = json.loads(QECR_METADATA.read_text(encoding="utf-8"))
    if metadata.get("contract_id") != "cross_layer_quantization_error_carry_final_output_v1":
        raise ValueError("cross-layer error-carry metadata contract differs")
    model = metadata.get("model", {})
    if model.get("repository") != "Qwen/Qwen2.5-0.5B":
        raise ValueError("cross-layer error-carry model repository differs")
    if model.get("revision") != "060db6499f32faf8b98477b0a26969ef7d8b9987":
        raise ValueError("cross-layer error-carry model revision differs")
    layers = metadata.get("layers")
    if not isinstance(layers, list) or len(layers) != 24:
        raise ValueError("cross-layer error-carry metadata must contain 24 layers")
    if [layer.get("layer_id") for layer in layers] != list(range(24)):
        raise ValueError("cross-layer error-carry layer order differs")
    for layer in layers:
        records = layer.get("accumulator_scale32")
        if not isinstance(records, list) or len(records) != RMS_HIDDEN_SIZE:
            raise ValueError("cross-layer error-carry lane count differs")
        for record in records:
            unpack_scale32(int(record))
        unpack_scale32(int(layer["residual_scale32"]))
        unpack_scale32(int(layer["destination_scale32"]))
    binary = QECR_METADATA.with_suffix(".bin")
    if sha256_file(binary) != metadata.get("identity", {}).get(
        "immutable_image_sha256"
    ):
        raise ValueError("cross-layer error-carry metadata image hash differs")
    return metadata


@lru_cache(maxsize=1)
def cross_layer_error_carry_metadata() -> CrossLayerCarryMetadata:
    layers = load_cross_layer_error_carry_metadata()["layers"]
    return CrossLayerCarryMetadata.from_sequences(
        [layer["accumulator_scale32"] for layer in layers],
        [layer["residual_scale32"] for layer in layers],
        [layer["destination_scale32"] for layer in layers],
    )


def write_json(path: Path, value: Any, *, default: Any | None = None) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=default) + "\n",
        encoding="utf-8",
    )


def hash_records(records: Iterable[str]) -> tuple[str, int]:
    digest = hashlib.sha256()
    count = 0
    for record in records:
        encoded = record.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        count += 1
    return digest.hexdigest(), count


def hash_json_records(records: Iterable[dict[str, Any]]) -> tuple[str, int]:
    return hash_records(
        json.dumps(record, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        for record in records
    )


def hash_token_sequences(sequences: Iterable[Tensor]) -> tuple[str, int, int]:
    digest = hashlib.sha256()
    sequence_count = 0
    token_count = 0
    for sequence in sequences:
        tokens = sequence.detach().cpu().reshape(-1).tolist()
        digest.update(len(tokens).to_bytes(8, "big"))
        for token in tokens:
            digest.update(int(token).to_bytes(4, "big", signed=False))
        sequence_count += 1
        token_count += len(tokens)
    return digest.hexdigest(), sequence_count, token_count


def load_contracts(
    *,
    require_rtl_binding: bool = True,
    candidate_evidence_path: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    manifest = json.loads(PROMPT_MANIFEST.read_text(encoding="utf-8"))
    config = json.loads(QUALITY_CONFIG.read_text(encoding="utf-8"))
    if manifest["model"] != {
        "repository": config["baseline"]["model_repository"],
        "revision": config["baseline"]["model_revision"],
    }:
        raise ValueError("model identity differs between quality contracts")
    if config["weight_quantization"]["packing_group_size"] != 128:
        raise ValueError("ACE-2 W4 packing groups must contain 128 elements")
    if (
        config["weight_quantization"]["granularity"]
        != "symmetric_per_output_channel_across_the_complete_projection_reduction"
    ):
        raise ValueError("projection weights must use one scale per output reduction")
    if (
        config["activation_quantization"]["granularity"]
        != "static_per_linear_tensor_with_dynamic_per_token_per_head_rope_requantization"
    ):
        raise ValueError("Q/K activation scales must use the frozen dynamic RoPE granularity")
    if config["arithmetic"]["rounding"] != "round_to_nearest_ties_to_even":
        raise ValueError("ACE-2 requires round-to-nearest ties-to-even")
    if (
        config["arithmetic"]["accumulator"]
        != "signed_int32_across_the_complete_projection_reduction"
    ):
        raise ValueError("ACE-2 projection requires one raw signed-int32 reduction")
    attention_scales = config["full_model_scope"]["attention_scale_derivation"]
    if attention_scales != {
        "attention_value_output_scale": "v_projection_output_scale",
        "o_projection_input_scale": "attention_value_output_scale",
        "producer_scale": "normalized_nonzero_Scale32_per_q_or_k_projection_tensor",
        "qk_basis_rotation": "none",
        "rope_output_scale": "smallest_normalized_Scale32_greater_than_or_equal_to_producer_scale_times_wide_head_maximum_divided_by_127_times_2^15",
        "score_requantization": "17_bit_pair_significand_round_even_q_sig_times_k_sig_divided_by_2^15_then_signed_50_bit_dot_product_round_shifted_by_9_minus_exponent_sum",
    }:
        raise ValueError("attention scale derivation differs from the frozen contract")
    if config["full_model_scope"].get("qk_basis_contract") != {
        "angle_degrees": ACTIVE_ROPE_PAIR_ROTATION_DEGREES,
        "command_flag_half_degree_units": None,
        "exact_model_invariance": "standard_qwen_rope_basis_is_preserved_without_an_added_qk_basis_rotation",
        "runtime_arithmetic_added": True,
        "scope": "all_layers_all_query_and_kv_heads_dynamic_per_token_per_head_requantization",
    }:
        raise ValueError("Q/K basis contract differs from the selected mechanism")
    if config["software"] != EXPECTED_SOFTWARE:
        raise ValueError("quality software pins differ from the accepted package set")
    if set(manifest["lm_eval"]["tasks"]) != {
        "arc_challenge",
        "arc_easy",
        "hellaswag",
        "piqa",
        "winogrande",
    }:
        raise ValueError("lm-eval task set differs from the accepted five-task set")
    piqa = manifest["lm_eval"]["tasks"]["piqa"]
    train_dev = piqa["raw_sources"]["train_dev"]
    if len(train_dev["sha256"]) != 64 or set(piqa["records"]) != {
        "train",
        "validation",
    }:
        raise ValueError("PIQA archive/record pins are incomplete")
    for split, expected in piqa["records"].items():
        if expected["count"] <= 0 or len(expected["sha256"]) != 64:
            raise ValueError(f"PIQA {split} record pin is invalid")
    if candidate_evidence_path is not None:
        evidence_path = candidate_evidence_path.resolve()
        if ROOT.resolve() not in evidence_path.parents or not evidence_path.is_file():
            raise ValueError("candidate evidence must be a repository file")
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        source_binding = evidence["source_binding"]
        source_list_path = (ROOT / source_binding["source_hash_list"]).resolve()
        if ROOT.resolve() not in source_list_path.parents or not source_list_path.is_file():
            raise ValueError("candidate source hash list is not a repository file")
        numerical_rtl = []
        for line in source_list_path.read_text(encoding="utf-8").splitlines():
            expected, separator, relative = line.partition("  ")
            if not separator or len(expected) != 64:
                raise ValueError(f"malformed candidate source hash line: {line!r}")
            path = (ROOT / relative).resolve()
            if ROOT.resolve() not in path.parents or not path.is_file():
                raise ValueError(f"candidate source is not a repository file: {relative}")
            observed = sha256_file(path)
            if observed != expected:
                raise ValueError(
                    f"candidate source hash differs for {relative}: {observed} != {expected}"
                )
            numerical_rtl.append(
                {
                    "bytes": path.stat().st_size,
                    "path": relative,
                    "sha256": observed,
                }
            )
        manifest_path = ROOT / "design" / "RTL_MANIFEST.json"
        expected_manifest_sha256 = evidence["accepted_frontier_preservation"][
            "rtl_manifest_sha256"
        ]
        if sha256_file(manifest_path) != expected_manifest_sha256:
            raise ValueError("accepted frontier manifest differs from candidate preservation binding")
        rtl_binding = {
            "binding": {
                "bytes": evidence_path.stat().st_size,
                "path": evidence_path.relative_to(ROOT).as_posix(),
                "sha256": sha256_file(evidence_path),
            },
            "candidate_id": evidence["candidate_id"],
            "candidate_rtl_hash": source_binding["ordered_source_hash_list_sha256"],
            "manifest": {
                "bytes": manifest_path.stat().st_size,
                "path": manifest_path.relative_to(ROOT).as_posix(),
                "sha256": expected_manifest_sha256,
            },
            "numerical_rtl": numerical_rtl,
            "source_hash_list": {
                "bytes": source_list_path.stat().st_size,
                "path": source_list_path.relative_to(ROOT).as_posix(),
                "sha256": sha256_file(source_list_path),
            },
            "valid": True,
        }
    else:
        rtl_binding = (
            validate_rtl_binding(ROOT)
            if require_rtl_binding
            else json.loads(
                (ROOT / "benchmark" / "quality" / "RTL_BINDING.json").read_text()
            )
        )
    return manifest, config, rtl_binding


def runtime_versions() -> dict[str, str]:
    return {
        name: importlib.metadata.version("lm-eval" if name == "lm_eval" else name)
        for name in EXPECTED_SOFTWARE
    }


def validate_runtime(config: dict[str, Any]) -> dict[str, str]:
    versions = runtime_versions()
    if versions != config["software"]:
        raise RuntimeError(f"runtime package versions differ from pins: {versions}")
    return versions


def seed_everything(config: dict[str, Any]) -> None:
    seeds = config["determinism"]
    required = {
        "datasets_seed",
        "numpy_seed",
        "python_seed",
        "torch_seed",
        "torch_deterministic_algorithms",
    }
    if set(seeds) != required:
        raise ValueError("determinism contract has missing or unexpected fields")
    os.environ["PYTHONHASHSEED"] = str(seeds["python_seed"])
    os.environ["HF_DATASETS_RANDOM_SEED"] = str(seeds["datasets_seed"])
    random.seed(seeds["python_seed"])
    np.random.seed(seeds["numpy_seed"])
    torch.manual_seed(seeds["torch_seed"])
    torch.use_deterministic_algorithms(seeds["torch_deterministic_algorithms"])


def round_shift_even(value: Tensor, shift: Tensor | int) -> Tensor:
    shift_tensor = torch.as_tensor(shift, dtype=torch.int64, device=value.device)
    if torch.any(shift_tensor < 0) or torch.any(shift_tensor > 63):
        raise ValueError("fixed-point right shifts must be in the accepted range 0..63")
    safe_shift = shift_tensor.clamp(min=1)
    magnitude = value.abs()
    base = torch.bitwise_right_shift(magnitude, safe_shift)
    shifted_one = torch.bitwise_left_shift(torch.ones_like(safe_shift), safe_shift)
    mask = torch.where(
        safe_shift == 63,
        torch.full_like(safe_shift, INT64_MAX),
        shifted_one - 1,
    )
    remainder = torch.bitwise_and(magnitude, mask)
    half = torch.bitwise_left_shift(torch.ones_like(safe_shift), safe_shift - 1)
    increment = (remainder > half) | ((remainder == half) & ((base & 1) == 1))
    rounded = base + increment.to(base.dtype)
    signed = torch.where(value < 0, -rounded, rounded)
    return torch.where(shift_tensor == 0, value, signed)


def derive_multiplier(real_multiplier: Tensor) -> tuple[Tensor, Tensor]:
    real = real_multiplier.detach().to(torch.float64)
    if torch.any(real < 0) or not torch.all(torch.isfinite(real)):
        raise ValueError("requantization multiplier must be finite and nonnegative")
    multiplier = torch.zeros_like(real, dtype=torch.int64)
    right_shift = torch.full_like(real, -1, dtype=torch.int64)
    for shift in range(63, -1, -1):
        candidate = torch.round(real * math.ldexp(1.0, shift))
        select = (right_shift < 0) & (candidate <= INT32_MAX)
        multiplier = torch.where(select, candidate.to(torch.int64), multiplier)
        right_shift = torch.where(select, torch.full_like(right_shift, shift), right_shift)
    if torch.any(right_shift < 0):
        raise OverflowError("real multiplier cannot be represented by signed-int32 metadata")
    return multiplier, right_shift


def quantize_int8(value: Tensor, scale: float) -> Tensor:
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("activation scale must be finite and positive")
    return torch.round(value.to(torch.float64) / scale).clamp(-128, 127).to(torch.int8)


@dataclass
class CalibrationRange:
    input_absmax: float = 0.0
    output_absmax: float = 0.0
    output_head_absmax: list[float] | None = None
    input_percentile_absmax: float | None = None
    output_percentile_absmax: float | None = None


@dataclass
class ObservedRange:
    absmax: float = 0.0
    percentile_absmax: float | None = None


@dataclass(frozen=True)
class AttentionScaleBinding:
    conversion_scale: float
    conversion_q9: int
    metadata_saturated: bool
    query_output_scale: float
    key_output_scale: float


def absolute_percentile(value: Tensor, quantile: float) -> float:
    """Return the exact linear-interpolated percentile of absolute BF16 values.

    PyTorch's quantile kernel rejects tensors with more than 2**24 elements.
    Calibration reaches that size at the lm-head output, while the frozen model
    executes in BF16.  Positive finite BF16 bit patterns are monotonically
    ordered, so a bounded 65,536-bin histogram preserves the exact order
    statistics without materializing a float64 copy of the complete tensor.
    """
    if not 0.0 < quantile < 1.0:
        raise ValueError("activation scale percentile must be between zero and one")
    flattened = value.detach().abs().reshape(-1)
    if flattened.numel() == 0:
        raise ValueError("cannot derive an activation percentile from an empty tensor")
    if flattened.dtype != torch.bfloat16:
        return float(torch.quantile(flattened.to(torch.float64), quantile))

    histogram = torch.zeros(1 << 16, dtype=torch.int64)
    chunk_elements = 1 << 20
    for offset in range(0, flattened.numel(), chunk_elements):
        chunk = flattened[offset : offset + chunk_elements].contiguous()
        if not torch.all(torch.isfinite(chunk)):
            raise ValueError("activation calibration tensor must be finite")
        bit_patterns = chunk.view(torch.uint16).to(device="cpu", dtype=torch.int64)
        histogram += torch.bincount(bit_patterns, minlength=1 << 16)

    position = quantile * (flattened.numel() - 1)
    lower_rank = math.floor(position)
    upper_rank = math.ceil(position)
    cumulative = torch.cumsum(histogram, dim=0)

    def value_at_rank(rank: int) -> float:
        bit_pattern = int(
            torch.searchsorted(
                cumulative,
                torch.tensor(rank + 1, dtype=torch.int64),
            )
        )
        return float(
            torch.tensor([bit_pattern], dtype=torch.uint16)
            .view(torch.bfloat16)
            .item()
        )

    lower = value_at_rank(lower_rank)
    upper = value_at_rank(upper_rank)
    return lower + (upper - lower) * (position - lower_rank)


def positive_scale(absmax: float, *, cap: float | None = None) -> float:
    if not math.isfinite(absmax) or absmax <= 0:
        raise ValueError(f"calibration range must be finite and positive, got {absmax}")
    scale = absmax / 127.0
    if cap is not None:
        if not math.isfinite(cap) or cap <= 0:
            raise ValueError(f"activation scale cap must be finite and positive, got {cap}")
        scale = min(scale, cap)
    return scale


def rmsnorm_output_scale(
    source: nn.Module,
    output_absmax: float,
    *,
    cap: float | None = None,
) -> float:
    scale = positive_scale(output_absmax, cap=cap)
    gain_scale_floor = (
        float(source.weight.detach().to(torch.float64).abs().amax())
        * (1 << RMS_GAIN_FRAC)
        / 32767.0
    )
    return max(scale, gain_scale_floor)


def diagnostic_factor(value: float, name: str) -> float:
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def apply_rope_commuting_pair_rotation(
    model: nn.Module,
    angle_degrees: float,
) -> None:
    """Fuse one shared orthogonal rotation into every Q/K rotary pair."""
    if not math.isfinite(angle_degrees) or not -90.0 < angle_degrees < 90.0:
        raise ValueError("RoPE pair-rotation angle must be finite and between -90 and 90")
    angle_radians = math.radians(angle_degrees)
    cosine = math.cos(angle_radians)
    sine = math.sin(angle_radians)
    with torch.no_grad():
        for layer in model.model.layers:
            attention = layer.self_attn
            for projection in (attention.q_proj, attention.k_proj):
                weight = projection.weight.detach().to(torch.float64)
                if weight.shape[0] % ROPE_HEAD_DIM:
                    raise ValueError("Q/K projection rows do not align to RoPE heads")
                heads = weight.shape[0] // ROPE_HEAD_DIM
                shaped = weight.reshape(heads, ROPE_HEAD_DIM, weight.shape[1])
                first = shaped[:, : ROPE_HEAD_DIM // 2, :].clone()
                second = shaped[:, ROPE_HEAD_DIM // 2 :, :].clone()
                shaped[:, : ROPE_HEAD_DIM // 2, :] = cosine * first - sine * second
                shaped[:, ROPE_HEAD_DIM // 2 :, :] = sine * first + cosine * second
                projection.weight.copy_(shaped.reshape_as(weight).to(projection.weight.dtype))
                if projection.bias is not None:
                    bias = projection.bias.detach().to(torch.float64).reshape(
                        heads, ROPE_HEAD_DIM
                    )
                    first_bias = bias[:, : ROPE_HEAD_DIM // 2].clone()
                    second_bias = bias[:, ROPE_HEAD_DIM // 2 :].clone()
                    bias[:, : ROPE_HEAD_DIM // 2] = (
                        cosine * first_bias - sine * second_bias
                    )
                    bias[:, ROPE_HEAD_DIM // 2 :] = (
                        sine * first_bias + cosine * second_bias
                    )
                    projection.bias.copy_(
                        bias.reshape_as(projection.bias).to(projection.bias.dtype)
                    )


def derive_attention_score_metadata(
    scale_binding: AttentionScaleBinding,
    head_dim: int,
    score_scale_factor: float = 1.0,
) -> tuple[Tensor, Tensor]:
    factor = diagnostic_factor(
        score_scale_factor,
        "attention score diagnostic scale factor",
    )
    score_real_multiplier = torch.tensor(
        [
            scale_binding.query_output_scale
            * scale_binding.key_output_scale
            * (1 << ATTENTION_SCORE_FRAC)
            / math.sqrt(head_dim)
            * factor
        ],
        dtype=torch.float64,
    )
    return derive_multiplier(score_real_multiplier)


def derive_attention_scale_binding(
    query_projection_scale: float,
    key_projection_scale: float,
) -> AttentionScaleBinding:
    if (
        not math.isfinite(query_projection_scale)
        or query_projection_scale <= 0
        or not math.isfinite(key_projection_scale)
        or key_projection_scale <= 0
    ):
        raise ValueError("Q/K projection scales must be finite and positive")
    if ROPE_HEAD_DIM != 1 << (2 * ATTENTION_HEAD_SCALE_SHIFT):
        raise ValueError("attention score shift does not implement 1/sqrt(head_dim)")
    conversion_q9 = ROPE_SAFE_CONVERSION_Q9
    realized_conversion_scale = conversion_q9 / float(1 << ROPE_SCALE_FRAC)
    query_output_scale = query_projection_scale / realized_conversion_scale
    key_output_scale = key_projection_scale / realized_conversion_scale
    return AttentionScaleBinding(
        conversion_scale=realized_conversion_scale,
        conversion_q9=conversion_q9,
        metadata_saturated=False,
        query_output_scale=query_output_scale,
        key_output_scale=key_output_scale,
    )


class W4A8Linear(nn.Module):
    def __init__(
        self,
        source: nn.Linear,
        calibration: CalibrationRange,
        *,
        input_is_quantized: bool = False,
        output_head_absmax: list[float] | None = None,
        output_head_size: int | None = None,
        scale_cap: float | None = None,
        use_percentile_scale: bool = False,
        scale32_output: bool = False,
    ) -> None:
        super().__init__()
        if calibration.input_absmax <= 0 or calibration.output_absmax <= 0:
            raise ValueError("linear calibration ranges must be positive")
        weight = source.weight.detach().to(torch.float64)
        self.in_features = source.in_features
        self.out_features = source.out_features
        input_absmax = (
            calibration.input_percentile_absmax
            if use_percentile_scale
            else calibration.input_absmax
        )
        output_absmax = (
            calibration.output_percentile_absmax
            if use_percentile_scale
            else calibration.output_absmax
        )
        if input_absmax is None or output_absmax is None:
            raise ValueError("requested percentile calibration ranges are missing")
        self.input_scale = positive_scale(input_absmax, cap=scale_cap)
        self.hardware_input_scale = self.input_scale
        if output_head_absmax is not None:
            if output_head_size is None or output_head_size <= 0:
                raise ValueError("per-head projection scales require a positive head size")
            if len(output_head_absmax) * output_head_size != self.out_features:
                raise ValueError("per-head projection scales do not cover all outputs")
            head_scales = torch.tensor(
                [positive_scale(value, cap=scale_cap) for value in output_head_absmax],
                dtype=torch.float64,
            )
            output_scale_per_channel = head_scales.repeat_interleave(
                output_head_size
            )
            self.output_scale = None
        else:
            head_scales = torch.empty(0, dtype=torch.float64)
            self.output_scale = positive_scale(output_absmax, cap=scale_cap)
            self.output_scale32_record = (
                ceil_scale32_from_float(self.output_scale) if scale32_output else None
            )
            if self.output_scale32_record is not None:
                scale_numerator, scale_denominator = scale32_ratio(
                    self.output_scale32_record
                )
                self.output_scale = scale_numerator / scale_denominator
            output_scale_per_channel = torch.full(
                (self.out_features,), self.output_scale, dtype=torch.float64
            )
        if output_head_absmax is not None:
            self.output_scale32_record = None
        self.output_head_size = output_head_size
        self.register_buffer(
            "output_head_scales", head_scales, persistent=True
        )
        self.register_buffer(
            "output_scale_per_channel",
            output_scale_per_channel,
            persistent=True,
        )
        self.input_is_quantized = input_is_quantized
        weight_scale = weight.abs().amax(dim=1) / 7.0
        weight_scale = torch.where(weight_scale > 0, weight_scale, torch.ones_like(weight_scale))
        qweight = torch.round(weight / weight_scale[:, None]).clamp(-8, 7).to(torch.int8)
        self.register_buffer("qweight", qweight, persistent=True)
        self.register_buffer("qweight_transposed", qweight.transpose(0, 1).contiguous(), persistent=False)
        self.register_buffer("weight_scale", weight_scale, persistent=True)
        self.register_buffer(
            "native_scale32_records",
            self._native_scale32_for_input_scale(self.input_scale),
            persistent=True,
        )
        self.register_buffer(
            "_source_bias",
            (
                source.bias.detach().to(torch.float64)
                if source.bias is not None
                else None
            ),
            persistent=False,
        )
        multiplier, right_shift, bias_accumulator = self._metadata_for_input_scale(
            self.input_scale
        )
        self.register_buffer("multiplier", multiplier, persistent=True)
        self.register_buffer("right_shift", right_shift, persistent=True)
        self.register_buffer("bias_accumulator", bias_accumulator, persistent=True)

    def _metadata_for_input_scale(
        self,
        input_scale: float,
    ) -> tuple[Tensor, Tensor, Tensor | None]:
        if not math.isfinite(input_scale) or input_scale <= 0:
            raise ValueError("projection input scale must be finite and positive")
        real_multiplier = (
            input_scale * self.weight_scale / self.output_scale_per_channel
        )
        multiplier, right_shift = derive_multiplier(real_multiplier)
        bias_accumulator = (
            None
            if self._source_bias is None
            else torch.round(
                self._source_bias / (input_scale * self.weight_scale)
            ).to(torch.int64)
        )
        return multiplier, right_shift, bias_accumulator

    def _native_scale32_for_input_scale(self, input_scale: float) -> Tensor:
        records = [
            ceil_scale32_from_float(input_scale * float(weight_scale))
            for weight_scale in self.weight_scale.detach().cpu().tolist()
        ]
        return torch.tensor(
            records,
            dtype=torch.int64,
            device=self.weight_scale.device,
        )

    def bind_hardware_input_scale(self, input_scale: float) -> None:
        multiplier, right_shift, bias_accumulator = self._metadata_for_input_scale(
            input_scale
        )
        self.hardware_input_scale = input_scale
        self.native_scale32_records.copy_(
            self._native_scale32_for_input_scale(input_scale)
        )
        self.multiplier.copy_(multiplier)
        self.right_shift.copy_(right_shift)
        if self.bias_accumulator is not None:
            if bias_accumulator is None:
                raise AssertionError("projection bias metadata unexpectedly disappeared")
            self.bias_accumulator.copy_(bias_accumulator)

    def forward_quantized(self, qinput: Tensor) -> Tensor:
        if qinput.dtype != torch.int8:
            raise TypeError("projection input must be signed int8")
        if qinput.shape[-1] != self.in_features:
            raise ValueError(
                f"projection input has {qinput.shape[-1]} features, expected {self.in_features}"
            )
        original_shape = qinput.shape[:-1]
        accumulator = self.accumulator_quantized(qinput)
        return self.requantize_accumulator(accumulator, original_shape)

    def requantize_accumulator(
        self,
        accumulator: Tensor,
        original_shape: tuple[int, ...],
    ) -> Tensor:
        if accumulator.dtype != torch.int64:
            raise TypeError("projection accumulator must be signed int64")
        if accumulator.shape != (math.prod(original_shape), self.out_features):
            raise ValueError("projection accumulator shape differs from output geometry")
        product = accumulator * self.multiplier
        output = round_shift_even(product, self.right_shift).clamp(-128, 127).to(torch.int8)
        return output.reshape(*original_shape, self.out_features)

    def accumulator_quantized(self, qinput: Tensor) -> Tensor:
        if qinput.dtype != torch.int8:
            raise TypeError("projection input must be signed int8")
        if qinput.shape[-1] != self.in_features:
            raise ValueError(
                f"projection input has {qinput.shape[-1]} features, expected {self.in_features}"
            )
        flat = qinput.reshape(-1, self.in_features).contiguous()
        accumulator = torch._int_mm(flat, self.qweight_transposed).to(torch.int64)
        if self.bias_accumulator is not None:
            accumulator = accumulator + self.bias_accumulator
        if torch.any(accumulator < -(1 << 31)) or torch.any(accumulator >= (1 << 31)):
            raise OverflowError("projection accumulator plus bias exceeds signed-32")
        return accumulator

    def forward_shadow_quantized(self, qinput: Tensor) -> Tensor:
        """Emit the checked signed-Q15.16 pre-clamp projection shadow."""
        original_shape = qinput.shape[:-1]
        accumulator = self.accumulator_quantized(qinput)
        product = accumulator * self.multiplier
        shadow_shift = self.right_shift - 16
        if torch.any(shadow_shift < 0):
            left_shift = (-shadow_shift).clamp(min=0)
            if torch.any(left_shift > 15):
                raise OverflowError("projection-shadow left shift exceeds bounded metadata")
            left = torch.bitwise_left_shift(product, left_shift)
            right = round_shift_even(product, shadow_shift.clamp(min=0))
            shadow = torch.where(shadow_shift < 0, left, right)
        else:
            shadow = round_shift_even(product, shadow_shift)
        if torch.any(shadow < -(1 << 31)) or torch.any(shadow >= (1 << 31)):
            raise OverflowError("projection shadow exceeds signed-32 Q15.16")
        return shadow.to(torch.int32).reshape(*original_shape, self.out_features)

    def forward_raw(self, inputs: Tensor) -> Tensor:
        return self.forward_quantized(quantize_int8(inputs, self.input_scale))

    def forward_hardware_input(self, inputs: Tensor) -> Tensor:
        if inputs.dtype == torch.int8:
            qinput = inputs
        elif inputs.is_floating_point():
            if not torch.all(torch.isfinite(inputs)):
                raise ValueError("projection hardware input must be finite")
            if torch.any(inputs < -128) or torch.any(inputs > 127):
                raise ValueError("projection hardware input is outside signed int8")
            if not torch.equal(inputs, torch.round(inputs)):
                raise ValueError("projection hardware input must contain exact integers")
            qinput = inputs.to(torch.int8)
        else:
            raise TypeError("projection hardware input must be signed int8 or an exact float container")
        return self.forward_quantized(qinput)

    def forward_shadow_hardware_input(self, inputs: Tensor) -> Tensor:
        if inputs.dtype == torch.int8:
            qinput = inputs
        elif inputs.is_floating_point():
            if not torch.all(torch.isfinite(inputs)):
                raise ValueError("projection hardware input must be finite")
            if torch.any(inputs < -128) or torch.any(inputs > 127):
                raise ValueError("projection hardware input is outside signed int8")
            if not torch.equal(inputs, torch.round(inputs)):
                raise ValueError("projection hardware input must contain exact integers")
            qinput = inputs.to(torch.int8)
        else:
            raise TypeError("projection hardware input must be signed int8 or an exact float container")
        return self.forward_shadow_quantized(qinput)

    def forward_native_hardware_input(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        if inputs.dtype == torch.int8:
            qinput = inputs
        elif inputs.is_floating_point():
            if not torch.all(torch.isfinite(inputs)):
                raise ValueError("projection hardware input must be finite")
            if torch.any(inputs < -128) or torch.any(inputs > 127):
                raise ValueError("projection hardware input is outside signed int8")
            if not torch.equal(inputs, torch.round(inputs)):
                raise ValueError("projection hardware input must contain exact integers")
            qinput = inputs.to(torch.int8)
        else:
            raise TypeError("projection hardware input must be signed int8 or an exact float container")
        original_shape = qinput.shape[:-1]
        accumulator = self.accumulator_quantized(qinput).to(torch.int32)
        return (
            accumulator.reshape(*original_shape, self.out_features),
            self.native_scale32_records,
        )

    def forward_qk_residual_hardware_input(
        self,
        inputs: Tensor,
        baseline_scale32_records: Tensor,
        residual_scale32_records: Tensor,
    ) -> tuple[Tensor, Tensor, int, int]:
        """Emit the unchanged q8 projection and its frozen signed-4 sidecar."""
        if inputs.dtype == torch.int8:
            qinput = inputs
        elif inputs.is_floating_point():
            if not torch.all(torch.isfinite(inputs)):
                raise ValueError("projection hardware input must be finite")
            if torch.any(inputs < -128) or torch.any(inputs > 127):
                raise ValueError("projection hardware input is outside signed int8")
            if not torch.equal(inputs, torch.round(inputs)):
                raise ValueError("projection hardware input must contain exact integers")
            qinput = inputs.to(torch.int8)
        else:
            raise TypeError(
                "projection hardware input must be signed int8 or an exact float container"
            )
        original_shape = qinput.shape[:-1]
        accumulator = self.accumulator_quantized(qinput)
        baseline = self.forward_quantized(qinput)
        residual, positive_clamps, negative_clamps = qk_residual_projection_raw(
            accumulator,
            self.multiplier,
            self.right_shift,
            baseline.reshape(-1, self.out_features),
            baseline_scale32_records,
            residual_scale32_records,
        )
        return (
            baseline.reshape(*original_shape, self.out_features),
            residual.reshape(*original_shape, self.out_features),
            positive_clamps,
            negative_clamps,
        )

    def forward_v_residual_hardware_input(
        self,
        inputs: Tensor,
        baseline_scale32_records: Tensor,
        residual_scale32_records: Tensor,
    ) -> tuple[Tensor, Tensor, int, int]:
        """Emit unchanged V int8 plus the frozen signed-4 projection remainder."""
        if inputs.dtype == torch.int8:
            qinput = inputs
        elif inputs.is_floating_point():
            if not torch.all(torch.isfinite(inputs)):
                raise ValueError("projection hardware input must be finite")
            if torch.any(inputs < -128) or torch.any(inputs > 127):
                raise ValueError("projection hardware input is outside signed int8")
            if not torch.equal(inputs, torch.round(inputs)):
                raise ValueError("projection hardware input must contain exact integers")
            qinput = inputs.to(torch.int8)
        else:
            raise TypeError(
                "projection hardware input must be signed int8 or an exact float container"
            )
        original_shape = qinput.shape[:-1]
        accumulator = self.accumulator_quantized(qinput)
        baseline = self.forward_quantized(qinput)
        residual, positive_clamps, negative_clamps = v_residual_projection_raw(
            accumulator,
            self.multiplier,
            self.right_shift,
            baseline.reshape(-1, self.out_features),
            baseline_scale32_records,
            residual_scale32_records,
        )
        return (
            baseline.reshape(*original_shape, self.out_features),
            residual.reshape(*original_shape, self.out_features),
            positive_clamps,
            negative_clamps,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        raw = (
            self.forward_hardware_input(inputs)
            if self.input_is_quantized
            else self.forward_raw(inputs)
        )
        return raw.to(inputs.dtype) * self.output_scale_per_channel.to(inputs.dtype)


@lru_cache(maxsize=128)
def _qk_residual_round_thresholds(
    baseline_records: tuple[int, ...],
    residual_records: tuple[int, ...],
    shifts: tuple[int, ...],
    head_size: int,
) -> tuple[tuple[int, ...], ...]:
    if len(baseline_records) != len(residual_records):
        raise ValueError("Q/K residual Scale32 head counts differ")
    if len(shifts) != len(baseline_records) * head_size:
        raise ValueError("Q/K residual requantization metadata geometry differs")
    rows: list[tuple[int, ...]] = []
    for channel, shift in enumerate(shifts):
        if not 0 <= shift <= 63:
            raise ValueError("Q/K residual right shift is outside 0..63")
        head = channel // head_size
        baseline_sig, baseline_exp = unpack_scale32(baseline_records[head])
        residual_sig, residual_exp = unpack_scale32(residual_records[head])
        delta = baseline_exp - residual_exp - shift
        if not -91 <= delta <= 28:
            raise ValueError("Q/K residual Scale32/shift delta is outside [-91,+28]")
        numerator_coefficient = baseline_sig << max(delta, 0)
        denominator = residual_sig << max(-delta, 0)
        channel_thresholds: list[int] = []
        for rounded_magnitude in range(1, 9):
            boundary = (2 * rounded_magnitude - 1) * denominator
            divisor = 2 * numerator_coefficient
            if (rounded_magnitude - 1) & 1:
                threshold = (boundary + divisor - 1) // divisor
            else:
                threshold = boundary // divisor + 1
            if threshold > INT64_MAX:
                raise OverflowError("Q/K residual threshold exceeds signed-64 observation path")
            channel_thresholds.append(threshold)
        rows.append(tuple(channel_thresholds))
    return tuple(rows)


def qk_residual_projection_raw(
    accumulator: Tensor,
    multiplier: Tensor,
    right_shift: Tensor,
    baseline_q8: Tensor,
    baseline_scale32_records: Tensor,
    residual_scale32_records: Tensor,
) -> tuple[Tensor, int, int]:
    """Tensor form of the frozen integer projection-error sidecar."""
    if accumulator.dtype != torch.int64 or baseline_q8.dtype != torch.int8:
        raise TypeError("Q/K residual projection requires signed-int64 accumulators and q8")
    if accumulator.shape != baseline_q8.shape:
        raise ValueError("Q/K residual accumulator and q8 shapes differ")
    channels = accumulator.shape[-1]
    if channels % ROPE_HEAD_DIM:
        raise ValueError("Q/K residual channels must align to 64-lane heads")
    head_count = channels // ROPE_HEAD_DIM
    if baseline_scale32_records.shape != (head_count,) or residual_scale32_records.shape != (
        head_count,
    ):
        raise ValueError("Q/K residual Scale32 records do not cover every head")
    if multiplier.shape != (channels,) or right_shift.shape != (channels,):
        raise ValueError("Q/K residual ordinary requantization metadata shape differs")
    product = accumulator * multiplier
    shifts = right_shift.to(torch.int64)
    baseline_reconstruction = torch.bitwise_left_shift(
        baseline_q8.to(torch.int64),
        shifts,
    )
    error = product - baseline_reconstruction
    if torch.any(error == torch.iinfo(torch.int64).min):
        raise OverflowError("Q/K residual observed error reaches signed-64 minimum")
    threshold_rows = _qk_residual_round_thresholds(
        tuple(int(value) for value in baseline_scale32_records.detach().cpu().tolist()),
        tuple(int(value) for value in residual_scale32_records.detach().cpu().tolist()),
        tuple(int(value) for value in shifts.detach().cpu().tolist()),
        ROPE_HEAD_DIM,
    )
    thresholds = torch.tensor(
        threshold_rows,
        dtype=torch.int64,
        device=accumulator.device,
    )
    magnitude = error.abs()
    rounded_magnitude = (magnitude.unsqueeze(-1) >= thresholds).sum(dim=-1)
    positive_clamp = (error > 0) & (rounded_magnitude >= 8)
    negative_clamp = (error < 0) & (rounded_magnitude >= 8)
    signed = torch.where(error < 0, -rounded_magnitude, rounded_magnitude)
    residual = signed.clamp(-7, 7).to(torch.int8)
    return residual, int(positive_clamp.sum()), int(negative_clamp.sum())


def v_residual_projection_raw(
    accumulator: Tensor,
    multiplier: Tensor,
    right_shift: Tensor,
    baseline_v8: Tensor,
    baseline_scale32_records: Tensor,
    residual_scale32_records: Tensor,
) -> tuple[Tensor, int, int]:
    """Tensor form of the identical frozen V projection-error equation."""
    return qk_residual_projection_raw(
        accumulator,
        multiplier,
        right_shift,
        baseline_v8,
        baseline_scale32_records,
        residual_scale32_records,
    )


def fixed_rmsnorm_raw(activations: Tensor, scaled_gains_q8: Tensor) -> Tensor:
    if activations.dtype != torch.int8 or activations.shape[-1] != RMS_HIDDEN_SIZE:
        raise ValueError("RMSNorm requires signed-int8 vectors of length 896")
    if scaled_gains_q8.dtype != torch.int16 or scaled_gains_q8.shape != (
        RMS_HIDDEN_SIZE,
    ):
        raise ValueError("RMSNorm requires 896 signed-int16 scaled Q7.8 gains")
    values = activations.to(torch.int64)
    sumsq = (values * values).sum(dim=-1, keepdim=True)
    mean_square = (sumsq + RMS_HIDDEN_SIZE // 2) // RMS_HIDDEN_SIZE
    root = torch.floor(torch.sqrt(mean_square.to(torch.float64))).to(torch.int64)
    root = root + (root * root < mean_square).to(torch.int64)
    root = root.clamp(min=1)
    inv_rms_q30 = (1 << RMS_INV_FRAC) // root
    product = values * scaled_gains_q8.to(torch.int64) * inv_rms_q30
    output = round_shift_even(
        product,
        RMS_INV_FRAC + RMS_GAIN_FRAC,
    )
    return output.clamp(-128, 127).to(torch.int8)


class FixedRMSNorm(nn.Module):
    def __init__(
        self,
        source: nn.Module,
        input_scale: float,
        output_scale: float,
    ) -> None:
        super().__init__()
        if not math.isfinite(output_scale) or output_scale <= 0:
            raise ValueError("RMSNorm output scale must be finite and positive")
        gain = torch.round(
            source.weight.detach().to(torch.float64)
            / output_scale
            * (1 << RMS_GAIN_FRAC)
        )
        if torch.any(gain < -32768) or torch.any(gain > 32767):
            raise OverflowError("RMSNorm scaled Q7.8 gain metadata is not representable")
        self.input_scale = input_scale
        self.output_scale = output_scale
        self.register_buffer("scaled_gains_q8", gain.to(torch.int16), persistent=True)

    def forward(self, hidden_states: Tensor) -> Tensor:
        raw = fixed_rmsnorm_raw(
            quantize_int8(hidden_states, self.input_scale),
            self.scaled_gains_q8,
        )
        return raw.to(hidden_states.dtype)


class CarryAwareFixedRMSNorm(FixedRMSNorm):
    def __init__(
        self,
        source: nn.Module,
        input_scale: float,
        output_scale: float,
        runtime: CrossLayerErrorCarryRuntime,
        consumer_layer_id: int,
    ) -> None:
        super().__init__(source, input_scale, output_scale)
        if not 1 <= consumer_layer_id <= 24:
            raise ValueError("carry-aware RMSNorm consumer id must be in 1..24")
        self.runtime = runtime
        self.consumer_layer_id = consumer_layer_id

    def forward(self, hidden_states: Tensor) -> Tensor:
        raw = self.runtime.consume(
            self.consumer_layer_id,
            self.scaled_gains_q8,
        )
        if raw.shape != hidden_states.shape:
            raise RuntimeError("carry-aware RMSNorm output geometry differs")
        return raw.to(hidden_states.dtype)


def fixed_rope_raw_with_saturation(
    activations: Tensor,
    scale_q9: int,
    cos: Tensor,
    sin: Tensor,
    output_bits: int = 8,
) -> tuple[Tensor, int]:
    if activations.dtype != torch.int8 or activations.shape[-1] != ROPE_HEAD_DIM:
        raise ValueError("RoPE requires signed-int8 head vectors of length 64")
    if not isinstance(scale_q9, int) or not -32768 <= scale_q9 <= 32767:
        raise ValueError("RoPE scale metadata must be a signed-int16 Q6.9 integer")
    if output_bits not in (8, 9):
        raise ValueError("RoPE output width must be 8 or 9 bits")
    scaled = activations.to(torch.int64) * scale_q9
    saturation_count = 0
    cos_q15 = torch.round(cos.to(torch.float64) * 32767.0).clamp(-32768, 32767).to(torch.int64)
    sin_q15 = torch.round(sin.to(torch.float64) * 32767.0).clamp(-32768, 32767).to(torch.int64)
    cos_q15 = cos_q15.unsqueeze(1)
    sin_q15 = sin_q15.unsqueeze(1)
    first, second = scaled[..., :32], scaled[..., 32:]
    rotated_first = first * cos_q15[..., :32] - second * sin_q15[..., :32]
    rotated_second = second * cos_q15[..., 32:] + first * sin_q15[..., 32:]
    rotated = torch.cat((rotated_first, rotated_second), dim=-1)
    rounded = round_shift_even(rotated, ROPE_SCALE_FRAC + 15)
    output_min = -(1 << (output_bits - 1))
    output_max = (1 << (output_bits - 1)) - 1
    saturation_count += int(
        ((rounded > output_max) | (rounded < output_min)).sum()
    )
    output = rounded.clamp(output_min, output_max).to(
        torch.int8 if output_bits == 8 else torch.int16
    )
    return output, saturation_count


def fixed_rope_raw(
    activations: Tensor,
    scale_q9: int,
    cos: Tensor,
    sin: Tensor,
) -> Tensor:
    return fixed_rope_raw_with_saturation(activations, scale_q9, cos, sin)[0]


def dynamic_rope_head_raw(
    activations: Tensor,
    producer_scale32: int,
    cos: Tensor,
    sin: Tensor,
) -> tuple[Tensor, Tensor, int]:
    """Bit-exact dynamic per-token/per-head RoPE requantization."""
    if activations.dtype != torch.int8 or activations.shape[-1] != ROPE_HEAD_DIM:
        raise ValueError("dynamic RoPE requires signed-int8 64-lane heads")
    if cos.shape[-1] != ROPE_HEAD_DIM or sin.shape[-1] != ROPE_HEAD_DIM:
        raise ValueError("dynamic RoPE requires 64 cosine and sine coefficients")
    if cos.shape != sin.shape:
        raise ValueError("dynamic RoPE cosine and sine shapes must match")
    producer_sig, producer_exp = unpack_scale32(producer_scale32)
    cos_q15 = (
        torch.round(cos.to(torch.float64) * 32767.0)
        .clamp(-32768, 32767)
        .to(torch.int64)
        .unsqueeze(1)
    )
    sin_q15 = (
        torch.round(sin.to(torch.float64) * 32767.0)
        .clamp(-32768, 32767)
        .to(torch.int64)
        .unsqueeze(1)
    )
    values = activations.to(torch.int64)
    first, second = values[..., :32], values[..., 32:]
    rotated = torch.cat(
        (
            first * cos_q15[..., :32] - second * sin_q15[..., :32],
            second * cos_q15[..., 32:] + first * sin_q15[..., 32:],
        ),
        dim=-1,
    )
    if torch.any(rotated < -(1 << 24)) or torch.any(rotated >= (1 << 24)):
        raise OverflowError("dynamic RoPE signed-25-bit staging overflow")

    maximum = rotated.abs().amax(dim=-1)
    records = torch.tensor(
        [
            dynamic_rope_output_scale(producer_scale32, int(value))
            for value in maximum.reshape(-1).tolist()
        ],
        dtype=torch.int64,
        device=activations.device,
    ).reshape(maximum.shape)
    output_sig = torch.bitwise_and(records, 0xFFFF)
    output_exp_u8 = torch.bitwise_and(torch.bitwise_right_shift(records, 16), 0xFF)
    output_exp = torch.where(output_exp_u8 >= 128, output_exp_u8 - 256, output_exp_u8)
    exponent_delta = producer_exp - output_exp - 15
    numerator = rotated * producer_sig
    numerator = torch.bitwise_left_shift(
        numerator,
        exponent_delta.clamp(min=0).unsqueeze(-1),
    )
    denominator = torch.bitwise_left_shift(
        output_sig,
        (-exponent_delta).clamp(min=0),
    ).unsqueeze(-1)
    magnitude = numerator.abs()
    quotient = torch.div(magnitude, denominator, rounding_mode="floor")
    remainder = magnitude - quotient * denominator
    doubled = remainder * 2
    increment = (doubled > denominator) | (
        (doubled == denominator) & ((quotient & 1) == 1)
    )
    rounded_magnitude = quotient + increment.to(torch.int64)
    rounded = torch.where(numerator < 0, -rounded_magnitude, rounded_magnitude)
    if torch.any(rounded < -127) or torch.any(rounded > 127):
        raise OverflowError("ceil-selected dynamic RoPE scale failed to bound int8")
    return rounded.to(torch.int8), records, 0


def fixed_q7_rope_head_raw(
    activations: Tensor,
    cos: Tensor,
    sin: Tensor,
) -> Tensor:
    """Reviewed layer-0 RoPE: signed-int8 input to signed-int16 fixed-Q7."""
    if activations.dtype != torch.int8 or activations.shape[-1] != ROPE_HEAD_DIM:
        raise ValueError("fixed-Q7 RoPE requires signed-int8 64-lane heads")
    if cos.shape != sin.shape or cos.shape[-1] != ROPE_HEAD_DIM:
        raise ValueError("fixed-Q7 RoPE requires matching 64-lane coefficient tensors")
    cos_q15 = (
        torch.round(cos.to(torch.float64) * 32767.0)
        .clamp(-32768, 32767)
        .to(torch.int64)
        .unsqueeze(1)
    )
    sin_q15 = (
        torch.round(sin.to(torch.float64) * 32767.0)
        .clamp(-32768, 32767)
        .to(torch.int64)
        .unsqueeze(1)
    )
    if not torch.equal(cos_q15[..., :32], cos_q15[..., 32:]):
        raise ValueError("fixed-Q7 RoPE coefficient halves differ")
    if not torch.equal(sin_q15[..., :32], sin_q15[..., 32:]):
        raise ValueError("fixed-Q7 RoPE sine halves differ")
    if torch.any(cos_q15[..., :32].abs() + sin_q15[..., :32].abs() > 46342):
        raise ValueError("fixed-Q7 RoPE coefficient L1 bound exceeded")
    values = activations.to(torch.int64)
    first, second = values[..., :32], values[..., 32:]
    coefficients_c = cos_q15[..., :32]
    coefficients_s = sin_q15[..., :32]
    rotated = torch.cat(
        (
            first * coefficients_c - second * coefficients_s,
            first * coefficients_s + second * coefficients_c,
        ),
        dim=-1,
    )
    if torch.any(rotated.abs() > 5_931_776):
        raise OverflowError("fixed-Q7 RoPE signed-25-bit reviewed bound exceeded")
    wide = round_shift_even(rotated, 8)
    if torch.any(wide.abs() > 23_171):
        raise OverflowError("fixed-Q7 RoPE signed-int16 reviewed bound exceeded")
    return wide.to(torch.int16)


def repeat_kv(hidden_states: Tensor, repeats: int) -> Tensor:
    batch, heads, sequence, head_dim = hidden_states.shape
    if repeats == 1:
        return hidden_states
    expanded = hidden_states[:, :, None, :, :].expand(
        batch, heads, repeats, sequence, head_dim
    )
    return expanded.reshape(batch, heads * repeats, sequence, head_dim)


def qk_residual_rope_raw(residual_s4: Tensor) -> Tensor:
    """Rotate frozen signed-4 residual lanes into checked signed-int8 values."""
    if residual_s4.dtype != torch.int8 or residual_s4.ndim != 4:
        raise TypeError("Q/K residual RoPE requires rank-4 signed-int8 storage")
    if residual_s4.shape[-1] != ROPE_HEAD_DIM:
        raise ValueError("Q/K residual RoPE requires 64-lane heads")
    if torch.any(residual_s4 < -7) or torch.any(residual_s4 > 7):
        raise ValueError("Q/K residual RoPE input must be signed-4 [-7,+7]")
    sequence = residual_s4.shape[-2]
    coefficients = [absolute_coefficients_q15(position) for position in range(sequence)]
    cosine = torch.tensor(
        [item[0] for item in coefficients],
        dtype=torch.int64,
        device=residual_s4.device,
    ).reshape(1, 1, sequence, ROPE_HEAD_DIM // 2)
    sine = torch.tensor(
        [item[1] for item in coefficients],
        dtype=torch.int64,
        device=residual_s4.device,
    ).reshape(1, 1, sequence, ROPE_HEAD_DIM // 2)
    values = residual_s4.to(torch.int64)
    real, imag = values[..., :32], values[..., 32:]
    real_accumulator = real * cosine - imag * sine
    imag_accumulator = imag * cosine + real * sine
    if (
        torch.any(real_accumulator < -(1 << 21))
        or torch.any(real_accumulator >= (1 << 21))
        or torch.any(imag_accumulator < -(1 << 21))
        or torch.any(imag_accumulator >= (1 << 21))
    ):
        raise OverflowError("Q/K residual RoPE exceeds signed-22 accumulation")
    rotated = torch.cat(
        (
            round_shift_even(real_accumulator, 15),
            round_shift_even(imag_accumulator, 15),
        ),
        dim=-1,
    )
    if torch.any(rotated < -128) or torch.any(rotated > 127):
        raise OverflowError("Q/K residual RoPE output exceeds signed-int8")
    return rotated.to(torch.int8)


def _scale32_records_to_q20_44(
    dots: Tensor,
    scale_a_records: Tensor,
    scale_b_records: Tensor,
) -> Tensor:
    if dots.dtype != torch.int64 or dots.ndim != 4:
        raise TypeError("Scale32 score conversion requires rank-4 signed-int64 dots")
    heads = dots.shape[1]
    if scale_a_records.shape != (heads,) or scale_b_records.shape != (heads,):
        raise ValueError("Scale32 score conversion records do not cover every head")
    converted = torch.empty_like(dots)
    for head in range(heads):
        sig_a, exp_a = unpack_scale32(int(scale_a_records[head]))
        sig_b, exp_b = unpack_scale32(int(scale_b_records[head]))
        product = dots[:, head] * (sig_a * sig_b)
        shift = exp_a + exp_b + 11
        if shift >= 0:
            limit = INT64_MAX >> shift
            if torch.any(product > limit) or torch.any(product < -limit - 1):
                raise OverflowError("Scale32 Q20.44 conversion exceeds signed-64")
            value = torch.bitwise_left_shift(product, shift)
        else:
            value = round_shift_even(product, -shift)
        converted[:, head] = value
    return converted


def qk_authoritative_base_scores_q20_44_raw(
    query_q8: Tensor,
    key_q8: Tensor,
    query_scale32_records: Tensor,
    key_scale32_records: Tensor,
) -> Tensor:
    """Producer-owned conventional base score before residual correction."""
    if query_q8.dtype != torch.int8 or key_q8.dtype != torch.int8:
        raise TypeError("authoritative Q/K base score requires signed-int8 tensors")
    batch, query_heads, sequence, head_dim = query_q8.shape
    if query_heads != 14 or head_dim != ROPE_HEAD_DIM:
        raise ValueError("authoritative base score requires fourteen 64-lane Q heads")
    if key_q8.shape != (batch, 2, sequence, head_dim):
        raise ValueError("authoritative base score requires two 64-lane K heads")
    repeated_key = repeat_kv(key_q8, 7)
    repeated_key_scale32 = key_scale32_records.repeat_interleave(7)
    dots = torch.matmul(
        query_q8.to(torch.int32),
        repeated_key.to(torch.int32).transpose(2, 3),
    ).to(torch.int64)
    return _scale32_records_to_q20_44(
        dots,
        query_scale32_records,
        repeated_key_scale32,
    )


def _center_q20_44_scores(
    scores: Tensor,
    attention_mask: Tensor | None,
) -> Tensor:
    batch, heads, query_length, key_length = scores.shape
    if attention_mask is None:
        valid = (
            torch.ones(
                query_length,
                key_length,
                dtype=torch.bool,
                device=scores.device,
            )
            .tril()
            .reshape(1, 1, query_length, key_length)
            .expand(batch, heads, query_length, key_length)
        )
    else:
        valid = (attention_mask[:, :, :, :key_length] >= 0).expand_as(scores)
    maximum = torch.where(
        valid,
        scores,
        torch.full_like(scores, torch.iinfo(torch.int64).min),
    ).amax(dim=-1, keepdim=True)
    centered = torch.where(
        valid,
        scores - maximum,
        torch.full_like(scores, TAGGED_SCORE_SENTINEL),
    )
    if torch.any(centered[valid] > 0):
        raise OverflowError("centered Q20.44 score became positive")
    return centered


def v_residual_baseline_scores_raw(
    authoritative_base_scores_q20_44: Tensor,
    attention_mask: Tensor | None,
) -> Tensor:
    """Expose the unchanged authoritative Q20.44 score path for tracing."""
    return _center_q20_44_scores(authoritative_base_scores_q20_44, attention_mask)


def qk_residual_cross_term_scores_raw(
    authoritative_base_scores_q20_44: Tensor,
    query_q8: Tensor,
    key_q8: Tensor,
    query_residual_s8: Tensor,
    key_residual_s8: Tensor,
    query_scale32_records: Tensor,
    key_scale32_records: Tensor,
    query_residual_scale32_records: Tensor,
    key_residual_scale32_records: Tensor,
    attention_mask: Tensor | None,
    *,
    enable_correction: bool,
) -> Tensor:
    """Carry the authoritative base score and optionally add three corrections."""
    if authoritative_base_scores_q20_44.dtype != torch.int64:
        raise TypeError("Q/K correction requires explicit signed-Q20.44 base scores")
    if not enable_correction:
        return _center_q20_44_scores(authoritative_base_scores_q20_44, attention_mask)
    repeated_key = repeat_kv(key_q8, 7)
    repeated_key_residual = repeat_kv(key_residual_s8, 7)
    repeated_key_scale32 = key_scale32_records.repeat_interleave(7)
    repeated_key_residual_scale32 = key_residual_scale32_records.repeat_interleave(7)
    correction_dots = (
        torch.matmul(
            query_q8.to(torch.int32),
            repeated_key_residual.to(torch.int32).transpose(2, 3),
        ).to(torch.int64),
        torch.matmul(
            query_residual_s8.to(torch.int32),
            repeated_key.to(torch.int32).transpose(2, 3),
        ).to(torch.int64),
        torch.matmul(
            query_residual_s8.to(torch.int32),
            repeated_key_residual.to(torch.int32).transpose(2, 3),
        ).to(torch.int64),
    )
    correction_terms = (
        _scale32_records_to_q20_44(
            correction_dots[0],
            query_scale32_records,
            repeated_key_residual_scale32,
        ),
        _scale32_records_to_q20_44(
            correction_dots[1],
            query_residual_scale32_records,
            repeated_key_scale32,
        ),
        _scale32_records_to_q20_44(
            correction_dots[2],
            query_residual_scale32_records,
            repeated_key_residual_scale32,
        ),
    )
    scores = authoritative_base_scores_q20_44
    for term in correction_terms:
        if torch.any((term > 0) & (scores > INT64_MAX - term)) or torch.any(
            (term < 0) & (scores < -INT64_MAX - 1 - term)
        ):
            raise OverflowError("Q/K residual corrected score exceeds signed-64")
        scores = scores + term
    return _center_q20_44_scores(scores, attention_mask)


def fixed_attention_scores_raw(
    query: Tensor,
    key: Tensor,
    attention_mask: Tensor | None,
    multiplier: Tensor | None = None,
    right_shift: Tensor | None = None,
) -> Tensor:
    if query.dtype not in (torch.int8, torch.int16) or key.dtype != query.dtype:
        raise TypeError("attention Q and K must use the same signed integer dtype")
    if query.dtype == torch.int16 and (
        torch.any(query < -256)
        or torch.any(query > 255)
        or torch.any(key < -256)
        or torch.any(key > 255)
    ):
        raise ValueError("wide attention Q/K must fit signed 9-bit")
    accumulator = torch.matmul(
        query.to(torch.int32),
        key.to(torch.int32).transpose(2, 3),
    ).to(torch.int64)
    if (multiplier is None) != (right_shift is None):
        raise ValueError("attention score multiplier and right shift must be provided together")
    scores = (
        round_shift_even(accumulator, 3)
        if multiplier is None
        else round_shift_even(accumulator * multiplier, right_shift)
    )
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key.shape[-2]]
        valid = causal_mask >= 0
        row_max = torch.where(
            valid,
            scores,
            torch.full_like(scores, torch.iinfo(torch.int64).min),
        ).amax(dim=-1, keepdim=True)
        scores = torch.where(valid, scores - row_max, torch.full_like(scores, -32768))
    else:
        scores = scores - scores.amax(dim=-1, keepdim=True)
    scores = scores.clamp(-32768, 0)
    return scores.to(torch.int16)


def fixed_dynamic_attention_scores_raw(
    query: Tensor,
    key: Tensor,
    query_scale32: Tensor,
    key_scale32: Tensor,
    attention_mask: Tensor | None,
) -> Tensor:
    if query.dtype != torch.int8 or key.dtype != torch.int8:
        raise TypeError("dynamic attention Q and K must be signed int8")
    if query_scale32.shape != query.shape[:-1] or key_scale32.shape != key.shape[:-1]:
        raise ValueError("dynamic attention Scale32 metadata shape mismatch")
    accumulator = torch.matmul(
        query.to(torch.int32),
        key.to(torch.int32).transpose(2, 3),
    ).to(torch.int64)
    if torch.any(accumulator < -1_032_256) or torch.any(accumulator > 1_032_256):
        raise OverflowError("dynamic attention signed-int32 dot bound was exceeded")

    query_sig = torch.bitwise_and(query_scale32, 0xFFFF).unsqueeze(-1)
    key_sig = torch.bitwise_and(key_scale32, 0xFFFF).unsqueeze(-2)
    sig_product = query_sig * key_sig
    pair_sig = round_div_even_unsigned_tensor(sig_product, 1 << 15)
    if torch.any(pair_sig < 0x08000) or torch.any(pair_sig > 0x1FFFC):
        raise OverflowError("dynamic attention pair significand is outside 17 bits")

    query_exp_u8 = torch.bitwise_and(
        torch.bitwise_right_shift(query_scale32, 16), 0xFF
    ).unsqueeze(-1)
    key_exp_u8 = torch.bitwise_and(
        torch.bitwise_right_shift(key_scale32, 16), 0xFF
    ).unsqueeze(-2)
    query_exp = torch.where(query_exp_u8 >= 128, query_exp_u8 - 256, query_exp_u8)
    key_exp = torch.where(key_exp_u8 >= 128, key_exp_u8 - 256, key_exp_u8)
    right_shift = 9 - (query_exp + key_exp)
    if torch.any(right_shift < 1) or torch.any(right_shift > 57):
        raise OverflowError("dynamic attention score shift is outside 1..57")
    product = accumulator * pair_sig
    if torch.any(product < -(1 << 49)) or torch.any(product >= (1 << 49)):
        raise OverflowError("dynamic attention signed-50-bit product overflow")
    scores = round_shift_even(product, right_shift)
    if attention_mask is not None:
        valid = attention_mask[:, :, :, : key.shape[-2]] >= 0
        row_max = torch.where(
            valid,
            scores,
            torch.full_like(scores, torch.iinfo(torch.int64).min),
        ).amax(dim=-1, keepdim=True)
        scores = torch.where(valid, scores - row_max, torch.full_like(scores, -32768))
    else:
        scores = scores - scores.amax(dim=-1, keepdim=True)
    return scores.clamp(-32768, 0).to(torch.int16)


def fixed_q7_attention_scores_raw(
    query: Tensor,
    key: Tensor,
    query_scale32: int,
    key_scale32: int,
    attention_mask: Tensor | None,
) -> Tensor:
    """Reviewed layer-0 fixed-Q7 dot product and Q6.9 centered score."""
    if query.dtype != torch.int16 or key.dtype != torch.int16:
        raise TypeError("fixed-Q7 attention Q and K must be signed int16")
    if query.shape[-1] != ROPE_HEAD_DIM or key.shape[-1] != ROPE_HEAD_DIM:
        raise ValueError("fixed-Q7 attention requires 64-lane heads")
    if torch.any(query.abs() > 23_171) or torch.any(key.abs() > 23_171):
        raise OverflowError("fixed-Q7 attention input exceeds the reviewed bound")
    accumulator = torch.matmul(
        query.to(torch.int64),
        key.to(torch.int64).transpose(2, 3),
    )
    if torch.any(accumulator < -(1 << 37)) or torch.any(accumulator >= (1 << 37)):
        raise OverflowError("fixed-Q7 signed-38-bit dot product overflow")
    pair_sig, right_shift = fixed_q7_score_pair_parameters(
        query_scale32,
        key_scale32,
    )
    product = accumulator * pair_sig
    if torch.any(product < -(1 << 54)) or torch.any(product >= (1 << 54)):
        raise OverflowError("fixed-Q7 signed-55-bit score product overflow")
    scores = round_shift_even(product, right_shift)
    if attention_mask is not None:
        valid = attention_mask[:, :, :, : key.shape[-2]] >= 0
        row_max = torch.where(
            valid,
            scores,
            torch.full_like(scores, torch.iinfo(torch.int64).min),
        ).amax(dim=-1, keepdim=True)
        scores = torch.where(valid, scores - row_max, torch.full_like(scores, -32768))
    else:
        scores = scores - scores.amax(dim=-1, keepdim=True)
    return scores.clamp(-32768, 0).to(torch.int16)


def relative_rope_attention_scores_raw(
    query: Tensor,
    key: Tensor,
    query_scale32: int,
    key_scale32: int,
    attention_mask: Tensor | None,
) -> Tensor:
    """Direct layer-0 relative-RoPE bilinear with exact signed-70 scaling."""
    if query.dtype != torch.int8 or key.dtype != torch.int8:
        raise TypeError("relative-RoPE attention Q and K must be signed int8")
    if query.ndim != 4 or key.ndim != 4:
        raise ValueError("relative-RoPE attention requires rank-4 Q/K tensors")
    if query.shape[0] != key.shape[0] or query.shape[2:] != key.shape[2:]:
        raise ValueError("relative-RoPE attention Q/K batch, sequence, or head dimension differs")
    if query.shape[1] != 14 or key.shape[1] != 2 or query.shape[-1] != ROPE_HEAD_DIM:
        raise ValueError("relative-RoPE attention requires fourteen Q and two K heads")
    sequence = query.shape[2]
    if not 1 <= sequence <= 32768:
        raise ValueError("relative-RoPE sequence is outside 1..32768")

    query_sig, query_exp = unpack_scale32(query_scale32)
    key_sig, key_exp = unpack_scale32(key_scale32)
    significand_product = query_sig * key_sig
    right_shift = 39 - (query_exp + key_exp)
    if not 31 <= right_shift <= 87:
        raise ValueError("relative-RoPE score shift is outside 31..87")
    denominator = 1 << right_shift

    repeated_key = repeat_kv(key, 7).to(torch.int64)
    query_wide = query.to(torch.int64)
    cosine_table = torch.tensor(
        [relative_coefficients_q15(distance)[0] for distance in range(sequence)],
        dtype=torch.int64,
        device=query.device,
    )
    sine_table = torch.tensor(
        [relative_coefficients_q15(distance)[1] for distance in range(sequence)],
        dtype=torch.int64,
        device=query.device,
    )
    precenter = torch.zeros(
        (*query.shape[:2], sequence, sequence),
        dtype=torch.int64,
        device=query.device,
    )
    for position in range(sequence):
        query_head = query_wide[:, :, position, :]
        keys = repeated_key[:, :, : position + 1, :]
        q0 = query_head[..., :32].unsqueeze(2)
        q1 = query_head[..., 32:].unsqueeze(2)
        k0 = keys[..., :32]
        k1 = keys[..., 32:]
        distances = torch.arange(
            position,
            -1,
            -1,
            dtype=torch.int64,
            device=query.device,
        )
        cosine = cosine_table[distances].reshape(1, 1, position + 1, 32)
        sine = sine_table[distances].reshape(1, 1, position + 1, 32)
        a_term = q0 * k0 + q1 * k1
        b_term = q1 * k0 - q0 * k1
        phase_acc = (a_term * cosine - b_term * sine).sum(dim=-1)
        if torch.any(phase_acc.abs() > 48_718_938_112):
            raise OverflowError("relative-RoPE signed-38 phase accumulator overflow")
        scaled_values = [
            round_divide_even_signed(
                int(value) * significand_product,
                denominator,
            )
            for value in phase_acc.reshape(-1).tolist()
        ]
        position_scores = torch.tensor(
            scaled_values,
            dtype=torch.int64,
            device=query.device,
        ).reshape(*phase_acc.shape)
        precenter[:, :, position, : position + 1] = position_scores

    if attention_mask is None:
        valid = (
            torch.ones(sequence, sequence, dtype=torch.bool, device=query.device)
            .tril()
            .reshape(1, 1, sequence, sequence)
            .expand_as(precenter)
        )
    else:
        valid = (attention_mask[:, :, :, :sequence] >= 0).expand_as(precenter)
    row_max = torch.where(
        valid,
        precenter,
        torch.full_like(precenter, torch.iinfo(torch.int64).min),
    ).amax(dim=-1, keepdim=True)
    centered = torch.where(
        valid,
        precenter - row_max,
        torch.full_like(precenter, -32768),
    )
    return centered.clamp(-32768, 0).to(torch.int16)


def absolute_rope_online_attention_raw(
    query: Tensor,
    key: Tensor,
    values: Tensor,
    attention_mask: Tensor | None,
    query_scale32: int,
    key_scale32: int,
) -> Tensor:
    """Exact layer-0 absolute-RoPE Q12.20 online attention recurrence."""
    if query.dtype != torch.int8 or key.dtype != torch.int8 or values.dtype != torch.int8:
        raise TypeError("absolute-RoPE online attention requires signed-int8 Q/K/V")
    if query.ndim != 4 or key.ndim != 4 or values.ndim != 4:
        raise ValueError("absolute-RoPE online attention requires rank-4 Q/K/V tensors")
    batch, query_heads, sequence, head_dim = query.shape
    if query_heads != 14 or key.shape != (batch, 2, sequence, head_dim):
        raise ValueError("absolute-RoPE online attention requires fourteen Q and two K heads")
    if values.shape != (batch, 14, sequence, head_dim) or head_dim != ROPE_HEAD_DIM:
        raise ValueError("absolute-RoPE online attention V mapping or head dimension differs")
    if not 1 <= sequence <= 32768:
        raise ValueError("absolute-RoPE online attention sequence is outside 1..32768")
    if query_scale32 != ABSOLUTE_ROPE_Q_SCALE32 or key_scale32 != ABSOLUTE_ROPE_K_SCALE32:
        raise ValueError("absolute-RoPE online attention static Q/K Scale32 records differ")

    cosine = torch.tensor(
        [absolute_coefficients_q15(position)[0] for position in range(sequence)],
        dtype=torch.int64,
        device=query.device,
    ).reshape(1, 1, sequence, 32)
    sine = torch.tensor(
        [absolute_coefficients_q15(position)[1] for position in range(sequence)],
        dtype=torch.int64,
        device=query.device,
    ).reshape(1, 1, sequence, 32)

    query_wide = query.to(torch.int64)
    q0, q1 = query_wide[..., :32], query_wide[..., 32:]
    rotated_query = torch.cat(
        (q0 * cosine - q1 * sine, q1 * cosine + q0 * sine), dim=-1
    )
    key_wide = key.to(torch.int64)
    k0, k1 = key_wide[..., :32], key_wide[..., 32:]
    rotated_key = torch.cat(
        (k0 * cosine - k1 * sine, k1 * cosine + k0 * sine), dim=-1
    )
    if torch.any(rotated_query < -(1 << 24)) or torch.any(rotated_query >= (1 << 24)):
        raise OverflowError("absolute-RoPE query staging exceeds signed-25")
    if torch.any(rotated_key < -(1 << 24)) or torch.any(rotated_key >= (1 << 24)):
        raise OverflowError("absolute-RoPE key staging exceeds signed-25")

    repeated_key = repeat_kv(rotated_key, 7)
    score_raw = torch.matmul(
        rotated_query, repeated_key.transpose(2, 3)
    )
    if torch.any(score_raw < -(1 << 53)) or torch.any(score_raw >= (1 << 53)):
        raise OverflowError("absolute-RoPE score exceeds signed-54")
    logits = torch.tensor(
        [
            score_raw_to_logit_q12_20(int(value), query_scale32, key_scale32)
            for value in score_raw.detach().cpu().reshape(-1).tolist()
        ],
        dtype=torch.int64,
        device=query.device,
    ).reshape(score_raw.shape)

    if attention_mask is None:
        valid = (
            torch.ones(sequence, sequence, dtype=torch.bool, device=query.device)
            .tril()
            .reshape(1, 1, sequence, sequence)
            .expand(batch, query_heads, sequence, sequence)
        )
    else:
        valid = (attention_mask[:, :, :, :sequence] >= 0).expand(
            batch, query_heads, sequence, sequence
        )

    result = torch.empty_like(values)
    logits_cpu = logits.detach().cpu()
    values_cpu = values.detach().cpu()
    valid_cpu = valid.detach().cpu()
    for batch_index in range(batch):
        for head in range(query_heads):
            for position in range(sequence):
                state = AbsoluteOnlineAttentionState()
                for key_position in range(sequence):
                    if not bool(valid_cpu[batch_index, head, position, key_position]):
                        continue
                    state = update_absolute_online_state(
                        state,
                        int(logits_cpu[batch_index, head, position, key_position]),
                        tuple(
                            int(lane)
                            for lane in values_cpu[
                                batch_index, head, key_position
                            ].tolist()
                        ),
                    )
                result[batch_index, head, position] = torch.tensor(
                    finalize_absolute_online_state(state),
                    dtype=torch.int8,
                    device=result.device,
                )
    return result


def projection_shadow_staged_attention_scores_raw(
    query_shadow: Tensor,
    key_shadow: Tensor,
    query_scale32: int,
    key_scale32: int,
    attention_mask: Tensor | None,
) -> Tensor:
    """Materialize exact layer-0 Q6.9 scores from signed-Q15.16 shadows."""
    if query_shadow.dtype != torch.int32 or key_shadow.dtype != torch.int32:
        raise TypeError("projection-shadow attention requires signed-int32 Q/K")
    if query_shadow.ndim != 4 or key_shadow.ndim != 4:
        raise ValueError("projection-shadow attention requires rank-4 Q/K tensors")
    batch, query_heads, sequence, head_dim = query_shadow.shape
    if query_heads != 14 or key_shadow.shape != (batch, 2, sequence, head_dim):
        raise ValueError("projection-shadow attention requires fourteen Q and two K heads")
    if head_dim != ROPE_HEAD_DIM or not 1 <= sequence <= 32768:
        raise ValueError("projection-shadow attention shape is outside the frozen contract")
    if query_scale32 != ABSOLUTE_ROPE_Q_SCALE32 or key_scale32 != ABSOLUTE_ROPE_K_SCALE32:
        raise ValueError("projection-shadow Q/K Scale32 records differ")

    cosine = torch.tensor(
        [absolute_coefficients_q15(position)[0] for position in range(sequence)],
        dtype=torch.int64,
        device=query_shadow.device,
    ).reshape(1, 1, sequence, 32)
    sine = torch.tensor(
        [absolute_coefficients_q15(position)[1] for position in range(sequence)],
        dtype=torch.int64,
        device=query_shadow.device,
    ).reshape(1, 1, sequence, 32)

    qwide = query_shadow.to(torch.int64)
    q0, q1 = qwide[..., :32], qwide[..., 32:]
    rotated_query = torch.cat(
        (
            round_shift_even(q0 * cosine - q1 * sine, 15),
            round_shift_even(q1 * cosine + q0 * sine, 15),
        ),
        dim=-1,
    )
    kwide = key_shadow.to(torch.int64)
    k0, k1 = kwide[..., :32], kwide[..., 32:]
    rotated_key = torch.cat(
        (
            round_shift_even(k0 * cosine - k1 * sine, 15),
            round_shift_even(k1 * cosine + k0 * sine, 15),
        ),
        dim=-1,
    )
    if torch.any(rotated_query < -(1 << 33)) or torch.any(rotated_query >= (1 << 33)):
        raise OverflowError("projection-shadow rotated query exceeds signed-34")
    if torch.any(rotated_key < -(1 << 33)) or torch.any(rotated_key >= (1 << 33)):
        raise OverflowError("projection-shadow rotated key exceeds signed-34")

    max_query = int(rotated_query.abs().amax())
    max_key = int(rotated_key.abs().amax())
    if max_query * max_key * ROPE_HEAD_DIM > INT64_MAX:
        raise OverflowError("observed projection-shadow dot requires the RTL signed-74 path")
    repeated_key = repeat_kv(rotated_key, 7)
    dot = torch.matmul(rotated_query, repeated_key.transpose(2, 3))

    query_sig, query_exp = unpack_scale32(query_scale32)
    key_sig, key_exp = unpack_scale32(key_scale32)
    shift = 56 - (query_exp + key_exp)
    if not 48 <= shift <= 104:
        raise ValueError("projection-shadow score shift is outside 48..104")
    significand_product = query_sig * key_sig
    scaled = [
        round_divide_even_signed(int(value) * significand_product, 1 << shift)
        for value in dot.detach().cpu().reshape(-1).tolist()
    ]
    scores = torch.tensor(scaled, dtype=torch.int64, device=query_shadow.device).reshape(dot.shape)
    scores = scores.clamp(-32768, 32767)
    if attention_mask is None:
        valid = (
            torch.ones(sequence, sequence, dtype=torch.bool, device=query_shadow.device)
            .tril()
            .reshape(1, 1, sequence, sequence)
            .expand(batch, query_heads, sequence, sequence)
        )
    else:
        valid = (attention_mask[:, :, :, :sequence] >= 0).expand(
            batch, query_heads, sequence, sequence
        )
    row_max = torch.where(
        valid,
        scores,
        torch.full_like(scores, torch.iinfo(torch.int64).min),
    ).amax(dim=-1, keepdim=True)
    centered = torch.where(
        valid,
        scores - row_max,
        torch.full_like(scores, -32768),
    )
    return centered.clamp(-32768, 0).to(torch.int16)


def tile_max_delta_attention_scores_raw(
    query_shadow: Tensor,
    key_shadow: Tensor,
    query_scale32: int,
    key_scale32: int,
    attention_mask: Tensor | None,
) -> Tensor:
    """Encode layer-0 scores as hierarchical signed-Q6.17 deltas."""
    if query_shadow.dtype != torch.int32 or key_shadow.dtype != torch.int32:
        raise TypeError("tile-max attention requires signed-int32 Q/K shadows")
    if query_shadow.ndim != 4 or key_shadow.ndim != 4:
        raise ValueError("tile-max attention requires rank-4 Q/K tensors")
    batch, query_heads, sequence, head_dim = query_shadow.shape
    if query_heads != 14 or key_shadow.shape != (batch, 2, sequence, head_dim):
        raise ValueError("tile-max attention requires fourteen Q and two K heads")
    if head_dim != ROPE_HEAD_DIM or not 1 <= sequence <= 32768:
        raise ValueError("tile-max attention shape is outside the frozen contract")
    if query_scale32 != ABSOLUTE_ROPE_Q_SCALE32 or key_scale32 != ABSOLUTE_ROPE_K_SCALE32:
        raise ValueError("tile-max Q/K Scale32 records differ")

    cosine = torch.tensor(
        [absolute_coefficients_q15(position)[0] for position in range(sequence)],
        dtype=torch.int64,
        device=query_shadow.device,
    ).reshape(1, 1, sequence, 32)
    sine = torch.tensor(
        [absolute_coefficients_q15(position)[1] for position in range(sequence)],
        dtype=torch.int64,
        device=query_shadow.device,
    ).reshape(1, 1, sequence, 32)
    qwide = query_shadow.to(torch.int64)
    q0, q1 = qwide[..., :32], qwide[..., 32:]
    rotated_query = torch.cat(
        (
            round_shift_even(q0 * cosine - q1 * sine, 15),
            round_shift_even(q1 * cosine + q0 * sine, 15),
        ),
        dim=-1,
    )
    kwide = key_shadow.to(torch.int64)
    k0, k1 = kwide[..., :32], kwide[..., 32:]
    rotated_key = torch.cat(
        (
            round_shift_even(k0 * cosine - k1 * sine, 15),
            round_shift_even(k1 * cosine + k0 * sine, 15),
        ),
        dim=-1,
    )
    if torch.any(rotated_query < -(1 << 33)) or torch.any(rotated_query >= (1 << 33)):
        raise OverflowError("tile-max rotated query exceeds signed-34")
    if torch.any(rotated_key < -(1 << 33)) or torch.any(rotated_key >= (1 << 33)):
        raise OverflowError("tile-max rotated key exceeds signed-34")
    max_query = int(rotated_query.abs().amax())
    max_key = int(rotated_key.abs().amax())
    if max_query * max_key * ROPE_HEAD_DIM > INT64_MAX:
        raise OverflowError("observed tile-max dot requires the RTL signed-74 path")
    repeated_key = repeat_kv(rotated_key, 7)
    dot = torch.matmul(rotated_query, repeated_key.transpose(2, 3))

    query_sig, query_exp = unpack_scale32(query_scale32)
    key_sig, key_exp = unpack_scale32(key_scale32)
    shift = 48 - (query_exp + key_exp)
    if not 40 <= shift <= 105:
        raise ValueError("tile-max Q6.17 shift is outside 40..105")
    significand_product = query_sig * key_sig
    if attention_mask is None:
        valid = (
            torch.ones(sequence, sequence, dtype=torch.bool, device=query_shadow.device)
            .tril()
            .reshape(1, 1, sequence, sequence)
            .expand(batch, query_heads, sequence, sequence)
        )
    else:
        valid = (attention_mask[:, :, :, :sequence] >= 0).expand(
            batch, query_heads, sequence, sequence
        )

    output = torch.full(
        dot.shape,
        TILE_DELTA_SENTINEL,
        dtype=torch.int32,
        device=query_shadow.device,
    )
    dot_cpu = dot.detach().cpu()
    valid_cpu = valid.detach().cpu()
    for batch_index in range(batch):
        for head in range(query_heads):
            for position in range(sequence):
                positions = [
                    key_position
                    for key_position in range(sequence)
                    if bool(valid_cpu[batch_index, head, position, key_position])
                ]
                if not positions:
                    raise ValueError("tile-max row has no valid key")
                numerators = [
                    int(dot_cpu[batch_index, head, position, key_position])
                    * significand_product
                    for key_position in positions
                ]
                for numerator in numerators:
                    if not -(1 << 105) <= numerator < (1 << 105):
                        raise OverflowError("tile-max score numerator exceeds signed-106")
                tile_maxima = [
                    max(numerators[start : start + 64])
                    for start in range(0, len(numerators), 64)
                ]
                row_maximum = max(tile_maxima)
                encoded: list[int] = []
                for tile_index, start in enumerate(range(0, len(numerators), 64)):
                    tile_maximum = tile_maxima[tile_index]
                    tile_offset = round_divide_even_signed(
                        tile_maximum - row_maximum,
                        1 << shift,
                    )
                    for numerator in numerators[start : start + 64]:
                        local_delta = round_divide_even_signed(
                            numerator - tile_maximum,
                            1 << shift,
                        )
                        merged = tile_offset + local_delta
                        if merged > 0:
                            raise OverflowError("tile-max hierarchical delta became positive")
                        if merged <= TILE_DELTA_SENTINEL:
                            encoded.append(TILE_DELTA_SENTINEL)
                        elif not -(1 << 23) <= merged < (1 << 23):
                            raise OverflowError("tile-max delta exceeds signed-24")
                        else:
                            encoded.append(merged)
                output[
                    batch_index,
                    head,
                    position,
                    torch.tensor(positions, device=output.device),
                ] = torch.tensor(encoded, dtype=torch.int32, device=output.device)
    return output


def tile_bfp_attention_scores_raw(
    query_shadow: Tensor,
    key_shadow: Tensor,
    query_scale32: int,
    key_scale32: int,
    attention_mask: Tensor | None,
) -> Tensor:
    """Encode layer-0 scores as tile block-floating common-Q17 deltas."""
    if query_shadow.dtype != torch.int32 or key_shadow.dtype != torch.int32:
        raise TypeError("tile-BFP attention requires signed-int32 Q/K shadows")
    if query_shadow.ndim != 4 or key_shadow.ndim != 4:
        raise ValueError("tile-BFP attention requires rank-4 Q/K tensors")
    batch, query_heads, sequence, head_dim = query_shadow.shape
    if query_heads != 14 or key_shadow.shape != (batch, 2, sequence, head_dim):
        raise ValueError("tile-BFP attention requires fourteen Q and two K heads")
    if head_dim != ROPE_HEAD_DIM or not 1 <= sequence <= 32768:
        raise ValueError("tile-BFP attention shape is outside the frozen contract")
    if query_scale32 != ABSOLUTE_ROPE_Q_SCALE32 or key_scale32 != ABSOLUTE_ROPE_K_SCALE32:
        raise ValueError("tile-BFP Q/K Scale32 records differ")

    cosine = torch.tensor(
        [absolute_coefficients_q15(position)[0] for position in range(sequence)],
        dtype=torch.int64,
        device=query_shadow.device,
    ).reshape(1, 1, sequence, 32)
    sine = torch.tensor(
        [absolute_coefficients_q15(position)[1] for position in range(sequence)],
        dtype=torch.int64,
        device=query_shadow.device,
    ).reshape(1, 1, sequence, 32)
    qwide = query_shadow.to(torch.int64)
    q0, q1 = qwide[..., :32], qwide[..., 32:]
    rotated_query = torch.cat(
        (
            round_shift_even(q0 * cosine - q1 * sine, 15),
            round_shift_even(q1 * cosine + q0 * sine, 15),
        ),
        dim=-1,
    )
    kwide = key_shadow.to(torch.int64)
    k0, k1 = kwide[..., :32], kwide[..., 32:]
    rotated_key = torch.cat(
        (
            round_shift_even(k0 * cosine - k1 * sine, 15),
            round_shift_even(k1 * cosine + k0 * sine, 15),
        ),
        dim=-1,
    )
    if torch.any(rotated_query < -(1 << 33)) or torch.any(rotated_query >= (1 << 33)):
        raise OverflowError("tile-BFP rotated query exceeds signed-34")
    if torch.any(rotated_key < -(1 << 33)) or torch.any(rotated_key >= (1 << 33)):
        raise OverflowError("tile-BFP rotated key exceeds signed-34")
    max_query = int(rotated_query.abs().amax())
    max_key = int(rotated_key.abs().amax())
    if max_query * max_key * ROPE_HEAD_DIM > INT64_MAX:
        raise OverflowError("observed tile-BFP dot requires the RTL signed-74 path")
    repeated_key = repeat_kv(rotated_key, 7)
    dot = torch.matmul(rotated_query, repeated_key.transpose(2, 3))

    query_sig, query_exp = unpack_scale32(query_scale32)
    key_sig, key_exp = unpack_scale32(key_scale32)
    score_exponent = query_exp + key_exp
    tile_offset_shift = 48 - score_exponent
    significand_product = query_sig * key_sig
    if attention_mask is None:
        valid = (
            torch.ones(sequence, sequence, dtype=torch.bool, device=query_shadow.device)
            .tril()
            .reshape(1, 1, sequence, sequence)
            .expand(batch, query_heads, sequence, sequence)
        )
    else:
        valid = (attention_mask[:, :, :, :sequence] >= 0).expand(
            batch, query_heads, sequence, sequence
        )

    output = torch.full(
        dot.shape,
        TILE_DELTA_SENTINEL,
        dtype=torch.int32,
        device=query_shadow.device,
    )
    dot_cpu = dot.detach().cpu()
    valid_cpu = valid.detach().cpu()
    for batch_index in range(batch):
        for head in range(query_heads):
            for position in range(sequence):
                positions = [
                    key_position
                    for key_position in range(sequence)
                    if bool(valid_cpu[batch_index, head, position, key_position])
                ]
                if not positions:
                    raise ValueError("tile-BFP row has no valid key")
                numerators = [
                    int(dot_cpu[batch_index, head, position, key_position])
                    * significand_product
                    for key_position in positions
                ]
                if any(not -(1 << 105) <= value < (1 << 105) for value in numerators):
                    raise OverflowError("tile-BFP score numerator exceeds signed-106")
                encoded_tiles: list[tuple[int, int, list[int]]] = []
                for start in range(0, len(numerators), 64):
                    tile = numerators[start : start + 64]
                    tile_maximum = max(tile)
                    tile_minimum = min(tile)
                    score_range = tile_maximum - tile_minimum
                    fraction_bits = None
                    for candidate_fraction in range(17, -1, -1):
                        shift = 65 - score_exponent - candidate_fraction
                        if round_divide_even_signed(score_range, 1 << shift) <= 8388607:
                            fraction_bits = candidate_fraction
                            break
                    if fraction_bits is None:
                        raise OverflowError("tile-BFP range cannot fit signed-24")
                    shift = 65 - score_exponent - fraction_bits
                    mantissas = [
                        round_divide_even_signed(value - tile_maximum, 1 << shift)
                        for value in tile
                    ]
                    if any(value > 0 or value < -8388608 for value in mantissas):
                        raise OverflowError("tile-BFP mantissa violates signed-24")
                    encoded_tiles.append((tile_maximum, fraction_bits, mantissas))
                row_maximum = max(tile[0] for tile in encoded_tiles)
                encoded: list[int] = []
                for tile_maximum, fraction_bits, mantissas in encoded_tiles:
                    tile_offset_q17 = round_divide_even_signed(
                        tile_maximum - row_maximum,
                        1 << tile_offset_shift,
                    )
                    for mantissa in mantissas:
                        global_delta_q17 = (
                            tile_offset_q17 + (mantissa << (17 - fraction_bits))
                        )
                        if not -(1 << 31) <= global_delta_q17 < (1 << 31):
                            raise OverflowError("tile-BFP reconstruction exceeds signed-32")
                        if global_delta_q17 > 0:
                            raise OverflowError("tile-BFP reconstruction became positive")
                        encoded.append(global_delta_q17)
                output[
                    batch_index,
                    head,
                    position,
                    torch.tensor(positions, device=output.device),
                ] = torch.tensor(encoded, dtype=torch.int32, device=output.device)
    return output


def _round_shift_even_or_zero(value: Tensor, shift: Tensor) -> Tensor:
    """RTL-compatible signed RNE; signed-64 values shifted by >=64 become zero."""
    if torch.any(shift < 0):
        raise ValueError("right shift must be nonnegative")
    rounded = round_shift_even(value, shift.clamp(max=63))
    return torch.where(shift >= 64, torch.zeros_like(rounded), rounded)


def _normalize_tagged_tensor(
    value: Tensor,
    base_exponent: Tensor,
) -> tuple[Tensor, Tensor]:
    """Vector form of normalize_tagged for observed signed-64 RoPE values."""
    magnitude = value.abs()
    shift = torch.zeros_like(value, dtype=torch.int64)
    for candidate in range(1, 33):
        shift = torch.where(
            magnitude >= (1 << (30 + candidate)),
            torch.full_like(shift, candidate),
            shift,
        )
    mantissa = round_shift_even(value, shift)
    carry = (mantissa < -(1 << 31)) | (mantissa >= (1 << 31))
    if torch.any(carry):
        shift = shift + carry.to(torch.int64)
        mantissa = round_shift_even(value, shift)
    exponent = base_exponent + shift
    zero = value == 0
    mantissa = torch.where(zero, torch.zeros_like(mantissa), mantissa)
    exponent = torch.where(zero, torch.zeros_like(exponent), exponent)
    if torch.any(mantissa < -(1 << 31)) or torch.any(mantissa >= (1 << 31)):
        raise OverflowError("tagged RoPE mantissa exceeds signed-32")
    if torch.any(exponent < -96) or torch.any(exponent > 31):
        raise OverflowError("tagged RoPE exponent exceeds the frozen range")
    return mantissa, exponent


def native_accumulator_tagged_rope_raw(
    accumulator: Tensor,
    scale32_records: Tensor,
) -> tuple[Tensor, Tensor]:
    """Apply the frozen per-channel Scale32 absolute-RoPE tagged transform."""
    if accumulator.dtype != torch.int32 or accumulator.ndim != 4:
        raise TypeError("native tagged RoPE requires rank-4 signed-int32 accumulators")
    _, heads, sequence, head_dim = accumulator.shape
    if head_dim != ROPE_HEAD_DIM or scale32_records.shape != (heads * head_dim,):
        raise ValueError("native tagged RoPE geometry or Scale32 record count differs")
    records = scale32_records.reshape(heads, head_dim)
    if torch.any(torch.bitwise_right_shift(records, 24) != 0):
        raise ValueError("native tagged RoPE Scale32 reserved byte is nonzero")
    significand = torch.bitwise_and(records, 0xFFFF)
    exponent_u8 = torch.bitwise_and(torch.bitwise_right_shift(records, 16), 0xFF)
    exponent = torch.where(exponent_u8 >= 128, exponent_u8 - 256, exponent_u8)
    if torch.any(significand < 0x8000) or torch.any(significand > 0xFFFF):
        raise ValueError("native tagged RoPE Scale32 significand is malformed")
    if torch.any(exponent < -24) or torch.any(exponent > 4):
        raise ValueError("native tagged RoPE Scale32 exponent is out of range")

    coefficient_pairs = [absolute_coefficients_q15(position) for position in range(sequence)]
    cosine = torch.tensor(
        [item[0] for item in coefficient_pairs],
        dtype=torch.int64,
        device=accumulator.device,
    ).reshape(1, 1, sequence, 32)
    sine = torch.tensor(
        [item[1] for item in coefficient_pairs],
        dtype=torch.int64,
        device=accumulator.device,
    ).reshape(1, 1, sequence, 32)
    wide = accumulator.to(torch.int64)
    low, high = wide[..., :32], wide[..., 32:]
    sig_low = significand[:, :32].reshape(1, heads, 1, 32)
    sig_high = significand[:, 32:].reshape(1, heads, 1, 32)
    exp_low = exponent[:, :32].reshape(1, heads, 1, 32)
    exp_high = exponent[:, 32:].reshape(1, heads, 1, 32)
    common = torch.maximum(exp_low, exp_high)
    low_cos = round_shift_even(low * sig_low * cosine, common - exp_low)
    high_sin = round_shift_even(high * sig_high * sine, common - exp_high)
    high_cos = round_shift_even(high * sig_high * cosine, common - exp_high)
    low_sin = round_shift_even(low * sig_low * sine, common - exp_low)
    term_bound = max(
        int(low_cos.abs().amax()),
        int(high_sin.abs().amax()),
        int(high_cos.abs().amax()),
        int(low_sin.abs().amax()),
    )
    if term_bound >= (1 << 62):
        raise OverflowError("observed native tagged RoPE terms require a wider tensor path")
    real = low_cos - high_sin
    imag = high_cos + low_sin
    real_mantissa, real_exponent = _normalize_tagged_tensor(real, common - 30)
    imag_mantissa, imag_exponent = _normalize_tagged_tensor(imag, common - 30)
    return (
        torch.cat((real_mantissa, imag_mantissa), dim=-1),
        torch.cat((real_exponent, imag_exponent), dim=-1),
    )


def _sum_tagged_products(aligned_products: Tensor) -> tuple[Tensor, Tensor]:
    """Return an exact signed accumulator as base-2^30 high/low limbs."""
    limb_base = 1 << 30
    high = torch.div(aligned_products, limb_base, rounding_mode="floor")
    low = aligned_products - high * limb_base
    low_sum = low.sum(dim=-1)
    high_sum = high.sum(dim=-1)
    carry = torch.bitwise_right_shift(low_sum, 30)
    low_total = torch.bitwise_and(low_sum, limb_base - 1)
    high_total = high_sum + carry
    return high_total, low_total


def _convert_tagged_accumulator_q20_44(
    high: Tensor,
    low: Tensor,
    common_exponent: Tensor,
) -> Tensor:
    """Convert an exact base-2^30 signed accumulator to signed Q20.44."""
    limb_base = 1 << 30
    conversion_shift = -(common_exponent + 41)

    negative = high < 0
    negative_low = torch.where(low == 0, torch.zeros_like(low), limb_base - low)
    negative_high = torch.where(low == 0, -high, -high - 1)
    magnitude_high = torch.where(negative, negative_high, high)
    magnitude_low = torch.where(negative, negative_low, low)

    high_bits = torch.zeros_like(magnitude_high)
    for width in range(1, 40):
        high_bits = torch.where(
            magnitude_high >= (1 << (width - 1)),
            torch.full_like(high_bits, width),
            high_bits,
        )
    low_bits = torch.zeros_like(magnitude_low)
    for width in range(1, 31):
        low_bits = torch.where(
            magnitude_low >= (1 << (width - 1)),
            torch.full_like(low_bits, width),
            low_bits,
        )
    full_bits = torch.where(magnitude_high > 0, high_bits + 30, low_bits)
    tensor_path = (
        (conversion_shift >= 0)
        & (conversion_shift <= 62)
        & ((full_bits - conversion_shift) < 63)
    )
    if not torch.all(tensor_path):
        flat_high = high.detach().cpu().reshape(-1).tolist()
        flat_low = low.detach().cpu().reshape(-1).tolist()
        flat_shift = conversion_shift.detach().cpu().reshape(-1).tolist()
        converted: list[int] = []
        for high_value, low_value, shift_value in zip(
            flat_high,
            flat_low,
            flat_shift,
            strict=True,
        ):
            value = int(high_value) * limb_base + int(low_value)
            shift_int = int(shift_value)
            if shift_int <= 0:
                rounded = value << -shift_int
            else:
                magnitude = abs(value)
                quotient, remainder = divmod(magnitude, 1 << shift_int)
                half = 1 << (shift_int - 1)
                if remainder > half or (remainder == half and quotient & 1):
                    quotient += 1
                rounded = -quotient if value < 0 else quotient
            if rounded < -(1 << 63) or rounded >= (1 << 63):
                raise OverflowError("tagged Q20.44 score exceeds signed-64")
            converted.append(rounded)
        return torch.tensor(
            converted,
            dtype=torch.int64,
            device=high.device,
        ).reshape(high.shape)

    shift = conversion_shift
    small = shift <= 30
    small_shift = shift.clamp(min=0, max=30)
    small_quotient = (
        torch.bitwise_left_shift(magnitude_high, 30 - small_shift)
        + torch.bitwise_right_shift(magnitude_low, small_shift)
    )
    small_mask = torch.where(
        small_shift == 0,
        torch.zeros_like(small_shift),
        torch.bitwise_left_shift(torch.ones_like(small_shift), small_shift) - 1,
    )
    small_remainder = torch.bitwise_and(magnitude_low, small_mask)

    large_shift = (shift - 30).clamp(min=1, max=32)
    large_quotient = torch.bitwise_right_shift(magnitude_high, large_shift)
    large_mask = torch.bitwise_left_shift(torch.ones_like(large_shift), large_shift) - 1
    large_remainder = (
        torch.bitwise_and(magnitude_high, large_mask) * limb_base
        + magnitude_low
    )
    quotient = torch.where(small, small_quotient, large_quotient)
    remainder = torch.where(small, small_remainder, large_remainder)
    half = torch.where(
        shift == 0,
        torch.zeros_like(shift),
        torch.bitwise_left_shift(torch.ones_like(shift), (shift - 1).clamp(min=0)),
    )
    increment = (remainder > half) | (
        (remainder == half) & ((quotient & 1) == 1) & (shift != 0)
    )
    rounded_magnitude = quotient + increment.to(torch.int64)
    return torch.where(negative, -rounded_magnitude, rounded_magnitude)


def native_accumulator_tagged_attention_scores_raw(
    query_accumulator: Tensor,
    key_accumulator: Tensor,
    query_scale32_records: Tensor,
    key_scale32_records: Tensor,
    attention_mask: Tensor | None,
) -> Tensor:
    """All-layer tagged absolute-RoPE score path with signed-Q20.44 output."""
    query_mantissa, query_exponent = native_accumulator_tagged_rope_raw(
        query_accumulator,
        query_scale32_records,
    )
    key_mantissa, key_exponent = native_accumulator_tagged_rope_raw(
        key_accumulator,
        key_scale32_records,
    )
    batch, query_heads, sequence, head_dim = query_mantissa.shape
    if query_heads != 14 or key_mantissa.shape != (batch, 2, sequence, head_dim):
        raise ValueError("native tagged attention requires fourteen Q and two K heads")
    repeated_key_mantissa = repeat_kv(key_mantissa, 7)
    repeated_key_exponent = repeat_kv(key_exponent, 7)
    scores = torch.zeros(
        (batch, query_heads, sequence, sequence),
        dtype=torch.int64,
        device=query_mantissa.device,
    )
    for query_position in range(sequence):
        q_mantissa = query_mantissa[:, :, query_position : query_position + 1, :]
        q_exponent = query_exponent[:, :, query_position : query_position + 1, :]
        k_mantissa = repeated_key_mantissa[:, :, : query_position + 1, :]
        k_exponent = repeated_key_exponent[:, :, : query_position + 1, :]
        products = q_mantissa * k_mantissa
        product_exponent = q_exponent + k_exponent
        common_exponent = product_exponent.amax(dim=-1)
        aligned = _round_shift_even_or_zero(
            products,
            common_exponent.unsqueeze(-1) - product_exponent,
        )
        accumulator_high, accumulator_low = _sum_tagged_products(aligned)
        converted = _convert_tagged_accumulator_q20_44(
            accumulator_high,
            accumulator_low,
            common_exponent,
        )
        scores[:, :, query_position, : query_position + 1] = converted

    if attention_mask is None:
        valid = (
            torch.ones(sequence, sequence, dtype=torch.bool, device=scores.device)
            .tril()
            .reshape(1, 1, sequence, sequence)
            .expand_as(scores)
        )
    else:
        valid = (attention_mask[:, :, :, :sequence] >= 0).expand_as(scores)
    row_maximum = torch.where(
        valid,
        scores,
        torch.full_like(scores, torch.iinfo(torch.int64).min),
    ).amax(dim=-1, keepdim=True)
    centered = torch.where(
        valid,
        scores - row_maximum,
        torch.full_like(scores, TAGGED_SCORE_SENTINEL),
    )
    if torch.any(centered[valid] > 0):
        raise OverflowError("tagged centered score became positive")
    return centered


def native_accumulator_tagged_softmax_raw(scores_q20_44: Tensor) -> Tensor:
    """Q20.44 -> Q5.26 -> Q1.31 exp -> Q0.15 staged softmax."""
    if scores_q20_44.dtype != torch.int64 or torch.any(scores_q20_44 > 0):
        raise TypeError("native tagged softmax requires nonpositive signed-int64 Q20.44")
    delta_q26 = round_shift_even(
        scores_q20_44,
        TAGGED_SCORE_FRAC - TAGGED_EXP_INPUT_FRAC,
    )
    magnitude = -delta_q26
    exp_limit = 16 << TAGGED_EXP_INPUT_FRAC
    table = torch.tensor(
        absolute_exp_table_q31(),
        dtype=torch.int64,
        device=scores_q20_44.device,
    )
    interpolation_fraction_bits = TAGGED_EXP_INPUT_FRAC - 4
    index = torch.bitwise_right_shift(magnitude, interpolation_fraction_bits)
    fraction = torch.bitwise_and(magnitude, (1 << interpolation_fraction_bits) - 1)
    interpolation_index = index.clamp(max=255)
    lower = table[interpolation_index]
    upper = table[interpolation_index + 1]
    interpolated = lower + round_shift_even(
        (upper - lower) * fraction,
        interpolation_fraction_bits,
    )
    weights = torch.where(
        delta_q26 >= 0,
        torch.full_like(interpolated, 1 << 31),
        torch.where(
            delta_q26 < -exp_limit,
            torch.zeros_like(interpolated),
            torch.where(index >= 256, table[256], interpolated),
        ),
    )
    denominator = weights.sum(dim=-1, keepdim=True)
    if torch.any(denominator <= 0) or torch.any(denominator >= (1 << 48)):
        raise OverflowError("native tagged softmax denominator violates unsigned-48")
    return round_div_even_unsigned(weights * 32767, denominator).clamp(max=32767)


def round_div_even_unsigned_tensor(value: Tensor, denominator: int) -> Tensor:
    if denominator <= 0 or torch.any(value < 0):
        raise ValueError("unsigned tensor division requires nonnegative values")
    quotient = torch.div(value, denominator, rounding_mode="floor")
    remainder = value - quotient * denominator
    doubled = remainder * 2
    increment = (doubled > denominator) | (
        (doubled == denominator) & ((quotient & 1) == 1)
    )
    return quotient + increment.to(torch.int64)


def round_div_even_unsigned(numerator: Tensor, denominator: Tensor) -> Tensor:
    if torch.any(numerator < 0) or torch.any(denominator <= 0):
        raise ValueError("unsigned rounded division requires numerator >= 0 and denominator > 0")
    quotient = torch.div(numerator, denominator, rounding_mode="floor")
    remainder = numerator - quotient * denominator
    doubled = remainder * 2
    increment = (doubled > denominator) | (
        (doubled == denominator) & ((quotient & 1) == 1)
    )
    return quotient + increment.to(torch.int64)


def tile_max_delta_softmax_raw(scores_q6_17: Tensor) -> Tensor:
    if scores_q6_17.dtype != torch.int32:
        raise TypeError("tile-max softmax scores must be signed int32 Q6.17")
    if torch.any(scores_q6_17 > 0):
        raise ValueError("tile-max softmax requires nonpositive centered scores")
    weights = torch.tensor(
        [
            0
            if int(score) <= TILE_DELTA_SENTINEL
            else absolute_exp_q31(int(score) << 3)
            for score in scores_q6_17.detach().cpu().reshape(-1).tolist()
        ],
        dtype=torch.int64,
        device=scores_q6_17.device,
    ).reshape(scores_q6_17.shape)
    exp_sum = weights.sum(dim=-1, keepdim=True)
    if torch.any(exp_sum <= 0):
        raise RuntimeError("tile-max softmax denominator is zero")
    return round_div_even_unsigned(weights * 32767, exp_sum).clamp(max=32767)


def tile_bfp_softmax_raw(scores_q17: Tensor) -> Tensor:
    if scores_q17.dtype != torch.int32:
        raise TypeError("tile-BFP softmax scores must be signed int32 Q17")
    if torch.any(scores_q17 > 0):
        raise ValueError("tile-BFP softmax requires nonpositive centered scores")
    weights = torch.tensor(
        [
            0
            if int(score) <= TILE_DELTA_SENTINEL
            else absolute_exp_q31(int(score) << 3)
            for score in scores_q17.detach().cpu().reshape(-1).tolist()
        ],
        dtype=torch.int64,
        device=scores_q17.device,
    ).reshape(scores_q17.shape)
    exp_sum = weights.sum(dim=-1, keepdim=True)
    if torch.any(exp_sum <= 0):
        raise RuntimeError("tile-BFP softmax denominator is zero")
    return round_div_even_unsigned(weights * 32767, exp_sum).clamp(max=32767)


def fixed_softmax_raw(scores_q6_9: Tensor) -> Tensor:
    if scores_q6_9.dtype != torch.int16:
        raise TypeError("softmax scores must be signed int16 Q6.9")
    scores = scores_q6_9.to(torch.int64)
    max_score = scores.amax(dim=-1, keepdim=True)
    magnitude = max_score - scores
    table_index = torch.div(
        magnitude + SOFTMAX_EXP_ROUND,
        SOFTMAX_EXP_STEP,
        rounding_mode="floor",
    )
    table = torch.tensor(SOFTMAX_EXP_LUT, dtype=torch.int64, device=scores.device)
    weights = table[table_index.clamp(max=len(SOFTMAX_EXP_LUT) - 1)]
    weights = torch.where(table_index < len(SOFTMAX_EXP_LUT), weights, torch.zeros_like(weights))
    exp_sum = weights.sum(dim=-1, keepdim=True)
    return round_div_even_unsigned(weights << SOFTMAX_PROB_FRAC, exp_sum).to(torch.int64)


def fixed_attention_value_raw(probabilities_q15: Tensor, values: Tensor) -> Tensor:
    if values.dtype != torch.int8:
        raise TypeError("attention V must be signed int8")
    accumulator = torch.matmul(
        probabilities_q15.to(torch.int64),
        values.to(torch.int64),
    )
    return round_shift_even(accumulator, SOFTMAX_PROB_FRAC).clamp(-128, 127).to(torch.int8)


def round_div_even_signed_tensor(numerator: Tensor, denominator: Tensor) -> Tensor:
    """Exact signed round-to-nearest-even division for broadcastable tensors."""
    if numerator.dtype != torch.int64 or denominator.dtype != torch.int64:
        raise TypeError("signed rounded division requires signed-int64 tensors")
    if torch.any(denominator <= 0):
        raise ValueError("signed rounded division requires positive denominators")
    if torch.any(numerator == torch.iinfo(torch.int64).min):
        raise OverflowError("signed rounded division cannot negate signed-int64 minimum")
    magnitude = numerator.abs()
    quotient = torch.div(magnitude, denominator, rounding_mode="floor")
    remainder = magnitude - quotient * denominator
    doubled = remainder * 2
    increment = (doubled > denominator) | (
        (doubled == denominator) & ((quotient & 1) == 1)
    )
    rounded = quotient + increment.to(torch.int64)
    return torch.where(numerator < 0, -rounded, rounded)


def v_residual_value_accumulators_raw(
    probabilities_q15: Tensor,
    values: Tensor,
    residual_values_s4: Tensor,
    baseline_v_scale32_records: Tensor,
    residual_v_scale32_records: Tensor,
    *,
    enable_correction: bool,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return baseline, converted correction, and checked corrected accumulators."""
    if probabilities_q15.dtype != torch.int64:
        raise TypeError("V residual correction requires unsigned Q0.15 int64 storage")
    if values.dtype != torch.int8 or residual_values_s4.dtype != torch.int8:
        raise TypeError("V residual correction requires signed-int8 V and residual storage")
    if probabilities_q15.ndim != 4 or values.ndim != 4:
        raise ValueError("V residual correction requires rank-4 attention tensors")
    batch, query_heads, query_length, key_length = probabilities_q15.shape
    if query_heads != 14 or values.shape != (batch, 14, key_length, ROPE_HEAD_DIM):
        raise ValueError("V residual correction requires fourteen repeated 64-lane V heads")
    if residual_values_s4.shape != values.shape:
        raise ValueError("V residual storage geometry differs from baseline V")
    if baseline_v_scale32_records.shape != (2,) or residual_v_scale32_records.shape != (2,):
        raise ValueError("V residual Scale32 metadata must cover two KV heads")
    if torch.any(probabilities_q15 < 0) or torch.any(probabilities_q15 > 0xFFFF):
        raise ValueError("V residual probabilities are outside unsigned Q0.15")
    if torch.any(residual_values_s4 < -7) or torch.any(residual_values_s4 > 7):
        raise ValueError("V residual values must be canonical signed-4 [-7,+7]")

    baseline = torch.matmul(
        probabilities_q15,
        values.to(torch.int64),
    )
    if torch.any(baseline < -(1 << 31)) or torch.any(baseline > INT32_MAX):
        raise OverflowError("authoritative attention-value accumulator exceeds signed-32")
    correction = torch.zeros_like(baseline)
    if enable_correction:
        correction_raw = torch.matmul(
            probabilities_q15,
            residual_values_s4.to(torch.int64),
        )
        for query_head in range(query_heads):
            kv_head = query_head // 7
            baseline_sig, baseline_exp = unpack_scale32(
                int(baseline_v_scale32_records[kv_head])
            )
            residual_sig, residual_exp = unpack_scale32(
                int(residual_v_scale32_records[kv_head])
            )
            numerator = correction_raw[:, query_head] * residual_sig
            delta = residual_exp - baseline_exp
            if delta >= 0:
                if torch.any(numerator > (INT64_MAX >> delta)) or torch.any(
                    numerator < (-(1 << 63) >> delta)
                ):
                    raise OverflowError("V residual Scale32 numerator exceeds signed-64")
                numerator = torch.bitwise_left_shift(numerator, delta)
                denominator_value = baseline_sig
            else:
                denominator_value = baseline_sig << -delta
            denominator = torch.full_like(numerator, denominator_value)
            correction[:, query_head] = round_div_even_signed_tensor(
                numerator,
                denominator,
            )
        if torch.any(correction < -(1 << 31)) or torch.any(correction > INT32_MAX):
            raise OverflowError("V residual correction exceeds signed-32")
    corrected = baseline + correction
    if torch.any(corrected < -(1 << 31)) or torch.any(corrected > INT32_MAX):
        raise OverflowError("corrected attention-value accumulator exceeds signed-32")
    return baseline, correction, corrected


def v_residual_attention_value_raw(
    probabilities_q15: Tensor,
    values: Tensor,
    residual_values_s4: Tensor,
    baseline_v_scale32_records: Tensor,
    residual_v_scale32_records: Tensor,
    *,
    enable_correction: bool,
) -> Tensor:
    """Apply the frozen sum-then-convert correction before unchanged requantization."""
    _, _, corrected = v_residual_value_accumulators_raw(
        probabilities_q15,
        values,
        residual_values_s4,
        baseline_v_scale32_records,
        residual_v_scale32_records,
        enable_correction=enable_correction,
    )
    return round_shift_even(corrected, SOFTMAX_PROB_FRAC).clamp(-128, 127).to(torch.int8)


def fixed_silu_gate_raw(
    gate: Tensor,
    up: Tensor,
    gate_scale: float,
    up_scale: float,
    multiplier: Tensor,
    right_shift: Tensor,
) -> Tensor:
    if gate.dtype != torch.int8 or up.dtype != torch.int8:
        raise TypeError("SiLU gate and up-projection values must be signed int8")
    gate_q6_9 = torch.round(gate.to(torch.float64) * gate_scale * (1 << 9))
    up_q6_9 = torch.round(up.to(torch.float64) * up_scale * (1 << 9))
    gate_q6_9 = gate_q6_9.clamp(-32768, 32767).to(torch.int64)
    up_q6_9 = up_q6_9.clamp(-32768, 32767).to(torch.int64)
    table_index = torch.bitwise_right_shift(gate_q6_9, 6).clamp(-64, 64) + 64
    table = torch.tensor(SILU_LUT, dtype=torch.int64, device=gate.device)
    silu_q3_12 = table[table_index]
    product_q9_21 = silu_q3_12 * up_q6_9
    requant_product = product_q9_21 * multiplier
    return round_shift_even(requant_product, right_shift).clamp(-128, 127).to(torch.int8)


def fixed_residual_add(lhs: Tensor, rhs: Tensor, scale: float) -> Tensor:
    lhs_raw = quantize_int8(lhs, scale).to(torch.int16)
    rhs_raw = quantize_int8(rhs, scale).to(torch.int16)
    output = (lhs_raw + rhs_raw).clamp(-128, 127).to(torch.int8)
    return output.to(lhs.dtype) * scale


def _independent_down_projection_fusion_layer(
    accumulator: Tensor,
    residual: Tensor,
    accumulator_scale32: tuple[int, ...],
    residual_scale32: int,
    destination_scale32: int,
) -> tuple[Tensor, dict[str, int]]:
    if accumulator.dtype != torch.int32 or residual.dtype != torch.int8:
        raise TypeError("independent fusion replay requires signed-int32/int8 inputs")
    if accumulator.shape != residual.shape or accumulator.shape[-1] != RMS_HIDDEN_SIZE:
        raise ValueError("independent fusion replay geometry differs")
    accumulator_rows = accumulator.detach().cpu().reshape(-1, RMS_HIDDEN_SIZE).tolist()
    residual_rows = residual.detach().cpu().reshape(-1, RMS_HIDDEN_SIZE).tolist()
    output_rows: list[list[int]] = []
    positive_saturations = 0
    negative_saturations = 0
    maximum_absolute_numerator = 0
    maximum_denominator = 0
    common_exponent_minimum = 127
    common_exponent_maximum = -128
    for accumulator_row, residual_row in zip(
        accumulator_rows, residual_rows, strict=True
    ):
        output_row: list[int] = []
        for lane, (accumulator_value, residual_value) in enumerate(
            zip(accumulator_row, residual_row, strict=True)
        ):
            result = fuse_lane(
                accumulator_value,
                residual_value,
                accumulator_scale32[lane],
                residual_scale32,
                destination_scale32,
            )
            output_row.append(result.output_s8)
            positive_saturations += int(result.positive_saturation)
            negative_saturations += int(result.negative_saturation)
            maximum_absolute_numerator = max(
                maximum_absolute_numerator,
                abs(result.numerator_s96),
            )
            maximum_denominator = max(maximum_denominator, result.denominator_u64)
            common_exponent_minimum = min(
                common_exponent_minimum,
                result.common_exponent,
            )
            common_exponent_maximum = max(
                common_exponent_maximum,
                result.common_exponent,
            )
        output_rows.append(output_row)
    output = torch.tensor(output_rows, dtype=torch.int8).reshape(residual.shape)
    return output.to(residual.device), {
        "rows": len(output_rows),
        "lanes": accumulator.numel(),
        "positive_saturations": positive_saturations,
        "negative_saturations": negative_saturations,
        "maximum_absolute_numerator": maximum_absolute_numerator,
        "maximum_denominator": maximum_denominator,
        "common_exponent_minimum": common_exponent_minimum,
        "common_exponent_maximum": common_exponent_maximum,
    }


class DownProjectionResidualFusionRuntime:
    """Hash-bound composed hook plus independent all-lane replay checking."""

    def __init__(self, mode: str) -> None:
        if mode not in {"baseline", "candidate"}:
            raise ValueError("down-projection fusion runtime mode differs")
        self.mode = mode
        self.metadata = down_projection_residual_fusion_metadata()
        self.hook = ExactScale32AllLayerHook(self.metadata)
        self._next_layer = 0
        self._layer_records: list[dict[str, Any]] = []
        self._completed: list[dict[str, Any]] = []

    def _begin_pass(self) -> None:
        self.hook.begin_pass()
        self._next_layer = 0
        self._layer_records = []

    def apply_layer(
        self,
        layer_index: int,
        accumulator: Tensor,
        residual: Tensor,
        baseline_down_projection: Tensor,
        baseline_post_mlp: Tensor,
    ) -> Tensor:
        if layer_index == 0:
            if self._next_layer not in {0, 24}:
                raise RuntimeError("fusion pass restarted before 24 ordered layers")
            self._begin_pass()
        if layer_index != self._next_layer:
            raise RuntimeError(
                f"fusion runtime expected layer {self._next_layer}, got {layer_index}"
            )
        implementation = self.hook.apply_layer(layer_index, accumulator, residual)
        independent, independent_trace = _independent_down_projection_fusion_layer(
            accumulator,
            residual,
            self.metadata.accumulator_scale32[layer_index],
            self.metadata.residual_scale32[layer_index],
            self.metadata.destination_scale32[layer_index],
        )
        if not torch.equal(implementation, independent):
            raise RuntimeError(
                f"layer {layer_index} exact hook differs from independent integer oracle"
            )
        self._layer_records.append(
            {
                "layer_index": layer_index,
                "input_descended_from_prior_fusion": self.mode == "candidate" and layer_index > 0,
                "accumulator_s32_sha256": sha256_tensor(accumulator),
                "residual_s8_sha256": sha256_tensor(residual),
                "baseline_down_projection_s8_sha256": sha256_tensor(
                    baseline_down_projection
                ),
                "baseline_post_mlp_s8_sha256": sha256_tensor(baseline_post_mlp),
                "isolated_or_composed_fused_s8_sha256": sha256_tensor(implementation),
                "independent_oracle_match": True,
                "independent_trace": independent_trace,
            }
        )
        self._next_layer += 1
        if layer_index == 23:
            hook_summary = self.hook.finish_pass()
            hook_layers = hook_summary["layers"]
            for record, hook_trace in zip(
                self._layer_records, hook_layers, strict=True
            ):
                expected = {
                    key: hook_trace[key]
                    for key in record["independent_trace"]
                }
                if record["independent_trace"] != expected:
                    raise RuntimeError(
                        f"layer {record['layer_index']} hook trace differs from oracle trace"
                    )
            encoded = json.dumps(
                self._layer_records,
                indent=2,
                sort_keys=True,
            ).encode()
            self._completed.append(
                {
                    "mode": self.mode,
                    "layers_executed": 24,
                    "ordered_layer_indices": list(range(24)),
                    "all_lane_independent_oracle_match": True,
                    "baseline_projection_diagnostic_not_used_by_candidate_output": True,
                    "ordered_trace_sha256": hashlib.sha256(encoded).hexdigest(),
                    "positive_saturations": hook_summary["positive_saturations"],
                    "negative_saturations": hook_summary["negative_saturations"],
                    "sticky_numeric_overflow": hook_summary["sticky_numeric_overflow"],
                    "layers": self._layer_records,
                }
            )
        return implementation

    def consume_completed_pass(self) -> dict[str, Any]:
        if not self._completed:
            raise RuntimeError("no completed down-projection fusion pass is available")
        if len(self._completed) != 1:
            raise RuntimeError("unexpected unconsumed down-projection fusion passes")
        return self._completed.pop()


def down_projection_residual_fusion_raw(
    runtime: DownProjectionResidualFusionRuntime,
    layer_index: int,
    accumulator: Tensor,
    residual: Tensor,
    baseline_down_projection: Tensor,
    baseline_post_mlp: Tensor,
) -> Tensor:
    return runtime.apply_layer(
        layer_index,
        accumulator,
        residual,
        baseline_down_projection,
        baseline_post_mlp,
    )


def cross_layer_error_carry_produce_raw(
    runtime: CrossLayerErrorCarryRuntime,
    layer_index: int,
    accumulator: Tensor,
    residual: Tensor,
    baseline_down_projection: Tensor,
    baseline_post_mlp: Tensor,
) -> Tensor:
    return runtime.produce(
        layer_index,
        accumulator,
        residual,
        baseline_down_projection,
        baseline_post_mlp,
    )


class FixedAttention(nn.Module):
    def __init__(
        self,
        source: nn.Module,
        normalized_input_scale: float,
        score_scale_factor: float = 1.0,
        rope_diagnostic_mechanism: str | None = None,
    ) -> None:
        super().__init__()
        self.layer_idx = source.layer_idx
        self.head_dim = source.head_dim
        self.num_key_value_groups = source.num_key_value_groups
        self.q_proj = source.q_proj
        self.k_proj = source.k_proj
        self.v_proj = source.v_proj
        self.o_proj = source.o_proj
        if not all(
            isinstance(module, W4A8Linear)
            for module in (self.q_proj, self.k_proj, self.v_proj, self.o_proj)
        ):
            raise TypeError("fixed attention requires W4A8 projection modules")
        for module in (self.q_proj, self.k_proj, self.v_proj):
            module.bind_hardware_input_scale(normalized_input_scale)
        self.dynamic_rope_head_scale = (
            rope_diagnostic_mechanism == "dynamic_rope_head_scale_v1"
        )
        self.layer0_fixed_q7 = (
            rope_diagnostic_mechanism == "layer0_fixed_q7_rope_score_v1"
            and self.layer_idx == 0
        )
        self.layer0_relative_rope_score_fusion = (
            rope_diagnostic_mechanism == "layer0_relative_rope_score_fusion_v1"
            and self.layer_idx == 0
        )
        self.layer0_absolute_rope_online_attention = (
            rope_diagnostic_mechanism == "layer0_absolute_rope_online_attention_v1"
            and self.layer_idx == 0
        )
        self.layer0_projection_shadow_staged_attention = (
            rope_diagnostic_mechanism == "layer0_projection_shadow_staged_attention_v1"
            and self.layer_idx == 0
        )
        self.layer0_tile_max_delta_attention = (
            rope_diagnostic_mechanism == "layer0_tile_max_delta_attention_v1"
            and self.layer_idx == 0
        )
        self.layer0_tile_bfp_score_attention = (
            rope_diagnostic_mechanism == "layer0_tile_bfp_score_attention_v1"
            and self.layer_idx == 0
        )
        self.shared_native_accumulator_tagged_attention = (
            rope_diagnostic_mechanism
            == "shared_native_accumulator_tagged_attention_v1"
        )
        self.shared_qk_residual_cross_term_baseline = (
            rope_diagnostic_mechanism
            == "shared_qk_residual_cross_term_baseline_v1"
        )
        self.shared_qk_residual_cross_term_attention = (
            rope_diagnostic_mechanism
            == "shared_qk_residual_cross_term_attention_v1"
        )
        self.shared_qk_residual_mode = (
            self.shared_qk_residual_cross_term_baseline
            or self.shared_qk_residual_cross_term_attention
        )
        self.shared_v_residual_value_correction_baseline = (
            rope_diagnostic_mechanism
            in {
                "shared_v_residual_value_correction_baseline_v1",
                "shared_down_projection_residual_fusion_baseline_v1",
                "shared_down_projection_residual_fusion_v1",
                "cross_layer_quantization_error_carry_final_output_v1",
            }
        )
        self.shared_v_residual_value_correction_attention = (
            rope_diagnostic_mechanism
            == "shared_v_residual_value_correction_attention_v1"
        )
        self.shared_v_residual_mode = (
            self.shared_v_residual_value_correction_baseline
            or self.shared_v_residual_value_correction_attention
        )
        self.shared_q20_44_score_mode = (
            self.shared_qk_residual_mode or self.shared_v_residual_mode
        )
        if (
            rope_diagnostic_mechanism
            in {
                "layer0_fixed_q7_rope_score_v1",
                "layer0_relative_rope_score_fusion_v1",
                "layer0_absolute_rope_online_attention_v1",
                "layer0_projection_shadow_staged_attention_v1",
                "layer0_tile_max_delta_attention_v1",
                "layer0_tile_bfp_score_attention_v1",
            }
            and not (
                self.layer0_fixed_q7
                or self.layer0_relative_rope_score_fusion
                or self.layer0_absolute_rope_online_attention
                or self.layer0_projection_shadow_staged_attention
                or self.layer0_tile_max_delta_attention
                or self.layer0_tile_bfp_score_attention
            )
        ):
            raise ValueError("structural RoPE-to-score replacement is legal only for layer 0")
        if rope_diagnostic_mechanism is None or self.shared_q20_44_score_mode:
            if self.q_proj.output_head_scales.numel() != 14:
                raise ValueError("Q projection requires fourteen calibrated head scales")
            if self.k_proj.output_head_scales.numel() != 2:
                raise ValueError("K projection requires two calibrated KV-head scales")
            query_projection_scales = self.q_proj.output_head_scales.detach().clone()
            key_projection_scales = self.k_proj.output_head_scales.detach().clone()
            rope_conversion_q9 = (
                ROPE_UNIT_CONVERSION_Q9
                if self.shared_q20_44_score_mode
                else ROPE_SAFE_CONVERSION_Q9
            )
            rope_output_bits = 8
        else:
            if rope_diagnostic_mechanism not in ROPE_DIAGNOSTIC_MECHANISMS:
                raise ValueError(
                    f"unsupported RoPE diagnostic mechanism: {rope_diagnostic_mechanism}"
                )
            if self.q_proj.output_scale is None or self.k_proj.output_scale is None:
                raise ValueError("structural RoPE diagnostics require scalar Q/K scales")
            if self.q_proj.output_head_scales.numel() or self.k_proj.output_head_scales.numel():
                raise ValueError("structural RoPE diagnostics exclude per-head Q/K scales")
            query_projection_scales = torch.full(
                (14,), self.q_proj.output_scale, dtype=torch.float64
            )
            key_projection_scales = torch.full(
                (2,), self.k_proj.output_scale, dtype=torch.float64
            )
            rope_conversion_q9, rope_output_bits = ROPE_DIAGNOSTIC_MECHANISMS[
                rope_diagnostic_mechanism
            ]
        realized_conversion_scale = (
            128.0
            if self.layer0_fixed_q7
            else (
                1.0
                if (
                    self.layer0_relative_rope_score_fusion
                    or self.layer0_absolute_rope_online_attention
                    or self.layer0_projection_shadow_staged_attention
                    or self.layer0_tile_max_delta_attention
                    or self.layer0_tile_bfp_score_attention
                    or self.shared_native_accumulator_tagged_attention
                    or self.shared_q20_44_score_mode
                )
                else rope_conversion_q9 / float(1 << ROPE_SCALE_FRAC)
            )
        )
        self.rope_diagnostic_mechanism = rope_diagnostic_mechanism
        self.rope_conversion_q9 = rope_conversion_q9
        self.rope_output_bits = rope_output_bits
        self.register_buffer(
            "query_projection_scales",
            query_projection_scales,
            persistent=True,
        )
        self.register_buffer(
            "key_projection_scales",
            key_projection_scales,
            persistent=True,
        )
        self.register_buffer(
            "query_rope_output_scales",
            self.query_projection_scales / realized_conversion_scale,
            persistent=not self.dynamic_rope_head_scale,
        )
        self.register_buffer(
            "key_rope_output_scales",
            self.key_projection_scales / realized_conversion_scale,
            persistent=not self.dynamic_rope_head_scale,
        )
        mapped_key_scales = self.key_rope_output_scales.repeat_interleave(7)
        self.score_scale_factor = diagnostic_factor(
            score_scale_factor,
            "attention score diagnostic scale factor",
        )
        score_multiplier, score_right_shift = derive_multiplier(
            self.query_rope_output_scales
            * mapped_key_scales
            * (1 << ATTENTION_SCORE_FRAC)
            / math.sqrt(self.head_dim)
            * self.score_scale_factor
        )
        self.register_buffer(
            "score_multiplier",
            score_multiplier,
            persistent=True,
        )
        self.register_buffer(
            "score_right_shift",
            score_right_shift,
            persistent=not self.dynamic_rope_head_scale,
        )
        if (
            self.dynamic_rope_head_scale
            or self.layer0_fixed_q7
            or self.layer0_relative_rope_score_fusion
            or self.layer0_absolute_rope_online_attention
            or self.layer0_projection_shadow_staged_attention
            or self.layer0_tile_max_delta_attention
            or self.layer0_tile_bfp_score_attention
        ):
            if (
                self.q_proj.output_scale32_record is None
                or self.k_proj.output_scale32_record is None
            ):
                raise ValueError("dynamic RoPE requires Scale32-bound Q/K producers")
            self.query_producer_scale32 = self.q_proj.output_scale32_record
            self.key_producer_scale32 = self.k_proj.output_scale32_record
        else:
            self.query_producer_scale32 = None
            self.key_producer_scale32 = None
        if self.shared_q20_44_score_mode:
            metadata = qk_residual_layer_metadata(self.layer_idx)
            expected_query_scales = torch.tensor(
                [value / 127.0 for value in metadata["q_proj_output_absmax"]],
                dtype=torch.float64,
            )
            expected_key_scales = torch.tensor(
                [value / 127.0 for value in metadata["k_proj_output_absmax"]],
                dtype=torch.float64,
            )
            if not torch.equal(self.query_projection_scales.cpu(), expected_query_scales):
                raise ValueError("candidate Q projection scales differ from frozen metadata")
            if not torch.equal(self.key_projection_scales.cpu(), expected_key_scales):
                raise ValueError("candidate K projection scales differ from frozen metadata")
            self.register_buffer(
                "query_baseline_scale32_records",
                torch.tensor(metadata["q_proj_baseline_scale32"], dtype=torch.int64),
                persistent=True,
            )
            self.register_buffer(
                "key_baseline_scale32_records",
                torch.tensor(metadata["k_proj_baseline_scale32"], dtype=torch.int64),
                persistent=True,
            )
        if self.shared_qk_residual_mode:
            self.register_buffer(
                "query_residual_scale32_records",
                torch.tensor(metadata["q_proj_residual_scale32"], dtype=torch.int64),
                persistent=True,
            )
            self.register_buffer(
                "key_residual_scale32_records",
                torch.tensor(metadata["k_proj_residual_scale32"], dtype=torch.int64),
                persistent=True,
            )
        if self.shared_v_residual_mode:
            v_metadata = v_residual_layer_metadata(self.layer_idx)
            if self.v_proj.output_scale != v_metadata["baseline_output_scale"]:
                raise ValueError("candidate V projection scale differs from frozen metadata")
            self.register_buffer(
                "v_baseline_scale32_records",
                torch.tensor(v_metadata["baseline_v_scale32"], dtype=torch.int64),
                persistent=True,
            )
            self.register_buffer(
                "v_residual_scale32_records",
                torch.tensor(v_metadata["residual_v_scale32"], dtype=torch.int64),
                persistent=True,
            )
        self.o_proj.bind_hardware_input_scale(self.v_proj.output_scale)
        self.query_rope_output_elements = 0
        self.query_rope_output_saturations = 0
        self.key_rope_output_elements = 0
        self.key_rope_output_saturations = 0
        self.query_dynamic_scale_min = None
        self.query_dynamic_scale_max = None
        self.key_dynamic_scale_min = None
        self.key_dynamic_scale_max = None
        self.query_residual_positive_clamps = 0
        self.query_residual_negative_clamps = 0
        self.key_residual_positive_clamps = 0
        self.key_residual_negative_clamps = 0
        self.v_residual_positive_clamps = 0
        self.v_residual_negative_clamps = 0
        self.baseline_equality_checks = (
            {
                "q_projection_int8": 0,
                "k_projection_int8": 0,
                "v_projection_int8": 0,
                "q_absolute_rope_int8": 0,
                "k_absolute_rope_int8": 0,
                "base_score_q20_44": 0,
                "softmax_probability_q0_15": 0,
            }
            if self.shared_v_residual_mode
            else {
                "q_projection_int8": 0,
                "k_projection_int8": 0,
                "q_absolute_rope_int8": 0,
                "k_absolute_rope_int8": 0,
                "base_score_before_residual_correction": 0,
            }
        )

    def forward(
        self,
        hidden_states: Tensor,
        position_embeddings: tuple[Tensor, Tensor],
        attention_mask: Tensor | None,
        past_key_values: Any = None,
        **_: Any,
    ) -> tuple[Tensor, None]:
        if past_key_values is not None:
            raise RuntimeError("official quality measurement requires cache-free paired scoring")
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        if self.shared_qk_residual_mode:
            (
                query,
                query_residual,
                query_positive_clamps,
                query_negative_clamps,
            ) = self.q_proj.forward_qk_residual_hardware_input(
                hidden_states,
                self.query_baseline_scale32_records,
                self.query_residual_scale32_records,
            )
            (
                key,
                key_residual,
                key_positive_clamps,
                key_negative_clamps,
            ) = self.k_proj.forward_qk_residual_hardware_input(
                hidden_states,
                self.key_baseline_scale32_records,
                self.key_residual_scale32_records,
            )
            authoritative_query = self.q_proj.forward_hardware_input(hidden_states)
            authoritative_key = self.k_proj.forward_hardware_input(hidden_states)
            if not torch.equal(query, authoritative_query):
                raise RuntimeError("candidate Q projection changed the authoritative q8 tensor")
            if not torch.equal(key, authoritative_key):
                raise RuntimeError("candidate K projection changed the authoritative q8 tensor")
            self.baseline_equality_checks["q_projection_int8"] += 1
            self.baseline_equality_checks["k_projection_int8"] += 1
            self.query_residual_positive_clamps += query_positive_clamps
            self.query_residual_negative_clamps += query_negative_clamps
            self.key_residual_positive_clamps += key_positive_clamps
            self.key_residual_negative_clamps += key_negative_clamps
            query = query.view(hidden_shape).transpose(1, 2)
            key = key.view(hidden_shape).transpose(1, 2)
            query_residual = query_residual.view(hidden_shape).transpose(1, 2)
            key_residual = key_residual.view(hidden_shape).transpose(1, 2)
            authoritative_query = authoritative_query.view(hidden_shape).transpose(1, 2)
            authoritative_key = authoritative_key.view(hidden_shape).transpose(1, 2)
        elif self.shared_v_residual_mode:
            query = self.q_proj.forward_hardware_input(hidden_states)
            key = self.k_proj.forward_hardware_input(hidden_states)
            authoritative_query = self.q_proj.forward_hardware_input(hidden_states)
            authoritative_key = self.k_proj.forward_hardware_input(hidden_states)
            if not torch.equal(query, authoritative_query):
                raise RuntimeError("candidate Q projection changed the authoritative q8 tensor")
            if not torch.equal(key, authoritative_key):
                raise RuntimeError("candidate K projection changed the authoritative q8 tensor")
            self.baseline_equality_checks["q_projection_int8"] += 1
            self.baseline_equality_checks["k_projection_int8"] += 1
            query = query.view(hidden_shape).transpose(1, 2)
            key = key.view(hidden_shape).transpose(1, 2)
            authoritative_query = authoritative_query.view(hidden_shape).transpose(1, 2)
            authoritative_key = authoritative_key.view(hidden_shape).transpose(1, 2)
        elif (
            self.layer0_projection_shadow_staged_attention
            or self.layer0_tile_max_delta_attention
            or self.layer0_tile_bfp_score_attention
        ):
            query = self.q_proj.forward_shadow_hardware_input(hidden_states).view(hidden_shape).transpose(1, 2)
            key = self.k_proj.forward_shadow_hardware_input(hidden_states).view(hidden_shape).transpose(1, 2)
        elif self.shared_native_accumulator_tagged_attention:
            query, query_native_scale32 = self.q_proj.forward_native_hardware_input(hidden_states)
            key, key_native_scale32 = self.k_proj.forward_native_hardware_input(hidden_states)
            query = query.view(hidden_shape).transpose(1, 2)
            key = key.view(hidden_shape).transpose(1, 2)
        else:
            query = self.q_proj.forward_hardware_input(hidden_states).view(hidden_shape).transpose(1, 2)
            key = self.k_proj.forward_hardware_input(hidden_states).view(hidden_shape).transpose(1, 2)
        if self.shared_v_residual_mode:
            (
                value,
                value_residual,
                v_positive_clamps,
                v_negative_clamps,
            ) = self.v_proj.forward_v_residual_hardware_input(
                hidden_states,
                self.v_baseline_scale32_records,
                self.v_residual_scale32_records,
            )
            authoritative_value = self.v_proj.forward_hardware_input(hidden_states)
            if not torch.equal(value, authoritative_value):
                raise RuntimeError("candidate V projection changed the authoritative v8 tensor")
            self.baseline_equality_checks["v_projection_int8"] += 1
            self.v_residual_positive_clamps += v_positive_clamps
            self.v_residual_negative_clamps += v_negative_clamps
            value = value.view(hidden_shape).transpose(1, 2)
            value_residual = value_residual.view(hidden_shape).transpose(1, 2)
        else:
            value = self.v_proj.forward_hardware_input(hidden_states).view(hidden_shape).transpose(1, 2)
        cos, sin = position_embeddings
        if not (
            self.layer0_relative_rope_score_fusion
            or self.layer0_absolute_rope_online_attention
            or self.layer0_projection_shadow_staged_attention
            or self.layer0_tile_max_delta_attention
            or self.layer0_tile_bfp_score_attention
            or self.shared_native_accumulator_tagged_attention
            or self.shared_q20_44_score_mode
        ):
            self.query_rope_output_elements += query.numel()
        if self.shared_q20_44_score_mode:
            query, query_saturations = fixed_rope_raw_with_saturation(
                query,
                self.rope_conversion_q9,
                cos,
                sin,
                self.rope_output_bits,
            )
            authoritative_query, _ = fixed_rope_raw_with_saturation(
                authoritative_query,
                self.rope_conversion_q9,
                cos,
                sin,
                self.rope_output_bits,
            )
            if not torch.equal(query, authoritative_query):
                raise RuntimeError("candidate Q RoPE changed the authoritative baseline tensor")
            self.baseline_equality_checks["q_absolute_rope_int8"] += 1
            if self.shared_qk_residual_mode:
                query_residual = qk_residual_rope_raw(query_residual)
        elif (
            self.layer0_relative_rope_score_fusion
            or self.layer0_absolute_rope_online_attention
            or self.layer0_projection_shadow_staged_attention
            or self.layer0_tile_max_delta_attention
            or self.layer0_tile_bfp_score_attention
            or self.shared_native_accumulator_tagged_attention
        ):
            query_saturations = 0
        elif self.layer0_fixed_q7:
            query = fixed_q7_rope_head_raw(query, cos, sin)
            query_saturations = 0
        elif self.dynamic_rope_head_scale:
            query, query_scale32, query_saturations = dynamic_rope_head_raw(
                query,
                self.query_producer_scale32,
                cos,
                sin,
            )
        else:
            query, query_saturations = fixed_rope_raw_with_saturation(
                query,
                self.rope_conversion_q9,
                cos,
                sin,
                self.rope_output_bits,
            )
        self.query_rope_output_saturations += query_saturations
        if not (
            self.layer0_relative_rope_score_fusion
            or self.layer0_absolute_rope_online_attention
            or self.layer0_projection_shadow_staged_attention
            or self.layer0_tile_max_delta_attention
            or self.layer0_tile_bfp_score_attention
            or self.shared_native_accumulator_tagged_attention
            or self.shared_q20_44_score_mode
        ):
            self.key_rope_output_elements += key.numel()
        if self.shared_q20_44_score_mode:
            key, key_saturations = fixed_rope_raw_with_saturation(
                key,
                self.rope_conversion_q9,
                cos,
                sin,
                self.rope_output_bits,
            )
            authoritative_key, _ = fixed_rope_raw_with_saturation(
                authoritative_key,
                self.rope_conversion_q9,
                cos,
                sin,
                self.rope_output_bits,
            )
            if not torch.equal(key, authoritative_key):
                raise RuntimeError("candidate K RoPE changed the authoritative baseline tensor")
            self.baseline_equality_checks["k_absolute_rope_int8"] += 1
            if self.shared_qk_residual_mode:
                key_residual = qk_residual_rope_raw(key_residual)
        elif (
            self.layer0_relative_rope_score_fusion
            or self.layer0_absolute_rope_online_attention
            or self.layer0_projection_shadow_staged_attention
            or self.layer0_tile_max_delta_attention
            or self.layer0_tile_bfp_score_attention
            or self.shared_native_accumulator_tagged_attention
        ):
            key_saturations = 0
        elif self.layer0_fixed_q7:
            key = fixed_q7_rope_head_raw(key, cos, sin)
            key_saturations = 0
        elif self.dynamic_rope_head_scale:
            key, key_scale32, key_saturations = dynamic_rope_head_raw(
                key,
                self.key_producer_scale32,
                cos,
                sin,
            )
            self.query_dynamic_scale_min = int(query_scale32.min())
            self.query_dynamic_scale_max = int(query_scale32.max())
            self.key_dynamic_scale_min = int(key_scale32.min())
            self.key_dynamic_scale_max = int(key_scale32.max())
        else:
            key, key_saturations = fixed_rope_raw_with_saturation(
                key,
                self.rope_conversion_q9,
                cos,
                sin,
                self.rope_output_bits,
            )
        self.key_rope_output_saturations += key_saturations
        value = repeat_kv(value, self.num_key_value_groups)
        if self.shared_v_residual_mode:
            value_residual = repeat_kv(value_residual, self.num_key_value_groups)
        if self.layer0_absolute_rope_online_attention:
            output = absolute_rope_online_attention_raw(
                query,
                key,
                value,
                attention_mask,
                self.query_producer_scale32,
                self.key_producer_scale32,
            )
            output = output.transpose(1, 2).contiguous().reshape(*input_shape, -1)
            projected = self.o_proj.forward_hardware_input(output)
            return projected.to(hidden_states.dtype) * self.o_proj.output_scale, None
        if self.layer0_relative_rope_score_fusion:
            scores = relative_rope_attention_scores_raw(
                query,
                key,
                self.query_producer_scale32,
                self.key_producer_scale32,
                attention_mask,
            )
        elif self.layer0_projection_shadow_staged_attention:
            scores = projection_shadow_staged_attention_scores_raw(
                query,
                key,
                self.query_producer_scale32,
                self.key_producer_scale32,
                attention_mask,
            )
        elif self.layer0_tile_max_delta_attention:
            scores = tile_max_delta_attention_scores_raw(
                query,
                key,
                self.query_producer_scale32,
                self.key_producer_scale32,
                attention_mask,
            )
        elif self.layer0_tile_bfp_score_attention:
            scores = tile_bfp_attention_scores_raw(
                query,
                key,
                self.query_producer_scale32,
                self.key_producer_scale32,
                attention_mask,
            )
        elif self.shared_native_accumulator_tagged_attention:
            scores = native_accumulator_tagged_attention_scores_raw(
                query,
                key,
                query_native_scale32,
                key_native_scale32,
                attention_mask,
            )
        elif self.shared_qk_residual_mode:
            authoritative_base_scores = qk_authoritative_base_scores_q20_44_raw(
                query,
                key,
                self.query_baseline_scale32_records,
                self.key_baseline_scale32_records,
            )
            duplicate_base_scores = qk_authoritative_base_scores_q20_44_raw(
                authoritative_query,
                authoritative_key,
                self.query_baseline_scale32_records,
                self.key_baseline_scale32_records,
            )
            if not torch.equal(authoritative_base_scores, duplicate_base_scores):
                raise RuntimeError("candidate changed the authoritative base-score tensor")
            self.baseline_equality_checks[
                "base_score_before_residual_correction"
            ] += 1
            scores = qk_residual_cross_term_scores_raw(
                authoritative_base_scores,
                query,
                key,
                query_residual,
                key_residual,
                self.query_baseline_scale32_records,
                self.key_baseline_scale32_records,
                self.query_residual_scale32_records,
                self.key_residual_scale32_records,
                attention_mask,
                enable_correction=self.shared_qk_residual_cross_term_attention,
            )
        elif self.shared_v_residual_mode:
            authoritative_base_scores = qk_authoritative_base_scores_q20_44_raw(
                query,
                key,
                self.query_baseline_scale32_records,
                self.key_baseline_scale32_records,
            )
            duplicate_base_scores = qk_authoritative_base_scores_q20_44_raw(
                authoritative_query,
                authoritative_key,
                self.query_baseline_scale32_records,
                self.key_baseline_scale32_records,
            )
            if not torch.equal(authoritative_base_scores, duplicate_base_scores):
                raise RuntimeError("candidate changed the authoritative base-score tensor")
            self.baseline_equality_checks["base_score_q20_44"] += 1
            scores = v_residual_baseline_scores_raw(
                authoritative_base_scores,
                attention_mask,
            )
        elif self.layer0_fixed_q7:
            key = repeat_kv(key, self.num_key_value_groups)
            scores = fixed_q7_attention_scores_raw(
                query,
                key,
                self.query_producer_scale32,
                self.key_producer_scale32,
                attention_mask,
            )
        elif self.dynamic_rope_head_scale:
            key = repeat_kv(key, self.num_key_value_groups)
            key_scale32 = key_scale32.repeat_interleave(
                self.num_key_value_groups,
                dim=1,
            )
            scores = fixed_dynamic_attention_scores_raw(
                query,
                key,
                query_scale32,
                key_scale32,
                attention_mask,
            )
        else:
            key = repeat_kv(key, self.num_key_value_groups)
            scores = fixed_attention_scores_raw(
                query,
                key,
                attention_mask,
                self.score_multiplier.reshape(1, 14, 1, 1),
                self.score_right_shift.reshape(1, 14, 1, 1),
            )
        if self.shared_v_residual_mode:
            probabilities = native_accumulator_tagged_softmax_raw(scores)
            self.baseline_equality_checks["softmax_probability_q0_15"] += 1
            output = v_residual_attention_value_raw(
                probabilities,
                value,
                value_residual,
                self.v_baseline_scale32_records,
                self.v_residual_scale32_records,
                enable_correction=self.shared_v_residual_value_correction_attention,
            )
        else:
            probabilities = (
                native_accumulator_tagged_softmax_raw(scores)
                if (
                    self.shared_native_accumulator_tagged_attention
                    or self.shared_qk_residual_mode
                )
                else tile_bfp_softmax_raw(scores)
                if self.layer0_tile_bfp_score_attention
                else tile_max_delta_softmax_raw(scores)
                if self.layer0_tile_max_delta_attention
                else fixed_softmax_raw(scores)
            )
            output = fixed_attention_value_raw(probabilities, value)
        output = output.transpose(1, 2).contiguous().reshape(*input_shape, -1)
        projected = self.o_proj.forward_hardware_input(output)
        return projected.to(hidden_states.dtype) * self.o_proj.output_scale, None


class FixedMLP(nn.Module):
    def __init__(self, source: nn.Module, normalized_input_scale: float) -> None:
        super().__init__()
        self.gate_proj = source.gate_proj
        self.up_proj = source.up_proj
        self.down_proj = source.down_proj
        if not all(
            isinstance(module, W4A8Linear)
            for module in (self.gate_proj, self.up_proj, self.down_proj)
        ):
            raise TypeError("fixed MLP requires W4A8 projection modules")
        for module in (self.gate_proj, self.up_proj):
            module.bind_hardware_input_scale(normalized_input_scale)
        real_multiplier = torch.tensor(
            [1.0 / ((1 << 21) * self.down_proj.input_scale)],
            dtype=torch.float64,
        )
        multiplier, right_shift = derive_multiplier(real_multiplier)
        self.register_buffer("silu_multiplier", multiplier, persistent=True)
        self.register_buffer("silu_right_shift", right_shift, persistent=True)

    def forward_components(
        self,
        hidden_states: Tensor,
    ) -> tuple[Tensor, Tensor]:
        gate = self.gate_proj.forward_hardware_input(hidden_states)
        up = self.up_proj.forward_hardware_input(hidden_states)
        gated = fixed_silu_gate_raw(
            gate,
            up,
            self.gate_proj.output_scale,
            self.up_proj.output_scale,
            self.silu_multiplier,
            self.silu_right_shift,
        )
        original_shape = gated.shape[:-1]
        accumulator = self.down_proj.accumulator_quantized(gated)
        output = self.down_proj.requantize_accumulator(accumulator, original_shape)
        return accumulator.to(torch.int32).reshape(*original_shape, -1), output

    def forward(self, hidden_states: Tensor) -> Tensor:
        _accumulator, output = self.forward_components(hidden_states)
        return output.to(hidden_states.dtype) * self.down_proj.output_scale


class FixedDecoderLayer(nn.Module):
    def __init__(
        self,
        source: nn.Module,
        input_scale: float,
        post_attention_scale: float,
        post_mlp_scale: float,
        input_norm_output_scale: float,
        post_attention_norm_output_scale: float,
        attention_score_scale_factor: float = 1.0,
        rope_diagnostic_mechanism: str | None = None,
        down_projection_fusion_runtime: DownProjectionResidualFusionRuntime | None = None,
        cross_layer_error_carry_runtime: CrossLayerErrorCarryRuntime | None = None,
    ) -> None:
        super().__init__()
        self.attention_type = source.attention_type
        layer_index = source.self_attn.layer_idx
        self.cross_layer_error_carry_runtime = cross_layer_error_carry_runtime
        self.cross_layer_error_carry_candidate = (
            rope_diagnostic_mechanism
            == "cross_layer_quantization_error_carry_final_output_v1"
        )
        if self.cross_layer_error_carry_candidate and cross_layer_error_carry_runtime is None:
            raise ValueError("cross-layer error-carry mechanism lacks shared runtime")
        if self.cross_layer_error_carry_candidate and layer_index > 0:
            self.input_layernorm = CarryAwareFixedRMSNorm(
                source.input_layernorm,
                input_scale,
                input_norm_output_scale,
                cross_layer_error_carry_runtime,
                layer_index,
            )
        else:
            self.input_layernorm = FixedRMSNorm(
                source.input_layernorm,
                input_scale,
                input_norm_output_scale,
            )
        self.post_attention_layernorm = FixedRMSNorm(
            source.post_attention_layernorm,
            post_attention_scale,
            post_attention_norm_output_scale,
        )
        self.self_attn = FixedAttention(
            source.self_attn,
            self.input_layernorm.output_scale,
            attention_score_scale_factor,
            rope_diagnostic_mechanism,
        )
        self.mlp = FixedMLP(
            source.mlp,
            self.post_attention_layernorm.output_scale,
        )
        self.post_attention_scale = post_attention_scale
        self.post_mlp_scale = post_mlp_scale
        self.down_projection_fusion_runtime = down_projection_fusion_runtime
        self.down_projection_fusion_candidate = (
            rope_diagnostic_mechanism == "shared_down_projection_residual_fusion_v1"
        )
        self.down_projection_fusion_baseline = (
            rope_diagnostic_mechanism
            == "shared_down_projection_residual_fusion_baseline_v1"
        )
        if (
            self.down_projection_fusion_candidate
            or self.down_projection_fusion_baseline
        ) and down_projection_fusion_runtime is None:
            raise ValueError("down-projection fusion mechanism lacks shared runtime")
        if self.down_projection_fusion_candidate or self.down_projection_fusion_baseline:
            frozen_layer = load_down_projection_residual_fusion_metadata()["layers"][
                self.self_attn.layer_idx
            ]
            if self.mlp.down_proj.native_scale32_records.cpu().tolist() != frozen_layer[
                "accumulator_scale32"
            ]:
                raise ValueError("live down-projection accumulator scales differ from frozen metadata")
            if ceil_scale32_from_float(self.post_attention_scale) != frozen_layer[
                "residual_scale32"
            ]:
                raise ValueError("live post-attention residual scale differs from frozen metadata")
            if ceil_scale32_from_float(self.post_mlp_scale) != frozen_layer[
                "destination_scale32"
            ]:
                raise ValueError("live post-MLP destination scale differs from frozen metadata")
            if sha256_tensor(self.mlp.down_proj.qweight) != frozen_layer["qweight_sha256"]:
                raise ValueError("live down-projection qweight differs from frozen metadata")
        if self.cross_layer_error_carry_candidate:
            frozen_layer = load_cross_layer_error_carry_metadata()["layers"][
                self.self_attn.layer_idx
            ]
            if self.mlp.down_proj.native_scale32_records.cpu().tolist() != frozen_layer[
                "accumulator_scale32"
            ]:
                raise ValueError(
                    "live down-projection accumulator scales differ from QECR metadata"
                )
            if ceil_scale32_from_float(self.post_attention_scale) != frozen_layer[
                "residual_scale32"
            ]:
                raise ValueError(
                    "live post-attention residual scale differs from QECR metadata"
                )
            if ceil_scale32_from_float(self.post_mlp_scale) != frozen_layer[
                "destination_scale32"
            ]:
                raise ValueError(
                    "live post-MLP destination scale differs from QECR metadata"
                )
            if sha256_tensor(self.mlp.down_proj.qweight) != frozen_layer["qweight_sha256"]:
                raise ValueError("live down-projection qweight differs from QECR metadata")

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None = None,
        past_key_values: Any = None,
        position_embeddings: tuple[Tensor, Tensor] | None = None,
        **kwargs: Any,
    ) -> Tensor:
        if position_embeddings is None:
            raise ValueError("position embeddings are required")
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = fixed_residual_add(
            residual,
            hidden_states,
            self.post_attention_scale,
        )
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        if not (
            self.down_projection_fusion_candidate
            or self.down_projection_fusion_baseline
            or self.cross_layer_error_carry_candidate
        ):
            hidden_states = self.mlp(hidden_states)
            return fixed_residual_add(residual, hidden_states, self.post_mlp_scale)

        accumulator, baseline_down_projection_s8 = self.mlp.forward_components(
            hidden_states
        )
        baseline_down_projection = (
            baseline_down_projection_s8.to(hidden_states.dtype)
            * self.mlp.down_proj.output_scale
        )
        baseline_post_mlp = fixed_residual_add(
            residual,
            baseline_down_projection,
            self.post_mlp_scale,
        )
        residual_s8 = quantize_int8(residual, self.post_attention_scale)
        baseline_post_mlp_s8 = quantize_int8(baseline_post_mlp, self.post_mlp_scale)
        if self.cross_layer_error_carry_candidate:
            if self.cross_layer_error_carry_runtime is None:
                raise AssertionError("cross-layer error-carry runtime disappeared")
            hidden_s8 = cross_layer_error_carry_produce_raw(
                self.cross_layer_error_carry_runtime,
                self.self_attn.layer_idx,
                accumulator,
                residual_s8,
                baseline_down_projection_s8,
                baseline_post_mlp_s8,
            )
            return hidden_s8.to(hidden_states.dtype) * (
                self.cross_layer_error_carry_runtime.destination_scale(
                    self.self_attn.layer_idx
                )
            )
        if self.down_projection_fusion_runtime is None:
            raise AssertionError("down-projection fusion runtime disappeared")
        fused_s8 = down_projection_residual_fusion_raw(
            self.down_projection_fusion_runtime,
            self.self_attn.layer_idx,
            accumulator,
            residual_s8,
            baseline_down_projection_s8,
            baseline_post_mlp_s8,
        )
        if self.down_projection_fusion_candidate:
            return fused_s8.to(hidden_states.dtype) * self.post_mlp_scale
        return baseline_post_mlp


def selected_texts(spec: dict[str, Any], limit: int | None = None) -> list[str]:
    dataset = load_dataset(
        spec["repository"],
        spec["config"],
        split=spec["split"],
        revision=spec["revision"],
        streaming=True,
    )
    start = spec["indices"]["start"]
    stop = spec["indices"]["stop"]
    if limit is not None:
        stop = min(stop, start + limit)
    texts: list[str] = []
    for index, row in enumerate(dataset):
        if index >= stop:
            break
        if index >= start:
            texts.append(row[spec["field"]])
    if len(texts) != stop - start:
        raise RuntimeError(
            f"{spec['repository']} at {spec['revision']} returned {len(texts)} records, "
            f"expected {stop - start}"
        )
    return texts


def observe_lm_eval_datasets(manifest: dict[str, Any]) -> dict[str, Any]:
    observations: dict[str, Any] = {}
    for task, spec in manifest["lm_eval"]["tasks"].items():
        if task == "piqa":
            sys.path.insert(0, str(LM_EVAL_TASKS))
            try:
                from piqa_dataset import load_piqa

                dataset = load_piqa(revision=spec["revision"])[spec["split"]]
            finally:
                sys.path.remove(str(LM_EVAL_TASKS))
        else:
            dataset = load_dataset(
                spec["repository"],
                spec["config"],
                split=spec["split"],
                revision=spec["revision"],
                streaming=True,
            )
        digest, count = hash_json_records(dataset)
        expected = spec.get("records", {}).get(spec["split"])
        if expected is not None and (
            digest != expected["sha256"] or count != expected["count"]
        ):
            raise RuntimeError(
                f"{task} records differ from the frozen manifest: "
                f"{count}/{digest} != {expected['count']}/{expected['sha256']}"
            )
        observations[task] = {
            "config": spec["config"],
            "record_count": count,
            "record_sha256": digest,
            "repository": spec["repository"],
            "revision": spec["revision"],
            "split": spec["split"],
        }
    return observations


def tokenize_prompts(tokenizer: Any, texts: list[str], token_limit: int) -> list[Tensor]:
    prompts: list[Tensor] = []
    for text in texts:
        encoded = tokenizer(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=token_limit,
            return_tensors="pt",
        )["input_ids"]
        if encoded.shape[1] < 2:
            raise RuntimeError("frozen prompt contains fewer than two scoreable tokens")
        prompts.append(encoded)
    return prompts


def tokenize_wikitext(
    tokenizer: Any,
    texts: list[str],
    token_limit: int,
    join: str,
) -> list[Tensor]:
    input_ids = tokenizer(
        join.join(texts),
        add_special_tokens=False,
        return_tensors="pt",
    )["input_ids"]
    prompts = [
        input_ids[:, start : start + token_limit]
        for start in range(0, input_ids.shape[1], token_limit)
        if input_ids[:, start : start + token_limit].shape[1] >= 2
    ]
    if not prompts:
        raise RuntimeError("WikiText-2 slice produced no scoreable token sequences")
    return prompts


def calibrate(
    model: nn.Module,
    prompts: list[Tensor],
    *,
    activation_scale_percentile: float | None = None,
) -> tuple[dict[str, CalibrationRange], dict[str, ObservedRange]]:
    if activation_scale_percentile is not None and not (
        0.0 < activation_scale_percentile < 1.0
    ):
        raise ValueError("activation scale percentile must be between zero and one")
    ranges = {
        name: CalibrationRange()
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear)
    }
    operator_ranges: dict[str, ObservedRange] = {}
    hooks: list[Any] = []

    def observe_operator(key: str, value: Tensor) -> None:
        current = operator_ranges.setdefault(key, ObservedRange())
        current.absmax = max(current.absmax, float(value.detach().abs().amax()))
        if activation_scale_percentile is not None:
            percentile_absmax = absolute_percentile(
                value,
                activation_scale_percentile,
            )
            current.percentile_absmax = max(
                current.percentile_absmax or 0.0,
                percentile_absmax,
            )

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue

        def observe_linear(
            _module: nn.Module,
            inputs: tuple[Tensor, ...],
            output: Tensor,
            *,
            layer_name: str = name,
        ) -> None:
            current = ranges[layer_name]
            current.input_absmax = max(
                current.input_absmax,
                float(inputs[0].detach().abs().amax()),
            )
            current.output_absmax = max(
                current.output_absmax,
                float(output.detach().abs().amax()),
            )
            if layer_name.endswith(".self_attn.q_proj") or layer_name.endswith(
                ".self_attn.k_proj"
            ):
                head_count = (
                    14
                    if layer_name.endswith(".self_attn.q_proj")
                    else 2
                )
                head_absmax = (
                    output.detach()
                    .reshape(-1, head_count, ROPE_HEAD_DIM)
                    .abs()
                    .amax(dim=(0, 2))
                    .to(torch.float64)
                    .tolist()
                )
                if current.output_head_absmax is None:
                    current.output_head_absmax = [0.0] * head_count
                current.output_head_absmax = [
                    max(previous, observed)
                    for previous, observed in zip(
                        current.output_head_absmax,
                        head_absmax,
                        strict=True,
                    )
                ]
            if activation_scale_percentile is not None:
                input_percentile_absmax = absolute_percentile(
                    inputs[0],
                    activation_scale_percentile,
                )
                output_percentile_absmax = absolute_percentile(
                    output,
                    activation_scale_percentile,
                )
                current.input_percentile_absmax = max(
                    current.input_percentile_absmax or 0.0,
                    input_percentile_absmax,
                )
                current.output_percentile_absmax = max(
                    current.output_percentile_absmax or 0.0,
                    output_percentile_absmax,
                )

        hooks.append(module.register_forward_hook(observe_linear))

    for index, layer in enumerate(model.model.layers):
        input_key = f"model.layers.{index}.input_layernorm.input"
        input_output_key = f"model.layers.{index}.input_layernorm.output"
        post_attention_key = f"model.layers.{index}.post_attention_residual"
        post_attention_output_key = (
            f"model.layers.{index}.post_attention_layernorm.output"
        )
        post_mlp_key = f"model.layers.{index}.post_mlp_residual"

        def observe_input_norm(
            _module: nn.Module,
            inputs: tuple[Tensor, ...],
            *,
            key: str = input_key,
        ) -> None:
            observe_operator(key, inputs[0])

        def observe_post_attention(
            _module: nn.Module,
            inputs: tuple[Tensor, ...],
            *,
            key: str = post_attention_key,
        ) -> None:
            observe_operator(key, inputs[0])

        def observe_input_norm_output(
            _module: nn.Module,
            _inputs: tuple[Tensor, ...],
            output: Tensor,
            *,
            key: str = input_output_key,
        ) -> None:
            observe_operator(key, output)

        def observe_post_attention_norm_output(
            _module: nn.Module,
            _inputs: tuple[Tensor, ...],
            output: Tensor,
            *,
            key: str = post_attention_output_key,
        ) -> None:
            observe_operator(key, output)

        def observe_layer_output(
            _module: nn.Module,
            _inputs: tuple[Tensor, ...],
            output: Tensor,
            *,
            key: str = post_mlp_key,
        ) -> None:
            observe_operator(key, output)

        hooks.append(layer.input_layernorm.register_forward_pre_hook(observe_input_norm))
        hooks.append(
            layer.input_layernorm.register_forward_hook(observe_input_norm_output)
        )
        hooks.append(
            layer.post_attention_layernorm.register_forward_pre_hook(observe_post_attention)
        )
        hooks.append(
            layer.post_attention_layernorm.register_forward_hook(
                observe_post_attention_norm_output
            )
        )
        hooks.append(layer.register_forward_hook(observe_layer_output))

    def observe_final_norm(_module: nn.Module, inputs: tuple[Tensor, ...]) -> None:
        observe_operator("model.norm.input", inputs[0])

    def observe_final_norm_output(
        _module: nn.Module,
        _inputs: tuple[Tensor, ...],
        output: Tensor,
    ) -> None:
        observe_operator("model.norm.output", output)

    hooks.append(model.model.norm.register_forward_pre_hook(observe_final_norm))
    hooks.append(model.model.norm.register_forward_hook(observe_final_norm_output))
    try:
        with torch.inference_mode():
            for input_ids in prompts:
                model(input_ids=input_ids, use_cache=False)
    finally:
        for hook in hooks:
            hook.remove()
    return ranges, operator_ranges


def replace_linears(
    model: nn.Module,
    ranges: dict[str, CalibrationRange],
    *,
    scale_cap: float | None = None,
    use_percentile_scale: bool = False,
    rope_diagnostic_mechanism: str | None = None,
) -> None:
    replacements = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear)
    ]
    for name, module in replacements:
        parent_name, _, child_name = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        use_per_head_output_scales, emit_scale32 = rope_linear_contract(
            name,
            rope_diagnostic_mechanism,
        )
        frozen_output_head_absmax = qk_residual_projection_output_absmax(
            name,
            rope_diagnostic_mechanism,
        )
        setattr(
            parent,
            child_name,
            W4A8Linear(
                module,
                ranges[name],
                output_head_absmax=(
                    frozen_output_head_absmax
                    if frozen_output_head_absmax is not None
                    else ranges[name].output_head_absmax
                    if use_per_head_output_scales
                    else None
                ),
                output_head_size=(
                    ROPE_HEAD_DIM
                    if use_per_head_output_scales
                    else None
                ),
                scale_cap=scale_cap,
                use_percentile_scale=use_percentile_scale,
                scale32_output=emit_scale32,
            ),
        )


def replace_fixed_operators(
    model: nn.Module,
    operator_ranges: dict[str, ObservedRange],
    *,
    scale_cap: float | None = None,
    use_percentile_scale: bool = False,
    diagnostic_attention_score_scale: float | None = None,
    diagnostic_rmsnorm_output_scale: float | None = None,
    rope_diagnostic_mechanism: str | None = None,
) -> None:
    def selected_absmax(name: str) -> float:
        observed = operator_ranges[name]
        selected = (
            observed.percentile_absmax
            if use_percentile_scale
            else observed.absmax
        )
        if selected is None:
            raise ValueError(f"requested percentile operator range is missing: {name}")
        return selected

    fusion_runtime = None
    if rope_diagnostic_mechanism in {
        "shared_down_projection_residual_fusion_baseline_v1",
        "shared_down_projection_residual_fusion_v1",
    }:
        fusion_runtime = DownProjectionResidualFusionRuntime(
            "candidate"
            if rope_diagnostic_mechanism == "shared_down_projection_residual_fusion_v1"
            else "baseline"
        )
    cross_layer_error_carry_runtime = None
    if (
        rope_diagnostic_mechanism
        == "cross_layer_quantization_error_carry_final_output_v1"
    ):
        cross_layer_error_carry_runtime = CrossLayerErrorCarryRuntime(
            cross_layer_error_carry_metadata()
        )
    fixed_layers = []
    for index, layer in enumerate(model.model.layers):
        input_scale = positive_scale(
            selected_absmax(f"model.layers.{index}.input_layernorm.input"),
            cap=scale_cap,
        )
        post_attention_scale = positive_scale(
            selected_absmax(f"model.layers.{index}.post_attention_residual"),
            cap=scale_cap,
        )
        post_mlp_scale = positive_scale(
            selected_absmax(f"model.layers.{index}.post_mlp_residual"),
            cap=scale_cap,
        )
        if fusion_runtime is not None:
            frozen_layer = load_down_projection_residual_fusion_metadata()["layers"][index]
            frozen_down_projection_input_scale = float(
                frozen_layer["hardware_input_scale"]
            )
            layer.mlp.down_proj.input_scale = frozen_down_projection_input_scale
            layer.mlp.down_proj.bind_hardware_input_scale(
                frozen_down_projection_input_scale
            )
            post_attention_scale = float(
                frozen_layer["residual_input_scale_source"]
            )
            post_mlp_scale = float(frozen_layer["destination_scale_source"])
        input_norm_output_scale = rmsnorm_output_scale(
            layer.input_layernorm,
            selected_absmax(f"model.layers.{index}.input_layernorm.output"),
            cap=scale_cap,
        )
        if index == 0 and diagnostic_rmsnorm_output_scale is not None:
            input_norm_output_scale *= diagnostic_factor(
                diagnostic_rmsnorm_output_scale,
                "RMSNorm diagnostic output scale factor",
            )
        post_attention_norm_output_scale = rmsnorm_output_scale(
            layer.post_attention_layernorm,
            selected_absmax(
                f"model.layers.{index}.post_attention_layernorm.output"
            ),
            cap=scale_cap,
        )
        fixed_layers.append(
            FixedDecoderLayer(
                layer,
                input_scale,
                post_attention_scale,
                post_mlp_scale,
                input_norm_output_scale,
                post_attention_norm_output_scale,
                (
                    diagnostic_factor(
                        diagnostic_attention_score_scale,
                        "attention score diagnostic scale factor",
                    )
                    if index == 0 and diagnostic_attention_score_scale is not None
                    else 1.0
                ),
                rope_mechanism_for_layer(index, rope_diagnostic_mechanism),
                fusion_runtime,
                cross_layer_error_carry_runtime,
            )
        )
    model.model.layers = nn.ModuleList(fixed_layers)
    model.ace2_down_projection_fusion_runtime = fusion_runtime
    model.ace2_cross_layer_error_carry_runtime = cross_layer_error_carry_runtime
    final_norm_source = model.model.norm
    final_norm_input_scale = positive_scale(
        selected_absmax("model.norm.input"),
        cap=scale_cap,
    )
    final_norm_output_scale = rmsnorm_output_scale(
        final_norm_source,
        selected_absmax("model.norm.output"),
        cap=scale_cap,
    )
    if cross_layer_error_carry_runtime is None:
        model.model.norm = FixedRMSNorm(
            final_norm_source,
            final_norm_input_scale,
            final_norm_output_scale,
        )
    else:
        model.model.norm = CarryAwareFixedRMSNorm(
            final_norm_source,
            final_norm_input_scale,
            final_norm_output_scale,
            cross_layer_error_carry_runtime,
            24,
        )
    if not isinstance(model.lm_head, W4A8Linear):
        raise TypeError("fixed final RMSNorm requires a W4A8 lm_head")
    model.lm_head.bind_hardware_input_scale(model.model.norm.output_scale)
    model.lm_head.input_is_quantized = True


def perplexity(model: nn.Module, prompts: list[Tensor]) -> dict[str, Any]:
    negative_log_likelihood = 0.0
    scored_tokens = 0
    sequences = []
    with torch.inference_mode():
        for index, input_ids in enumerate(prompts):
            logits = model(input_ids=input_ids, use_cache=False).logits[:, :-1, :].to(
                torch.float32
            )
            labels = input_ids[:, 1:]
            nll = float(
                nn.functional.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    labels.reshape(-1),
                    reduction="sum",
                )
            )
            tokens = labels.numel()
            negative_log_likelihood += nll
            scored_tokens += tokens
            token_hash, _, _ = hash_token_sequences([input_ids])
            sequences.append(
                {
                    "index": index,
                    "input_ids_sha256": token_hash,
                    "negative_log_likelihood": nll,
                    "scored_tokens": tokens,
                }
            )
    mean_nll = negative_log_likelihood / scored_tokens
    return {
        "mean_negative_log_likelihood": mean_nll,
        "negative_log_likelihood": negative_log_likelihood,
        "perplexity": math.exp(mean_nll),
        "scored_tokens": scored_tokens,
        "sequences": sequences,
    }


def task_manager_for_manifest(manifest: dict[str, Any]) -> Any:
    from lm_eval.tasks import TaskManager

    manager = TaskManager(include_path=str(LM_EVAL_TASKS), include_defaults=False)
    expected = set(manifest["lm_eval"]["tasks"])
    if set(manager.all_subtasks) != expected:
        raise RuntimeError(
            f"project-local lm-eval tasks differ from manifest: {manager.all_subtasks}"
        )
    for task in expected:
        path = Path(manager.task_index[task]["yaml_path"]).resolve()
        if path.parent != LM_EVAL_TASKS.resolve():
            raise RuntimeError(f"{task} did not resolve to the project-local frozen task")
    return manager


def run_lm_eval(
    model: nn.Module,
    tokenizer: Any,
    manifest: dict[str, Any],
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        import lm_eval
        from lm_eval.models.huggingface import HFLM
    except ImportError as exc:
        raise RuntimeError("official mode requires lm_eval 0.4.9.2") from exc
    if getattr(lm_eval, "__version__", None) != config["software"]["lm_eval"]:
        raise RuntimeError(
            f"lm_eval version must be {config['software']['lm_eval']}, "
            f"got {getattr(lm_eval, '__version__', None)}"
        )
    tasks = list(manifest["lm_eval"]["tasks"])
    manager = task_manager_for_manifest(manifest)
    loaded_tasks = manager.load_task_or_group(tasks)
    # lm-eval 0.4.9.2's informational pretty-printer assumes every YAML path is
    # under its installed task tree. Evaluation uses the already-loaded frozen
    # task objects; these synthetic paths affect only that log label.
    installed_task_root = Path(sys.modules["lm_eval.tasks"].__file__).parent
    for task in tasks:
        manager.task_index[task]["yaml_path"] = str(
            installed_task_root / "ace2_frozen_project" / f"{task}.yaml"
        )
    model.config.use_cache = False
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.use_cache = False
    wrapped = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        batch_size=config["evaluation"]["lm_eval"]["batch_size"],
    )
    seeds = config["determinism"]
    raw = lm_eval.simple_evaluate(
        model=wrapped,
        tasks=[loaded_tasks[task] for task in tasks],
        num_fewshot=manifest["lm_eval"]["fewshot"],
        batch_size=config["evaluation"]["lm_eval"]["batch_size"],
        limit=manifest["lm_eval"]["limit"],
        bootstrap_iters=config["evaluation"]["lm_eval"]["bootstrap_iters"],
        log_samples=config["evaluation"]["lm_eval"]["log_samples"],
        task_manager=manager,
        random_seed=seeds["python_seed"],
        numpy_random_seed=seeds["numpy_seed"],
        torch_random_seed=seeds["torch_seed"],
        fewshot_random_seed=seeds["datasets_seed"],
    )
    for task, spec in manifest["lm_eval"]["tasks"].items():
        task_config = raw["configs"][task]
        observed = {
            "config": task_config["dataset_name"],
            "repository": task_config["dataset_path"],
            "revision": task_config["dataset_kwargs"]["revision"],
            "split": task_config["validation_split"],
        }
        expected = {
            "config": spec["config"],
            "repository": spec["repository"],
            "revision": spec["revision"],
            "split": spec["split"],
        }
        if observed != expected:
            raise RuntimeError(
                f"lm-eval did not consume the frozen {task} dataset identity: {observed}"
            )
    metric_name = config["evaluation"]["lm_eval"]["metric"]
    metrics = {task: raw["results"][task][metric_name] for task in tasks}
    summary = {
        "average_normalized_accuracy": sum(metrics.values()) / len(metrics),
        "metric": metric_name,
        "tasks": metrics,
    }
    return summary, raw


def source_artifacts(rtl_binding: dict[str, Any]) -> list[dict[str, Any]]:
    paths = [
        PROMPT_MANIFEST,
        QUALITY_CONFIG,
        QUALITY_REQUIREMENTS,
        ROOT / rtl_binding["binding"]["path"],
        ROOT / rtl_binding["manifest"]["path"],
        Path(__file__),
        ROOT / "tools" / "ace2_quality_contracts.py",
        ROOT / "tools" / "ace2_projection_reference.py",
        ROOT / "tools" / "ace2_rmsnorm_reference.py",
        ROOT / "tools" / "ace2_rope_reference.py",
        ROOT / "tools" / "ace2_attention_score_reference.py",
        ROOT / "tools" / "ace2_softmax_reference.py",
        ROOT / "tools" / "ace2_attention_value_reference.py",
        ROOT / "tools" / "ace2_attention_compose_reference.py",
        ROOT / "tools" / "ace2_silu_gate_reference.py",
        ROOT / "tools" / "ace2_residual_reference.py",
        ROOT / "tools" / "ace2_mlp_residual_reference.py",
        ROOT / "tools" / "run_official_quality.py",
        ROOT / "tools" / "run_quality_gate.py",
        *sorted(LM_EVAL_TASKS.iterdir()),
        *(ROOT / item["path"] for item in rtl_binding["numerical_rtl"]),
    ]
    return [
        {
            "bytes": path.stat().st_size,
            "path": path.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(path),
        }
        for path in paths
        if path.is_file()
    ]


def derived_scale_table(
    ranges: dict[str, CalibrationRange],
    operator_ranges: dict[str, ObservedRange],
    model: nn.Module,
) -> dict[str, Any]:
    linears: dict[str, Any] = {}
    for name, module in model.named_modules():
        if not isinstance(module, W4A8Linear):
            continue
        linears[name] = {
            "bias_accumulator": (
                module.bias_accumulator.tolist()
                if module.bias_accumulator is not None
                else None
            ),
            "hardware_input_scale": module.hardware_input_scale,
            "input_absmax": ranges[name].input_absmax,
            "input_scale": module.input_scale,
            "multiplier": module.multiplier.tolist(),
            "output_absmax": ranges[name].output_absmax,
            "output_scale": module.output_scale,
            "output_head_scales": module.output_head_scales.tolist(),
            "qweight_sha256": sha256_tensor(module.qweight),
            "right_shift": module.right_shift.tolist(),
            "weight_scale": module.weight_scale.tolist(),
        }
    attention: dict[str, Any] = {}
    for index, layer in enumerate(model.model.layers):
        module = layer.self_attn
        if not isinstance(module, FixedAttention):
            raise TypeError("derived attention scales require fixed attention modules")
        query_saturation_fraction = (
            module.query_rope_output_saturations / module.query_rope_output_elements
            if module.query_rope_output_elements
            else 0.0
        )
        key_saturation_fraction = (
            module.key_rope_output_saturations / module.key_rope_output_elements
            if module.key_rope_output_elements
            else 0.0
        )
        attention[f"model.layers.{index}.self_attn"] = {
            "attention_value_output_scale": module.v_proj.output_scale,
            "key_projection_output_scales":
                module.key_projection_scales.tolist(),
            "key_rope_output_scales": module.key_rope_output_scales.tolist(),
            "o_projection_calibrated_input_scale": module.o_proj.input_scale,
            "o_projection_hardware_input_scale": module.o_proj.hardware_input_scale,
            "query_key_rope_output_scale_products": (
                module.query_rope_output_scales
                * module.key_rope_output_scales.repeat_interleave(7)
            ).tolist(),
            "query_projection_output_scales":
                module.query_projection_scales.tolist(),
            "query_rope_output_saturation": {
                "elements": module.query_rope_output_elements,
                "fraction": query_saturation_fraction,
                "saturated_elements": module.query_rope_output_saturations,
            },
            "query_rope_output_scales":
                module.query_rope_output_scales.tolist(),
            "rope_conversion_q9": module.rope_conversion_q9,
            "rope_conversion_scale": (
                module.rope_conversion_q9 / float(1 << ROPE_SCALE_FRAC)
            ),
            "rope_diagnostic_mechanism": module.rope_diagnostic_mechanism,
            "dynamic_rope_head_scale": {
                "contract_id": "dynamic_rope_head_scale_v1",
                "enabled": module.dynamic_rope_head_scale,
                "query_producer_scale32": module.query_producer_scale32,
                "key_producer_scale32": module.key_producer_scale32,
                "query_observed_record_min": module.query_dynamic_scale_min,
                "query_observed_record_max": module.query_dynamic_scale_max,
                "key_observed_record_min": module.key_dynamic_scale_min,
                "key_observed_record_max": module.key_dynamic_scale_max,
            },
            "relative_rope_score_fusion": {
                "contract_id": "layer0_relative_rope_score_fusion_v1",
                "enabled": module.layer0_relative_rope_score_fusion,
                "materializes_post_rope_qk": False,
                "query_producer_scale32": module.query_producer_scale32,
                "key_producer_scale32": module.key_producer_scale32,
                "centering": "one_complete_valid_row_maximum_before_int16_clamp",
            },
            "absolute_rope_online_attention": {
                "contract_id": "layer0_absolute_rope_online_attention_v1",
                "enabled": module.layer0_absolute_rope_online_attention,
                "materializes_post_rope_qk": False,
                "materializes_score_or_probability_matrix": False,
                "query_producer_scale32": module.query_producer_scale32,
                "key_producer_scale32": module.key_producer_scale32,
                "logit_format": "signed_Q12_20",
                "online_weight_format": "unsigned_Q1_31",
            },
            "rope_output_bits": module.rope_output_bits,
            "rope_range_safe": module.rope_conversion_q9 == ROPE_SAFE_CONVERSION_Q9,
            "rope_scale_fraction_bits": ROPE_SCALE_FRAC,
            "score_fraction_bits": ATTENTION_SCORE_FRAC,
            "score_head_scale_shift": ATTENTION_HEAD_SCALE_SHIFT,
            "score_multiplier": module.score_multiplier.tolist(),
            "score_scale_factor": module.score_scale_factor,
            "score_real_multiplier": (
                module.query_rope_output_scales
                * module.key_rope_output_scales.repeat_interleave(7)
                * (1 << ATTENTION_SCORE_FRAC)
                / math.sqrt(module.head_dim)
                * module.score_scale_factor
            ).tolist(),
            "score_realized_multiplier": (
                module.score_multiplier.to(torch.float64)
                / torch.pow(
                    torch.tensor(2.0, dtype=torch.float64),
                    module.score_right_shift.to(torch.float64),
                )
            ).tolist(),
            "score_right_shift": module.score_right_shift.tolist(),
            "key_rope_output_saturation": {
                "elements": module.key_rope_output_elements,
                "fraction": key_saturation_fraction,
                "saturated_elements": module.key_rope_output_saturations,
            },
        }
    return {
        "attention": attention,
        "linears": linears,
        "operators": {
            name: {
                "absmax": observed.absmax,
                "scale": positive_scale(observed.absmax),
            }
            for name, observed in sorted(operator_ranges.items())
        },
        "schema_version": 2,
    }


def lm_eval_json_default(value: Any) -> Any:
    from lm_eval.utils import handle_non_serializable

    return handle_non_serializable(value)


def run(
    mode: str,
    output_dir: Path,
    *,
    activation_scale_cap: float | None = None,
    activation_scale_percentile: float | None = None,
    diagnostic_attention_score_scale: float | None = None,
    diagnostic_rmsnorm_output_scale: float | None = None,
    diagnostic_rope_mechanism: str | None = None,
    diagnostic_rope_angle_degrees: float | None = None,
    candidate_evidence_path: Path | None = None,
    smoke_token_limit: int = 32,
) -> dict[str, Any]:
    total_started = time.perf_counter()
    active_rope_mechanism = (
        diagnostic_rope_mechanism or ACTIVE_ROPE_MECHANISM
    )
    active_rope_angle_degrees = (
        diagnostic_rope_angle_degrees
        if diagnostic_rope_mechanism == "scalar_pair_rotate_safe_int8"
        else (
            45.0
            if diagnostic_rope_mechanism == "scalar_pair_rotate45_safe_int8"
            else (
                ACTIVE_ROPE_PAIR_ROTATION_DEGREES
                if diagnostic_rope_mechanism is None
                else None
            )
        )
    )
    manifest, config, rtl_binding = load_contracts(
        require_rtl_binding=diagnostic_rope_mechanism is None,
        candidate_evidence_path=candidate_evidence_path
    )
    if diagnostic_rope_mechanism is not None and "binding" not in rtl_binding:
        historical_binding_path = ROOT / "benchmark" / "quality" / "RTL_BINDING.json"
        rtl_binding = {
            **rtl_binding,
            "binding": {
                "bytes": historical_binding_path.stat().st_size,
                "path": historical_binding_path.relative_to(ROOT).as_posix(),
                "sha256": sha256_file(historical_binding_path),
            },
            "candidate_id": None,
            "source_hash_list": None,
            "valid": False,
        }
    versions = validate_runtime(config)
    seed_everything(config)
    output_dir.mkdir(parents=True, exist_ok=False)
    smoke = mode == "smoke"
    if smoke and not 2 <= smoke_token_limit <= 512:
        raise ValueError("smoke token limit must be in the range 2..512")
    diagnostic_lanes = [
        diagnostic_attention_score_scale is not None,
        diagnostic_rmsnorm_output_scale is not None,
        diagnostic_rope_mechanism is not None,
    ]
    if sum(diagnostic_lanes) > 1:
        raise ValueError("choose one numerical repair diagnostic lane at a time")
    if diagnostic_attention_score_scale is not None:
        diagnostic_factor(
            diagnostic_attention_score_scale,
            "attention score diagnostic scale factor",
        )
    if diagnostic_rmsnorm_output_scale is not None:
        diagnostic_factor(
            diagnostic_rmsnorm_output_scale,
            "RMSNorm diagnostic output scale factor",
        )
    if (
        diagnostic_rope_mechanism is not None
        and diagnostic_rope_mechanism not in ROPE_DIAGNOSTIC_MECHANISMS
    ):
        raise ValueError(
            f"unsupported RoPE diagnostic mechanism: {diagnostic_rope_mechanism}"
        )
    if diagnostic_rope_mechanism == "scalar_pair_rotate_safe_int8":
        if diagnostic_rope_angle_degrees is None:
            raise ValueError("pair-rotation diagnostic requires an angle")
    elif diagnostic_rope_angle_degrees is not None:
        raise ValueError("RoPE angle is valid only for scalar_pair_rotate_safe_int8")

    calibration_text = selected_texts(
        manifest["datasets"]["c4_calibration"],
        limit=1 if smoke else None,
    )
    wiki_text = selected_texts(
        manifest["datasets"]["wikitext2"],
        limit=16 if smoke else None,
    )
    c4_text = selected_texts(
        manifest["datasets"]["c4_en_512"],
        limit=1 if smoke else None,
    )
    observations: dict[str, Any] = {}
    for name, records in {
        "c4_calibration": calibration_text,
        "wikitext2": wiki_text,
        "c4_en_512": c4_text,
    }.items():
        digest, count = hash_records(records)
        observations[name] = {
            "record_count": count,
            "record_sha256": digest,
            **{
                key: manifest["datasets"][name][key]
                for key in ("config", "repository", "revision", "split")
            },
        }
    if not smoke:
        observations["lm_eval"] = observe_lm_eval_datasets(manifest)

    model_spec = manifest["model"]
    tokenizer = AutoTokenizer.from_pretrained(
        model_spec["repository"],
        revision=model_spec["revision"],
    )
    calibration_prompts = tokenize_prompts(
        tokenizer,
        calibration_text,
        manifest["datasets"]["c4_calibration"]["token_limit"],
    )
    wiki_prompts = tokenize_wikitext(
        tokenizer,
        wiki_text,
        manifest["datasets"]["wikitext2"]["token_limit"],
        manifest["datasets"]["wikitext2"]["join"],
    )
    c4_prompts = tokenize_prompts(
        tokenizer,
        c4_text,
        manifest["datasets"]["c4_en_512"]["token_limit"],
    )
    if smoke:
        calibration_prompts = [calibration_prompts[0][:, :smoke_token_limit]]
        wiki_prompts = [wiki_prompts[0][:, :smoke_token_limit]]
        c4_prompts = [c4_prompts[0][:, :smoke_token_limit]]
    for name, prompts in {
        "c4_calibration": calibration_prompts,
        "wikitext2": wiki_prompts,
        "c4_en_512": c4_prompts,
    }.items():
        digest, sequence_count, token_count = hash_token_sequences(prompts)
        observations[name]["tokenized"] = {
            "sequence_count": sequence_count,
            "token_count": token_count,
            "token_sequence_sha256": digest,
        }
    write_json(output_dir / "input_observations.json", observations)

    tokenizer_revision = tokenizer.init_kwargs.get("_commit_hash")
    if tokenizer_revision is not None and tokenizer_revision != model_spec["revision"]:
        raise RuntimeError(
            f"tokenizer resolved to {tokenizer_revision}, expected {model_spec['revision']}"
        )
    run_contract = {
        "schema_version": 1,
        "status": "frozen_before_measurement",
        "created_at_utc": utc_now(),
        "mode": mode,
        "command": [sys.executable, *sys.argv],
        "candidate": {
            "candidate_id": rtl_binding.get("candidate_id"),
            "ordered_source_hash_list_sha256": rtl_binding["candidate_rtl_hash"],
            "binding": rtl_binding["binding"],
            "source_hash_list": rtl_binding.get("source_hash_list"),
        },
        "model": {
            **model_spec,
            "tokenizer_requested_revision": model_spec["revision"],
            "tokenizer_resolved_revision": tokenizer_revision,
        },
        "public_input_slice": {
            "record_limits": {
                "c4_calibration": 1 if smoke else None,
                "c4_en_512": 1 if smoke else None,
                "wikitext2": 16 if smoke else None,
            },
            "token_limit": smoke_token_limit if smoke else None,
            "observations": observations,
        },
        "numerical_contract": {
            "activation_quantization": config["activation_quantization"],
            "arithmetic": config["arithmetic"],
            "full_model_scope": config["full_model_scope"],
            "weight_quantization": config["weight_quantization"],
        },
        "active_qk_basis": {
            "mechanism": active_rope_mechanism,
            "pair_rotation_angle_degrees": active_rope_angle_degrees,
            "scope": (
                "layer0_only_other_layers_unchanged"
                if active_rope_mechanism
                in {
                    "layer0_fixed_q7_rope_score_v1",
                    "layer0_relative_rope_score_fusion_v1",
                    "layer0_absolute_rope_online_attention_v1",
                    "layer0_projection_shadow_staged_attention_v1",
                    "layer0_tile_max_delta_attention_v1",
                    "layer0_tile_bfp_score_attention_v1",
                }
                else "all_layers"
            ),
            "runtime_arithmetic_added": (
                active_rope_mechanism in {
                    "dynamic_rope_head_scale_v1",
                    "layer0_fixed_q7_rope_score_v1",
                    "layer0_relative_rope_score_fusion_v1",
                    "layer0_absolute_rope_online_attention_v1",
                    "layer0_projection_shadow_staged_attention_v1",
                    "layer0_tile_max_delta_attention_v1",
                    "layer0_tile_bfp_score_attention_v1",
                }
            ),
        },
        "diagnostic_numerical_repair": (
            {
                "lane": "A_layer0_attention_score_requantization",
                "layer": 0,
                "score_scale_factor": diagnostic_attention_score_scale,
            }
            if diagnostic_attention_score_scale is not None
            else (
                {
                    "lane": "B_layer0_input_rmsnorm_requantization",
                    "layer": 0,
                    "output_scale_factor": diagnostic_rmsnorm_output_scale,
                }
                if diagnostic_rmsnorm_output_scale is not None
                else (
                    {
                        "lane": (
                            "layer0_fixed_q7_rope_score"
                            if diagnostic_rope_mechanism == "layer0_fixed_q7_rope_score_v1"
                            else (
                                "layer0_relative_rope_score_fusion"
                                if diagnostic_rope_mechanism
                                == "layer0_relative_rope_score_fusion_v1"
                                else (
                                    "layer0_absolute_rope_online_attention"
                                    if diagnostic_rope_mechanism
                                    == "layer0_absolute_rope_online_attention_v1"
                                    else (
                                        "layer0_projection_shadow_staged_attention"
                                        if diagnostic_rope_mechanism
                                        == "layer0_projection_shadow_staged_attention_v1"
                                        else (
                                            "layer0_tile_max_delta_attention"
                                        if diagnostic_rope_mechanism
                                        == "layer0_tile_max_delta_attention_v1"
                                            else (
                                                "layer0_tile_bfp_score_attention"
                                                if diagnostic_rope_mechanism
                                                == "layer0_tile_bfp_score_attention_v1"
                                                else "C_scalar_unit_gain_rope"
                                            )
                                        )
                                    )
                                )
                            )
                        ),
                        "mechanism": diagnostic_rope_mechanism,
                        "qk_scale_granularity": (
                            "layer0_static_per_tensor_other_layers_static_per_head"
                            if diagnostic_rope_mechanism
                            in {
                                "layer0_fixed_q7_rope_score_v1",
                                "layer0_relative_rope_score_fusion_v1",
                                "layer0_absolute_rope_online_attention_v1",
                                "layer0_projection_shadow_staged_attention_v1",
                                "layer0_tile_max_delta_attention_v1",
                                "layer0_tile_bfp_score_attention_v1",
                            }
                            else "static_per_tensor"
                        ),
                        "rope_conversion_q9": ROPE_DIAGNOSTIC_MECHANISMS[
                            diagnostic_rope_mechanism
                        ][0],
                        "rope_output_bits": ROPE_DIAGNOSTIC_MECHANISMS[
                            diagnostic_rope_mechanism
                        ][1],
                        "pair_rotation_angle_degrees": diagnostic_rope_angle_degrees,
                    }
                    if diagnostic_rope_mechanism is not None
                    else None
                )
            )
        ),
        "seeds": config["determinism"],
        "environment": {
            "device": "cpu",
            "executable": sys.executable,
            "packages": versions,
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch_runtime": torch.__version__,
        },
        "source_contracts": {
            "prompt_manifest_sha256": sha256_file(PROMPT_MANIFEST),
            "quality_config_sha256": sha256_file(QUALITY_CONFIG),
            "runner_sha256": sha256_file(Path(__file__)),
        },
    }
    run_contract_path = output_dir / "run_contract.json"
    write_json(run_contract_path, run_contract)

    model_load_started = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        model_spec["repository"],
        revision=model_spec["revision"],
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    ).eval()
    resolved_revision = getattr(model.config, "_commit_hash", None)
    if resolved_revision != model_spec["revision"]:
        raise RuntimeError(
            f"model resolved to {resolved_revision}, expected {model_spec['revision']}"
        )
    model_load_seconds = time.perf_counter() - model_load_started
    calibration_started = time.perf_counter()
    ranges, operator_ranges = calibrate(
        model,
        calibration_prompts,
        activation_scale_percentile=activation_scale_percentile,
    )
    calibration_seconds = time.perf_counter() - calibration_started

    bf16_started = time.perf_counter()
    baseline_perplexity = {
        "wikitext2": perplexity(model, wiki_prompts),
        "c4_en_512": perplexity(model, c4_prompts),
    }
    bf16_seconds = time.perf_counter() - bf16_started
    baseline_perplexity["runtime_seconds"] = bf16_seconds
    write_json(output_dir / "perplexity_bf16_raw.json", baseline_perplexity)
    baseline_lm_eval = None
    if not smoke:
        seed_everything(config)
        baseline_lm_eval, baseline_raw = run_lm_eval(model, tokenizer, manifest, config)
        write_json(
            output_dir / "lm_eval_bf16_raw.json",
            baseline_raw,
            default=lm_eval_json_default,
        )

    if active_rope_mechanism in {
        "scalar_pair_rotate45_safe_int8",
        "scalar_pair_rotate_safe_int8",
    }:
        candidate_calibration_started = time.perf_counter()
        apply_rope_commuting_pair_rotation(
            model,
            float(active_rope_angle_degrees),
        )
        seed_everything(config)
        ranges, operator_ranges = calibrate(
            model,
            calibration_prompts,
            activation_scale_percentile=activation_scale_percentile,
        )
        calibration_seconds += time.perf_counter() - candidate_calibration_started

    quantization_started = time.perf_counter()
    replace_linears(
        model,
        ranges,
        scale_cap=activation_scale_cap,
        use_percentile_scale=activation_scale_percentile is not None,
        rope_diagnostic_mechanism=active_rope_mechanism,
    )
    replace_fixed_operators(
        model,
        operator_ranges,
        scale_cap=activation_scale_cap,
        use_percentile_scale=activation_scale_percentile is not None,
        diagnostic_attention_score_scale=diagnostic_attention_score_scale,
        diagnostic_rmsnorm_output_scale=diagnostic_rmsnorm_output_scale,
        rope_diagnostic_mechanism=active_rope_mechanism,
    )
    quantization_seconds = time.perf_counter() - quantization_started
    scales_path = output_dir / "derived_scales.json"

    w4a8_started = time.perf_counter()
    fixed_perplexity = {
        "wikitext2": perplexity(model, wiki_prompts),
        "c4_en_512": perplexity(model, c4_prompts),
    }
    w4a8_seconds = time.perf_counter() - w4a8_started
    fixed_perplexity["runtime_seconds"] = w4a8_seconds
    write_json(output_dir / "perplexity_w4a8_raw.json", fixed_perplexity)
    fixed_lm_eval = None
    if not smoke:
        seed_everything(config)
        fixed_lm_eval, fixed_raw = run_lm_eval(model, tokenizer, manifest, config)
        write_json(
            output_dir / "lm_eval_w4a8_raw.json",
            fixed_raw,
            default=lm_eval_json_default,
        )
    scale_table = derived_scale_table(ranges, operator_ranges, model)
    write_json(scales_path, scale_table)
    attention_diagnostics = list(scale_table["attention"].values())
    saturation_diagnostics = {
        "rope_output": {
            "key": {
                "elements": sum(
                    item["key_rope_output_saturation"]["elements"]
                    for item in attention_diagnostics
                ),
                "saturated_elements": sum(
                    item["key_rope_output_saturation"]["saturated_elements"]
                    for item in attention_diagnostics
                ),
            },
            "query": {
                "elements": sum(
                    item["query_rope_output_saturation"]["elements"]
                    for item in attention_diagnostics
                ),
                "saturated_elements": sum(
                    item["query_rope_output_saturation"]["saturated_elements"]
                    for item in attention_diagnostics
                ),
            },
        }
    }
    for value in saturation_diagnostics["rope_output"].values():
        value["fraction"] = (
            value["saturated_elements"] / value["elements"]
            if value["elements"]
            else 0.0
        )

    thresholds = config["acceptance_thresholds"]
    wiki_ratio = (
        fixed_perplexity["wikitext2"]["perplexity"]
        / baseline_perplexity["wikitext2"]["perplexity"]
    )
    c4_ratio = (
        fixed_perplexity["c4_en_512"]["perplexity"]
        / baseline_perplexity["c4_en_512"]["perplexity"]
    )
    metrics: dict[str, Any] = {
        "wikitext2": {
            "bf16": baseline_perplexity["wikitext2"],
            "ratio": wiki_ratio,
            "w4a8": fixed_perplexity["wikitext2"],
        },
        "c4_en_512": {
            "bf16": baseline_perplexity["c4_en_512"],
            "ratio": c4_ratio,
            "w4a8": fixed_perplexity["c4_en_512"],
        },
    }
    checks = {
        "wikitext2_perplexity_ratio": (
            wiki_ratio <= thresholds["wikitext2_perplexity_ratio_max"]
        ),
        "c4_en_512_perplexity_ratio": (
            c4_ratio <= thresholds["c4_en_512_perplexity_ratio_max"]
        ),
    }
    if not smoke:
        if baseline_lm_eval is None or fixed_lm_eval is None:
            raise AssertionError("official lm-eval summaries are missing")
        drop = 100.0 * (
            baseline_lm_eval["average_normalized_accuracy"]
            - fixed_lm_eval["average_normalized_accuracy"]
        )
        metrics["lm_eval"] = {
            "bf16": baseline_lm_eval,
            "drop_percentage_points": drop,
            "w4a8": fixed_lm_eval,
        }
        checks["lm_eval_average_normalized_accuracy_drop"] = (
            drop
            <= thresholds[
                "lm_eval_average_normalized_accuracy_drop_percentage_points_max"
            ]
        )

    result = {
        "schema_version": 2,
        "generated_at_utc": utc_now(),
        "classification": (
            (
                "diagnostic_scale_contract_smoke_not_acceptance_evidence"
                if (
                    activation_scale_cap is not None
                    or activation_scale_percentile is not None
                    or diagnostic_attention_score_scale is not None
                    or diagnostic_rmsnorm_output_scale is not None
                    or diagnostic_rope_mechanism is not None
                )
                else "smoke_not_acceptance_evidence"
            )
            if smoke
            else ("supported" if all(checks.values()) else "negative")
        ),
        "mode": mode,
        "gate_passed": all(checks.values()) if not smoke else False,
        "model": {**model_spec, "resolved_revision": resolved_revision},
        "active_qk_basis": {
            "mechanism": active_rope_mechanism,
            "pair_rotation_angle_degrees": active_rope_angle_degrees,
            "scope": (
                "layer0_only_other_layers_unchanged"
                if active_rope_mechanism
                in {
                    "layer0_fixed_q7_rope_score_v1",
                    "layer0_relative_rope_score_fusion_v1",
                    "layer0_absolute_rope_online_attention_v1",
                    "layer0_projection_shadow_staged_attention_v1",
                    "layer0_tile_max_delta_attention_v1",
                    "layer0_tile_bfp_score_attention_v1",
                }
                else "all_layers"
            ),
            "runtime_arithmetic_added": (
                active_rope_mechanism in {
                    "dynamic_rope_head_scale_v1",
                    "layer0_fixed_q7_rope_score_v1",
                    "layer0_relative_rope_score_fusion_v1",
                    "layer0_absolute_rope_online_attention_v1",
                    "layer0_projection_shadow_staged_attention_v1",
                    "layer0_tile_max_delta_attention_v1",
                    "layer0_tile_bfp_score_attention_v1",
                }
            ),
        },
        "input_observations": observations,
        "metrics": metrics,
        "checks": checks,
        "thresholds": thresholds,
        "artifacts": {
            "accepted_rtl": rtl_binding,
            "derived_scales_sha256": sha256_file(scales_path),
            "run_contract_sha256": sha256_file(run_contract_path),
            "sources": source_artifacts(rtl_binding),
        },
        "runtime": {
            "device": "cpu",
            "executable": sys.executable,
            "packages": versions,
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch_runtime": torch.__version__,
            "seconds": {
                "bf16_perplexity": bf16_seconds,
                "calibration": calibration_seconds,
                "model_load": model_load_seconds,
                "w4a8_conversion": quantization_seconds,
                "w4a8_perplexity": w4a8_seconds,
                "total_before_result_write": time.perf_counter() - total_started,
            },
        },
        "saturation_diagnostics": saturation_diagnostics,
        "seeds": config["determinism"],
    }
    if activation_scale_cap is not None or activation_scale_percentile is not None:
        result["diagnostic_scale_policy"] = {
            "maximum_static_scale": activation_scale_cap,
            "per_prompt_absolute_percentile": activation_scale_percentile,
            "rmsnorm_gain_scale_floor": (
                "maximum_absolute_gain_times_2^8_divided_by_32767"
            ),
            "status": "diagnostic_hypothesis_not_accepted",
        }
    if diagnostic_attention_score_scale is not None:
        result["diagnostic_numerical_repair"] = {
            "lane": "A_layer0_attention_score_requantization",
            "layer": 0,
            "score_scale_factor": diagnostic_attention_score_scale,
            "factor_derivation": "mean least-squares scalar from frozen 32-token C4 and WikiText-2 layer-0 centered-score traces",
            "status": "diagnostic_hypothesis_not_accepted",
        }
    if diagnostic_rmsnorm_output_scale is not None:
        result["diagnostic_numerical_repair"] = {
            "lane": "B_layer0_input_rmsnorm_requantization",
            "layer": 0,
            "output_scale_factor": diagnostic_rmsnorm_output_scale,
            "factor_derivation": "least-squares scalar from the frozen 32-token C4 layer-0 input-RMSNorm trace",
            "status": "diagnostic_hypothesis_not_accepted",
        }
    if diagnostic_rope_mechanism is not None:
        result["diagnostic_numerical_repair"] = {
            "lane": (
                "layer0_fixed_q7_rope_score"
                if diagnostic_rope_mechanism == "layer0_fixed_q7_rope_score_v1"
                else (
                    "layer0_relative_rope_score_fusion"
                    if diagnostic_rope_mechanism
                    == "layer0_relative_rope_score_fusion_v1"
                    else (
                        "layer0_absolute_rope_online_attention"
                        if diagnostic_rope_mechanism
                        == "layer0_absolute_rope_online_attention_v1"
                        else (
                            "layer0_projection_shadow_staged_attention"
                            if diagnostic_rope_mechanism
                            == "layer0_projection_shadow_staged_attention_v1"
                            else (
                                "layer0_tile_max_delta_attention"
                                if diagnostic_rope_mechanism
                                == "layer0_tile_max_delta_attention_v1"
                                else (
                                    "layer0_tile_bfp_score_attention"
                                    if diagnostic_rope_mechanism
                                    == "layer0_tile_bfp_score_attention_v1"
                                    else "C_scalar_unit_gain_rope"
                                )
                            )
                        )
                    )
                )
            ),
            "mechanism": diagnostic_rope_mechanism,
            "qk_scale_granularity": (
                "layer0_static_per_tensor_other_layers_static_per_head"
                if diagnostic_rope_mechanism
                in {
                    "layer0_fixed_q7_rope_score_v1",
                    "layer0_relative_rope_score_fusion_v1",
                    "layer0_absolute_rope_online_attention_v1",
                    "layer0_projection_shadow_staged_attention_v1",
                    "layer0_tile_max_delta_attention_v1",
                    "layer0_tile_bfp_score_attention_v1",
                }
                else "static_per_tensor_not_per_head"
            ),
            "rope_conversion_q9": ROPE_DIAGNOSTIC_MECHANISMS[
                diagnostic_rope_mechanism
            ][0],
            "rope_output_bits": ROPE_DIAGNOSTIC_MECHANISMS[
                diagnostic_rope_mechanism
            ][1],
            "pair_rotation_angle_degrees": diagnostic_rope_angle_degrees,
            "status": "diagnostic_hypothesis_not_accepted",
        }
    write_json(output_dir / "results.json", result)
    return result


def assert_tensor_equal(actual: Tensor, expected: list[int], name: str) -> None:
    expected_tensor = torch.tensor(expected, dtype=actual.dtype).reshape(actual.shape)
    if not torch.equal(actual.cpu(), expected_tensor):
        raise AssertionError(f"{name} tensor kernel differs from accepted scalar reference")


def self_test() -> None:
    from ace2_attention_compose_reference import (
        AttentionComposeCase,
        reference_attention_compose,
    )
    from ace2_attention_score_reference import (
        AttentionScoreCase,
        DynamicAttentionScoreCase,
        reference_attention_score,
        reference_dynamic_attention_score,
    )
    from ace2_attention_value_reference import (
        AttentionValueCase,
        reference_attention_value,
    )
    from ace2_projection_reference import (
        ProjectionCase,
        reference_projection,
        round_shift_even as scalar_round_shift_even,
    )
    from ace2_residual_reference import reference_residual_add
    from ace2_rmsnorm_reference import derive_scaled_gains_q8, reference_rmsnorm
    from ace2_rope_reference import (
        DynamicRopeHeadCase,
        RopeCase,
        reference_dynamic_rope_head,
        reference_rope,
    )
    from ace2_silu_gate_reference import SiluGateCase, reference_silu_gate
    from ace2_softmax_reference import SoftmaxCase, reference_softmax

    values = torch.tensor(
        [0, 1, -1, 4, 12, 20, -4, -12, -20, (1 << 62), -(1 << 62)],
        dtype=torch.int64,
    )
    for shift in (0, 1, 3, 31, 62, 63):
        actual = round_shift_even(values, shift).tolist()
        expected = [scalar_round_shift_even(value, shift) for value in values.tolist()]
        if actual != expected:
            raise AssertionError(f"round-shift-even differs at shift {shift}")

    real = torch.tensor([0.0, 0.5, 1.0, 3.25, float(INT32_MAX)], dtype=torch.float64)
    multipliers, shifts = derive_multiplier(real)
    for value, multiplier, shift in zip(
        real.tolist(),
        multipliers.tolist(),
        shifts.tolist(),
        strict=True,
    ):
        if not 0 <= shift <= 63 or multiplier != round(value * math.ldexp(1.0, shift)):
            raise AssertionError("derived multiplier does not encode the requested real value")
        if shift < 63 and round(value * math.ldexp(1.0, shift + 1)) <= INT32_MAX:
            raise AssertionError("derived right shift is not the largest representable shift")

    source = nn.Linear(128, 3, bias=False)
    with torch.no_grad():
        weights = torch.tensor(
            [[((row * 13 + column * 7) % 31) - 15 for column in range(128)] for row in range(3)],
            dtype=torch.float32,
        )
        source.weight.copy_(weights / 8.0)
    fixed = W4A8Linear(source, CalibrationRange(4.0, 32.0))
    inputs = torch.tensor(
        [[((column * 11) % 17) - 8 for column in range(128)]],
        dtype=torch.bfloat16,
    )
    qinput = quantize_int8(inputs, fixed.input_scale)
    projection_case = ProjectionCase(
        name="full_model_projection_binding",
        rows=1,
        reduction_size=128,
        activations=[qinput[0].tolist()],
        weights=fixed.qweight.tolist(),
        multipliers=fixed.multiplier.tolist(),
        right_shifts=fixed.right_shift.tolist(),
        output_zero_points=[0, 0, 0],
        bias_accumulators=[0, 0, 0],
    )
    projection_expected = reference_projection(projection_case).outputs[0]
    assert_tensor_equal(
        fixed.forward_quantized(qinput),
        projection_expected,
        "projection",
    )
    fixed.input_is_quantized = True
    hardware_output = fixed(qinput.to(torch.bfloat16))
    expected_hardware_output = (
        torch.tensor(projection_expected, dtype=torch.bfloat16).reshape(hardware_output.shape)
        * fixed.output_scale
    )
    if not torch.equal(hardware_output, expected_hardware_output):
        raise AssertionError("composed integer projection boundary requantized its input")

    rms_input = torch.tensor(
        [[((index * 29) % 255) - 127 for index in range(RMS_HIDDEN_SIZE)]],
        dtype=torch.int8,
    )
    gains = torch.tensor(
        [6000 + (index % 4000) for index in range(RMS_HIDDEN_SIZE)],
        dtype=torch.int16,
    )
    rms_expected = reference_rmsnorm(rms_input[0].tolist(), gains.tolist()).outputs
    rms_output = fixed_rmsnorm_raw(rms_input, gains)
    assert_tensor_equal(rms_output, rms_expected, "RMSNorm")
    test_output_scale = math.ldexp(1.0, -5)
    unity = torch.full(
        (RMS_HIDDEN_SIZE,),
        round((1.0 / test_output_scale) * (1 << RMS_GAIN_FRAC)),
        dtype=torch.int16,
    )
    all_one = torch.ones((1, RMS_HIDDEN_SIZE), dtype=torch.int8)
    all_one_output = fixed_rmsnorm_raw(all_one, unity)
    if not torch.all(all_one_output == 32):
        raise AssertionError("RMSNorm scaled output collapsed a unit-normalized vector")
    alternate_output_scale = 3.0 / 64.0
    alternate_weights = [
        0.875 + (index % 17) / 64.0 for index in range(RMS_HIDDEN_SIZE)
    ]
    alternate_gains = derive_scaled_gains_q8(
        alternate_weights,
        alternate_output_scale,
    )
    alternate_source = nn.RMSNorm(RMS_HIDDEN_SIZE)
    with torch.no_grad():
        alternate_source.weight.copy_(torch.tensor(alternate_weights))
    alternate_fixed = FixedRMSNorm(
        alternate_source,
        input_scale=0.125,
        output_scale=alternate_output_scale,
    )
    assert_tensor_equal(
        alternate_fixed.scaled_gains_q8,
        alternate_gains,
        "per-tensor RMSNorm gain metadata",
    )
    alternate_expected = reference_rmsnorm(
        rms_input[0].tolist(),
        alternate_gains,
    ).outputs
    assert_tensor_equal(
        fixed_rmsnorm_raw(rms_input, alternate_fixed.scaled_gains_q8),
        alternate_expected,
        "per-tensor RMSNorm output",
    )
    tie_gains = torch.tensor(
        [128, 384, -128, -384] * (RMS_HIDDEN_SIZE // 4),
        dtype=torch.int16,
    )
    tie_output = fixed_rmsnorm_raw(all_one, tie_gains)
    assert_tensor_equal(
        tie_output,
        [0, 2, 0, -2] * (RMS_HIDDEN_SIZE // 4),
        "RMSNorm ties-to-even",
    )

    chain_input_scales = {
        "q_proj": 0.25,
        "k_proj": 0.125,
        "v_proj": 0.5,
        "gate_proj": 0.0625,
        "up_proj": 0.375,
        "final_norm_to_lm_head": 1.75,
    }
    for chain_index, (chain_name, calibrated_input_scale) in enumerate(
        chain_input_scales.items()
    ):
        chain_source = nn.Linear(RMS_HIDDEN_SIZE, 1, bias=False)
        with torch.no_grad():
            chain_source.weight.copy_(
                rms_output.to(torch.float32)
                * (0.25 + chain_index / 32.0)
            )
        chain_projection = W4A8Linear(
            chain_source,
            CalibrationRange(calibrated_input_scale * 127.0, 6350.0),
        )
        unbound_multiplier = chain_projection.multiplier.clone()
        unbound_right_shift = chain_projection.right_shift.clone()
        chain_projection.bind_hardware_input_scale(test_output_scale)
        if chain_projection.hardware_input_scale != test_output_scale:
            raise AssertionError(f"{chain_name} did not bind the RMSNorm output scale")
        if torch.equal(chain_projection.multiplier, unbound_multiplier) and torch.equal(
            chain_projection.right_shift,
            unbound_right_shift,
        ):
            raise AssertionError(f"{chain_name} scale binding did not change metadata")
        if chain_projection.hardware_input_scale == chain_projection.input_scale:
            raise AssertionError(f"{chain_name} did not exercise unequal scale domains")
        chain_case = ProjectionCase(
            name=f"rmsnorm_to_{chain_name}_scale_binding",
            rows=1,
            reduction_size=RMS_HIDDEN_SIZE,
            activations=[rms_expected],
            weights=chain_projection.qweight.tolist(),
            multipliers=chain_projection.multiplier.tolist(),
            right_shifts=chain_projection.right_shift.tolist(),
            output_zero_points=[0],
            bias_accumulators=[0],
        )
        chain_expected = reference_projection(chain_case).outputs[0]
        chain_actual = chain_projection.forward_hardware_input(rms_output)
        assert_tensor_equal(chain_actual, chain_expected, chain_name)

    rope_activations = [((index * 17) % 255) - 127 for index in range(RMS_HIDDEN_SIZE)]
    rope_scales = [37 for _ in range(RMS_HIDDEN_SIZE)]
    rope_cos = [
        32767 if (index % ROPE_HEAD_DIM) % 3 else 23170
        for index in range(RMS_HIDDEN_SIZE)
    ]
    rope_sin = [
        0 if (index % ROPE_HEAD_DIM) % 3 else 23170
        for index in range(RMS_HIDDEN_SIZE)
    ]
    rope_case = RopeCase(
        "full_model_rope_binding",
        7,
        rope_activations,
        rope_scales,
        rope_cos,
        rope_sin,
    )
    rope_expected = reference_rope(rope_case).outputs
    rope_input_tensor = torch.tensor(rope_activations, dtype=torch.int8).reshape(
        1, 14, 1, 64
    )
    rope_cos_tensor = (
        torch.tensor(rope_cos, dtype=torch.float64).reshape(1, 14, 1, 64)[:, 0, :, :]
        / 32767.0
    )
    rope_sin_tensor = (
        torch.tensor(rope_sin, dtype=torch.float64).reshape(1, 14, 1, 64)[:, 0, :, :]
        / 32767.0
    )
    assert_tensor_equal(
        fixed_rope_raw(rope_input_tensor, 37, rope_cos_tensor, rope_sin_tensor),
        rope_expected,
        "RoPE",
    )
    saturated_rope_input = torch.tensor(
        [127, -128, *([0] * (ROPE_HEAD_DIM - 2))],
        dtype=torch.int8,
    ).reshape(1, 1, 1, ROPE_HEAD_DIM)
    saturated_rope_output, saturated_rope_count = fixed_rope_raw_with_saturation(
        saturated_rope_input,
        ROPE_SCALE_Q9_MAX,
        torch.ones((1, 1, ROPE_HEAD_DIM), dtype=torch.float64),
        torch.zeros((1, 1, ROPE_HEAD_DIM), dtype=torch.float64),
    )
    saturated_rope_expected = reference_rope(
        RopeCase(
            "full_model_rope_q9_output_saturation",
            0,
            saturated_rope_input.reshape(-1).tolist(),
            [ROPE_SCALE_Q9_MAX] * ROPE_HEAD_DIM,
            [32767] * ROPE_HEAD_DIM,
            [0] * ROPE_HEAD_DIM,
        )
    ).outputs
    assert_tensor_equal(
        saturated_rope_output,
        saturated_rope_expected,
        "RoPE Q6.9 output saturation",
    )
    if saturated_rope_count != 2:
        raise AssertionError("RoPE Q6.9 output saturation count is incorrect")

    minimum_scale = pack_scale32(0x8000, -24)
    maximum_scale = pack_scale32(0xFFFF, 4)
    if minimum_scale != SCALE32_ALL_ZERO_RECORD:
        raise AssertionError("canonical all-zero Scale32 record is incorrect")
    if ceil_scale32_from_ratio(1, 1 << 80) != minimum_scale:
        raise AssertionError("Scale32 below-range ceil did not select the minimum")
    max_num, max_den = scale32_ratio(maximum_scale)
    if ceil_scale32_from_ratio(max_num, max_den) != maximum_scale:
        raise AssertionError("Scale32 maximum endpoint did not round-trip")
    try:
        ceil_scale32_from_ratio(max_num + 1, max_den)
    except OverflowError:
        pass
    else:
        raise AssertionError("Scale32 overflow above the maximum was not rejected")
    if round_divide_even_signed(5, 2) != 2 or round_divide_even_signed(-5, 2) != -2:
        raise AssertionError("signed Scale32 ties-to-even rounding is incorrect")
    max_pair_sig, max_pair_shift = dynamic_score_pair_parameters(
        pack_scale32(0xFFFF, 4),
        pack_scale32(0xFFFF, 4),
    )
    if max_pair_sig != 0x1FFFC or max_pair_shift != 1:
        raise AssertionError("dynamic score retained-width endpoint is incorrect")

    dynamic_identity_cos = [32767] * ROPE_HEAD_DIM
    dynamic_identity_sin = [0] * ROPE_HEAD_DIM
    producer_record = ceil_scale32_from_float(0.125)
    low_dynamic_case = DynamicRopeHeadCase(
        "dynamic_low_amplitude",
        [((index * 3) % 7) - 3 for index in range(ROPE_HEAD_DIM)],
        producer_record,
        dynamic_identity_cos,
        dynamic_identity_sin,
    )
    high_dynamic_case = DynamicRopeHeadCase(
        "dynamic_high_amplitude",
        [((index * 29) % 255) - 127 for index in range(ROPE_HEAD_DIM)],
        producer_record,
        dynamic_identity_cos,
        dynamic_identity_sin,
    )
    low_dynamic = reference_dynamic_rope_head(low_dynamic_case)
    high_dynamic = reference_dynamic_rope_head(high_dynamic_case)
    if low_dynamic.output_scale32 == high_dynamic.output_scale32:
        raise AssertionError("dynamic RoPE collapsed unequal head amplitudes")
    dynamic_tensor_input = torch.tensor(
        [low_dynamic_case.activations, high_dynamic_case.activations],
        dtype=torch.int8,
    ).reshape(1, 1, 2, ROPE_HEAD_DIM)
    dynamic_tensor_output, dynamic_records, dynamic_saturation = dynamic_rope_head_raw(
        dynamic_tensor_input,
        producer_record,
        torch.ones((1, 2, ROPE_HEAD_DIM), dtype=torch.float64),
        torch.zeros((1, 2, ROPE_HEAD_DIM), dtype=torch.float64),
    )
    assert_tensor_equal(
        dynamic_tensor_output,
        low_dynamic.outputs + high_dynamic.outputs,
        "dynamic RoPE",
    )
    if dynamic_records.reshape(-1).tolist() != [
        low_dynamic.output_scale32,
        high_dynamic.output_scale32,
    ] or dynamic_saturation != 0:
        raise AssertionError("dynamic RoPE metadata differs from scalar reference")

    alternate_key_record = pack_scale32(
        *unpack_scale32(high_dynamic.output_scale32)[:1],
        unpack_scale32(high_dynamic.output_scale32)[1] - 1,
    )
    dynamic_score_case = DynamicAttentionScoreCase(
        "identical_payload_different_key_scale",
        low_dynamic.outputs,
        [high_dynamic.outputs, high_dynamic.outputs],
        low_dynamic.output_scale32,
        [high_dynamic.output_scale32, alternate_key_record],
    )
    dynamic_score_expected = reference_dynamic_attention_score(dynamic_score_case)
    dynamic_score_actual = fixed_dynamic_attention_scores_raw(
        torch.tensor(low_dynamic.outputs, dtype=torch.int8).reshape(1, 1, 1, 64),
        torch.tensor(
            [high_dynamic.outputs, high_dynamic.outputs], dtype=torch.int8
        ).reshape(1, 1, 2, 64),
        torch.tensor([low_dynamic.output_scale32], dtype=torch.int64).reshape(1, 1, 1),
        torch.tensor(
            [high_dynamic.output_scale32, alternate_key_record], dtype=torch.int64
        ).reshape(1, 1, 2),
        None,
    )
    assert_tensor_equal(
        dynamic_score_actual,
        dynamic_score_expected.scores_q6_9,
        "dynamic attention score",
    )

    scale_binding = derive_attention_scale_binding(0.125, 0.0625)
    if scale_binding.conversion_q9 != ROPE_SAFE_CONVERSION_Q9:
        raise AssertionError("range-safe Q/K RoPE conversion metadata is incorrect")
    realized_conversion = ROPE_SAFE_CONVERSION_Q9 / float(1 << ROPE_SCALE_FRAC)
    if not math.isclose(
        scale_binding.query_output_scale,
        0.125 / realized_conversion,
        rel_tol=0.0,
        abs_tol=0.0,
    ) or not math.isclose(
        scale_binding.key_output_scale,
        0.0625 / realized_conversion,
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise AssertionError("Q/K RoPE output scales do not preserve producer scale")
    if math.isclose(
        scale_binding.query_output_scale,
        scale_binding.key_output_scale,
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise AssertionError("unequal Q/K producer scales collapsed to one scale")

    scaled_q_values = [(index % 3) - 1 for index in range(64)]
    scaled_k_values = [((index * 5) % 3) - 1 for index in range(64)]
    identity_cos = torch.ones((1, 1, 64), dtype=torch.float64)
    identity_sin = torch.zeros((1, 1, 64), dtype=torch.float64)
    scaled_query = fixed_rope_raw(
        torch.tensor(scaled_q_values, dtype=torch.int8).reshape(1, 1, 1, 64),
        scale_binding.conversion_q9,
        identity_cos,
        identity_sin,
    )
    scaled_key = fixed_rope_raw(
        torch.tensor(scaled_k_values, dtype=torch.int8).reshape(1, 1, 1, 64),
        scale_binding.conversion_q9,
        identity_cos,
        identity_sin,
    )
    scaled_score_real_multiplier = torch.tensor(
        [
            scale_binding.query_output_scale
            * scale_binding.key_output_scale
            * (1 << ATTENTION_SCORE_FRAC)
            / math.sqrt(ROPE_HEAD_DIM)
        ],
        dtype=torch.float64,
    )
    scaled_score_multiplier, scaled_score_right_shift = derive_multiplier(
        scaled_score_real_multiplier
    )
    corrected_score_multiplier, corrected_score_right_shift = (
        derive_attention_score_metadata(scale_binding, ROPE_HEAD_DIM, 0.5)
    )
    baseline_realized = float(scaled_score_multiplier.item()) / math.ldexp(
        1.0, int(scaled_score_right_shift.item())
    )
    corrected_realized = float(corrected_score_multiplier.item()) / math.ldexp(
        1.0, int(corrected_score_right_shift.item())
    )
    if not math.isclose(
        corrected_realized / baseline_realized,
        0.5,
        rel_tol=1e-8,
        abs_tol=0.0,
    ):
        raise AssertionError("attention score diagnostic factor did not bind requantization metadata")
    scaled_score = fixed_attention_scores_raw(
        scaled_query,
        scaled_key,
        None,
        scaled_score_multiplier,
        scaled_score_right_shift,
    )
    if scaled_score.item() != 0:
        raise AssertionError("single-token attention score was not max-centered")

    q_projection_values = [((index * 5) % 31) - 15 for index in range(64)]
    k_projection_values = [
        [((token * 7 + index * 3) % 29) - 14 for index in range(64)]
        for token in range(5)
    ]
    chain_binding = derive_attention_scale_binding(0.75, 1.25)
    chain_cos = torch.ones((1, 5, ROPE_HEAD_DIM), dtype=torch.float64)
    chain_sin = torch.zeros((1, 5, ROPE_HEAD_DIM), dtype=torch.float64)
    q_rope_actual = fixed_rope_raw(
        torch.tensor(q_projection_values, dtype=torch.int8).reshape(1, 1, 1, 64),
        chain_binding.conversion_q9,
        chain_cos[:, :1, :],
        chain_sin[:, :1, :],
    )
    k_rope_actual = fixed_rope_raw(
        torch.tensor(k_projection_values, dtype=torch.int8).reshape(1, 1, 5, 64),
        chain_binding.conversion_q9,
        chain_cos,
        chain_sin,
    )
    q_values = q_rope_actual.reshape(-1).tolist()
    k_values = k_rope_actual.reshape(5, 64).tolist()
    q_rope_expected = reference_rope(
        RopeCase(
            "unequal_scale_chain_query",
            0,
            q_projection_values,
            [chain_binding.conversion_q9] * ROPE_HEAD_DIM,
            [32767] * ROPE_HEAD_DIM,
            [0] * ROPE_HEAD_DIM,
        )
    ).outputs
    k_rope_expected = reference_rope(
        RopeCase(
            "unequal_scale_chain_key",
            0,
            [value for row in k_projection_values for value in row],
            [chain_binding.conversion_q9] * (5 * ROPE_HEAD_DIM),
            [32767] * (5 * ROPE_HEAD_DIM),
            [0] * (5 * ROPE_HEAD_DIM),
        )
    ).outputs
    assert_tensor_equal(q_rope_actual, q_rope_expected, "unequal-scale query RoPE chain")
    assert_tensor_equal(k_rope_actual, k_rope_expected, "unequal-scale key RoPE chain")
    score_reference = reference_attention_score(
        AttentionScoreCase(
            "full_model_unequal_scale_score_chain",
            q_values,
            k_values,
            query_scale=chain_binding.query_output_scale,
            key_scale=chain_binding.key_output_scale,
        )
    )
    score_expected = score_reference.scores_q6_9
    score_actual = fixed_attention_scores_raw(
        torch.tensor(q_values, dtype=torch.int8).reshape(1, 1, 1, 64),
        torch.tensor(k_values, dtype=torch.int8).reshape(1, 1, 5, 64),
        None,
        torch.tensor([score_reference.multiplier], dtype=torch.int64),
        torch.tensor([score_reference.right_shift], dtype=torch.int64),
    )
    assert_tensor_equal(score_actual, score_expected, "attention score")

    softmax_expected = reference_softmax(
        SoftmaxCase("full_model_softmax_binding", score_expected)
    ).probabilities_q0_15[: len(score_expected)]
    probabilities = fixed_softmax_raw(score_actual)
    assert_tensor_equal(probabilities, softmax_expected, "softmax")

    value_rows = [
        [((token * 19 + index * 11) % 255) - 127 for index in range(64)]
        for token in range(5)
    ]
    value_expected = reference_attention_value(
        AttentionValueCase("full_model_value_binding", softmax_expected, value_rows)
    ).outputs
    value_actual = fixed_attention_value_raw(
        probabilities,
        torch.tensor(value_rows, dtype=torch.int8).reshape(1, 1, 5, 64),
    )
    assert_tensor_equal(value_actual, value_expected, "attention value")

    value_scale = 0.03125
    o_source = nn.Linear(ROPE_HEAD_DIM, 3, bias=False)
    with torch.no_grad():
        o_source.weight.copy_(
            torch.tensor(
                [
                    [((row * 17 + column * 11) % 31) - 15 for column in range(ROPE_HEAD_DIM)]
                    for row in range(3)
                ],
                dtype=torch.float32,
            )
            / 8.0
        )
    o_projection = W4A8Linear(
        o_source,
        CalibrationRange(0.5 * 127.0, 48.0),
        input_is_quantized=True,
    )
    unbound_o_output = o_projection.forward_hardware_input(value_actual)
    o_projection.bind_hardware_input_scale(value_scale)
    if o_projection.hardware_input_scale == o_projection.input_scale:
        raise AssertionError("attention-value-to-o_proj did not exercise unequal scales")
    o_case = ProjectionCase(
        name="attention_value_to_o_proj_scale_binding",
        rows=1,
        reduction_size=ROPE_HEAD_DIM,
        activations=[value_expected],
        weights=o_projection.qweight.tolist(),
        multipliers=o_projection.multiplier.tolist(),
        right_shifts=o_projection.right_shift.tolist(),
        output_zero_points=[0, 0, 0],
        bias_accumulators=[0, 0, 0],
    )
    o_expected = reference_projection(o_case).outputs[0]
    o_actual = o_projection.forward_hardware_input(value_actual)
    assert_tensor_equal(o_actual, o_expected, "attention-value-to-o_proj")
    if torch.equal(o_actual, unbound_o_output):
        raise AssertionError(
            "attention-value-to-o_proj unequal-scale discriminator did not change output"
        )

    compose_scores = [((index * 97) % 2048) - 1024 for index in range(11)]
    compose_values = [
        [((token * 23 + index * 13) % 255) - 127 for index in range(64)]
        for token in range(11)
    ]
    compose_expected = reference_attention_compose(
        AttentionComposeCase(
            "full_model_compose_binding",
            compose_scores,
            compose_values,
        )
    )
    compose_probabilities = fixed_softmax_raw(
        torch.tensor(compose_scores, dtype=torch.int16).reshape(1, 1, 1, 11)
    )
    assert_tensor_equal(
        compose_probabilities,
        compose_expected.probabilities_q15,
        "long-context softmax compose",
    )
    assert_tensor_equal(
        fixed_attention_value_raw(
            compose_probabilities,
            torch.tensor(compose_values, dtype=torch.int8).reshape(1, 1, 11, 64),
        ),
        compose_expected.outputs,
        "long-context attention compose",
    )

    gate_raw = torch.tensor(
        [[-128, -64, -17, -1, 0, 1, 31, 63, 127]],
        dtype=torch.int8,
    )
    up_raw = torch.tensor(
        [[127, -128, 29, -31, 3, -7, 61, -73, 11]],
        dtype=torch.int8,
    )
    gate_scale = 0.125
    up_scale = 0.0625
    silu_multiplier = torch.tensor(37, dtype=torch.int64)
    silu_shift = torch.tensor(17, dtype=torch.int64)
    gate_q6_9 = (
        torch.round(gate_raw.to(torch.float64) * gate_scale * 512)
        .to(torch.int64)
        .tolist()[0]
    )
    up_q6_9 = (
        torch.round(up_raw.to(torch.float64) * up_scale * 512)
        .to(torch.int64)
        .tolist()[0]
    )
    silu_expected = reference_silu_gate(
        SiluGateCase(
            "full_model_silu_binding",
            gate_q6_9,
            up_q6_9,
            int(silu_multiplier),
            int(silu_shift),
            0,
        )
    ).outputs
    assert_tensor_equal(
        fixed_silu_gate_raw(
            gate_raw,
            up_raw,
            gate_scale,
            up_scale,
            silu_multiplier,
            silu_shift,
        ),
        silu_expected,
        "SiLU gate",
    )

    lhs = [((index * 37) % 255) - 127 for index in range(RMS_HIDDEN_SIZE)]
    rhs = [((index * 43) % 255) - 127 for index in range(RMS_HIDDEN_SIZE)]
    residual_expected, _ = reference_residual_add(lhs, rhs)
    residual_actual = fixed_residual_add(
        torch.tensor(lhs, dtype=torch.int8).to(torch.float32),
        torch.tensor(rhs, dtype=torch.int8).to(torch.float32),
        1.0,
    ).to(torch.int8)
    assert_tensor_equal(residual_actual, residual_expected, "residual add")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "official", "self-test"], required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--activation-scale-cap", type=float)
    parser.add_argument("--activation-scale-percentile", type=float)
    parser.add_argument("--diagnostic-attention-score-scale", type=float)
    parser.add_argument("--diagnostic-rmsnorm-output-scale", type=float)
    parser.add_argument(
        "--diagnostic-rope-mechanism",
        choices=sorted(ROPE_DIAGNOSTIC_MECHANISMS),
    )
    parser.add_argument("--diagnostic-rope-angle-degrees", type=float)
    parser.add_argument("--candidate-evidence", type=Path)
    parser.add_argument("--smoke-token-limit", type=int, default=32)
    args = parser.parse_args()
    if args.mode == "self-test":
        validate_runtime(json.loads(QUALITY_CONFIG.read_text(encoding="utf-8")))
        self_test()
        print("ACE2_FULL_MODEL_FIXED_POINT_SELF_TEST status=pass")
        return
    if args.output_dir is None:
        raise SystemExit("--output-dir is required for smoke and official modes")
    if (
        args.activation_scale_cap is not None
        or args.activation_scale_percentile is not None
        or args.diagnostic_attention_score_scale is not None
        or args.diagnostic_rmsnorm_output_scale is not None
        or args.diagnostic_rope_mechanism is not None
    ) and args.mode != "smoke":
        raise SystemExit("numerical diagnostics require --mode smoke")
    diagnostic_choices = [
        args.activation_scale_cap is not None,
        args.activation_scale_percentile is not None,
        args.diagnostic_attention_score_scale is not None,
        args.diagnostic_rmsnorm_output_scale is not None,
        args.diagnostic_rope_mechanism is not None,
    ]
    if sum(diagnostic_choices) > 1:
        raise SystemExit("choose one numerical diagnostic at a time")
    if args.mode != "smoke" and args.smoke_token_limit != 32:
        raise SystemExit("--smoke-token-limit is valid only for --mode smoke")
    if args.mode != "smoke" and args.candidate_evidence is not None:
        raise SystemExit("--candidate-evidence is valid only for --mode smoke")
    result = run(
        args.mode,
        args.output_dir,
        activation_scale_cap=args.activation_scale_cap,
        activation_scale_percentile=args.activation_scale_percentile,
        diagnostic_attention_score_scale=args.diagnostic_attention_score_scale,
        diagnostic_rmsnorm_output_scale=args.diagnostic_rmsnorm_output_scale,
        diagnostic_rope_mechanism=args.diagnostic_rope_mechanism,
        diagnostic_rope_angle_degrees=args.diagnostic_rope_angle_degrees,
        candidate_evidence_path=args.candidate_evidence,
        smoke_token_limit=args.smoke_token_limit,
    )
    print(
        "ACE2_FULL_MODEL_FIXED_POINT "
        f"mode={args.mode} classification={result['classification']} "
        f"gate_passed={str(result['gate_passed']).lower()}"
    )


if __name__ == "__main__":
    main()
