#!/usr/bin/env python3
"""Generate and validate ACE-2 Qwen2.5 model hardware contracts."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT / "design/model_hardware_contract.schema.json"
DESCRIPTOR_DIR = ROOT / "design/model_hardware_contracts"
SCHEMA_VERSION = 1
PRECISION_MODE = "w4a8_scale32"


class ContractError(ValueError):
    """Raised when a model hardware contract is incompatible."""


@dataclass(frozen=True)
class QwenConfig:
    model_id: str
    repository: str
    revision: str
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    max_position_embeddings: int
    vocab_size: int
    tie_word_embeddings: bool


QWEN_CONFIGS = (
    QwenConfig(
        model_id="qwen2.5-0.5b",
        repository="Qwen/Qwen2.5-0.5B",
        revision="060db6499f32faf8b98477b0a26969ef7d8b9987",
        hidden_size=896,
        intermediate_size=4864,
        num_hidden_layers=24,
        num_attention_heads=14,
        num_key_value_heads=2,
        max_position_embeddings=32768,
        vocab_size=151936,
        tie_word_embeddings=True,
    ),
    QwenConfig(
        model_id="qwen2.5-1.5b",
        repository="Qwen/Qwen2.5-1.5B",
        revision="8faed761d45a263340a0528343f099c05c9a4323",
        hidden_size=1536,
        intermediate_size=8960,
        num_hidden_layers=28,
        num_attention_heads=12,
        num_key_value_heads=2,
        max_position_embeddings=131072,
        vocab_size=151936,
        tie_word_embeddings=True,
    ),
    QwenConfig(
        model_id="qwen2.5-3b",
        repository="Qwen/Qwen2.5-3B",
        revision="3aab1f1954e9cc14eb9509a215f9e5ca08227a9b",
        hidden_size=2048,
        intermediate_size=11008,
        num_hidden_layers=36,
        num_attention_heads=16,
        num_key_value_heads=2,
        max_position_embeddings=32768,
        vocab_size=151936,
        tie_word_embeddings=True,
    ),
    QwenConfig(
        model_id="qwen2.5-7b",
        repository="Qwen/Qwen2.5-7B",
        revision="d149729398750b98c0af14eb82c78cfe92750796",
        hidden_size=3584,
        intermediate_size=18944,
        num_hidden_layers=28,
        num_attention_heads=28,
        num_key_value_heads=4,
        max_position_embeddings=131072,
        vocab_size=152064,
        tie_word_embeddings=False,
    ),
)
CONFIG_BY_ID = {config.model_id: config for config in QWEN_CONFIGS}

PRECISION = {
    "supported_modes": [PRECISION_MODE],
    "default_mode": PRECISION_MODE,
    "weight_bits": 4,
    "activation_bits": 8,
    "accumulator_bits": 32,
    "kv_cache_bits": 8,
    "scale_format": "scale32",
}
WEIGHT_LAYOUT = {
    "matrix_order": "output_channel_major_row_major",
    "signed_encoding": "twos_complement",
    "packing": "even_input_low_nibble_odd_input_high_nibble",
    "weight_scale_granularity": "per_output_channel",
    "projection_record_bytes": 16,
    "embedding_storage": "bf16",
    "lm_head_representation": "separate_packed_w4_copy",
}
HARDWARE = {
    "compute_tile": {
        "vector_lanes": 16,
        "projection_m": 1,
        "projection_n": 32,
        "projection_k": 32,
        "lm_head_vocab": 32,
    },
    "streaming_memory": {
        "boundary": "abstract_streaming_memory",
        "address_width_bits": 64,
        "data_width_bits": 128,
        "request_channels": 1,
        "response_channels": 1,
        "sram_banks": 8,
    },
}


def _strict_object(properties: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": list(properties),
        "properties": properties,
    }


def _positive_integer() -> dict[str, Any]:
    return {"type": "integer", "minimum": 1}


DIMENSION_FIELDS = (
    "hidden_size",
    "intermediate_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "max_position_embeddings",
    "vocab_size",
)
DERIVED_FIELDS = (
    "head_dim",
    "query_heads_per_kv_head",
    "kv_width_elements",
    "transformer_linear_weight_elements",
    "packed_linear_weight_elements",
    "packed_w4_bytes",
    "projection_output_rows",
    "projection_metadata_bytes",
    "rmsnorm_metadata_bytes",
    "operator_aux_metadata_bytes",
    "bf16_embedding_bytes",
    "estimated_weight_bytes",
    "kv_payload_bytes_per_token_per_layer",
    "kv_scale_metadata_bytes_per_token_per_layer",
    "kv_bytes_per_token_per_layer",
    "maximum_kv_bytes",
    "maximum_weight_plus_kv_bytes",
)

SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": (
        "https://github.com/Argus-AiTeam/ace-2/"
        "design/model_hardware_contract.schema.json"
    ),
    "title": "ACE-2 Qwen2.5 Model Hardware Contract",
    **_strict_object(
        {
            "schema_version": {"const": SCHEMA_VERSION},
            "contract_id": {"type": "string", "minLength": 1},
            "model_id": {"enum": [config.model_id for config in QWEN_CONFIGS]},
            "source": _strict_object(
                {
                    "repository": {"type": "string", "minLength": 1},
                    "revision": {"type": "string", "pattern": "^[0-9a-f]{40}$"},
                    "config_path": {"const": "config.json"},
                    "config_url": {"type": "string", "format": "uri"},
                }
            ),
            "dimensions": _strict_object(
                {
                    **{field: _positive_integer() for field in DIMENSION_FIELDS},
                    "tie_word_embeddings": {"type": "boolean"},
                }
            ),
            "precision": _strict_object(
                {
                    "supported_modes": {
                        "type": "array",
                        "minItems": 1,
                        "uniqueItems": True,
                        "items": {"enum": [PRECISION_MODE]},
                    },
                    "default_mode": {"enum": [PRECISION_MODE]},
                    "weight_bits": {"const": 4},
                    "activation_bits": {"const": 8},
                    "accumulator_bits": {"const": 32},
                    "kv_cache_bits": {"const": 8},
                    "scale_format": {"const": "scale32"},
                }
            ),
            "weight_layout": _strict_object(
                {
                    key: {"const": value}
                    for key, value in WEIGHT_LAYOUT.items()
                }
            ),
            "hardware": _strict_object(
                {
                    "compute_tile": _strict_object(
                        {
                            key: {"const": value}
                            for key, value in HARDWARE["compute_tile"].items()
                        }
                    ),
                    "streaming_memory": _strict_object(
                        {
                            key: {"const": value}
                            for key, value in HARDWARE["streaming_memory"].items()
                        }
                    ),
                }
            ),
            "compatibility": _strict_object(
                {
                    "validation_scope": {
                        "enum": [
                            "existing_0p5b_package_runtime_preflight",
                            "structural_only_no_rtl_execution",
                        ]
                    },
                    "existing_ace2_runtime_preflight": {"type": "boolean"},
                    "full_rtl_execution_validated": {"const": False},
                }
            ),
            "derived": _strict_object(
                {field: _positive_integer() for field in DERIVED_FIELDS}
            ),
            "estimate_scope": _strict_object(
                {
                    "weight": {"type": "string", "minLength": 1},
                    "kv_cache": {"type": "string", "minLength": 1},
                    "units": {"const": "bytes"},
                }
            ),
        }
    ),
}


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def descriptor_path(model_id: str) -> Path:
    if model_id not in CONFIG_BY_ID:
        raise ContractError(f"unsupported model_id: {model_id}")
    return DESCRIPTOR_DIR / f"{model_id}.json"


def derive(config: QwenConfig) -> dict[str, int]:
    head_dim = config.hidden_size // config.num_attention_heads
    kv_width = config.num_key_value_heads * head_dim
    per_layer_weights = (
        2 * config.hidden_size * config.hidden_size
        + 2 * config.hidden_size * kv_width
        + 3 * config.hidden_size * config.intermediate_size
    )
    transformer_weights = config.num_hidden_layers * per_layer_weights
    packed_weights = transformer_weights + config.vocab_size * config.hidden_size
    projection_rows_per_layer = (
        3 * config.hidden_size
        + 2 * kv_width
        + 2 * config.intermediate_size
    )
    projection_rows = (
        config.num_hidden_layers * projection_rows_per_layer + config.vocab_size
    )
    packed_w4_bytes = (packed_weights + 1) // 2
    projection_metadata_bytes = (
        projection_rows * WEIGHT_LAYOUT["projection_record_bytes"]
    )
    rmsnorm_metadata_bytes = (2 * config.num_hidden_layers + 1) * (
        config.hidden_size * 2 + 16
    )
    operator_aux_bytes_per_layer = (
        config.hidden_size * 2
        + kv_width * 2
        + 2 * config.num_key_value_heads * 4
        + config.num_attention_heads * 16
        + 16
    )
    operator_aux_metadata_bytes = (
        config.num_hidden_layers * operator_aux_bytes_per_layer
    )
    embedding_bytes = config.vocab_size * config.hidden_size * 2
    estimated_weight_bytes = (
        packed_w4_bytes
        + projection_metadata_bytes
        + rmsnorm_metadata_bytes
        + operator_aux_metadata_bytes
        + embedding_bytes
    )
    kv_payload_bytes = (
        2
        * config.num_key_value_heads
        * head_dim
        * PRECISION["kv_cache_bits"]
        // 8
    )
    kv_scale_bytes = 2 * config.num_key_value_heads * 4
    kv_bytes = kv_payload_bytes + kv_scale_bytes
    maximum_kv_bytes = (
        config.num_hidden_layers * config.max_position_embeddings * kv_bytes
    )
    return {
        "head_dim": head_dim,
        "query_heads_per_kv_head": (
            config.num_attention_heads // config.num_key_value_heads
        ),
        "kv_width_elements": kv_width,
        "transformer_linear_weight_elements": transformer_weights,
        "packed_linear_weight_elements": packed_weights,
        "packed_w4_bytes": packed_w4_bytes,
        "projection_output_rows": projection_rows,
        "projection_metadata_bytes": projection_metadata_bytes,
        "rmsnorm_metadata_bytes": rmsnorm_metadata_bytes,
        "operator_aux_metadata_bytes": operator_aux_metadata_bytes,
        "bf16_embedding_bytes": embedding_bytes,
        "estimated_weight_bytes": estimated_weight_bytes,
        "kv_payload_bytes_per_token_per_layer": kv_payload_bytes,
        "kv_scale_metadata_bytes_per_token_per_layer": kv_scale_bytes,
        "kv_bytes_per_token_per_layer": kv_bytes,
        "maximum_kv_bytes": maximum_kv_bytes,
        "maximum_weight_plus_kv_bytes": estimated_weight_bytes + maximum_kv_bytes,
    }


def build_descriptor(config: QwenConfig) -> dict[str, Any]:
    runtime_compatible = config.model_id == "qwen2.5-0.5b"
    return {
        "schema_version": SCHEMA_VERSION,
        "contract_id": f"ace2-{config.model_id}-{PRECISION_MODE}-v1",
        "model_id": config.model_id,
        "source": {
            "repository": config.repository,
            "revision": config.revision,
            "config_path": "config.json",
            "config_url": (
                f"https://huggingface.co/{config.repository}/"
                f"blob/{config.revision}/config.json"
            ),
        },
        "dimensions": {
            field: getattr(config, field)
            for field in DIMENSION_FIELDS
        }
        | {"tie_word_embeddings": config.tie_word_embeddings},
        "precision": copy.deepcopy(PRECISION),
        "weight_layout": copy.deepcopy(WEIGHT_LAYOUT),
        "hardware": copy.deepcopy(HARDWARE),
        "compatibility": {
            "validation_scope": (
                "existing_0p5b_package_runtime_preflight"
                if runtime_compatible
                else "structural_only_no_rtl_execution"
            ),
            "existing_ace2_runtime_preflight": runtime_compatible,
            "full_rtl_execution_validated": False,
        },
        "derived": derive(config),
        "estimate_scope": {
            "weight": (
                "packed W4 transformer and LM-head matrices, 16-byte per-output "
                "projection records, RMSNorm/operator metadata, and one BF16 "
                "input embedding table"
            ),
            "kv_cache": (
                "all layers at max_position_embeddings with signed-int8 K/V "
                "payload and one Scale32 value per K/V head per token"
            ),
            "units": "bytes",
        },
    }


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def _record(value: Any, path: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{path} must be an object")
    return value


def _expect_keys(
    value: Mapping[str, Any],
    expected: set[str],
    path: str,
) -> None:
    actual = set(value)
    _require(actual == expected, f"{path} fields differ: {sorted(actual ^ expected)}")


def _integer(value: Any, path: str) -> int:
    _require(type(value) is int and value > 0, f"{path} must be a positive integer")
    return value


def validate_descriptor(value: Any) -> dict[str, Any]:
    descriptor = _record(value, "descriptor")
    _expect_keys(
        descriptor,
        {
            "schema_version",
            "contract_id",
            "model_id",
            "source",
            "dimensions",
            "precision",
            "weight_layout",
            "hardware",
            "compatibility",
            "derived",
            "estimate_scope",
        },
        "descriptor",
    )
    _require(
        descriptor["schema_version"] == SCHEMA_VERSION,
        "schema_version differs",
    )
    model_id = descriptor["model_id"]
    _require(isinstance(model_id, str) and model_id in CONFIG_BY_ID, "unsupported model_id")
    config = CONFIG_BY_ID[model_id]
    _require(
        descriptor["contract_id"] == f"ace2-{model_id}-{PRECISION_MODE}-v1",
        "contract_id differs",
    )

    dimensions = _record(descriptor["dimensions"], "dimensions")
    _expect_keys(
        dimensions,
        set(DIMENSION_FIELDS) | {"tie_word_embeddings"},
        "dimensions",
    )
    values = {
        field: _integer(dimensions[field], f"dimensions.{field}")
        for field in DIMENSION_FIELDS
    }
    _require(
        values["hidden_size"] % values["num_attention_heads"] == 0,
        "hidden_size must be divisible by num_attention_heads",
    )
    _require(
        values["num_attention_heads"] % values["num_key_value_heads"] == 0,
        "num_attention_heads must be divisible by num_key_value_heads",
    )
    _require(
        values["hidden_size"] % HARDWARE["compute_tile"]["projection_k"] == 0,
        "hidden_size must align to projection_k",
    )
    _require(
        values["intermediate_size"] % HARDWARE["compute_tile"]["projection_n"] == 0,
        "intermediate_size must align to projection_n",
    )
    _require(
        values["vocab_size"] % HARDWARE["compute_tile"]["lm_head_vocab"] == 0,
        "vocab_size must align to lm_head_vocab",
    )
    _require(
        type(dimensions["tie_word_embeddings"]) is bool,
        "dimensions.tie_word_embeddings must be boolean",
    )
    for field in DIMENSION_FIELDS:
        _require(
            dimensions[field] == getattr(config, field),
            f"dimensions.{field} differs from authoritative Qwen config",
        )
    _require(
        dimensions["tie_word_embeddings"] == config.tie_word_embeddings,
        "dimensions.tie_word_embeddings differs from authoritative Qwen config",
    )

    source = _record(descriptor["source"], "source")
    expected_source = build_descriptor(config)["source"]
    _expect_keys(source, set(expected_source), "source")
    _require(dict(source) == expected_source, "source identity differs")

    precision = _record(descriptor["precision"], "precision")
    _expect_keys(precision, set(PRECISION), "precision")
    _require(
        precision["supported_modes"] == [PRECISION_MODE],
        "unsupported precision mode",
    )
    _require(dict(precision) == PRECISION, "precision contract differs")

    layout = _record(descriptor["weight_layout"], "weight_layout")
    _expect_keys(layout, set(WEIGHT_LAYOUT), "weight_layout")
    _require(dict(layout) == WEIGHT_LAYOUT, "weight layout differs")

    hardware = _record(descriptor["hardware"], "hardware")
    _expect_keys(hardware, set(HARDWARE), "hardware")
    _require(dict(hardware) == HARDWARE, "compute-tile or memory-channel contract differs")

    compatibility = _record(descriptor["compatibility"], "compatibility")
    expected_compatibility = build_descriptor(config)["compatibility"]
    _expect_keys(compatibility, set(expected_compatibility), "compatibility")
    _require(
        dict(compatibility) == expected_compatibility,
        "compatibility claim differs",
    )

    derived = _record(descriptor["derived"], "derived")
    expected_derived = derive(config)
    _expect_keys(derived, set(DERIVED_FIELDS), "derived")
    for field in DERIVED_FIELDS:
        _integer(derived[field], f"derived.{field}")
    _require(dict(derived) == expected_derived, "derived memory or geometry estimate differs")

    estimate_scope = _record(descriptor["estimate_scope"], "estimate_scope")
    expected_scope = build_descriptor(config)["estimate_scope"]
    _expect_keys(estimate_scope, set(expected_scope), "estimate_scope")
    _require(dict(estimate_scope) == expected_scope, "estimate scope differs")
    return dict(descriptor)


def load_descriptor(model_id: str) -> dict[str, Any]:
    path = descriptor_path(model_id)
    if not path.is_file():
        raise ContractError(f"missing descriptor: {path.relative_to(ROOT)}")
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ContractError(f"descriptor is not valid JSON: {model_id}") from error
    validated = validate_descriptor(value)
    expected = canonical_bytes(build_descriptor(CONFIG_BY_ID[model_id]))
    _require(raw == expected, f"descriptor is not canonical or generated: {model_id}")
    return validated


def runtime_preflight(
    *,
    model_id: str,
    embedding_shape: list[int],
    max_sequence_positions: int,
    kv_bytes_per_token_per_layer: int,
) -> dict[str, Any]:
    descriptor = load_descriptor(model_id)
    dimensions = descriptor["dimensions"]
    derived = descriptor["derived"]
    compatibility = descriptor["compatibility"]
    _require(
        compatibility["existing_ace2_runtime_preflight"] is True,
        f"{model_id} has structural compatibility only",
    )
    _require(type(embedding_shape) is list, "runtime embedding shape must be a list")
    _require(len(embedding_shape) == 2, "runtime embedding shape must have two dimensions")
    for index, value in enumerate(embedding_shape):
        _integer(value, f"runtime embedding_shape[{index}]")
    _integer(max_sequence_positions, "runtime max_sequence_positions")
    _integer(
        kv_bytes_per_token_per_layer,
        "runtime kv_bytes_per_token_per_layer",
    )
    _require(
        embedding_shape == [dimensions["vocab_size"], dimensions["hidden_size"]],
        "runtime embedding shape differs from model hardware contract",
    )
    _require(
        max_sequence_positions == dimensions["max_position_embeddings"],
        "runtime sequence capacity differs from model hardware contract",
    )
    _require(
        kv_bytes_per_token_per_layer == derived["kv_bytes_per_token_per_layer"],
        "runtime KV stride differs from model hardware contract",
    )
    path = descriptor_path(model_id)
    return {
        "status": "PASS_MODEL_HARDWARE_CONTRACT",
        "model_id": model_id,
        "descriptor": {
            "artifact": path.relative_to(ROOT).as_posix(),
            "sha256": sha256_bytes(path.read_bytes()),
        },
        "geometry": {
            "hidden_size": dimensions["hidden_size"],
            "head_dim": derived["head_dim"],
            "attention_heads": dimensions["num_attention_heads"],
            "kv_heads": dimensions["num_key_value_heads"],
            "max_sequence_positions": dimensions["max_position_embeddings"],
            "kv_bytes_per_token_per_layer": derived[
                "kv_bytes_per_token_per_layer"
            ],
        },
        "precision_mode": descriptor["precision"]["default_mode"],
        "validation_scope": compatibility["validation_scope"],
        "claims": {
            "package_runtime_compatibility": True,
            "full_model_rtl_execution": False,
            "timing_area_or_power": False,
        },
    }


def write_artifacts() -> None:
    SCHEMA_PATH.parent.mkdir(parents=True, exist_ok=True)
    DESCRIPTOR_DIR.mkdir(parents=True, exist_ok=True)
    SCHEMA_PATH.write_bytes(canonical_bytes(SCHEMA))
    for config in QWEN_CONFIGS:
        descriptor_path(config.model_id).write_bytes(
            canonical_bytes(build_descriptor(config))
        )


def validate_repository() -> dict[str, Any]:
    if not SCHEMA_PATH.is_file():
        raise ContractError(f"missing schema: {SCHEMA_PATH.relative_to(ROOT)}")
    schema_raw = SCHEMA_PATH.read_bytes()
    _require(
        schema_raw == canonical_bytes(SCHEMA),
        "model hardware contract schema is stale or noncanonical",
    )
    records = []
    for config in QWEN_CONFIGS:
        descriptor = load_descriptor(config.model_id)
        path = descriptor_path(config.model_id)
        records.append(
            {
                "model_id": config.model_id,
                "artifact": path.relative_to(ROOT).as_posix(),
                "sha256": sha256_bytes(path.read_bytes()),
                "validation_scope": descriptor["compatibility"][
                    "validation_scope"
                ],
                "dimensions": {
                    field: descriptor["dimensions"][field]
                    for field in DIMENSION_FIELDS
                },
                "estimated_memory": {
                    "weight_bytes": descriptor["derived"][
                        "estimated_weight_bytes"
                    ],
                    "maximum_kv_bytes": descriptor["derived"]["maximum_kv_bytes"],
                    "maximum_weight_plus_kv_bytes": descriptor["derived"][
                        "maximum_weight_plus_kv_bytes"
                    ],
                },
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "classification": "ace2_model_hardware_contract_validation",
        "status": "PASS",
        "schema": {
            "artifact": SCHEMA_PATH.relative_to(ROOT).as_posix(),
            "sha256": sha256_bytes(schema_raw),
        },
        "descriptors": records,
        "claims": {
            "descriptor_validation": True,
            "existing_0p5b_package_runtime_compatibility": True,
            "larger_model_structural_compatibility": True,
            "larger_model_rtl_execution": False,
            "full_model_rtl_execution_performed": False,
            "timing_area_or_power": False,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--write",
        action="store_true",
        help="write the canonical schema and four generated descriptors",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="validate checked-in artifacts and emit a deterministic report",
    )
    args = parser.parse_args(argv)
    if args.write:
        write_artifacts()
    if not args.write or args.check:
        try:
            report = validate_repository()
        except (ContractError, OSError) as error:
            print(f"FAIL: {error}", file=sys.stderr)
            return 1
        sys.stdout.buffer.write(canonical_bytes(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
