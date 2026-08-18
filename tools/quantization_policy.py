#!/usr/bin/env python3
"""Generate and validate ACE-2 mixed-precision quantization plans."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

if __package__:
    from tools import model_hardware_contract as hardware
else:
    import model_hardware_contract as hardware


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT / "design/quantization_policy.schema.json"
SCHEMA_VERSION = 1
POLICY_IDS = ("w4a8", "w8a8", "mixed_w4a8_a16_bf16")
OPERATOR_CLASSES = (
    "attention_projection",
    "mlp_projection",
    "rmsnorm",
    "softmax",
    "residual_accumulation",
    "kv_cache",
    "lm_head",
)
WEIGHT_FORMATS = (
    "none",
    "int4_twos_complement",
    "int8_twos_complement",
    "bf16",
    "bf16_parameter",
)
ACTIVATION_FORMATS = ("int8_scale32", "int16", "bf16")
ACCUMULATOR_FORMATS = ("none", "int32", "bf16")
SUPPORT_STATUSES = (
    "implemented_rtl_format",
    "structural_candidate_no_rtl_execution",
)
DEPLOYMENT_STATUSES = (
    "current_rtl_format",
    "structural_candidate_no_full_rtl_execution",
)


class QuantizationPolicyError(ValueError):
    """Raised when a quantization plan is unsupported or inconsistent."""


def _strict_object(properties: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": list(properties),
        "properties": properties,
    }


def _positive_integer() -> dict[str, Any]:
    return {"type": "integer", "minimum": 1}


OPERATOR_SCHEMA = _strict_object(
    {
        "operator_class": {"enum": list(OPERATOR_CLASSES)},
        "weight_format": {"enum": list(WEIGHT_FORMATS)},
        "activation_format": {"enum": list(ACTIVATION_FORMATS)},
        "accumulator_format": {"enum": list(ACCUMULATOR_FORMATS)},
        "hardware_support": {"enum": list(SUPPORT_STATUSES)},
    }
)
BANDWIDTH_SCHEMA = _strict_object(
    {
        "weight_stream_bytes": _positive_integer(),
        "embedding_lookup_bytes": _positive_integer(),
        "kv_write_bytes": _positive_integer(),
        "kv_read_bytes": _positive_integer(),
        "total_stream_bytes": _positive_integer(),
    }
)
ESTIMATE_SCHEMA = _strict_object(
    {
        "units": {"const": "bytes"},
        "weight_memory_bytes": _positive_integer(),
        "kv_cache_bytes_per_token_per_layer": _positive_integer(),
        "maximum_kv_cache_bytes": _positive_integer(),
        "maximum_weight_plus_kv_bytes": _positive_integer(),
        "bandwidth_per_decode_token_at_max_context": BANDWIDTH_SCHEMA,
    }
)
CLAIMS_SCHEMA = _strict_object(
    {
        "full_model_rtl_execution_validated": {"const": False},
        "w8a8_rtl_execution": {"const": False},
        "a16_bf16_rtl_execution": {"const": False},
        "timing_area_or_power": {"const": False},
    }
)
SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": (
        "https://github.com/Argus-AiTeam/ace-2/"
        "design/quantization_policy.schema.json"
    ),
    "title": "ACE-2 Quantization Policy Plan",
    **_strict_object(
        {
            "schema_version": {"const": SCHEMA_VERSION},
            "plan_id": {"type": "string", "minLength": 1},
            "policy_id": {"enum": list(POLICY_IDS)},
            "model_contract": _strict_object(
                {
                    "model_id": {
                        "enum": list(hardware.CONFIG_BY_ID),
                    },
                    "contract_id": {"type": "string", "minLength": 1},
                    "artifact": {"type": "string", "minLength": 1},
                    "sha256": {
                        "type": "string",
                        "pattern": "^[0-9a-f]{64}$",
                    },
                }
            ),
            "deployment_status": {"enum": list(DEPLOYMENT_STATUSES)},
            "operators": {
                "type": "array",
                "minItems": len(OPERATOR_CLASSES),
                "maxItems": len(OPERATOR_CLASSES),
                "items": OPERATOR_SCHEMA,
            },
            "estimates": ESTIMATE_SCHEMA,
            "claims": CLAIMS_SCHEMA,
        }
    ),
}


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise QuantizationPolicyError(message)


def _record(value: Any, path: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{path} must be an object")
    return value


def _expect_keys(value: Mapping[str, Any], expected: set[str], path: str) -> None:
    actual = set(value)
    _require(actual == expected, f"{path} fields differ: {sorted(actual ^ expected)}")


def _validate_schema_types(
    value: Any,
    schema: Mapping[str, Any],
    path: str,
) -> None:
    schema_type = schema.get("type")
    if schema_type == "object":
        _require(isinstance(value, Mapping), f"{path} must be an object")
    elif schema_type == "array":
        _require(type(value) is list, f"{path} must be an array")
    elif schema_type == "string":
        _require(type(value) is str, f"{path} must be a string")
    elif schema_type == "integer":
        _require(type(value) is int, f"{path} must be an integer")
    elif schema_type == "number":
        _require(type(value) in (int, float), f"{path} must be a number")
    elif schema_type == "boolean":
        _require(type(value) is bool, f"{path} must be a boolean")
    elif schema_type == "null":
        _require(value is None, f"{path} must be null")

    if "const" in schema:
        expected = schema["const"]
        _require(
            type(value) is type(expected) and value == expected,
            f"{path} differs from schema constant",
        )
    if "enum" in schema:
        _require(
            any(type(value) is type(item) and value == item for item in schema["enum"]),
            f"{path} is not a schema enum value",
        )

    if schema_type == "object" and isinstance(value, Mapping):
        for key, property_schema in schema.get("properties", {}).items():
            if key in value:
                _validate_schema_types(value[key], property_schema, f"{path}.{key}")
    elif schema_type == "array" and type(value) is list:
        item_schema = schema.get("items")
        if item_schema is not None:
            for index, item in enumerate(value):
                _validate_schema_types(item, item_schema, f"{path}[{index}]")


def _operator(
    operator_class: str,
    weight_format: str,
    activation_format: str,
    accumulator_format: str,
    hardware_support: str,
) -> dict[str, str]:
    return {
        "operator_class": operator_class,
        "weight_format": weight_format,
        "activation_format": activation_format,
        "accumulator_format": accumulator_format,
        "hardware_support": hardware_support,
    }


def _operators(policy_id: str) -> list[dict[str, str]]:
    implemented = "implemented_rtl_format"
    candidate = "structural_candidate_no_rtl_execution"
    if policy_id == "w4a8":
        return [
            _operator("attention_projection", "int4_twos_complement", "int8_scale32", "int32", implemented),
            _operator("mlp_projection", "int4_twos_complement", "int8_scale32", "int32", implemented),
            _operator("rmsnorm", "bf16_parameter", "int8_scale32", "int32", implemented),
            _operator("softmax", "none", "int8_scale32", "int32", implemented),
            _operator("residual_accumulation", "none", "int8_scale32", "int32", implemented),
            _operator("kv_cache", "none", "int8_scale32", "none", implemented),
            _operator("lm_head", "int4_twos_complement", "int8_scale32", "int32", implemented),
        ]
    if policy_id == "w8a8":
        return [
            _operator("attention_projection", "int8_twos_complement", "int8_scale32", "int32", candidate),
            _operator("mlp_projection", "int8_twos_complement", "int8_scale32", "int32", candidate),
            _operator("rmsnorm", "bf16_parameter", "int8_scale32", "int32", implemented),
            _operator("softmax", "none", "int8_scale32", "int32", implemented),
            _operator("residual_accumulation", "none", "int8_scale32", "int32", implemented),
            _operator("kv_cache", "none", "int8_scale32", "none", implemented),
            _operator("lm_head", "int8_twos_complement", "int8_scale32", "int32", candidate),
        ]
    if policy_id == "mixed_w4a8_a16_bf16":
        return [
            _operator("attention_projection", "int4_twos_complement", "int8_scale32", "int32", implemented),
            _operator("mlp_projection", "int4_twos_complement", "int8_scale32", "int32", implemented),
            _operator("rmsnorm", "bf16_parameter", "bf16", "bf16", candidate),
            _operator("softmax", "none", "bf16", "bf16", candidate),
            _operator("residual_accumulation", "none", "bf16", "bf16", candidate),
            _operator("kv_cache", "none", "int16", "none", candidate),
            _operator("lm_head", "bf16", "bf16", "bf16", candidate),
        ]
    raise QuantizationPolicyError(f"unsupported policy_id: {policy_id}")


def _estimates(descriptor: Mapping[str, Any], policy_id: str) -> dict[str, Any]:
    dimensions = descriptor["dimensions"]
    derived = descriptor["derived"]
    transformer_elements = derived["transformer_linear_weight_elements"]
    lm_head_elements = dimensions["vocab_size"] * dimensions["hidden_size"]
    projection_record_bytes = descriptor["weight_layout"]["projection_record_bytes"]
    transformer_projection_rows = (
        derived["projection_output_rows"] - dimensions["vocab_size"]
    )
    fixed_bytes = (
        derived["rmsnorm_metadata_bytes"]
        + derived["operator_aux_metadata_bytes"]
        + derived["bf16_embedding_bytes"]
    )

    if policy_id == "w4a8":
        matrix_bytes = (transformer_elements + lm_head_elements + 1) // 2
        projection_metadata_bytes = derived["projection_metadata_bytes"]
        kv_bytes_per_layer = derived["kv_bytes_per_token_per_layer"]
    elif policy_id == "w8a8":
        matrix_bytes = transformer_elements + lm_head_elements
        projection_metadata_bytes = derived["projection_metadata_bytes"]
        kv_bytes_per_layer = derived["kv_bytes_per_token_per_layer"]
    elif policy_id == "mixed_w4a8_a16_bf16":
        matrix_bytes = (transformer_elements + 1) // 2 + 2 * lm_head_elements
        projection_metadata_bytes = (
            transformer_projection_rows * projection_record_bytes
        )
        kv_bytes_per_layer = 4 * derived["kv_width_elements"]
    else:
        raise QuantizationPolicyError(f"unsupported policy_id: {policy_id}")

    weight_bytes = matrix_bytes + projection_metadata_bytes + fixed_bytes
    if policy_id == "w4a8":
        _require(
            weight_bytes == derived["estimated_weight_bytes"],
            "W4A8 weight estimate differs from the published model contract",
        )
    maximum_kv_bytes = (
        dimensions["num_hidden_layers"]
        * dimensions["max_position_embeddings"]
        * kv_bytes_per_layer
    )
    weight_stream_bytes = weight_bytes - derived["bf16_embedding_bytes"]
    embedding_lookup_bytes = 2 * dimensions["hidden_size"]
    kv_write_bytes = dimensions["num_hidden_layers"] * kv_bytes_per_layer
    kv_read_bytes = maximum_kv_bytes
    return {
        "units": "bytes",
        "weight_memory_bytes": weight_bytes,
        "kv_cache_bytes_per_token_per_layer": kv_bytes_per_layer,
        "maximum_kv_cache_bytes": maximum_kv_bytes,
        "maximum_weight_plus_kv_bytes": weight_bytes + maximum_kv_bytes,
        "bandwidth_per_decode_token_at_max_context": {
            "weight_stream_bytes": weight_stream_bytes,
            "embedding_lookup_bytes": embedding_lookup_bytes,
            "kv_write_bytes": kv_write_bytes,
            "kv_read_bytes": kv_read_bytes,
            "total_stream_bytes": (
                weight_stream_bytes
                + embedding_lookup_bytes
                + kv_write_bytes
                + kv_read_bytes
            ),
        },
    }


def build_plan(model_id: str, policy_id: str) -> dict[str, Any]:
    _require(policy_id in POLICY_IDS, f"unsupported policy_id: {policy_id}")
    descriptor = hardware.load_descriptor(model_id)
    path = hardware.descriptor_path(model_id)
    return {
        "schema_version": SCHEMA_VERSION,
        "plan_id": f"ace2-{model_id}-{policy_id}-v1",
        "policy_id": policy_id,
        "model_contract": {
            "model_id": model_id,
            "contract_id": descriptor["contract_id"],
            "artifact": path.relative_to(ROOT).as_posix(),
            "sha256": sha256_bytes(path.read_bytes()),
        },
        "deployment_status": (
            "current_rtl_format"
            if policy_id == "w4a8"
            else "structural_candidate_no_full_rtl_execution"
        ),
        "operators": _operators(policy_id),
        "estimates": _estimates(descriptor, policy_id),
        "claims": {
            "full_model_rtl_execution_validated": False,
            "w8a8_rtl_execution": False,
            "a16_bf16_rtl_execution": False,
            "timing_area_or_power": False,
        },
    }


def validate_plan(value: Any) -> dict[str, Any]:
    plan = _record(value, "plan")
    _validate_schema_types(plan, SCHEMA, "plan")
    expected_fields = {
        "schema_version",
        "plan_id",
        "policy_id",
        "model_contract",
        "deployment_status",
        "operators",
        "estimates",
        "claims",
    }
    _expect_keys(plan, expected_fields, "plan")
    _require(plan["schema_version"] == SCHEMA_VERSION, "schema_version differs")
    policy_id = plan["policy_id"]
    _require(
        isinstance(policy_id, str) and policy_id in POLICY_IDS,
        "unsupported policy_id",
    )
    model_contract = _record(plan["model_contract"], "model_contract")
    _expect_keys(
        model_contract,
        {"model_id", "contract_id", "artifact", "sha256"},
        "model_contract",
    )
    model_id = model_contract["model_id"]
    _require(
        isinstance(model_id, str) and model_id in hardware.CONFIG_BY_ID,
        "unsupported model_id",
    )
    operators = plan["operators"]
    _require(type(operators) is list, "operators must be an array")
    _require(
        len(operators) == len(OPERATOR_CLASSES),
        "operator frontier is incomplete",
    )
    for index, operator_class in enumerate(OPERATOR_CLASSES):
        operator = _record(operators[index], f"operators[{index}]")
        _expect_keys(
            operator,
            {
                "operator_class",
                "weight_format",
                "activation_format",
                "accumulator_format",
                "hardware_support",
            },
            f"operators[{index}]",
        )
        _require(
            operator["operator_class"] == operator_class,
            "operator frontier order differs",
        )
    expected = build_plan(model_id, policy_id)
    _require(dict(plan) == expected, "quantization plan is inconsistent with generated policy")
    return dict(plan)


def deployment_preflight(
    value: Any,
    *,
    require_current_rtl: bool,
) -> dict[str, Any]:
    plan = validate_plan(value)
    descriptor = hardware.load_descriptor(plan["model_contract"]["model_id"])
    if require_current_rtl:
        _require(
            plan["policy_id"] == "w4a8",
            "current RTL preflight requires the W4A8 policy",
        )
        _require(
            descriptor["compatibility"]["existing_ace2_runtime_preflight"] is True,
            "model has structural compatibility only",
        )
        _require(
            all(
                operator["hardware_support"] == "implemented_rtl_format"
                for operator in plan["operators"]
            ),
            "plan contains operator formats not implemented in current RTL",
        )
    return {
        "schema_version": 1,
        "classification": "ace2_quantization_policy_preflight",
        "status": (
            "PASS_CURRENT_RTL_QUANTIZATION_POLICY"
            if require_current_rtl
            else "PASS_STRUCTURAL_QUANTIZATION_POLICY"
        ),
        "plan_id": plan["plan_id"],
        "model_id": plan["model_contract"]["model_id"],
        "policy_id": plan["policy_id"],
        "plan_sha256": sha256_bytes(canonical_bytes(plan)),
        "deployment_status": plan["deployment_status"],
        "claims": plan["claims"],
    }


def runtime_preflight(
    *,
    model_id: str,
    policy_id: str,
    require_current_rtl: bool,
) -> dict[str, Any]:
    return deployment_preflight(
        build_plan(model_id, policy_id),
        require_current_rtl=require_current_rtl,
    )


def pack_signed_int4(values: Sequence[int]) -> bytes:
    packed = bytearray()
    for index in range(0, len(values), 2):
        low = values[index]
        _require(type(low) is int and -8 <= low <= 7, "signed-int4 value is out of range")
        high = values[index + 1] if index + 1 < len(values) else 0
        _require(type(high) is int and -8 <= high <= 7, "signed-int4 value is out of range")
        packed.append((low & 0xF) | ((high & 0xF) << 4))
    return bytes(packed)


def unpack_signed_int4(packed: bytes, element_count: int) -> list[int]:
    _require(type(packed) is bytes, "packed signed-int4 payload must be bytes")
    _require(
        type(element_count) is int and element_count >= 0,
        "element_count must be a nonnegative integer",
    )
    _require(
        len(packed) == (element_count + 1) // 2,
        "packed signed-int4 payload length differs",
    )
    if element_count % 2 and packed:
        _require(packed[-1] >> 4 == 0, "odd signed-int4 payload has nonzero padding")
    values = []
    for byte in packed:
        for nibble in (byte & 0xF, byte >> 4):
            values.append(nibble - 16 if nibble >= 8 else nibble)
    return values[:element_count]


def quantize_signed_int4(values: Sequence[float], scale: float) -> list[int]:
    _require(
        isinstance(scale, (int, float))
        and not isinstance(scale, bool)
        and math.isfinite(scale)
        and scale > 0,
        "scale must be finite and positive",
    )
    quantized = []
    for value in values:
        _require(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value),
            "numeric reference values must be finite",
        )
        rounded = round(value / scale)
        quantized.append(max(-8, min(7, rounded)))
    return quantized


def numeric_reference() -> dict[str, Any]:
    values = [-1.0, -0.5, 0.0, 0.5, 0.875]
    scale = 0.125
    quantized = quantize_signed_int4(values, scale)
    packed = pack_signed_int4(quantized)
    unpacked = unpack_signed_int4(packed, len(values))
    reconstructed = [value * scale for value in unpacked]
    _require(unpacked == quantized, "signed-int4 packing round trip differs")
    maximum_error = max(
        abs(reference - actual)
        for reference, actual in zip(values, reconstructed, strict=True)
    )
    return {
        "format": "signed_int4_twos_complement",
        "packing": "even_input_low_nibble_odd_input_high_nibble",
        "scale": scale,
        "zero_point": 0,
        "input": values,
        "quantized": quantized,
        "packed_hex": packed.hex(),
        "reconstructed": reconstructed,
        "maximum_absolute_error": maximum_error,
    }


def write_artifacts() -> None:
    SCHEMA_PATH.parent.mkdir(parents=True, exist_ok=True)
    SCHEMA_PATH.write_bytes(canonical_bytes(SCHEMA))


def validate_repository() -> dict[str, Any]:
    _require(SCHEMA_PATH.is_file(), "quantization policy schema is missing")
    schema_raw = SCHEMA_PATH.read_bytes()
    _require(
        schema_raw == canonical_bytes(SCHEMA),
        "quantization policy schema is stale or noncanonical",
    )
    plans = []
    for model_id in hardware.CONFIG_BY_ID:
        for policy_id in POLICY_IDS:
            plan = validate_plan(build_plan(model_id, policy_id))
            plans.append(
                {
                    "model_id": model_id,
                    "policy_id": policy_id,
                    "plan_sha256": sha256_bytes(canonical_bytes(plan)),
                    "deployment_status": plan["deployment_status"],
                    "estimates": plan["estimates"],
                }
            )
    return {
        "schema_version": SCHEMA_VERSION,
        "classification": "ace2_quantization_policy_validation",
        "status": "PASS",
        "schema": {
            "artifact": SCHEMA_PATH.relative_to(ROOT).as_posix(),
            "sha256": sha256_bytes(schema_raw),
        },
        "plans": plans,
        "numeric_reference": numeric_reference(),
        "claims": {
            "deterministic_plan_generation": True,
            "current_w4a8_estimates_preserved": True,
            "w8a8_rtl_execution": False,
            "a16_bf16_rtl_execution": False,
            "full_model_rtl_execution_performed": False,
            "timing_area_or_power": False,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true", help="write the canonical schema")
    parser.add_argument("--check", action="store_true", help="validate all plans")
    parser.add_argument("--model", choices=list(hardware.CONFIG_BY_ID))
    parser.add_argument("--policy", choices=list(POLICY_IDS))
    parser.add_argument(
        "--preflight-current-rtl",
        action="store_true",
        help="require the selected plan to match the current RTL format",
    )
    args = parser.parse_args(argv)
    try:
        _require(
            (args.model is None) == (args.policy is None),
            "--model and --policy must be provided together",
        )
        _require(
            not args.preflight_current_rtl or args.model is not None,
            "--preflight-current-rtl requires --model and --policy",
        )
        if args.write:
            write_artifacts()
        if args.model is not None:
            plan = build_plan(args.model, args.policy)
            report = (
                deployment_preflight(plan, require_current_rtl=True)
                if args.preflight_current_rtl
                else plan
            )
        elif not args.write or args.check:
            report = validate_repository()
        else:
            return 0
    except (QuantizationPolicyError, hardware.ContractError, OSError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1
    sys.stdout.buffer.write(canonical_bytes(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
