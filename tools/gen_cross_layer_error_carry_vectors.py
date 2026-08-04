#!/usr/bin/env python3
"""Generate deterministic RTL vectors for the cross-layer error-carry cores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ace2_cross_layer_error_carry_reference import (
    accepted_consumer_completion_tag,
    pack_scale32,
    producer_lane,
    rmsnorm,
    workspace_proof,
)
from ace2_rmsnorm_reference import derive_scaled_gains_q8


ROOT = Path(__file__).resolve().parents[1]
OUT_JSON = ROOT / "verification/generated/cross_layer_error_carry_vectors.json"
OUT_SVH = ROOT / "verification/generated/cross_layer_error_carry_vectors.svh"
UNIT = pack_scale32(0x8000, 0)
HALF = pack_scale32(0x8000, -1)
QUARTER = pack_scale32(0x8000, -2)


PRODUCER_INPUTS = [
    (1, 2, UNIT, UNIT, UNIT, "unit_exact"),
    (1, 0, HALF, UNIT, UNIT, "positive_half_tie_even"),
    (3, 0, HALF, UNIT, UNIT, "positive_one_half_tie_even"),
    (-1, 0, HALF, UNIT, UNIT, "negative_half_tie_even"),
    (-3, 0, HALF, UNIT, UNIT, "negative_one_half_tie_even"),
    (126, 1, UNIT, HALF, UNIT, "mixed_scale"),
    (7, -3, QUARTER, HALF, UNIT, "exponent_alignment"),
    (127, 0, UNIT, UNIT, UNIT, "positive_limit"),
    (-128, 0, UNIT, UNIT, UNIT, "negative_limit"),
]


def payload() -> dict[str, object]:
    producer = []
    for accumulator, residual, accumulator_scale, residual_scale, destination_scale, name in PRODUCER_INPUTS:
        result = producer_lane(accumulator, residual, accumulator_scale, residual_scale, destination_scale)
        producer.append({
            "name": name,
            "accumulator_s32": accumulator,
            "residual_s8": residual,
            "accumulator_scale32": accumulator_scale,
            "residual_scale32": residual_scale,
            "destination_scale32": destination_scale,
            "hidden_s8": result.hidden_s8,
            "carry_s16_q15": result.carry_s16_q15,
            "numerator_s96": result.numerator_s96,
            "denominator_u64": result.denominator_u64,
            "common_exponent": result.common_exponent,
            "latency_cycles": result.latency_cycles,
        })
    rms_hidden = [1, -2, 3, -4]
    rms_carry = [16384, -16384, 0, 8192]
    rms_output_scale = 1.0 / 32.0
    rms_weights = [1.0, 1.25, -0.75, 0.5]
    rms_scaled_gain = derive_scaled_gains_q8(rms_weights, rms_output_scale)
    rms_result = rmsnorm(rms_hidden, rms_carry, rms_scaled_gain)
    consumer_tag_cases = [
        {
            "name": "valid_consumer_tag_held_through_completion_backpressure",
            "accepted_completion_tag": 0x0ACE,
            "subsequent_completion_tags": [0xBEEF, 0x1234],
            "expected_completion_tag": accepted_consumer_completion_tag(
                True, True, 0x0ACE, (0xBEEF, 0x1234)
            ),
            "descriptor_error": False,
        },
        {
            "name": "invalid_consumer_start_reports_accepted_tag",
            "accepted_completion_tag": 0xDEAD,
            "subsequent_completion_tags": [0xCAFE],
            "expected_completion_tag": accepted_consumer_completion_tag(
                True, True, 0xDEAD, (0xCAFE,)
            ),
            "descriptor_error": True,
        },
    ]
    return {
        "schema_version": 1,
        "contract_id": "cross_layer_quantization_error_carry_final_output_v1",
        "implementation_provenance": {
            "role": "engineer",
            "status": "fresh_regeneration_from_frozen_contract",
            "supersedes": "unaccepted_planner_draft",
        },
        "producer_vectors": producer,
        "producer_overflow_vectors": [
            {"accumulator_s32": 128, "residual_s8": 0, "accumulator_scale32": UNIT,
             "residual_scale32": UNIT, "destination_scale32": UNIT, "name": "positive_hidden_overflow"},
            {"accumulator_s32": -129, "residual_s8": 0, "accumulator_scale32": UNIT,
             "residual_scale32": UNIT, "destination_scale32": UNIT, "name": "negative_hidden_overflow"},
        ],
        "invalid_scale32": 0x00007FFF,
        "consumer_completion_tag_cases": consumer_tag_cases,
        "rmsnorm_small": {
            "hidden_s8": rms_hidden,
            "carry_s16_q15": rms_carry,
            "output_scale": rms_output_scale,
            "weights": rms_weights,
            "scaled_gain_s16_q8": rms_scaled_gain,
            "outputs_s8": list(rms_result.outputs_s8),
            "reconstructed_q15": list(rms_result.reconstructed_q15),
            "sum_squares_q30": rms_result.sum_squares_q30,
            "mean_square_q30": rms_result.mean_square_q30,
            "root_q15": rms_result.root_q15,
            "inverse_q30": rms_result.inverse_q30,
            "saturation_seen": rms_result.saturation_seen,
        },
        "workspace_proof": workspace_proof(),
    }


def svh(value: dict[str, object]) -> str:
    def signed_literal(width: int, number: int) -> str:
        return f"-{width}'sd{abs(number)}" if number < 0 else f"{width}'sd{number}"

    producer = value["producer_vectors"]
    rms = value["rmsnorm_small"]
    consumer_tags = value["consumer_completion_tag_cases"]
    lines = [
        "// Generated by tools/gen_cross_layer_error_carry_vectors.py",
        f"localparam integer QECR_PRODUCER_VECTOR_COUNT = {len(producer)};",
    ]
    for index, vector in enumerate(producer):
        lines.extend([
            f"localparam logic signed [31:0] QECR_ACC_{index} = {signed_literal(32, vector['accumulator_s32'])};",
            f"localparam logic signed [7:0] QECR_RES_{index} = {signed_literal(8, vector['residual_s8'])};",
            f"localparam logic [31:0] QECR_AS_{index} = 32'h{vector['accumulator_scale32']:08x};",
            f"localparam logic [31:0] QECR_RS_{index} = 32'h{vector['residual_scale32']:08x};",
            f"localparam logic [31:0] QECR_DS_{index} = 32'h{vector['destination_scale32']:08x};",
            f"localparam logic signed [7:0] QECR_HIDDEN_{index} = {signed_literal(8, vector['hidden_s8'])};",
            f"localparam logic signed [15:0] QECR_CARRY_{index} = {signed_literal(16, vector['carry_s16_q15'])};",
        ])
    lines.append("localparam integer QECR_RMS_LANES = 4;")
    lines.extend([
        f"localparam logic [15:0] QECR_VALID_CONSUMER_ACCEPTED_TAG = 16'h{consumer_tags[0]['accepted_completion_tag']:04x};",
        f"localparam logic [15:0] QECR_VALID_CONSUMER_CHANGED_TAG = 16'h{consumer_tags[0]['subsequent_completion_tags'][0]:04x};",
        f"localparam logic [15:0] QECR_VALID_CONSUMER_EXPECTED_TAG = 16'h{consumer_tags[0]['expected_completion_tag']:04x};",
        f"localparam logic [15:0] QECR_INVALID_CONSUMER_ACCEPTED_TAG = 16'h{consumer_tags[1]['accepted_completion_tag']:04x};",
        f"localparam logic [15:0] QECR_INVALID_CONSUMER_CHANGED_TAG = 16'h{consumer_tags[1]['subsequent_completion_tags'][0]:04x};",
        f"localparam logic [15:0] QECR_INVALID_CONSUMER_EXPECTED_TAG = 16'h{consumer_tags[1]['expected_completion_tag']:04x};",
    ])
    for index in range(4):
        lines.extend([
            f"localparam logic signed [7:0] QECR_RMS_H_{index} = {signed_literal(8, rms['hidden_s8'][index])};",
            f"localparam logic signed [15:0] QECR_RMS_C_{index} = {signed_literal(16, rms['carry_s16_q15'][index])};",
            f"localparam logic signed [15:0] QECR_RMS_SCALED_G_{index} = {signed_literal(16, rms['scaled_gain_s16_q8'][index])};",
            f"localparam logic signed [7:0] QECR_RMS_Y_{index} = {signed_literal(8, rms['outputs_s8'][index])};",
        ])
    lines.extend([
        f"localparam logic [55:0] QECR_RMS_SUMSQ = 56'd{rms['sum_squares_q30']};",
        f"localparam logic [55:0] QECR_RMS_MEAN = 56'd{rms['mean_square_q30']};",
        f"localparam logic [23:0] QECR_RMS_ROOT = 24'd{rms['root_q15']};",
        f"localparam logic [45:0] QECR_RMS_INV = 46'd{rms['inverse_q30']};",
    ])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    value = payload()
    json_bytes = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    svh_bytes = svh(value).encode()
    if args.check:
        if not OUT_JSON.is_file() or OUT_JSON.read_bytes() != json_bytes:
            raise SystemExit("generated QECR vector JSON is stale")
        if not OUT_SVH.is_file() or OUT_SVH.read_bytes() != svh_bytes:
            raise SystemExit("generated QECR vector include is stale")
        print("ACE2_QECR_VECTOR_CHECK_PASS producer=9 rms_lanes=4 consumer_tags=2")
        return 0
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_bytes(json_bytes)
    OUT_SVH.write_bytes(svh_bytes)
    print("ACE2_QECR_VECTORS_GENERATED producer=9 rms_lanes=4 consumer_tags=2")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
