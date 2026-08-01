#!/usr/bin/env python3
"""Generate ACE-2 W4A8 projection RTL vectors from the Python reference."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ace2_projection_reference import (
    HIDDEN_SIZE,
    MLP_INTERMEDIATE_SIZE,
    PROJ_GROUPS_PER_WEIGHT_BEAT,
    PROJ_MAC_LANES,
    PROJ_MAX_GROUPS,
    PROJ_MAX_K,
    ProjectionCase,
    pack_int8,
    pack_meta,
    pack_w4,
    projection_groups,
    projection_weight_beats_per_output,
    projection_weight_bytes_per_output,
    reference_projection,
)
from ace2_rmsnorm_reference import derive_scaled_gains_q8, reference_rmsnorm


ROOT = Path(__file__).resolve().parents[1]
OUT_SVH = ROOT / "verification" / "generated" / "projection_vectors.svh"
OUT_JSON = ROOT / "verification" / "generated" / "projection_vectors.json"
RMSNORM_CONSUMER_OUTPUT_SCALE = 3.0 / 64.0


def _balanced_case() -> ProjectionCase:
    rows = 2
    activations = [
        [((row * 19 + index * 7 + 3) % 128) - 64 for index in range(HIDDEN_SIZE)]
        for row in range(rows)
    ]
    weights = [
        [((out_index * 5 + k_index * 3 + 1) % 5) - 2 for k_index in range(HIDDEN_SIZE)]
        for out_index in range(HIDDEN_SIZE)
    ]
    multipliers = [(out_index % 3) + 1 for out_index in range(HIDDEN_SIZE)]
    right_shifts = [9 + (out_index % 2) for out_index in range(HIDDEN_SIZE)]
    output_zero_points = [(out_index % 5) - 2 for out_index in range(HIDDEN_SIZE)]
    return ProjectionCase(
        "balanced_rows2",
        rows,
        HIDDEN_SIZE,
        activations,
        weights,
        multipliers,
        right_shifts,
        output_zero_points,
    )


def _saturation_case() -> ProjectionCase:
    rows = 1
    activations = [[127 if (index & 1) else -128 for index in range(HIDDEN_SIZE)]]
    weights = [
        [((out_index + k_index * 7) % 16) - 8 for k_index in range(HIDDEN_SIZE)]
        for out_index in range(HIDDEN_SIZE)
    ]
    multipliers = [16 + (out_index & 3) for out_index in range(HIDDEN_SIZE)]
    right_shifts = [3 for _ in range(HIDDEN_SIZE)]
    output_zero_points = [0 for _ in range(HIDDEN_SIZE)]
    return ProjectionCase(
        "saturation_edges",
        rows,
        HIDDEN_SIZE,
        activations,
        weights,
        multipliers,
        right_shifts,
        output_zero_points,
    )


def _mlp_gate_case() -> ProjectionCase:
    rows = 1
    activations = [[((index * 5 + 11) % 16) - 8 for index in range(HIDDEN_SIZE)]]
    weights = [
        [((out_index * 3 + k_index * 5 + 7) % 5) - 2 for k_index in range(HIDDEN_SIZE)]
        for out_index in range(MLP_INTERMEDIATE_SIZE)
    ]
    multipliers = [(out_index % 3) + 1 for out_index in range(MLP_INTERMEDIATE_SIZE)]
    right_shifts = [10 + (out_index & 1) for out_index in range(MLP_INTERMEDIATE_SIZE)]
    output_zero_points = [(out_index % 7) - 3 for out_index in range(MLP_INTERMEDIATE_SIZE)]
    return ProjectionCase(
        "mlp_gate_proj_rows1",
        rows,
        HIDDEN_SIZE,
        activations,
        weights,
        multipliers,
        right_shifts,
        output_zero_points,
    )


def _mlp_down_case() -> ProjectionCase:
    rows = 1
    activations = [[((index * 7 + 13) % 8) - 4 for index in range(MLP_INTERMEDIATE_SIZE)]]
    weights = [
        [((out_index * 7 + k_index * 5 + 3) % 5) - 2 for k_index in range(MLP_INTERMEDIATE_SIZE)]
        for out_index in range(HIDDEN_SIZE)
    ]
    multipliers = [(out_index & 1) + 1 for out_index in range(HIDDEN_SIZE)]
    right_shifts = [10 + (out_index & 1) for out_index in range(HIDDEN_SIZE)]
    output_zero_points = [(out_index % 5) - 2 for out_index in range(HIDDEN_SIZE)]
    return ProjectionCase(
        "mlp_down_proj_rows1",
        rows,
        MLP_INTERMEDIATE_SIZE,
        activations,
        weights,
        multipliers,
        right_shifts,
        output_zero_points,
    )


def _rmsnorm_consumer_case() -> ProjectionCase:
    activations = [((index * 29) % 255) - 127 for index in range(HIDDEN_SIZE)]
    weights = [0.875 + (index % 17) / 64.0 for index in range(HIDDEN_SIZE)]
    gains = derive_scaled_gains_q8(weights, RMSNORM_CONSUMER_OUTPUT_SCALE)
    rms_output = reference_rmsnorm(activations, gains).outputs
    output_count = 32
    weights = [
        [((out_index * 5 + k_index * 3 + 1) % 5) - 2 for k_index in range(HIDDEN_SIZE)]
        for out_index in range(output_count)
    ]
    return ProjectionCase(
        "rmsnorm_per_tensor_consumer",
        1,
        HIDDEN_SIZE,
        [rms_output],
        weights,
        [3 + (out_index % 3) for out_index in range(output_count)],
        [12 + (out_index & 1) for out_index in range(output_count)],
        [0] * output_count,
    )


def _cases() -> list[ProjectionCase]:
    return [
        _balanced_case(),
        _saturation_case(),
        _mlp_gate_case(),
        _mlp_down_case(),
        _rmsnorm_consumer_case(),
    ]


def _hex(value: int, width_bits: int) -> str:
    return f"{width_bits}'h{value & ((1 << width_bits) - 1):0{width_bits // 4}x}"


def _chunks(values: list[int], chunk_size: int) -> list[list[int]]:
    return [values[index : index + chunk_size] for index in range(0, len(values), chunk_size)]


def main() -> None:
    OUT_SVH.parent.mkdir(parents=True, exist_ok=True)
    cases = _cases()
    max_rows = max(case.rows for case in cases)
    max_input_beats = max(case.reduction_size // 16 for case in cases)
    case_outputs = [len(case.weights) for case in cases]
    case_output_beats = [(outputs + 15) // 16 for outputs in case_outputs]
    case_input_beats = [case.reduction_size // 16 for case in cases]
    case_groups = [projection_groups(case.reduction_size) for case in cases]
    case_weight_beats_per_output = [
        projection_weight_beats_per_output(case.reduction_size) for case in cases
    ]
    max_outputs = max(case_outputs)
    max_output_beats = max(case_output_beats)
    weight_offsets: list[int] = []
    meta_offsets: list[int] = []
    weight_offset = 0
    meta_offset = 0
    for outputs in case_outputs:
        weight_offsets.append(weight_offset)
        meta_offsets.append(meta_offset)
        weight_offset += outputs * case_weight_beats_per_output[len(weight_offsets) - 1]
        meta_offset += outputs
    total_weight_beats = weight_offset
    total_meta_beats = meta_offset
    records = []
    sv_lines = [
        "// Generated by tools/gen_projection_vectors.py; do not edit by hand.",
        f"localparam integer PROJ_CASE_COUNT = {len(cases)};",
        "localparam integer PROJ_CASE_BALANCED = 0;",
        "localparam integer PROJ_CASE_SATURATION = 1;",
        "localparam integer PROJ_CASE_MLP_GATE = 2;",
        "localparam integer PROJ_CASE_MLP_DOWN = 3;",
        "localparam integer PROJ_CASE_RMSNORM_CONSUMER = 4;",
        f"localparam integer PROJ_MAX_ROWS = {max_rows};",
        f"localparam integer PROJ_MAX_K = {PROJ_MAX_K};",
        f"localparam integer PROJ_MAX_GROUPS = {PROJ_MAX_GROUPS};",
        f"localparam integer PROJ_GROUP_INDEX_WIDTH = {PROJ_MAX_GROUPS.bit_length()};",
        f"localparam integer PROJ_BEATS = {max_input_beats};",
        f"localparam integer PROJ_INPUT_BEATS = {max_input_beats};",
        f"localparam integer PROJ_MAX_INPUT_BEATS = {max_input_beats};",
        f"localparam integer PROJ_MAX_OUTPUTS = {max_outputs};",
        f"localparam integer PROJ_MAX_OUTPUT_BEATS = {max_output_beats};",
        f"localparam integer PROJ_MAC_LANES = {PROJ_MAC_LANES};",
        f"localparam integer PROJ_GROUPS_PER_WEIGHT_BEAT = {PROJ_GROUPS_PER_WEIGHT_BEAT};",
        f"localparam integer PROJ_WEIGHT_BEATS_PER_OUTPUT = {projection_weight_beats_per_output(HIDDEN_SIZE)};",
        f"localparam integer PROJ_MAX_WEIGHT_BEATS_PER_OUTPUT = {max(case_weight_beats_per_output)};",
        f"localparam integer PROJ_TOTAL_WEIGHT_BEATS = {total_weight_beats};",
        f"localparam integer PROJ_TOTAL_META_BEATS = {total_meta_beats};",
        "reg [15:0] proj_case_rows [0:PROJ_CASE_COUNT-1];",
        "reg [15:0] proj_case_k [0:PROJ_CASE_COUNT-1];",
        "reg [15:0] proj_case_input_beats [0:PROJ_CASE_COUNT-1];",
        "reg [15:0] proj_case_groups [0:PROJ_CASE_COUNT-1];",
        "reg [15:0] proj_case_outputs [0:PROJ_CASE_COUNT-1];",
        "reg [15:0] proj_case_output_beats [0:PROJ_CASE_COUNT-1];",
        "reg [15:0] proj_case_weight_beats_per_output [0:PROJ_CASE_COUNT-1];",
        "reg [31:0] proj_case_weight_offset [0:PROJ_CASE_COUNT-1];",
        "reg [31:0] proj_case_meta_offset [0:PROJ_CASE_COUNT-1];",
        "reg proj_expected_saturation [0:PROJ_CASE_COUNT-1];",
        "reg [8*16-1:0] proj_input_beats [0:PROJ_CASE_COUNT*PROJ_MAX_ROWS*PROJ_MAX_INPUT_BEATS-1];",
        "reg [4*32-1:0] proj_weight_beats [0:PROJ_TOTAL_WEIGHT_BEATS-1];",
        "reg [8*16-1:0] proj_meta_beats [0:PROJ_TOTAL_META_BEATS-1];",
        "reg [8*16-1:0] proj_expected_beats [0:PROJ_CASE_COUNT*PROJ_MAX_ROWS*PROJ_MAX_OUTPUT_BEATS-1];",
        "initial begin",
    ]

    for case_index, case in enumerate(cases):
        result = reference_projection(case)
        output_count = case_outputs[case_index]
        output_beats = case_output_beats[case_index]
        input_beats = case_input_beats[case_index]
        weight_beats_per_output = case_weight_beats_per_output[case_index]
        input_digest = hashlib.sha256()
        weight_digest = hashlib.sha256()
        output_digest = hashlib.sha256()
        sv_lines.append(f"  // projection case {case_index}: {case.name}")
        sv_lines.append(f"  proj_case_rows[{case_index}] = 16'd{case.rows};")
        sv_lines.append(f"  proj_case_k[{case_index}] = 16'd{case.reduction_size};")
        sv_lines.append(f"  proj_case_input_beats[{case_index}] = 16'd{input_beats};")
        sv_lines.append(f"  proj_case_groups[{case_index}] = 16'd{case_groups[case_index]};")
        sv_lines.append(f"  proj_case_outputs[{case_index}] = 16'd{output_count};")
        sv_lines.append(f"  proj_case_output_beats[{case_index}] = 16'd{output_beats};")
        sv_lines.append(
            f"  proj_case_weight_beats_per_output[{case_index}] = 16'd{weight_beats_per_output};"
        )
        sv_lines.append(f"  proj_case_weight_offset[{case_index}] = 32'd{weight_offsets[case_index]};")
        sv_lines.append(f"  proj_case_meta_offset[{case_index}] = 32'd{meta_offsets[case_index]};")
        sv_lines.append(f"  proj_expected_saturation[{case_index}] = 1'b{1 if result.saturation_seen else 0};")

        for row in range(max_rows):
            act_values = case.activations[row] if row < case.rows else [0] * case.reduction_size
            expected_values = result.outputs[row] if row < case.rows else [0] * output_count
            input_digest.update(bytes(value & 0xFF for value in act_values))
            output_digest.update(bytes(value & 0xFF for value in expected_values))
            padded_act_values = act_values + [0] * ((max_input_beats * 16) - len(act_values))
            for beat_index, chunk in enumerate(_chunks(padded_act_values, 16)):
                flat = case_index * max_rows * max_input_beats + row * max_input_beats + beat_index
                sv_lines.append(f"  proj_input_beats[{flat}] = {_hex(pack_int8(chunk), 128)};")
            for beat_index, chunk in enumerate(_chunks(expected_values, 16)):
                flat = case_index * max_rows * max_output_beats + row * max_output_beats + beat_index
                sv_lines.append(f"  proj_expected_beats[{flat}] = {_hex(pack_int8(chunk), 128)};")

        for out_index in range(output_count):
            weight_digest.update(bytes(value & 0xF for value in case.weights[out_index]))
            for group_index, chunk in enumerate(_chunks(case.weights[out_index], 16)):
                flat = weight_offsets[case_index] + out_index * weight_beats_per_output + group_index
                sv_lines.append(f"  proj_weight_beats[{flat}] = {_hex(pack_w4(chunk), 128)};")
            meta_flat = meta_offsets[case_index] + out_index
            sv_lines.append(
                f"  proj_meta_beats[{meta_flat}] = "
                f"{_hex(pack_meta(case.multipliers[out_index], case.right_shifts[out_index], case.output_zero_points[out_index]), 128)};"
            )

        records.append(
            {
                "name": case.name,
                "rows": case.rows,
                "reduction_size": case.reduction_size,
                "input_beats": input_beats,
                "groups": case_groups[case_index],
                "output_count": output_count,
                "output_beats": output_beats,
                "saturation_seen": result.saturation_seen,
                "input_sha256": input_digest.hexdigest(),
                "weight_sha256": weight_digest.hexdigest(),
                "output_sha256": output_digest.hexdigest(),
                "weight_beats_per_output": weight_beats_per_output,
                "weight_bytes_per_output": projection_weight_bytes_per_output(case.reduction_size),
            }
        )

    sv_lines.append("end")
    OUT_SVH.write_text("\n".join(sv_lines) + "\n", encoding="utf-8")
    OUT_JSON.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "generator": "tools/gen_projection_vectors.py",
                "reference": "tools/ace2_projection_reference.py",
                "hidden_size": HIDDEN_SIZE,
                "mlp_intermediate_size": MLP_INTERMEDIATE_SIZE,
                "projection_max_k": PROJ_MAX_K,
                "projection_mac_lanes": PROJ_MAC_LANES,
                "projection_max_input_beats": max_input_beats,
                "projection_max_output_beats": max_output_beats,
                "projection_max_groups": PROJ_MAX_GROUPS,
                "groups_per_weight_beat": PROJ_GROUPS_PER_WEIGHT_BEAT,
                "total_weight_beats": total_weight_beats,
                "total_meta_beats": total_meta_beats,
                "rmsnorm_consumer_contract": {
                    "case": "rmsnorm_per_tensor_consumer",
                    "producer_output_scale": RMSNORM_CONSUMER_OUTPUT_SCALE,
                    "projection_input_scale": RMSNORM_CONSUMER_OUTPUT_SCALE,
                },
                "cases": records,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
