#!/usr/bin/env python3
"""Generate SiLU-gate RTL lookup data and independent oracle vectors."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ace2_silu_gate_reference import (
    MLP_INTERMEDIATE_SIZE,
    SILU_LANES,
    SILU_LUT,
    SILU_LUT_MAX_INDEX,
    SILU_LUT_MIN_INDEX,
    SiluGateCase,
    SiluGateResult,
    reference_silu_gate,
    reference_silu_gate_packed_int8,
)


ROOT = Path(__file__).resolve().parents[1]
LUT_OUT = ROOT / "rtl" / "generated" / "ace2_silu_lut.svh"
SV_OUT = ROOT / "verification" / "generated" / "silu_gate_vectors.svh"
JSON_OUT = ROOT / "verification" / "generated" / "silu_gate_vectors.json"
SHELL_SV_OUT = ROOT / "verification" / "generated" / "silu_gate_shell_vectors.svh"
SHELL_JSON_OUT = ROOT / "verification" / "generated" / "silu_gate_shell_vectors.json"

BOUNDARY_BITS = {
    "tie_even_keep": 0,
    "tie_even_increment": 1,
    "positive_saturation": 2,
    "negative_saturation": 3,
    "lut_clip_low": 4,
    "lut_clip_high": 5,
}
REQUIRED_BOUNDARY_CASES = {
    "tie_even_keep": "tie_even_rounding",
    "tie_even_increment": "tie_even_rounding",
    "positive_saturation": "positive_saturation",
    "negative_saturation": "negative_saturation",
    "lut_clip_low": "lut_clip_low",
    "lut_clip_high": "lut_clip_high",
}
SHELL_REQUIRED_BOUNDARY_CASES = {
    "tie_even_keep": "packed_tie_even_rounding",
    "tie_even_increment": "packed_tie_even_rounding",
    "positive_saturation": "packed_positive_saturation",
    "negative_saturation": "packed_negative_saturation",
}


def cases() -> list[SiluGateCase]:
    return [
        SiluGateCase("single_negative", [-512], [1024], 1, 21, -3),
        SiluGateCase(
            "tie_even_rounding",
            [-3904, -3904, -3904, -3904],
            [-15, -13, 15, 13],
            1,
            1,
            0,
        ),
        SiluGateCase("positive_saturation", [4032], [32767], 1, 21, 0),
        SiluGateCase("negative_saturation", [4032], [-32768], 1, 21, 0),
        SiluGateCase("lut_clip_low", [-4160, -4096], [-32768, -32768], 1, 12, 0),
        SiluGateCase("lut_clip_high", [4160, 4096], [8, 8], 1, 12, 0),
        SiluGateCase(
            "ragged_mixed_values",
            [((index * 173 + 91) % 8192) - 4096 for index in range(17)],
            [((index * 257 + 31) % 4096) - 2048 for index in range(17)],
            7,
            18,
            1,
        ),
        SiluGateCase(
            "full_mlp_intermediate",
            [((index * 43 + 19) % 6144) - 3072 for index in range(MLP_INTERMEDIATE_SIZE)],
            [((index * 61 + 7) % 3072) - 1536 for index in range(MLP_INTERMEDIATE_SIZE)],
            3,
            20,
            -2,
        ),
    ]


def shell_cases() -> list[SiluGateCase]:
    lane_order_gate = [-128, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, -64]
    lane_order_up = [1, -2, 3, -4, 5, -6, 7, -8, 9, -10, 11, -12, 13, -14, 15, -16, 17]
    full_gate = [((index * 29) % 256) - 128 for index in range(MLP_INTERMEDIATE_SIZE)]
    full_up = [((index * 53 + 1) % 256) - 128 for index in range(MLP_INTERMEDIATE_SIZE)]
    return [
        SiluGateCase("packed_single_negative", [-128], [1], 1, 12, 0),
        SiluGateCase("packed_tie_even_rounding", [-64, -64], [-15, -13], 1, 5, 0),
        SiluGateCase("packed_positive_saturation", [-128], [-128], 1, 0, 0),
        SiluGateCase("packed_negative_saturation", [-128], [1], 1, 0, 0),
        SiluGateCase("packed_lane_order", lane_order_gate, lane_order_up, 17, 12, -3),
        SiluGateCase("packed_full_mlp_intermediate", full_gate, full_up, 1, 12, -2),
    ]


def pack(values: list[int], width: int, lanes: int) -> int:
    packed = 0
    for lane in range(lanes):
        value = values[lane] if lane < len(values) else 0
        packed |= (value & ((1 << width) - 1)) << (lane * width)
    return packed


def hex_value(value: int, width: int) -> str:
    return f"{width}'h{value & ((1 << width) - 1):0{width // 4}x}"


def boundary_hits(result: SiluGateResult) -> dict[str, int]:
    return {
        "tie_even_keep": result.tie_even_keep_count,
        "tie_even_increment": result.tie_even_increment_count,
        "positive_saturation": result.positive_saturation_count,
        "negative_saturation": result.negative_saturation_count,
        "lut_clip_low": result.lut_clip_low_count,
        "lut_clip_high": result.lut_clip_high_count,
    }


def boundary_mask(hits: dict[str, int]) -> int:
    return sum(1 << BOUNDARY_BITS[name] for name, count in hits.items() if count)


def emit_shell_vectors() -> None:
    vector_cases = shell_cases()
    max_length = max(len(case.gate_q6_9) for case in vector_cases)
    max_input_beats = (max_length + 15) // 16
    max_output_beats = (max_length + 15) // 16
    full_case_index = next(
        index for index, case in enumerate(vector_cases)
        if case.name == "packed_full_mlp_intermediate"
    )
    required_mask = sum(1 << BOUNDARY_BITS[name] for name in SHELL_REQUIRED_BOUNDARY_CASES)
    lines = [
        "// Generated by tools/gen_silu_gate_vectors.py; do not edit.",
        f"localparam integer SILU_CASE_COUNT = {len(vector_cases)};",
        f"localparam integer SILU_MAX_LENGTH = {max_length};",
        f"localparam integer SILU_MAX_INPUT_BEATS = {max_input_beats};",
        f"localparam integer SILU_MAX_OUTPUT_BEATS = {max_output_beats};",
        f"localparam integer SILU_FULL_CASE_INDEX = {full_case_index};",
        f"localparam [5:0] SILU_REQUIRED_BOUNDARY_COVERAGE = 6'h{required_mask:02x};",
        "reg [15:0] silu_case_length [0:SILU_CASE_COUNT-1];",
        "reg signed [31:0] silu_case_multiplier [0:SILU_CASE_COUNT-1];",
        "reg [5:0] silu_case_right_shift [0:SILU_CASE_COUNT-1];",
        "reg signed [7:0] silu_case_zero_point [0:SILU_CASE_COUNT-1];",
        "reg silu_expected_saturation [0:SILU_CASE_COUNT-1];",
        "reg [5:0] silu_case_boundary_coverage [0:SILU_CASE_COUNT-1];",
        "reg [127:0] silu_gate_beats [0:SILU_CASE_COUNT*SILU_MAX_INPUT_BEATS-1];",
        "reg [127:0] silu_up_beats [0:SILU_CASE_COUNT*SILU_MAX_INPUT_BEATS-1];",
        "reg [127:0] silu_expected_beats [0:SILU_CASE_COUNT*SILU_MAX_OUTPUT_BEATS-1];",
        "initial begin",
    ]
    records = []
    for case_index, case in enumerate(vector_cases):
        result = reference_silu_gate_packed_int8(case)
        hits = boundary_hits(result)
        length = len(case.gate_q6_9)
        input_beats = (length + 15) // 16
        lines.extend(
            [
                f"  silu_case_length[{case_index}] = 16'd{length};",
                f"  silu_case_multiplier[{case_index}] = 32'sh{case.multiplier & 0xffffffff:08x};",
                f"  silu_case_right_shift[{case_index}] = 6'd{case.right_shift};",
                f"  silu_case_zero_point[{case_index}] = 8'sh{case.output_zero_point & 0xff:02x};",
                f"  silu_expected_saturation[{case_index}] = 1'b{int(result.saturation_seen)};",
                f"  silu_case_boundary_coverage[{case_index}] = 6'h{boundary_mask(hits):02x};",
            ]
        )
        for beat in range(input_beats):
            start = beat * 16
            flat = case_index * max_input_beats + beat
            lines.append(
                f"  silu_gate_beats[{flat}] = {hex_value(pack(case.gate_q6_9[start:start + 16], 8, 16), 128)};"
            )
            lines.append(
                f"  silu_up_beats[{flat}] = {hex_value(pack(case.up_q6_9[start:start + 16], 8, 16), 128)};"
            )
            lines.append(
                f"  silu_expected_beats[{flat}] = {hex_value(pack(result.outputs[start:start + 16], 8, 16), 128)};"
            )
        records.append(
            {
                "name": case.name,
                "length": length,
                "input_beats": input_beats,
                "output_beats": input_beats,
                "multiplier": case.multiplier,
                "right_shift": case.right_shift,
                "output_zero_point": case.output_zero_point,
                "saturation_seen": result.saturation_seen,
                "boundary_hits": hits,
                "source_gate_sha256": hashlib.sha256(bytes(value & 0xff for value in case.gate_q6_9)).hexdigest(),
                "source_up_sha256": hashlib.sha256(bytes(value & 0xff for value in case.up_q6_9)).hexdigest(),
                "output_sha256": hashlib.sha256(bytes(value & 0xff for value in result.outputs)).hexdigest(),
            }
        )
    records_by_name = {record["name"]: record for record in records}
    for boundary, case_name in SHELL_REQUIRED_BOUNDARY_CASES.items():
        if records_by_name[case_name]["boundary_hits"][boundary] < 1:
            raise RuntimeError(f"{case_name} does not cover required boundary {boundary}")
    lines.append("end")
    SHELL_SV_OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    SHELL_JSON_OUT.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "generator": "tools/gen_silu_gate_vectors.py",
                "reference": "tools/ace2_silu_gate_reference.py",
                "input_format": "packed signed int8, sixteen lanes per 128-bit beat",
                "core_adapter": "ordered lower eight then upper eight, sign-extended to signed int16",
                "full_case_index": full_case_index,
                "required_boundary_cases": SHELL_REQUIRED_BOUNDARY_CASES,
                "cases": records,
            },
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    LUT_OUT.parent.mkdir(parents=True, exist_ok=True)
    SV_OUT.parent.mkdir(parents=True, exist_ok=True)
    lut_lines = [
        "// Generated by tools/gen_silu_gate_vectors.py; do not edit.",
        "function automatic signed [15:0] silu_lookup_q3_12;",
        "    input signed [15:0] gate_q6_9;",
        "    reg signed [9:0] table_index;",
        "    begin",
        "        table_index = 10'(gate_q6_9 >>> 6);",
        "        case (table_index)",
    ]
    for index in range(SILU_LUT_MIN_INDEX + 1, SILU_LUT_MAX_INDEX):
        index_literal = f"-10'sd{abs(index)}" if index < 0 else f"10'sd{index}"
        lut_lines.append(
            f"            {index_literal}: silu_lookup_q3_12 = "
            f"16'sh{SILU_LUT[index] & 0xffff:04x};"
        )
    lut_lines.extend(
        [
            f"            default: silu_lookup_q3_12 = (table_index <= -10'sd{abs(SILU_LUT_MIN_INDEX)}) ? "
            f"16'sh{SILU_LUT[SILU_LUT_MIN_INDEX] & 0xffff:04x} : "
            f"16'sh{SILU_LUT[SILU_LUT_MAX_INDEX] & 0xffff:04x};",
            "        endcase",
            "    end",
            "endfunction",
        ]
    )
    LUT_OUT.write_text("\n".join(lut_lines) + "\n", encoding="utf-8")

    vector_cases = cases()
    max_length = max(len(case.gate_q6_9) for case in vector_cases)
    max_input_beats = (max_length + SILU_LANES - 1) // SILU_LANES
    max_output_beats = (max_length + 15) // 16
    lines = [
        "// Generated by tools/gen_silu_gate_vectors.py; do not edit.",
        f"localparam integer SILU_CASE_COUNT = {len(vector_cases)};",
        f"localparam integer SILU_MAX_LENGTH = {max_length};",
        f"localparam integer SILU_MAX_INPUT_BEATS = {max_input_beats};",
        f"localparam integer SILU_MAX_OUTPUT_BEATS = {max_output_beats};",
        "localparam [5:0] SILU_REQUIRED_BOUNDARY_COVERAGE = 6'h3f;",
        "reg [15:0] silu_case_length [0:SILU_CASE_COUNT-1];",
        "reg signed [31:0] silu_case_multiplier [0:SILU_CASE_COUNT-1];",
        "reg [5:0] silu_case_right_shift [0:SILU_CASE_COUNT-1];",
        "reg signed [7:0] silu_case_zero_point [0:SILU_CASE_COUNT-1];",
        "reg silu_expected_saturation [0:SILU_CASE_COUNT-1];",
        "reg [5:0] silu_case_boundary_coverage [0:SILU_CASE_COUNT-1];",
        "reg [127:0] silu_gate_beats [0:SILU_CASE_COUNT*SILU_MAX_INPUT_BEATS-1];",
        "reg [127:0] silu_up_beats [0:SILU_CASE_COUNT*SILU_MAX_INPUT_BEATS-1];",
        "reg [127:0] silu_expected_beats [0:SILU_CASE_COUNT*SILU_MAX_OUTPUT_BEATS-1];",
        "initial begin",
    ]
    records = []
    for case_index, case in enumerate(vector_cases):
        result = reference_silu_gate(case)
        hits = boundary_hits(result)
        length = len(case.gate_q6_9)
        input_beats = (length + SILU_LANES - 1) // SILU_LANES
        output_beats = (length + 15) // 16
        lines.extend(
            [
                f"  silu_case_length[{case_index}] = 16'd{length};",
                f"  silu_case_multiplier[{case_index}] = 32'sh{case.multiplier & 0xffffffff:08x};",
                f"  silu_case_right_shift[{case_index}] = 6'd{case.right_shift};",
                f"  silu_case_zero_point[{case_index}] = 8'sh{case.output_zero_point & 0xff:02x};",
                f"  silu_expected_saturation[{case_index}] = 1'b{int(result.saturation_seen)};",
                f"  silu_case_boundary_coverage[{case_index}] = 6'h{boundary_mask(hits):02x};",
            ]
        )
        for beat in range(input_beats):
            start = beat * SILU_LANES
            gate = case.gate_q6_9[start : start + SILU_LANES]
            up = case.up_q6_9[start : start + SILU_LANES]
            flat = case_index * max_input_beats + beat
            lines.append(f"  silu_gate_beats[{flat}] = {hex_value(pack(gate, 16, SILU_LANES), 128)};")
            lines.append(f"  silu_up_beats[{flat}] = {hex_value(pack(up, 16, SILU_LANES), 128)};")
        for beat in range(output_beats):
            start = beat * 16
            output = result.outputs[start : start + 16]
            flat = case_index * max_output_beats + beat
            lines.append(f"  silu_expected_beats[{flat}] = {hex_value(pack(output, 8, 16), 128)};")
        digest = hashlib.sha256(bytes(value & 0xff for value in result.outputs)).hexdigest()
        records.append(
            {
                "name": case.name,
                "length": length,
                "input_beats": input_beats,
                "output_beats": output_beats,
                "multiplier": case.multiplier,
                "right_shift": case.right_shift,
                "output_zero_point": case.output_zero_point,
                "saturation_seen": result.saturation_seen,
                "boundary_hits": hits,
                "output_sha256": digest,
            }
        )
    records_by_name = {record["name"]: record for record in records}
    for boundary, case_name in REQUIRED_BOUNDARY_CASES.items():
        hit_count = records_by_name[case_name]["boundary_hits"][boundary]
        if hit_count < 1:
            raise RuntimeError(
                f"{case_name} does not cover required boundary {boundary}"
            )
    lines.append("end")
    SV_OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    JSON_OUT.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "generator": "tools/gen_silu_gate_vectors.py",
                "reference": "tools/ace2_silu_gate_reference.py",
                "input_format": "signed int16 Q6.9",
                "lookup_output_format": "signed int16 Q3.12",
                "product_format": "signed int32 Q9.21",
                "lookup_step": "1/8, floor-indexed, clipped to [-8,8]",
                "requant_metadata_scope": "one multiplier/right-shift/zero-point tuple per descriptor",
                "boundary_coverage_bits": BOUNDARY_BITS,
                "required_boundary_cases": REQUIRED_BOUNDARY_CASES,
                "cases": records,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    emit_shell_vectors()


if __name__ == "__main__":
    main()
