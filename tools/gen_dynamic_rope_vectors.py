#!/usr/bin/env python3
"""Generate independent dynamic-RoPE head vectors for RTL parity."""

from __future__ import annotations

import json
from pathlib import Path

from ace2_quality_contracts import ceil_scale32_from_float
from ace2_rope_reference import DynamicRopeHeadCase, Q15_ONE, pack_int8, pack_int16, reference_dynamic_rope_head

ROOT = Path(__file__).resolve().parents[1]
OUT_SVH = ROOT / "verification/generated/dynamic_rope_vectors.svh"
OUT_JSON = ROOT / "verification/generated/dynamic_rope_vectors.json"


def cases() -> list[DynamicRopeHeadCase]:
    cos_mix, sin_mix = [], []
    for index in range(64):
        choices = ((30274, 12540), (23170, 23170), (12540, 30274), (Q15_ONE, 0))
        cosine, sine = choices[index % 4]
        cos_mix.append(cosine)
        sin_mix.append(sine)
    return [
        DynamicRopeHeadCase("all_zero", [0] * 64, ceil_scale32_from_float(0.03125), [Q15_ONE] * 64, [0] * 64),
        DynamicRopeHeadCase("identity_ramp", [((i * 17 + 3) % 63) - 31 for i in range(64)], ceil_scale32_from_float(0.0625), [Q15_ONE] * 64, [0] * 64),
        DynamicRopeHeadCase("quarter_turn_extrema", [127 if i % 4 == 0 else -128 if i % 4 == 1 else i - 32 for i in range(64)], ceil_scale32_from_float(0.0078125), [0] * 64, [Q15_ONE] * 64),
        DynamicRopeHeadCase("mixed_coefficients", [((i * 29 + 11) % 255) - 127 for i in range(64)], ceil_scale32_from_float(0.015625), cos_mix, sin_mix),
    ]


def hex_literal(value: int, bits: int) -> str:
    return f"{bits}'h{value:0{bits // 4}x}"


def main() -> None:
    rows = []
    lines = [
        f"localparam integer DYNAMIC_ROPE_CASE_COUNT = {len(cases())};",
        "task load_dynamic_rope_case;",
        " input integer case_index; output reg [31:0] case_producer_scale32;",
        " output reg [511:0] case_activations; output reg [1023:0] case_cos_q15;",
        " output reg [1023:0] case_sin_q15; output reg [511:0] case_expected_outputs;",
        " output reg [31:0] case_expected_scale32; begin",
        " case_producer_scale32=0; case_activations=0; case_cos_q15=0; case_sin_q15=0; case_expected_outputs=0; case_expected_scale32=0;",
        " case (case_index)",
    ]
    for index, case in enumerate(cases()):
        result = reference_dynamic_rope_head(case)
        rows.append({"name": case.name, "producer_scale32": case.producer_scale32, "activations": case.activations, "cos_q15": case.cos_q15, "sin_q15": case.sin_q15, "expected_outputs": result.outputs, "expected_scale32": result.output_scale32, "maximum_magnitude": result.maximum_magnitude})
        lines += [
            f" {index}: begin",
            f"  case_producer_scale32={hex_literal(case.producer_scale32, 32)};",
            f"  case_activations={hex_literal(pack_int8(case.activations), 512)};",
            f"  case_cos_q15={hex_literal(pack_int16(case.cos_q15), 1024)};",
            f"  case_sin_q15={hex_literal(pack_int16(case.sin_q15), 1024)};",
            f"  case_expected_outputs={hex_literal(pack_int8(result.outputs), 512)};",
            f"  case_expected_scale32={hex_literal(result.output_scale32, 32)}; end",
        ]
    lines += [" default: begin end", " endcase end endtask", ""]
    OUT_SVH.write_text("\n".join(lines), encoding="utf-8")
    OUT_JSON.write_text(json.dumps({"contract_id": "dynamic_rope_head_scale_v1", "cases": rows}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
