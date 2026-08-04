#!/usr/bin/env python3
"""Generate deterministic JSON/SVH vectors for residual-sidecar RTL."""

from __future__ import annotations

import json
from pathlib import Path

try:
    from ace2_qk_residual_cross_term_reference import (
        projection_residual,
        residual_cross_term_score,
        residual_rope_pair,
        staged_softmax_attention_value,
    )
except ModuleNotFoundError:
    from tools.ace2_qk_residual_cross_term_reference import (
        projection_residual,
        residual_cross_term_score,
        residual_rope_pair,
        staged_softmax_attention_value,
    )


ROOT = Path(__file__).resolve().parents[1]
OUT_JSON = ROOT / "verification/generated/qk_residual_cross_term_vectors.json"
OUT_SVH = ROOT / "verification/generated/qk_residual_cross_term_vectors.svh"
SCALE_ONE = 0x00008000
SCALE_HALF = 0x00FF8000


PROJECTION_INPUTS = [
    (3, 5, 1, SCALE_ONE, SCALE_HALF),
    (-17, 257, 5, SCALE_HALF, SCALE_ONE),
    (1000, 4096, 12, SCALE_ONE, 0x00F88000),
    (-1000, 4096, 12, SCALE_ONE, 0x00F88000),
]

ROPE_INPUTS = [
    (7, -7, 32767, 0),
    (1, 0, 16384, 0),
    (4, -3, 23170, 23170),
    (-7, 7, -12540, 30274),
]


def score_case(seed: int) -> dict[str, object]:
    query = [((lane * (7 + seed)) % 31) - 15 for lane in range(64)]
    key = [((lane * (5 + seed)) % 29) - 14 for lane in range(64)]
    query_residual = [((lane * (3 + seed)) % 15) - 7 for lane in range(64)]
    key_residual = [((lane * (11 + seed)) % 15) - 7 for lane in range(64)]
    scales = (SCALE_ONE, SCALE_HALF, SCALE_HALF, SCALE_ONE)
    authoritative_base_score = ((seed + 1) * 17 - 23) << 40
    result = residual_cross_term_score(
        authoritative_base_score,
        query,
        key,
        query_residual,
        key_residual,
        *scales,
    )
    return {
        "authoritative_base_score_q20_44_s64": authoritative_base_score,
        "query_q8": query,
        "key_q8": key,
        "query_residual_s8": query_residual,
        "key_residual_s8": key_residual,
        "scales": list(scales),
        "correction_dots_s32": list(result.correction_dots_s32),
        "correction_terms_q20_44_s67": list(result.correction_terms_q20_44_s67),
        "score_q20_44_s64": result.score_q20_44_s64,
    }


def attention_row_case(length: int) -> dict[str, object]:
    scores = [((index * 17) % 41 - 20) << 40 for index in range(length)]
    values = [
        [((token * 11 + lane * 3) % 255) - 127 for lane in range(64)]
        for token in range(length)
    ]
    result = staged_softmax_attention_value(scores, values)
    return {
        "row_length": length,
        "scores_q20_44": scores,
        "values_s8": values,
        "weights_q1_31": list(result.weights_q1_31),
        "probabilities_q0_15": list(result.probabilities_q0_15),
        "output_s8": list(result.output_s8),
    }


def sv_signed(width: int, value: int) -> str:
    return f"-{width}'sd{abs(value)}" if value < 0 else f"{width}'sd{value}"


def main() -> int:
    projection_cases = []
    for values in PROJECTION_INPUTS:
        result = projection_residual(*values)
        projection_cases.append(
            {
                "input": {
                    "accumulator_s32": values[0],
                    "multiplier_s32": values[1],
                    "shift_u6": values[2],
                    "baseline_scale32": values[3],
                    "residual_scale32": values[4],
                },
                "expected": {
                    "baseline_q8": result.baseline_q8,
                    "residual_s4": result.residual_s4,
                    "positive_clamp": result.positive_clamp,
                    "negative_clamp": result.negative_clamp,
                },
            }
        )
    rope_cases = []
    for values in ROPE_INPUTS:
        result = residual_rope_pair(*values)
        rope_cases.append(
            {
                "input": {
                    "real_s4": values[0],
                    "imag_s4": values[1],
                    "cosine_q1_15": values[2],
                    "sine_q1_15": values[3],
                },
                "expected": {"real_s8": result.real_s8, "imag_s8": result.imag_s8},
            }
        )
    score_cases = [score_case(0), score_case(2)]
    attention_row_cases = [attention_row_case(length) for length in (1, 2, 63, 64, 65)]
    payload = {
        "schema_version": 1,
        "contract_id": "shared_qk_residual_cross_term_attention_v1",
        "coverage": {
            "layer_ids": list(range(24)),
            "query_to_kv_head": [0] * 7 + [1] * 7,
            "row_lengths": [1, 2, 63, 64, 65],
            "reserved_s4_code": -8,
            "score_seed": "authoritative_base_score_q20_44_s64",
            "score_terms": ["q_x_rk", "rq_x_k", "rq_x_rk"],
        },
        "projection_cases": projection_cases,
        "rope_cases": rope_cases,
        "score_cases": score_cases,
        "attention_row_cases": attention_row_cases,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    lines = ["// Generated by tools/gen_qk_residual_cross_term_vectors.py"]
    for index, case in enumerate(projection_cases):
        i, e = case["input"], case["expected"]
        lines.extend(
            [
                f"localparam logic signed [31:0] QKR_P{index}_ACC = {sv_signed(32, i['accumulator_s32'])};",
                f"localparam logic signed [31:0] QKR_P{index}_MUL = {sv_signed(32, i['multiplier_s32'])};",
                f"localparam logic [5:0] QKR_P{index}_SHIFT = 6'd{i['shift_u6']};",
                f"localparam logic [31:0] QKR_P{index}_BASE_SCALE = 32'h{i['baseline_scale32']:08x};",
                f"localparam logic [31:0] QKR_P{index}_RES_SCALE = 32'h{i['residual_scale32']:08x};",
                f"localparam logic signed [7:0] QKR_P{index}_Q8 = {sv_signed(8, e['baseline_q8'])};",
                f"localparam logic signed [3:0] QKR_P{index}_R4 = {sv_signed(4, e['residual_s4'])};",
                f"localparam logic QKR_P{index}_POS_CLAMP = 1'b{int(e['positive_clamp'])};",
                f"localparam logic QKR_P{index}_NEG_CLAMP = 1'b{int(e['negative_clamp'])};",
            ]
        )
    for index, case in enumerate(rope_cases):
        i, e = case["input"], case["expected"]
        lines.extend(
            [
                f"localparam logic signed [3:0] QKR_R{index}_REAL = {sv_signed(4, i['real_s4'])};",
                f"localparam logic signed [3:0] QKR_R{index}_IMAG = {sv_signed(4, i['imag_s4'])};",
                f"localparam logic signed [15:0] QKR_R{index}_COS = {sv_signed(16, i['cosine_q1_15'])};",
                f"localparam logic signed [15:0] QKR_R{index}_SIN = {sv_signed(16, i['sine_q1_15'])};",
                f"localparam logic signed [7:0] QKR_R{index}_REAL_OUT = {sv_signed(8, e['real_s8'])};",
                f"localparam logic signed [7:0] QKR_R{index}_IMAG_OUT = {sv_signed(8, e['imag_s8'])};",
            ]
        )
    case = score_cases[0]
    for lane in range(64):
        lines.extend(
            [
                f"localparam logic signed [7:0] QKR_S_Q_{lane} = {sv_signed(8, case['query_q8'][lane])};",
                f"localparam logic signed [7:0] QKR_S_K_{lane} = {sv_signed(8, case['key_q8'][lane])};",
                f"localparam logic signed [7:0] QKR_S_RQ_{lane} = {sv_signed(8, case['query_residual_s8'][lane])};",
                f"localparam logic signed [7:0] QKR_S_RK_{lane} = {sv_signed(8, case['key_residual_s8'][lane])};",
            ]
        )
    lines.extend(
        [
            f"localparam logic signed [63:0] QKR_S_BASE_SCORE = {sv_signed(64, case['authoritative_base_score_q20_44_s64'])};",
            f"localparam logic signed [31:0] QKR_S_DOT_Q_RK = {sv_signed(32, case['correction_dots_s32'][0])};",
            f"localparam logic signed [31:0] QKR_S_DOT_RQ_K = {sv_signed(32, case['correction_dots_s32'][1])};",
            f"localparam logic signed [31:0] QKR_S_DOT_RQ_RK = {sv_signed(32, case['correction_dots_s32'][2])};",
            f"localparam logic signed [63:0] QKR_S_SCORE = {sv_signed(64, case['score_q20_44_s64'])};",
            f"localparam logic [31:0] QKR_S_Q_SCALE = 32'h{case['scales'][0]:08x};",
            f"localparam logic [31:0] QKR_S_K_SCALE = 32'h{case['scales'][1]:08x};",
            f"localparam logic [31:0] QKR_S_RQ_SCALE = 32'h{case['scales'][2]:08x};",
            f"localparam logic [31:0] QKR_S_RK_SCALE = 32'h{case['scales'][3]:08x};",
        ]
    )
    OUT_SVH.write_text("\n".join(lines) + "\n")
    print(f"generated {OUT_JSON.relative_to(ROOT)} and {OUT_SVH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
