#!/usr/bin/env python3
"""Independent integer oracle for projection-shadow staged attention v1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

try:
    from ace2_absolute_rope_online_attention_reference import absolute_coefficients_q15
    from ace2_quality_contracts import round_divide_even_signed, unpack_scale32
    from ace2_softmax_reference import EXP_LUT_Q15, EXP_ROUND_Q6_9, EXP_STEP_Q6_9
except ModuleNotFoundError:
    from tools.ace2_absolute_rope_online_attention_reference import absolute_coefficients_q15
    from tools.ace2_quality_contracts import round_divide_even_signed, unpack_scale32
    from tools.ace2_softmax_reference import EXP_LUT_Q15, EXP_ROUND_Q6_9, EXP_STEP_Q6_9


CONTRACT_ID = "layer0_projection_shadow_staged_attention_v1"
HEAD_DIM = 64
PAIR_COUNT = 32
Q_SCALE32 = 0x00FFA245
K_SCALE32 = 0x00008307
PROB_FRAC = 15


def check_signed(value: int, width: int, label: str) -> int:
    if not -(1 << (width - 1)) <= value < (1 << (width - 1)):
        raise OverflowError(f"{label} does not fit signed-{width}")
    return value


def rne_integer(value: int, shift: int) -> int:
    if shift < 0:
        return value << (-shift)
    return round_divide_even_signed(value, 1 << shift)


def projection_shadow_q15_16(
    dot_s32: int,
    bias_accumulator_s32: int,
    multiplier_s32: int,
    right_shift_u6: int,
) -> int:
    check_signed(dot_s32, 32, "projection dot")
    check_signed(bias_accumulator_s32, 32, "projection bias")
    check_signed(multiplier_s32, 32, "projection multiplier")
    if multiplier_s32 <= 0 or not 0 <= right_shift_u6 <= 63:
        raise ValueError("projection metadata is outside the frozen domain")
    accumulator = check_signed(dot_s32 + bias_accumulator_s32, 32, "post-bias accumulator")
    return check_signed(
        rne_integer(accumulator * multiplier_s32, right_shift_u6 - 16),
        32,
        "projection shadow",
    )


def rotate_shadow_head(values: Sequence[int], position: int) -> tuple[int, ...]:
    if len(values) != HEAD_DIM:
        raise ValueError("shadow RoPE requires 64 lanes")
    cosine, sine = absolute_coefficients_q15(position)
    rotated = [0] * HEAD_DIM
    for pair in range(PAIR_COUNT):
        q0 = check_signed(int(values[pair]), 32, "shadow lane")
        q1 = check_signed(int(values[pair + PAIR_COUNT]), 32, "shadow lane")
        real = rne_integer(q0 * cosine[pair] - q1 * sine[pair], 15)
        imag = rne_integer(q1 * cosine[pair] + q0 * sine[pair], 15)
        rotated[pair] = check_signed(real, 34, "rotated real")
        rotated[pair + PAIR_COUNT] = check_signed(imag, 34, "rotated imag")
    return tuple(rotated)


def shadow_score_q6_9(
    query_shadow: Sequence[int],
    key_shadow: Sequence[int],
    query_position: int,
    key_position: int,
    query_scale32: int = Q_SCALE32,
    key_scale32: int = K_SCALE32,
) -> int:
    if not 0 <= key_position <= query_position <= 32767:
        raise ValueError("shadow score positions violate the causal range")
    query = rotate_shadow_head(query_shadow, query_position)
    key = rotate_shadow_head(key_shadow, key_position)
    dot = check_signed(sum(q * k for q, k in zip(query, key, strict=True)), 74, "shadow dot")
    q_sig, q_exp = unpack_scale32(query_scale32)
    k_sig, k_exp = unpack_scale32(key_scale32)
    shift = 56 - (q_exp + k_exp)
    if not 48 <= shift <= 104:
        raise ValueError("shadow score shift is outside 48..104")
    score = rne_integer(dot * q_sig * k_sig, shift)
    return max(-32768, min(32767, score))


def softmax_q15(scores_q6_9: Sequence[int]) -> tuple[int, ...]:
    if not scores_q6_9:
        raise ValueError("softmax requires a nonempty row")
    maximum = max(scores_q6_9)
    weights = []
    for score in scores_q6_9:
        magnitude = maximum - int(score)
        index = (magnitude + EXP_ROUND_Q6_9) // EXP_STEP_Q6_9
        weights.append(EXP_LUT_Q15[index] if index < len(EXP_LUT_Q15) else 0)
    total = sum(weights)
    return tuple(round_divide_even_signed(weight << PROB_FRAC, total) for weight in weights)


def attention_value_s8(probabilities_q15: Sequence[int], values: Sequence[Sequence[int]]) -> tuple[int, ...]:
    if len(probabilities_q15) != len(values) or not values:
        raise ValueError("attention value requires matched probabilities and V rows")
    if any(len(row) != HEAD_DIM for row in values):
        raise ValueError("attention V rows require 64 lanes")
    output = []
    for lane in range(HEAD_DIM):
        accumulator = sum(
            int(probabilities_q15[token]) * int(values[token][lane])
            for token in range(len(values))
        )
        rounded = rne_integer(accumulator, PROB_FRAC)
        output.append(max(-128, min(127, rounded)))
    return tuple(output)


@dataclass(frozen=True)
class ShadowAttentionResult:
    scores_q6_9: tuple[int, ...]
    probabilities_q15: tuple[int, ...]
    output_s8: tuple[int, ...]


def staged_attention_row(
    query_shadow: Sequence[int],
    key_shadows: Sequence[Sequence[int]],
    values: Sequence[Sequence[int]],
    query_position: int,
) -> ShadowAttentionResult:
    if len(key_shadows) != query_position + 1 or len(values) != query_position + 1:
        raise ValueError("staged row requires every causal key through query_position")
    scores = tuple(
        shadow_score_q6_9(query_shadow, key, query_position, position)
        for position, key in enumerate(key_shadows)
    )
    probabilities = softmax_q15(scores)
    return ShadowAttentionResult(
        scores_q6_9=scores,
        probabilities_q15=probabilities,
        output_s8=attention_value_s8(probabilities, values),
    )
