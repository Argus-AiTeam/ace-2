#!/usr/bin/env python3
"""Independent fixed-point reference for the ACE-2 softmax RTL slice."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


CONTEXT_MAX = 8
SCORE_FRAC = 9
PROB_FRAC = 15
EXP_STEP_Q6_9 = 64
EXP_ROUND_Q6_9 = EXP_STEP_Q6_9 // 2
EXP_LUT_Q15 = [
    32768, 28918, 25520, 22521, 19875, 17539, 15479, 13660,
    12055, 10638, 9388, 8285, 7312, 6452, 5694, 5025,
    4435, 3914, 3454, 3048, 2690, 2374, 2095, 1849,
    1631, 1440, 1271, 1121, 990, 873, 771, 680,
    600, 530, 467, 412, 364, 321, 283, 250,
    221, 195, 172, 152, 134, 118, 104, 92,
    81, 72, 63, 56, 49, 43, 38, 34,
    30, 26, 23, 21, 18, 16, 14, 12,
    11,
]


@dataclass(frozen=True)
class SoftmaxCase:
    name: str
    scores_q6_9: list[int]


@dataclass(frozen=True)
class SoftmaxResult:
    max_score_q6_9: int
    exp_weights_q15: list[int]
    exp_sum_q15: int
    probabilities_q0_15: list[int]


def to_sint(value: int, width: int) -> int:
    mask = (1 << width) - 1
    value &= mask
    sign = 1 << (width - 1)
    return value - (1 << width) if value & sign else value


def round_div_even(numerator: int, denominator: int) -> int:
    quotient, remainder = divmod(numerator, denominator)
    doubled = remainder * 2
    if doubled > denominator or (doubled == denominator and (quotient & 1)):
        quotient += 1
    return quotient


def exp_weight_q15(delta_q6_9: int) -> int:
    if delta_q6_9 >= 0:
        return EXP_LUT_Q15[0]
    magnitude = -delta_q6_9
    table_index = (magnitude + EXP_ROUND_Q6_9) // EXP_STEP_Q6_9
    if table_index >= len(EXP_LUT_Q15):
        return 0
    return EXP_LUT_Q15[table_index]


def pack_uint16(values: Iterable[int]) -> int:
    packed = 0
    for lane, value in enumerate(values):
        packed |= (value & 0xFFFF) << (lane * 16)
    return packed


def reference_softmax(case: SoftmaxCase) -> SoftmaxResult:
    if not 1 <= len(case.scores_q6_9) <= CONTEXT_MAX:
        raise ValueError("softmax case context must be 1..CONTEXT_MAX")
    scores = [to_sint(value, 16) for value in case.scores_q6_9]
    max_score = max(scores)
    weights = [exp_weight_q15(score - max_score) for score in scores]
    exp_sum = sum(weights)
    probabilities = [
        round_div_even(weight << PROB_FRAC, exp_sum)
        for weight in weights
    ]
    padded_weights = weights + [0 for _ in range(CONTEXT_MAX - len(weights))]
    padded_probabilities = probabilities + [0 for _ in range(CONTEXT_MAX - len(probabilities))]
    return SoftmaxResult(
        max_score_q6_9=max_score,
        exp_weights_q15=padded_weights,
        exp_sum_q15=exp_sum,
        probabilities_q0_15=padded_probabilities,
    )
