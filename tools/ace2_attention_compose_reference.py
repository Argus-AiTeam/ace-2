#!/usr/bin/env python3
"""Independent fixed-point reference for tiled long-context attention."""

from __future__ import annotations

from dataclasses import dataclass

from ace2_softmax_reference import exp_weight_q15, round_div_even, to_sint
from ace2_attention_value_reference import round_shift_even_signed


HEAD_DIM = 64
CONTEXT_MAX = 32768
PROB_FRAC = 15


@dataclass(frozen=True)
class AttentionComposeCase:
    name: str
    scores_q6_9: list[int]
    values: list[list[int]]


@dataclass(frozen=True)
class AttentionComposeResult:
    max_score_q6_9: int
    exp_sum_q15: int
    probabilities_q15: list[int]
    accumulators: list[int]
    outputs: list[int]
    saturation_seen: bool


def reference_attention_compose(
    case: AttentionComposeCase,
) -> AttentionComposeResult:
    context_count = len(case.scores_q6_9)
    if not 9 <= context_count <= CONTEXT_MAX:
        raise ValueError("composed attention context must be 9..CONTEXT_MAX")
    if len(case.values) != context_count:
        raise ValueError("one V row is required per score")
    if any(len(row) != HEAD_DIM for row in case.values):
        raise ValueError("each V row must contain HEAD_DIM elements")

    scores = [to_sint(value, 16) for value in case.scores_q6_9]
    values = [[to_sint(value, 8) for value in row] for row in case.values]
    max_score = max(scores)
    weights = [exp_weight_q15(score - max_score) for score in scores]
    exp_sum = sum(weights)
    probabilities = [
        round_div_even(weight << PROB_FRAC, exp_sum) for weight in weights
    ]
    accumulators = [
        sum(
            probabilities[token] * values[token][lane]
            for token in range(context_count)
        )
        for lane in range(HEAD_DIM)
    ]
    rounded = [
        round_shift_even_signed(value, PROB_FRAC) for value in accumulators
    ]
    outputs = [max(-128, min(127, value)) for value in rounded]
    return AttentionComposeResult(
        max_score_q6_9=max_score,
        exp_sum_q15=exp_sum,
        probabilities_q15=probabilities,
        accumulators=accumulators,
        outputs=outputs,
        saturation_seen=any(
            output != value for output, value in zip(outputs, rounded, strict=True)
        ),
    )
