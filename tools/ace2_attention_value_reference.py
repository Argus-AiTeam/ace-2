#!/usr/bin/env python3
"""Independent fixed-point reference for the ACE-2 attention-value RTL slice."""

from __future__ import annotations

from dataclasses import dataclass


CONTEXT_MAX = 8
HEAD_DIM = 64
PROB_FRAC = 15


@dataclass(frozen=True)
class AttentionValueCase:
    name: str
    probabilities_q15: list[int]
    values: list[list[int]]


@dataclass(frozen=True)
class AttentionValueResult:
    accumulators: list[int]
    outputs: list[int]
    saturation_seen: bool


def to_sint(value: int, width: int) -> int:
    mask = (1 << width) - 1
    value &= mask
    sign = 1 << (width - 1)
    return value - (1 << width) if value & sign else value


def round_shift_even_signed(value: int, shift: int) -> int:
    negative = value < 0
    magnitude = -value if negative else value
    base = magnitude >> shift
    remainder = magnitude & ((1 << shift) - 1)
    half = 1 << (shift - 1)
    if remainder > half or (remainder == half and (base & 1)):
        base += 1
    return -base if negative else base


def reference_attention_value(case: AttentionValueCase) -> AttentionValueResult:
    context_count = len(case.probabilities_q15)
    if not 1 <= context_count <= CONTEXT_MAX:
        raise ValueError("attention-value context must be 1..CONTEXT_MAX")
    if len(case.values) != context_count:
        raise ValueError("one V row is required per probability")
    if any(len(row) != HEAD_DIM for row in case.values):
        raise ValueError("each V row must contain HEAD_DIM elements")

    probabilities = [value & 0xFFFF for value in case.probabilities_q15]
    values = [[to_sint(value, 8) for value in row] for row in case.values]
    accumulators = [
        sum(probabilities[token] * values[token][lane] for token in range(context_count))
        for lane in range(HEAD_DIM)
    ]
    rounded = [round_shift_even_signed(value, PROB_FRAC) for value in accumulators]
    outputs = [max(-128, min(127, value)) for value in rounded]
    return AttentionValueResult(
        accumulators=accumulators,
        outputs=outputs,
        saturation_seen=any(output != value for output, value in zip(outputs, rounded, strict=True)),
    )
