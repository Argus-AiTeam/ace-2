#!/usr/bin/env python3
"""Independent fixed-point reference for the ACE-2 RoPE RTL slice."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from ace2_quality_contracts import (
    dynamic_rope_output_scale,
    requantize_dynamic_rope_value,
)


HIDDEN_SIZE = 896
HEAD_DIM = 64
LANES = 16
ROPE_BEATS = HIDDEN_SIZE // LANES
ROPE_SCALE_FRAC = 9
Q9_SCALE_ONE = 1 << ROPE_SCALE_FRAC
Q15_ONE = (1 << 15) - 1


@dataclass(frozen=True)
class RopeCase:
    name: str
    sequence_position: int
    activations: list[int]
    scales_q9: list[int]
    cos_q15: list[int]
    sin_q15: list[int]


@dataclass(frozen=True)
class RopeResult:
    outputs: list[int]
    saturation_seen: bool


@dataclass(frozen=True)
class DynamicRopeHeadCase:
    name: str
    activations: list[int]
    producer_scale32: int
    cos_q15: list[int]
    sin_q15: list[int]


@dataclass(frozen=True)
class DynamicRopeHeadResult:
    outputs: list[int]
    output_scale32: int
    rotated_s25: list[int]
    maximum_magnitude: int


def to_sint(value: int, width: int) -> int:
    mask = (1 << width) - 1
    value &= mask
    sign = 1 << (width - 1)
    return value - (1 << width) if value & sign else value


def saturate(value: int, low: int, high: int) -> tuple[int, bool]:
    if value > high:
        return high, True
    if value < low:
        return low, True
    return value, False


def round_shift_even(value: int, shift: int) -> int:
    if shift <= 0:
        return value
    sign = -1 if value < 0 else 1
    magnitude = abs(value)
    base = magnitude >> shift
    remainder = magnitude & ((1 << shift) - 1)
    half = 1 << (shift - 1)
    if remainder > half or (remainder == half and (base & 1)):
        base += 1
    return sign * base


def pack_int8(values: Iterable[int]) -> int:
    packed = 0
    for lane, value in enumerate(values):
        packed |= (value & 0xFF) << (lane * 8)
    return packed


def pack_int16(values: Iterable[int]) -> int:
    packed = 0
    for lane, value in enumerate(values):
        packed |= (value & 0xFFFF) << (lane * 16)
    return packed


def _pair_index(index: int) -> int:
    head_base = (index // HEAD_DIM) * HEAD_DIM
    dim = index % HEAD_DIM
    return head_base + dim + 32 if dim < 32 else head_base + dim - 32


def reference_rope(case: RopeCase) -> RopeResult:
    q9_values: list[int] = []
    saturation_seen = False
    for activation, scale in zip(case.activations, case.scales_q9, strict=True):
        q9_values.append(to_sint(activation, 8) * to_sint(scale, 16))

    outputs: list[int] = []
    for index, current_q9 in enumerate(q9_values):
        dim = index % HEAD_DIM
        pair_q9 = q9_values[_pair_index(index)]
        cos_q15 = to_sint(case.cos_q15[index], 16)
        sin_q15 = to_sint(case.sin_q15[index], 16)
        if dim < 32:
            rotated_q24 = current_q9 * cos_q15 - pair_q9 * sin_q15
        else:
            rotated_q24 = current_q9 * cos_q15 + pair_q9 * sin_q15
        rounded = round_shift_even(rotated_q24, ROPE_SCALE_FRAC + 15)
        clipped, saturated = saturate(rounded, -128, 127)
        outputs.append(clipped)
        saturation_seen = saturation_seen or saturated
    return RopeResult(outputs, saturation_seen)


def reference_dynamic_rope_head(case: DynamicRopeHeadCase) -> DynamicRopeHeadResult:
    if len(case.activations) != HEAD_DIM:
        raise ValueError("dynamic RoPE requires one 64-lane head")
    if len(case.cos_q15) != HEAD_DIM or len(case.sin_q15) != HEAD_DIM:
        raise ValueError("dynamic RoPE requires 64 cosine and sine coefficients")

    rotated: list[int] = []
    for index, activation in enumerate(case.activations):
        pair = case.activations[_pair_index(index)]
        current = to_sint(activation, 8)
        paired = to_sint(pair, 8)
        cosine = to_sint(case.cos_q15[index], 16)
        sine = to_sint(case.sin_q15[index], 16)
        if index < HEAD_DIM // 2:
            value = current * cosine - paired * sine
        else:
            value = current * cosine + paired * sine
        if not -(1 << 24) <= value < (1 << 24):
            raise OverflowError("dynamic RoPE signed-25-bit staging overflow")
        rotated.append(value)

    maximum = max(abs(value) for value in rotated)
    output_scale32 = dynamic_rope_output_scale(case.producer_scale32, maximum)
    outputs = [
        requantize_dynamic_rope_value(value, case.producer_scale32, output_scale32)
        for value in rotated
    ]
    if any(value == -128 for value in outputs):
        raise AssertionError("dynamic RoPE generated forbidden -128 payload")
    return DynamicRopeHeadResult(outputs, output_scale32, rotated, maximum)
