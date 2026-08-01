#!/usr/bin/env python3
"""Independent fixed-point reference for ACE-2 W4A8 projection RTL slices."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


HIDDEN_SIZE = 896
MLP_INTERMEDIATE_SIZE = 4864
PROJ_MAC_LANES = 4
PROJ_MAX_K = MLP_INTERMEDIATE_SIZE
PROJ_MAX_GROUPS = PROJ_MAX_K // PROJ_MAC_LANES
PROJ_GROUPS_PER_WEIGHT_BEAT = 16 // PROJ_MAC_LANES
ACT_WIDTH = 8
WEIGHT_WIDTH = 4


@dataclass(frozen=True)
class ProjectionCase:
    name: str
    rows: int
    reduction_size: int
    activations: list[list[int]]
    weights: list[list[int]]
    multipliers: list[int]
    right_shifts: list[int]
    output_zero_points: list[int]


@dataclass(frozen=True)
class ProjectionResult:
    outputs: list[list[int]]
    saturation_seen: bool


def to_sint(value: int, width: int) -> int:
    mask = (1 << width) - 1
    value &= mask
    sign = 1 << (width - 1)
    return value - (1 << width) if value & sign else value


def saturate_int8(value: int) -> tuple[int, bool]:
    if value > 127:
        return 127, True
    if value < -128:
        return -128, True
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
        packed |= (value & 0xFF) << (lane * ACT_WIDTH)
    return packed


def pack_w4(values: Iterable[int]) -> int:
    packed = 0
    for lane, value in enumerate(values):
        packed |= (value & 0xF) << (lane * WEIGHT_WIDTH)
    return packed


def pack_meta(multiplier: int, right_shift: int, output_zero_point: int) -> int:
    return (multiplier & 0xFFFFFFFF) | ((right_shift & 0x3F) << 32) | ((output_zero_point & 0xFF) << 40)


def projection_groups(reduction_size: int) -> int:
    if reduction_size % PROJ_MAC_LANES:
        raise ValueError(f"reduction size {reduction_size} is not divisible by MAC lanes")
    return reduction_size // PROJ_MAC_LANES


def projection_weight_beats_per_output(reduction_size: int) -> int:
    if reduction_size % 16:
        raise ValueError(f"reduction size {reduction_size} is not divisible by packed W4 beat width")
    return reduction_size // 16


def projection_weight_bytes_per_output(reduction_size: int) -> int:
    return projection_weight_beats_per_output(reduction_size) * 16


def reference_projection(case: ProjectionCase) -> ProjectionResult:
    outputs: list[list[int]] = []
    saturation_seen = False
    output_count = len(case.weights)
    if not (
        len(case.multipliers) == len(case.right_shifts) == len(case.output_zero_points) == output_count
    ):
        raise ValueError(f"{case.name}: metadata length does not match weight output count")
    for row in range(case.rows):
        row_outputs: list[int] = []
        act = [to_sint(value, ACT_WIDTH) for value in case.activations[row]]
        if len(act) != case.reduction_size:
            raise ValueError(
                f"{case.name}: activation row {row} has {len(act)} elements, expected {case.reduction_size}"
            )
        for out_index in range(output_count):
            weights = [to_sint(value, WEIGHT_WIDTH) for value in case.weights[out_index]]
            if len(weights) != case.reduction_size:
                raise ValueError(
                    f"{case.name}: output {out_index} has {len(weights)} weights, expected {case.reduction_size}"
                )
            acc = sum(a * w for a, w in zip(act, weights, strict=True))
            scaled = round_shift_even(acc * to_sint(case.multipliers[out_index], 32), case.right_shifts[out_index])
            clipped, saturated = saturate_int8(scaled + to_sint(case.output_zero_points[out_index], ACT_WIDTH))
            row_outputs.append(clipped)
            saturation_seen = saturation_seen or saturated
        outputs.append(row_outputs)
    return ProjectionResult(outputs, saturation_seen)
