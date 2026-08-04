#!/usr/bin/env python3
"""Independent full-shape post-MLP residual reference for ACE-2."""

from __future__ import annotations

from dataclasses import dataclass


HIDDEN_SIZE = 896
LANES = 16
BEATS = HIDDEN_SIZE // LANES


@dataclass(frozen=True)
class MlpResidualCase:
    name: str
    down_projection: list[int]
    residual_stream: list[int]


@dataclass(frozen=True)
class MlpResidualResult:
    outputs: list[int]
    saturation_seen: bool
    positive_saturation_count: int
    negative_saturation_count: int


def to_int8(value: int) -> int:
    value &= 0xFF
    return value - 256 if value & 0x80 else value


def reference_mlp_residual_add(case: MlpResidualCase) -> MlpResidualResult:
    if len(case.down_projection) != HIDDEN_SIZE:
        raise ValueError(f"{case.name}: down projection must have {HIDDEN_SIZE} elements")
    if len(case.residual_stream) != HIDDEN_SIZE:
        raise ValueError(f"{case.name}: residual stream must have {HIDDEN_SIZE} elements")

    outputs: list[int] = []
    positive_saturation_count = 0
    negative_saturation_count = 0
    for down_value, residual_value in zip(
        case.down_projection, case.residual_stream, strict=True
    ):
        total = to_int8(down_value) + to_int8(residual_value)
        if total > 127:
            total = 127
            positive_saturation_count += 1
        elif total < -128:
            total = -128
            negative_saturation_count += 1
        outputs.append(total)

    return MlpResidualResult(
        outputs=outputs,
        saturation_seen=(positive_saturation_count + negative_saturation_count) != 0,
        positive_saturation_count=positive_saturation_count,
        negative_saturation_count=negative_saturation_count,
    )


def pack_int8_beats(values: list[int]) -> list[int]:
    if len(values) != HIDDEN_SIZE:
        raise ValueError(f"expected {HIDDEN_SIZE} values")
    beats: list[int] = []
    for offset in range(0, HIDDEN_SIZE, LANES):
        packed = 0
        for lane, value in enumerate(values[offset : offset + LANES]):
            packed |= (value & 0xFF) << (lane * 8)
        beats.append(packed)
    return beats
