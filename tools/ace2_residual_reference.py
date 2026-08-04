#!/usr/bin/env python3
"""Independent signed-int8 residual-add reference for ACE-2."""

from __future__ import annotations


HIDDEN_SIZE = 896
LANES = 16
BEATS = HIDDEN_SIZE // LANES


def _to_int8(value: int) -> int:
    value &= 0xFF
    return value - 256 if value & 0x80 else value


def reference_residual_add(lhs: list[int], rhs: list[int]) -> tuple[list[int], bool]:
    if len(lhs) != HIDDEN_SIZE or len(rhs) != HIDDEN_SIZE:
        raise ValueError(f"expected two {HIDDEN_SIZE}-element vectors")

    outputs: list[int] = []
    saturation_seen = False
    for lhs_value, rhs_value in zip(lhs, rhs, strict=True):
        total = _to_int8(lhs_value) + _to_int8(rhs_value)
        if total > 127:
            total = 127
            saturation_seen = True
        elif total < -128:
            total = -128
            saturation_seen = True
        outputs.append(total)
    return outputs, saturation_seen


def pack_int8_beats(values: list[int]) -> list[int]:
    if len(values) != HIDDEN_SIZE:
        raise ValueError(f"expected {HIDDEN_SIZE} values")
    beats = []
    for offset in range(0, HIDDEN_SIZE, LANES):
        packed = 0
        for lane, value in enumerate(values[offset : offset + LANES]):
            packed |= (value & 0xFF) << (lane * 8)
        beats.append(packed)
    return beats
