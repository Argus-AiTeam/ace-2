#!/usr/bin/env python3
"""Independent fixed-point reference for static per-tensor int8 RMSNorm."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable


HIDDEN_SIZE = 896
LANES = 16
BEATS = HIDDEN_SIZE // LANES
ACT_WIDTH = 8
GAIN_WIDTH = 16
ACC_WIDTH = 48
INV_RMS_FRAC = 30
GAIN_FRAC = 8
GAIN_MAX = (1 << (GAIN_WIDTH - 1)) - 1


@dataclass(frozen=True)
class RmsNormResult:
    outputs: list[int]
    sumsq: int
    inv_rms_q30: int
    saturation_seen: bool


def _to_sint(value: int, width: int) -> int:
    mask = (1 << width) - 1
    value &= mask
    sign = 1 << (width - 1)
    return value - (1 << width) if value & sign else value


def _saturate_int8(value: int) -> tuple[int, bool]:
    if value > 127:
        return 127, True
    if value < -128:
        return -128, True
    return value, False


def _round_shift_even(value: int, shift: int) -> int:
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


def _isqrt_ceil(value: int) -> int:
    root = 0
    for candidate in range(256):
        if candidate * candidate <= value:
            root = candidate
    if root * root < value:
        root += 1
    return max(root, 1)


def _pack(values: Iterable[int], width: int) -> int:
    packed = 0
    mask = (1 << width) - 1
    for lane, value in enumerate(values):
        packed |= (value & mask) << (lane * width)
    return packed


def _chunks(values: list[int], lanes: int = LANES) -> list[list[int]]:
    return [values[index : index + lanes] for index in range(0, len(values), lanes)]


def derive_scaled_gains_q8(
    weights: Iterable[float],
    output_scale: float,
) -> list[int]:
    if not math.isfinite(output_scale) or output_scale <= 0:
        raise ValueError("RMSNorm output scale must be finite and positive")
    gains: list[int] = []
    for weight in weights:
        if not math.isfinite(weight):
            raise ValueError("RMSNorm weights must be finite")
        gain = round(weight / output_scale * (1 << GAIN_FRAC))
        if gain < -(1 << (GAIN_WIDTH - 1)) or gain > GAIN_MAX:
            raise OverflowError("RMSNorm scaled Q7.8 gain metadata is not representable")
        gains.append(gain)
    return gains


def rmsnorm_gain_scale_floor(weights: Iterable[float]) -> float:
    values = list(weights)
    if not values:
        raise ValueError("RMSNorm weights must not be empty")
    if not all(math.isfinite(weight) for weight in values):
        raise ValueError("RMSNorm weights must be finite")
    return max(abs(weight) for weight in values) * (1 << GAIN_FRAC) / GAIN_MAX


def derive_rmsnorm_output_scale(
    weights: Iterable[float],
    calibrated_output_absmax: float,
) -> float:
    values = list(weights)
    if not math.isfinite(calibrated_output_absmax) or calibrated_output_absmax <= 0:
        raise ValueError("calibrated RMSNorm output absmax must be finite and positive")
    return max(
        calibrated_output_absmax / 127.0,
        rmsnorm_gain_scale_floor(values),
    )


def reference_rmsnorm(activations: list[int], scaled_gains_q8: list[int]) -> RmsNormResult:
    if len(activations) != HIDDEN_SIZE:
        raise ValueError(f"expected {HIDDEN_SIZE} activations")
    if len(scaled_gains_q8) != HIDDEN_SIZE:
        raise ValueError(f"expected {HIDDEN_SIZE} gains")

    act = [_to_sint(value, ACT_WIDTH) for value in activations]
    gain = [_to_sint(value, GAIN_WIDTH) for value in scaled_gains_q8]
    sumsq = sum(value * value for value in act)
    mean_square = (sumsq + (HIDDEN_SIZE // 2)) // HIDDEN_SIZE
    rms_ceil = _isqrt_ceil(mean_square)
    inv_rms_q30 = (1 << INV_RMS_FRAC) // rms_ceil

    outputs: list[int] = []
    saturation_seen = False
    for value, lane_gain in zip(act, gain, strict=True):
        product = value * lane_gain * inv_rms_q30
        rounded = _round_shift_even(product, INV_RMS_FRAC + GAIN_FRAC)
        clipped, saturated = _saturate_int8(rounded)
        outputs.append(clipped)
        saturation_seen = saturation_seen or saturated

    return RmsNormResult(outputs, sumsq, inv_rms_q30, saturation_seen)


def pack_int8_beats(values: list[int]) -> list[int]:
    if len(values) != HIDDEN_SIZE:
        raise ValueError(f"expected {HIDDEN_SIZE} int8 values")
    return [_pack(chunk, ACT_WIDTH) for chunk in _chunks(values)]


def pack_gain_beats(values: list[int]) -> list[int]:
    if len(values) != HIDDEN_SIZE:
        raise ValueError(f"expected {HIDDEN_SIZE} gain values")
    return [_pack(chunk, GAIN_WIDTH) for chunk in _chunks(values)]
