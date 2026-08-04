#!/usr/bin/env python3
"""Independent scalar oracle and schedule model for DPRF RTL."""

from __future__ import annotations

from dataclasses import dataclass


SIGNED32_MIN = -(1 << 31)
SIGNED32_MAX = (1 << 31) - 1
SIGNED96_MIN = -(1 << 95)
SIGNED96_MAX = (1 << 95) - 1
UNSIGNED64_MAX = (1 << 64) - 1
VALID_LATENCY_CYCLES = 10


@dataclass(frozen=True)
class FusionLaneResult:
    output_s8: int
    numerator_s96: int
    denominator_u64: int
    common_exponent: int
    rounded_integer: int
    positive_saturation: bool
    negative_saturation: bool
    latency_cycles: int = VALID_LATENCY_CYCLES


def pack_scale32(significand: int, exponent: int) -> int:
    if not 0x8000 <= significand <= 0xFFFF:
        raise ValueError("Scale32 significand is not normalized")
    if not -24 <= exponent <= 4:
        raise ValueError("Scale32 exponent is outside -24..4")
    return significand | ((exponent & 0xFF) << 16)


def unpack_scale32(record: int) -> tuple[int, int]:
    if not 0 <= record <= 0xFFFFFFFF or record >> 24:
        raise ValueError("Scale32 reserved byte must be zero")
    significand = record & 0xFFFF
    exponent_u8 = (record >> 16) & 0xFF
    exponent = exponent_u8 - 256 if exponent_u8 & 0x80 else exponent_u8
    pack_scale32(significand, exponent)
    return significand, exponent


def round_divide_ties_to_even(numerator: int, denominator: int) -> int:
    if denominator <= 0:
        raise ValueError("denominator must be positive")
    quotient, remainder = divmod(abs(numerator), denominator)
    doubled = 2 * remainder
    quotient += int(doubled > denominator or (doubled == denominator and quotient & 1))
    return -quotient if numerator < 0 else quotient


def fuse_lane(
    accumulator_s32: int,
    residual_s8: int,
    accumulator_scale32: int,
    residual_scale32: int,
    destination_scale32: int,
) -> FusionLaneResult:
    if not SIGNED32_MIN <= accumulator_s32 <= SIGNED32_MAX:
        raise OverflowError("accumulator is outside signed-32")
    if not -128 <= residual_s8 <= 127:
        raise ValueError("residual is outside signed-int8")
    accumulator_sig, accumulator_exp = unpack_scale32(accumulator_scale32)
    residual_sig, residual_exp = unpack_scale32(residual_scale32)
    destination_sig, destination_exp = unpack_scale32(destination_scale32)
    common_exp = min(accumulator_exp, residual_exp, destination_exp)
    numerator = (
        accumulator_s32 * accumulator_sig * (1 << (accumulator_exp - common_exp))
        + residual_s8 * residual_sig * (1 << (residual_exp - common_exp))
    )
    denominator = destination_sig * (1 << (destination_exp - common_exp))
    if not SIGNED96_MIN <= numerator <= SIGNED96_MAX:
        raise OverflowError("numerator exceeds signed-96")
    if not 1 <= denominator <= UNSIGNED64_MAX:
        raise OverflowError("denominator exceeds unsigned-64")
    rounded = round_divide_ties_to_even(numerator, denominator)
    return FusionLaneResult(
        output_s8=max(-128, min(127, rounded)),
        numerator_s96=numerator,
        denominator_u64=denominator,
        common_exponent=common_exp,
        rounded_integer=rounded,
        positive_saturation=rounded > 127,
        negative_saturation=rounded < -128,
    )


def bounded_quotient_schedule(numerator: int, denominator: int) -> tuple[int, int, bool, bool]:
    """Return rounded magnitude, remainder, and exact positive/negative clamp flags."""
    if denominator <= 0:
        raise ValueError("denominator must be positive")
    magnitude = abs(numerator)
    positive_clamp = numerator >= 0 and 2 * magnitude >= 255 * denominator
    negative_clamp = numerator < 0 and 2 * magnitude > 257 * denominator
    if positive_clamp or negative_clamp:
        return 0, 0, positive_clamp, negative_clamp
    remainder = magnitude
    quotient = 0
    for bit in range(7, -1, -1):
        shifted = denominator << bit
        if remainder >= shifted:
            remainder -= shifted
            quotient |= 1 << bit
    increment = 2 * remainder > denominator or (
        2 * remainder == denominator and quotient & 1
    )
    return quotient + int(increment), remainder, False, False


def fuse_lane_schedule_model(*args: int) -> FusionLaneResult:
    exact = fuse_lane(*args)
    magnitude, _remainder, positive_clamp, negative_clamp = bounded_quotient_schedule(
        exact.numerator_s96, exact.denominator_u64
    )
    if positive_clamp:
        scheduled_output = 127
    elif negative_clamp:
        scheduled_output = -128
    else:
        scheduled_output = -magnitude if exact.numerator_s96 < 0 else magnitude
    if max(-128, min(127, scheduled_output)) != exact.output_s8:
        raise AssertionError("bounded quotient schedule differs from exact division")
    return exact


def workspace_proof() -> dict[str, int]:
    max_accumulator_term = (1 << 31) * 0xFFFF * (1 << 28)
    max_residual_term = 128 * 0xFFFF * (1 << 28)
    max_abs_numerator = max_accumulator_term + max_residual_term
    max_denominator = 0xFFFF * (1 << 28)
    if max_abs_numerator >= (1 << 95):
        raise AssertionError("signed-96 numerator proof failed")
    if max_denominator >= (1 << 64):
        raise AssertionError("unsigned-64 denominator proof failed")
    return {
        "max_abs_numerator_bound": max_abs_numerator,
        "max_denominator_bound": max_denominator,
        "numerator_required_magnitude_bits": max_abs_numerator.bit_length(),
        "denominator_required_bits": max_denominator.bit_length(),
    }
