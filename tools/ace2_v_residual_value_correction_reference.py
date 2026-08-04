#!/usr/bin/env python3
"""Exact integer reference for shared V-residual attention-value correction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


LAYERS = 24
QUERY_HEADS = 14
KV_HEADS = 2
HEAD_DIM = 64
CONTEXT_MAX = 32768


def check_signed(value: int, bits: int, label: str) -> int:
    minimum = -(1 << (bits - 1))
    maximum = (1 << (bits - 1)) - 1
    if not minimum <= value <= maximum:
        raise OverflowError(f"{label} does not fit signed-{bits}: {value}")
    return value


def check_unsigned(value: int, bits: int, label: str) -> int:
    if not 0 <= value < (1 << bits):
        raise OverflowError(f"{label} does not fit unsigned-{bits}: {value}")
    return value


def unpack_scale32(record: int) -> tuple[int, int]:
    check_unsigned(record, 32, "Scale32 record")
    if record >> 24:
        raise ValueError("Scale32 reserved byte must be zero")
    significand = record & 0xFFFF
    exponent_u8 = (record >> 16) & 0xFF
    exponent = exponent_u8 - 256 if exponent_u8 & 0x80 else exponent_u8
    if not 0x8000 <= significand <= 0xFFFF:
        raise ValueError("Scale32 significand must be normalized")
    if not -24 <= exponent <= 4:
        raise ValueError("Scale32 exponent is outside the frozen range")
    return significand, exponent


def round_shift_even_signed(value: int, shift: int) -> int:
    if not 0 <= shift <= 63:
        raise ValueError("right shift must be 0..63")
    if shift == 0:
        return value
    magnitude = abs(value)
    quotient, remainder = divmod(magnitude, 1 << shift)
    half = 1 << (shift - 1)
    if remainder > half or (remainder == half and (quotient & 1)):
        quotient += 1
    return -quotient if value < 0 else quotient


def round_div_even_signed(numerator: int, denominator: int) -> int:
    if denominator <= 0:
        raise ValueError("division denominator must be positive")
    quotient, remainder = divmod(abs(numerator), denominator)
    doubled = remainder * 2
    if doubled > denominator or (doubled == denominator and (quotient & 1)):
        quotient += 1
    return -quotient if numerator < 0 else quotient


def saturate_s8(value: int) -> int:
    return max(-128, min(127, value))


def canonical_s4_byte(value: int) -> int:
    if not -7 <= value <= 7:
        raise ValueError("canonical residual V must be signed-4 [-7,+7]")
    return value & 0xFF


def decode_canonical_s4_byte(value: int) -> int:
    check_unsigned(value, 8, "canonical residual V byte")
    low = value & 0xF
    signed = low - 16 if low & 0x8 else low
    if signed == -8:
        raise ValueError("signed-4 code -8 is reserved")
    expected = signed & 0xFF
    if value != expected:
        raise ValueError("residual V byte is not canonical sign extension")
    return signed


@dataclass(frozen=True)
class VResidualProjectionResult:
    baseline_v8: int
    residual_v_s4: int
    residual_v_canonical_u8: int
    positive_clamp: bool
    negative_clamp: bool
    product_s64: int
    error_s72: int


def v_projection_residual(
    accumulator_s32: int,
    multiplier_s32: int,
    shift_u6: int,
    baseline_v_scale32: int,
    residual_v_scale32: int,
) -> VResidualProjectionResult:
    check_signed(accumulator_s32, 32, "V projection accumulator")
    if not 0 < multiplier_s32 < (1 << 31):
        raise ValueError("V projection multiplier must be positive signed-32")
    check_unsigned(shift_u6, 6, "V projection right shift")
    baseline_sig, baseline_exp = unpack_scale32(baseline_v_scale32)
    residual_sig, residual_exp = unpack_scale32(residual_v_scale32)

    product = check_signed(
        accumulator_s32 * multiplier_s32,
        64,
        "V projection product",
    )
    baseline_unclamped = round_shift_even_signed(product, shift_u6)
    baseline_v8 = saturate_s8(baseline_unclamped)
    error = check_signed(
        product - (baseline_v8 << shift_u6),
        72,
        "V projection exact remainder",
    )
    delta = baseline_exp - residual_exp - shift_u6
    if not -91 <= delta <= 28:
        raise ValueError("V residual Scale32 delta is outside the frozen range")
    numerator = error * baseline_sig
    denominator = residual_sig
    if delta >= 0:
        numerator <<= delta
    else:
        denominator <<= -delta
    check_signed(numerator, 116, "V residual numerator")
    check_unsigned(denominator, 107, "V residual denominator")
    unclamped = round_div_even_signed(numerator, denominator)
    positive_clamp = unclamped > 7
    negative_clamp = unclamped < -7
    residual = max(-7, min(7, unclamped))
    return VResidualProjectionResult(
        baseline_v8=baseline_v8,
        residual_v_s4=residual,
        residual_v_canonical_u8=canonical_s4_byte(residual),
        positive_clamp=positive_clamp,
        negative_clamp=negative_clamp,
        product_s64=product,
        error_s72=error,
    )


@dataclass(frozen=True)
class VResidualValueCorrectionResult:
    correction_raw_s64: int
    correction_baseline_domain_s32: int
    corrected_accumulator_s32: int


def v_residual_value_correction(
    probabilities_q0_15: Sequence[int],
    residual_v_canonical_u8: Sequence[int],
    baseline_v_scale32: int,
    residual_v_scale32: int,
    authoritative_baseline_accumulator_s32: int,
) -> VResidualValueCorrectionResult:
    lane_count = len(probabilities_q0_15)
    if not 1 <= lane_count <= CONTEXT_MAX:
        raise ValueError("V correction lane count must be 1..32768")
    if len(residual_v_canonical_u8) != lane_count:
        raise ValueError("one residual V byte is required per probability")
    baseline_sig, baseline_exp = unpack_scale32(baseline_v_scale32)
    residual_sig, residual_exp = unpack_scale32(residual_v_scale32)
    baseline = check_signed(
        authoritative_baseline_accumulator_s32,
        32,
        "authoritative baseline attention-value accumulator",
    )

    correction_raw = 0
    for probability, residual_byte in zip(
        probabilities_q0_15,
        residual_v_canonical_u8,
        strict=True,
    ):
        check_unsigned(probability, 16, "Q0.15 probability")
        residual = decode_canonical_s4_byte(residual_byte)
        correction_raw = check_signed(
            correction_raw + probability * residual,
            64,
            "V residual correction accumulator",
        )

    numerator = correction_raw * residual_sig
    denominator = baseline_sig
    delta = residual_exp - baseline_exp
    if delta >= 0:
        numerator <<= delta
    else:
        denominator <<= -delta
    check_signed(numerator, 108, "V correction Scale32 numerator")
    check_unsigned(denominator, 44, "V correction Scale32 denominator")
    correction = check_signed(
        round_div_even_signed(numerator, denominator),
        32,
        "V correction in baseline accumulator domain",
    )
    corrected = check_signed(
        baseline + correction,
        32,
        "corrected attention-value accumulator",
    )
    return VResidualValueCorrectionResult(
        correction_raw_s64=correction_raw,
        correction_baseline_domain_s32=correction,
        corrected_accumulator_s32=corrected,
    )


def kv_head_for_query_head(query_head: int) -> int:
    if not 0 <= query_head < QUERY_HEADS:
        raise ValueError("query head must be 0..13")
    return query_head // 7


def validate_layer_head(layer_id: int, query_head: int, kv_head: int) -> None:
    if not 0 <= layer_id < LAYERS:
        raise ValueError("layer_id must be 0..23")
    if kv_head_for_query_head(query_head) != kv_head:
        raise ValueError("query/KV head mapping violates the frozen 14:2 mapping")
