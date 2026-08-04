#!/usr/bin/env python3
"""Exact scalar reference for the frozen cross-layer error-carry contract."""

from __future__ import annotations

from dataclasses import dataclass
from math import isqrt


HIDDEN_SIZE = 896
SIGNED16_MIN = -(1 << 15)
SIGNED16_MAX = (1 << 15) - 1
SIGNED24_MIN = -(1 << 23)
SIGNED24_MAX = (1 << 23) - 1
SIGNED32_MIN = -(1 << 31)
SIGNED32_MAX = (1 << 31) - 1
SIGNED96_MIN = -(1 << 95)
SIGNED96_MAX = (1 << 95) - 1
UNSIGNED64_MAX = (1 << 64) - 1
CARRY_MIN = -16384
CARRY_MAX = 16384
PRODUCER_LATENCY_CYCLES = 26


@dataclass(frozen=True)
class ProducerLaneResult:
    hidden_s8: int
    carry_s16_q15: int
    numerator_s96: int
    denominator_u64: int
    common_exponent: int
    latency_cycles: int = PRODUCER_LATENCY_CYCLES


@dataclass(frozen=True)
class RmsnormResult:
    outputs_s8: tuple[int, ...]
    reconstructed_q15: tuple[int, ...]
    sum_squares_q30: int
    mean_square_q30: int
    root_q15: int
    inverse_q30: int
    saturation_seen: bool


def accepted_consumer_completion_tag(
    start_valid: bool,
    start_ready: bool,
    completion_tag: int,
    subsequent_completion_tags: tuple[int, ...] = (),
) -> int:
    """Return the tag owned by one accepted consumer transaction.

    Later bus values are checked for width but cannot alter the accepted tag.
    """
    if not start_valid or not start_ready:
        raise ValueError("consumer start was not accepted")
    if not 0 <= completion_tag <= 0xFFFF:
        raise ValueError("completion tag is outside unsigned-16")
    if any(not 0 <= value <= 0xFFFF for value in subsequent_completion_tags):
        raise ValueError("subsequent completion tag is outside unsigned-16")
    return completion_tag


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


def round_divide_ties_to_even_signed(numerator: int, denominator: int) -> int:
    if denominator <= 0:
        raise ValueError("denominator must be positive")
    quotient, remainder = divmod(abs(numerator), denominator)
    doubled = remainder * 2
    quotient += int(doubled > denominator or (doubled == denominator and quotient & 1))
    return -quotient if numerator < 0 else quotient


def round_divide_ties_to_even_unsigned(numerator: int, denominator: int) -> int:
    if numerator < 0 or denominator <= 0:
        raise ValueError("unsigned division operands are invalid")
    quotient, remainder = divmod(numerator, denominator)
    doubled = remainder * 2
    return quotient + int(doubled > denominator or (doubled == denominator and quotient & 1))


def producer_lane(
    accumulator_s32: int,
    residual_s8: int,
    accumulator_scale32: int,
    residual_scale32: int,
    destination_scale32: int,
) -> ProducerLaneResult:
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
    hidden = round_divide_ties_to_even_signed(numerator, denominator)
    if not -128 <= hidden <= 127:
        raise OverflowError("carry mode forbids emitted-hidden saturation")
    error = numerator - hidden * denominator
    carry = round_divide_ties_to_even_signed(error << 15, denominator)
    if not CARRY_MIN <= carry <= CARRY_MAX:
        raise OverflowError("carry exceeds frozen signed-Q0.15 range")
    return ProducerLaneResult(
        hidden_s8=hidden,
        carry_s16_q15=carry,
        numerator_s96=numerator,
        denominator_u64=denominator,
        common_exponent=common_exp,
    )


def reconstruct_q15(hidden_s8: int, carry_s16_q15: int) -> int:
    if not -128 <= hidden_s8 <= 127:
        raise ValueError("hidden value is outside signed-int8")
    if not CARRY_MIN <= carry_s16_q15 <= CARRY_MAX:
        raise OverflowError("carry is outside the successful contract range")
    value = (hidden_s8 << 15) + carry_s16_q15
    if not SIGNED24_MIN <= value <= SIGNED24_MAX:
        raise OverflowError("reconstruction exceeds signed-24")
    return value


def rmsnorm(
    hidden_s8: list[int] | tuple[int, ...],
    carry_s16_q15: list[int] | tuple[int, ...],
    scaled_gain_s16_q8: list[int] | tuple[int, ...],
) -> RmsnormResult:
    """Apply carry-aware RMSNorm with output-scale-folded Q7.8 gains.

    ``scaled_gain_s16_q8`` is the frozen RMSNorm metadata
    ``round(weight / output_scale * 2**8)``. It is not the unscaled model
    weight encoded directly as Q7.8.
    """
    if not (
        len(hidden_s8) == len(carry_s16_q15) == len(scaled_gain_s16_q8)
    ):
        raise ValueError("RMSNorm vectors have different lengths")
    if not hidden_s8:
        raise ValueError("RMSNorm vector is empty")
    reconstructed = tuple(
        reconstruct_q15(hidden, carry)
        for hidden, carry in zip(hidden_s8, carry_s16_q15, strict=True)
    )
    if any(
        not SIGNED16_MIN <= gain <= SIGNED16_MAX
        for gain in scaled_gain_s16_q8
    ):
        raise ValueError("scaled gain is outside signed-16 Q7.8")
    sum_squares = sum(value * value for value in reconstructed)
    if sum_squares >= (1 << 56):
        raise OverflowError("sum of squares exceeds unsigned-56")
    mean_square = round_divide_ties_to_even_unsigned(sum_squares, len(reconstructed))
    floor_root = isqrt(mean_square)
    root = max(1, floor_root + int(floor_root * floor_root != mean_square))
    inverse = (1 << 45) // root
    outputs: list[int] = []
    saturation_seen = False
    for value, gain in zip(reconstructed, scaled_gain_s16_q8, strict=True):
        scaled = round_divide_ties_to_even_signed(value * gain * inverse, 1 << 53)
        saturation_seen |= scaled < -128 or scaled > 127
        outputs.append(max(-128, min(127, scaled)))
    return RmsnormResult(
        outputs_s8=tuple(outputs),
        reconstructed_q15=reconstructed,
        sum_squares_q30=sum_squares,
        mean_square_q30=mean_square,
        root_q15=root,
        inverse_q30=inverse,
        saturation_seen=saturation_seen,
    )


def workspace_proof() -> dict[str, int]:
    max_numerator = (1 << 31) * 0xFFFF * (1 << 28) + 128 * 0xFFFF * (1 << 28)
    max_denominator = 0xFFFF * (1 << 28)
    max_reconstruction = max(abs((-128 << 15) - 16384), abs((127 << 15) + 16384))
    max_sum_squares = HIDDEN_SIZE * max_reconstruction * max_reconstruction
    if max_numerator >= (1 << 95):
        raise AssertionError("signed-96 producer numerator proof failed")
    if max_denominator >= (1 << 64):
        raise AssertionError("unsigned-64 denominator proof failed")
    if max_reconstruction >= (1 << 23):
        raise AssertionError("signed-24 reconstruction proof failed")
    if max_sum_squares >= (1 << 56):
        raise AssertionError("unsigned-56 square-sum proof failed")
    return {
        "producer_numerator_magnitude_bits": max_numerator.bit_length(),
        "producer_denominator_bits": max_denominator.bit_length(),
        "reconstruction_magnitude_bits": max_reconstruction.bit_length(),
        "square_sum_bits": max_sum_squares.bit_length(),
    }
