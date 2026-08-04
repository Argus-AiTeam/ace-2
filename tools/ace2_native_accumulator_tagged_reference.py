#!/usr/bin/env python3
"""Independent integer reference for native-accumulator tagged attention."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence


CONTRACT_ID = "shared_native_accumulator_tagged_attention_v1"
SCALE32_SIG_MIN = 0x8000
SCALE32_SIG_MAX = 0xFFFF
SCALE32_EXP_MIN = -24
SCALE32_EXP_MAX = 4
TAG_EXP_MIN = -96
TAG_EXP_MAX = 31


def check_signed(value: int, width: int, label: str) -> int:
    if not -(1 << (width - 1)) <= value < (1 << (width - 1)):
        raise OverflowError(f"{label} exceeds signed-{width}")
    return value


def unpack_scale32(record: int) -> tuple[int, int]:
    if not 0 <= record <= 0xFFFFFFFF or record >> 24:
        raise ValueError("Scale32 reserved byte must be zero")
    significand = record & 0xFFFF
    exponent = (record >> 16) & 0xFF
    if exponent & 0x80:
        exponent -= 256
    if not SCALE32_SIG_MIN <= significand <= SCALE32_SIG_MAX:
        raise ValueError("Scale32 significand is not normalized")
    if not SCALE32_EXP_MIN <= exponent <= SCALE32_EXP_MAX:
        raise ValueError("Scale32 exponent is outside the frozen range")
    return significand, exponent


def round_shift_even(value: int, shift: int) -> int:
    if shift <= 0:
        return value << -shift
    magnitude = abs(value)
    quotient, remainder = divmod(magnitude, 1 << shift)
    half = 1 << (shift - 1)
    if remainder > half or (remainder == half and quotient & 1):
        quotient += 1
    return -quotient if value < 0 else quotient


@dataclass(frozen=True)
class TaggedLane:
    mantissa: int
    exponent: int


def normalize_tagged(value: int, base_exponent: int) -> TaggedLane:
    check_signed(value, 68, "RoPE aligned value")
    if value == 0:
        return TaggedLane(0, 0)
    shift = max(abs(value).bit_length() - 31, 0)
    while True:
        mantissa = round_shift_even(value, shift)
        if -(1 << 31) <= mantissa < (1 << 31):
            break
        shift += 1
    exponent = base_exponent + shift
    if not TAG_EXP_MIN <= exponent <= TAG_EXP_MAX:
        raise OverflowError("tagged exponent exceeds the frozen range")
    return TaggedLane(check_signed(mantissa, 32, "tagged mantissa"), exponent)


@dataclass(frozen=True)
class TaggedRopePair:
    real: TaggedLane
    imag: TaggedLane


def tagged_rope_pair(
    acc0: int,
    scale0: int,
    acc1: int,
    scale1: int,
    cosine_q15: int,
    sine_q15: int,
) -> TaggedRopePair:
    check_signed(acc0, 32, "acc0")
    check_signed(acc1, 32, "acc1")
    check_signed(cosine_q15, 16, "cosine")
    check_signed(sine_q15, 16, "sine")
    sig0, exp0 = unpack_scale32(scale0)
    sig1, exp1 = unpack_scale32(scale1)
    common_exp = max(exp0, exp1) - 30

    def aligned(acc: int, sig: int, coeff: int, exp: int) -> int:
        product = acc * sig * coeff
        return round_shift_even(product, max(exp0, exp1) - exp)

    real = aligned(acc0, sig0, cosine_q15, exp0) - aligned(
        acc1, sig1, sine_q15, exp1
    )
    imag = aligned(acc1, sig1, cosine_q15, exp1) + aligned(
        acc0, sig0, sine_q15, exp0
    )
    return TaggedRopePair(
        normalize_tagged(real, common_exp),
        normalize_tagged(imag, common_exp),
    )


def tagged_score_q20_44(
    queries: Sequence[TaggedLane], keys: Sequence[TaggedLane]
) -> int:
    if not queries or len(queries) != len(keys) or len(queries) > 64:
        raise ValueError("tagged score requires 1..64 matched lanes")
    products: list[tuple[int, int]] = []
    for query, key in zip(queries, keys, strict=True):
        check_signed(query.mantissa, 32, "query mantissa")
        check_signed(key.mantissa, 32, "key mantissa")
        if not TAG_EXP_MIN <= query.exponent <= TAG_EXP_MAX:
            raise ValueError("query exponent is outside the frozen range")
        if not TAG_EXP_MIN <= key.exponent <= TAG_EXP_MAX:
            raise ValueError("key exponent is outside the frozen range")
        products.append(
            (query.mantissa * key.mantissa, query.exponent + key.exponent)
        )
    common_exp = max(exponent for _, exponent in products)
    accumulator = sum(
        round_shift_even(product, common_exp - exponent)
        for product, exponent in products
    )
    check_signed(accumulator, 128, "tagged score accumulator")
    score = round_shift_even(accumulator, -(common_exp + 41))
    return check_signed(score, 64, "Q20.44 score")


def score_from_tuples(
    query: Iterable[tuple[int, int]], key: Iterable[tuple[int, int]]
) -> int:
    return tagged_score_q20_44(
        [TaggedLane(*item) for item in query],
        [TaggedLane(*item) for item in key],
    )

