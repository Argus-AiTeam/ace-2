#!/usr/bin/env python3
"""Independent integer reference for the shared Q/K residual cross-term contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


CONTRACT_ID = "shared_qk_residual_cross_term_attention_v1"
SCALE32_SIG_MIN = 0x8000
SCALE32_SIG_MAX = 0xFFFF
SCALE32_EXP_MIN = -24
SCALE32_EXP_MAX = 4
HEAD_DIM = 64
QUERY_HEADS = 14
KV_HEADS = 2
LAYERS = 24


def check_signed(value: int, width: int, label: str) -> int:
    if not -(1 << (width - 1)) <= value < (1 << (width - 1)):
        raise OverflowError(f"{label} exceeds signed-{width}")
    return value


def check_unsigned(value: int, width: int, label: str) -> int:
    if not 0 <= value < (1 << width):
        raise OverflowError(f"{label} exceeds unsigned-{width}")
    return value


def unpack_scale32(record: int) -> tuple[int, int]:
    check_unsigned(record, 32, "Scale32")
    if record >> 24:
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


def round_div_even_signed(numerator: int, denominator: int) -> int:
    if denominator <= 0:
        raise ValueError("rounding denominator must be positive")
    quotient, remainder = divmod(abs(numerator), denominator)
    doubled = 2 * remainder
    if doubled > denominator or (doubled == denominator and quotient & 1):
        quotient += 1
    return -quotient if numerator < 0 else quotient


def round_shift_even_signed(value: int, shift: int) -> int:
    if shift <= 0:
        return value << -shift
    return round_div_even_signed(value, 1 << shift)


def saturate_s8(value: int) -> int:
    return max(-128, min(127, value))


@dataclass(frozen=True)
class ResidualProjectionResult:
    baseline_q8: int
    residual_s4: int
    positive_clamp: bool
    negative_clamp: bool
    product_s64: int
    error_s72: int


def projection_residual(
    accumulator_s32: int,
    multiplier_s32: int,
    shift_u6: int,
    baseline_scale32: int,
    residual_scale32: int,
) -> ResidualProjectionResult:
    check_signed(accumulator_s32, 32, "projection accumulator")
    if not 0 < multiplier_s32 < (1 << 31):
        raise ValueError("projection multiplier must be positive signed-32")
    check_unsigned(shift_u6, 6, "projection shift")
    baseline_sig, baseline_exp = unpack_scale32(baseline_scale32)
    residual_sig, residual_exp = unpack_scale32(residual_scale32)

    product = check_signed(accumulator_s32 * multiplier_s32, 64, "P")
    baseline_unclamped = round_shift_even_signed(product, shift_u6)
    baseline_q8 = saturate_s8(baseline_unclamped)
    error = check_signed(product - (baseline_q8 << shift_u6), 72, "E")
    delta = baseline_exp - residual_exp - shift_u6
    if not -91 <= delta <= 28:
        raise ValueError("Scale32/shift delta is outside the frozen range")

    numerator = error * baseline_sig
    denominator = residual_sig
    if delta >= 0:
        numerator <<= delta
    else:
        denominator <<= -delta
    check_signed(numerator, 116, "shifted residual numerator")
    check_unsigned(denominator, 107, "shifted residual denominator")
    residual_unclamped = round_div_even_signed(numerator, denominator)
    positive_clamp = residual_unclamped > 7
    negative_clamp = residual_unclamped < -7
    residual_s4 = max(-7, min(7, residual_unclamped))
    return ResidualProjectionResult(
        baseline_q8=baseline_q8,
        residual_s4=residual_s4,
        positive_clamp=positive_clamp,
        negative_clamp=negative_clamp,
        product_s64=product,
        error_s72=error,
    )


@dataclass(frozen=True)
class ResidualRopePair:
    real_s8: int
    imag_s8: int


def residual_rope_pair(
    real_s4: int,
    imag_s4: int,
    cosine_q1_15: int,
    sine_q1_15: int,
) -> ResidualRopePair:
    if real_s4 == -8 or imag_s4 == -8:
        raise ValueError("signed-4 code -8 is reserved")
    if not -7 <= real_s4 <= 7 or not -7 <= imag_s4 <= 7:
        raise ValueError("residual RoPE input must be signed-4 [-7,+7]")
    check_signed(cosine_q1_15, 16, "cosine")
    check_signed(sine_q1_15, 16, "sine")
    real_acc = check_signed(
        real_s4 * cosine_q1_15 - imag_s4 * sine_q1_15,
        22,
        "residual RoPE real accumulator",
    )
    imag_acc = check_signed(
        imag_s4 * cosine_q1_15 + real_s4 * sine_q1_15,
        22,
        "residual RoPE imag accumulator",
    )
    real_out = round_shift_even_signed(real_acc, 15)
    imag_out = round_shift_even_signed(imag_acc, 15)
    check_signed(real_out, 8, "residual RoPE real output")
    check_signed(imag_out, 8, "residual RoPE imag output")
    return ResidualRopePair(real_s8=real_out, imag_s8=imag_out)


@dataclass(frozen=True)
class CrossTermScoreResult:
    authoritative_base_score_q20_44_s64: int
    correction_dots_s32: tuple[int, int, int]
    correction_terms_q20_44_s67: tuple[int, int, int]
    score_q20_44_s64: int


def _scaled_dot_q20_44(dot: int, scale_a: int, scale_b: int) -> int:
    check_signed(dot, 32, "dot product")
    sig_a, exp_a = unpack_scale32(scale_a)
    sig_b, exp_b = unpack_scale32(scale_b)
    product = dot * sig_a * sig_b
    shift = exp_a + exp_b + 11  # -15-15-3+44
    scaled = product << shift if shift >= 0 else round_shift_even_signed(product, -shift)
    return check_signed(scaled, 67, "Q20.44 term")


def residual_cross_term_score(
    authoritative_base_score_q20_44_s64: int,
    query_q8: Sequence[int],
    key_q8: Sequence[int],
    query_residual_s8: Sequence[int],
    key_residual_s8: Sequence[int],
    query_scale32: int,
    key_scale32: int,
    query_residual_scale32: int,
    key_residual_scale32: int,
) -> CrossTermScoreResult:
    lanes = len(query_q8)
    if lanes != HEAD_DIM or not (
        len(key_q8) == len(query_residual_s8) == len(key_residual_s8) == lanes
    ):
        raise ValueError("cross-term score requires four matched 64-lane vectors")
    vectors = (query_q8, key_q8, query_residual_s8, key_residual_s8)
    for vector in vectors:
        for value in vector:
            check_signed(value, 8, "score lane")

    authoritative_base_score_q20_44_s64 = check_signed(
        authoritative_base_score_q20_44_s64, 64, "authoritative Q20.44 base score"
    )
    correction_dots = (
        sum(q * rk for q, rk in zip(query_q8, key_residual_s8, strict=True)),
        sum(rq * k for rq, k in zip(query_residual_s8, key_q8, strict=True)),
        sum(rq * rk for rq, rk in zip(query_residual_s8, key_residual_s8, strict=True)),
    )
    correction_dots = tuple(
        check_signed(value, 32, "correction dot product") for value in correction_dots
    )
    correction_terms = (
        _scaled_dot_q20_44(correction_dots[0], query_scale32, key_residual_scale32),
        _scaled_dot_q20_44(correction_dots[1], query_residual_scale32, key_scale32),
        _scaled_dot_q20_44(
            correction_dots[2], query_residual_scale32, key_residual_scale32
        ),
    )
    accumulator = check_signed(
        authoritative_base_score_q20_44_s64, 67, "scaled score accumulator"
    )
    for term in correction_terms:
        accumulator = check_signed(accumulator + term, 67, "scaled score accumulator")
    return CrossTermScoreResult(
        authoritative_base_score_q20_44_s64=authoritative_base_score_q20_44_s64,
        correction_dots_s32=correction_dots,
        correction_terms_q20_44_s67=correction_terms,
        score_q20_44_s64=check_signed(accumulator, 64, "Q20.44 score"),
    )


def kv_head_for_query_head(query_head: int) -> int:
    if not 0 <= query_head < QUERY_HEADS:
        raise ValueError("query head must be 0..13")
    return query_head // (QUERY_HEADS // KV_HEADS)


def validate_layer_head(layer_id: int, query_head: int, kv_head: int) -> None:
    if not 0 <= layer_id < LAYERS:
        raise ValueError("layer_id must be 0..23")
    if kv_head_for_query_head(query_head) != kv_head:
        raise ValueError("query/KV head mapping violates the frozen 14:2 mapping")


@dataclass(frozen=True)
class StagedAttentionResult:
    weights_q1_31: tuple[int, ...]
    probabilities_q0_15: tuple[int, ...]
    output_s8: tuple[int, ...]


def staged_softmax_attention_value(
    scores_q20_44: Sequence[int], values_s8: Sequence[Sequence[int]]
) -> StagedAttentionResult:
    """Frozen staged Q1.31 exp, Q0.15 normalize, and int8 V accumulation."""
    if not 1 <= len(scores_q20_44) <= 32768 or len(values_s8) != len(scores_q20_44):
        raise ValueError("staged attention requires 1..32768 matched score/value rows")
    if any(len(row) != HEAD_DIM for row in values_s8):
        raise ValueError("each attention-value row must contain 64 lanes")
    for score in scores_q20_44:
        check_signed(score, 64, "Q20.44 score")
    for row in values_s8:
        for value in row:
            check_signed(value, 8, "attention value")

    try:
        from ace2_absolute_rope_online_attention_reference import exp_q31
    except ModuleNotFoundError:
        from tools.ace2_absolute_rope_online_attention_reference import exp_q31

    maximum = max(scores_q20_44)
    weights = tuple(
        exp_q31(round_shift_even_signed(score - maximum, 24))
        for score in scores_q20_44
    )
    denominator = check_unsigned(sum(weights), 48, "Q1.31 softmax sum")
    if denominator == 0:
        raise RuntimeError("softmax denominator is zero")
    probabilities = tuple(
        check_unsigned(round_div_even_signed(weight << 15, denominator), 16, "Q0.15 probability")
        for weight in weights
    )
    output = []
    for lane in range(HEAD_DIM):
        accumulator = check_signed(
            sum(probabilities[index] * int(values_s8[index][lane])
                for index in range(len(probabilities))),
            32,
            "attention-value accumulator",
        )
        output.append(saturate_s8(round_shift_even_signed(accumulator, 15)))
    return StagedAttentionResult(weights, probabilities, tuple(output))
