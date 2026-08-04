#!/usr/bin/env python3
"""Independent fixed-point reference for the ACE-2 attention-score RTL slice."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

from ace2_quality_contracts import (
    dynamic_score_pair_parameters,
    round_divide_even_signed,
)


HEAD_DIM = 64
MAC_LANES = 1
CONTEXT_MAX = 8
BEATS_PER_VECTOR = HEAD_DIM // 16
SCORE_FRAC = 9
ROPE_CONVERSION_Q9 = 359
INT32_MAX = (1 << 31) - 1
METADATA_WIDTH = 128


@dataclass(frozen=True)
class AttentionScoreCase:
    name: str
    q_values: list[int]
    k_values: list[list[int]]
    query_scale: float = 1.0 / 32.0
    key_scale: float = 1.0 / 16.0


@dataclass(frozen=True)
class AttentionScoreResult:
    accumulators: list[int]
    core_scores_q6_9: list[int]
    scores_q6_9: list[int]
    core_saturation_by_token: list[bool]
    saturation_by_token: list[bool]
    saturation_seen: bool
    multiplier: int
    right_shift: int


@dataclass(frozen=True)
class DynamicAttentionScoreCase:
    name: str
    q_values: list[int]
    k_values: list[list[int]]
    query_scale32: int
    key_scale32: list[int]
    valid_keys: list[bool] | None = None


@dataclass(frozen=True)
class DynamicAttentionScoreResult:
    accumulators: list[int]
    pair_significands: list[int]
    right_shifts: list[int]
    precenter_scores: list[int]
    scores_q6_9: list[int]
    saturation_by_token: list[bool]
    saturation_seen: bool


def to_sint(value: int, width: int) -> int:
    mask = (1 << width) - 1
    value &= mask
    sign = 1 << (width - 1)
    return value - (1 << width) if value & sign else value


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


def saturate(value: int, low: int, high: int) -> tuple[int, bool]:
    if value > high:
        return high, True
    if value < low:
        return low, True
    return value, False


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


def derive_score_requantization(query_scale: float, key_scale: float) -> tuple[int, int]:
    if (
        not math.isfinite(query_scale)
        or query_scale <= 0
        or not math.isfinite(key_scale)
        or key_scale <= 0
    ):
        raise ValueError("attention Q/K scales must be finite and positive")
    real_multiplier = (
        query_scale * key_scale * (1 << SCORE_FRAC) / math.sqrt(HEAD_DIM)
    )
    for right_shift in range(63, -1, -1):
        multiplier = round(real_multiplier * math.ldexp(1.0, right_shift))
        if multiplier <= INT32_MAX:
            return multiplier, right_shift
    raise OverflowError("attention score multiplier is not signed-int32 representable")


def pack_score_metadata(
    multiplier: int,
    right_shift: int,
    *,
    query_head: int = 0,
    conversion_q9: int = ROPE_CONVERSION_Q9,
) -> int:
    if not 1 <= multiplier <= INT32_MAX:
        raise ValueError("attention score multiplier must be positive signed-int32")
    if not 0 <= right_shift <= 63:
        raise ValueError("attention score right shift must be unsigned 6-bit")
    if not 0 <= query_head < 14:
        raise ValueError("attention query-head ID must be in the range 0..13")
    if not 1 <= conversion_q9 <= 32767:
        raise ValueError("RoPE conversion must be positive signed-int16 Q6.9")
    mapped_kv_head = query_head // 7
    return (
        conversion_q9
        | (multiplier << 16)
        | (right_shift << 48)
        | (mapped_kv_head << 54)
        | (query_head << 55)
    )


def reference_attention_score(case: AttentionScoreCase) -> AttentionScoreResult:
    multiplier, right_shift = derive_score_requantization(
        case.query_scale,
        case.key_scale,
    )
    accumulators: list[int] = []
    scaled_scores: list[int] = []
    core_scores: list[int] = []
    core_saturation_by_token: list[bool] = []
    for k_vector in case.k_values:
        acc = 0
        for q_value, k_value in zip(case.q_values, k_vector, strict=True):
            acc += to_sint(q_value, 8) * to_sint(k_value, 8)
        rounded = round_shift_even(acc * multiplier, right_shift)
        core_score, core_saturated = saturate(rounded, -32768, 32767)
        accumulators.append(acc)
        scaled_scores.append(rounded)
        core_scores.append(core_score)
        core_saturation_by_token.append(core_saturated)
    row_max = max(scaled_scores)
    scores: list[int] = []
    saturation_by_token: list[bool] = []
    for scaled_score in scaled_scores:
        score, saturated = saturate(scaled_score - row_max, -32768, 0)
        scores.append(score)
        saturation_by_token.append(saturated)
    return AttentionScoreResult(
        accumulators=accumulators,
        core_scores_q6_9=core_scores,
        scores_q6_9=scores,
        core_saturation_by_token=core_saturation_by_token,
        saturation_by_token=saturation_by_token,
        saturation_seen=any(saturation_by_token),
        multiplier=multiplier,
        right_shift=right_shift,
    )


def reference_dynamic_attention_score(
    case: DynamicAttentionScoreCase,
) -> DynamicAttentionScoreResult:
    if len(case.q_values) != HEAD_DIM:
        raise ValueError("dynamic attention score requires one 64-lane query head")
    if len(case.k_values) != len(case.key_scale32):
        raise ValueError("every key vector requires one Scale32 record")
    valid_keys = (
        [True] * len(case.k_values) if case.valid_keys is None else case.valid_keys
    )
    if len(valid_keys) != len(case.k_values) or not any(valid_keys):
        raise ValueError("dynamic attention score requires at least one valid key")

    accumulators: list[int] = []
    pair_significands: list[int] = []
    right_shifts: list[int] = []
    precenter_scores: list[int] = []
    for key_vector, key_record in zip(
        case.k_values, case.key_scale32, strict=True
    ):
        if len(key_vector) != HEAD_DIM:
            raise ValueError("dynamic attention key head must contain 64 lanes")
        accumulator = sum(
            to_sint(q_value, 8) * to_sint(k_value, 8)
            for q_value, k_value in zip(case.q_values, key_vector, strict=True)
        )
        if not -1_032_256 <= accumulator <= 1_032_256:
            raise OverflowError("dynamic attention signed-int32 dot bound was exceeded")
        pair_sig, right_shift = dynamic_score_pair_parameters(
            case.query_scale32,
            key_record,
        )
        product = accumulator * pair_sig
        if not -(1 << 49) <= product < (1 << 49):
            raise OverflowError("dynamic attention signed-50-bit product overflow")
        precenter = round_divide_even_signed(product, 1 << right_shift)
        accumulators.append(accumulator)
        pair_significands.append(pair_sig)
        right_shifts.append(right_shift)
        precenter_scores.append(precenter)

    row_max = max(
        score for score, valid in zip(precenter_scores, valid_keys, strict=True) if valid
    )
    scores: list[int] = []
    saturation_by_token: list[bool] = []
    for precenter, valid in zip(precenter_scores, valid_keys, strict=True):
        if not valid:
            scores.append(-32768)
            saturation_by_token.append(False)
            continue
        centered = precenter - row_max
        score, saturated = saturate(centered, -32768, 0)
        scores.append(score)
        saturation_by_token.append(saturated)

    return DynamicAttentionScoreResult(
        accumulators=accumulators,
        pair_significands=pair_significands,
        right_shifts=right_shifts,
        precenter_scores=precenter_scores,
        scores_q6_9=scores,
        saturation_by_token=saturation_by_token,
        saturation_seen=any(saturation_by_token),
    )
