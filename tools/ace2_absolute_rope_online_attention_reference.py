#!/usr/bin/env python3
"""Independent integer oracle for layer0_absolute_rope_online_attention_v1."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, Sequence

import gmpy2

try:
    from ace2_quality_contracts import round_divide_even_signed, unpack_scale32
except ModuleNotFoundError:
    from tools.ace2_quality_contracts import round_divide_even_signed, unpack_scale32


CONTRACT_ID = "layer0_absolute_rope_online_attention_v1"
HEAD_DIM = 64
PAIR_COUNT = 32
Q_SCALE32 = 0x00FFA245
K_SCALE32 = 0x00008307
V_SCALE32 = 0x00F78C19
Q31_ONE = 1 << 31
LOGIT_FRAC = 20
EXP_LIMIT = 16 << LOGIT_FRAC


def _ratio(value: gmpy2.mpfr) -> tuple[int, int]:
    return tuple(map(int, value.as_integer_ratio()))


def _check_signed(value: int, width: int, label: str) -> int:
    if not -(1 << (width - 1)) <= value < (1 << (width - 1)):
        raise OverflowError(f"{label} does not fit signed-{width}")
    return value


def _check_unsigned(value: int, width: int, label: str) -> int:
    if not 0 <= value < (1 << width):
        raise OverflowError(f"{label} does not fit unsigned-{width}")
    return value


@lru_cache(maxsize=32768)
def absolute_coefficients_q15(position: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Generate the frozen F32 -> BF16 -> signed-Q1.15 coefficient record."""
    if not 0 <= position <= 32767:
        raise ValueError("absolute RoPE position is outside 0..32767")
    cosine: list[int] = []
    sine: list[int] = []
    for pair in range(PAIR_COUNT):
        with gmpy2.context(
            gmpy2.get_context(), precision=24, round=gmpy2.RoundToNearest
        ):
            alpha = gmpy2.mpfr(2 * pair) / gmpy2.mpfr(HEAD_DIM)
            omega = gmpy2.mpfr(1) / (gmpy2.mpfr(1_000_000) ** alpha)
            angle = gmpy2.mpfr(position) * omega
            cos_f32 = gmpy2.cos(angle)
            sin_f32 = gmpy2.sin(angle)
        with gmpy2.context(
            gmpy2.get_context(), precision=8, round=gmpy2.RoundToNearest
        ):
            cos_bf16 = gmpy2.mpfr(cos_f32)
            sin_bf16 = gmpy2.mpfr(sin_f32)
        cos_n, cos_d = _ratio(cos_bf16)
        sin_n, sin_d = _ratio(sin_bf16)
        cosine.append(
            max(-32768, min(32767, round_divide_even_signed(32767 * cos_n, cos_d)))
        )
        sine.append(
            max(-32768, min(32767, round_divide_even_signed(32767 * sin_n, sin_d)))
        )
    return tuple(cosine), tuple(sine)


def rotate_split_half(
    values: Sequence[int], cosine: Sequence[int], sine: Sequence[int]
) -> tuple[int, ...]:
    if len(values) != HEAD_DIM or len(cosine) != PAIR_COUNT or len(sine) != PAIR_COUNT:
        raise ValueError("wide RoPE requires 64 values and 32 coefficient pairs")
    if any(not -128 <= value <= 127 for value in values):
        raise ValueError("wide RoPE input must be signed int8")
    rotated = [0] * HEAD_DIM
    for pair in range(PAIR_COUNT):
        low = int(values[pair])
        high = int(values[pair + PAIR_COUNT])
        real = low * int(cosine[pair]) - high * int(sine[pair])
        imag = high * int(cosine[pair]) + low * int(sine[pair])
        rotated[pair] = _check_signed(real, 25, "wide RoPE real")
        rotated[pair + PAIR_COUNT] = _check_signed(imag, 25, "wide RoPE imag")
    return tuple(rotated)


def absolute_rope_score_raw(
    query: Sequence[int],
    key: Sequence[int],
    query_position: int,
    key_position: int,
) -> tuple[int, tuple[int, ...], tuple[int, ...]]:
    if not 0 <= key_position <= query_position <= 32767:
        raise ValueError("key/query positions violate the causal absolute-RoPE range")
    q_cos, q_sin = absolute_coefficients_q15(query_position)
    k_cos, k_sin = absolute_coefficients_q15(key_position)
    rotated_query = rotate_split_half(query, q_cos, q_sin)
    rotated_key = rotate_split_half(key, k_cos, k_sin)
    score = sum(q * k for q, k in zip(rotated_query, rotated_key, strict=True))
    return _check_signed(score, 54, "wide score"), rotated_query, rotated_key


def score_raw_to_logit_q12_20(
    score_raw: int, query_scale32: int = Q_SCALE32, key_scale32: int = K_SCALE32
) -> int:
    _check_signed(score_raw, 54, "wide score")
    query_sig, query_exp = unpack_scale32(query_scale32)
    key_sig, key_exp = unpack_scale32(key_scale32)
    shift = 43 - (query_exp + key_exp)
    if not 35 <= shift <= 91:
        raise ValueError("absolute-RoPE score shift is outside 35..91")
    scaled = score_raw * query_sig * key_sig
    return _check_signed(
        round_divide_even_signed(scaled, 1 << shift), 64, "Q12.20 logit"
    )


@lru_cache(maxsize=1)
def exp_table_q31() -> tuple[int, ...]:
    table: list[int] = []
    for index in range(257):
        with gmpy2.context(
            gmpy2.get_context(), precision=53, round=gmpy2.RoundToNearest
        ):
            value = gmpy2.exp(-gmpy2.mpfr(index) / gmpy2.mpfr(16))
        numerator, denominator = _ratio(value)
        table.append(round_divide_even_signed(numerator * Q31_ONE, denominator))
    if table[0] != Q31_ONE or table[-1] < 0:
        raise AssertionError("invalid Q1.31 exponential table endpoints")
    return tuple(table)


def exp_q31(delta_q12_20: int) -> int:
    if delta_q12_20 >= 0:
        return Q31_ONE
    if delta_q12_20 <= -EXP_LIMIT:
        return 0
    magnitude = -delta_q12_20
    index = magnitude >> 16
    fraction = magnitude & 0xFFFF
    table = exp_table_q31()
    correction = round_divide_even_signed(
        (table[index + 1] - table[index]) * fraction, 1 << 16
    )
    return _check_unsigned(table[index] + correction, 32, "Q1.31 exponential")


@dataclass(frozen=True)
class OnlineAttentionState:
    maximum: int | None = None
    denominator: int = 0
    numerators: tuple[int, ...] = (0,) * HEAD_DIM

    @property
    def valid(self) -> bool:
        return self.maximum is not None


def update_online_state(
    state: OnlineAttentionState, logit_q12_20: int, value: Sequence[int]
) -> OnlineAttentionState:
    _check_signed(logit_q12_20, 64, "Q12.20 logit")
    if len(value) != HEAD_DIM or any(not -128 <= lane <= 127 for lane in value):
        raise ValueError("online attention value must be one signed-int8 head")
    if len(state.numerators) != HEAD_DIM:
        raise ValueError("online attention state must contain 64 numerators")
    if not state.valid:
        return OnlineAttentionState(
            maximum=logit_q12_20,
            denominator=Q31_ONE,
            numerators=tuple(
                _check_signed(int(lane) * Q31_ONE, 56, "numerator") for lane in value
            ),
        )

    assert state.maximum is not None
    _check_unsigned(state.denominator, 48, "denominator")
    for numerator in state.numerators:
        _check_signed(numerator, 56, "numerator")
    if logit_q12_20 <= state.maximum:
        weight = exp_q31(logit_q12_20 - state.maximum)
        denominator = _check_unsigned(state.denominator + weight, 48, "denominator")
        numerators = tuple(
            _check_signed(old + weight * int(lane), 56, "numerator")
            for old, lane in zip(state.numerators, value, strict=True)
        )
        return OnlineAttentionState(state.maximum, denominator, numerators)

    rescale = exp_q31(state.maximum - logit_q12_20)
    denominator = _check_unsigned(
        round_divide_even_signed(state.denominator * rescale, Q31_ONE) + Q31_ONE,
        48,
        "denominator",
    )
    numerators = tuple(
        _check_signed(
            round_divide_even_signed(old * rescale, Q31_ONE) + int(lane) * Q31_ONE,
            56,
            "numerator",
        )
        for old, lane in zip(state.numerators, value, strict=True)
    )
    return OnlineAttentionState(logit_q12_20, denominator, numerators)


def finalize_online_state(state: OnlineAttentionState) -> tuple[int, ...]:
    if not state.valid or state.denominator == 0:
        raise ValueError("cannot finalize an empty online attention state")
    output = []
    for numerator in state.numerators:
        rounded = round_divide_even_signed(numerator, state.denominator)
        output.append(max(-128, min(127, rounded)))
    return tuple(output)


def online_attention_row(
    query: Sequence[int],
    keys: Iterable[Sequence[int]],
    values: Iterable[Sequence[int]],
    query_position: int,
    key_base: int = 0,
) -> tuple[tuple[int, ...], OnlineAttentionState, tuple[int, ...]]:
    key_rows = tuple(tuple(int(lane) for lane in key) for key in keys)
    value_rows = tuple(tuple(int(lane) for lane in value) for value in values)
    if len(key_rows) != len(value_rows) or not key_rows:
        raise ValueError("online attention row requires matched nonempty K/V rows")
    if key_base + len(key_rows) - 1 > query_position:
        raise ValueError("online attention row exceeds causal position")
    state = OnlineAttentionState()
    logits: list[int] = []
    for offset, (key, value) in enumerate(zip(key_rows, value_rows, strict=True)):
        score, _, _ = absolute_rope_score_raw(query, key, query_position, key_base + offset)
        logit = score_raw_to_logit_q12_20(score)
        logits.append(logit)
        state = update_online_state(state, logit, value)
    return finalize_online_state(state), state, tuple(logits)
