#!/usr/bin/env python3
"""Independent integer oracle for layer0_tile_max_delta_attention_v1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

try:
    from ace2_absolute_rope_online_attention_reference import (
        absolute_coefficients_q15,
        exp_q31,
    )
    from ace2_quality_contracts import round_divide_even_signed, unpack_scale32
except ModuleNotFoundError:
    from tools.ace2_absolute_rope_online_attention_reference import (
        absolute_coefficients_q15,
        exp_q31,
    )
    from tools.ace2_quality_contracts import round_divide_even_signed, unpack_scale32


CONTRACT_ID = "layer0_tile_max_delta_attention_v1"
HEAD_DIM = 64
PAIR_COUNT = 32
TILE_KEYS = 64
Q_SCALE32 = 0x00FFA245
K_SCALE32 = 0x00008307
DELTA_FRAC = 17
DELTA_SENTINEL = -(16 << DELTA_FRAC)
PROB_FRAC = 15


def check_signed(value: int, width: int, label: str) -> int:
    if not -(1 << (width - 1)) <= value < (1 << (width - 1)):
        raise OverflowError(f"{label} does not fit signed-{width}")
    return value


def rne_shift(value: int, shift: int) -> int:
    if shift < 0:
        return value << -shift
    return round_divide_even_signed(value, 1 << shift)


def rotate_shadow_head(values: Sequence[int], position: int) -> tuple[int, ...]:
    if len(values) != HEAD_DIM:
        raise ValueError("tile-max RoPE requires 64 Q15.16 lanes")
    cosine, sine = absolute_coefficients_q15(position)
    rotated = [0] * HEAD_DIM
    for pair in range(PAIR_COUNT):
        low = check_signed(int(values[pair]), 32, "shadow lane")
        high = check_signed(int(values[pair + PAIR_COUNT]), 32, "shadow lane")
        real = rne_shift(low * cosine[pair] - high * sine[pair], 15)
        imag = rne_shift(high * cosine[pair] + low * sine[pair], 15)
        rotated[pair] = check_signed(real, 34, "rotated real")
        rotated[pair + PAIR_COUNT] = check_signed(imag, 34, "rotated imag")
    return tuple(rotated)


def score_numerator_s106(
    query_shadow: Sequence[int],
    key_shadow: Sequence[int],
    query_position: int,
    key_position: int,
    query_scale32: int = Q_SCALE32,
    key_scale32: int = K_SCALE32,
) -> int:
    if not 0 <= key_position <= query_position <= 32767:
        raise ValueError("tile-max positions violate causal range")
    query = rotate_shadow_head(query_shadow, query_position)
    key = rotate_shadow_head(key_shadow, key_position)
    dot = check_signed(
        sum(q * k for q, k in zip(query, key, strict=True)),
        74,
        "shadow dot",
    )
    q_sig, _ = unpack_scale32(query_scale32)
    k_sig, _ = unpack_scale32(key_scale32)
    return check_signed(dot * q_sig * k_sig, 106, "score numerator")


def encode_score_row_from_numerators(
    numerators: Sequence[int],
    *,
    query_scale32: int = Q_SCALE32,
    key_scale32: int = K_SCALE32,
) -> tuple[int, ...]:
    if not numerators:
        raise ValueError("tile-max row must be nonempty")
    for value in numerators:
        check_signed(int(value), 106, "score numerator")
    _, q_exp = unpack_scale32(query_scale32)
    _, k_exp = unpack_scale32(key_scale32)
    shift = 48 - (q_exp + k_exp)
    if not 40 <= shift <= 105:
        raise ValueError("Q6.17 score shift is outside 40..105")
    tile_maxima = [
        max(int(value) for value in numerators[start : start + TILE_KEYS])
        for start in range(0, len(numerators), TILE_KEYS)
    ]
    row_maximum = max(tile_maxima)
    encoded: list[int] = []
    for tile_index, start in enumerate(range(0, len(numerators), TILE_KEYS)):
        tile_maximum = tile_maxima[tile_index]
        tile_offset = rne_shift(tile_maximum - row_maximum, shift)
        for value in numerators[start : start + TILE_KEYS]:
            local_delta = rne_shift(int(value) - tile_maximum, shift)
            merged = tile_offset + local_delta
            if merged > 0:
                raise OverflowError("hierarchical delta became positive")
            encoded.append(
                DELTA_SENTINEL if merged <= DELTA_SENTINEL else check_signed(merged, 24, "Q6.17 delta")
            )
    return tuple(encoded)


def softmax_q15_from_q6_17(scores: Sequence[int]) -> tuple[int, ...]:
    if not scores:
        raise ValueError("softmax requires a nonempty row")
    weights = []
    for score in scores:
        value = check_signed(int(score), 24, "Q6.17 score")
        if value > 0:
            raise ValueError("centered score must be nonpositive")
        weights.append(0 if value <= DELTA_SENTINEL else exp_q31(value << 3))
    total = sum(weights)
    if total <= 0:
        raise RuntimeError("softmax denominator is zero")
    return tuple(
        min(32767, round_divide_even_signed(weight * 32767, total))
        for weight in weights
    )


def attention_value_s8(
    probabilities_q15: Sequence[int], values: Sequence[Sequence[int]]
) -> tuple[int, ...]:
    if len(probabilities_q15) != len(values) or not values:
        raise ValueError("attention value requires matched nonempty rows")
    output = []
    for lane in range(HEAD_DIM):
        accumulator = sum(
            int(probabilities_q15[index]) * int(values[index][lane])
            for index in range(len(values))
        )
        rounded = rne_shift(accumulator, PROB_FRAC)
        output.append(max(-128, min(127, rounded)))
    return tuple(output)


@dataclass(frozen=True)
class TileMaxDeltaResult:
    scores_q6_17: tuple[int, ...]
    probabilities_q15: tuple[int, ...]
    output_s8: tuple[int, ...]


def staged_attention_row(
    query_shadow: Sequence[int],
    key_shadows: Sequence[Sequence[int]],
    values: Sequence[Sequence[int]],
    query_position: int,
) -> TileMaxDeltaResult:
    if len(key_shadows) != query_position + 1 or len(values) != query_position + 1:
        raise ValueError("tile-max row requires all causal keys")
    numerators = tuple(
        score_numerator_s106(
            query_shadow,
            key,
            query_position,
            key_position,
        )
        for key_position, key in enumerate(key_shadows)
    )
    scores = encode_score_row_from_numerators(numerators)
    probabilities = softmax_q15_from_q6_17(scores)
    return TileMaxDeltaResult(
        scores_q6_17=scores,
        probabilities_q15=probabilities,
        output_s8=attention_value_s8(probabilities, values),
    )
