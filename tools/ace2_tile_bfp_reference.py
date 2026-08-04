#!/usr/bin/env python3
"""Independent integer oracle for layer0_tile_bfp_score_attention_v1."""

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


CONTRACT_ID = "layer0_tile_bfp_score_attention_v1"
HEAD_DIM = 64
PAIR_COUNT = 32
TILE_KEYS = 64
Q_SCALE32 = 0x00FFA245
K_SCALE32 = 0x00008307
MANTISSA_BITS = 24
FRACTION_BITS_MIN = 0
FRACTION_BITS_MAX = 17
SCORE_FRAC = 17
EXP_UNDERFLOW_Q17 = -(16 << SCORE_FRAC)
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
        raise ValueError("tile-BFP RoPE requires 64 Q15.16 lanes")
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
        raise ValueError("tile-BFP positions violate causal range")
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


@dataclass(frozen=True)
class EncodedTile:
    tile_max_s128: int
    tile_min_s128: int
    fraction_bits_u5: int
    mantissas_s24: tuple[int, ...]


def encode_tile(
    numerators: Sequence[int],
    *,
    score_exponent: int = -1,
) -> EncodedTile:
    if not 1 <= len(numerators) <= TILE_KEYS:
        raise ValueError("tile-BFP tile must contain 1..64 scores")
    checked = tuple(check_signed(int(value), 106, "score numerator") for value in numerators)
    tile_maximum = max(checked)
    tile_minimum = min(checked)
    score_range = tile_maximum - tile_minimum
    selected_fraction: int | None = None
    for fraction_bits in range(FRACTION_BITS_MAX, FRACTION_BITS_MIN - 1, -1):
        shift = 65 - score_exponent - fraction_bits
        if rne_shift(score_range, shift) <= (1 << (MANTISSA_BITS - 1)) - 1:
            selected_fraction = fraction_bits
            break
    if selected_fraction is None:
        raise OverflowError("tile-BFP score range cannot fit signed-24")
    shift = 65 - score_exponent - selected_fraction
    mantissas = tuple(
        check_signed(rne_shift(value - tile_maximum, shift), MANTISSA_BITS, "BFP mantissa")
        for value in checked
    )
    if any(value > 0 for value in mantissas):
        raise OverflowError("tile-BFP mantissa became positive")
    return EncodedTile(
        tile_max_s128=check_signed(tile_maximum, 128, "tile maximum"),
        tile_min_s128=check_signed(tile_minimum, 128, "tile minimum"),
        fraction_bits_u5=selected_fraction,
        mantissas_s24=mantissas,
    )


def encode_score_row_from_numerators(
    numerators: Sequence[int],
    *,
    query_scale32: int = Q_SCALE32,
    key_scale32: int = K_SCALE32,
) -> tuple[int, ...]:
    if not numerators:
        raise ValueError("tile-BFP row must be nonempty")
    _, q_exp = unpack_scale32(query_scale32)
    _, k_exp = unpack_scale32(key_scale32)
    score_exponent = q_exp + k_exp
    tiles = [
        encode_tile(numerators[start : start + TILE_KEYS], score_exponent=score_exponent)
        for start in range(0, len(numerators), TILE_KEYS)
    ]
    row_maximum = max(tile.tile_max_s128 for tile in tiles)
    tile_offset_shift = 48 - score_exponent
    reconstructed: list[int] = []
    for tile in tiles:
        tile_offset_q17 = rne_shift(tile.tile_max_s128 - row_maximum, tile_offset_shift)
        for mantissa in tile.mantissas_s24:
            local_delta_q17 = mantissa << (SCORE_FRAC - tile.fraction_bits_u5)
            global_delta_q17 = tile_offset_q17 + local_delta_q17
            check_signed(global_delta_q17, 32, "reconstructed Q17 score")
            if global_delta_q17 > 0:
                raise OverflowError("tile-BFP reconstructed score became positive")
            reconstructed.append(global_delta_q17)
    return tuple(reconstructed)


def softmax_q15_from_q17(scores: Sequence[int]) -> tuple[int, ...]:
    if not scores:
        raise ValueError("softmax requires a nonempty row")
    weights = []
    for score in scores:
        value = check_signed(int(score), 32, "Q17 score")
        if value > 0:
            raise ValueError("centered score must be nonpositive")
        weights.append(0 if value <= EXP_UNDERFLOW_Q17 else exp_q31(value << 3))
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
class TileBfpResult:
    scores_q17: tuple[int, ...]
    probabilities_q15: tuple[int, ...]
    output_s8: tuple[int, ...]


def staged_attention_row(
    query_shadow: Sequence[int],
    key_shadows: Sequence[Sequence[int]],
    values: Sequence[Sequence[int]],
    query_position: int,
) -> TileBfpResult:
    if len(key_shadows) != query_position + 1 or len(values) != query_position + 1:
        raise ValueError("tile-BFP row requires all causal keys")
    numerators = tuple(
        score_numerator_s106(query_shadow, key, query_position, key_position)
        for key_position, key in enumerate(key_shadows)
    )
    scores = encode_score_row_from_numerators(numerators)
    probabilities = softmax_q15_from_q17(scores)
    return TileBfpResult(
        scores_q17=scores,
        probabilities_q15=probabilities,
        output_s8=attention_value_s8(probabilities, values),
    )
