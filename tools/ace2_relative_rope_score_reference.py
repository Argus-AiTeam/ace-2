#!/usr/bin/env python3
"""Independent oracle for ``layer0_relative_rope_score_fusion_v1``.

The oracle intentionally uses Python integers for the signed-70 Scale32
product so no host int64 wrap can be mistaken for the hardware contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, Sequence

import torch

from ace2_quality_contracts import round_divide_even_signed, unpack_scale32


CONTRACT_ID = "layer0_relative_rope_score_fusion_v1"
HEAD_DIM = 64
PAIR_COUNT = 32
MAX_CONTEXT = 32768
MAX_COEFFICIENT_L1 = 46462
MAX_PHASE_ACC = 48_718_938_112
SRAM_PEAK_BYTES = 495_120
SRAM_MARGIN_BYTES = 29_168

PHASE_SINGLE = 0
PHASE_MAX_SCAN = 1
PHASE_CENTER_EMIT = 2

STATE_EMPTY = 0
STATE_SCAN = 1
STATE_SEALED = 2
STATE_EMIT = 3


def center_score(value: int, maximum: int) -> int:
    return max(-32768, min(0, value - maximum))


@lru_cache(maxsize=MAX_CONTEXT)
def relative_coefficients_q15(distance: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Generate one frozen BF16 Qwen relative-RoPE coefficient record."""
    if not 0 <= distance < MAX_CONTEXT:
        raise ValueError("relative-RoPE distance is outside 0..32767")
    pair_index = torch.arange(PAIR_COUNT, dtype=torch.float32)
    omega = torch.reciprocal(
        torch.pow(
            torch.tensor(1_000_000.0, dtype=torch.float32),
            (2.0 * pair_index) / float(HEAD_DIM),
        )
    )
    angle = torch.tensor(float(distance), dtype=torch.float32) * omega
    cosine = torch.round(
        torch.cos(angle).to(torch.bfloat16).to(torch.float64) * 32767.0
    ).clamp(-32768, 32767).to(torch.int64)
    sine = torch.round(
        torch.sin(angle).to(torch.bfloat16).to(torch.float64) * 32767.0
    ).clamp(-32768, 32767).to(torch.int64)
    cosine_values = tuple(int(value) for value in cosine.tolist())
    sine_values = tuple(int(value) for value in sine.tolist())
    if any(
        abs(cosine_value) + abs(sine_value) > MAX_COEFFICIENT_L1
        for cosine_value, sine_value in zip(
            cosine_values, sine_values, strict=True
        )
    ):
        raise ValueError("relative-RoPE coefficient L1 bound exceeded")
    return cosine_values, sine_values


def relative_rope_phase_acc(
    query: Iterable[int],
    key: Iterable[int],
    distance: int,
) -> int:
    query_values = tuple(int(value) for value in query)
    key_values = tuple(int(value) for value in key)
    if len(query_values) != HEAD_DIM or len(key_values) != HEAD_DIM:
        raise ValueError("relative-RoPE score requires 64-lane Q/K heads")
    if any(not -128 <= value <= 127 for value in (*query_values, *key_values)):
        raise ValueError("relative-RoPE score operands must be signed int8")
    cosine, sine = relative_coefficients_q15(distance)
    phase_acc = 0
    for pair in range(PAIR_COUNT):
        q0 = query_values[pair]
        q1 = query_values[pair + PAIR_COUNT]
        k0 = key_values[pair]
        k1 = key_values[pair + PAIR_COUNT]
        a_term = q0 * k0 + q1 * k1
        b_term = q1 * k0 - q0 * k1
        phase_acc += a_term * cosine[pair] - b_term * sine[pair]
    if abs(phase_acc) > MAX_PHASE_ACC:
        raise OverflowError("relative-RoPE signed-38 phase accumulator overflow")
    return phase_acc


def relative_rope_precenter_score(
    query: Iterable[int],
    key: Iterable[int],
    query_scale32: int,
    key_scale32: int,
    distance: int,
) -> tuple[int, int]:
    phase_acc = relative_rope_phase_acc(query, key, distance)
    query_sig, query_exp = unpack_scale32(query_scale32)
    key_sig, key_exp = unpack_scale32(key_scale32)
    significand_product = query_sig * key_sig
    shift = 39 - (query_exp + key_exp)
    if not 31 <= shift <= 87:
        raise ValueError("relative-RoPE score shift is outside 31..87")
    scaled = phase_acc * significand_product
    if not -(1 << 69) <= scaled < (1 << 69):
        raise OverflowError("relative-RoPE signed-70 scale product overflow")
    precenter = round_divide_even_signed(scaled, 1 << shift)
    if not -(1 << 63) <= precenter < (1 << 63):
        raise OverflowError("relative-RoPE signed-64 precenter overflow")
    return precenter, phase_acc


def relative_rope_centered_row(
    query: Iterable[int],
    keys: Iterable[Iterable[int]],
    query_scale32: int,
    key_scale32: int,
    query_position: int,
    key_base: int = 0,
) -> list[int]:
    key_values = [tuple(int(value) for value in key) for key in keys]
    if not key_values:
        raise ValueError("relative-RoPE row requires at least one valid key")
    if key_base < 0 or key_base + len(key_values) - 1 > query_position:
        raise ValueError("relative-RoPE key range is outside the causal row")
    precenter = [
        relative_rope_precenter_score(
            query,
            key,
            query_scale32,
            key_scale32,
            query_position - (key_base + index),
        )[0]
        for index, key in enumerate(key_values)
    ]
    maximum = max(precenter)
    return [center_score(value, maximum) for value in precenter]


@dataclass(frozen=True)
class GlobalRowState:
    maximum: int | None = None
    query_position: int = 0
    next_key_base: int = 0
    phase: int = STATE_EMPTY
    query_head: int = 0


def scan_precenter_tile(
    state: GlobalRowState,
    values: Sequence[int],
    *,
    query_head: int,
    query_position: int,
    key_base: int,
) -> GlobalRowState:
    if not values or not 0 <= query_head < 14:
        raise ValueError("MAX_SCAN requires a nonempty tile and legal query head")
    if key_base == 0:
        if state != GlobalRowState():
            raise ValueError("first MAX_SCAN requires EMPTY state")
        maximum = max(values)
    else:
        if (
            state.phase != STATE_SCAN
            or state.query_head != query_head
            or state.query_position != query_position
            or state.next_key_base != key_base
            or state.maximum is None
        ):
            raise ValueError("MAX_SCAN state is not contiguous")
        maximum = max(state.maximum, max(values))
    next_key_base = key_base + len(values)
    if next_key_base > query_position + 1:
        raise ValueError("MAX_SCAN exceeds the valid causal row")
    return GlobalRowState(
        maximum=maximum,
        query_position=query_position,
        next_key_base=next_key_base,
        phase=STATE_SEALED if next_key_base == query_position + 1 else STATE_SCAN,
        query_head=query_head,
    )


def emit_centered_tile(
    state: GlobalRowState,
    values: Sequence[int],
    *,
    query_head: int,
    query_position: int,
    key_base: int,
) -> tuple[GlobalRowState, list[int]]:
    if not values or state.maximum is None:
        raise ValueError("CENTER_EMIT requires a nonempty tile and sealed maximum")
    expected_phase = STATE_SEALED if key_base == 0 else STATE_EMIT
    if (
        state.phase != expected_phase
        or state.query_head != query_head
        or state.query_position != query_position
        or (key_base != 0 and state.next_key_base != key_base)
    ):
        raise ValueError("CENTER_EMIT state is not contiguous")
    next_key_base = key_base + len(values)
    if next_key_base > query_position + 1:
        raise ValueError("CENTER_EMIT exceeds the valid causal row")
    centered = [center_score(value, state.maximum) for value in values]
    if next_key_base == query_position + 1:
        return GlobalRowState(), centered
    return (
        GlobalRowState(
            maximum=state.maximum,
            query_position=query_position,
            next_key_base=next_key_base,
            phase=STATE_EMIT,
            query_head=query_head,
        ),
        centered,
    )
