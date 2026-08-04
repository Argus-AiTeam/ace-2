#!/usr/bin/env python3
"""Independent scalar model for layer0_fixed_q7_rope_score_v1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from ace2_quality_contracts import (
    fixed_q7_score_pair_parameters,
    round_divide_even_signed,
)


SRAM_TAG = 1 << 63
SRAM_LIMIT = 1 << 19
SRAM_PEAK_BYTES = 506_368


def signed(value: int, bits: int) -> int:
    mask = (1 << bits) - 1
    value &= mask
    return value - (1 << bits) if value & (1 << (bits - 1)) else value


def fixed_q7_rope_head(
    activations: Iterable[int],
    cos_q15: Iterable[int],
    sin_q15: Iterable[int],
) -> list[int]:
    values = list(activations)
    cosine = list(cos_q15)
    sine = list(sin_q15)
    if len(values) != 64 or len(cosine) != 64 or len(sine) != 64:
        raise ValueError("fixed-Q7 RoPE requires 64 lanes")
    if any(not -128 <= value <= 127 for value in values):
        raise ValueError("fixed-Q7 RoPE activation is outside signed int8")
    outputs = [0] * 64
    for pair in range(32):
        if cosine[pair] != cosine[pair + 32] or sine[pair] != sine[pair + 32]:
            raise ValueError("fixed-Q7 RoPE coefficient halves differ")
        c = cosine[pair]
        s = sine[pair]
        if not -32768 <= c <= 32767 or not -32768 <= s <= 32767:
            raise ValueError("fixed-Q7 coefficient is outside signed int16")
        if abs(c) + abs(s) > 46_342:
            raise ValueError("fixed-Q7 coefficient L1 bound exceeded")
        x0 = values[pair]
        x1 = values[pair + 32]
        r0 = x0 * c - x1 * s
        r1 = x0 * s + x1 * c
        if abs(r0) > 5_931_776 or abs(r1) > 5_931_776:
            raise OverflowError("fixed-Q7 signed-25-bit rotation bound exceeded")
        outputs[pair] = round_divide_even_signed(r0, 1 << 8)
        outputs[pair + 32] = round_divide_even_signed(r1, 1 << 8)
        if abs(outputs[pair]) > 23_171 or abs(outputs[pair + 32]) > 23_171:
            raise OverflowError("fixed-Q7 signed-int16 output bound exceeded")
    return outputs


def four_phase_signed16_product(query: int, key: int) -> tuple[int, tuple[int, ...]]:
    if not -32768 <= query <= 32767 or not -32768 <= key <= 32767:
        raise ValueError("four-phase operands must be signed int16")
    q_u16 = query & 0xFFFF
    k_u16 = key & 0xFFFF
    ql = signed(q_u16, 8)
    qh = signed(q_u16 >> 8, 8)
    kl = signed(k_u16, 8)
    kh = signed(k_u16 >> 8, 8)
    cq = 1 if q_u16 & 0x80 else 0
    ck = 1 if k_u16 & 0x80 else 0
    t0 = ql * kl + (kl << 8 if cq else 0) + (ql << 8 if ck else 0)
    if cq and ck:
        t0 += 1 << 16
    t1 = qh * kl + (qh << 8 if ck else 0)
    t2 = ql * kh + (kh << 8 if cq else 0)
    t3 = qh * kh
    product = t0 + (t1 << 8) + (t2 << 8) + (t3 << 16)
    return product, (t0, t1, t2, t3)


def fixed_q7_precenter_score(
    query: Iterable[int],
    key: Iterable[int],
    query_scale32: int,
    key_scale32: int,
) -> tuple[int, int]:
    q_values = list(query)
    k_values = list(key)
    if len(q_values) != 64 or len(k_values) != 64:
        raise ValueError("fixed-Q7 score requires 64 lanes")
    dot = 0
    for q_value, k_value in zip(q_values, k_values, strict=True):
        product, _ = four_phase_signed16_product(q_value, k_value)
        if product != q_value * k_value:
            raise AssertionError("four-phase product identity failed")
        dot += product
    if not -(1 << 37) <= dot < (1 << 37):
        raise OverflowError("fixed-Q7 signed-38-bit dot overflow")
    pair_sig, right_shift = fixed_q7_score_pair_parameters(
        query_scale32, key_scale32
    )
    scaled = dot * pair_sig
    if not -(1 << 54) <= scaled < (1 << 54):
        raise OverflowError("fixed-Q7 signed-55-bit product overflow")
    return round_divide_even_signed(scaled, 1 << right_shift), dot


def is_canonical_sram_range(address: int, size: int) -> bool:
    if address < 0 or size <= 0 or address & 0xF:
        return False
    if address & SRAM_TAG == 0 or address & ~(SRAM_TAG | (SRAM_LIMIT - 1)):
        return False
    offset = address & (SRAM_LIMIT - 1)
    return offset + size <= SRAM_LIMIT


@dataclass(frozen=True)
class Descriptor:
    opcode: int
    flags: int
    layer_id: int
    m: int
    n: int
    k: int
    src0_addr: int = 0
    src1_addr: int = 0
    dst_addr: int = 0
    scale_addr: int = 0
    scratch_addr: int = 0
    stride1: int = 0
    aux: int = 0


def legal_fixed_q7_tuple(descriptor: Descriptor) -> bool:
    flags_without_last = descriptor.flags & ~0x08
    if descriptor.flags & 0xF0 or flags_without_last & 0x04:
        return False
    if descriptor.layer_id != 0:
        return False
    if descriptor.opcode == 0x03:
        output_size = descriptor.n * 2
        return (
            descriptor.m == 1
            and descriptor.n in {896, 128}
            and descriptor.k == 64
            and descriptor.scratch_addr == 0
            and descriptor.src1_addr >> 32 == 0
            and descriptor.src0_addr == descriptor.dst_addr
            and is_canonical_sram_range(descriptor.src0_addr, output_size)
        )
    if descriptor.opcode == 0x04:
        return (
            descriptor.m == 1
            and 1 <= descriptor.n <= 256
            and descriptor.k == 64
            and descriptor.stride1 == 400
            and descriptor.scratch_addr == 0
            and descriptor.aux <= 13
            and is_canonical_sram_range(descriptor.src0_addr, 128)
            and is_canonical_sram_range(descriptor.src1_addr, 400 * descriptor.n)
        )
    if descriptor.opcode in {0x09, 0x0A}:
        return descriptor.stride1 == 400 and descriptor.scratch_addr == 0 and descriptor.aux == 0
    return False
