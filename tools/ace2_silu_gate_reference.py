#!/usr/bin/env python3
"""Independent fixed-point reference for the ACE-2 SiLU-gate operator."""

from __future__ import annotations

from dataclasses import dataclass
import math


SILU_INPUT_FRAC = 9
SILU_OUTPUT_FRAC = 12
SILU_LUT_STEP_SHIFT = 6
SILU_LUT_MIN_INDEX = -64
SILU_LUT_MAX_INDEX = 64
SILU_LANES = 8
MLP_INTERMEDIATE_SIZE = 4864


@dataclass(frozen=True)
class SiluGateCase:
    name: str
    gate_q6_9: list[int]
    up_q6_9: list[int]
    multiplier: int
    right_shift: int
    output_zero_point: int


@dataclass(frozen=True)
class SiluGateResult:
    outputs: list[int]
    saturation_seen: bool
    tie_even_keep_count: int
    tie_even_increment_count: int
    positive_saturation_count: int
    negative_saturation_count: int
    lut_clip_low_count: int
    lut_clip_high_count: int


def to_sint(value: int, width: int) -> int:
    value &= (1 << width) - 1
    return value - (1 << width) if value & (1 << (width - 1)) else value


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


def saturate_int8(value: int) -> tuple[int, bool]:
    if value > 127:
        return 127, True
    if value < -128:
        return -128, True
    return value, False


def silu_lut() -> dict[int, int]:
    table: dict[int, int] = {}
    for index in range(SILU_LUT_MIN_INDEX, SILU_LUT_MAX_INDEX + 1):
        x = index / 8.0
        quantized = round((x / (1.0 + math.exp(-x))) * (1 << SILU_OUTPUT_FRAC))
        table[index] = max(-32768, min(32767, quantized))
    return table


SILU_LUT = silu_lut()


def silu_q3_12(gate_q6_9: int) -> int:
    gate = to_sint(gate_q6_9, 16)
    index = gate >> SILU_LUT_STEP_SHIFT
    if index <= SILU_LUT_MIN_INDEX:
        return SILU_LUT[SILU_LUT_MIN_INDEX]
    if index >= SILU_LUT_MAX_INDEX:
        return SILU_LUT[SILU_LUT_MAX_INDEX]
    return SILU_LUT[index]


def reference_silu_gate(case: SiluGateCase) -> SiluGateResult:
    if len(case.gate_q6_9) != len(case.up_q6_9):
        raise ValueError(f"{case.name}: gate/up lengths differ")
    if not 1 <= len(case.gate_q6_9) <= MLP_INTERMEDIATE_SIZE:
        raise ValueError(f"{case.name}: length is outside 1..{MLP_INTERMEDIATE_SIZE}")
    outputs: list[int] = []
    saturation_seen = False
    tie_even_keep_count = 0
    tie_even_increment_count = 0
    positive_saturation_count = 0
    negative_saturation_count = 0
    lut_clip_low_count = 0
    lut_clip_high_count = 0
    multiplier = to_sint(case.multiplier, 32)
    zero_point = to_sint(case.output_zero_point, 8)
    for gate, up in zip(case.gate_q6_9, case.up_q6_9, strict=True):
        gate_value = to_sint(gate, 16)
        table_index = gate_value >> SILU_LUT_STEP_SHIFT
        lut_clip_low_count += int(table_index < SILU_LUT_MIN_INDEX)
        lut_clip_high_count += int(table_index > SILU_LUT_MAX_INDEX)
        product_q9_21 = silu_q3_12(gate) * to_sint(up, 16)
        requant_product = product_q9_21 * multiplier
        if case.right_shift > 0:
            magnitude = abs(requant_product)
            base = magnitude >> case.right_shift
            remainder = magnitude & ((1 << case.right_shift) - 1)
            if remainder == 1 << (case.right_shift - 1):
                if base & 1:
                    tie_even_increment_count += 1
                else:
                    tie_even_keep_count += 1
        scaled = round_shift_even(requant_product, case.right_shift)
        shifted_output = scaled + zero_point
        positive_saturation_count += int(shifted_output > 127)
        negative_saturation_count += int(shifted_output < -128)
        output, saturated = saturate_int8(shifted_output)
        outputs.append(output)
        saturation_seen |= saturated
    return SiluGateResult(
        outputs=outputs,
        saturation_seen=saturation_seen,
        tie_even_keep_count=tie_even_keep_count,
        tie_even_increment_count=tie_even_increment_count,
        positive_saturation_count=positive_saturation_count,
        negative_saturation_count=negative_saturation_count,
        lut_clip_low_count=lut_clip_low_count,
        lut_clip_high_count=lut_clip_high_count,
    )


def reference_silu_gate_packed_int8(case: SiluGateCase) -> SiluGateResult:
    """Model the shell adapter before applying the unchanged int16 core math."""
    for field_name, values in (
        ("gate", case.gate_q6_9),
        ("up", case.up_q6_9),
    ):
        for index, value in enumerate(values):
            if not -128 <= value <= 127:
                raise ValueError(
                    f"{case.name}: {field_name}[{index}]={value} is not signed int8"
                )
    widened = SiluGateCase(
        name=case.name,
        gate_q6_9=[to_sint(value, 8) for value in case.gate_q6_9],
        up_q6_9=[to_sint(value, 8) for value in case.up_q6_9],
        multiplier=case.multiplier,
        right_shift=case.right_shift,
        output_zero_point=case.output_zero_point,
    )
    return reference_silu_gate(widened)
