"""Independent integer reference for shared_token_group_dynamic_scale32_v1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from ace2_quality_contracts import pack_scale32, unpack_scale32


DELTA_MIN = -24
DELTA_MAX = 24
MANTISSA_MIN = -127
MANTISSA_MAX = 127
GROUP_LANES = (64, 128)
MAX_GROUPS = 38
VALUE_WIDTH = 40
ACCUMULATOR_WIDTH = 160
CANONICAL_PRODUCT_EXPONENT = -78


def round_shift_even_signed(value: int, shift: int) -> int:
    if shift < 0:
        raise ValueError("shift must be nonnegative")
    if shift == 0:
        return value
    magnitude = abs(value)
    quotient, remainder = divmod(magnitude, 1 << shift)
    half = 1 << (shift - 1)
    rounded = quotient + int(
        remainder > half or (remainder == half and (quotient & 1))
    )
    return -rounded if value < 0 else rounded


def quantize_for_delta(value: int, delta: int) -> int:
    if not DELTA_MIN <= delta <= DELTA_MAX:
        raise ValueError("dynamic Scale32 delta outside [-24,24]")
    if delta < 0:
        return value << -delta
    return round_shift_even_signed(value, delta)


@dataclass(frozen=True)
class DynamicGroupResult:
    delta: int
    effective_scale32: int
    mantissas: tuple[int, ...]


def dynamic_group(values: Sequence[int], base_scale32: int) -> DynamicGroupResult:
    if len(values) not in GROUP_LANES:
        raise ValueError("dynamic group must contain exactly 64 or 128 lanes")
    lower = -(1 << (VALUE_WIDTH - 1))
    upper = (1 << (VALUE_WIDTH - 1)) - 1
    if any(not lower <= value <= upper for value in values):
        raise OverflowError("wide group value exceeds signed-40 storage")

    significand, base_exponent = unpack_scale32(base_scale32)
    if all(value == 0 for value in values):
        return DynamicGroupResult(
            delta=0,
            effective_scale32=base_scale32,
            mantissas=tuple(0 for _ in values),
        )

    for delta in range(DELTA_MIN, DELTA_MAX + 1):
        effective_exponent = base_exponent + delta
        if not -24 <= effective_exponent <= 4:
            continue
        mantissas = tuple(quantize_for_delta(value, delta) for value in values)
        if all(MANTISSA_MIN <= value <= MANTISSA_MAX for value in mantissas):
            if any(value == -128 for value in mantissas):
                raise AssertionError("reserved -128 mantissa escaped symmetric clamp")
            return DynamicGroupResult(
                delta=delta,
                effective_scale32=pack_scale32(significand, effective_exponent),
                mantissas=mantissas,
            )
    raise OverflowError("no legal dynamic Scale32 exponent delta")


def _expected_group_count(elements: int, lanes: int) -> int:
    if lanes not in GROUP_LANES or elements <= 0:
        raise ValueError("invalid tensor shape")
    return (elements + lanes - 1) // lanes


def build_sidecar(
    *,
    payload_addr: int,
    group_lanes: int,
    producer_tag: int,
    layer_id: int,
    producer_opcode: int,
    tensor_elements: int,
    model_identity: int,
    deltas: Sequence[int],
    base_scales: Sequence[int],
) -> bytes:
    group_count = _expected_group_count(tensor_elements, group_lanes)
    if not 1 <= group_count <= MAX_GROUPS:
        raise ValueError("sidecar group count outside [1,38]")
    if len(deltas) != group_count or len(base_scales) != group_count:
        raise ValueError("sidecar delta/base-scale count mismatch")
    if payload_addr < 64 or payload_addr % 64:
        raise ValueError("candidate payload address must be nonzero 64-byte aligned")
    if not 0 <= producer_tag <= 0xFFFF:
        raise ValueError("producer tag is not uint16")
    if not 0 <= layer_id <= 0xFF or not 0 <= producer_opcode <= 0xFF:
        raise ValueError("layer/opcode is not uint8")
    if not 0 <= model_identity <= 0xFFFFFFFFFFFFFFFF:
        raise ValueError("model identity is not uint64")

    sidecar = bytearray(64)
    sidecar[0:4] = b"BFP1"
    sidecar[4] = 1
    sidecar[5] = group_lanes
    sidecar[6] = group_count
    sidecar[8:10] = producer_tag.to_bytes(2, "little")
    sidecar[10] = layer_id
    sidecar[11] = producer_opcode
    sidecar[12:16] = tensor_elements.to_bytes(4, "little")
    sidecar[16:24] = model_identity.to_bytes(8, "little")
    for index, (delta, base_scale) in enumerate(zip(deltas, base_scales, strict=True)):
        _, base_exponent = unpack_scale32(base_scale)
        if not DELTA_MIN <= delta <= DELTA_MAX:
            raise OverflowError("sidecar delta outside [-24,24]")
        if not -24 <= base_exponent + delta <= 4:
            raise OverflowError("sidecar effective exponent outside [-24,4]")
        sidecar[24 + index] = delta & 0xFF
    return bytes(sidecar)


def validate_sidecar(
    sidecar: bytes,
    *,
    payload_addr: int,
    group_lanes: int,
    producer_tag: int,
    layer_id: int,
    producer_opcode: int,
    tensor_elements: int,
    model_identity: int,
    base_scales: Sequence[int],
) -> tuple[int, ...]:
    if len(sidecar) != 64:
        raise ValueError("sidecar must be exactly 64 bytes")
    group_count = _expected_group_count(tensor_elements, group_lanes)
    if payload_addr < 64 or payload_addr % 64:
        raise ValueError("invalid payload address")
    if sidecar[0:4] != b"BFP1" or sidecar[4] != 1:
        raise ValueError("invalid sidecar magic/schema")
    if sidecar[5] != group_lanes or sidecar[6] != group_count or sidecar[7] != 0:
        raise ValueError("invalid sidecar shape/flags")
    if int.from_bytes(sidecar[8:10], "little") != producer_tag:
        raise ValueError("producer tag mismatch")
    if sidecar[10] != layer_id or sidecar[11] != producer_opcode:
        raise ValueError("producer identity mismatch")
    if int.from_bytes(sidecar[12:16], "little") != tensor_elements:
        raise ValueError("element count mismatch")
    if int.from_bytes(sidecar[16:24], "little") != model_identity:
        raise ValueError("model identity mismatch")
    if sidecar[62:64] != b"\0\0" or any(sidecar[24 + group_count : 62]):
        raise ValueError("nonzero sidecar reserved/zero-fill bytes")
    if len(base_scales) != group_count:
        raise ValueError("base-scale count mismatch")

    deltas: list[int] = []
    for raw_delta, base_scale in zip(sidecar[24 : 24 + group_count], base_scales, strict=True):
        delta = raw_delta - 256 if raw_delta & 0x80 else raw_delta
        _, base_exponent = unpack_scale32(base_scale)
        if not DELTA_MIN <= delta <= DELTA_MAX:
            raise OverflowError("sidecar delta outside [-24,24]")
        if not -24 <= base_exponent + delta <= 4:
            raise OverflowError("sidecar effective exponent outside [-24,4]")
        deltas.append(delta)
    return tuple(deltas)


@dataclass(frozen=True)
class TaggedEvent:
    partial_s32: int
    scale_a32: int
    scale_b32: int


def tagged_accumulate(events: Iterable[TaggedEvent]) -> int:
    accumulator = 0
    count = 0
    for event in events:
        if not -(1 << 31) <= event.partial_s32 < (1 << 31):
            raise OverflowError("tagged partial is not signed-32")
        sig_a, exp_a = unpack_scale32(event.scale_a32)
        sig_b, exp_b = unpack_scale32(event.scale_b32)
        shift = exp_a + exp_b + 48
        if not 0 <= shift <= 56:
            raise ValueError("tagged Scale32 alignment shift outside [0,56]")
        accumulator += event.partial_s32 * sig_a * sig_b * (1 << shift)
        count += 1
        if count > MAX_GROUPS:
            raise ValueError("tagged accumulator exceeds 38 events")
        if not -(1 << (ACCUMULATOR_WIDTH - 1)) <= accumulator < (1 << (ACCUMULATOR_WIDTH - 1)):
            raise OverflowError("tagged accumulator exceeds signed-160")
    if count == 0:
        raise ValueError("tagged accumulator requires at least one event")
    return accumulator

