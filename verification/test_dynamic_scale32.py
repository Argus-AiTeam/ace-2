from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from ace2_dynamic_scale32_reference import (  # noqa: E402
    TaggedEvent,
    build_sidecar,
    dynamic_group,
    tagged_accumulate,
    validate_sidecar,
)
from ace2_quality_contracts import pack_scale32  # noqa: E402


class DynamicScale32Test(unittest.TestCase):
    def test_smallest_legal_delta_and_ties_even(self) -> None:
        scale = pack_scale32(0x8000, 0)
        small = dynamic_group([(-2, -1, 0, 1, 2)[index % 5] for index in range(64)], scale)
        self.assertEqual(-5, small.delta)
        self.assertEqual((-64, -32, 0, 32, 64), small.mantissas[:5])

        tie = dynamic_group([255, -255] + [0] * 62, scale)
        self.assertEqual(2, tie.delta)
        self.assertEqual((64, -64), tie.mantissas[:2])

    def test_effective_exponent_floor_and_zero_rule(self) -> None:
        floor = pack_scale32(0x9234, -24)
        result = dynamic_group([1 if index & 1 else -1 for index in range(64)], floor)
        self.assertEqual(0, result.delta)
        self.assertEqual(floor, result.effective_scale32)
        zero = dynamic_group([0] * 128, pack_scale32(0xC000, -3))
        self.assertEqual(0, zero.delta)

    def test_no_legal_delta_fails_closed(self) -> None:
        with self.assertRaises(OverflowError):
            dynamic_group([(1 << 39) - 1] + [0] * 63, pack_scale32(0x8000, 4))

    def test_sidecar_round_trip_and_reserved_bytes(self) -> None:
        scale = pack_scale32(0x8000, 0)
        sidecar = build_sidecar(
            payload_addr=0x1000,
            group_lanes=64,
            producer_tag=0x1234,
            layer_id=7,
            producer_opcode=1,
            tensor_elements=128,
            model_identity=0x0123456789ABCDEF,
            deltas=(-5, 2),
            base_scales=(scale, scale),
        )
        self.assertEqual(
            (-5, 2),
            validate_sidecar(
                sidecar,
                payload_addr=0x1000,
                group_lanes=64,
                producer_tag=0x1234,
                layer_id=7,
                producer_opcode=1,
                tensor_elements=128,
                model_identity=0x0123456789ABCDEF,
                base_scales=(scale, scale),
            ),
        )
        corrupt = bytearray(sidecar)
        corrupt[61] = 1
        with self.assertRaises(ValueError):
            validate_sidecar(
                bytes(corrupt),
                payload_addr=0x1000,
                group_lanes=64,
                producer_tag=0x1234,
                layer_id=7,
                producer_opcode=1,
                tensor_elements=128,
                model_identity=0x0123456789ABCDEF,
                base_scales=(scale, scale),
            )

    def test_tagged_accumulator_has_no_intermediate_rounding(self) -> None:
        one = pack_scale32(0x8000, 0)
        half = pack_scale32(0x8000, -1)
        self.assertEqual(
            2 << 78,
            tagged_accumulate((TaggedEvent(3, one, one), TaggedEvent(-1, one, one))),
        )
        self.assertEqual(3 << 77, tagged_accumulate((TaggedEvent(3, half, one),)))


if __name__ == "__main__":
    unittest.main()
