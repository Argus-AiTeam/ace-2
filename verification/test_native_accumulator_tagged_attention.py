from __future__ import annotations

import unittest
from fractions import Fraction

from tools.ace2_native_accumulator_tagged_reference import (
    TaggedLane,
    normalize_tagged,
    round_shift_even,
    tagged_rope_pair,
    tagged_score_q20_44,
    unpack_scale32,
)


class NativeAccumulatorTaggedAttentionTest(unittest.TestCase):
    def test_scale32_and_ties_even(self) -> None:
        self.assertEqual(unpack_scale32(0x00FFA245), (0xA245, -1))
        self.assertEqual(round_shift_even(5, 1), 2)
        self.assertEqual(round_shift_even(7, 1), 4)
        self.assertEqual(round_shift_even(-5, 1), -2)
        with self.assertRaises(ValueError):
            unpack_scale32(0x01808000)

    def test_rope_zero_and_nonzero(self) -> None:
        zero = tagged_rope_pair(0, 0x00FFA245, 0, 0x00008307, 32767, 0)
        self.assertEqual(zero.real, TaggedLane(0, 0))
        self.assertEqual(zero.imag, TaggedLane(0, 0))
        value = tagged_rope_pair(
            123456, 0x00FFA245, -234567, 0x00008307, 32767, 0
        )
        self.assertNotEqual(value.real.mantissa, 0)
        self.assertNotEqual(value.imag.mantissa, 0)

    def test_normalization_boundaries(self) -> None:
        self.assertEqual(normalize_tagged((1 << 30) - 1, -30), TaggedLane((1 << 30) - 1, -30))
        normalized = normalize_tagged(1 << 40, -30)
        self.assertTrue(-(1 << 31) <= normalized.mantissa < (1 << 31))
        realized = (
            Fraction(normalized.mantissa << normalized.exponent, 1)
            if normalized.exponent >= 0
            else Fraction(normalized.mantissa, 1 << -normalized.exponent)
        )
        self.assertEqual(realized, 1 << 10)

    def test_tagged_score_mixed_exponents_and_64_lanes(self) -> None:
        query = [TaggedLane(1 << 29, -20), TaggedLane(-(1 << 28), -19)]
        key = [TaggedLane(-(1 << 29), -20), TaggedLane(1 << 28, -19)]
        score = tagged_score_q20_44(query, key)
        self.assertIsInstance(score, int)
        self.assertEqual(
            tagged_score_q20_44([TaggedLane(1, -30)] * 64, [TaggedLane(1, -30)] * 64),
            0,
        )


if __name__ == "__main__":
    unittest.main()
