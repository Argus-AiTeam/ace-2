from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from ace2_projection_reference import (  # noqa: E402
    HIDDEN_SIZE,
    ProjectionCase,
    pack_meta,
    reference_projection,
    round_shift_even,
)


class ProjectionBiasContractTest(unittest.TestCase):
    def setUp(self) -> None:
        weights = [7] * 154 + [0] * (HIDDEN_SIZE - 154)
        self.case = ProjectionCase(
            "c4_v_bias_discriminator",
            1,
            HIDDEN_SIZE,
            [[1] * HIDDEN_SIZE],
            [weights],
            [1_350_522_781],
            [34],
            [0],
            [593],
        )

    def test_exact_c4_bias_discriminator(self) -> None:
        dot = sum(
            activation * weight
            for activation, weight in zip(
                self.case.activations[0], self.case.weights[0], strict=True
            )
        )
        self.assertEqual(1078, dot)
        biased_accumulator = dot + self.case.bias_accumulators[0]
        self.assertEqual(1671, biased_accumulator)
        self.assertEqual(
            131,
            round_shift_even(
                biased_accumulator * self.case.multipliers[0],
                self.case.right_shifts[0],
            ),
        )
        result = reference_projection(self.case)
        self.assertEqual([[127]], result.outputs)
        self.assertTrue(result.saturation_seen)
        self.assertEqual(
            593,
            (pack_meta(1_350_522_781, 34, 0, 593) >> 48) & 0xFFFFFFFF,
        )

    def test_dot_only_negative_check(self) -> None:
        dot_only = ProjectionCase(
            self.case.name + "_dot_only",
            self.case.rows,
            self.case.reduction_size,
            self.case.activations,
            self.case.weights,
            self.case.multipliers,
            self.case.right_shifts,
            self.case.output_zero_points,
            [0],
        )
        result = reference_projection(dot_only)
        self.assertEqual(85, round_shift_even(1078 * 1_350_522_781, 34))
        self.assertEqual([[85]], result.outputs)
        self.assertFalse(result.saturation_seen)

    def test_dot_plus_bias_must_fit_signed32(self) -> None:
        overflow = ProjectionCase(
            "bias_overflow",
            1,
            4,
            [[1, 0, 0, 0]],
            [[1, 0, 0, 0]],
            [1],
            [0],
            [0],
            [(1 << 31) - 1],
        )
        with self.assertRaises(OverflowError):
            reference_projection(overflow)


if __name__ == "__main__":
    unittest.main()
