from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from ace2_cross_layer_error_carry_reference import (  # noqa: E402
    CARRY_MAX,
    CARRY_MIN,
    accepted_consumer_completion_tag,
    pack_scale32,
    producer_lane,
    reconstruct_q15,
    rmsnorm,
    workspace_proof,
)
from ace2_rmsnorm_reference import derive_scaled_gains_q8  # noqa: E402


class CrossLayerErrorCarryTest(unittest.TestCase):
    def test_accepted_consumer_completion_tag_is_transaction_owned(self) -> None:
        self.assertEqual(
            0x0ACE,
            accepted_consumer_completion_tag(True, True, 0x0ACE, (0xBEEF, 0x1234)),
        )
        with self.assertRaises(ValueError):
            accepted_consumer_completion_tag(True, False, 0xDEAD)

    def test_half_ties_preserve_error(self) -> None:
        unit = pack_scale32(0x8000, 0)
        half = pack_scale32(0x8000, -1)
        self.assertEqual((0, CARRY_MAX), (producer_lane(1, 0, half, unit, unit).hidden_s8,
                                         producer_lane(1, 0, half, unit, unit).carry_s16_q15))
        self.assertEqual((0, CARRY_MIN), (producer_lane(-1, 0, half, unit, unit).hidden_s8,
                                         producer_lane(-1, 0, half, unit, unit).carry_s16_q15))
        self.assertEqual((2, CARRY_MIN), (producer_lane(3, 0, half, unit, unit).hidden_s8,
                                         producer_lane(3, 0, half, unit, unit).carry_s16_q15))

    def test_hidden_saturation_is_an_error(self) -> None:
        unit = pack_scale32(0x8000, 0)
        with self.assertRaises(OverflowError):
            producer_lane(128, 0, unit, unit, unit)
        with self.assertRaises(OverflowError):
            producer_lane(-129, 0, unit, unit, unit)

    def test_reconstruction_bounds(self) -> None:
        self.assertEqual((127 << 15) + CARRY_MAX, reconstruct_q15(127, CARRY_MAX))
        self.assertEqual((-128 << 15) + CARRY_MIN, reconstruct_q15(-128, CARRY_MIN))

    def test_rmsnorm_exact_equations(self) -> None:
        scaled_gains = derive_scaled_gains_q8(
            [1.0, 1.25, -0.75, 0.5],
            1.0 / 32.0,
        )
        result = rmsnorm(
            [1, -2, 3, -4],
            [16384, -16384, 0, 8192],
            scaled_gains,
        )
        self.assertEqual(4, len(result.outputs_s8))
        self.assertGreater(result.root_q15, 0)
        self.assertEqual((1 << 45) // result.root_q15, result.inverse_q30)
        self.assertFalse(result.saturation_seen)
        self.assertEqual((17, -36, -26, -21), result.outputs_s8)

    def test_output_scale_folded_gains_avoid_zero_collapse(self) -> None:
        hidden = [((index * 37) % 31) - 15 for index in range(896)]
        carry = [((index * 8191) % 32769) - 16384 for index in range(896)]
        weights = [
            (0.5, 0.75, 1.0, 1.25)[index % 4]
            * (-1.0 if index % 11 == 0 else 1.0)
            for index in range(896)
        ]
        scaled_gains = derive_scaled_gains_q8(weights, 1.0 / 32.0)
        result = rmsnorm(hidden, carry, scaled_gains)
        nonzero_outputs = sum(value != 0 for value in result.outputs_s8)
        self.assertGreaterEqual(nonzero_outputs, 850)
        self.assertFalse(result.saturation_seen)

    def test_workspace_proof(self) -> None:
        proof = workspace_proof()
        self.assertLessEqual(proof["producer_numerator_magnitude_bits"], 95)
        self.assertLessEqual(proof["producer_denominator_bits"], 64)
        self.assertLessEqual(proof["reconstruction_magnitude_bits"], 23)
        self.assertLessEqual(proof["square_sum_bits"], 56)


if __name__ == "__main__":
    unittest.main()
