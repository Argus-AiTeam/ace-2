from __future__ import annotations

import random
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from ace2_down_projection_residual_fusion_hook import fuse_scale32_lane_exact
from ace2_down_projection_residual_fusion_reference import (
    bounded_quotient_schedule,
    fuse_lane,
    fuse_lane_schedule_model,
    pack_scale32,
    unpack_scale32,
    workspace_proof,
)


class DownProjectionResidualFusionTest(unittest.TestCase):
    def test_ties_and_asymmetric_saturation_boundaries(self) -> None:
        unit = pack_scale32(0x8000, 0)
        half = pack_scale32(0x8000, -1)
        self.assertEqual(fuse_lane(1, 0, half, unit, unit).output_s8, 0)
        self.assertEqual(fuse_lane(3, 0, half, unit, unit).output_s8, 2)
        self.assertEqual(fuse_lane(-3, 0, half, unit, unit).output_s8, -2)
        positive_tie = fuse_lane(255, 0, half, unit, unit)
        negative_tie = fuse_lane(-257, 0, half, unit, unit)
        self.assertEqual((positive_tie.output_s8, positive_tie.positive_saturation), (127, True))
        self.assertEqual((negative_tie.output_s8, negative_tie.negative_saturation), (-128, False))

    def test_scale32_validation(self) -> None:
        for exponent in range(-24, 5):
            self.assertEqual(unpack_scale32(pack_scale32(0x8000, exponent)), (0x8000, exponent))
        for invalid in (0x01008000, 0x00007FFF, 0x00E78000, 0x00058000):
            with self.assertRaises(ValueError):
                unpack_scale32(invalid)

    def test_schedule_matches_exact_oracle_randomized(self) -> None:
        rng = random.Random(0xACE2D0F)
        for _ in range(2000):
            accumulator = rng.randint(-(1 << 31), (1 << 31) - 1)
            residual = rng.randint(-128, 127)
            records = [
                pack_scale32(rng.randint(0x8000, 0xFFFF), rng.randint(-24, 4))
                for _ in range(3)
            ]
            expected = fuse_lane_schedule_model(accumulator, residual, *records)
            hook_output, hook_numerator, hook_denominator, hook_exp = fuse_scale32_lane_exact(
                accumulator, residual, *records
            )
            self.assertEqual(
                (expected.output_s8, expected.numerator_s96, expected.denominator_u64, expected.common_exponent),
                (hook_output, hook_numerator, hook_denominator, hook_exp),
            )

    def test_bounded_divider_never_needs_more_than_eight_bits(self) -> None:
        rng = random.Random(0x896)
        for _ in range(1000):
            denominator = rng.randint(1, (1 << 44) - 1)
            numerator = rng.randint(-400, 400) * denominator + rng.randint(0, denominator - 1)
            magnitude, _remainder, positive_clamp, negative_clamp = bounded_quotient_schedule(
                numerator, denominator
            )
            if not (positive_clamp or negative_clamp):
                self.assertLessEqual(magnitude, 128)

    def test_workspace_static_proof(self) -> None:
        proof = workspace_proof()
        self.assertLessEqual(proof["numerator_required_magnitude_bits"], 76)
        self.assertLessEqual(proof["denominator_required_bits"], 44)

    def test_no_source_term_rounding(self) -> None:
        accumulator_scale = pack_scale32(0x8001, -7)
        residual_scale = pack_scale32(0x8003, -6)
        destination_scale = pack_scale32(0xF123, -5)
        result = fuse_lane(1234567, -93, accumulator_scale, residual_scale, destination_scale)
        expected_numerator = 1234567 * 0x8001 + (-93) * 0x8003 * 2
        expected_denominator = 0xF123 * 4
        self.assertEqual(result.common_exponent, -7)
        self.assertEqual(result.numerator_s96, expected_numerator)
        self.assertEqual(result.denominator_u64, expected_denominator)


if __name__ == "__main__":
    unittest.main()
