from __future__ import annotations

import json
import unittest
from pathlib import Path

from tools.ace2_quality_contracts import scale32_ratio
from tools.ace2_v_residual_value_correction_reference import (
    canonical_s4_byte,
    decode_canonical_s4_byte,
    kv_head_for_query_head,
    round_div_even_signed,
    unpack_scale32,
    validate_layer_head,
    v_projection_residual,
    v_residual_value_correction,
)


ROOT = Path(__file__).resolve().parents[1]
SCALE_ONE = 0x00008000
SCALE_HALF = 0x00FF8000


class VResidualValueCorrectionTest(unittest.TestCase):
    def test_signed_rounding_and_canonical_encoding(self) -> None:
        self.assertEqual(round_div_even_signed(5, 2), 2)
        self.assertEqual(round_div_even_signed(7, 2), 4)
        self.assertEqual(round_div_even_signed(-5, 2), -2)
        self.assertEqual(canonical_s4_byte(-7), 0xF9)
        self.assertEqual(decode_canonical_s4_byte(0xF9), -7)
        with self.assertRaises(ValueError):
            decode_canonical_s4_byte(0xF8)
        with self.assertRaises(ValueError):
            decode_canonical_s4_byte(0x09)

    def test_projection_residual_exact_and_clamps(self) -> None:
        exact = v_projection_residual(3, 5, 1, SCALE_ONE, SCALE_HALF)
        self.assertEqual(exact.baseline_v8, 8)
        self.assertEqual(exact.residual_v_s4, -1)
        self.assertEqual(exact.residual_v_canonical_u8, 0xFF)
        positive = v_projection_residual(1000, 4096, 12, SCALE_ONE, 0x00F88000)
        negative = v_projection_residual(-1000, 4096, 12, SCALE_ONE, 0x00F88000)
        self.assertEqual((positive.residual_v_s4, negative.residual_v_s4), (7, -7))
        self.assertTrue(positive.positive_clamp)
        self.assertTrue(negative.negative_clamp)

    def test_value_correction_sum_then_convert(self) -> None:
        probabilities = [32768, 16384, 8192, 4096]
        residuals = [canonical_s4_byte(value) for value in (1, -2, 3, -4)]
        result = v_residual_value_correction(
            probabilities,
            residuals,
            SCALE_ONE,
            SCALE_HALF,
            1234,
        )
        expected_raw = sum(p * r for p, r in zip(probabilities, (1, -2, 3, -4), strict=True))
        self.assertEqual(result.correction_raw_s64, expected_raw)
        self.assertEqual(result.correction_baseline_domain_s32, round_div_even_signed(expected_raw, 2))
        self.assertEqual(result.corrected_accumulator_s32, 1234 + round_div_even_signed(expected_raw, 2))

    def test_row_lengths_and_mapping(self) -> None:
        for length in (1, 2, 63, 64, 65):
            probabilities = [((index * 7919) + 4096) & 0xFFFF for index in range(length)]
            residuals = [canonical_s4_byte(((index * 5) % 15) - 7) for index in range(length)]
            result = v_residual_value_correction(
                probabilities,
                residuals,
                SCALE_ONE,
                SCALE_HALF,
                0,
            )
            self.assertIsInstance(result.corrected_accumulator_s32, int)
        for layer in range(24):
            for query_head in range(14):
                validate_layer_head(layer, query_head, kv_head_for_query_head(query_head))
        self.assertEqual([kv_head_for_query_head(head) for head in range(14)], [0] * 7 + [1] * 7)

    def test_metadata_has_exact_48_row_rule(self) -> None:
        metadata = json.loads(
            (ROOT / "reference/generated/v_residual_scale32_metadata.json").read_text(encoding="utf-8")
        )
        self.assertEqual(metadata["contract_id"], "shared_v_residual_value_correction_attention_v1")
        self.assertEqual(len(metadata["records"]), 48)
        for layer in range(24):
            rows = metadata["records"][2 * layer : 2 * layer + 2]
            self.assertEqual([row["kv_head"] for row in rows], [0, 1])
            self.assertEqual(
                rows[0]["baseline_v_scale32"]["packed_u32"],
                rows[1]["baseline_v_scale32"]["packed_u32"],
            )
            baseline = rows[0]["baseline_v_scale32"]["packed_u32"]
            residual = rows[0]["residual_v_scale32"]["packed_u32"]
            unpack_scale32(baseline)
            unpack_scale32(residual)
            baseline_num, baseline_den = scale32_ratio(baseline)
            residual_num, residual_den = scale32_ratio(residual)
            self.assertGreaterEqual(residual_num * 14 * baseline_den, baseline_num * residual_den)

    def test_errors_are_visible(self) -> None:
        with self.assertRaises(ValueError):
            v_residual_value_correction([1], [0xF8], SCALE_ONE, SCALE_HALF, 0)
        with self.assertRaises(ValueError):
            v_residual_value_correction([], [], SCALE_ONE, SCALE_HALF, 0)
        with self.assertRaises(OverflowError):
            v_residual_value_correction([0xFFFF], [0x07], SCALE_ONE, 0x0000FFFF, (1 << 31) - 1)


if __name__ == "__main__":
    unittest.main()
