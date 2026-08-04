from __future__ import annotations

import unittest

from tools.ace2_qk_residual_cross_term_reference import (
    kv_head_for_query_head,
    projection_residual,
    residual_cross_term_score,
    residual_rope_pair,
    round_div_even_signed,
    staged_softmax_attention_value,
    unpack_scale32,
    validate_layer_head,
)


SCALE_ONE = 0x00008000  # 0x8000 * 2^-15 == 1
SCALE_HALF = 0x00FF8000  # 0x8000 * 2^-16 == 0.5


class QkResidualCrossTermTest(unittest.TestCase):
    def test_scale32_and_signed_ties_even(self) -> None:
        self.assertEqual(unpack_scale32(SCALE_ONE), (0x8000, 0))
        self.assertEqual(round_div_even_signed(5, 2), 2)
        self.assertEqual(round_div_even_signed(7, 2), 4)
        self.assertEqual(round_div_even_signed(-5, 2), -2)
        with self.assertRaises(ValueError):
            unpack_scale32(0x01008000)

    def test_projection_residual_exact_and_clamps(self) -> None:
        exact = projection_residual(3, 5, 1, SCALE_ONE, SCALE_HALF)
        self.assertEqual(exact.baseline_q8, 8)
        self.assertEqual(exact.residual_s4, -1)
        self.assertFalse(exact.positive_clamp or exact.negative_clamp)

        positive = projection_residual(1000, 4096, 12, SCALE_ONE, 0x00F88000)
        negative = projection_residual(-1000, 4096, 12, SCALE_ONE, 0x00F88000)
        self.assertEqual(positive.residual_s4, 7)
        self.assertEqual(negative.residual_s4, -7)
        self.assertTrue(positive.positive_clamp)
        self.assertTrue(negative.negative_clamp)

    def test_residual_rope_reserved_code_ties_and_identity(self) -> None:
        self.assertEqual(residual_rope_pair(7, -7, 32767, 0).real_s8, 7)
        self.assertEqual(residual_rope_pair(7, -7, 32767, 0).imag_s8, -7)
        tied = residual_rope_pair(1, 0, 16384, 0)
        self.assertEqual(tied.real_s8, 0)
        with self.assertRaises(ValueError):
            residual_rope_pair(-8, 0, 32767, 0)

    def test_authoritative_base_score_plus_three_correction_terms(self) -> None:
        q = [((lane * 7) % 31) - 15 for lane in range(64)]
        k = [((lane * 5) % 29) - 14 for lane in range(64)]
        rq = [((lane * 3) % 15) - 7 for lane in range(64)]
        rk = [((lane * 11) % 15) - 7 for lane in range(64)]
        authoritative_base_score = -6 << 40
        result = residual_cross_term_score(
            authoritative_base_score,
            q,
            k,
            rq,
            rk,
            SCALE_ONE,
            SCALE_HALF,
            SCALE_HALF,
            SCALE_ONE,
        )
        self.assertEqual(result.authoritative_base_score_q20_44_s64, authoritative_base_score)
        self.assertEqual(len(result.correction_dots_s32), 3)
        self.assertEqual(len(result.correction_terms_q20_44_s67), 3)
        self.assertEqual(
            result.score_q20_44_s64,
            authoritative_base_score + sum(result.correction_terms_q20_44_s67),
        )

    def test_all_layers_and_14_to_2_mapping(self) -> None:
        for layer_id in range(24):
            for query_head in range(14):
                kv_head = kv_head_for_query_head(query_head)
                validate_layer_head(layer_id, query_head, kv_head)
        self.assertEqual([kv_head_for_query_head(head) for head in range(14)], [0] * 7 + [1] * 7)
        with self.assertRaises(ValueError):
            validate_layer_head(24, 0, 0)
        with self.assertRaises(ValueError):
            validate_layer_head(0, 13, 0)

    def test_staged_softmax_and_value_row_lengths(self) -> None:
        for length in (1, 2, 63, 64, 65):
            scores = [((index * 17) % 41 - 20) << 40 for index in range(length)]
            values = [
                [((token * 11 + lane * 3) % 255) - 127 for lane in range(64)]
                for token in range(length)
            ]
            result = staged_softmax_attention_value(scores, values)
            self.assertEqual(len(result.weights_q1_31), length)
            self.assertEqual(len(result.probabilities_q0_15), length)
            self.assertEqual(len(result.output_s8), 64)
            self.assertTrue(all(0 <= value <= 0xFFFF for value in result.probabilities_q0_15))
            self.assertTrue(all(-128 <= value <= 127 for value in result.output_s8))


if __name__ == "__main__":
    unittest.main()
