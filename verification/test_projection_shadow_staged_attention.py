from __future__ import annotations

import random
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from tools.ace2_full_model_fixed_point import (
    projection_shadow_staged_attention_scores_raw,
    fixed_attention_value_raw,
    fixed_softmax_raw,
)
from tools.ace2_projection_shadow_reference import (
    K_SCALE32,
    Q_SCALE32,
    projection_shadow_q15_16,
    staged_attention_row,
)


class ProjectionShadowStagedAttentionTest(unittest.TestCase):
    def test_projection_shadow_rounding_and_left_shift(self) -> None:
        self.assertEqual(projection_shadow_q15_16(3, 0, 1, 17), 2)
        self.assertEqual(projection_shadow_q15_16(5, 0, 1, 17), 2)
        self.assertEqual(projection_shadow_q15_16(-3, 0, 1, 17), -2)
        self.assertEqual(projection_shadow_q15_16(7, -2, 3, 14), 60)

    def test_tensor_scores_and_staged_value_match_scalar_oracle(self) -> None:
        rng = random.Random(0xACE25015)
        sequence = 4
        query = torch.tensor(
            [[[[rng.randrange(-4_000_000, 4_000_001) for _ in range(64)]
               for _ in range(sequence)] for _ in range(14)]],
            dtype=torch.int32,
        )
        key = torch.tensor(
            [[[[rng.randrange(-4_000_000, 4_000_001) for _ in range(64)]
               for _ in range(sequence)] for _ in range(2)]],
            dtype=torch.int32,
        )
        value_kv = torch.tensor(
            [[[[rng.randrange(-128, 128) for _ in range(64)]
               for _ in range(sequence)] for _ in range(2)]],
            dtype=torch.int8,
        )
        values = value_kv.repeat_interleave(7, dim=1)
        scores = projection_shadow_staged_attention_scores_raw(
            query, key, Q_SCALE32, K_SCALE32, None
        )
        probabilities = fixed_softmax_raw(scores)
        output = fixed_attention_value_raw(probabilities, values)
        for head in (0, 6, 7, 13):
            kv_head = 0 if head <= 6 else 1
            for position in range(sequence):
                expected = staged_attention_row(
                    query[0, head, position].tolist(),
                    [key[0, kv_head, index].tolist() for index in range(position + 1)],
                    [values[0, head, index].tolist() for index in range(position + 1)],
                    position,
                )
                expected_scores = tuple(
                    max(-32768, score - max(expected.scores_q6_9))
                    for score in expected.scores_q6_9
                )
                self.assertEqual(
                    tuple(scores[0, head, position, : position + 1].tolist()),
                    expected_scores,
                )
                self.assertEqual(
                    tuple(probabilities[0, head, position, : position + 1].tolist()),
                    expected.probabilities_q15,
                )
                self.assertEqual(
                    tuple(output[0, head, position].tolist()),
                    expected.output_s8,
                )

    def test_causal_mask_materialization(self) -> None:
        query = torch.zeros((1, 14, 2, 64), dtype=torch.int32)
        key = torch.zeros((1, 2, 2, 64), dtype=torch.int32)
        scores = projection_shadow_staged_attention_scores_raw(
            query, key, Q_SCALE32, K_SCALE32, None
        )
        self.assertEqual(int(scores[0, 0, 0, 1]), -32768)
        self.assertEqual(int(scores[0, 0, 1, 0]), 0)
        self.assertEqual(int(scores[0, 0, 1, 1]), 0)


if __name__ == "__main__":
    unittest.main()
