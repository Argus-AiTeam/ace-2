from __future__ import annotations

import random
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from tools.ace2_full_model_fixed_point import (
    fixed_attention_value_raw,
    tile_max_delta_attention_scores_raw,
    tile_max_delta_softmax_raw,
)
from tools.ace2_tile_max_delta_reference import (
    DELTA_SENTINEL,
    K_SCALE32,
    Q_SCALE32,
    encode_score_row_from_numerators,
    staged_attention_row,
)


class TileMaxDeltaAttentionTest(unittest.TestCase):
    def test_distinct_preclamp_scores_remain_distinct(self) -> None:
        scale = 1 << 66
        scores = encode_score_row_from_numerators(
            [1000 * scale, 999 * scale, 998 * scale]
        )
        self.assertEqual(scores, (0, -(1 << 17), -(2 << 17)))

    def test_underflow_sentinel_and_cross_tile_merge(self) -> None:
        scale = 1 << 66
        numerators = [0] * 64 + [8 * scale, -40 * scale]
        scores = encode_score_row_from_numerators(numerators)
        self.assertEqual(scores[64], 0)
        self.assertEqual(scores[0], -(8 << 17))
        self.assertEqual(scores[65], DELTA_SENTINEL)

    def test_tensor_scores_softmax_and_value_match_scalar_oracle(self) -> None:
        rng = random.Random(0xACE25016)
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
        scores = tile_max_delta_attention_scores_raw(
            query, key, Q_SCALE32, K_SCALE32, None
        )
        probabilities = tile_max_delta_softmax_raw(scores)
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
                self.assertEqual(
                    tuple(scores[0, head, position, : position + 1].tolist()),
                    expected.scores_q6_17,
                )
                self.assertEqual(
                    tuple(probabilities[0, head, position, : position + 1].tolist()),
                    expected.probabilities_q15,
                )
                self.assertEqual(
                    tuple(output[0, head, position].tolist()),
                    expected.output_s8,
                )

    def test_causal_mask_uses_underflow_sentinel(self) -> None:
        query = torch.zeros((1, 14, 2, 64), dtype=torch.int32)
        key = torch.zeros((1, 2, 2, 64), dtype=torch.int32)
        scores = tile_max_delta_attention_scores_raw(
            query, key, Q_SCALE32, K_SCALE32, None
        )
        self.assertEqual(int(scores[0, 0, 0, 1]), DELTA_SENTINEL)
        self.assertEqual(int(scores[0, 0, 1, 0]), 0)
        self.assertEqual(int(scores[0, 0, 1, 1]), 0)


if __name__ == "__main__":
    unittest.main()
