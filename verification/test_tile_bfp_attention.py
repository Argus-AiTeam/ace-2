from __future__ import annotations

import random
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from tools.ace2_full_model_fixed_point import (
    fixed_attention_value_raw,
    tile_bfp_attention_scores_raw,
    tile_bfp_softmax_raw,
)
from tools.ace2_tile_bfp_reference import (
    K_SCALE32,
    Q_SCALE32,
    encode_score_row_from_numerators,
    encode_tile,
    staged_attention_row,
)


class TileBfpAttentionTest(unittest.TestCase):
    def test_fraction_selection_boundary_and_complete_range(self) -> None:
        unit = 1 << 66
        high_precision = encode_tile([0, -(63 * unit)])
        boundary = encode_tile([0, -(64 * unit)])
        self.assertEqual(high_precision.fraction_bits_u5, 17)
        self.assertEqual(boundary.fraction_bits_u5, 16)
        self.assertEqual(boundary.mantissas_s24, (0, -(64 << 16)))
        self.assertEqual(
            encode_score_row_from_numerators([0, -(64 * unit)]),
            (0, -(64 << 17)),
        )

    def test_signed_ties_round_to_even(self) -> None:
        tile = encode_tile([0, -(1 << 48), -(3 << 48)])
        self.assertEqual(tile.fraction_bits_u5, 17)
        self.assertEqual(tile.mantissas_s24, (0, 0, -2))

    def test_row_lengths_cover_tile_boundary(self) -> None:
        unit = 1 << 66
        for count in (1, 2, 63, 64, 65):
            numerators = [(count - index) * unit for index in range(count)]
            scores = encode_score_row_from_numerators(numerators)
            self.assertEqual(len(scores), count)
            self.assertEqual(scores[0], 0)
            self.assertEqual(scores[-1], -((count - 1) << 17))

    def test_tensor_scores_softmax_and_value_match_scalar_oracle(self) -> None:
        rng = random.Random(0xACE2B0F)
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
        scores = tile_bfp_attention_scores_raw(query, key, Q_SCALE32, K_SCALE32, None)
        probabilities = tile_bfp_softmax_raw(scores)
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
                    expected.scores_q17,
                )
                self.assertEqual(
                    tuple(probabilities[0, head, position, : position + 1].tolist()),
                    expected.probabilities_q15,
                )
                self.assertEqual(
                    tuple(output[0, head, position].tolist()),
                    expected.output_s8,
                )


if __name__ == "__main__":
    unittest.main()
