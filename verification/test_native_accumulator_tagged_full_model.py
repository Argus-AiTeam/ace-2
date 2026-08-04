from __future__ import annotations

import random
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from tools.ace2_absolute_rope_online_attention_reference import absolute_coefficients_q15
from tools.ace2_full_model_fixed_point import (
    native_accumulator_tagged_attention_scores_raw,
    native_accumulator_tagged_rope_raw,
    native_accumulator_tagged_softmax_raw,
)
from tools.ace2_native_accumulator_tagged_reference import (
    TaggedLane,
    tagged_rope_pair,
    tagged_score_q20_44,
)


class NativeAccumulatorTaggedFullModelTest(unittest.TestCase):
    def test_tensor_rope_and_score_match_scalar_reference(self) -> None:
        rng = random.Random(0xACE2A11)
        sequence = 3
        query = torch.tensor(
            [[[[rng.randrange(-200_000, 200_001) for _ in range(64)]
               for _ in range(sequence)] for _ in range(14)]],
            dtype=torch.int32,
        )
        key = torch.tensor(
            [[[[rng.randrange(-200_000, 200_001) for _ in range(64)]
               for _ in range(sequence)] for _ in range(2)]],
            dtype=torch.int32,
        )
        query_records = torch.tensor(
            [0x00F69000 + (index & 0x0FFF) for index in range(14 * 64)],
            dtype=torch.int64,
        )
        key_records = torch.tensor(
            [0x00F58A00 + (index & 0x0FFF) for index in range(2 * 64)],
            dtype=torch.int64,
        )
        query_mantissa, query_exponent = native_accumulator_tagged_rope_raw(
            query,
            query_records,
        )
        cosine, sine = absolute_coefficients_q15(2)
        for pair in range(32):
            result = tagged_rope_pair(
                int(query[0, 0, 2, pair]),
                int(query_records[pair]),
                int(query[0, 0, 2, pair + 32]),
                int(query_records[pair + 32]),
                cosine[pair],
                sine[pair],
            )
            self.assertEqual(
                (int(query_mantissa[0, 0, 2, pair]), int(query_exponent[0, 0, 2, pair])),
                (result.real.mantissa, result.real.exponent),
            )
            self.assertEqual(
                (
                    int(query_mantissa[0, 0, 2, pair + 32]),
                    int(query_exponent[0, 0, 2, pair + 32]),
                ),
                (result.imag.mantissa, result.imag.exponent),
            )

        scores = native_accumulator_tagged_attention_scores_raw(
            query,
            key,
            query_records,
            key_records,
            None,
        )
        key_mantissa, key_exponent = native_accumulator_tagged_rope_raw(
            key,
            key_records,
        )
        query_lanes = [
            TaggedLane(
                int(query_mantissa[0, 0, 2, lane]),
                int(query_exponent[0, 0, 2, lane]),
            )
            for lane in range(64)
        ]
        expected_scores = []
        for key_position in range(sequence):
            key_lanes = [
                TaggedLane(
                    int(key_mantissa[0, 0, key_position, lane]),
                    int(key_exponent[0, 0, key_position, lane]),
                )
                for lane in range(64)
            ]
            expected_scores.append(tagged_score_q20_44(query_lanes, key_lanes))
        maximum = max(expected_scores)
        self.assertEqual(
            scores[0, 0, 2, :sequence].tolist(),
            [value - maximum for value in expected_scores],
        )
        probabilities = native_accumulator_tagged_softmax_raw(scores)
        self.assertTrue(torch.all(probabilities.sum(dim=-1) > 0))
        self.assertTrue(torch.all(probabilities <= 32767))


if __name__ == "__main__":
    unittest.main()
