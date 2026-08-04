from __future__ import annotations

import random
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from tools.ace2_full_model_fixed_point import (
    qk_authoritative_base_scores_q20_44_raw,
    qk_residual_cross_term_scores_raw,
    qk_residual_layer_metadata,
    qk_residual_projection_raw,
    qk_residual_rope_raw,
    round_shift_even,
)
from tools.ace2_qk_residual_cross_term_reference import (
    projection_residual,
    residual_cross_term_score,
    residual_rope_pair,
)


class QkResidualCrossTermFullModelTest(unittest.TestCase):
    def test_tensor_projection_residual_matches_scalar_reference(self) -> None:
        rng = random.Random(0xACE25101)
        metadata = qk_residual_layer_metadata(0)
        baseline_records = torch.tensor(
            metadata["k_proj_baseline_scale32"], dtype=torch.int64
        )
        residual_records = torch.tensor(
            metadata["k_proj_residual_scale32"], dtype=torch.int64
        )
        accumulators = torch.tensor(
            [[rng.randrange(-2_000_000, 2_000_001) for _ in range(128)] for _ in range(3)],
            dtype=torch.int64,
        )
        multipliers = torch.tensor(
            [rng.randrange(1 << 20, 1 << 27) for _ in range(128)],
            dtype=torch.int64,
        )
        shifts = torch.tensor(
            [rng.randrange(27, 38) for _ in range(128)],
            dtype=torch.int64,
        )
        product = accumulators * multipliers
        baseline = round_shift_even(product, shifts).clamp(-128, 127).to(torch.int8)
        residual, positive_clamps, negative_clamps = qk_residual_projection_raw(
            accumulators,
            multipliers,
            shifts,
            baseline,
            baseline_records,
            residual_records,
        )
        expected_positive = 0
        expected_negative = 0
        for row in range(accumulators.shape[0]):
            for channel in range(accumulators.shape[1]):
                head = channel // 64
                expected = projection_residual(
                    int(accumulators[row, channel]),
                    int(multipliers[channel]),
                    int(shifts[channel]),
                    int(baseline_records[head]),
                    int(residual_records[head]),
                )
                self.assertEqual(int(baseline[row, channel]), expected.baseline_q8)
                self.assertEqual(int(residual[row, channel]), expected.residual_s4)
                expected_positive += int(expected.positive_clamp)
                expected_negative += int(expected.negative_clamp)
        self.assertEqual(positive_clamps, expected_positive)
        self.assertEqual(negative_clamps, expected_negative)

    def test_tensor_residual_rope_matches_scalar_reference(self) -> None:
        rng = random.Random(0xACE25102)
        residual = torch.tensor(
            [[[[rng.randrange(-7, 8) for _ in range(64)] for _ in range(4)] for _ in range(2)]],
            dtype=torch.int8,
        )
        rotated = qk_residual_rope_raw(residual)
        from tools.ace2_absolute_rope_online_attention_reference import (
            absolute_coefficients_q15,
        )

        for position in range(4):
            cosine, sine = absolute_coefficients_q15(position)
            for head in range(2):
                for pair in range(32):
                    expected = residual_rope_pair(
                        int(residual[0, head, position, pair]),
                        int(residual[0, head, position, pair + 32]),
                        cosine[pair],
                        sine[pair],
                    )
                    self.assertEqual(
                        int(rotated[0, head, position, pair]), expected.real_s8
                    )
                    self.assertEqual(
                        int(rotated[0, head, position, pair + 32]), expected.imag_s8
                    )

    def test_tensor_score_carries_base_and_matches_scalar_corrections(self) -> None:
        rng = random.Random(0xACE25103)
        sequence = 3
        metadata = qk_residual_layer_metadata(7)
        query = torch.tensor(
            [[[[rng.randrange(-64, 65) for _ in range(64)] for _ in range(sequence)] for _ in range(14)]],
            dtype=torch.int8,
        )
        key = torch.tensor(
            [[[[rng.randrange(-64, 65) for _ in range(64)] for _ in range(sequence)] for _ in range(2)]],
            dtype=torch.int8,
        )
        query_residual = torch.tensor(
            [[[[rng.randrange(-7, 8) for _ in range(64)] for _ in range(sequence)] for _ in range(14)]],
            dtype=torch.int8,
        )
        key_residual = torch.tensor(
            [[[[rng.randrange(-7, 8) for _ in range(64)] for _ in range(sequence)] for _ in range(2)]],
            dtype=torch.int8,
        )
        query_scale = torch.tensor(metadata["q_proj_baseline_scale32"], dtype=torch.int64)
        key_scale = torch.tensor(metadata["k_proj_baseline_scale32"], dtype=torch.int64)
        query_residual_scale = torch.tensor(
            metadata["q_proj_residual_scale32"], dtype=torch.int64
        )
        key_residual_scale = torch.tensor(
            metadata["k_proj_residual_scale32"], dtype=torch.int64
        )
        base = qk_authoritative_base_scores_q20_44_raw(
            query, key, query_scale, key_scale
        )
        baseline_centered = qk_residual_cross_term_scores_raw(
            base,
            query,
            key,
            query_residual,
            key_residual,
            query_scale,
            key_scale,
            query_residual_scale,
            key_residual_scale,
            None,
            enable_correction=False,
        )
        corrected = qk_residual_cross_term_scores_raw(
            base,
            query,
            key,
            query_residual,
            key_residual,
            query_scale,
            key_scale,
            query_residual_scale,
            key_residual_scale,
            None,
            enable_correction=True,
        )
        for query_head in range(14):
            kv_head = query_head // 7
            for query_position in range(sequence):
                expected_row = []
                for key_position in range(query_position + 1):
                    expected = residual_cross_term_score(
                        int(base[0, query_head, query_position, key_position]),
                        query[0, query_head, query_position].tolist(),
                        key[0, kv_head, key_position].tolist(),
                        query_residual[0, query_head, query_position].tolist(),
                        key_residual[0, kv_head, key_position].tolist(),
                        int(query_scale[query_head]),
                        int(key_scale[kv_head]),
                        int(query_residual_scale[query_head]),
                        int(key_residual_scale[kv_head]),
                    )
                    expected_row.append(expected.score_q20_44_s64)
                maximum = max(expected_row)
                self.assertEqual(
                    corrected[0, query_head, query_position, : query_position + 1].tolist(),
                    [value - maximum for value in expected_row],
                )
                base_row = base[0, query_head, query_position, : query_position + 1]
                self.assertEqual(
                    baseline_centered[
                        0, query_head, query_position, : query_position + 1
                    ].tolist(),
                    (base_row - base_row.max()).tolist(),
                )

    def test_metadata_covers_all_layers_and_heads(self) -> None:
        for layer in range(24):
            metadata = qk_residual_layer_metadata(layer)
            self.assertEqual(len(metadata["q_proj_baseline_scale32"]), 14)
            self.assertEqual(len(metadata["q_proj_residual_scale32"]), 14)
            self.assertEqual(len(metadata["k_proj_baseline_scale32"]), 2)
            self.assertEqual(len(metadata["k_proj_residual_scale32"]), 2)


if __name__ == "__main__":
    unittest.main()
