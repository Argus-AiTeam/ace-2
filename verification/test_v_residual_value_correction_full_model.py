from __future__ import annotations

import random
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from tools.ace2_full_model_fixed_point import (
    repeat_kv,
    round_shift_even,
    v_residual_layer_metadata,
    v_residual_projection_raw,
    v_residual_value_accumulators_raw,
)
from tools.ace2_v_residual_value_correction_reference import (
    canonical_s4_byte,
    v_projection_residual,
    v_residual_value_correction,
)


class VResidualValueCorrectionFullModelTest(unittest.TestCase):
    def test_tensor_projection_residual_matches_scalar_reference(self) -> None:
        rng = random.Random(0xACE25201)
        metadata = v_residual_layer_metadata(0)
        baseline_records = torch.tensor(
            metadata["baseline_v_scale32"], dtype=torch.int64
        )
        residual_records = torch.tensor(
            metadata["residual_v_scale32"], dtype=torch.int64
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
        residual, positive_clamps, negative_clamps = v_residual_projection_raw(
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
                kv_head = channel // 64
                expected = v_projection_residual(
                    int(accumulators[row, channel]),
                    int(multipliers[channel]),
                    int(shifts[channel]),
                    int(baseline_records[kv_head]),
                    int(residual_records[kv_head]),
                )
                self.assertEqual(int(baseline[row, channel]), expected.baseline_v8)
                self.assertEqual(int(residual[row, channel]), expected.residual_v_s4)
                expected_positive += int(expected.positive_clamp)
                expected_negative += int(expected.negative_clamp)
        self.assertEqual(positive_clamps, expected_positive)
        self.assertEqual(negative_clamps, expected_negative)

    def test_tensor_sum_then_convert_matches_scalar_reference(self) -> None:
        rng = random.Random(0xACE25202)
        metadata = v_residual_layer_metadata(7)
        baseline_records = torch.tensor(
            metadata["baseline_v_scale32"], dtype=torch.int64
        )
        residual_records = torch.tensor(
            metadata["residual_v_scale32"], dtype=torch.int64
        )
        query_length = 3
        key_length = 5
        probabilities = torch.tensor(
            [
                [
                    [rng.randrange(0, 32768) for _ in range(key_length)]
                    for _ in range(query_length)
                ]
                for _ in range(14)
            ],
            dtype=torch.int64,
        ).unsqueeze(0)
        kv_values = torch.tensor(
            [
                [
                    [[rng.randrange(-128, 128) for _ in range(64)] for _ in range(key_length)]
                    for _ in range(2)
                ]
            ],
            dtype=torch.int8,
        )
        kv_residual = torch.tensor(
            [
                [
                    [[rng.randrange(-7, 8) for _ in range(64)] for _ in range(key_length)]
                    for _ in range(2)
                ]
            ],
            dtype=torch.int8,
        )
        values = repeat_kv(kv_values, 7)
        residuals = repeat_kv(kv_residual, 7)
        baseline, correction, corrected = v_residual_value_accumulators_raw(
            probabilities,
            values,
            residuals,
            baseline_records,
            residual_records,
            enable_correction=True,
        )
        for query_head in range(14):
            kv_head = query_head // 7
            for query_position in range(query_length):
                probability_row = probabilities[
                    0, query_head, query_position
                ].tolist()
                for lane in range(64):
                    value_row = values[0, query_head, :, lane].tolist()
                    residual_row = residuals[0, query_head, :, lane].tolist()
                    baseline_accumulator = sum(
                        probability * value
                        for probability, value in zip(
                            probability_row, value_row, strict=True
                        )
                    )
                    expected = v_residual_value_correction(
                        probability_row,
                        [canonical_s4_byte(value) for value in residual_row],
                        int(baseline_records[kv_head]),
                        int(residual_records[kv_head]),
                        baseline_accumulator,
                    )
                    self.assertEqual(
                        int(baseline[0, query_head, query_position, lane]),
                        baseline_accumulator,
                    )
                    self.assertEqual(
                        int(correction[0, query_head, query_position, lane]),
                        expected.correction_baseline_domain_s32,
                    )
                    self.assertEqual(
                        int(corrected[0, query_head, query_position, lane]),
                        expected.corrected_accumulator_s32,
                    )

    def test_disabled_correction_preserves_authoritative_accumulator(self) -> None:
        metadata = v_residual_layer_metadata(23)
        probabilities = torch.tensor([[[[32767]] for _ in range(14)]], dtype=torch.int64)
        values = torch.arange(-32, 32, dtype=torch.int8).reshape(1, 1, 1, 64).repeat(
            1, 14, 1, 1
        )
        residuals = torch.full_like(values, 7)
        baseline, correction, corrected = v_residual_value_accumulators_raw(
            probabilities,
            values,
            residuals,
            torch.tensor(metadata["baseline_v_scale32"], dtype=torch.int64),
            torch.tensor(metadata["residual_v_scale32"], dtype=torch.int64),
            enable_correction=False,
        )
        self.assertTrue(torch.equal(correction, torch.zeros_like(correction)))
        self.assertTrue(torch.equal(corrected, baseline))

    def test_metadata_covers_all_layers_and_kv_heads(self) -> None:
        for layer in range(24):
            metadata = v_residual_layer_metadata(layer)
            self.assertEqual(len(metadata["baseline_v_scale32"]), 2)
            self.assertEqual(len(metadata["residual_v_scale32"]), 2)
            self.assertEqual(
                metadata["baseline_v_scale32"][0],
                metadata["baseline_v_scale32"][1],
            )


if __name__ == "__main__":
    unittest.main()
