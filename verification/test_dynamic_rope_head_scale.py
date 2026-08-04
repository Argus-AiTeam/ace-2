from __future__ import annotations

import random
import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from ace2_attention_score_reference import (  # noqa: E402
    DynamicAttentionScoreCase,
    reference_dynamic_attention_score,
)
from ace2_full_model_fixed_point import (  # noqa: E402
    dynamic_rope_head_raw,
    fixed_dynamic_attention_scores_raw,
)
from ace2_quality_contracts import (  # noqa: E402
    SCALE32_ALL_ZERO_RECORD,
    ceil_scale32_from_float,
    ceil_scale32_from_ratio,
    dynamic_score_pair_parameters,
    pack_scale32,
    unpack_scale32,
)
from ace2_rope_reference import (  # noqa: E402
    DynamicRopeHeadCase,
    reference_dynamic_rope_head,
)


class DynamicRopeHeadScaleTests(unittest.TestCase):
    def test_scale32_boundaries_and_invalid_records(self) -> None:
        self.assertEqual(SCALE32_ALL_ZERO_RECORD, pack_scale32(0x8000, -24))
        self.assertEqual(
            ceil_scale32_from_ratio(1, 1 << 100), SCALE32_ALL_ZERO_RECORD
        )
        self.assertEqual(
            dynamic_score_pair_parameters(
                pack_scale32(0xFFFF, 4), pack_scale32(0xFFFF, 4)
            ),
            (0x1FFFC, 1),
        )
        with self.assertRaises(ValueError):
            unpack_scale32(0x01008000)
        with self.assertRaises(ValueError):
            unpack_scale32(0x00007FFF)
        with self.assertRaises(OverflowError):
            ceil_scale32_from_ratio((0xFFFF << 4) + 1, 1 << 15)

    def test_dynamic_rope_tensor_matches_independent_scalar(self) -> None:
        for seed in (0, 1, 7, 31):
            with self.subTest(seed=seed):
                rng = random.Random(seed)
                producer = ceil_scale32_from_float(0.03125 * (seed + 1))
                activations = [rng.randint(-128, 127) for _ in range(64)]
                cos_q15 = [rng.randint(-32768, 32767) for _ in range(64)]
                sin_q15 = [rng.randint(-32768, 32767) for _ in range(64)]
                scalar = reference_dynamic_rope_head(
                    DynamicRopeHeadCase(
                        f"seed_{seed}", activations, producer, cos_q15, sin_q15
                    )
                )
                actual, records, saturation_count = dynamic_rope_head_raw(
                    torch.tensor(activations, dtype=torch.int8).reshape(1, 1, 1, 64),
                    producer,
                    torch.tensor(cos_q15, dtype=torch.float64).reshape(1, 1, 64)
                    / 32767.0,
                    torch.tensor(sin_q15, dtype=torch.float64).reshape(1, 1, 64)
                    / 32767.0,
                )
                self.assertEqual(actual.reshape(-1).tolist(), scalar.outputs)
                self.assertEqual(int(records.item()), scalar.output_scale32)
                self.assertEqual(saturation_count, 0)
                self.assertNotIn(-128, actual.reshape(-1).tolist())

    def test_all_zero_and_amplitude_specific_records(self) -> None:
        producer = ceil_scale32_from_float(0.03125)
        cos = torch.ones((1, 1, 64), dtype=torch.float64)
        sin = torch.zeros((1, 1, 64), dtype=torch.float64)
        zero, zero_record, zero_saturation = dynamic_rope_head_raw(torch.zeros((1, 1, 1, 64), dtype=torch.int8), producer, cos, sin)
        low, low_record, _ = dynamic_rope_head_raw(torch.tensor([1, -1] * 32, dtype=torch.int8).reshape(1, 1, 1, 64), producer, cos, sin)
        high, high_record, _ = dynamic_rope_head_raw(torch.tensor([64, -64] * 32, dtype=torch.int8).reshape(1, 1, 1, 64), producer, cos, sin)
        self.assertEqual(zero.reshape(-1).tolist(), [0] * 64)
        self.assertEqual(int(zero_record.item()), SCALE32_ALL_ZERO_RECORD)
        self.assertEqual(zero_saturation, 0)
        self.assertNotEqual(int(low_record.item()), int(high_record.item()))
        self.assertEqual(max(abs(value) for value in low.reshape(-1).tolist()), 127)
        self.assertEqual(max(abs(value) for value in high.reshape(-1).tolist()), 127)

    def test_dynamic_rope_rejects_bad_trig_shape(self) -> None:
        with self.assertRaisesRegex(ValueError, "64 cosine and sine"):
            dynamic_rope_head_raw(torch.zeros((1, 1, 1, 64), dtype=torch.int8), ceil_scale32_from_float(0.03125), torch.ones((1, 1, 32), dtype=torch.float64), torch.zeros((1, 1, 32), dtype=torch.float64))

    def test_dynamic_score_all_query_head_mappings_and_variable_key_scales(
        self,
    ) -> None:
        rng = random.Random(20260731)
        key_payloads = [
            [rng.randint(-127, 127) for _ in range(64)] for _ in range(3)
        ]
        key_records_by_kv_head = [
            [
                pack_scale32(0x9000 + token * 0x100, -8 + kv_head)
                for token in range(3)
            ]
            for kv_head in range(2)
        ]
        for query_head in range(14):
            with self.subTest(query_head=query_head):
                query_payload = [rng.randint(-127, 127) for _ in range(64)]
                query_record = pack_scale32(0x8800 + query_head * 0x80, -7)
                key_records = key_records_by_kv_head[query_head // 7]
                scalar = reference_dynamic_attention_score(
                    DynamicAttentionScoreCase(
                        f"query_head_{query_head}",
                        query_payload,
                        key_payloads,
                        query_record,
                        key_records,
                    )
                )
                actual = fixed_dynamic_attention_scores_raw(
                    torch.tensor(query_payload, dtype=torch.int8).reshape(
                        1, 1, 1, 64
                    ),
                    torch.tensor(key_payloads, dtype=torch.int8).reshape(1, 1, 3, 64),
                    torch.tensor([query_record], dtype=torch.int64).reshape(1, 1, 1),
                    torch.tensor(key_records, dtype=torch.int64).reshape(1, 1, 3),
                    None,
                )
                self.assertEqual(
                    actual.reshape(-1).tolist(), scalar.scores_q6_9
                )


if __name__ == "__main__":
    unittest.main()
