from __future__ import annotations

import random
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from ace2_fixed_q7_reference import (  # noqa: E402
    Descriptor,
    SRAM_PEAK_BYTES,
    SRAM_TAG,
    fixed_q7_precenter_score,
    fixed_q7_rope_head,
    four_phase_signed16_product,
    legal_fixed_q7_tuple,
)
from ace2_full_model_fixed_point import (  # noqa: E402
    fixed_q7_attention_scores_raw,
    fixed_q7_rope_head_raw,
    rope_linear_contract,
    rope_mechanism_for_layer,
)
from ace2_quality_contracts import pack_scale32  # noqa: E402


class Layer0FixedQ7Test(unittest.TestCase):
    def test_contract_is_strictly_layer0_only(self) -> None:
        mechanism = "layer0_fixed_q7_rope_score_v1"
        self.assertEqual(
            rope_linear_contract("model.layers.0.self_attn.q_proj", mechanism),
            (False, True),
        )
        self.assertEqual(
            rope_linear_contract("model.layers.0.self_attn.k_proj", mechanism),
            (False, True),
        )
        for layer_index in (1, 12, 23):
            self.assertEqual(
                rope_linear_contract(
                    f"model.layers.{layer_index}.self_attn.q_proj",
                    mechanism,
                ),
                (True, False),
            )
            self.assertIsNone(rope_mechanism_for_layer(layer_index, mechanism))
        self.assertEqual(
            rope_linear_contract("model.layers.0.self_attn.v_proj", mechanism),
            (False, False),
        )
        self.assertEqual(rope_mechanism_for_layer(0, mechanism), mechanism)

    def test_rope_scalar_tensor_and_bounds(self) -> None:
        activations = [((index * 37) % 256) - 128 for index in range(64)]
        cosine_half = [32767 if index & 1 else 23171 for index in range(32)]
        sine_half = [0 if index & 1 else 23171 for index in range(32)]
        cosine = cosine_half * 2
        sine = sine_half * 2
        expected = fixed_q7_rope_head(activations, cosine, sine)
        actual = fixed_q7_rope_head_raw(
            torch.tensor(activations, dtype=torch.int8).reshape(1, 1, 1, 64),
            torch.tensor(cosine, dtype=torch.float64).reshape(1, 1, 64) / 32767.0,
            torch.tensor(sine, dtype=torch.float64).reshape(1, 1, 64) / 32767.0,
        )
        self.assertEqual(actual.reshape(-1).tolist(), expected)
        self.assertLessEqual(max(abs(value) for value in expected), 23_171)
        corrupted = cosine.copy()
        corrupted[0] = 32767
        corrupted[32] = 32767
        bad_sine = sine.copy()
        bad_sine[0] = 32767
        bad_sine[32] = 32767
        with self.assertRaises(ValueError):
            fixed_q7_rope_head(activations, corrupted, bad_sine)

    def test_four_phase_product(self) -> None:
        corners = [-32768, -32641, -256, -255, -129, -128, -127, -1, 0, 1, 127, 128, 255, 256, 32639, 32767]
        for query in corners:
            for key in corners:
                product, _ = four_phase_signed16_product(query, key)
                self.assertEqual(product, query * key)
        for query_low in range(256):
            for key_low in range(256):
                query = ((0x81 << 8) | query_low) - 0x10000
                key = ((0x7E << 8) | key_low)
                product, _ = four_phase_signed16_product(query, key)
                self.assertEqual(product, query * key)

    def test_score_scalar_tensor(self) -> None:
        rng = random.Random(20260731)
        query = [rng.randint(-23_171, 23_171) for _ in range(64)]
        keys = [
            [rng.randint(-23_171, 23_171) for _ in range(64)]
            for _ in range(3)
        ]
        query_scale = pack_scale32(0xC123, -5)
        key_scale = pack_scale32(0x9ABC, -6)
        precenter = [
            fixed_q7_precenter_score(query, key, query_scale, key_scale)[0]
            for key in keys
        ]
        maximum = max(precenter)
        expected = [max(-32768, min(0, value - maximum)) for value in precenter]
        actual = fixed_q7_attention_scores_raw(
            torch.tensor(query, dtype=torch.int16).reshape(1, 1, 1, 64),
            torch.tensor(keys, dtype=torch.int16).reshape(1, 1, 3, 64),
            query_scale,
            key_scale,
            None,
        )
        self.assertEqual(actual.reshape(-1).tolist(), expected)

    def test_descriptor_tuple_and_sram_contract(self) -> None:
        q_base = SRAM_TAG | 0x1000
        rope = Descriptor(0x03, 0x00, 0, 1, 896, 64, q_base, pack_scale32(0x8000, -8), q_base)
        self.assertTrue(legal_fixed_q7_tuple(rope))
        self.assertTrue(legal_fixed_q7_tuple(Descriptor(**{**rope.__dict__, "flags": 0x08})))
        self.assertFalse(legal_fixed_q7_tuple(Descriptor(**{**rope.__dict__, "flags": 0x04})))
        self.assertFalse(legal_fixed_q7_tuple(Descriptor(**{**rope.__dict__, "n": 128, "dst_addr": q_base + 16})))
        score = Descriptor(0x04, 0, 0, 1, 16, 64, q_base, SRAM_TAG | 0x2000, scale_addr=0, stride1=400, aux=13)
        self.assertTrue(legal_fixed_q7_tuple(score))
        self.assertFalse(legal_fixed_q7_tuple(Descriptor(**{**score.__dict__, "stride1": 272})))
        self.assertEqual(SRAM_PEAK_BYTES, 506_368)


if __name__ == "__main__":
    unittest.main()
