from __future__ import annotations

import hashlib
import math
import random
import struct
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from ace2_full_model_fixed_point import (  # noqa: E402
    relative_rope_attention_scores_raw,
    rope_linear_contract,
    rope_mechanism_for_layer,
)
from ace2_quality_contracts import pack_scale32  # noqa: E402
from ace2_relative_rope_score_reference import (  # noqa: E402
    GlobalRowState,
    SRAM_MARGIN_BYTES,
    SRAM_PEAK_BYTES,
    STATE_EMPTY,
    STATE_SEALED,
    center_score,
    emit_centered_tile,
    relative_coefficients_q15,
    relative_rope_centered_row,
    relative_rope_phase_acc,
    scan_precenter_tile,
)


class Layer0RelativeRopeScoreFusionTest(unittest.TestCase):
    def test_contract_is_strictly_layer0_only(self) -> None:
        mechanism = "layer0_relative_rope_score_fusion_v1"
        self.assertEqual(
            rope_linear_contract("model.layers.0.self_attn.q_proj", mechanism),
            (False, True),
        )
        self.assertEqual(
            rope_linear_contract("model.layers.0.self_attn.k_proj", mechanism),
            (False, True),
        )
        self.assertEqual(rope_mechanism_for_layer(0, mechanism), mechanism)
        for layer_index in (1, 12, 23):
            self.assertEqual(
                rope_linear_contract(
                    f"model.layers.{layer_index}.self_attn.q_proj", mechanism
                ),
                (True, False),
            )
            self.assertIsNone(rope_mechanism_for_layer(layer_index, mechanism))

    def test_coefficient_prefix_hashes_and_bound(self) -> None:
        payload = bytearray()
        maximum_l1 = 0
        prefix_hashes: dict[int, str] = {}
        for distance in range(32768):
            cosine, sine = relative_coefficients_q15(distance)
            for c_value, s_value in zip(cosine, sine, strict=True):
                payload.extend(struct.pack("<hh", c_value, s_value))
                maximum_l1 = max(maximum_l1, abs(c_value) + abs(s_value))
            if distance + 1 in {93, 128}:
                prefix_hashes[distance + 1] = hashlib.sha256(payload).hexdigest()
        self.assertEqual(maximum_l1, 46462)
        self.assertEqual(
            prefix_hashes[93],
            "e9607786693cf4c0ae21f0afd8401c640cbee1d67d57312340c688524d897bfc",
        )
        self.assertEqual(
            prefix_hashes[128],
            "9617827682c06ac9fb50740e2f51d51f003e9d42433d19cc41b0959da9dbdd9d",
        )
        self.assertEqual(
            hashlib.sha256(payload).hexdigest(),
            "62fd37a6e4dabc6abf89301dca6fb56cda61be329477155dbb7ffa02ca0fb325",
        )

    def test_bilinear_matches_explicit_relative_rotation(self) -> None:
        rng = random.Random(20260731)
        for distance in (0, 1, 17, 127, 256, 32767):
            query = [rng.randint(-128, 127) for _ in range(64)]
            key = [rng.randint(-128, 127) for _ in range(64)]
            cosine, sine = relative_coefficients_q15(distance)
            explicit = 0
            for pair in range(32):
                q0, q1 = query[pair], query[pair + 32]
                k0, k1 = key[pair], key[pair + 32]
                c_value, s_value = cosine[pair], sine[pair]
                rq0 = q0 * c_value - q1 * s_value
                rq1 = q0 * s_value + q1 * c_value
                explicit += rq0 * k0 + rq1 * k1
            self.assertEqual(relative_rope_phase_acc(query, key, distance), explicit)

    def test_257_key_global_centering(self) -> None:
        precenter = [10, 9] + [-20] * 254 + [100]
        state = scan_precenter_tile(
            GlobalRowState(),
            precenter[:256],
            query_head=0,
            query_position=256,
            key_base=0,
        )
        self.assertEqual(state.maximum, 10)
        state = scan_precenter_tile(
            state,
            precenter[256:],
            query_head=0,
            query_position=256,
            key_base=256,
        )
        self.assertEqual(state.phase, STATE_SEALED)
        self.assertEqual(state.maximum, 100)
        state, first = emit_centered_tile(
            state,
            precenter[:256],
            query_head=0,
            query_position=256,
            key_base=0,
        )
        state, second = emit_centered_tile(
            state,
            precenter[256:],
            query_head=0,
            query_position=256,
            key_base=256,
        )
        payload = b"".join(struct.pack("<h", value) for value in first + second)
        self.assertEqual(
            hashlib.sha256(payload).hexdigest(),
            "e570bb9241979e9891868465ac910a87cd1e57f698503f649d57d5342c63ae3f",
        )
        self.assertEqual(state.phase, STATE_EMPTY)
        self.assertEqual(SRAM_PEAK_BYTES, 495_120)
        self.assertEqual(SRAM_MARGIN_BYTES, 29_168)

    def test_tensor_path_matches_scalar_oracle(self) -> None:
        rng = random.Random(2026073101)
        query = torch.tensor(
            [rng.randint(-128, 127) for _ in range(2 * 14 * 4 * 64)],
            dtype=torch.int8,
        ).reshape(2, 14, 4, 64)
        key = torch.tensor(
            [rng.randint(-128, 127) for _ in range(2 * 2 * 4 * 64)],
            dtype=torch.int8,
        ).reshape(2, 2, 4, 64)
        query_scale = pack_scale32(0xA245, -1)
        key_scale = pack_scale32(0x8307, 0)
        actual = relative_rope_attention_scores_raw(
            query, key, query_scale, key_scale, None
        )
        repeated_key = key.repeat_interleave(7, dim=1)
        for batch in range(2):
            for head in range(14):
                for position in range(4):
                    expected = relative_rope_centered_row(
                        query[batch, head, position].tolist(),
                        [
                            repeated_key[batch, head, key_position].tolist()
                            for key_position in range(position + 1)
                        ],
                        query_scale,
                        key_scale,
                        position,
                    )
                    row = actual[batch, head, position].tolist()
                    self.assertEqual(row[: position + 1], expected)
                    self.assertEqual(row[position + 1 :], [-32768] * (3 - position))


if __name__ == "__main__":
    unittest.main()
