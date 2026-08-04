from __future__ import annotations

import random
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from tools.ace2_absolute_rope_online_attention_reference import (
    K_SCALE32,
    Q_SCALE32,
    EXP_LIMIT,
    OnlineAttentionState,
    absolute_coefficients_q15,
    absolute_rope_score_raw,
    exp_q31,
    exp_table_q31,
    finalize_online_state,
    score_raw_to_logit_q12_20,
    update_online_state,
    online_attention_row,
)
from tools.ace2_full_model_fixed_point import (
    absolute_rope_online_attention_raw,
    repeat_kv,
)


class AbsoluteRopeOnlineAttentionTest(unittest.TestCase):
    def test_exact_environment_anchor(self) -> None:
        cosine, sine = absolute_coefficients_q15(32767)
        self.assertEqual(cosine[7], 31487)
        self.assertEqual(sine[7], -9152)
        self.assertEqual(absolute_coefficients_q15(0), ((32767,) * 32, (0,) * 32))

    def test_exp_table_and_interpolation_boundaries(self) -> None:
        table = exp_table_q31()
        self.assertEqual(len(table), 257)
        self.assertEqual(table[0], 1 << 31)
        self.assertTrue(all(a >= b for a, b in zip(table, table[1:])))
        self.assertEqual(exp_q31(0), 1 << 31)
        self.assertEqual(exp_q31(-EXP_LIMIT), 0)
        self.assertEqual(exp_q31(-EXP_LIMIT - 1), 0)
        for index in range(256):
            self.assertEqual(exp_q31(-(index << 16)), table[index])

    def test_score_and_online_branches(self) -> None:
        rng = random.Random(0xACE2A801)
        query = [rng.randrange(-128, 128) for _ in range(64)]
        state = OnlineAttentionState()
        saw_no_new_max = False
        saw_new_max = False
        previous_max = None
        for key_position in range(8):
            key = [rng.randrange(-128, 128) for _ in range(64)]
            value = [rng.randrange(-128, 128) for _ in range(64)]
            score, rotated_query, rotated_key = absolute_rope_score_raw(
                query, key, 7, key_position
            )
            self.assertTrue(all(-(1 << 24) <= lane < (1 << 24) for lane in rotated_query))
            self.assertTrue(all(-(1 << 24) <= lane < (1 << 24) for lane in rotated_key))
            logit = score_raw_to_logit_q12_20(score, Q_SCALE32, K_SCALE32)
            state = update_online_state(state, logit, value)
            if previous_max is not None:
                saw_new_max |= state.maximum != previous_max
                saw_no_new_max |= state.maximum == previous_max
            previous_max = state.maximum
        self.assertTrue(saw_new_max)
        self.assertTrue(saw_no_new_max)
        output = finalize_online_state(state)
        self.assertEqual(len(output), 64)
        self.assertTrue(all(-128 <= lane <= 127 for lane in output))

    def test_ties_to_even_final_division(self) -> None:
        state = OnlineAttentionState(maximum=0, denominator=4, numerators=(2, 6, -2, -6) + (0,) * 60)
        output = finalize_online_state(state)
        self.assertEqual(output[:4], (0, 2, 0, -2))

    def test_tensor_path_matches_independent_rows(self) -> None:
        rng = random.Random(0xA850A11)
        query = torch.tensor(
            [[[[rng.randrange(-128, 128) for _ in range(64)] for _ in range(2)]
              for _ in range(14)]],
            dtype=torch.int8,
        )
        key = torch.tensor(
            [[[[rng.randrange(-128, 128) for _ in range(64)] for _ in range(2)]
              for _ in range(2)]],
            dtype=torch.int8,
        )
        value_kv = torch.tensor(
            [[[[rng.randrange(-128, 128) for _ in range(64)] for _ in range(2)]
              for _ in range(2)]],
            dtype=torch.int8,
        )
        values = repeat_kv(value_kv, 7)
        actual = absolute_rope_online_attention_raw(
            query, key, values, None, Q_SCALE32, K_SCALE32
        )
        for head in range(14):
            kv_head = 0 if head <= 6 else 1
            for position in range(2):
                expected, _, _ = online_attention_row(
                    query[0, head, position].tolist(),
                    [key[0, kv_head, index].tolist() for index in range(position + 1)],
                    [values[0, head, index].tolist() for index in range(position + 1)],
                    position,
                )
                self.assertEqual(tuple(actual[0, head, position].tolist()), expected)


if __name__ == "__main__":
    unittest.main()
