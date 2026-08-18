from __future__ import annotations

import copy
import json
import subprocess
import sys
import unittest
from pathlib import Path

from tools import model_hardware_contract as hardware
from tools import quantization_policy as policy


ROOT = Path(__file__).resolve().parents[1]


class QuantizationPolicyTest(unittest.TestCase):
    def test_all_models_and_policies_are_deterministic_and_valid(self) -> None:
        for model_id in hardware.CONFIG_BY_ID:
            for policy_id in policy.POLICY_IDS:
                with self.subTest(model_id=model_id, policy_id=policy_id):
                    first = policy.build_plan(model_id, policy_id)
                    second = policy.build_plan(model_id, policy_id)
                    self.assertEqual(policy.canonical_bytes(first), policy.canonical_bytes(second))
                    self.assertEqual(policy.validate_plan(first), first)
                    self.assertEqual(
                        [record["operator_class"] for record in first["operators"]],
                        list(policy.OPERATOR_CLASSES),
                    )

    def test_w4a8_preserves_published_memory_and_current_format(self) -> None:
        descriptor = hardware.load_descriptor("qwen2.5-0.5b")
        plan = policy.build_plan("qwen2.5-0.5b", "w4a8")
        self.assertEqual(
            plan["estimates"]["weight_memory_bytes"],
            descriptor["derived"]["estimated_weight_bytes"],
        )
        self.assertEqual(
            plan["estimates"]["maximum_kv_cache_bytes"],
            descriptor["derived"]["maximum_kv_bytes"],
        )
        self.assertEqual(plan["deployment_status"], "current_rtl_format")
        self.assertTrue(
            all(
                record["hardware_support"] == "implemented_rtl_format"
                for record in plan["operators"]
            )
        )

    def test_candidate_formats_never_claim_rtl_execution(self) -> None:
        w8 = policy.build_plan("qwen2.5-1.5b", "w8a8")
        mixed = policy.build_plan("qwen2.5-1.5b", "mixed_w4a8_a16_bf16")
        self.assertEqual(
            w8["operators"][0]["hardware_support"],
            "structural_candidate_no_rtl_execution",
        )
        self.assertEqual(mixed["operators"][2]["activation_format"], "bf16")
        self.assertEqual(mixed["operators"][5]["activation_format"], "int16")
        for plan in (w8, mixed):
            self.assertEqual(
                plan["deployment_status"],
                "structural_candidate_no_full_rtl_execution",
            )
            self.assertFalse(any(plan["claims"].values()))
            with self.assertRaisesRegex(
                policy.QuantizationPolicyError,
                "requires the W4A8 policy",
            ):
                policy.deployment_preflight(plan, require_current_rtl=True)

    def test_inconsistent_plans_fail_closed(self) -> None:
        plan = policy.build_plan("qwen2.5-0.5b", "w4a8")
        cases = []
        drifted_format = copy.deepcopy(plan)
        drifted_format["operators"][0]["weight_format"] = "int8_twos_complement"
        cases.append(drifted_format)
        drifted_estimate = copy.deepcopy(plan)
        drifted_estimate["estimates"]["weight_memory_bytes"] += 1
        cases.append(drifted_estimate)
        reordered = copy.deepcopy(plan)
        reordered["operators"][0], reordered["operators"][1] = (
            reordered["operators"][1],
            reordered["operators"][0],
        )
        cases.append(reordered)
        for value in cases:
            with self.subTest(value=value):
                with self.assertRaises(policy.QuantizationPolicyError):
                    policy.validate_plan(value)

    def test_schema_type_substitutions_fail_closed(self) -> None:
        plan = policy.build_plan("qwen2.5-0.5b", "w4a8")
        cases = {}
        boolean_for_integer = copy.deepcopy(plan)
        boolean_for_integer["schema_version"] = True
        cases["boolean for integer"] = boolean_for_integer
        integer_for_boolean = copy.deepcopy(plan)
        integer_for_boolean["claims"]["full_model_rtl_execution_validated"] = 0
        cases["integer for boolean"] = integer_for_boolean
        float_for_integer = copy.deepcopy(plan)
        float_for_integer["estimates"]["weight_memory_bytes"] = float(
            plan["estimates"]["weight_memory_bytes"]
        )
        cases["float for integer"] = float_for_integer
        for substitution, value in cases.items():
            with self.subTest(substitution=substitution):
                with self.assertRaises(policy.QuantizationPolicyError):
                    policy.validate_plan(value)

    def test_current_rtl_preflight_is_limited_to_existing_0p5b_w4a8(self) -> None:
        result = policy.runtime_preflight(
            model_id="qwen2.5-0.5b",
            policy_id="w4a8",
            require_current_rtl=True,
        )
        self.assertEqual(result["status"], "PASS_CURRENT_RTL_QUANTIZATION_POLICY")
        with self.assertRaisesRegex(
            policy.QuantizationPolicyError,
            "structural compatibility only",
        ):
            policy.runtime_preflight(
                model_id="qwen2.5-1.5b",
                policy_id="w4a8",
                require_current_rtl=True,
            )

    def test_signed_int4_numeric_reference_round_trips_exactly(self) -> None:
        reference = policy.numeric_reference()
        self.assertEqual(reference["quantized"], [-8, -4, 0, 4, 7])
        self.assertEqual(reference["packed_hex"], "c84007")
        self.assertEqual(reference["reconstructed"], reference["input"])
        self.assertEqual(reference["maximum_absolute_error"], 0.0)
        with self.assertRaisesRegex(policy.QuantizationPolicyError, "out of range"):
            policy.pack_signed_int4([8])
        with self.assertRaisesRegex(policy.QuantizationPolicyError, "nonzero padding"):
            policy.unpack_signed_int4(bytes.fromhex("f0"), 1)

    def test_signed_int4_quantization_uses_ties_to_even_and_saturates(self) -> None:
        scale = 0.125
        self.assertEqual(
            policy.quantize_signed_int4(
                [-0.3125, -0.1875, -0.0625, 0.0625, 0.1875, 0.3125],
                scale,
            ),
            [-2, -2, 0, 0, 2, 2],
        )
        self.assertEqual(
            policy.quantize_signed_int4([-100.0, -1.125, 1.0, 100.0], scale),
            [-8, -8, 7, 7],
        )

    def test_cli_is_deterministic_and_fail_closed(self) -> None:
        command = [sys.executable, "tools/quantization_policy.py", "--check"]
        first = subprocess.run(command, cwd=ROOT, check=True, capture_output=True)
        second = subprocess.run(command, cwd=ROOT, check=True, capture_output=True)
        self.assertEqual(first.stdout, second.stdout)
        self.assertNotIn(str(ROOT).encode(), first.stdout)
        report = json.loads(first.stdout)
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(len(report["plans"]), 12)
        rejected = subprocess.run(
            [
                sys.executable,
                "tools/quantization_policy.py",
                "--model",
                "qwen2.5-0.5b",
                "--policy",
                "w8a8",
                "--preflight-current-rtl",
            ],
            cwd=ROOT,
            check=False,
            capture_output=True,
        )
        self.assertNotEqual(rejected.returncode, 0)
        self.assertIn(b"requires the W4A8 policy", rejected.stderr)


if __name__ == "__main__":
    unittest.main()
