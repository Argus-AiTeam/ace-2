#!/usr/bin/env python3
"""Focused regression tests for quality-gate evidence precedence."""

from __future__ import annotations

import hashlib
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import run_quality_gate
import run_official_quality
import torch
from ace2_full_model_fixed_point import absolute_percentile
from ace2_quality_contracts import validate_oracle_manifest, validate_rtl_binding
from run_quality_gate import enforce_quality_blocker_precedence


def aggregate_source_hash(source_hashes: list[dict[str, str]]) -> str:
    digest = hashlib.sha256()
    for source in sorted(source_hashes, key=lambda entry: entry["path"]):
        digest.update(source["path"].encode())
        digest.update(b"\0")
        digest.update(source["sha256"].encode())
        digest.update(b"\n")
    return digest.hexdigest()


class Bfloat16PercentileTest(unittest.TestCase):
    def test_histogram_percentile_matches_torch_linear_interpolation(self) -> None:
        values = torch.tensor(
            [-7.5, -3.0, -1.5, 0.0, 0.5, 2.0, 4.0, 8.0],
            dtype=torch.bfloat16,
        )
        for quantile in (0.125, 0.5, 0.875, 0.999):
            expected = float(
                torch.quantile(values.abs().to(torch.float64), quantile)
            )
            self.assertTrue(
                math.isclose(
                    absolute_percentile(values, quantile),
                    expected,
                    rel_tol=0.0,
                    abs_tol=0.0,
                )
            )

    def test_histogram_percentile_rejects_nonfinite_values(self) -> None:
        values = torch.tensor([1.0, float("inf")], dtype=torch.bfloat16)
        with self.assertRaisesRegex(ValueError, "must be finite"):
            absolute_percentile(values, 0.999)


class QualityBlockerPrecedenceTest(unittest.TestCase):
    def test_active_blocker_rejects_official_evidence(self) -> None:
        official_result = {"gate_passed": True, "mode": "official"}
        official_status = {"pointer": {"path": "sealed-run"}, "valid": True}
        blocker = {"classification": "independently_reproduced_numerical_contract_failure"}

        result, status = enforce_quality_blocker_precedence(
            official_result,
            official_status,
            blocker,
        )

        self.assertIsNone(result)
        self.assertFalse(status["valid"])
        self.assertEqual(
            status["rejected_by_active_blocker"],
            blocker["classification"],
        )
        self.assertTrue(official_status["valid"])

    def test_unblocked_official_evidence_is_preserved(self) -> None:
        official_result = {"gate_passed": False, "mode": "official"}
        official_status = {"valid": True}

        result, status = enforce_quality_blocker_precedence(
            official_result,
            official_status,
            None,
        )

        self.assertIs(result, official_result)
        self.assertIs(status, official_status)


class OfficialRunManifestTest(unittest.TestCase):
    def test_manifest_freezes_public_inputs_numerics_and_provenance(self) -> None:
        accepted_rtl = {
            "binding": {"path": "benchmark/quality/RTL_BINDING.json", "sha256": "a" * 64},
            "candidate_rtl_hash": "b" * 64,
            "manifest": {"path": "evidence/candidate/RESULTS.json", "sha256": "c" * 64},
            "numerical_rtl": [],
            "valid": True,
        }
        sources = [
            {
                "bytes": 1,
                "path": "benchmark/quality/PROMPT_MANIFEST.json",
                "sha256": "d" * 64,
            }
        ]
        oracle_contract = {
            "manifest": {
                "bytes": 1,
                "path": "reference/ORACLE_MANIFEST.json",
                "sha256": "e" * 64,
            },
            "entry_count": 1,
            "artifacts": [],
            "valid": True,
        }
        command = [
            "./.venv/bin/python",
            "tools/ace2_full_model_fixed_point.py",
            "--mode",
            "official",
            "--output-dir",
            "benchmark/raw/quality/test/artifacts",
        ]

        manifest = run_official_quality.build_run_manifest(
            accepted_rtl=accepted_rtl,
            command=command,
            generated_at_utc="2026-07-31T00:00:00Z",
            runtime_packages={
                "datasets": "4.8.5",
                "lm_eval": "0.4.9.2",
                "torch": "2.11.0",
                "transformers": "4.57.6",
            },
            sources=sources,
            oracle_contract=oracle_contract,
        )

        self.assertEqual(manifest["command"], command)
        self.assertEqual(manifest["accepted_rtl"], accepted_rtl)
        self.assertEqual(manifest["source_artifacts"], sources)
        self.assertEqual(manifest["independent_reference_oracles"], oracle_contract)
        self.assertIn("datasets", manifest["public_inputs"])
        self.assertIn("lm_eval", manifest["public_inputs"])
        self.assertIn("arithmetic", manifest["numerical_contract"])
        self.assertIn("full_model_scope", manifest["numerical_contract"])
        self.assertIn("python_seed", manifest["determinism"])
        self.assertIn("wikitext2_perplexity_ratio_max", manifest["acceptance_thresholds"])
        self.assertNotIn(str(ROOT), json.dumps(manifest["command"]))


class OracleManifestValidationTest(unittest.TestCase):
    def test_oracle_artifacts_are_hash_validated(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "build") as temporary:
            temporary_root = Path(temporary)
            manifest_dir = temporary_root / "reference"
            artifact_dir = temporary_root / "tools"
            manifest_dir.mkdir(parents=True)
            artifact_dir.mkdir(parents=True)
            artifact_path = artifact_dir / "oracle.py"
            artifact_path.write_text("ORACLE = 1\n", encoding="utf-8")
            artifact_sha256 = run_quality_gate.sha256_file(artifact_path)
            oracle = {
                "case_count": 1,
                "numeric_acceptance": "bit_exact_fixed_point_vector_match",
            }
            for field in ("generator", "reference", "vector_json", "vector_svh"):
                oracle[field] = "tools/oracle.py"
                oracle[f"{field}_sha256"] = artifact_sha256
            (manifest_dir / "ORACLE_MANIFEST.json").write_text(
                json.dumps(
                    {"schema_version": 1, "oracles": [oracle]},
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )

            validated = validate_oracle_manifest(temporary_root)
            self.assertTrue(validated["valid"])
            self.assertEqual(validated["entry_count"], 1)
            self.assertEqual(len(validated["artifacts"]), 1)

            artifact_path.write_text("ORACLE = 2\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "oracle hash differs"):
                validate_oracle_manifest(temporary_root)


class AcceptedRtlBindingTest(unittest.TestCase):
    def _write_accepted_binding(
        self,
        root: Path,
        *,
        frequency_mhz: float = 100.0,
        wns_ns: float = 0.0,
    ) -> None:
        binding_dir = root / "benchmark" / "quality"
        manifest_dir = root / "evidence" / "candidate"
        rtl_dir = root / "rtl"
        constraint_dir = root / "constraints"
        flow_dir = root / "flow" / "yosys"
        binding_dir.mkdir(parents=True)
        manifest_dir.mkdir(parents=True)
        rtl_dir.mkdir(parents=True)
        constraint_dir.mkdir(parents=True)
        flow_dir.mkdir(parents=True)

        rtl_path = rtl_dir / "candidate.sv"
        constraint_path = constraint_dir / "candidate.sdc"
        synth_script = flow_dir / "candidate.ys"
        sta_script = flow_dir / "candidate_sta.tcl"
        yosys_log = manifest_dir / "sky130_yosys.log"
        sta_log = manifest_dir / "sky130_sta.log"
        rtl_path.write_text("module candidate; endmodule\n", encoding="utf-8")
        constraint_path.write_text("create_clock -period 10 clk\n", encoding="utf-8")
        synth_script.write_text("read_verilog rtl/candidate.sv\n", encoding="utf-8")
        sta_script.write_text("read_sdc constraints/candidate.sdc\n", encoding="utf-8")
        yosys_log.write_text("Chip area 1000\n", encoding="utf-8")
        sta_log.write_text(f"wns {wns_ns}\n", encoding="utf-8")
        rtl_sha256 = run_quality_gate.sha256_file(rtl_path)
        constraint_sources = [
            {
                "path": "constraints/candidate.sdc",
                "sha256": run_quality_gate.sha256_file(constraint_path),
            },
            {
                "path": "flow/yosys/candidate.ys",
                "sha256": run_quality_gate.sha256_file(synth_script),
            },
            {
                "path": "flow/yosys/candidate_sta.tcl",
                "sha256": run_quality_gate.sha256_file(sta_script),
            },
        ]
        constraint_hash = aggregate_source_hash(constraint_sources)
        candidate_hash = "c" * 64
        manifest = {
            "candidate_rtl_hash": candidate_hash,
            "rtl_hash": candidate_hash,
            "candidate_status": "published_independent_l2_accepted",
            "candidate_review_binding": {"decision": "accepted", "level": "L2"},
            "candidate_source_hashes": [
                {"path": "rtl/candidate.sv", "sha256": rtl_sha256},
                *constraint_sources,
            ],
            "constraint_hash": constraint_hash,
            "ppa_evidence_binding": {
                "constraint_hash": constraint_hash,
                "rtl_hash": candidate_hash,
            },
            "evidence": {
                "sky130_synthesis_log": "evidence/candidate/sky130_yosys.log",
                "sky130_sta_log": "evidence/candidate/sky130_sta.log",
                "sky130_yosys": {
                    "log_sha256": run_quality_gate.sha256_file(yosys_log),
                    "non_sram_area_mm2": 0.5,
                },
                "sky130_sta": {
                    "estimated_fmax_mhz_floor_bound": frequency_mhz,
                    "log_sha256": run_quality_gate.sha256_file(sta_log),
                    "wns_ns": wns_ns,
                },
            },
        }
        manifest_path = manifest_dir / "RESULTS.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        binding = {
            "schema_version": 1,
            "status": "published_independent_l2_accepted",
            "candidate_rtl_hash": candidate_hash,
            "manifest": {
                "path": "evidence/candidate/RESULTS.json",
                "sha256": run_quality_gate.sha256_file(manifest_path),
            },
            "numerical_rtl": [
                {"path": "rtl/candidate.sv", "sha256": rtl_sha256}
            ],
        }
        (binding_dir / "RTL_BINDING.json").write_text(
            json.dumps(binding, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def test_independent_l2_binding_with_bound_passing_ppa_validates(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "build") as temporary:
            temporary_root = Path(temporary)
            self._write_accepted_binding(temporary_root)
            validated = validate_rtl_binding(temporary_root)

        self.assertTrue(validated["valid"])
        self.assertEqual(validated["ppa_contract"]["non_sram_area_mm2"], 0.5)
        self.assertEqual(validated["ppa_contract"]["reported_frequency_mhz"], 100.0)
        self.assertEqual(
            validated["ppa_contract"]["constraint_hash"],
            aggregate_source_hash(
                [
                    {
                        "path": artifact["path"],
                        "sha256": artifact["sha256"],
                    }
                    for artifact in validated["constraints"]
                ]
            ),
        )
        self.assertEqual(len(validated["constraints"]), 3)

    def test_accepted_label_cannot_bypass_timing_floor(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "build") as temporary:
            temporary_root = Path(temporary)
            self._write_accepted_binding(
                temporary_root,
                frequency_mhz=64.39,
                wns_ns=-5.53,
            )
            with self.assertRaisesRegex(ValueError, "misses the 100 MHz timing floor"):
                validate_rtl_binding(temporary_root)

    def test_ppa_threshold_miss_is_not_accepted_rtl(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "build") as temporary:
            temporary_root = Path(temporary)
            binding_dir = temporary_root / "benchmark" / "quality"
            manifest_dir = temporary_root / "evidence" / "candidate"
            rtl_dir = temporary_root / "rtl"
            binding_dir.mkdir(parents=True)
            manifest_dir.mkdir(parents=True)
            rtl_dir.mkdir(parents=True)

            rtl_path = rtl_dir / "candidate.sv"
            rtl_path.write_text("module candidate; endmodule\n", encoding="utf-8")
            rtl_sha256 = run_quality_gate.sha256_file(rtl_path)
            manifest = {
                "candidate_rtl_hash": "candidate-hash",
                "rtl_hash": "candidate-hash",
                "candidate_source_hashes": [
                    {"path": "rtl/candidate.sv", "sha256": rtl_sha256}
                ],
            }
            manifest_path = manifest_dir / "RESULTS.json"
            manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            binding = {
                "schema_version": 1,
                "status": "rmsnorm_downstream_scale_contract_ppa_threshold_miss",
                "candidate_rtl_hash": "candidate-hash",
                "manifest": {
                    "path": "evidence/candidate/RESULTS.json",
                    "sha256": run_quality_gate.sha256_file(manifest_path),
                },
                "numerical_rtl": [
                    {"path": "rtl/candidate.sv", "sha256": rtl_sha256}
                ],
            }
            (binding_dir / "RTL_BINDING.json").write_text(
                json.dumps(binding, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                ValueError,
                "RTL binding is not independently accepted",
            ):
                validate_rtl_binding(temporary_root)

    def test_unaccepted_binding_also_reports_live_source_drift(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "build") as temporary:
            temporary_root = Path(temporary)
            binding_dir = temporary_root / "benchmark" / "quality"
            manifest_dir = temporary_root / "evidence" / "candidate"
            rtl_dir = temporary_root / "rtl"
            binding_dir.mkdir(parents=True)
            manifest_dir.mkdir(parents=True)
            rtl_dir.mkdir(parents=True)

            rtl_path = rtl_dir / "candidate.sv"
            rtl_path.write_text("module candidate; endmodule\n", encoding="utf-8")
            bound_sha256 = run_quality_gate.sha256_file(rtl_path)
            manifest = {
                "candidate_rtl_hash": "candidate-hash",
                "rtl_hash": "candidate-hash",
                "candidate_source_hashes": [
                    {"path": "rtl/candidate.sv", "sha256": bound_sha256}
                ],
            }
            manifest_path = manifest_dir / "RESULTS.json"
            manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            binding = {
                "schema_version": 1,
                "status": "ppa_threshold_miss",
                "candidate_rtl_hash": "candidate-hash",
                "manifest": {
                    "path": "evidence/candidate/RESULTS.json",
                    "sha256": run_quality_gate.sha256_file(manifest_path),
                },
                "numerical_rtl": [
                    {"path": "rtl/candidate.sv", "sha256": bound_sha256}
                ],
            }
            (binding_dir / "RTL_BINDING.json").write_text(
                json.dumps(binding, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            rtl_path.write_text("module candidate; wire drift; endmodule\n", encoding="utf-8")

            with self.assertRaises(ValueError) as raised:
                validate_rtl_binding(temporary_root)

        error = str(raised.exception)
        self.assertIn("RTL binding is not independently accepted", error)
        self.assertIn("numerical RTL hash differs for rtl/candidate.sv", error)


class QualityGatePreflightPrecedenceTest(unittest.TestCase):
    def test_failed_rtl_binding_is_retained_as_a_blocking_condition(self) -> None:
        rtl_error = (
            "RTL binding is not independently accepted: "
            "status='ppa_threshold_miss'"
        )
        official_status = {
            "pointer": {"path": "benchmark/raw/latest/quality_official_pointer.json"},
            "valid": False,
        }

        with tempfile.TemporaryDirectory(dir=ROOT / "build") as temporary:
            temporary_path = Path(temporary)
            results_path = temporary_path / "quality_results.json"
            preflight_path = temporary_path / "quality_gate_preflight.json"
            with (
                patch.object(run_quality_gate, "RESULTS", results_path),
                patch.object(run_quality_gate, "PREFLIGHT", preflight_path),
                patch.object(run_quality_gate, "PROJECT_VENV", Path(sys.prefix)),
                patch.object(
                    run_quality_gate,
                    "package_version",
                    side_effect=lambda distribution: (
                        run_quality_gate.EXPECTED_SOFTWARE["lm_eval"]
                        if distribution == "lm-eval"
                        else run_quality_gate.EXPECTED_SOFTWARE[distribution]
                    ),
                ),
                patch.object(
                    run_quality_gate.importlib.util,
                    "find_spec",
                    return_value=object(),
                ),
                patch.object(
                    run_quality_gate,
                    "run_fixed_point_self_test",
                    return_value={"passed": True},
                ),
                patch.object(
                    run_quality_gate,
                    "validate_oracle_manifest",
                    return_value={"valid": True},
                ),
                patch.object(
                    run_quality_gate,
                    "validate_rtl_binding",
                    side_effect=ValueError(rtl_error),
                ),
                patch.object(
                    run_quality_gate,
                    "load_quality_blocker",
                    return_value=(None, {"valid": True}),
                ),
                patch.object(
                    run_quality_gate,
                    "load_official_evidence",
                    return_value=(None, official_status),
                ),
            ):
                result = run_quality_gate.run_gate()

            written_result = json.loads(results_path.read_text(encoding="utf-8"))

        self.assertEqual(result["status"], "blocked_precondition_failure")
        self.assertEqual(
            result["blocking_conditions"],
            ["accepted_rtl_binding_validation"],
        )
        self.assertEqual(
            result["prerequisites"]["accepted_rtl_validation"]["error"],
            rtl_error,
        )
        self.assertEqual(written_result, result)

    def test_invalid_blocker_validation_rejects_valid_official_evidence(self) -> None:
        official_result = {
            "classification": "pass",
            "gate_passed": True,
            "mode": "official",
        }
        official_status = {
            "result": {"path": "benchmark/raw/quality/sealed/artifacts/results.json"},
            "valid": True,
        }
        invalid_blocker_status = {
            "binding": {"path": "benchmark/quality/QUALITY_BLOCKER.json"},
            "error": "quality blocker source hash differs",
            "valid": False,
        }

        with tempfile.TemporaryDirectory(dir=ROOT / "build") as temporary:
            temporary_path = Path(temporary)
            results_path = temporary_path / "quality_results.json"
            preflight_path = temporary_path / "quality_gate_preflight.json"
            with (
                patch.object(run_quality_gate, "RESULTS", results_path),
                patch.object(run_quality_gate, "PREFLIGHT", preflight_path),
                patch.object(run_quality_gate, "PROJECT_VENV", Path(sys.prefix)),
                patch.object(
                    run_quality_gate,
                    "package_version",
                    side_effect=lambda distribution: (
                        run_quality_gate.EXPECTED_SOFTWARE["lm_eval"]
                        if distribution == "lm-eval"
                        else run_quality_gate.EXPECTED_SOFTWARE[distribution]
                    ),
                ),
                patch.object(
                    run_quality_gate.importlib.util,
                    "find_spec",
                    return_value=object(),
                ),
                patch.object(
                    run_quality_gate,
                    "run_fixed_point_self_test",
                    return_value={"passed": True},
                ),
                patch.object(
                    run_quality_gate,
                    "validate_rtl_binding",
                    return_value={"candidate_rtl_hash": "accepted", "valid": True},
                ),
                patch.object(
                    run_quality_gate,
                    "load_quality_blocker",
                    return_value=(None, invalid_blocker_status),
                ),
                patch.object(
                    run_quality_gate,
                    "load_official_evidence",
                    return_value=(official_result, official_status),
                ),
            ):
                result = run_quality_gate.run_gate()

            preflight = json.loads(preflight_path.read_text(encoding="utf-8"))

        self.assertEqual(result["status"], "blocked_precondition_failure")
        self.assertFalse(result["gate_passed"])
        self.assertFalse(preflight["measurement_ready"])
        self.assertFalse(preflight["measurement_started"])
        self.assertFalse(preflight["official_evidence"]["valid"])
        self.assertIn(
            "quality_blocker_validation",
            preflight["preflight"]["missing_artifacts"],
        )
        self.assertEqual(
            preflight["official_evidence"]["rejected_by_precondition_failures"],
            ["quality_blocker_validation"],
        )


if __name__ == "__main__":
    unittest.main()
