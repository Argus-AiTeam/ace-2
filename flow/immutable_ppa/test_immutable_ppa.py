from __future__ import annotations

import copy
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from flow.immutable_ppa import immutable_ppa as flow


class DockerParsingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.digest = (
            "openroad/orfs@sha256:"
            "3bc303869d5e4caac8f72c854f2b1614c726b2961bbb372f54bc8fbc0e725e71"
        )
        self.image_id = self.digest.split("@", 1)[1]

    def test_unformatted_inspect_array_is_parsed(self) -> None:
        raw = json.dumps([{"Id": self.image_id, "RepoDigests": [self.digest]}]).encode()
        self.assertEqual(
            flow.parse_docker_image_inspect(raw, self.digest),
            {"image_id": self.image_id, "repo_digest": self.digest},
        )

    def test_formatted_object_and_ambiguous_results_are_rejected(self) -> None:
        image = {"Id": self.image_id, "RepoDigests": [self.digest]}
        for value in (image, [], [image, image]):
            with self.subTest(value=value):
                with self.assertRaises(flow.PreflightError):
                    flow.parse_docker_image_inspect(json.dumps(value).encode(), self.digest)

    def test_missing_pinned_digest_is_rejected(self) -> None:
        raw = json.dumps([{"Id": self.image_id, "RepoDigests": []}]).encode()
        with self.assertRaises(flow.PreflightError):
            flow.parse_docker_image_inspect(raw, self.digest)


class ToolDiscoveryTest(unittest.TestCase):
    def test_discovery_sources_environment_without_invoking_tools(self) -> None:
        config = flow.load_config()
        inspect_json = json.dumps(
            [
                {
                    "Id": config["docker"]["expected_image_id"],
                    "RepoDigests": [config["docker"]["expected_repo_digest"]],
                }
            ]
        ).encode()
        paths = {
            "yosys": config["toolchain"]["yosys"],
            "opensta": config["toolchain"]["opensta"],
            "liberty": config["toolchain"]["liberty"],
            "sed": config["netlist_transform"]["executable"],
            "tee": config["toolchain"]["tee"],
        }
        discovery = (
            "OPENROAD: /OpenROAD-flow-scripts/tools/OpenROAD\n"
            + "".join(
                f"ACE2_IMMUTABLE_PPA_PATH\t{name}\t{path}\n"
                for name, path in paths.items()
            )
        ).encode()
        calls: list[list[str]] = []

        def runner(argv: list[str], cwd: Path | None) -> bytes:
            self.assertIsNone(cwd)
            calls.append(list(argv))
            return inspect_json if len(calls) == 1 else discovery

        metadata, found = flow.inspect_environment(config, runner)
        self.assertEqual(metadata["image_id"], config["docker"]["expected_image_id"])
        self.assertEqual(found, paths)
        self.assertEqual(calls[0][1:3], ["image", "inspect"])
        self.assertIn("--network", calls[1])
        self.assertIn("--read-only", calls[1])
        self.assertNotIn("--mount", calls[1])
        self.assertNotIn(config["toolchain"]["yosys"], calls[1][:-1])
        self.assertNotIn(config["toolchain"]["opensta"], calls[1][:-1])

    def test_empty_default_path_discovery_is_rejected(self) -> None:
        with self.assertRaises(flow.PreflightError):
            flow.parse_tool_discovery(b"")


class ConfigValidationTest(unittest.TestCase):
    def test_schema_fixed_config_values_are_rejected(self) -> None:
        cases = [
            (("flow_id",), "different-flow"),
            (("benchmark_interface", "sidecar"), "design/other.json"),
            (("namespace", "create_mode"), "0o755"),
            (("netlist_transform", "executable"), "/bin/sed"),
            (("target", "clock_period_ns"), 20.0),
            (("target", "frequency_floor_mhz"), 50.0),
            (("target", "non_sram_area_cap_mm2"), 3.0),
            (("target", "top_module"), "wrong_top"),
        ]
        for keys, value in cases:
            with self.subTest(keys=keys, value=value):
                config = copy.deepcopy(flow.load_config())
                destination = config
                for key in keys[:-1]:
                    destination = destination[key]
                destination[keys[-1]] = value
                with self.assertRaises(flow.PreflightError):
                    flow.validate_config(config)

    def test_package_base_flow_input_list_matches_frozen_scripts(self) -> None:
        config = flow.load_config()
        repo = flow.HERE.parents[1]
        base = config["package_base_commit"]

        def frozen(path: str) -> bytes:
            return subprocess.run(
                ["git", "show", f"{base}:{path}"],
                cwd=repo,
                check=True,
                stdout=subprocess.PIPE,
            ).stdout

        flow.validate_flow_input_closure(
            frozen(config["inputs"]["synthesis_script"]),
            frozen(config["inputs"]["sta_script"]),
            config,
            "package base",
        )

    def test_unlisted_script_input_is_rejected(self) -> None:
        config = copy.deepcopy(flow.load_config())
        synthesis = (
            "\n".join(
                [
                    *(f"read_verilog -sv {path}" for path in config["inputs"]["rtl"]),
                    "read_verilog -sv rtl/unlisted.sv",
                    f"write_verilog {config['netlist_transform']['input']}",
                ]
            )
            + "\n"
        ).encode()
        sta = (
            f"read_verilog {config['netlist_transform']['output']}\n"
            + "\n".join(
                f"read_sdc {path}" for path in config["inputs"]["constraints"]
            )
            + "\n"
        ).encode()
        with self.assertRaisesRegex(flow.PreflightError, "synthesis RTL closure differs"):
            flow.validate_flow_input_closure(synthesis, sta, config, "candidate")


class ManifestAndNamespaceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.repo = self.root / "repo"
        self.repo.mkdir()
        self._git("init", "-q")
        self._git("config", "user.email", "immutable-ppa@example.invalid")
        self._git("config", "user.name", "Immutable PPA Test")
        for path, text in {
            "rtl/top.sv": "module top; endmodule\n",
            "constraints/top.sdc": "create_clock -period 10 clk\n",
            "flow/synth.ys": (
                "read_verilog rtl/top.sv\n"
                "write_verilog build/sky130_rmsnorm/ace2_shell_mapped.v\n"
            ),
            "flow/sta.tcl": (
                "read_verilog build/sky130_rmsnorm/ace2_shell_mapped_sta.v\n"
                "read_sdc constraints/top.sdc\n"
            ),
        }.items():
            target = self.repo / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text, encoding="utf-8")
        self._git("add", ".")
        self._git("commit", "-q", "-m", "base")
        self.base = self._git("rev-parse", "HEAD").strip()
        (self.repo / "rtl/top.sv").write_text(
            "module top; wire changed; endmodule\n", encoding="utf-8"
        )
        self._git("add", "rtl/top.sv")
        self._git("commit", "-q", "-m", "candidate")
        self.candidate = self._git("rev-parse", "HEAD").strip()

        self.config = copy.deepcopy(flow.load_config())
        self.config["package_base_commit"] = self.base
        self.config["inputs"] = {
            "constraints": ["constraints/top.sdc"],
            "rtl": ["rtl/top.sv"],
            "sta_script": "flow/sta.tcl",
            "synthesis_script": "flow/synth.ys",
        }
        self.config["namespace"]["root"] = "evidence/immutable_ppa"
        self.metadata = {
            "image_id": self.config["docker"]["expected_image_id"],
            "repo_digest": self.config["docker"]["expected_repo_digest"],
        }
        self.discovered = {
            "yosys": self.config["toolchain"]["yosys"],
            "opensta": self.config["toolchain"]["opensta"],
            "liberty": self.config["toolchain"]["liberty"],
            "sed": self.config["netlist_transform"]["executable"],
            "tee": self.config["toolchain"]["tee"],
        }
        self.interface_bundle = (
            {"base_commit": self.base},
            {
                "base_commit": self.base,
                "module": "ace2_shell",
                "parameter_count": 12,
                "port_count": 64,
                "schema_path": "design/benchmark_interface.schema.json",
                "schema_sha256": "1" * 64,
                "sidecar_path": "design/BENCHMARK_INTERFACE.json",
                "sidecar_sha256": "2" * 64,
                "validator_path": "flow/immutable_ppa/benchmark_interface.py",
                "validator_sha256": "3" * 64,
            },
        )

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _git(self, *args: str) -> str:
        return subprocess.run(
            ["git", *args],
            cwd=self.repo,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
        ).stdout

    def test_manifest_binds_two_revisions_and_is_deterministic(self) -> None:
        namespace = self.repo / "evidence/immutable_ppa/comparison-unit"
        kwargs = {
            "repo": self.repo,
            "base_ref": self.base,
            "candidate_ref": self.candidate,
            "namespace": namespace,
            "config": self.config,
            "metadata": self.metadata,
            "discovered": self.discovered,
            "uid": 123,
            "gid": 456,
            "interface_bundle": self.interface_bundle,
        }
        with mock.patch.object(flow.interface_contract, "validate_revision"):
            first = flow.build_manifest(**kwargs)
            second = flow.build_manifest(**kwargs)
        self.assertEqual(first, second)
        self.assertEqual([item["role"] for item in first["revisions"]], ["base", "candidate"])
        self.assertNotEqual(
            first["revisions"][0]["rtl_sha256"],
            first["revisions"][1]["rtl_sha256"],
        )
        self.assertEqual(
            [step["name"] for step in first["commands"][0]["steps"]],
            ["synthesis", "netlist_transform", "sta"],
        )
        self.assertEqual(
            first["benchmark_interface"]["sidecar_sha256"],
            self.interface_bundle[1]["sidecar_sha256"],
        )
        self.assertTrue(
            first["benchmark_interface"]["candidate_compatibility_validated"]
        )
        self.assertFalse(namespace.exists())

    def test_candidate_constraint_drift_is_rejected_before_interface_check(self) -> None:
        (self.repo / "constraints/top.sdc").write_text(
            "create_clock -period 11 clk\n", encoding="utf-8"
        )
        self._git("add", "constraints/top.sdc")
        self._git("commit", "-q", "-m", "constraint drift")
        changed = self._git("rev-parse", "HEAD").strip()
        with mock.patch.object(
            flow.interface_contract, "validate_revision"
        ) as validate_revision:
            with self.assertRaisesRegex(flow.PreflightError, "constraints differ"):
                flow.build_manifest(
                    self.repo,
                    self.base,
                    changed,
                    self.repo / "evidence/immutable_ppa/comparison-constraint-drift",
                    self.config,
                    self.metadata,
                    self.discovered,
                    interface_bundle=self.interface_bundle,
                )
        validate_revision.assert_not_called()

    def test_candidate_interface_mismatch_is_rejected(self) -> None:
        with mock.patch.object(
            flow.interface_contract,
            "validate_revision",
            side_effect=flow.interface_contract.InterfaceError("port mismatch"),
        ):
            with self.assertRaisesRegex(
                flow.PreflightError, "candidate benchmark interface is incompatible"
            ):
                flow.build_manifest(
                    self.repo,
                    self.base,
                    self.candidate,
                    self.repo / "evidence/immutable_ppa/comparison-interface-drift",
                    self.config,
                    self.metadata,
                    self.discovered,
                    interface_bundle=self.interface_bundle,
                )

    def test_revision_input_must_be_a_git_blob(self) -> None:
        config = copy.deepcopy(self.config)
        config["inputs"]["rtl"] = ["rtl"]
        with self.assertRaisesRegex(flow.PreflightError, "not a blob"):
            flow.resolve_revision(self.repo, self.base, "base", config)

    def test_manifest_is_validated_before_it_can_be_returned(self) -> None:
        discovered = dict(self.discovered)
        discovered["yosys"] = "relative/yosys"
        with self.assertRaisesRegex(
            flow.PreflightError, "manifest violates bundled schema"
        ):
            with mock.patch.object(flow.interface_contract, "validate_revision"):
                flow.build_manifest(
                    self.repo,
                    self.base,
                    self.candidate,
                    self.repo / "evidence/immutable_ppa/comparison-invalid",
                    self.config,
                    self.metadata,
                    discovered,
                    interface_bundle=self.interface_bundle,
                )

    def test_existing_namespace_and_manifest_overwrite_are_refused(self) -> None:
        namespace_root = self.repo / "evidence/immutable_ppa"
        namespace_root.mkdir(parents=True)
        namespace = namespace_root / "comparison-exclusive"
        digest = "a" * 64
        flow.reserve_namespace(
            namespace_root,
            namespace,
            self.config["namespace"]["leaf_pattern"],
            digest,
        )
        with self.assertRaises(flow.PreflightError):
            flow.reserve_namespace(
                namespace_root,
                namespace,
                self.config["namespace"]["leaf_pattern"],
                digest,
            )
        output = self.root / "manifest.json"
        flow.write_json_exclusive(output, {"first": True})
        with self.assertRaises(flow.PreflightError):
            flow.write_json_exclusive(output, {"second": True})
        self.assertEqual(json.loads(output.read_text()), {"first": True})

    def test_namespace_creation_requires_explicit_acknowledgement(self) -> None:
        with self.assertRaises(flow.PreflightError):
            flow.main(
                [
                    "reserve-namespace",
                    "--namespace-root",
                    str(self.root),
                    "--namespace",
                    str(self.root / "comparison-no-ack"),
                    "--manifest-sha256",
                    "b" * 64,
                ]
            )


if __name__ == "__main__":
    unittest.main()
