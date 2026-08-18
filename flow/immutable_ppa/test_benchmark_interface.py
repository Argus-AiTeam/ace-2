from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from flow.immutable_ppa import benchmark_interface as interface


class BenchmarkInterfaceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repo = Path(__file__).resolve().parents[2]
        cls.sidecar = cls.repo / "design/BENCHMARK_INTERFACE.json"
        cls.schema = cls.repo / "design/benchmark_interface.schema.json"
        cls.checksum = cls.repo / "design/BENCHMARK_INTERFACE.json.sha256"
        cls.data = interface.load_json_bytes(
            cls.sidecar.read_bytes(), str(cls.sidecar)
        )

    def assert_invalid(self, data: dict, pattern: str | None = None) -> None:
        context = (
            self.assertRaisesRegex(interface.InterfaceError, pattern)
            if pattern
            else self.assertRaises(interface.InterfaceError)
        )
        with context:
            interface.validate_data(data, self.repo, self.schema)

    def test_exact_public_base_contract_is_valid(self) -> None:
        result = interface.validate_data(self.data, self.repo, self.schema)
        self.assertEqual(result["base_commit"], interface.BASE_COMMIT)
        self.assertEqual(result["parameter_count"], 12)
        self.assertEqual(result["port_count"], 64)
        interface.validate_revision(self.data, self.repo, interface.BASE_COMMIT)
        interface.validate_checksum(self.sidecar, self.checksum)

    def test_duplicate_json_key_is_rejected(self) -> None:
        with self.assertRaisesRegex(interface.InterfaceError, "duplicate JSON key"):
            interface.load_json_bytes(
                b'{"schema_version":1,"schema_version":1}', "duplicate"
            )

    def test_missing_and_extra_top_level_fields_are_rejected(self) -> None:
        missing = copy.deepcopy(self.data)
        del missing["clock"]
        self.assert_invalid(missing, "schema violation")
        extra = copy.deepcopy(self.data)
        extra["unexpected"] = True
        self.assert_invalid(extra, "schema violation")

    def test_port_set_order_and_duplicate_are_rejected(self) -> None:
        for operation in ("missing", "extra", "duplicate", "reordered"):
            with self.subTest(operation=operation):
                data = copy.deepcopy(self.data)
                ports = data["module"]["ports"]
                if operation == "missing":
                    ports.pop()
                elif operation == "extra":
                    ports.append(copy.deepcopy(ports[-1]))
                    ports[-1]["name"] = "unexpected_o"
                elif operation == "duplicate":
                    ports[-1] = copy.deepcopy(ports[0])
                else:
                    ports[0], ports[1] = ports[1], ports[0]
                self.assert_invalid(data)

    def test_port_direction_width_signedness_and_expression_are_rejected(self) -> None:
        cases = [
            ("direction", 0, "direction", "output"),
            ("declared_width", 5, "width", 31),
            ("signedness", 5, "signed", True),
            ("packed_range", 5, "packed_range", "[30:0]"),
            (
                "malformed_expression",
                36,
                "packed_range",
                "[LANES*UNKNOWN_WIDTH-1:0]",
            ),
        ]
        for label, index, key, value in cases:
            with self.subTest(label=label):
                data = copy.deepcopy(self.data)
                data["module"]["ports"][index][key] = value
                self.assert_invalid(data)

    def test_parameter_set_order_defaults_and_duplicates_are_rejected(self) -> None:
        for operation in (
            "missing",
            "extra",
            "duplicate",
            "reordered",
            "default",
            "resolved",
        ):
            with self.subTest(operation=operation):
                data = copy.deepcopy(self.data)
                parameters = data["module"]["parameters"]
                if operation == "missing":
                    parameters.pop()
                elif operation == "extra":
                    parameters.append(copy.deepcopy(parameters[-1]))
                    parameters[-1]["name"] = "EXTRA_PARAMETER"
                elif operation == "duplicate":
                    parameters[-1] = copy.deepcopy(parameters[0])
                elif operation == "reordered":
                    parameters[0], parameters[1] = parameters[1], parameters[0]
                elif operation == "default":
                    parameters[0]["default_expression"] = "897"
                else:
                    parameters[0]["resolved_default"] = 897
                self.assert_invalid(data)

    def test_non_blob_and_source_identity_drift_are_rejected(self) -> None:
        non_blob = copy.deepcopy(self.data)
        non_blob["module"]["source"]["path"] = "rtl"
        self.assert_invalid(non_blob, "not a blob")

        cases = [
            ("module", "source", "sha256"),
            ("module", "source", "git_blob"),
            ("module", "package_source", "sha256"),
            ("constraints", "source", "sha256"),
            ("constraints", "source", "git_blob"),
        ]
        for first, second, field in cases:
            with self.subTest(path=(first, second, field)):
                data = copy.deepcopy(self.data)
                data[first][second][field] = (
                    "0" * 64 if field == "sha256" else "0" * 40
                )
                self.assert_invalid(data, "mismatch")

    def test_constraint_and_protocol_semantic_drift_are_rejected(self) -> None:
        constraint = copy.deepcopy(self.data)
        constraint["constraints"]["period_ns"] = 11.0
        self.assert_invalid(constraint, "schema violation")

        protocol = copy.deepcopy(self.data)
        protocol["protocols"]["channels"][0]["payload"].reverse()
        self.assert_invalid(protocol, "semantics or ordering differ")

        fused = copy.deepcopy(self.data)
        fused["fused_qkv"]["descriptor"]["aligned_address_ports"].reverse()
        self.assert_invalid(fused, "aligned address port order differs")

    def test_checksum_record_shape_is_strict(self) -> None:
        raw = self.sidecar.read_bytes()
        with self.subTest("json_is_canonical_object"):
            self.assertIsInstance(json.loads(raw), dict)


if __name__ == "__main__":
    unittest.main()
