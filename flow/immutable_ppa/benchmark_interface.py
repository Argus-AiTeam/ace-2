#!/usr/bin/env python3
"""Validate the strict ACE-2 benchmark interface without executing PPA tools."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import operator
import re
import subprocess
import sys
from pathlib import Path, PurePosixPath
from typing import Any, Sequence


BASE_COMMIT = "bc0ff4d89341646b948564cd59e2c67307bdea38"
TOP_MODULE = "ace2_shell"
EXPECTED_CHANNELS = [
    {
        "name": "csr_request",
        "source": "external",
        "valid": "csr_valid_i",
        "ready": "csr_ready_o",
        "payload": ["csr_write_i", "csr_addr_i", "csr_wdata_i", "csr_wstrb_i"],
    },
    {
        "name": "csr_response",
        "source": "ace2_shell",
        "valid": "csr_rvalid_o",
        "ready": "csr_rready_i",
        "payload": ["csr_rdata_o", "csr_error_o"],
    },
    {
        "name": "command",
        "source": "external",
        "valid": "cmd_valid_i",
        "ready": "cmd_ready_o",
        "payload": [
            "cmd_opcode_i",
            "cmd_flags_i",
            "cmd_layer_id_i",
            "cmd_m_i",
            "cmd_n_i",
            "cmd_k_i",
            "cmd_sequence_position_i",
            "cmd_completion_tag_i",
            "cmd_src0_addr_i",
            "cmd_src1_addr_i",
            "cmd_dst_addr_i",
            "cmd_scale_addr_i",
            "cmd_scratch_addr_i",
        ],
    },
    {
        "name": "memory_request",
        "source": "ace2_shell",
        "valid": "mem_req_valid_o",
        "ready": "mem_req_ready_i",
        "payload": [
            "mem_req_write_o",
            "mem_req_addr_o",
            "mem_req_len_o",
            "mem_req_tag_o",
        ],
    },
    {
        "name": "memory_write",
        "source": "ace2_shell",
        "valid": "mem_wvalid_o",
        "ready": "mem_wready_i",
        "payload": ["mem_wdata_o", "mem_wstrb_o", "mem_wtag_o"],
    },
    {
        "name": "memory_read",
        "source": "external",
        "valid": "mem_rvalid_i",
        "ready": "mem_rready_o",
        "payload": ["mem_rdata_i", "mem_rtag_i", "mem_rerror_i"],
    },
    {
        "name": "memory_write_response",
        "source": "external",
        "valid": "mem_bvalid_i",
        "ready": "mem_bready_o",
        "payload": ["mem_btag_i", "mem_berror_i"],
    },
    {
        "name": "completion",
        "source": "ace2_shell",
        "valid": "cmd_done_valid_o",
        "ready": "cmd_done_ready_i",
        "payload": [
            "cmd_done_tag_o",
            "cmd_done_error_o",
            "cmd_done_sumsq_o",
            "cmd_done_inv_rms_q30_o",
            "cmd_done_saturation_seen_o",
        ],
    },
]
EXPECTED_SRAM_PROTOCOL = {
    "request": (
        "each_bank_transfers_when_matching_sram_req_valid_o_and_"
        "sram_req_ready_i_bits_are_high_on_a_rising_edge"
    ),
    "read_response": (
        "each_sram_rvalid_i_bit_qualifies_the_matching_bank_slice_of_"
        "sram_rdata_i_on_a_rising_edge"
    ),
    "read_backpressure": False,
}
EXPECTED_ALIGNED_PORTS = [
    "cmd_src0_addr_i",
    "cmd_src1_addr_i",
    "cmd_dst_addr_i",
    "cmd_scale_addr_i",
]


class InterfaceError(RuntimeError):
    """The sidecar or the source contract is invalid."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise InterfaceError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_json_bytes(raw: bytes, label: str) -> dict[str, Any]:
    try:
        decoded = raw.decode("utf-8", errors="strict")
        value = json.loads(decoded, object_pairs_hook=_unique_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise InterfaceError(f"could not parse {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise InterfaceError(f"{label} root must be an object")
    return value


def validate_schema(data: dict[str, Any], schema_path: Path) -> None:
    schema = load_json_bytes(schema_path.read_bytes(), str(schema_path))
    try:
        from jsonschema import Draft202012Validator
        from jsonschema.exceptions import SchemaError
    except ImportError as exc:
        raise InterfaceError("validation requires the Python 'jsonschema' package") from exc
    try:
        Draft202012Validator.check_schema(schema)
    except SchemaError as exc:
        raise InterfaceError(f"invalid benchmark-interface schema: {exc.message}") from exc
    errors = sorted(
        Draft202012Validator(schema).iter_errors(data),
        key=lambda error: tuple(str(part) for part in error.absolute_path),
    )
    if errors:
        error = errors[0]
        location = "$" + "".join(
            f"[{part}]" if isinstance(part, int) else f".{part}"
            for part in error.absolute_path
        )
        raise InterfaceError(f"schema violation at {location}: {error.message}")


def _git(repo: Path, *args: str) -> bytes:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=repo,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except OSError as exc:
        raise InterfaceError(f"could not launch git: {exc}") from exc
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise InterfaceError(f"git {' '.join(args)} failed: {detail}") from None
    return result.stdout


def _repo_path(value: str, label: str) -> str:
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts or "\\" in value:
        raise InterfaceError(f"{label} is not a safe repository-relative POSIX path")
    return value


def _resolve_commit(repo: Path, ref: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", ref):
        raise InterfaceError("interface revision must be a full lowercase SHA-1 commit")
    commit = _git(repo, "rev-parse", "--verify", f"{ref}^{{commit}}").decode().strip()
    if commit != ref:
        raise InterfaceError(f"revision did not resolve exactly: expected={ref}, actual={commit}")
    return commit


def _read_git_file(
    repo: Path,
    commit: str,
    record: dict[str, str],
    label: str,
    *,
    check_identity: bool,
) -> bytes:
    path = _repo_path(record["path"], f"{label}.path")
    object_id = _git(repo, "rev-parse", f"{commit}:{path}").decode().strip()
    object_type = _git(repo, "cat-file", "-t", object_id).decode().strip()
    if object_type != "blob":
        raise InterfaceError(
            f"{label} resolved to Git object type {object_type!r}, not a blob"
        )
    content = _git(repo, "show", f"{commit}:{path}")
    if check_identity and object_id != record["git_blob"]:
        raise InterfaceError(
            f"{label} Git blob mismatch: expected={record['git_blob']}, actual={object_id}"
        )
    digest = sha256_bytes(content)
    if check_identity and digest != record["sha256"]:
        raise InterfaceError(
            f"{label} SHA-256 mismatch: expected={record['sha256']}, actual={digest}"
        )
    return content


_BINARY_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.floordiv,
    ast.FloorDiv: operator.floordiv,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _eval_expression(expression: str, values: dict[str, int], label: str) -> int:
    if not re.fullmatch(r"[A-Za-z0-9_+\-*/() \t]+", expression):
        raise InterfaceError(f"{label} contains unsupported syntax: {expression!r}")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise InterfaceError(f"{label} is malformed: {expression!r}") from exc

    def evaluate(node: ast.AST) -> int:
        if isinstance(node, ast.Expression):
            return evaluate(node.body)
        if isinstance(node, ast.Constant) and type(node.value) is int:
            return node.value
        if isinstance(node, ast.Name):
            if node.id not in values:
                raise InterfaceError(f"{label} references unknown identifier {node.id!r}")
            return values[node.id]
        if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPS:
            right = evaluate(node.right)
            if isinstance(node.op, (ast.Div, ast.FloorDiv)) and right == 0:
                raise InterfaceError(f"{label} divides by zero")
            return _BINARY_OPS[type(node.op)](evaluate(node.left), right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
            return _UNARY_OPS[type(node.op)](evaluate(node.operand))
        raise InterfaceError(f"{label} contains unsupported expression syntax")

    return evaluate(tree)


def _canonical_expression(value: str) -> str:
    return re.sub(r"\s+", "", value)


def _parse_package(source: bytes) -> tuple[dict[str, int], int]:
    try:
        text = source.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise InterfaceError(f"package source is not UTF-8: {exc}") from exc
    values: dict[str, int] = {}
    for match in re.finditer(
        r"\blocalparam\s+integer\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([^;]+);",
        text,
    ):
        name, expression = match.groups()
        values[name] = _eval_expression(expression, values, f"package constant {name}")
    opcode = re.search(
        r"\blocalparam\s+\[7:0\]\s+ACE2_OPCODE_FUSED_QKV\s*=\s*8'h([0-9a-fA-F]+)\s*;",
        text,
    )
    if opcode is None:
        raise InterfaceError("package does not define 8-bit ACE2_OPCODE_FUSED_QKV")
    return values, int(opcode.group(1), 16)


def _strip_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    return re.sub(r"//[^\n]*", "", text)


def _parse_module(
    source: bytes, module_name: str, package_values: dict[str, int]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    try:
        text = source.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise InterfaceError(f"module source is not UTF-8: {exc}") from exc
    clean = _strip_comments(text)
    match = re.search(
        rf"\bmodule\s+{re.escape(module_name)}\s*#\s*\((.*?)\)\s*\((.*?)\)\s*;",
        clean,
        flags=re.DOTALL,
    )
    if match is None:
        raise InterfaceError(f"could not parse ANSI header for module {module_name}")
    parameter_block, port_block = match.groups()

    parameters: list[dict[str, Any]] = []
    parameter_values = dict(package_values)
    for index, declaration in enumerate(parameter_block.split(",")):
        declaration = declaration.strip()
        item = re.fullmatch(
            r"parameter\s+(integer)(?:\s+(signed|unsigned))?\s+"
            r"([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.+)",
            declaration,
            flags=re.DOTALL,
        )
        if item is None:
            raise InterfaceError(f"could not parse parameter declaration {index}")
        data_type, explicit_sign, name, expression = item.groups()
        expression = _canonical_expression(expression)
        value = _eval_expression(expression, parameter_values, f"parameter {name}")
        parameter_values[name] = value
        parameters.append(
            {
                "name": name,
                "data_type": data_type,
                "signed": explicit_sign != "unsigned",
                "default_expression": expression,
                "resolved_default": value,
            }
        )

    ports: list[dict[str, Any]] = []
    for index, declaration in enumerate(port_block.split(",")):
        declaration = declaration.strip()
        item = re.fullmatch(
            r"(input|output|inout)\s+(wire|reg|logic)"
            r"(?:\s+(signed|unsigned))?\s*(\[[^\]]+\])?\s*"
            r"([A-Za-z_][A-Za-z0-9_]*)",
            declaration,
        )
        if item is None:
            raise InterfaceError(f"could not parse port declaration {index}")
        direction, net_type, explicit_sign, packed_range, name = item.groups()
        canonical_range = (
            _canonical_expression(packed_range) if packed_range is not None else None
        )
        width = _range_width(
            canonical_range, parameter_values, f"source port {name}.packed_range"
        )
        ports.append(
            {
                "name": name,
                "direction": direction,
                "net_type": net_type,
                "signed": explicit_sign == "signed",
                "packed_range": canonical_range,
                "width": width,
            }
        )
    return parameters, ports, text


def _range_width(
    packed_range: str | None, values: dict[str, int], label: str
) -> int:
    if packed_range is None:
        return 1
    match = re.fullmatch(r"\[(.+):(.+)\]", packed_range)
    if match is None:
        raise InterfaceError(f"{label} is malformed: {packed_range!r}")
    msb = _eval_expression(match.group(1), values, f"{label}.msb")
    lsb = _eval_expression(match.group(2), values, f"{label}.lsb")
    if msb < lsb:
        raise InterfaceError(f"{label} has descending bounds in the wrong order")
    return msb - lsb + 1


def _require_unique_names(items: list[dict[str, Any]], label: str) -> None:
    names = [item["name"] for item in items]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise InterfaceError(f"{label} contains duplicate names: {duplicates}")


def _validate_interface_lists(
    expected_parameters: list[dict[str, Any]],
    expected_ports: list[dict[str, Any]],
    actual_parameters: list[dict[str, Any]],
    actual_ports: list[dict[str, Any]],
) -> None:
    _require_unique_names(expected_parameters, "sidecar parameters")
    _require_unique_names(expected_ports, "sidecar ports")
    values = {
        parameter["name"]: parameter["resolved_default"]
        for parameter in expected_parameters
    }
    for port in expected_ports:
        width = _range_width(
            port["packed_range"], values, f"sidecar port {port['name']}.packed_range"
        )
        if width != port["width"]:
            raise InterfaceError(
                f"sidecar port {port['name']} width mismatch: "
                f"expression={width}, declared={port['width']}"
            )
    if expected_parameters != actual_parameters:
        raise InterfaceError("parameter declarations differ from the frozen source")
    if expected_ports != actual_ports:
        raise InterfaceError("port declarations differ from the frozen source")


def _strip_outer_parentheses(value: str) -> str:
    value = _canonical_expression(value)
    while value.startswith("(") and value.endswith(")"):
        depth = 0
        balanced = True
        for index, character in enumerate(value):
            if character == "(":
                depth += 1
            elif character == ")":
                depth -= 1
                if depth == 0 and index != len(value) - 1:
                    balanced = False
                    break
        if not balanced or depth != 0:
            break
        value = value[1:-1]
    return value


def _validate_fused_qkv(
    fused: dict[str, Any], package_opcode: int, module_text: str
) -> None:
    if fused["opcode"]["value"] != package_opcode:
        raise InterfaceError(
            "fused-QKV opcode differs from ACE2_OPCODE_FUSED_QKV in ace2_pkg"
        )
    descriptor = fused["descriptor"]
    if descriptor["aligned_address_ports"] != EXPECTED_ALIGNED_PORTS:
        raise InterfaceError("fused-QKV aligned address port order differs")
    block = re.search(
        r"\bwire\s+fused_qkv_descriptor_valid_w\s*=\s*(.*?);",
        _strip_comments(module_text),
        flags=re.DOTALL,
    )
    if block is None:
        raise InterfaceError("fused-QKV descriptor predicate is missing")
    observed = [
        _strip_outer_parentheses(condition)
        for condition in block.group(1).split("&&")
    ]
    expected = [
        "cmd_opcode_buf_q==ACE2_OPCODE_FUSED_QKV",
        f"cmd_flags_buf_q==8'd{descriptor['cmd_flags_i']}",
        f"cmd_layer_id_buf_q<8'd{descriptor['cmd_layer_id_i']['maximum'] + 1}",
        f"cmd_m_buf_q==16'd{descriptor['cmd_m_i']}",
        "cmd_n_buf_q==HIDDEN_SIZE_16",
        "cmd_k_buf_q==HIDDEN_SIZE_16",
        *(
            f"{port.replace('_i', '_buf_q')}[3:0]==4'd0"
            for port in descriptor["aligned_address_ports"]
        ),
        f"cmd_scratch_addr_buf_q==64'd{descriptor['cmd_scratch_addr_i']}",
    ]
    if observed != expected:
        raise InterfaceError(
            f"fused-QKV descriptor predicate differs: expected={expected}, observed={observed}"
        )
    canonical_module = _canonical_expression(module_text)
    projection_mapping = (
        "((cmd_opcode_i==ACE2_OPCODE_W4A8_PROJ)||"
        "(cmd_opcode_i==ACE2_OPCODE_FUSED_QKV))?OP_KIND_PROJ:"
    )
    if projection_mapping not in canonical_module:
        raise InterfaceError("fused-QKV opcode is not mapped to projection execution")


def _validate_protocols(data: dict[str, Any], ports: list[dict[str, Any]]) -> None:
    protocols = data["protocols"]
    if protocols["channels"] != EXPECTED_CHANNELS:
        raise InterfaceError("ready/valid channel semantics or ordering differ")
    if protocols["sram"] != EXPECTED_SRAM_PROTOCOL:
        raise InterfaceError("SRAM protocol semantics differ")
    port_map = {port["name"]: port for port in ports}
    for channel in protocols["channels"]:
        source_direction = "output" if channel["source"] == TOP_MODULE else "input"
        sink_direction = "input" if source_direction == "output" else "output"
        if port_map[channel["valid"]]["direction"] != source_direction:
            raise InterfaceError(f"{channel['name']} valid direction is inconsistent")
        if port_map[channel["ready"]]["direction"] != sink_direction:
            raise InterfaceError(f"{channel['name']} ready direction is inconsistent")
        for payload in channel["payload"]:
            if port_map[payload]["direction"] != source_direction:
                raise InterfaceError(f"{channel['name']} payload direction is inconsistent")


def _validate_sdc(data: dict[str, Any], content: bytes) -> None:
    constraints = data["constraints"]
    expected = (
        f"create_clock -name {constraints['clock_name']} "
        f"-period {constraints['period_ns']:.3f} "
        f"[get_ports {constraints['clock_port']}]\n"
        f"set_input_delay {constraints['input_delay_ns']:.3f} "
        f"-clock {constraints['clock_name']} "
        "[get_ports -filter {direction == input && name != clk_i} *]\n"
        f"set_output_delay {constraints['output_delay_ns']:.3f} "
        f"-clock {constraints['clock_name']} [all_outputs]\n"
        f"set_clock_uncertainty {constraints['clock_uncertainty_ns']:.3f} "
        f"[get_clocks {constraints['clock_name']}]\n"
    ).encode("utf-8")
    if content != expected:
        raise InterfaceError("SDC content differs from the frozen 10 ns binding")


def _validate_contract_sources(
    data: dict[str, Any], shell_source: bytes, package_source: bytes
) -> None:
    package_values, opcode = _parse_package(package_source)
    parameters, ports, module_text = _parse_module(
        shell_source, data["module"]["name"], package_values
    )
    _validate_interface_lists(
        data["module"]["parameters"],
        data["module"]["ports"],
        parameters,
        ports,
    )
    _validate_protocols(data, ports)
    _validate_fused_qkv(data["fused_qkv"], opcode, module_text)
    if "always @(posedge clk_i or negedge rst_ni)" not in module_text:
        raise InterfaceError("asynchronous active-low reset event control is missing")


def validate_data(
    data: dict[str, Any], repo: Path, schema_path: Path
) -> dict[str, Any]:
    repo = repo.resolve()
    validate_schema(data, schema_path)
    if data["base_commit"] != BASE_COMMIT:
        raise InterfaceError(f"base_commit must be exactly {BASE_COMMIT}")
    commit = _resolve_commit(repo, data["base_commit"])
    shell_source = _read_git_file(
        repo, commit, data["module"]["source"], "module source", check_identity=True
    )
    package_source = _read_git_file(
        repo,
        commit,
        data["module"]["package_source"],
        "package source",
        check_identity=True,
    )
    sdc = _read_git_file(
        repo,
        commit,
        data["constraints"]["source"],
        "constraint source",
        check_identity=True,
    )
    _validate_contract_sources(data, shell_source, package_source)
    _validate_sdc(data, sdc)
    return {
        "base_commit": commit,
        "module": data["module"]["name"],
        "parameter_count": len(data["module"]["parameters"]),
        "port_count": len(data["module"]["ports"]),
    }


def validate_revision(data: dict[str, Any], repo: Path, revision: str) -> None:
    repo = repo.resolve()
    commit = _resolve_commit(repo, revision)
    shell_source = _read_git_file(
        repo,
        commit,
        data["module"]["source"],
        "candidate module source",
        check_identity=False,
    )
    package_source = _read_git_file(
        repo,
        commit,
        data["module"]["package_source"],
        "candidate package source",
        check_identity=False,
    )
    _validate_contract_sources(data, shell_source, package_source)


def validate_file(
    sidecar_path: Path, repo: Path, schema_path: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        raw = sidecar_path.read_bytes()
    except OSError as exc:
        raise InterfaceError(f"could not read sidecar {sidecar_path}: {exc}") from exc
    data = load_json_bytes(raw, str(sidecar_path))
    result = validate_data(data, repo, schema_path)
    result["sidecar_sha256"] = sha256_bytes(raw)
    return data, result


def validate_checksum(sidecar_path: Path, checksum_path: Path) -> None:
    try:
        fields = checksum_path.read_text(encoding="utf-8").strip().split()
    except OSError as exc:
        raise InterfaceError(f"could not read checksum {checksum_path}: {exc}") from exc
    if len(fields) != 2 or fields[1] != "design/BENCHMARK_INTERFACE.json":
        raise InterfaceError("sidecar checksum file has an invalid record")
    if not re.fullmatch(r"[0-9a-f]{64}", fields[0]):
        raise InterfaceError("sidecar checksum is not a lowercase SHA-256 digest")
    if fields[0] != sha256_bytes(sidecar_path.read_bytes()):
        raise InterfaceError("sidecar checksum does not match")


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=".")
    parser.add_argument("--sidecar", default="design/BENCHMARK_INTERFACE.json")
    parser.add_argument(
        "--schema", default="design/benchmark_interface.schema.json"
    )
    parser.add_argument(
        "--checksum", default="design/BENCHMARK_INTERFACE.json.sha256"
    )
    parser.add_argument("--revision")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    repo = Path(args.repo).resolve()
    sidecar = (repo / args.sidecar).resolve()
    schema = (repo / args.schema).resolve()
    checksum = (repo / args.checksum).resolve()
    data, result = validate_file(sidecar, repo, schema)
    validate_checksum(sidecar, checksum)
    if args.revision is not None:
        validate_revision(data, repo, args.revision)
    print(
        "ACE2_BENCHMARK_INTERFACE_VALID "
        f"base={result['base_commit']} "
        f"module={result['module']} "
        f"parameters={result['parameter_count']} "
        f"ports={result['port_count']} "
        f"sidecar_sha256={result['sidecar_sha256']}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except InterfaceError as exc:
        print(f"benchmark-interface: ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
