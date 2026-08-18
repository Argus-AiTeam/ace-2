#!/usr/bin/env python3
"""Fail-closed, non-executing preparation for immutable ACE-2 PPA runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Sequence

try:
    from . import benchmark_interface as interface_contract
except ImportError:
    import benchmark_interface as interface_contract


SELF = Path(__file__).resolve()
HERE = SELF.parent
DEFAULT_CONFIG = HERE / "config.json"
DEFAULT_SCHEMA = HERE / "manifest.schema.json"
FIXED_FLOW_ID = "ace2-sky130hd-yosys-opensta-immutable-v1"
FIXED_PACKAGE_BASE_COMMIT = "bc0ff4d89341646b948564cd59e2c67307bdea38"
FIXED_NAMESPACE_CREATE_MODE = "0o750"
FIXED_BENCHMARK_INTERFACE = {
    "schema": "design/benchmark_interface.schema.json",
    "sidecar": "design/BENCHMARK_INTERFACE.json",
    "validator": "flow/immutable_ppa/benchmark_interface.py",
}
FIXED_TARGET = {
    "clock_period_ns": 10.0,
    "frequency_floor_mhz": 100.0,
    "non_sram_area_cap_mm2": 2.0,
    "top_module": "ace2_shell",
}
Run = Callable[[Sequence[str], Path | None], bytes]


class PreflightError(RuntimeError):
    """The immutable flow contract could not be established."""


def canonical_json(data: Any) -> bytes:
    return (json.dumps(data, indent=2, sort_keys=True) + "\n").encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def _run(argv: Sequence[str], cwd: Path | None = None) -> bytes:
    try:
        result = subprocess.run(
            list(argv),
            cwd=cwd,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except OSError as exc:
        raise PreflightError(f"could not launch {argv[0]}: {exc}") from exc
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace").strip()
        raise PreflightError(
            f"command failed ({result.returncode}): {shlex.join(argv)}"
            + (f": {stderr}" if stderr else "")
        )
    return result.stdout


def _require_exact_keys(data: dict[str, Any], expected: set[str], label: str) -> None:
    actual = set(data)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise PreflightError(f"{label} keys differ: missing={missing}, extra={extra}")


def _validate_repo_path(value: str, label: str) -> None:
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or ".." in path.parts
        or "\\" in value
        or ":" in value
    ):
        raise PreflightError(f"{label} must be a safe repository-relative POSIX path")


def validate_config(config: dict[str, Any]) -> None:
    _require_exact_keys(
        config,
        {
            "attempt_policy",
            "benchmark_interface",
            "docker",
            "flow_id",
            "inputs",
            "namespace",
            "netlist_transform",
            "package_base_commit",
            "schema_version",
            "target",
            "toolchain",
        },
        "config",
    )
    if config["schema_version"] != 1:
        raise PreflightError("unsupported config schema_version")
    if config["flow_id"] != FIXED_FLOW_ID:
        raise PreflightError(f"flow_id must be {FIXED_FLOW_ID!r}")
    if config["package_base_commit"] != FIXED_PACKAGE_BASE_COMMIT:
        raise PreflightError(
            f"package_base_commit must be exactly {FIXED_PACKAGE_BASE_COMMIT}"
        )
    if config["target"] != FIXED_TARGET:
        raise PreflightError(f"target must be exactly {FIXED_TARGET}")
    if config["benchmark_interface"] != FIXED_BENCHMARK_INTERFACE:
        raise PreflightError(
            f"benchmark_interface must be exactly {FIXED_BENCHMARK_INTERFACE}"
        )
    for name, value in config["benchmark_interface"].items():
        _validate_repo_path(value, f"benchmark_interface.{name}")

    docker = config["docker"]
    _require_exact_keys(
        docker,
        {"executable", "expected_image_id", "expected_repo_digest", "image"},
        "docker config",
    )
    digest = r"sha256:[0-9a-f]{64}"
    if not re.fullmatch(digest, docker["expected_image_id"]):
        raise PreflightError("expected_image_id is not a SHA-256 image ID")
    if not re.fullmatch(r"[^@\s]+@" + digest, docker["image"]):
        raise PreflightError("Docker image must be pinned by digest")
    if docker["image"] != docker["expected_repo_digest"]:
        raise PreflightError("Docker image and expected_repo_digest must match exactly")
    if not Path(docker["executable"]).is_absolute():
        raise PreflightError("Docker executable must be absolute")

    inputs = config["inputs"]
    _require_exact_keys(
        inputs, {"constraints", "rtl", "sta_script", "synthesis_script"}, "inputs"
    )
    for category in ("rtl", "constraints"):
        values = inputs[category]
        if not isinstance(values, list) or not values or len(values) != len(set(values)):
            raise PreflightError(f"{category} must be a non-empty unique list")
        for value in values:
            _validate_repo_path(value, category)
    _validate_repo_path(inputs["synthesis_script"], "synthesis_script")
    _validate_repo_path(inputs["sta_script"], "sta_script")

    namespace = config["namespace"]
    _require_exact_keys(
        namespace, {"create_mode", "leaf_pattern", "root"}, "namespace config"
    )
    if namespace["create_mode"] != FIXED_NAMESPACE_CREATE_MODE:
        raise PreflightError(
            f"namespace.create_mode must be {FIXED_NAMESPACE_CREATE_MODE!r}"
        )
    _validate_repo_path(namespace["root"], "namespace root")
    try:
        re.compile(namespace["leaf_pattern"])
    except (TypeError, re.error) as exc:
        raise PreflightError(f"namespace.leaf_pattern is invalid: {exc}") from exc

    toolchain = config["toolchain"]
    _require_exact_keys(
        toolchain, {"environment_script", "liberty", "opensta", "tee", "yosys"}, "toolchain"
    )
    for name, value in toolchain.items():
        if not isinstance(value, str) or not PurePosixPath(value).is_absolute():
            raise PreflightError(f"toolchain.{name} must be an absolute container path")

    transform = config["netlist_transform"]
    _require_exact_keys(
        transform, {"executable", "expression", "input", "kind", "output"}, "netlist_transform"
    )
    if transform["kind"] != "gnu_sed_ere":
        raise PreflightError("unsupported netlist transform")
    if transform["executable"] != "/usr/bin/sed":
        raise PreflightError("netlist transform executable must be '/usr/bin/sed'")
    _validate_repo_path(transform["input"], "netlist transform input")
    _validate_repo_path(transform["output"], "netlist transform output")

    policy = config["attempt_policy"]
    if policy != {
        "max_official_attempts": 1,
        "overwrite_allowed": False,
        "retry_allowed": False,
    }:
        raise PreflightError("attempt policy must be exactly-once and fail closed")


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PreflightError(f"could not load config {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise PreflightError("config root must be an object")
    validate_config(data)
    return data


def load_interface_contract(
    repo: Path, config: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    repo = repo.resolve()
    binding = config["benchmark_interface"]
    sidecar_path = repo / binding["sidecar"]
    schema_path = repo / binding["schema"]
    validator_path = repo / binding["validator"]
    if validator_path.resolve() != Path(interface_contract.__file__).resolve():
        raise PreflightError("configured benchmark-interface validator path differs")
    try:
        data, result = interface_contract.validate_file(
            sidecar_path, repo, schema_path
        )
    except interface_contract.InterfaceError as exc:
        raise PreflightError(f"benchmark interface is invalid: {exc}") from exc
    result.update(
        {
            "schema_path": binding["schema"],
            "schema_sha256": sha256_file(schema_path),
            "sidecar_path": binding["sidecar"],
            "validator_path": binding["validator"],
            "validator_sha256": sha256_file(validator_path),
        }
    )
    return data, result


def validate_manifest(
    manifest: dict[str, Any], schema_path: Path = DEFAULT_SCHEMA
) -> None:
    try:
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PreflightError(f"could not load manifest schema {schema_path}: {exc}") from exc
    if not isinstance(schema, dict):
        raise PreflightError("manifest schema root must be an object")
    try:
        from jsonschema import Draft202012Validator
        from jsonschema.exceptions import SchemaError
    except ImportError as exc:
        raise PreflightError(
            "manifest validation requires the Python 'jsonschema' package"
        ) from exc
    try:
        Draft202012Validator.check_schema(schema)
    except SchemaError as exc:
        raise PreflightError(f"bundled manifest schema is invalid: {exc.message}") from exc
    errors = sorted(
        Draft202012Validator(schema).iter_errors(manifest),
        key=lambda error: tuple(str(part) for part in error.absolute_path),
    )
    if errors:
        error = errors[0]
        location = "$" + "".join(
            f"[{part}]" if isinstance(part, int) else f".{part}"
            for part in error.absolute_path
        )
        raise PreflightError(
            f"manifest violates bundled schema at {location}: {error.message}"
        )


def parse_docker_image_inspect(raw: bytes, expected_repo_digest: str) -> dict[str, str]:
    """Parse the unformatted JSON array emitted by `docker image inspect`."""
    try:
        parsed = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PreflightError(f"invalid Docker inspect JSON: {exc}") from exc
    if not isinstance(parsed, list) or len(parsed) != 1 or not isinstance(parsed[0], dict):
        raise PreflightError("Docker inspect must return exactly one image object")
    image = parsed[0]
    image_id = image.get("Id")
    repo_digests = image.get("RepoDigests")
    if not isinstance(image_id, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", image_id):
        raise PreflightError("Docker inspect returned an invalid image ID")
    if (
        not isinstance(repo_digests, list)
        or any(not isinstance(item, str) for item in repo_digests)
        or expected_repo_digest not in repo_digests
    ):
        raise PreflightError("Docker inspect did not contain the pinned repository digest")
    return {"image_id": image_id, "repo_digest": expected_repo_digest}


def parse_tool_discovery(raw: bytes) -> dict[str, str]:
    prefix = "ACE2_IMMUTABLE_PPA_PATH\t"
    found: dict[str, str] = {}
    for line in raw.decode("utf-8", errors="strict").splitlines():
        if not line.startswith(prefix):
            continue
        parts = line.split("\t")
        if len(parts) != 3 or parts[1] in found or not parts[2].startswith("/"):
            raise PreflightError("malformed or duplicate tool-discovery record")
        found[parts[1]] = parts[2]
    expected = {"liberty", "opensta", "sed", "tee", "yosys"}
    if set(found) != expected:
        raise PreflightError(
            f"tool discovery keys differ: missing={sorted(expected - set(found))}, "
            f"extra={sorted(set(found) - expected)}"
        )
    return found


def inspect_environment(
    config: dict[str, Any], runner: Run = _run
) -> tuple[dict[str, str], dict[str, str]]:
    docker = config["docker"]
    inspect_raw = runner(
        [docker["executable"], "image", "inspect", docker["image"]], None
    )
    metadata = parse_docker_image_inspect(inspect_raw, docker["expected_repo_digest"])
    if metadata["image_id"] != docker["expected_image_id"]:
        raise PreflightError("installed image ID differs from the pinned image ID")

    tools = config["toolchain"]
    script = "\n".join(
        [
            "set -euo pipefail",
            f"source {shlex.quote(tools['environment_script'])}",
            'y=$(readlink -f "$(command -v yosys)")',
            's=$(readlink -f "$(command -v sta)")',
            f"l={shlex.quote(tools['liberty'])}",
            f"d={shlex.quote(config['netlist_transform']['executable'])}",
            f"t={shlex.quote(tools['tee'])}",
            'test -x "$y" && test -x "$s" && test -r "$l"',
            'test -x "$d" && test -x "$t"',
            "printf 'ACE2_IMMUTABLE_PPA_PATH\\tyosys\\t%s\\n' \"$y\"",
            "printf 'ACE2_IMMUTABLE_PPA_PATH\\topensta\\t%s\\n' \"$s\"",
            "printf 'ACE2_IMMUTABLE_PPA_PATH\\tliberty\\t%s\\n' \"$l\"",
            "printf 'ACE2_IMMUTABLE_PPA_PATH\\tsed\\t%s\\n' \"$d\"",
            "printf 'ACE2_IMMUTABLE_PPA_PATH\\ttee\\t%s\\n' \"$t\"",
        ]
    )
    discovery_raw = runner(
        [
            docker["executable"],
            "run",
            "--rm",
            "--network",
            "none",
            "--read-only",
            "--entrypoint",
            "/bin/bash",
            docker["image"],
            "-lc",
            script,
        ],
        None,
    )
    discovered = parse_tool_discovery(discovery_raw)
    pinned = {
        "yosys": tools["yosys"],
        "opensta": tools["opensta"],
        "liberty": tools["liberty"],
        "sed": config["netlist_transform"]["executable"],
        "tee": tools["tee"],
    }
    if discovered != pinned:
        raise PreflightError(f"discovered container paths differ from pinned paths: {discovered}")
    return metadata, discovered


def _validate_ref(ref: str) -> None:
    if (
        not ref
        or ref.startswith("-")
        or ".." in ref
        or "@{" in ref
        or not re.fullmatch(r"[A-Za-z0-9_./-]+", ref)
    ):
        raise PreflightError(f"unsafe Git revision: {ref!r}")


def _input_set_hash(entries: list[dict[str, str]]) -> str:
    payload = b"".join(
        entry["path"].encode("utf-8")
        + b"\0"
        + entry["sha256"].encode("ascii")
        + b"\n"
        for entry in entries
    )
    return sha256_bytes(payload)


def _script_paths(script: bytes, command: str, suffixes: tuple[str, ...]) -> list[str]:
    try:
        text = script.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise PreflightError(f"{command} script is not UTF-8: {exc}") from exc
    paths: list[str] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        try:
            fields = shlex.split(line, comments=True, posix=True)
        except ValueError as exc:
            raise PreflightError(
                f"could not parse {command} script line {line_number}: {exc}"
            ) from exc
        if not fields or fields[0] != command:
            continue
        operands = [field for field in fields[1:] if field.endswith(suffixes)]
        if not operands:
            raise PreflightError(
                f"{command} script line {line_number} has no supported path operand"
            )
        for operand in operands:
            _validate_repo_path(operand, f"{command} path")
        paths.extend(operands)
    return paths


def validate_flow_input_closure(
    synthesis_script: bytes,
    sta_script: bytes,
    config: dict[str, Any],
    role: str,
) -> None:
    inputs = config["inputs"]
    transform = config["netlist_transform"]
    observed = {
        "synthesis RTL": _script_paths(
            synthesis_script, "read_verilog", (".v", ".sv", ".vh", ".svh")
        ),
        "synthesis netlist output": _script_paths(
            synthesis_script, "write_verilog", (".v", ".sv")
        ),
        "STA netlist input": _script_paths(
            sta_script, "read_verilog", (".v", ".sv")
        ),
        "STA constraints": _script_paths(sta_script, "read_sdc", (".sdc",)),
    }
    expected = {
        "synthesis RTL": list(inputs["rtl"]),
        "synthesis netlist output": [transform["input"]],
        "STA netlist input": [transform["output"]],
        "STA constraints": list(inputs["constraints"]),
    }
    for label, expected_paths in expected.items():
        if observed[label] != expected_paths:
            raise PreflightError(
                f"{role} {label} closure differs: "
                f"expected={expected_paths}, observed={observed[label]}"
            )


def resolve_revision(
    repo: Path,
    ref: str,
    role: str,
    config: dict[str, Any],
    runner: Run = _run,
) -> dict[str, Any]:
    _validate_ref(ref)
    commit = runner(["git", "rev-parse", "--verify", f"{ref}^{{commit}}"], repo).decode().strip()
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise PreflightError(f"{role} did not resolve to a full SHA-1 commit")
    tree = runner(["git", "show", "-s", "--format=%T", commit], repo).decode().strip()
    if not re.fullmatch(r"[0-9a-f]{40}", tree):
        raise PreflightError(f"{role} did not resolve to a full SHA-1 tree")

    inputs = config["inputs"]
    groups = {
        "rtl": list(inputs["rtl"]),
        "constraints": list(inputs["constraints"]),
        "flow": [inputs["synthesis_script"], inputs["sta_script"]],
    }
    resolved: dict[str, list[dict[str, str]]] = {}
    hashes: dict[str, str] = {}
    contents: dict[str, bytes] = {}
    for group, paths in groups.items():
        entries: list[dict[str, str]] = []
        for path in paths:
            blob = runner(["git", "rev-parse", f"{commit}:{path}"], repo).decode().strip()
            if not re.fullmatch(r"[0-9a-f]{40}", blob):
                raise PreflightError(f"{role}:{path} did not resolve to a Git blob")
            object_type = runner(["git", "cat-file", "-t", blob], repo).decode().strip()
            if object_type != "blob":
                raise PreflightError(
                    f"{role}:{path} resolved to Git object type {object_type!r}, not a blob"
                )
            content = runner(["git", "show", f"{commit}:{path}"], repo)
            contents[path] = content
            entries.append({"git_blob": blob, "path": path, "sha256": sha256_bytes(content)})
        resolved[group] = entries
        hashes[f"{group}_sha256"] = _input_set_hash(entries)
    validate_flow_input_closure(
        contents[inputs["synthesis_script"]],
        contents[inputs["sta_script"]],
        config,
        role,
    )
    return {
        "commit": commit,
        "constraints": resolved["constraints"],
        "constraints_sha256": hashes["constraints_sha256"],
        "flow": resolved["flow"],
        "flow_sha256": hashes["flow_sha256"],
        "requested_ref": ref,
        "role": role,
        "rtl": resolved["rtl"],
        "rtl_sha256": hashes["rtl_sha256"],
        "tree": tree,
    }


def validate_namespace(root: Path, namespace: Path, leaf_pattern: str) -> tuple[Path, Path]:
    root = root.resolve()
    namespace = namespace.resolve()
    if not root.is_absolute() or namespace.parent != root:
        raise PreflightError("namespace must be a direct child of the configured absolute root")
    if not re.fullmatch(leaf_pattern, namespace.name):
        raise PreflightError("namespace leaf does not match the configured naming rule")
    if namespace.exists():
        raise PreflightError(f"namespace already exists; overwrite/retry refused: {namespace}")
    return root, namespace


def render_commands(
    config: dict[str, Any], namespace: Path, role: str, uid: int, gid: int
) -> list[dict[str, Any]]:
    docker = config["docker"]
    tools = config["toolchain"]
    inputs = config["inputs"]
    transform = config["netlist_transform"]
    role_root = namespace / role
    prefix = [
        docker["executable"],
        "run",
        "--rm",
        "--network",
        "none",
        "--user",
        f"{uid}:{gid}",
        "--mount",
        f"type=bind,src={role_root},dst=/run",
        "--workdir",
        "/run/source",
        docker["image"],
        "/bin/bash",
        "-lc",
    ]
    synth = (
        f"set -euo pipefail; source {shlex.quote(tools['environment_script'])}; "
        f"exec {shlex.quote(tools['yosys'])} -l /run/yosys.log "
        f"-s {shlex.quote(inputs['synthesis_script'])}"
    )
    sanitize = (
        f"set -euo pipefail; {shlex.quote(transform['executable'])} -E "
        f"{shlex.quote(transform['expression'])} "
        f"{shlex.quote('/run/source/' + transform['input'])} > "
        f"{shlex.quote('/run/source/' + transform['output'])}"
    )
    sta = (
        f"set -euo pipefail; source {shlex.quote(tools['environment_script'])}; "
        f"{shlex.quote(tools['opensta'])} -exit {shlex.quote(inputs['sta_script'])} "
        f"2>&1 | {shlex.quote(tools['tee'])} /run/sta.log"
    )
    return [
        {"argv": prefix + [synth], "name": "synthesis"},
        {"argv": prefix + [sanitize], "name": "netlist_transform"},
        {"argv": prefix + [sta], "name": "sta"},
    ]


def build_manifest(
    repo: Path,
    base_ref: str,
    candidate_ref: str,
    namespace: Path,
    config: dict[str, Any],
    metadata: dict[str, str],
    discovered: dict[str, str],
    *,
    uid: int | None = None,
    gid: int | None = None,
    config_path: Path = DEFAULT_CONFIG,
    schema_path: Path = DEFAULT_SCHEMA,
    runner: Run = _run,
    interface_bundle: tuple[dict[str, Any], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    repo = repo.resolve()
    if interface_bundle is None:
        interface_bundle = load_interface_contract(repo, config)
    interface_data, interface_result = interface_bundle
    root = (repo / config["namespace"]["root"]).resolve()
    root, namespace = validate_namespace(
        root, namespace, config["namespace"]["leaf_pattern"]
    )
    runner(
        ["git", "cat-file", "-e", f"{config['package_base_commit']}^{{commit}}"],
        repo,
    )
    revisions = [
        resolve_revision(repo, base_ref, "base", config, runner),
        resolve_revision(repo, candidate_ref, "candidate", config, runner),
    ]
    if revisions[0]["commit"] != interface_data["base_commit"]:
        raise PreflightError(
            "base revision must equal the benchmark-interface public base commit"
        )
    if revisions[0]["constraints_sha256"] != revisions[1]["constraints_sha256"]:
        raise PreflightError("candidate constraints differ from immutable public base")
    if revisions[0]["flow_sha256"] != revisions[1]["flow_sha256"]:
        raise PreflightError("candidate synthesis/STA scripts differ from immutable public base")
    try:
        interface_contract.validate_revision(
            interface_data, repo, revisions[1]["commit"]
        )
    except interface_contract.InterfaceError as exc:
        raise PreflightError(f"candidate benchmark interface is incompatible: {exc}") from exc
    actual_uid = os.getuid() if uid is None else uid
    actual_gid = os.getgid() if gid is None else gid
    manifest = {
        "attempt_policy": dict(config["attempt_policy"]),
        "claim_policy": {
            "ppa_execution_permitted": False,
            "preflight_only": True,
            "timing_claim_permitted": False,
        },
        "commands": [
            {
                "role": role,
                "steps": render_commands(config, namespace, role, actual_uid, actual_gid),
            }
            for role in ("base", "candidate")
        ],
        "benchmark_interface": {
            "base_commit": interface_result["base_commit"],
            "candidate_compatibility_validated": True,
            "module": interface_result["module"],
            "parameter_count": interface_result["parameter_count"],
            "port_count": interface_result["port_count"],
            "schema_path": interface_result["schema_path"],
            "schema_sha256": interface_result["schema_sha256"],
            "sidecar_path": interface_result["sidecar_path"],
            "sidecar_sha256": interface_result["sidecar_sha256"],
            "validator_path": interface_result["validator_path"],
            "validator_sha256": interface_result["validator_sha256"],
        },
        "flow_id": config["flow_id"],
        "namespace": {
            "create_mode": config["namespace"]["create_mode"],
            "exclusive_create": True,
            "leaf_pattern": config["namespace"]["leaf_pattern"],
            "overwrite_allowed": False,
            "path": str(namespace),
            "retry_allowed": False,
            "root": str(root),
            "source_layout": "<namespace>/<role>/source",
        },
        "netlist_transform": dict(config["netlist_transform"]),
        "package": {
            "base_commit": config["package_base_commit"],
            "config_sha256": sha256_file(config_path),
            "implementation_sha256": sha256_file(SELF),
            "schema_sha256": sha256_file(schema_path),
        },
        "revisions": revisions,
        "schema_version": 1,
        "target": dict(config["target"]),
        "toolchain": {
            "docker_executable": config["docker"]["executable"],
            "environment_script": config["toolchain"]["environment_script"],
            "image": config["docker"]["image"],
            "image_id": metadata["image_id"],
            "liberty": discovered["liberty"],
            "opensta": discovered["opensta"],
            "repo_digest": metadata["repo_digest"],
            "tee": discovered["tee"],
            "yosys": discovered["yosys"],
        },
    }
    validate_manifest(manifest, schema_path)
    return manifest


def write_json_exclusive(path: Path, data: Any) -> None:
    if not path.parent.is_dir():
        raise PreflightError(f"output parent must already exist: {path.parent}")
    try:
        with path.open("xb") as handle:
            handle.write(canonical_json(data))
    except FileExistsError as exc:
        raise PreflightError(f"refusing to overwrite existing file: {path}") from exc


def reserve_namespace(
    root: Path,
    namespace: Path,
    leaf_pattern: str,
    manifest_sha256: str,
) -> None:
    root, namespace = validate_namespace(root, namespace, leaf_pattern)
    if not root.is_dir():
        raise PreflightError(f"namespace root must already exist: {root}")
    if not re.fullmatch(r"[0-9a-f]{64}", manifest_sha256):
        raise PreflightError("manifest_sha256 must be a lowercase SHA-256 digest")
    try:
        os.mkdir(namespace, mode=0o750)
    except FileExistsError as exc:
        raise PreflightError(f"namespace already exists; retry refused: {namespace}") from exc
    try:
        write_json_exclusive(
            namespace / "RESERVATION.json",
            {
                "manifest_sha256": manifest_sha256,
                "overwrite_allowed": False,
                "retry_allowed": False,
                "state": "reserved_unconsumed",
            },
        )
    except Exception:
        os.rmdir(namespace)
        raise


def _preflight(args: argparse.Namespace) -> int:
    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    repo = Path(args.repo).resolve()
    interface_bundle = load_interface_contract(repo, config)
    metadata, discovered = inspect_environment(config)
    manifest = build_manifest(
        repo,
        args.base_ref,
        args.candidate_ref,
        Path(args.namespace),
        config,
        metadata,
        discovered,
        config_path=config_path,
        schema_path=DEFAULT_SCHEMA,
        interface_bundle=interface_bundle,
    )
    if args.output:
        write_json_exclusive(Path(args.output), manifest)
    else:
        sys.stdout.buffer.write(canonical_json(manifest))
    return 0


def _reserve(args: argparse.Namespace) -> int:
    if not args.acknowledge_create:
        raise PreflightError("namespace creation requires --acknowledge-create")
    config = load_config(Path(args.config))
    reserve_namespace(
        Path(args.namespace_root),
        Path(args.namespace),
        config["namespace"]["leaf_pattern"],
        args.manifest_sha256,
    )
    return 0


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.set_defaults(func=None)
    sub = parser.add_subparsers(dest="command")

    preflight = sub.add_parser(
        "preflight",
        help="inspect and render only; never create a namespace or execute PPA tools",
    )
    preflight.add_argument("--repo", default=".")
    preflight.add_argument("--config", default=str(DEFAULT_CONFIG))
    preflight.add_argument("--base-ref", required=True)
    preflight.add_argument("--candidate-ref", required=True)
    preflight.add_argument("--namespace", required=True)
    preflight.add_argument("--output")
    preflight.set_defaults(func=_preflight)

    reserve = sub.add_parser("reserve-namespace", help="exclusively reserve one future namespace")
    reserve.add_argument("--config", default=str(DEFAULT_CONFIG))
    reserve.add_argument("--namespace-root", required=True)
    reserve.add_argument("--namespace", required=True)
    reserve.add_argument("--manifest-sha256", required=True)
    reserve.add_argument("--acknowledge-create", action="store_true")
    reserve.set_defaults(func=_reserve)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    if args.func is None:
        raise PreflightError("a subcommand is required")
    return args.func(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PreflightError as exc:
        print(f"immutable-ppa: ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
