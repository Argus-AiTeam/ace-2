"""Shared validation for frozen ACE-2 quality provenance contracts."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any


AREA_CAP_MM2 = 2.0
FREQUENCY_FLOOR_MHZ = 100.0
SCALE32_SIGNIFICAND_MIN = 0x8000
SCALE32_SIGNIFICAND_MAX = 0xFFFF
SCALE32_EXPONENT_MIN = -24
SCALE32_EXPONENT_MAX = 4
SCALE32_ALL_ZERO_RECORD = 0x00E88000


def pack_scale32(significand: int, exponent: int) -> int:
    """Pack the architecture-stage dynamic RoPE scale record."""
    if not SCALE32_SIGNIFICAND_MIN <= significand <= SCALE32_SIGNIFICAND_MAX:
        raise ValueError("Scale32 significand must be normalized unsigned Q1.15")
    if not SCALE32_EXPONENT_MIN <= exponent <= SCALE32_EXPONENT_MAX:
        raise ValueError("Scale32 exponent is outside the frozen range")
    return significand | ((exponent & 0xFF) << 16)


def unpack_scale32(record: int) -> tuple[int, int]:
    if not 0 <= record <= 0xFFFFFFFF or (record >> 24) != 0:
        raise ValueError("Scale32 reserved byte must be zero")
    significand = record & 0xFFFF
    exponent_u8 = (record >> 16) & 0xFF
    exponent = exponent_u8 - 256 if exponent_u8 & 0x80 else exponent_u8
    pack_scale32(significand, exponent)
    return significand, exponent


def round_divide_even_unsigned(numerator: int, denominator: int) -> int:
    if numerator < 0 or denominator <= 0:
        raise ValueError("round_divide_even_unsigned requires numerator >= 0")
    quotient, remainder = divmod(numerator, denominator)
    doubled = remainder * 2
    return quotient + int(
        doubled > denominator or (doubled == denominator and (quotient & 1))
    )


def round_divide_even_signed(numerator: int, denominator: int) -> int:
    """Exact signed round-to-nearest, ties-to-even division."""
    if denominator <= 0:
        raise ValueError("round_divide_even_signed requires denominator > 0")
    magnitude = round_divide_even_unsigned(abs(numerator), denominator)
    return -magnitude if numerator < 0 else magnitude


def ceil_scale32_from_ratio(numerator: int, denominator: int) -> int:
    """Encode the smallest Scale32 value greater than or equal to a ratio."""
    if numerator < 0 or denominator <= 0:
        raise ValueError("Scale32 ceil encoding requires a nonnegative ratio")
    if numerator == 0:
        return SCALE32_ALL_ZERO_RECORD

    for exponent in range(SCALE32_EXPONENT_MIN, SCALE32_EXPONENT_MAX + 1):
        shift = 15 - exponent
        if shift >= 0:
            scaled_numerator = numerator << shift
            scaled_denominator = denominator
        else:
            scaled_numerator = numerator
            scaled_denominator = denominator << -shift
        significand = (scaled_numerator + scaled_denominator - 1) // scaled_denominator
        significand = max(significand, SCALE32_SIGNIFICAND_MIN)
        if significand <= SCALE32_SIGNIFICAND_MAX:
            return pack_scale32(significand, exponent)
    raise OverflowError("positive scale exceeds the frozen Scale32 range")


def ceil_scale32_from_float(value: float) -> int:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("ProducerScale32 input must be finite and positive")
    numerator, denominator = value.as_integer_ratio()
    return ceil_scale32_from_ratio(numerator, denominator)


def scale32_ratio(record: int) -> tuple[int, int]:
    """Return an exact positive numerator/denominator for one Scale32 record."""
    significand, exponent = unpack_scale32(record)
    if exponent >= 15:
        return significand << (exponent - 15), 1
    return significand, 1 << (15 - exponent)


def dynamic_rope_output_scale(producer_record: int, maximum_magnitude: int) -> int:
    """Encode producer_scale*M/(127*2^15) exactly as required by RoPE."""
    producer_sig, producer_exp = unpack_scale32(producer_record)
    if maximum_magnitude < 0:
        raise ValueError("RoPE maximum magnitude must be nonnegative")
    if maximum_magnitude == 0:
        return SCALE32_ALL_ZERO_RECORD
    numerator = producer_sig * maximum_magnitude
    denominator = 127 << 30
    if producer_exp >= 0:
        numerator <<= producer_exp
    else:
        denominator <<= -producer_exp
    return ceil_scale32_from_ratio(numerator, denominator)


def requantize_dynamic_rope_value(
    rotated_s25: int,
    producer_record: int,
    output_record: int,
) -> int:
    """Requantize one staged signed-25-bit RoPE lane to symmetric int8."""
    producer_sig, producer_exp = unpack_scale32(producer_record)
    output_sig, output_exp = unpack_scale32(output_record)
    exponent_delta = producer_exp - output_exp - 15
    numerator = rotated_s25 * producer_sig
    denominator = output_sig
    if exponent_delta >= 0:
        numerator <<= exponent_delta
    else:
        denominator <<= -exponent_delta
    result = round_divide_even_signed(numerator, denominator)
    return max(-127, min(127, result))


def dynamic_score_pair_parameters(query_record: int, key_record: int) -> tuple[int, int]:
    """Return (pair_significand, right_shift) for dynamic Q/K score scaling."""
    query_sig, query_exp = unpack_scale32(query_record)
    key_sig, key_exp = unpack_scale32(key_record)
    pair_sig = round_divide_even_unsigned(query_sig * key_sig, 1 << 15)
    right_shift = 9 - (query_exp + key_exp)
    if not 1 <= right_shift <= 57:
        raise ValueError("dynamic score shift is outside the frozen safe range")
    return pair_sig, right_shift


def fixed_q7_score_pair_parameters(query_record: int, key_record: int) -> tuple[int, int]:
    """Return the fixed-Q7 layer-0 score pair significand and right shift."""
    query_sig, query_exp = unpack_scale32(query_record)
    key_sig, key_exp = unpack_scale32(key_record)
    pair_sig = round_divide_even_unsigned(query_sig * key_sig, 1 << 15)
    right_shift = 23 - (query_exp + key_exp)
    if not 15 <= right_shift <= 71:
        raise ValueError("fixed-Q7 score shift is outside the reviewed safe range")
    return pair_sig, right_shift


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _aggregate_source_hash(source_hashes: list[dict[str, str]]) -> str:
    digest = hashlib.sha256()
    for source in sorted(source_hashes, key=lambda entry: entry["path"]):
        digest.update(source["path"].encode())
        digest.update(b"\0")
        digest.update(source["sha256"].encode())
        digest.update(b"\n")
    return digest.hexdigest()


def artifact(root: Path, path: Path) -> dict[str, Any]:
    return {
        "bytes": path.stat().st_size,
        "path": path.relative_to(root).as_posix(),
        "sha256": sha256_file(path),
    }


def _resolve_file(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if root.resolve() not in path.parents or not path.is_file():
        raise ValueError(f"quality provenance path is not a repository file: {relative}")
    return path


def validate_oracle_manifest(root: Path) -> dict[str, Any]:
    manifest_path = root / "reference" / "ORACLE_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validation_errors: list[str] = []
    if manifest.get("schema_version") != 1:
        validation_errors.append("oracle manifest schema differs from version 1")
    oracles = manifest.get("oracles")
    if not isinstance(oracles, list) or not oracles:
        validation_errors.append("oracle manifest has no oracle entries")
        oracles = []

    artifacts_by_path: dict[str, dict[str, Any]] = {}
    artifact_fields = ("generator", "reference", "vector_json", "vector_svh")
    for index, oracle in enumerate(oracles):
        if not isinstance(oracle, dict):
            validation_errors.append(f"oracle entry {index} is not an object")
            continue
        if not isinstance(oracle.get("case_count"), int) or oracle["case_count"] <= 0:
            validation_errors.append(f"oracle entry {index} has invalid case_count")
        if oracle.get("numeric_acceptance") != "bit_exact_fixed_point_vector_match":
            validation_errors.append(
                f"oracle entry {index} has unsupported numeric_acceptance"
            )
        for field in artifact_fields:
            relative = oracle.get(field)
            expected = oracle.get(f"{field}_sha256")
            if not isinstance(relative, str) or not isinstance(expected, str):
                validation_errors.append(
                    f"oracle entry {index} is missing {field} provenance"
                )
                continue
            try:
                path = _resolve_file(root, relative)
            except ValueError as exc:
                validation_errors.append(str(exc))
                continue
            observed = sha256_file(path)
            if observed != expected:
                validation_errors.append(
                    f"oracle hash differs for {relative}: {observed} != {expected}"
                )
            artifacts_by_path[relative] = artifact(root, path)

    if validation_errors:
        raise ValueError("; ".join(validation_errors))
    return {
        "manifest": artifact(root, manifest_path),
        "entry_count": len(oracles),
        "artifacts": [artifacts_by_path[path] for path in sorted(artifacts_by_path)],
        "valid": True,
    }


def validate_rtl_binding(root: Path) -> dict[str, Any]:
    binding_path = root / "benchmark" / "quality" / "RTL_BINDING.json"
    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    if binding.get("schema_version") != 1:
        raise ValueError("accepted RTL binding schema differs from version 1")
    binding_status = binding.get("status")
    validation_errors: list[str] = []
    if binding_status not in {"accepted", "published_independent_l2_accepted"}:
        validation_errors.append(
            "RTL binding is not independently accepted: "
            f"status={binding_status!r}"
        )

    manifest_spec = binding["manifest"]
    manifest_path = _resolve_file(root, manifest_spec["path"])
    manifest_sha256 = sha256_file(manifest_path)
    if manifest_sha256 != manifest_spec["sha256"]:
        validation_errors.append(
            "accepted RTL manifest hash differs from the quality binding: "
            f"{manifest_sha256} != {manifest_spec['sha256']}"
        )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    candidate_rtl_hash = binding["candidate_rtl_hash"]
    if (
        manifest.get("candidate_rtl_hash") != candidate_rtl_hash
        or manifest.get("rtl_hash") != candidate_rtl_hash
    ):
        validation_errors.append(
            "accepted candidate RTL hash differs from the quality binding"
        )
    manifest_sources = {
        entry["path"]: entry["sha256"]
        for entry in manifest.get("candidate_source_hashes", [])
    }

    constraint_artifacts: list[dict[str, Any]] = []
    ppa_contract: dict[str, Any] | None = None
    if binding_status in {"accepted", "published_independent_l2_accepted"}:
        review = manifest.get("candidate_review_binding", {})
        if (
            manifest.get("candidate_status") != "published_independent_l2_accepted"
            or review.get("decision") != "accepted"
            or review.get("level") != "L2"
        ):
            validation_errors.append(
                "accepted RTL manifest lacks a published independent L2 acceptance"
            )

        evidence = manifest.get("evidence", {})
        ppa_binding = manifest.get("ppa_evidence_binding")
        if not isinstance(ppa_binding, dict):
            ppa_binding = evidence.get("ppa_evidence_binding")
        if not isinstance(ppa_binding, dict):
            validation_errors.append("accepted RTL manifest lacks a PPA evidence binding")
            ppa_binding = {}
        if ppa_binding.get("rtl_hash") != candidate_rtl_hash:
            validation_errors.append("PPA evidence is bound to different RTL")

        constraint_hash = ppa_binding.get("constraint_hash")
        if not isinstance(constraint_hash, str) or len(constraint_hash) != 64:
            validation_errors.append("PPA evidence has no valid constraint hash")
        else:
            constraint_sources = manifest.get("constraint_source_hashes")
            if not isinstance(constraint_sources, list):
                constraint_sources = [
                    entry
                    for entry in manifest.get("candidate_source_hashes", [])
                    if isinstance(entry, dict)
                    and isinstance(entry.get("path"), str)
                    and entry["path"].startswith(("constraints/", "flow/yosys/"))
                ]
            if not constraint_sources:
                validation_errors.append(
                    "accepted RTL manifest does not bind constraint/flow sources"
                )
            validated_constraint_sources: list[dict[str, str]] = []
            seen_constraint_paths: set[str] = set()
            for source in constraint_sources:
                if not isinstance(source, dict):
                    validation_errors.append(
                        "accepted RTL manifest has a malformed constraint source"
                    )
                    continue
                relative = source.get("path")
                expected = source.get("sha256")
                if not isinstance(relative, str) or not isinstance(expected, str):
                    validation_errors.append(
                        "accepted RTL manifest has incomplete constraint provenance"
                    )
                    continue
                if relative in seen_constraint_paths:
                    validation_errors.append(
                        f"accepted RTL manifest repeats constraint source {relative}"
                    )
                    continue
                seen_constraint_paths.add(relative)
                try:
                    path = _resolve_file(root, relative)
                except ValueError as exc:
                    validation_errors.append(str(exc))
                    continue
                observed = sha256_file(path)
                if observed != expected:
                    validation_errors.append(
                        f"constraint hash differs for {relative}: {observed} != {expected}"
                    )
                constraint_artifacts.append(artifact(root, path))
                validated_constraint_sources.append(
                    {"path": relative, "sha256": expected}
                )

            if validated_constraint_sources:
                aggregate_constraint_hash = _aggregate_source_hash(
                    validated_constraint_sources
                )
                if aggregate_constraint_hash != constraint_hash:
                    validation_errors.append(
                        "PPA constraint hash differs from the aggregate bound "
                        "constraint/flow sources"
                    )
            manifest_constraint_hash = manifest.get("constraint_hash")
            if (
                manifest_constraint_hash is not None
                and manifest_constraint_hash != constraint_hash
            ):
                validation_errors.append(
                    "manifest constraint hash differs from the PPA evidence binding"
                )

        ppa = manifest.get("ppa")
        if isinstance(ppa, dict):
            area_mm2 = ppa.get("non_sram_area_mm2")
            frequency_mhz = ppa.get("frequency_floor_mhz")
            wns_ns = ppa.get("sta_wns_ns")
            area_met = ppa.get("area_cap_met")
            frequency_met = ppa.get("frequency_floor_met")
            yosys_log = ppa.get("yosys_log")
            sta_log = ppa.get("opensta_log")
        else:
            yosys = evidence.get("sky130_yosys", {})
            sta = evidence.get("sky130_sta", {})
            area_mm2 = yosys.get("non_sram_area_mm2")
            frequency_mhz = sta.get(
                "estimated_fmax_mhz_floor_bound",
                sta.get("estimated_fmax_mhz"),
            )
            wns_ns = sta.get("wns_ns")
            area_met = isinstance(area_mm2, (int, float)) and area_mm2 <= AREA_CAP_MM2
            frequency_met = (
                isinstance(frequency_mhz, (int, float))
                and frequency_mhz >= FREQUENCY_FLOOR_MHZ
                and isinstance(wns_ns, (int, float))
                and wns_ns >= 0.0
            )
            yosys_log = {
                "path": evidence.get("sky130_synthesis_log"),
                "sha256": yosys.get("log_sha256"),
            }
            sta_log = {
                "path": evidence.get("sky130_sta_log"),
                "sha256": sta.get("log_sha256"),
            }

        if not isinstance(area_mm2, (int, float)) or area_mm2 > AREA_CAP_MM2:
            validation_errors.append(
                f"accepted RTL exceeds the {AREA_CAP_MM2} mm2 non-SRAM cap: {area_mm2!r}"
            )
        if area_met is not True:
            validation_errors.append("accepted RTL PPA evidence does not mark area met")
        if (
            not isinstance(frequency_mhz, (int, float))
            or frequency_mhz < FREQUENCY_FLOOR_MHZ
            or not isinstance(wns_ns, (int, float))
            or wns_ns < 0.0
        ):
            validation_errors.append(
                "accepted RTL misses the 100 MHz timing floor: "
                f"frequency_mhz={frequency_mhz!r}, wns_ns={wns_ns!r}"
            )
        if frequency_met is not True:
            validation_errors.append("accepted RTL PPA evidence does not mark timing met")

        ppa_logs: dict[str, dict[str, Any]] = {}
        for name, spec in (("yosys", yosys_log), ("opensta", sta_log)):
            if not isinstance(spec, dict):
                validation_errors.append(f"accepted RTL lacks bound {name} log evidence")
                continue
            relative = spec.get("path")
            expected = spec.get("sha256")
            if not isinstance(relative, str) or not isinstance(expected, str):
                validation_errors.append(f"accepted RTL lacks bound {name} log evidence")
                continue
            try:
                path = _resolve_file(root, relative)
            except ValueError as exc:
                validation_errors.append(str(exc))
                continue
            observed = sha256_file(path)
            if observed != expected:
                validation_errors.append(
                    f"{name} log hash differs: {observed} != {expected}"
                )
            ppa_logs[name] = artifact(root, path)
        ppa_contract = {
            "area_cap_mm2": AREA_CAP_MM2,
            "frequency_floor_mhz": FREQUENCY_FLOOR_MHZ,
            "non_sram_area_mm2": area_mm2,
            "reported_frequency_mhz": frequency_mhz,
            "wns_ns": wns_ns,
            "constraint_hash": constraint_hash,
            "logs": ppa_logs,
        }

    numerical_rtl = []
    seen_paths: set[str] = set()
    for expected in binding["numerical_rtl"]:
        relative = expected["path"]
        if relative in seen_paths:
            raise ValueError(f"duplicate numerical RTL binding: {relative}")
        seen_paths.add(relative)
        path = _resolve_file(root, relative)
        observed = sha256_file(path)
        if observed != expected["sha256"]:
            validation_errors.append(
                f"numerical RTL hash differs for {relative}: "
                f"{observed} != {expected['sha256']}"
            )
        if manifest_sources.get(relative) != expected["sha256"]:
            validation_errors.append(
                f"RTL manifest does not bind the accepted hash for {relative}"
            )
        numerical_rtl.append(artifact(root, path))
    if not numerical_rtl:
        validation_errors.append("accepted numerical RTL binding is empty")
    if validation_errors:
        raise ValueError("; ".join(validation_errors))

    return {
        "binding": artifact(root, binding_path),
        "binding_status": binding_status,
        "candidate_rtl_hash": candidate_rtl_hash,
        "constraints": constraint_artifacts,
        "manifest": artifact(root, manifest_path),
        "numerical_rtl": numerical_rtl,
        "ppa_contract": ppa_contract,
        "valid": True,
    }
