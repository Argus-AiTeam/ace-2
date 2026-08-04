#!/usr/bin/env python3
"""Deterministic distribution-version identity for ACE-2 quality tools."""

from __future__ import annotations

from importlib import metadata
from typing import Any, Callable, Mapping

from packaging.version import InvalidVersion, Version


DISTRIBUTION_NAMES = {
    "lm_eval": "lm-eval",
}


def evaluate_distribution_versions(
    expected: Mapping[str, str],
    *,
    version_getter: Callable[[str], str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Compare installed distributions to frozen PEP 440 public versions.

    A frozen public version such as ``2.11.0`` accepts an observed local build
    such as ``2.11.0+cu130``.  Local labels are reported separately and are not
    a reproducibility binding unless a higher-level contract freezes one.
    """

    lookup = metadata.version if version_getter is None else version_getter
    records: dict[str, dict[str, Any]] = {}
    for package, expected_raw in sorted(expected.items()):
        distribution = DISTRIBUTION_NAMES.get(package, package)
        record: dict[str, Any] = {
            "distribution": distribution,
            "expected": expected_raw,
            "expected_public": None,
            "observed": None,
            "observed_public": None,
            "observed_local": None,
            "matches": False,
            "reason": None,
        }
        try:
            expected_version = Version(expected_raw)
        except InvalidVersion:
            record["reason"] = "invalid_frozen_version"
            records[package] = record
            continue
        record["expected_public"] = expected_version.public
        if expected_version.local is not None:
            record["reason"] = "frozen_local_version_requires_separate_binding"
            records[package] = record
            continue
        try:
            observed_raw = lookup(distribution)
        except metadata.PackageNotFoundError:
            record["reason"] = "distribution_not_found"
            records[package] = record
            continue
        record["observed"] = observed_raw
        try:
            observed_version = Version(observed_raw)
        except InvalidVersion:
            record["reason"] = "invalid_observed_version"
            records[package] = record
            continue
        record["observed_public"] = observed_version.public
        record["observed_local"] = observed_version.local
        expected_public = Version(expected_version.public)
        observed_public = Version(observed_version.public)
        record["matches"] = observed_public == expected_public
        record["reason"] = "match" if record["matches"] else "public_version_mismatch"
        records[package] = record
    return records


def require_distribution_versions(
    expected: Mapping[str, str],
    *,
    version_getter: Callable[[str], str] | None = None,
) -> dict[str, str]:
    """Fail closed unless every installed distribution has the frozen public version."""

    records = evaluate_distribution_versions(expected, version_getter=version_getter)
    mismatches = {name: record for name, record in records.items() if not record["matches"]}
    if mismatches:
        raise RuntimeError(f"runtime distribution versions differ: {mismatches}")
    return dict(expected)
