#!/usr/bin/env python3
"""Validate the public ACE-2 IP catalog and its command mappings."""

from __future__ import annotations

import json
from pathlib import Path

from run_single_operator_demo import OPERATORS


ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = ROOT / "ip" / "catalog.json"
EXPECTED_PACKAGES = {
    "w4a8_projection",
    "rmsnorm",
    "rope",
    "kv_cache",
    "attention",
    "softmax",
    "silu_swiglu",
    "mlp",
    "qwen25_transformer_layer",
}
VALID_CLASSIFICATIONS = {
    "standalone_core",
    "standalone_cores_with_shared_shell",
    "shared_shell_path",
    "integration_bundle",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"IP catalog validation failed: {message}")


def main() -> None:
    catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    require(catalog.get("schema_version") == 1, "schema_version must be 1")
    packages = catalog.get("packages")
    require(isinstance(packages, list), "packages must be an array")
    names = [package.get("name") for package in packages]
    require(len(names) == len(set(names)), "package names must be unique")
    require(set(names) == EXPECTED_PACKAGES, "package set is incomplete")

    for package in packages:
        name = package["name"]
        package_dir = ROOT / "ip" / name
        manifest_path = package_dir / "manifest.json"
        require((package_dir / "README.md").is_file(), f"{name} README missing")
        require(manifest_path.is_file(), f"{name} manifest missing")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        require(manifest == package, f"{name} manifest differs from catalog")
        require(
            package["classification"] in VALID_CLASSIFICATIONS,
            f"{name} has invalid classification",
        )
        require(
            package["license"]["inheritance"] == "Apache-2.0",
            f"{name} license inheritance must be Apache-2.0",
        )
        for source in package["canonical_rtl"]:
            require((ROOT / source).is_file(), f"{name} source missing: {source}")
        for source in package["dependencies"]["shared_paths"]:
            require((ROOT / source).exists(), f"{name} dependency missing: {source}")
        operators = package["verification"]["operators"]
        require(operators, f"{name} must declare at least one proof operator")
        for operator in operators:
            require(
                operator in OPERATORS,
                f"{name} maps unknown operator {operator}",
            )
        require(
            package["verification"]["command"] == f"make ip-demo IP={name}",
            f"{name} command does not match Makefile surface",
        )

    print(
        f"ACE2_IP_CATALOG_VALID packages={len(packages)} "
        f"operator_mappings={sum(len(p['verification']['operators']) for p in packages)}"
    )


if __name__ == "__main__":
    main()
