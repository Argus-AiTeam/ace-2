#!/usr/bin/env python3
"""Run catalog-declared ACE-2 IP proofs and emit package-level results."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = ROOT / "ip" / "catalog.json"
BUILD_ROOT = ROOT / "build" / "ip_library"


def load_catalog() -> dict:
    return json.loads(CATALOG_PATH.read_text(encoding="utf-8"))


def packages_by_name(catalog: dict) -> dict[str, dict]:
    return {package["name"]: package for package in catalog["packages"]}


def run_package(package: dict) -> bool:
    output_dir = BUILD_ROOT / package["name"]
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    proofs = []
    status = "PASS"

    for operator in package["verification"]["operators"]:
        command = [sys.executable, "scripts/run_single_operator_demo.py", operator]
        result = subprocess.run(
            command,
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        log_path = output_dir / f"{operator}.log"
        log_path.write_text(result.stdout, encoding="utf-8")
        operator_result_path = (
            ROOT / "build" / "single_operator" / operator / "result.json"
        )
        operator_result = None
        if result.returncode == 0 and operator_result_path.is_file():
            operator_result = json.loads(
                operator_result_path.read_text(encoding="utf-8")
            )
        if (
            result.returncode != 0
            or not operator_result
            or operator_result.get("status") != "PASS"
        ):
            status = "FAIL"
        proofs.append(
            {
                "operator": operator,
                "status": (
                    operator_result.get("status", "FAIL")
                    if result.returncode == 0 and operator_result
                    else "FAIL"
                ),
                "returncode": result.returncode,
                "marker": (
                    operator_result.get("marker") if operator_result else None
                ),
                "shared_path_note": (
                    operator_result.get("shared_path_note")
                    if operator_result
                    else None
                ),
                "log": str(log_path.relative_to(ROOT)),
                "underlying_result": (
                    str(operator_result_path.relative_to(ROOT))
                    if operator_result_path.is_file()
                    else None
                ),
            }
        )
        if status == "FAIL":
            break

    package_result = {
        "schema_version": 1,
        "package": package["name"],
        "classification": package["classification"],
        "status": status,
        "seconds": round(time.monotonic() - started, 3),
        "proof_kind": package["verification"]["proof_kind"],
        "proof_note": package["verification"]["proof_note"],
        "proofs": proofs,
    }
    result_path = output_dir / "result.json"
    result_path.write_text(
        json.dumps(package_result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"ACE2_IP_DEMO_{status} package={package['name']} "
        f"result={result_path.relative_to(ROOT)}"
    )
    return status == "PASS"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("package", nargs="?")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    catalog = load_catalog()
    packages = packages_by_name(catalog)
    if args.list:
        for package in catalog["packages"]:
            speed = "slow" if package["verification"]["slow"] else "fast"
            print(
                f"{package['name']}\t{package['classification']}\t{speed}\t"
                f"{package['summary']}"
            )
        return

    if args.all:
        selected = catalog["packages"]
    else:
        if not args.package:
            parser.error("provide a package name, --list, or --all")
        if args.package not in packages:
            parser.error(
                f"unknown package {args.package!r}; use --list to inspect names"
            )
        selected = [packages[args.package]]

    BUILD_ROOT.mkdir(parents=True, exist_ok=True)
    all_passed = True
    for package in selected:
        if not run_package(package):
            all_passed = False
            break
    if not all_passed:
        raise SystemExit(1)
    if args.all:
        print("ACE2_IP_DEMO_ALL_PASS")


if __name__ == "__main__":
    main()
