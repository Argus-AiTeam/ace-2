#!/usr/bin/env python3
"""Verify the compact publication and optionally its raw evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_hash(path: Path, expected: str) -> None:
    actual = sha256(path)
    if actual != expected:
        raise SystemExit(f"hash mismatch: {path}: expected {expected}, got {actual}")


def verify_package(manifest: dict[str, object]) -> None:
    files = manifest["package_files"]
    assert isinstance(files, dict)
    for name, expected in files.items():
        require_hash(ROOT / name, str(expected))


def verify_raw(manifest: dict[str, object], raw_root: Path, full_raw: bool) -> None:
    raw = manifest["raw_archive"]
    assert isinstance(raw, dict)
    files = raw["terminal_files"]
    assert isinstance(files, dict)
    for name, record in files.items():
        assert isinstance(record, dict)
        path = raw_root / name
        if path.stat().st_size != int(record["bytes"]):
            raise SystemExit(f"size mismatch: {path}")
        require_hash(path, str(record["sha256"]))

    if full_raw:
        for line in (raw_root / "SHA256SUMS").read_text().splitlines():
            expected, relative = line.split("  ", 1)
            require_hash(raw_root / relative, expected)


def verify_sources(project_root: Path) -> None:
    bindings = json.loads((ROOT / "SOURCE_BINDINGS.json").read_text())
    for record in bindings:
        require_hash(project_root / record["path"], record["sha256"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path)
    parser.add_argument("--project-root", type=Path)
    parser.add_argument("--full-raw", action="store_true")
    args = parser.parse_args()

    manifest = json.loads((ROOT / "PUBLICATION_MANIFEST.json").read_text())
    verify_package(manifest)
    if args.raw_root:
        verify_raw(manifest, args.raw_root, args.full_raw)
    elif args.full_raw:
        parser.error("--full-raw requires --raw-root")
    if args.project_root:
        verify_sources(args.project_root)

    print("PASS_COMPACT_RTL_CHAT_ATTEMPT_0002_PUBLICATION")


if __name__ == "__main__":
    main()
