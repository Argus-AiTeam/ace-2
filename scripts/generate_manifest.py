#!/usr/bin/env python3
from __future__ import annotations

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "MANIFEST.sha256"
EXCLUDED_DIRS = {"build", "__pycache__", ".git", ".cache", ".pytest_cache"}


def included(path: Path) -> bool:
    relative = path.relative_to(ROOT)
    return (
        path.is_file()
        and path != OUTPUT
        and not any(part in EXCLUDED_DIRS for part in relative.parts)
    )


lines = []
for path in sorted((path for path in ROOT.rglob("*") if included(path)), key=lambda p: p.relative_to(ROOT).as_posix()):
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    lines.append(f"{digest}  {path.relative_to(ROOT).as_posix()}")

OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
