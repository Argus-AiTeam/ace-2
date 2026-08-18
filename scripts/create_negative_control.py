#!/usr/bin/env python3
"""Corrupt one challenge expectation to prove the checker fails closed."""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "build" / "demo_challenge" / "rmsnorm_vectors.svh"
OUTPUT = ROOT / "build" / "demo_challenge" / "rmsnorm_vectors_negative.svh"


def flip_first_expected(match: re.Match[str]) -> str:
    value = int(match.group(2), 16) ^ 1
    return f"{match.group(1)}{value:032x}{match.group(3)}"


def main() -> None:
    text = SOURCE.read_text(encoding="utf-8")
    corrupted, count = re.subn(
        r"(test_expected_beats\[0\] = 128'h)([0-9a-f]{32})(;)",
        flip_first_expected,
        text,
        count=1,
    )
    if count != 1:
        raise SystemExit("could not locate the first expected RTL beat")
    OUTPUT.write_text(corrupted, encoding="utf-8")
    print("ACE2_NEGATIVE_CONTROL_VECTOR_CREATED index=0 bit=0")


if __name__ == "__main__":
    main()
