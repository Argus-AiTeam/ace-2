#!/usr/bin/env python3
"""Create a fresh local RMSNorm challenge and provenance record."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import platform
import secrets
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build" / "demo_challenge"


def command_output(command: list[str]) -> str:
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    return (result.stdout + result.stderr).strip().splitlines()[0]


def main() -> None:
    BUILD.mkdir(parents=True, exist_ok=True)
    challenge_id = secrets.token_hex(16)
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "gen_rmsnorm_vectors.py"),
            "--challenge-seed",
            challenge_id,
            "--out-json",
            str(BUILD / "rmsnorm_vectors.json"),
            "--out-svh",
            str(BUILD / "rmsnorm_vectors.svh"),
        ],
        cwd=ROOT,
        check=True,
    )
    git_commit = command_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"])
    manifest = ROOT / "CERTIFIED_RTL.sha256"
    record = {
        "schema_version": 1,
        "challenge_id": challenge_id,
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "git_commit": git_commit,
        "certified_rtl_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "tools": {
            "python": command_output([sys.executable, "--version"]),
            "verilator": command_output(["verilator", "--version"]),
            "iverilog": command_output(["iverilog", "-V"]),
            "vvp": command_output(["vvp", "-V"]),
        },
    }
    (BUILD / "challenge.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"ACE2_LOCAL_CHALLENGE_CREATED {challenge_id}")


if __name__ == "__main__":
    main()
