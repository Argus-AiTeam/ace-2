#!/usr/bin/env python3
"""Create reproducible random Python-oracle vectors for the operator demo."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import random
import secrets
import shutil
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
BUILD = ROOT / "build" / "operator_demo" / "random_challenge"
GENERATED = BUILD / "verification" / "generated"
TB = BUILD / "verification" / "tb"

sys.path.insert(0, str(TOOLS))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed",
        help="Hex or text seed. A fresh 128-bit seed is generated when omitted.",
    )
    return parser.parse_args()


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def redirect(module: object, **paths: Path) -> None:
    for name, path in paths.items():
        setattr(module, name, path)


def main() -> None:
    args = parse_args()
    seed = args.seed or secrets.token_hex(16)
    rng = random.Random(seed)
    if BUILD.exists():
        shutil.rmtree(BUILD)
    GENERATED.mkdir(parents=True)
    TB.mkdir(parents=True)

    rope_ref = importlib.import_module("ace2_rope_reference")
    rope_gen = importlib.import_module("gen_rope_vectors")
    rope_standard = rope_gen._cases

    def rope_cases() -> list[object]:
        angles = ((32767, 0), (30274, 12540), (23170, 23170), (12540, 30274))
        cos_q15: list[int] = []
        sin_q15: list[int] = []
        for _ in range(rope_ref.HIDDEN_SIZE):
            cos_value, sin_value = rng.choice(angles)
            if rng.randrange(2):
                sin_value = -sin_value
            cos_q15.append(cos_value)
            sin_q15.append(sin_value)
        random_case = rope_ref.RopeCase(
            "random_python_oracle",
            rng.randrange(0, 65536),
            [rng.randint(-96, 95) for _ in range(rope_ref.HIDDEN_SIZE)],
            [rng.randint(256, 768) for _ in range(rope_ref.HIDDEN_SIZE)],
            cos_q15,
            sin_q15,
        )
        return [*rope_standard(), random_case]

    rope_gen._cases = rope_cases
    redirect(
        rope_gen,
        OUT_SVH=GENERATED / "rope_vectors.svh",
        OUT_JSON=GENERATED / "rope_vectors.json",
    )
    rope_gen.main()

    score_ref = importlib.import_module("ace2_attention_score_reference")
    score_gen = importlib.import_module("gen_attention_score_vectors")
    score_standard = score_gen._cases

    def score_cases() -> list[object]:
        context = rng.randint(2, score_ref.CONTEXT_MAX)
        random_case = score_ref.AttentionScoreCase(
            "random_python_oracle",
            [rng.randint(-48, 47) for _ in range(score_ref.HEAD_DIM)],
            [
                [rng.randint(-48, 47) for _ in range(score_ref.HEAD_DIM)]
                for _ in range(context)
            ],
        )
        return [*score_standard(), random_case]

    score_gen._cases = score_cases
    redirect(
        score_gen,
        OUT_SVH=GENERATED / "attention_score_vectors.svh",
        OUT_JSON=GENERATED / "attention_score_vectors.json",
    )
    score_gen.main()

    softmax_ref = importlib.import_module("ace2_softmax_reference")
    softmax_gen = importlib.import_module("gen_softmax_vectors")
    softmax_standard = softmax_gen._cases

    def softmax_cases() -> list[object]:
        context = rng.randint(2, softmax_ref.CONTEXT_MAX)
        random_case = softmax_ref.SoftmaxCase(
            "random_python_oracle",
            [rng.randint(-4096, 3072) for _ in range(context)],
        )
        return [*softmax_standard(), random_case]

    softmax_gen._cases = softmax_cases
    redirect(
        softmax_gen,
        OUT_SVH=GENERATED / "softmax_vectors.svh",
        OUT_JSON=GENERATED / "softmax_vectors.json",
    )
    softmax_gen.main()

    compose_ref = importlib.import_module("ace2_attention_compose_reference")
    compose_gen = importlib.import_module("gen_attention_compose_vectors")
    compose_standard = compose_gen._cases

    def compose_cases() -> list[object]:
        context = rng.randint(9, compose_gen.MAX_CONTEXT)
        random_case = compose_ref.AttentionComposeCase(
            "random_python_oracle",
            [rng.randint(-3072, 3072) for _ in range(context)],
            [
                [rng.randint(-96, 95) for _ in range(compose_ref.HEAD_DIM)]
                for _ in range(context)
            ],
        )
        return [*compose_standard(), random_case]

    compose_gen._cases = compose_cases
    redirect(
        compose_gen,
        OUT_SVH=GENERATED / "attention_compose_vectors.svh",
        OUT_JSON=GENERATED / "attention_compose_vectors.json",
    )
    compose_gen.main()

    silu_ref = importlib.import_module("ace2_silu_gate_reference")
    silu_gen = importlib.import_module("gen_silu_gate_vectors")
    silu_standard = silu_gen.cases

    def silu_cases() -> list[object]:
        length = rng.randint(17, 47)
        random_case = silu_ref.SiluGateCase(
            "random_python_oracle",
            [rng.randint(-4096, 4096) for _ in range(length)],
            [rng.randint(-2048, 2047) for _ in range(length)],
            rng.choice((1, 3, 5, 7)),
            rng.randint(16, 21),
            rng.randint(-4, 4),
        )
        return [*silu_standard(), random_case]

    silu_gen.cases = silu_cases
    redirect(
        silu_gen,
        LUT_OUT=BUILD / "ace2_silu_lut.svh",
        SV_OUT=GENERATED / "silu_gate_vectors.svh",
        JSON_OUT=GENERATED / "silu_gate_vectors.json",
        SHELL_SV_OUT=GENERATED / "silu_gate_shell_vectors.svh",
        SHELL_JSON_OUT=GENERATED / "silu_gate_shell_vectors.json",
    )
    silu_gen.main()

    testbenches = (
        "ace2_rope_tb.sv",
        "ace2_attention_score_tb.sv",
        "ace2_softmax_tb.sv",
        "ace2_attention_compose_tb.sv",
        "ace2_silu_gate_tb.sv",
    )
    for name in testbenches:
        shutil.copy2(ROOT / "verification" / "tb" / name, TB / name)

    vector_files = sorted(GENERATED.glob("*.json"))
    record = {
        "schema_version": 1,
        "seed": seed,
        "random_cases_per_operator": 1,
        "operators": [
            {
                "name": path.stem.removesuffix("_vectors"),
                "oracle_json": str(path.relative_to(ROOT)),
                "oracle_sha256": digest(path),
                "case_count": len(json.loads(path.read_text(encoding="utf-8"))["cases"]),
            }
            for path in vector_files
            if path.name != "silu_gate_shell_vectors.json"
        ],
        "boundary": (
            "Random inputs are generated from this seed. Expected outputs are "
            "computed by the checked-in bit-accurate Python references, then "
            "compiled into temporary RTL testbench vectors under build/."
        ),
    }
    (BUILD / "challenge.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"ACE2_RANDOM_OPERATOR_CHALLENGE_CREATED seed={seed}")


if __name__ == "__main__":
    main()
