#!/usr/bin/env python3
"""Render selected scalar signals from the local challenge VCD as SVG."""

from __future__ import annotations

from html import escape
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VCD = ROOT / "build" / "demo_challenge" / "rmsnorm-waveform.vcd"
SVG = ROOT / "build" / "demo_challenge" / "rmsnorm-waveform.svg"
SIGNALS = (
    "start_valid",
    "start_ready",
    "in_valid",
    "in_ready",
    "gain_valid",
    "out_valid",
    "done_valid",
    "saturation_seen",
)


def main() -> None:
    codes: dict[str, str] = {}
    transitions: dict[str, list[tuple[int, str]]] = {name: [] for name in SIGNALS}
    current_time = 0
    definitions_done = False

    for raw_line in VCD.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line.startswith("$var "):
            parts = line.split()
            name = parts[4]
            if name in transitions:
                codes[parts[3]] = name
        elif line == "$enddefinitions $end":
            definitions_done = True
        elif definitions_done and line.startswith("#"):
            current_time = int(line[1:])
        elif definitions_done and len(line) >= 2 and line[0] in "01xz":
            name = codes.get(line[1:])
            if name is not None:
                value = line[0]
                if not transitions[name] or transitions[name][-1][1] != value:
                    transitions[name].append((current_time, value))

    observed = [time for rows in transitions.values() for time, _ in rows]
    if not observed:
        raise SystemExit("no selected scalar transitions found in challenge VCD")
    start, end = min(observed), max(observed)
    span = max(1, end - start)
    width = 1200
    left = 145
    right = 30
    plot_width = width - left - right
    row_height = 42
    height = 58 + row_height * len(SIGNALS)

    def x(time: int) -> float:
        return left + (time - start) * plot_width / span

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>text{font-family:ui-monospace,monospace;fill:#dcecff}"
        ".label{font-size:13px}.tick{font-size:11px;fill:#8da7c2}"
        ".grid{stroke:#29445f;stroke-width:1}.wave{fill:none;stroke:#51e5c2;stroke-width:2}"
        ".unknown{stroke:#ffbd66}</style>",
        '<rect width="100%" height="100%" fill="#07111f"/>',
        '<text x="18" y="26" font-size="16">Fresh local RMSNorm challenge waveform</text>',
        f'<text class="tick" x="{left}" y="45">0 us</text>',
        f'<text class="tick" x="{width-right-70}" y="45">{span / 1_000_000:.1f} us</text>',
    ]

    for index, name in enumerate(SIGNALS):
        top = 58 + index * row_height
        high = top + 7
        low = top + 29
        svg.append(f'<text class="label" x="12" y="{top + 22}">{escape(name)}</text>')
        svg.append(f'<line class="grid" x1="{left}" y1="{low}" x2="{width-right}" y2="{low}"/>')
        rows = transitions[name]
        if not rows:
            continue
        points: list[tuple[float, float]] = []
        previous_value = rows[0][1]
        previous_y = high if previous_value == "1" else low
        points.append((left, previous_y))
        for time, value in rows:
            next_x = x(time)
            next_y = high if value == "1" else low
            points.append((next_x, previous_y))
            points.append((next_x, next_y))
            previous_y = next_y
        points.append((width - right, previous_y))
        serialized = " ".join(f"{px:.2f},{py:.2f}" for px, py in points)
        svg.append(f'<polyline class="wave" points="{serialized}"/>')

    svg.append("</svg>")
    SVG.write_text("\n".join(svg) + "\n", encoding="utf-8")
    print(f"ACE2_LOCAL_WAVEFORM_SVG_WRITTEN {SVG.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
