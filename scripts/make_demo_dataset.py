#!/usr/bin/env python3
"""Generate a tiny deterministic XRD-like dataset for platform validation."""

from __future__ import annotations

import argparse
import math
import shutil
from pathlib import Path


PHASES = {
    "J12": [23.4, 31.1, 37.8, 45.2],
    "J13": [20.8, 28.9, 35.7, 50.3],
}


def gaussian(x: float, center: float, width: float) -> float:
    return math.exp(-0.5 * ((x - center) / width) ** 2)


def write_spectrum(path: Path, peaks: list[float], replicate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# two_theta intensity"]
    for idx in range(800):
        x = 10.0 + idx * 0.07
        baseline = 0.08 + 0.015 * math.sin(x / 4.0 + replicate)
        y = baseline
        for peak_index, peak in enumerate(peaks):
            shifted = peak + 0.03 * (replicate - 1)
            width = 0.18 + 0.015 * peak_index
            height = 1.0 - 0.08 * peak_index + 0.03 * replicate
            y += height * gaussian(x, shifted, width)
        lines.append(f"{x:.4f} {y:.8f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/demo_xrd")
    parser.add_argument("--replicates", type=int, default=4)
    parser.add_argument("--zip", action="store_true", help="Also create a zip next to the dataset directory.")
    args = parser.parse_args()

    out = Path(args.output).expanduser().resolve()
    if out.exists():
        shutil.rmtree(out)
    for label, peaks in PHASES.items():
        for replicate in range(args.replicates):
            write_spectrum(out / label / f"{label}_sample_{replicate + 1}.xy", peaks, replicate)

    if args.zip:
        archive = shutil.make_archive(str(out), "zip", root_dir=out)
        print(archive)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
