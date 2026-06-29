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
    "K10": [24.0, 31.7, 38.4, 46.0],
    "Z99": [17.5, 26.1, 42.0, 55.8],
}

CIF_PHASES = {
    "J12": {
        "cell": (5.42, 5.42, 9.81),
        "atoms": [("Al", 0.000, 0.000, 0.000), ("In", 0.333, 0.667, 0.250), ("Se", 0.210, 0.420, 0.620)],
    },
    "J13": {
        "cell": (5.18, 5.72, 10.34),
        "atoms": [("Al", 0.100, 0.000, 0.000), ("In", 0.410, 0.650, 0.280), ("Se", 0.260, 0.390, 0.690)],
    },
    "K10": {
        "cell": (4.92, 6.02, 9.44),
        "atoms": [("K", 0.030, 0.130, 0.070), ("Al", 0.360, 0.520, 0.330), ("Se", 0.190, 0.760, 0.610)],
    },
    "Z99": {
        "cell": (6.15, 4.88, 8.77),
        "atoms": [("Zn", 0.080, 0.090, 0.160), ("In", 0.580, 0.270, 0.430), ("Se", 0.320, 0.710, 0.780)],
    },
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


def write_cif(path: Path, label: str, replicate: int) -> None:
    phase = CIF_PHASES[label]
    a, b, c = phase["cell"]
    jitter = 0.004 * replicate
    lines = [
        f"data_{label}_sample_{replicate + 1}",
        "_symmetry_space_group_name_H-M 'P 1'",
        f"_cell_length_a {a + jitter:.5f}",
        f"_cell_length_b {b + jitter / 2:.5f}",
        f"_cell_length_c {c + jitter / 3:.5f}",
        "_cell_angle_alpha 90",
        "_cell_angle_beta 90",
        "_cell_angle_gamma 90",
        "loop_",
        "_atom_site_label",
        "_atom_site_type_symbol",
        "_atom_site_fract_x",
        "_atom_site_fract_y",
        "_atom_site_fract_z",
    ]
    for idx, (symbol, fx, fy, fz) in enumerate(phase["atoms"]):
        offset = 0.002 * replicate * (idx + 1)
        lines.append(
            f"{symbol}{idx + 1} {symbol} "
            f"{(fx + offset) % 1:.5f} {(fy + offset / 2) % 1:.5f} {(fz + offset / 3) % 1:.5f}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/demo_xrd")
    parser.add_argument("--replicates", type=int, default=4)
    parser.add_argument("--format", choices=["spectrum", "cif"], default="spectrum")
    parser.add_argument("--zip", action="store_true", help="Also create a zip next to the dataset directory.")
    args = parser.parse_args()

    out = Path(args.output).expanduser().resolve()
    if out.exists():
        shutil.rmtree(out)
    labels = CIF_PHASES if args.format == "cif" else PHASES
    for label, peaks in labels.items():
        for replicate in range(args.replicates):
            if args.format == "cif":
                write_cif(out / label / f"{label}_sample_{replicate + 1}.cif", label, replicate)
            else:
                write_spectrum(out / label / f"{label}_sample_{replicate + 1}.xy", peaks, replicate)

    if args.zip:
        archive = shutil.make_archive(str(out), "zip", root_dir=out)
        print(archive)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
