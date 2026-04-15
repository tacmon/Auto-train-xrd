#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path

DEFAULT_B_POOL = [
    "Si",
    "Al2O3",
    "TiO2",
    "ZnO",
    "MgO",
    "NaCl",
    "CaF2",
    "BaSO4",
    "SnO2",
    "Fe2O3",
    "WO3",
    "MoS2",
    "Bi2Se3",
]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def load_project_env() -> None:
    from dotenv import load_dotenv

    root = repo_root()
    load_dotenv(root / ".env", override=False)
    load_dotenv(root / "libs" / "XRD-1.1" / "Novel-Space" / ".env", override=False)


def require_api_key() -> str:
    key = os.getenv("MP_API_KEY")
    if not key:
        raise SystemExit(
            "MP_API_KEY is missing. Set it in ./.env or libs/XRD-1.1/Novel-Space/.env before querying Materials Project."
        )
    return key


def safe_float(value: object, default: float = math.inf) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_bool(value: object) -> bool:
    return bool(value) if value is not None else False


def normalized_formula(formula: str) -> str:
    return re.sub(r"\s+", "", formula or "").lower()


def get_symmetry_fields(doc: object) -> tuple[str | None, int | None]:
    symmetry = getattr(doc, "symmetry", None)
    if symmetry is None:
        return None, None
    symbol = getattr(symmetry, "symbol", None)
    number = getattr(symmetry, "number", None)
    if symbol is None and isinstance(symmetry, dict):
        symbol = symmetry.get("symbol")
        number = symmetry.get("number")
    return symbol, number


def fingerprint_from_structure(structure, min_angle: float = 20.0, max_angle: float = 60.0) -> tuple[np.ndarray, list[dict[str, float]]]:
    import numpy as np
    from pymatgen.analysis.diffraction.xrd import XRDCalculator

    calc = XRDCalculator(wavelength="CuKa")
    grid = np.linspace(min_angle, max_angle, 4001)
    profile = np.zeros_like(grid)
    pattern = calc.get_pattern(structure, two_theta_range=(min_angle, max_angle))
    sigma = 0.12
    peaks: list[dict[str, float]] = []

    for x_val, y_val in zip(pattern.x, pattern.y):
        profile += y_val * np.exp(-0.5 * ((grid - x_val) / sigma) ** 2)
        peaks.append({"two_theta": round(float(x_val), 3), "intensity": round(float(y_val), 2)})

    if profile.max() > 0:
        profile /= profile.max()

    top_peaks = sorted(peaks, key=lambda item: item["intensity"], reverse=True)[:5]
    return profile, top_peaks


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    import numpy as np

    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    similarity = float(np.dot(a, b) / denom)
    return round(1.0 - similarity, 4)


def search_formula_docs(mpr, formula: str):
    fields = [
        "material_id",
        "formula_pretty",
        "energy_above_hull",
        "is_stable",
        "theoretical",
        "symmetry",
        "nsites",
    ]

    attempts = [
        {"formula": [formula], "fields": fields},
        {"formula": formula, "fields": fields},
        {"formula_pretty": formula, "fields": fields},
    ]

    last_error = None
    for kwargs in attempts:
        try:
            docs = list(mpr.materials.summary.search(**kwargs))
            if docs:
                return docs
        except Exception as exc:  # pragma: no cover - depends on API version
            last_error = exc
    if last_error:
        raise last_error
    return []


def rank_docs(docs: list[object], requested_formula: str) -> list[object]:
    requested = normalized_formula(requested_formula)

    def key(doc: object):
        formula_pretty = normalized_formula(str(getattr(doc, "formula_pretty", "")))
        exact_penalty = 0 if formula_pretty == requested else 1
        stable_penalty = 0 if safe_bool(getattr(doc, "is_stable", False)) else 1
        hull = safe_float(getattr(doc, "energy_above_hull", None))
        theoretical_penalty = 1 if safe_bool(getattr(doc, "theoretical", False)) else 0
        nsites = safe_float(getattr(doc, "nsites", None))
        return (exact_penalty, stable_penalty, hull, theoretical_penalty, nsites, str(getattr(doc, "material_id", "")))

    unique = {}
    for doc in docs:
        material_id = str(getattr(doc, "material_id", ""))
        if material_id and material_id not in unique:
            unique[material_id] = doc
    return sorted(unique.values(), key=key)


def summarize_docs(mpr, formula: str, limit: int = 3) -> list[dict[str, object]]:
    ranked_docs = rank_docs(search_formula_docs(mpr, formula), formula)
    results: list[dict[str, object]] = []

    for doc in ranked_docs[:limit]:
        material_id = str(getattr(doc, "material_id", ""))
        structure = mpr.get_structure_by_material_id(material_id)
        profile, top_peaks = fingerprint_from_structure(structure)
        spg_symbol, spg_number = get_symmetry_fields(doc)
        results.append(
            {
                "material_id": material_id,
                "formula_pretty": str(getattr(doc, "formula_pretty", formula)),
                "energy_above_hull": None
                if getattr(doc, "energy_above_hull", None) is None
                else round(float(getattr(doc, "energy_above_hull")), 6),
                "is_stable": safe_bool(getattr(doc, "is_stable", False)),
                "theoretical": safe_bool(getattr(doc, "theoretical", False)),
                "spacegroup_symbol": spg_symbol,
                "spacegroup_number": spg_number,
                "top_peaks": top_peaks,
                "fingerprint": profile.tolist(),
            }
        )
    return results


def make_b_suggestions(mpr, formula_a: str, suggestion_count: int) -> list[dict[str, object]]:
    import numpy as np

    a_candidates = summarize_docs(mpr, formula_a, limit=1)
    if not a_candidates:
        return []
    a_profile = np.array(a_candidates[0]["fingerprint"], dtype=float)

    suggestions: list[dict[str, object]] = []
    for formula in DEFAULT_B_POOL:
        if normalized_formula(formula) == normalized_formula(formula_a):
            continue
        try:
            candidate = summarize_docs(mpr, formula, limit=1)
        except Exception:
            continue
        if not candidate:
            continue
        top = candidate[0]
        b_profile = np.array(top["fingerprint"], dtype=float)
        top["formula_query"] = formula
        top["contrast_vs_a"] = cosine_distance(a_profile, b_profile)
        suggestions.append(top)

    suggestions.sort(key=lambda item: item["contrast_vs_a"], reverse=True)
    return suggestions[:suggestion_count]


def print_candidates(title: str, entries: list[dict[str, object]]) -> None:
    print(title)
    if not entries:
        print("  (none)")
        return
    for idx, entry in enumerate(entries, start=1):
        peaks = ", ".join(f"{peak['two_theta']}/{peak['intensity']}" for peak in entry["top_peaks"])
        sg_symbol = entry.get("spacegroup_symbol") or "?"
        sg_number = entry.get("spacegroup_number") or "?"
        print(
            f"  {idx}. {entry['material_id']} | {entry['formula_pretty']} | "
            f"stable={entry['is_stable']} | hull={entry['energy_above_hull']} | "
            f"sg={sg_symbol} ({sg_number})"
        )
        if "contrast_vs_a" in entry:
            print(f"     contrast_vs_a={entry['contrast_vs_a']}")
        print(f"     top_peaks(20-60deg): {peaks}")


def command_candidates(args: argparse.Namespace) -> int:
    from mp_api.client import MPRester

    load_project_env()
    api_key = require_api_key()

    with MPRester(api_key) as mpr:
        a_candidates = summarize_docs(mpr, args.formula_a, limit=args.top_k)
        b_candidates = summarize_docs(mpr, args.formula_b, limit=args.top_k) if args.formula_b else []
        b_suggestions = make_b_suggestions(mpr, args.formula_a, args.suggest_b_count) if not args.formula_b else []

    payload = {
        "formula_a": args.formula_a,
        "formula_b": args.formula_b,
        "a_candidates": a_candidates,
        "b_candidates": b_candidates,
        "b_suggestions": b_suggestions,
    }

    if args.json:
        serializable = json.loads(json.dumps(payload))
        for section in ("a_candidates", "b_candidates", "b_suggestions"):
            for entry in serializable[section]:
                entry.pop("fingerprint", None)
        print(json.dumps(serializable, ensure_ascii=False, indent=2))
        return 0

    print_candidates(f"A candidates for {args.formula_a}:", a_candidates)
    print()
    if args.formula_b:
        print_candidates(f"B candidates for {args.formula_b}:", b_candidates)
    else:
        print_candidates("Suggested B formulas from the default contrast pool:", b_suggestions)
    return 0


def sanitize_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("._-") or "material"


def command_download(args: argparse.Namespace) -> int:
    from mp_api.client import MPRester

    load_project_env()
    api_key = require_api_key()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    downloaded = []
    with MPRester(api_key) as mpr:
        for material_id in args.material_id:
            structure = mpr.get_structure_by_material_id(material_id)
            formula = sanitize_name(structure.composition.reduced_formula)
            filename = output_dir / f"{formula}__{material_id}.cif"
            structure.to(fmt="cif", filename=str(filename))
            downloaded.append({"material_id": material_id, "formula": structure.composition.reduced_formula, "path": str(filename)})

    if args.manifest:
        manifest_path = (output_dir / args.manifest).resolve()
        manifest_path.write_text(json.dumps(downloaded, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(downloaded, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Materials Project helper for XRD formula-based training.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    candidates_parser = subparsers.add_parser("candidates", help="List candidate material IDs for formula A/B.")
    candidates_parser.add_argument("--formula-a", required=True)
    candidates_parser.add_argument("--formula-b")
    candidates_parser.add_argument("--top-k", type=int, default=3)
    candidates_parser.add_argument("--suggest-b-count", type=int, default=3)
    candidates_parser.add_argument("--json", action="store_true")
    candidates_parser.set_defaults(func=command_candidates)

    download_parser = subparsers.add_parser("download", help="Download CIFs for one or more material IDs.")
    download_parser.add_argument("--material-id", action="append", required=True)
    download_parser.add_argument("--output-dir", required=True)
    download_parser.add_argument("--manifest", default="download_manifest.json")
    download_parser.set_defaults(func=command_download)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
