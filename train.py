#!/usr/bin/env python3
"""OpenI/MMP-compatible training entry for Auto-train-xrd.

This entry is intentionally non-interactive. It reads XRD spectra from a dataset
directory, extracts compact numeric features, trains a lightweight centroid
classifier, and writes all artifacts to the platform task output directory.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


SPECTRUM_EXTENSIONS = {".txt", ".xy", ".gk"}


@dataclass
class SpectrumSample:
    path: Path
    label: str
    x: np.ndarray
    y: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a lightweight XRD spectrum classifier for platform validation."
    )
    parser.add_argument("--dataset", default="", help="Dataset path injected by platform or local path.")
    parser.add_argument("--model_name", default="", help="Optional pretrained model path injected by platform.")
    parser.add_argument("--task_output", default="/result", help="Output path injected by platform.")
    parser.add_argument("--epochs", type=int, default=1, help="Compatibility hyperparameter.")
    parser.add_argument("--bs", type=int, default=16, help="Compatibility hyperparameter.")
    parser.add_argument("--lr", type=float, default=0.001, help="Compatibility hyperparameter.")
    parser.add_argument("--bins", type=int, default=256, help="Number of interpolated XRD feature bins.")
    parser.add_argument("--max_files", type=int, default=0, help="Limit spectra for quick validation; 0 means all.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for deterministic split.")
    args, unknown = parser.parse_known_args()
    if unknown:
        print(
            json.dumps(
                {
                    "event": "ignored_platform_args",
                    "args": unknown,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    return args


def prepare_platform_context() -> dict[str, str]:
    """Use c2net paths when running on OpenI; return empty values locally."""
    try:
        from c2net.context import prepare  # type: ignore

        ctx = prepare()
        return {
            "dataset_path": getattr(ctx, "dataset_path", "") or "",
            "model_path": getattr(ctx, "pretrain_model_path", "") or "",
            "output_path": getattr(ctx, "output_path", "") or "",
        }
    except Exception:
        return {"dataset_path": "", "model_path": "", "output_path": ""}


def first_existing_path(candidates: Iterable[str | Path]) -> Path | None:
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate).expanduser().resolve()
        if path.exists():
            return path
    return None


def _count_spectrum_files(path: Path) -> int:
    if path.is_file():
        return int(path.suffix.lower() in SPECTRUM_EXTENSIONS)
    return sum(
        1
        for candidate in path.rglob("*")
        if candidate.is_file() and candidate.suffix.lower() in SPECTRUM_EXTENSIONS
    )


def _extract_zip_dataset(path: Path) -> Path | None:
    zip_files: list[Path] = []
    if path.is_file() and path.suffix.lower() == ".zip":
        zip_files = [path]
    elif path.is_dir():
        zip_files = sorted(p for p in path.rglob("*.zip") if p.is_file())
    if not zip_files:
        return None

    extract_root = Path(tempfile.mkdtemp(prefix="autotrain_xrd_dataset_"))
    try:
        with zipfile.ZipFile(zip_files[0], "r") as zf:
            zf.extractall(extract_root)
    except Exception:
        shutil.rmtree(extract_root, ignore_errors=True)
        return None
    return extract_root


def select_dataset_root(path: Path) -> Path:
    if _count_spectrum_files(path) > 0:
        return path

    extracted = _extract_zip_dataset(path)
    candidates: list[Path] = []
    if extracted is not None:
        candidates.append(extracted)
        candidates.extend(p for p in extracted.rglob("*") if p.is_dir())
    if path.is_dir():
        candidates.extend(p for p in path.rglob("*") if p.is_dir())

    ranked = sorted(
        ((candidate, _count_spectrum_files(candidate)) for candidate in candidates),
        key=lambda item: (item[1], len(item[0].parts)),
        reverse=True,
    )
    if ranked and ranked[0][1] > 0:
        selected = ranked[0][0]
        print(
            json.dumps(
                {
                    "event": "selected_dataset_root",
                    "input": str(path),
                    "selected": str(selected),
                    "spectra": ranked[0][1],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        return selected
    return path


def resolve_paths(args: argparse.Namespace, platform: dict[str, str]) -> tuple[Path, Path, Path | None]:
    repo_root = Path(__file__).resolve().parent
    dataset_root = first_existing_path(
        [
            args.dataset,
            platform.get("dataset_path", ""),
            os.getenv("C2NET_DATASET_PATH", ""),
            os.getenv("OPENI_DATASET_PATH", ""),
            repo_root / "data",
        ]
    )
    if dataset_root is None:
        raise FileNotFoundError("No dataset directory found. Pass --dataset or provide platform dataset_path.")
    dataset_root = select_dataset_root(dataset_root)

    output_root = Path(
        args.task_output
        or platform.get("output_path", "")
        or os.getenv("C2NET_OUTPUT_PATH", "")
        or os.getenv("OPENI_OUTPUT_PATH", "")
        or (repo_root / "outputs" / "openi-xrd-training")
    ).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    model_root = first_existing_path(
        [
            args.model_name,
            platform.get("model_path", ""),
            os.getenv("C2NET_PRETRAIN_MODEL_PATH", ""),
            os.getenv("OPENI_PRETRAIN_MODEL_PATH", ""),
        ]
    )
    return dataset_root, output_root, model_root


def spectrum_files(dataset_root: Path, max_files: int) -> list[Path]:
    if dataset_root.is_file() and dataset_root.suffix.lower() in SPECTRUM_EXTENSIONS:
        return [dataset_root]
    files = [
        path
        for path in dataset_root.rglob("*")
        if path.is_file() and path.suffix.lower() in SPECTRUM_EXTENSIONS
    ]
    files.sort()
    if max_files > 0:
        files = files[:max_files]
    return files


def infer_label(path: Path, dataset_root: Path) -> str:
    try:
        rel = path.relative_to(dataset_root)
        if len(rel.parts) > 1:
            return rel.parts[0]
    except ValueError:
        pass
    stem = path.stem
    match = re.match(r"([A-Za-z0-9]+(?:[A-Z][a-z]?[0-9]*)*)", stem)
    return match.group(1) if match else "unknown"


def parse_numeric_pairs(path: Path) -> tuple[np.ndarray, np.ndarray]:
    xs: list[float] = []
    ys: list[float] = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith(("#", "*", ";", "/" + "/")):
                continue
            parts = re.split(r"[\s,;\t]+", line)
            numeric: list[float] = []
            for part in parts:
                try:
                    numeric.append(float(part))
                except ValueError:
                    continue
                if len(numeric) == 2:
                    break
            if len(numeric) == 2:
                xs.append(numeric[0])
                ys.append(numeric[1])
    if len(xs) < 8:
        raise ValueError(f"not enough numeric XRD points in {path}")
    order = np.argsort(np.asarray(xs, dtype=np.float32))
    x = np.asarray(xs, dtype=np.float32)[order]
    y = np.asarray(ys, dtype=np.float32)[order]
    return x, y


def load_samples(dataset_root: Path, max_files: int) -> tuple[list[SpectrumSample], list[dict[str, str]]]:
    samples: list[SpectrumSample] = []
    skipped: list[dict[str, str]] = []
    for path in spectrum_files(dataset_root, max_files):
        try:
            x, y = parse_numeric_pairs(path)
            samples.append(SpectrumSample(path=path, label=infer_label(path, dataset_root), x=x, y=y))
        except Exception as exc:
            skipped.append({"path": str(path), "reason": str(exc)})
    return samples, skipped


def featurize(sample: SpectrumSample, bins: int) -> np.ndarray:
    x_min = float(np.nanmin(sample.x))
    x_max = float(np.nanmax(sample.x))
    grid = np.linspace(x_min, x_max, bins, dtype=np.float32)
    y = np.interp(grid, sample.x, sample.y).astype(np.float32)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    y = y - float(y.min())
    denom = float(y.max())
    if denom > 0:
        y = y / denom
    stats = np.asarray(
        [
            float(sample.x.size),
            x_min,
            x_max,
            float(np.mean(sample.y)),
            float(np.std(sample.y)),
            float(np.max(sample.y)),
        ],
        dtype=np.float32,
    )
    stats = stats / (np.linalg.norm(stats) + 1e-8)
    return np.concatenate([y, stats]).astype(np.float32)


def train_centroid_model(samples: list[SpectrumSample], bins: int, seed: int) -> dict:
    if not samples:
        raise ValueError("no valid spectra found")
    labels = sorted({sample.label for sample in samples})
    label_to_id = {label: idx for idx, label in enumerate(labels)}
    features = np.stack([featurize(sample, bins) for sample in samples])
    y = np.asarray([label_to_id[sample.label] for sample in samples], dtype=np.int64)

    rng = np.random.default_rng(seed)
    indices = np.arange(len(samples))
    rng.shuffle(indices)
    split = max(1, int(len(indices) * 0.8))
    train_idx = indices[:split]
    test_idx = indices[split:] if split < len(indices) else indices[:0]

    centroids = []
    for label_id in range(len(labels)):
        label_features = features[train_idx][y[train_idx] == label_id]
        if len(label_features) == 0:
            label_features = features[y == label_id]
        centroids.append(label_features.mean(axis=0))
    centroid_matrix = np.stack(centroids)

    def predict(batch: np.ndarray) -> np.ndarray:
        distances = ((batch[:, None, :] - centroid_matrix[None, :, :]) ** 2).sum(axis=2)
        return np.argmin(distances, axis=1)

    train_pred = predict(features[train_idx])
    test_pred = predict(features[test_idx]) if len(test_idx) else np.asarray([], dtype=np.int64)
    train_acc = float(np.mean(train_pred == y[train_idx])) if len(train_idx) else 0.0
    test_acc = float(np.mean(test_pred == y[test_idx])) if len(test_idx) else None

    return {
        "labels": labels,
        "label_to_id": label_to_id,
        "centroids": centroid_matrix.tolist(),
        "feature_bins": bins,
        "feature_size": int(features.shape[1]),
        "sample_count": len(samples),
        "train_count": int(len(train_idx)),
        "test_count": int(len(test_idx)),
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
    }


def write_outputs(
    output_root: Path,
    model: dict,
    samples: list[SpectrumSample],
    skipped: list[dict[str, str]],
    args: argparse.Namespace,
    dataset_root: Path,
    model_root: Path | None,
) -> None:
    model_file = output_root / "xrd_centroid_model.json"
    metrics_file = output_root / "metrics.json"
    manifest_file = output_root / "run_manifest.json"
    sample_file = output_root / "samples.csv"

    model_file.write_text(json.dumps(model, ensure_ascii=False, indent=2), encoding="utf-8")
    metrics = {
        "sample_count": model["sample_count"],
        "label_count": len(model["labels"]),
        "labels": model["labels"],
        "train_accuracy": model["train_accuracy"],
        "test_accuracy": model["test_accuracy"],
        "skipped_count": len(skipped),
    }
    metrics_file.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = {
        "entry": "train.py",
            "dataset_root": str(dataset_root),
        "pretrain_model_root": str(model_root) if model_root else "",
        "task_output": str(output_root),
        "hyperparameters": {
            "epochs": args.epochs,
            "bs": args.bs,
            "lr": args.lr,
            "bins": args.bins,
            "max_files": args.max_files,
            "seed": args.seed,
        },
        "artifacts": {
            "model": str(model_file),
            "metrics": str(metrics_file),
            "samples": str(sample_file),
        },
        "skipped": skipped[:50],
    }
    manifest_file.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    with sample_file.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "label", "points"])
        writer.writeheader()
        for sample in samples:
            writer.writerow({"path": str(sample.path), "label": sample.label, "points": int(sample.x.size)})


def upload_platform_output() -> None:
    try:
        from c2net.context import upload_output  # type: ignore

        upload_output()
    except Exception:
        return


def main() -> int:
    args = parse_args()
    platform = prepare_platform_context()
    dataset_root, output_root, model_root = resolve_paths(args, platform)
    samples, skipped = load_samples(dataset_root, args.max_files)
    model = train_centroid_model(samples, args.bins, args.seed)
    write_outputs(output_root, model, samples, skipped, args, dataset_root, model_root)
    upload_platform_output()
    print(
        json.dumps(
            {
                "ok": True,
                "samples": model["sample_count"],
                "labels": model["labels"],
                "output": str(output_root),
                "train_accuracy": model["train_accuracy"],
                "test_accuracy": model["test_accuracy"],
                "skipped": len(skipped),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        raise
