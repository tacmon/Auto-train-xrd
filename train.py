#!/usr/bin/env python3
"""OpenI/MMP-compatible single-run XRD training entry."""

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


SPECTRUM_EXTENSIONS = {".txt", ".xy", ".gk", ".xrd", ".csv"}


@dataclass
class SpectrumSample:
    path: Path
    label: str
    x: np.ndarray
    y: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a lightweight XRD classifier.")
    parser.add_argument("--dataset", default="", help="Dataset path injected by platform or local path.")
    parser.add_argument("--model_name", default="", help="Optional pretrained model path injected by platform.")
    parser.add_argument("--task_output", default="/result", help="Output path injected by platform.")
    parser.add_argument("--epochs", type=int, default=1, help="Compatibility hyperparameter.")
    parser.add_argument("--bs", type=int, default=16, help="Compatibility hyperparameter.")
    parser.add_argument("--lr", type=float, default=0.001, help="Compatibility hyperparameter.")
    parser.add_argument("--bins", type=int, default=256, help="Number of interpolated feature bins.")
    parser.add_argument("--max_files", type=int, default=0, help="Limit spectra; 0 means all.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for deterministic split.")
    parser.add_argument("--target_formula", default="", help="Target formula/label A for negative-loop scoring.")
    parser.add_argument("--candidate_formula", default="", help="Candidate negative formula/label for this run.")
    parser.add_argument("--known_formula", action="append", default=[], help="Known measured-data weak label. Repeatable.")
    parser.add_argument("--confidence_threshold", type=float, default=50.0, help="Target postprocess confidence threshold.")
    args, unknown = parser.parse_known_args()
    if unknown:
        print(json.dumps({"event": "ignored_platform_args", "args": unknown}, ensure_ascii=False), flush=True)
    return args


def expand_known_formulas(values: list[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        normalized = str(value).replace("+", ",")
        for item in normalized.split(","):
            item = item.strip()
            if item and item not in out:
                out.append(item)
    return out


def prepare_platform_context() -> dict[str, str]:
    try:
        from c2net.context import prepare  # type: ignore

        ctx = prepare()
        return {
            "dataset_path": getattr(ctx, "dataset_path", "") or "",
            "model_path": getattr(ctx, "pretrain_model_path", "") or "",
            "output_path": getattr(ctx, "output_path", "") or "",
        }
    except Exception as exc:
        print(json.dumps({"event": "c2net_prepare_unavailable", "reason": str(exc)}, ensure_ascii=False), flush=True)
        return {"dataset_path": "", "model_path": "", "output_path": ""}


def first_existing_path(candidates: Iterable[str | Path]) -> Path | None:
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate).expanduser().resolve()
        if path.exists():
            return path
    return None


def count_spectrum_files(path: Path) -> int:
    if path.is_file():
        return int(path.suffix.lower() in SPECTRUM_EXTENSIONS)
    return sum(1 for p in path.rglob("*") if p.is_file() and p.suffix.lower() in SPECTRUM_EXTENSIONS)


def extract_first_zip(path: Path) -> Path | None:
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
    if count_spectrum_files(path) > 0:
        return path

    candidates: list[Path] = []
    extracted = extract_first_zip(path)
    if extracted:
        candidates.append(extracted)
        candidates.extend(p for p in extracted.rglob("*") if p.is_dir())
    if path.is_dir():
        candidates.extend(p for p in path.rglob("*") if p.is_dir())

    ranked = sorted(
        ((candidate, count_spectrum_files(candidate)) for candidate in candidates),
        key=lambda item: (item[1], -len(item[0].parts)),
        reverse=True,
    )
    if ranked and ranked[0][1] > 0:
        print(
            json.dumps(
                {"event": "selected_dataset_root", "input": str(path), "selected": str(ranked[0][0]), "spectra": ranked[0][1]},
                ensure_ascii=False,
            ),
            flush=True,
        )
        return ranked[0][0]
    return path


def resolve_paths(args: argparse.Namespace, platform: dict[str, str]) -> tuple[Path, Path, Path | None]:
    repo_root = Path(__file__).resolve().parent
    dataset_root = first_existing_path(
        [
            args.dataset,
            platform.get("dataset_path", ""),
            os.getenv("C2NET_DATASET_PATH", ""),
            os.getenv("OPENI_DATASET_PATH", ""),
            repo_root / "data" / "demo_xrd",
            repo_root / "data",
        ]
    )
    if dataset_root is None:
        raise FileNotFoundError("No dataset found. Pass --dataset or provide a platform-mounted dataset.")
    dataset_root = select_dataset_root(dataset_root)

    platform_output = platform.get("output_path", "") or os.getenv("C2NET_OUTPUT_PATH", "") or os.getenv("OPENI_OUTPUT_PATH", "")
    output_root = Path(platform_output or args.task_output or (repo_root / "outputs")).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    model_root = first_existing_path(
        [args.model_name, platform.get("model_path", ""), os.getenv("C2NET_PRETRAIN_MODEL_PATH", ""), os.getenv("OPENI_PRETRAIN_MODEL_PATH", "")]
    )
    return dataset_root, output_root, model_root


def spectrum_files(dataset_root: Path, max_files: int) -> list[Path]:
    if dataset_root.is_file() and dataset_root.suffix.lower() in SPECTRUM_EXTENSIONS:
        files = [dataset_root]
    else:
        files = [p for p in dataset_root.rglob("*") if p.is_file() and p.suffix.lower() in SPECTRUM_EXTENSIONS]
    files.sort()
    return files[:max_files] if max_files > 0 else files


def infer_label(path: Path, dataset_root: Path) -> str:
    try:
        rel = path.relative_to(dataset_root)
        if len(rel.parts) > 1:
            return rel.parts[0]
    except ValueError:
        pass
    match = re.search(r"([A-Za-z]+[0-9]+)", path.stem)
    return match.group(1) if match else "unknown"


def parse_numeric_pairs(path: Path) -> tuple[np.ndarray, np.ndarray]:
    xs: list[float] = []
    ys: list[float] = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith(("#", "*", ";", "//")):
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
    return np.asarray(xs, dtype=np.float32)[order], np.asarray(ys, dtype=np.float32)[order]


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


def split_train_measured_roots(dataset_root: Path) -> tuple[Path, Path]:
    train_root = dataset_root / "train"
    measured_root = dataset_root / "measured"
    if train_root.is_dir() and measured_root.is_dir():
        return train_root, measured_root
    return dataset_root, dataset_root


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
        [float(sample.x.size), x_min, x_max, float(np.mean(sample.y)), float(np.std(sample.y)), float(np.max(sample.y))],
        dtype=np.float32,
    )
    stats = stats / (np.linalg.norm(stats) + 1e-8)
    return np.concatenate([y, stats]).astype(np.float32)


def fit_centroids(
    features: np.ndarray,
    targets: np.ndarray,
    train_idx: np.ndarray,
    labels: list[str],
) -> np.ndarray:
    centroids = []
    for label_id in range(len(labels)):
        label_features = features[train_idx][targets[train_idx] == label_id]
        if len(label_features) == 0:
            label_features = features[targets == label_id]
        centroids.append(label_features.mean(axis=0))
    return np.stack(centroids)


def predict_with_confidence(batch: np.ndarray, centroid_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    distances = ((batch[:, None, :] - centroid_matrix[None, :, :]) ** 2).sum(axis=2)
    pred = np.argmin(distances, axis=1)
    if centroid_matrix.shape[0] == 1:
        confidence = np.full(batch.shape[0], 100.0, dtype=np.float32)
    else:
        similarity = 1.0 / (distances + 1e-8)
        confidence = similarity.max(axis=1) / similarity.sum(axis=1) * 100.0
    return pred, confidence.astype(np.float32)


def train_centroid_model(samples: list[SpectrumSample], bins: int, seed: int) -> dict:
    if not samples:
        raise ValueError("no valid spectra found")
    labels = sorted({sample.label for sample in samples})
    label_to_id = {label: idx for idx, label in enumerate(labels)}
    features = np.stack([featurize(sample, bins) for sample in samples])
    targets = np.asarray([label_to_id[sample.label] for sample in samples], dtype=np.int64)

    rng = np.random.default_rng(seed)
    indices = np.arange(len(samples))
    rng.shuffle(indices)
    split = max(1, int(len(indices) * 0.8))
    train_idx = indices[:split]
    test_idx = indices[split:] if split < len(indices) else indices[:0]

    centroid_matrix = fit_centroids(features, targets, train_idx, labels)

    train_pred, _ = predict_with_confidence(features[train_idx], centroid_matrix)
    test_pred, _ = predict_with_confidence(features[test_idx], centroid_matrix) if len(test_idx) else (np.asarray([], dtype=np.int64), np.asarray([], dtype=np.float32))
    return {
        "labels": labels,
        "label_to_id": label_to_id,
        "centroids": centroid_matrix.tolist(),
        "feature_bins": bins,
        "feature_size": int(features.shape[1]),
        "sample_count": len(samples),
        "train_count": int(len(train_idx)),
        "test_count": int(len(test_idx)),
        "train_accuracy": float(np.mean(train_pred == targets[train_idx])) if len(train_idx) else 0.0,
        "test_accuracy": float(np.mean(test_pred == targets[test_idx])) if len(test_idx) else None,
    }


def selected_training_samples(samples: list[SpectrumSample], target: str, candidate: str) -> list[SpectrumSample]:
    wanted = {x for x in [target, candidate] if x}
    if len(wanted) < 2:
        return samples
    selected = [sample for sample in samples if sample.label in wanted]
    if len({sample.label for sample in selected}) < 2:
        raise ValueError(f"candidate run requires both target and candidate labels, got labels={sorted({sample.label for sample in selected})}")
    return selected


def score_predictions(rows: list[dict[str, object]], target: str, candidate: str, known_formulas: list[str]) -> dict:
    tp = fp = tn = fn = 0
    labeled_rows = 0
    target_predictions = 0
    unidentified = 0
    known = set(known_formulas or [target, candidate])
    known.add(target)
    for row in rows:
        truth = str(row["truth"])
        pred = str(row["processed_prediction"])
        is_target_truth = truth == target
        is_target_pred = pred == target
        if pred == "未识别":
            unidentified += 1
        if is_target_pred:
            target_predictions += 1
        if truth not in known:
            continue
        labeled_rows += 1
        if is_target_truth and is_target_pred:
            tp += 1
        elif not is_target_truth and is_target_pred:
            fp += 1
        elif is_target_truth and not is_target_pred:
            fn += 1
        else:
            tn += 1

    def div(num: int, den: int) -> float | None:
        return None if den == 0 else round(num / den, 6)

    precision = div(tp, tp + fp)
    recall = div(tp, tp + fn)
    f1 = None if precision is None or recall is None or precision + recall == 0 else round(2 * precision * recall / (precision + recall), 6)
    return {
        "target_formula": target,
        "candidate_formula": candidate,
        "known_formulas": sorted(known),
        "evaluation_mode": "weak_labels",
        "decision": "pass" if (precision or 0) >= 0.85 and (recall or 0) >= 0.70 and (f1 or 0) >= 0.75 else "retry",
        "counts": {
            "total_rows": len(rows),
            "labeled_rows": labeled_rows,
            "target_predictions": target_predictions,
            "unidentified_rows": unidentified,
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
        },
        "metrics": {
            "coverage": div(labeled_rows, len(rows)),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "target_prediction_ratio": div(target_predictions, len(rows)),
            "unidentified_ratio": div(unidentified, len(rows)),
        },
    }


def write_negative_loop_outputs(
    output_root: Path,
    measured_samples: list[SpectrumSample],
    model: dict,
    args: argparse.Namespace,
) -> dict | None:
    target = (args.target_formula or "").strip()
    candidate = (args.candidate_formula or "").strip()
    if not target or not candidate:
        return None

    centroid_matrix = np.asarray(model["centroids"], dtype=np.float32)
    rows: list[dict[str, object]] = []
    for sample in measured_samples:
        feat = featurize(sample, int(model["feature_bins"]))[None, :]
        pred_ids, confidences = predict_with_confidence(feat, centroid_matrix)
        pred_label = model["labels"][int(pred_ids[0])]
        confidence = float(confidences[0])
        processed = pred_label if pred_label == target and confidence > args.confidence_threshold else "未识别"
        rows.append(
            {
                "filename": str(sample.path),
                "truth": sample.label,
                "predicted": pred_label,
                "confidence": round(confidence, 4),
                "processed_prediction": processed,
            }
        )

    result_csv = output_root / "result.csv"
    with result_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["Filename", "Predicted phases", "Confidence", "Truth"])
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "Filename": row["filename"],
                    "Predicted phases": [row["predicted"]],
                    "Confidence": [row["confidence"]],
                    "Truth": row["truth"],
                }
            )

    processed_csv = output_root / "processed_result.csv"
    with processed_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["Filename", "Predicted phases", "Confidence", "Truth"])
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "Filename": row["filename"],
                    "Predicted phases": row["processed_prediction"],
                    "Confidence": row["confidence"] if row["processed_prediction"] != "未识别" else "",
                    "Truth": row["truth"],
                }
            )

    score = score_predictions(rows, target, candidate, expand_known_formulas(args.known_formula))
    (output_root / "score.json").write_text(json.dumps(score, ensure_ascii=False, indent=2), encoding="utf-8")
    return score


def write_outputs(
    output_root: Path,
    model: dict,
    train_samples: list[SpectrumSample],
    measured_samples: list[SpectrumSample],
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
        "target_formula": args.target_formula,
        "candidate_formula": args.candidate_formula,
    }
    metrics_file.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    negative_loop_score = write_negative_loop_outputs(output_root, measured_samples, model, args)
    if negative_loop_score is not None:
        metrics["negative_loop"] = negative_loop_score
        metrics_file.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = {
        "entry": "train.py",
        "dataset_root": str(dataset_root),
        "train_sample_count": len(train_samples),
        "measured_sample_count": len(measured_samples),
        "pretrain_model_root": str(model_root) if model_root else "",
        "task_output": str(output_root),
        "hyperparameters": {
            "epochs": args.epochs,
            "bs": args.bs,
            "lr": args.lr,
            "bins": args.bins,
            "max_files": args.max_files,
            "seed": args.seed,
            "target_formula": args.target_formula,
            "candidate_formula": args.candidate_formula,
            "known_formula": expand_known_formulas(args.known_formula),
            "confidence_threshold": args.confidence_threshold,
        },
        "artifacts": {
            "model": str(model_file),
            "metrics": str(metrics_file),
            "samples": str(sample_file),
            "result_csv": str(output_root / "result.csv"),
            "processed_result_csv": str(output_root / "processed_result.csv"),
            "score_json": str(output_root / "score.json"),
        },
        "skipped": skipped[:50],
    }
    manifest_file.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    with sample_file.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "label", "points"])
        writer.writeheader()
        for sample in train_samples:
            writer.writerow({"path": str(sample.path), "label": sample.label, "points": int(sample.x.size)})


def mirror_task_output(output_root: Path, requested_output: str) -> None:
    target = Path((requested_output or "").strip()).expanduser().resolve() if (requested_output or "").strip() else output_root
    if target == output_root:
        return
    try:
        target.mkdir(parents=True, exist_ok=True)
        for item in output_root.iterdir():
            if item.is_file():
                shutil.copy2(item, target / item.name)
    except Exception as exc:
        print(json.dumps({"event": "mirror_task_output_failed", "error": str(exc)}, ensure_ascii=False), flush=True)


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
    train_root, measured_root = split_train_measured_roots(dataset_root)
    samples, skipped = load_samples(train_root, args.max_files)
    measured_samples, measured_skipped = load_samples(measured_root, 0)
    skipped.extend(measured_skipped)
    train_samples = selected_training_samples(samples, args.target_formula.strip(), args.candidate_formula.strip())
    measured_known = set(expand_known_formulas(args.known_formula) or [args.target_formula.strip(), args.candidate_formula.strip()])
    measured_known.add(args.target_formula.strip())
    measured_selected = [sample for sample in measured_samples if sample.label in measured_known]
    model = train_centroid_model(train_samples, args.bins, args.seed)
    write_outputs(output_root, model, train_samples, measured_selected, skipped, args, dataset_root, model_root)
    mirror_task_output(output_root, args.task_output)
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
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        raise
