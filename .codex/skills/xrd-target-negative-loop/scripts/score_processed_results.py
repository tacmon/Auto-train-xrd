#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path


def normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (text or "").lower())


def infer_weak_label(filename: str, known_formulas: list[str]) -> str | None:
    norm_name = normalize(filename)
    matches = []
    for formula in known_formulas:
        token = normalize(formula)
        if token and token in norm_name:
            matches.append(formula)
    unique = sorted(set(matches))
    if len(unique) == 1:
        return unique[0]
    return None


def predicted_positive(predicted_phase: str, target_formula: str, positive_tags: list[str]) -> bool:
    if not predicted_phase or predicted_phase == "未识别":
        return False
    predicted_norm = normalize(predicted_phase)
    if positive_tags:
        return predicted_phase in positive_tags
    target_norm = normalize(target_formula)
    return predicted_norm == target_norm or predicted_norm.startswith(target_norm) or target_norm in predicted_norm


def safe_div(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def rounded(value: float | None) -> float | None:
    return None if value is None else round(value, 6)


def main() -> int:
    parser = argparse.ArgumentParser(description="Score processed_result.csv against weak labels inferred from filenames.")
    parser.add_argument("--input", required=True, help="Path to processed_result.csv")
    parser.add_argument("--target-formula", required=True, help="Target formula A")
    parser.add_argument(
        "--known-formula",
        action="append",
        default=[],
        help="Known formula that may appear in filenames or directories. Repeatable.",
    )
    parser.add_argument(
        "--positive-tag",
        action="append",
        default=[],
        help="Explicit predicted tag to treat as target-positive. Repeatable.",
    )
    parser.add_argument("--output-json", help="Optional path to write score.json")
    parser.add_argument("--min-labeled-rows", type=int, default=10)
    parser.add_argument("--min-coverage", type=float, default=0.10)
    parser.add_argument("--min-precision", type=float, default=0.85)
    parser.add_argument("--min-recall", type=float, default=0.70)
    parser.add_argument("--min-f1", type=float, default=0.75)
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[ERROR] Input file does not exist: {input_path}", file=sys.stderr)
        return 1

    known_formulas = []
    seen = set()
    for formula in [args.target_formula, *args.known_formula]:
        key = normalize(formula)
        if key and key not in seen:
            known_formulas.append(formula)
            seen.add(key)

    with input_path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))

    tp = fp = tn = fn = 0
    target_predictions = 0
    unidentified = 0
    labeled_rows = 0
    ambiguous_or_unknown = 0
    row_details = []

    for row in rows:
        filename = row.get("Filename", "")
        predicted_phase = row.get("Predicted phases", "")
        is_pred_positive = predicted_positive(predicted_phase, args.target_formula, args.positive_tag)
        weak_label = infer_weak_label(filename, known_formulas)
        is_unidentified = predicted_phase == "未识别"

        if is_pred_positive:
            target_predictions += 1
        if is_unidentified:
            unidentified += 1

        if weak_label is None:
            ambiguous_or_unknown += 1
            row_details.append(
                {
                    "filename": filename,
                    "weak_label": None,
                    "predicted_phase": predicted_phase,
                    "counted": False,
                }
            )
            continue

        labeled_rows += 1
        truth_positive = normalize(weak_label) == normalize(args.target_formula)

        if truth_positive and is_pred_positive:
            tp += 1
        elif not truth_positive and is_pred_positive:
            fp += 1
        elif truth_positive and not is_pred_positive:
            fn += 1
        else:
            tn += 1

        row_details.append(
            {
                "filename": filename,
                "weak_label": weak_label,
                "predicted_phase": predicted_phase,
                "counted": True,
            }
        )

    total_rows = len(rows)
    coverage = safe_div(labeled_rows, total_rows)
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    if precision is None or recall is None or precision + recall == 0:
        f1 = None
    else:
        f1 = 2 * precision * recall / (precision + recall)

    reliable = labeled_rows >= args.min_labeled_rows and (coverage or 0.0) >= args.min_coverage
    if reliable:
        evaluation_mode = "weak_labels"
        passed = (
            (precision or 0.0) >= args.min_precision
            and (recall or 0.0) >= args.min_recall
            and (f1 or 0.0) >= args.min_f1
        )
        decision = "pass" if passed else "retry"
    else:
        evaluation_mode = "proxy_only"
        passed = None
        decision = "insufficient_labels"

    payload = {
        "target_formula": args.target_formula,
        "known_formulas": known_formulas,
        "positive_tags": args.positive_tag,
        "evaluation_mode": evaluation_mode,
        "decision": decision,
        "passed": passed,
        "thresholds": {
            "min_labeled_rows": args.min_labeled_rows,
            "min_coverage": args.min_coverage,
            "min_precision": args.min_precision,
            "min_recall": args.min_recall,
            "min_f1": args.min_f1,
        },
        "counts": {
            "total_rows": total_rows,
            "labeled_rows": labeled_rows,
            "unknown_or_ambiguous_rows": ambiguous_or_unknown,
            "target_predictions": target_predictions,
            "unidentified_rows": unidentified,
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
        },
        "metrics": {
            "coverage": rounded(coverage),
            "precision": rounded(precision),
            "recall": rounded(recall),
            "f1": rounded(f1),
            "target_prediction_ratio": rounded(safe_div(target_predictions, total_rows)),
            "unidentified_ratio": rounded(safe_div(unidentified, total_rows)),
        },
        "rows": row_details,
    }

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
