#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import csv
import re
import sys
from pathlib import Path


def parse_list_string(raw: str) -> list[object]:
    if raw is None:
        return []
    text = str(raw).strip()
    if not text:
        return []
    cleaned = re.sub(r"np\.float64\s*\(\s*(.*?)\s*\)", r"\1", text)
    try:
        value = ast.literal_eval(cleaned)
    except (SyntaxError, ValueError):
        return []
    return value if isinstance(value, list) else []


def normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (text or "").lower())


def tags_from_result(rows: list[dict[str, str]]) -> list[str]:
    tags: set[str] = set()
    for row in rows:
        for phase in parse_list_string(row.get("Predicted phases", "")):
            if isinstance(phase, str) and phase:
                tags.add(phase)
    return sorted(tags)


def resolve_main_tags(all_tags: list[str], target_formula: str, explicit_tags: list[str]) -> list[str]:
    if explicit_tags:
        return explicit_tags

    target_norm = normalize(target_formula)
    matches = []
    for tag in all_tags:
        tag_norm = normalize(tag)
        if not tag_norm:
            continue
        if tag_norm == target_norm or tag_norm.startswith(target_norm) or target_norm in tag_norm:
            matches.append(tag)
    return matches


def process_rows(
    rows: list[dict[str, str]],
    main_tags: set[str],
    threshold: float,
) -> list[dict[str, object]]:
    processed = []
    for row in rows:
        phases = parse_list_string(row.get("Predicted phases", ""))
        confidences = parse_list_string(row.get("Confidence", ""))

        final_phase = "未识别"
        final_confidence: float | str = ""

        winner_phase = None
        winner_conf = float("-inf")
        for phase, confidence in zip(phases, confidences):
            if not isinstance(phase, str):
                continue
            try:
                numeric_conf = float(confidence)
            except (TypeError, ValueError):
                continue
            if numeric_conf > winner_conf:
                winner_conf = numeric_conf
                winner_phase = phase

        if winner_phase in main_tags and winner_conf > threshold:
            final_phase = winner_phase
            final_confidence = round(winner_conf, 4)

        processed.append(
            {
                "Filename": row.get("Filename", ""),
                "Predicted phases": final_phase,
                "Confidence": final_confidence,
            }
        )
    return processed


def main() -> int:
    parser = argparse.ArgumentParser(description="Non-interactively post-process result.csv for a target formula.")
    parser.add_argument("--input", required=True, help="Path to result.csv")
    parser.add_argument("--output", required=True, help="Path to processed_result.csv")
    parser.add_argument("--target-formula", required=True, help="Target chemical formula A")
    parser.add_argument(
        "--main-tag",
        action="append",
        default=[],
        help="Explicit tag to treat as the main substance. Repeatable.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=50.0,
        help="Minimum confidence required for the winner tag. Default: 50.0",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    if not input_path.exists():
        print(f"[ERROR] Input file does not exist: {input_path}", file=sys.stderr)
        return 1

    with input_path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))

    all_tags = tags_from_result(rows)
    main_tags = resolve_main_tags(all_tags, args.target_formula, args.main_tag)
    if not main_tags:
        print(
            "[ERROR] Could not resolve any main tags. "
            "Pass --main-tag explicitly or check whether result.csv contains the target tag.",
            file=sys.stderr,
        )
        if all_tags:
            print(f"[INFO] Available tags: {', '.join(all_tags)}", file=sys.stderr)
        return 1

    processed = process_rows(rows, set(main_tags), args.confidence_threshold)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=["Filename", "Predicted phases", "Confidence"])
        writer.writeheader()
        writer.writerows(processed)

    identified = sum(1 for row in processed if row["Predicted phases"] != "未识别")
    print(f"main_tags={','.join(main_tags)}")
    print(f"rows={len(processed)}")
    print(f"identified_rows={identified}")
    print(f"output={output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
