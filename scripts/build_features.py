#!/usr/bin/env python3
"""Convert attribution outputs into model-ready features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from ads.detector.logistic import encode_label
from ads.features.density import compute_density_features, compute_h_at_k


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores-path", type=Path, default=Path("artifacts/scores.jsonl"))
    parser.add_argument("--output-path", type=Path, default=Path("artifacts/features.csv"))
    parser.add_argument("--max-score-floor", type=float, default=0.05)
    parser.add_argument(
        "--h-k",
        type=str,
        default="20",
        help="Comma-separated K list for entropy features, e.g. '5,10,20'",
    )
    parser.add_argument(
        "--h-weight-mode",
        type=str,
        choices=("shifted", "softmax"),
        default="shifted",
    )
    return parser.parse_args()


def _parse_h_k_values(raw: str) -> list[int]:
    values = [chunk.strip() for chunk in raw.split(",") if chunk.strip()]
    if not values:
        raise ValueError("--h-k must include at least one integer")
    parsed = [int(item) for item in values]
    if any(item <= 0 for item in parsed):
        raise ValueError("All --h-k values must be positive integers")
    return parsed


def _iter_raw_records(path: Path) -> list[tuple[int, Any]]:
    records: list[tuple[int, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_index, line in enumerate(handle):
            stripped = line.strip()
            if not stripped:
                continue
            records.append((line_index, json.loads(stripped)))
    return records


def _extract_items(raw_record: Any) -> list[dict[str, Any]]:
    if isinstance(raw_record, list):
        items = raw_record
    elif isinstance(raw_record, dict):
        if isinstance(raw_record.get("items"), list):
            items = raw_record["items"]
        elif isinstance(raw_record.get("attribution"), list):
            items = raw_record["attribution"]
        else:
            raise KeyError("Attribution payload missing both 'items' and legacy 'attribution'")
    else:
        raise TypeError(f"Unsupported attribution row type: {type(raw_record)!r}")

    parsed_items: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            raise TypeError("Each attribution item must be a JSON object")
        parsed_items.append(item)
    return parsed_items


def _extract_item_rank(item: dict[str, Any]) -> float | None:
    raw_rank: Any | None = item.get("rank")
    if raw_rank is None and isinstance(item.get("meta"), dict):
        raw_rank = item["meta"].get("rank")
    if raw_rank is None:
        return None
    try:
        return float(raw_rank)
    except (TypeError, ValueError):
        return None


def _extract_sorted_scores(items: list[dict[str, Any]]) -> list[float]:
    parsed_rows: list[tuple[float, float | None, int]] = []
    for index, item in enumerate(items):
        if "score" not in item:
            raise KeyError("Attribution item missing required 'score' field")
        score = float(item["score"])
        rank = _extract_item_rank(item)
        parsed_rows.append((score, rank, index))

    has_rank = any(rank is not None for _, rank, _ in parsed_rows)
    if has_rank:
        parsed_rows.sort(
            key=lambda entry: (
                entry[1] is None,
                entry[1] if entry[1] is not None else float("inf"),
                -entry[0],
                entry[2],
            )
        )
    else:
        parsed_rows.sort(key=lambda entry: (-entry[0], entry[2]))
    return [score for score, _, _ in parsed_rows]


def _extract_attribution_mode(raw_record: Any, items: list[dict[str, Any]]) -> str:
    candidate_values: list[Any] = []
    if isinstance(raw_record, dict):
        candidate_values.append(raw_record.get("attribution_mode"))
        sample_meta = raw_record.get("sample_meta")
        if isinstance(sample_meta, dict):
            candidate_values.append(sample_meta.get("attribution_mode"))
        record_meta = raw_record.get("meta")
        if isinstance(record_meta, dict):
            candidate_values.append(record_meta.get("attribution_mode"))
    if items:
        item_meta = items[0].get("meta")
        if isinstance(item_meta, dict):
            candidate_values.append(item_meta.get("mode"))

    for value in candidate_values:
        if value is None:
            continue
        normalized = str(value).strip()
        if normalized:
            return normalized
    return "unknown"


def load_attribution_record(
    raw_record: Any,
    *,
    line_index: int,
    source_path: Path,
) -> dict[str, Any]:
    """Load one attribution row from mixed schema variants."""
    items = _extract_items(raw_record)
    scores = _extract_sorted_scores(items)
    fallback_sample_id = f"{source_path.stem}-{line_index:06d}"

    if isinstance(raw_record, dict):
        sample_id_raw = raw_record.get("sample_id")
        prompt_raw = raw_record.get("prompt", "")
        answer_raw = raw_record.get("answer", "")
        label_raw = raw_record.get("label", "unknown")
    else:
        sample_id_raw = None
        prompt_raw = ""
        answer_raw = ""
        label_raw = "unknown"

    sample_id = (
        str(sample_id_raw).strip() if sample_id_raw is not None and str(sample_id_raw).strip() else fallback_sample_id
    )
    prompt = str(prompt_raw)
    answer = str(answer_raw)
    label = str(label_raw)
    attribution_mode = _extract_attribution_mode(raw_record, items)
    return {
        "sample_id": sample_id,
        "prompt": prompt,
        "answer": answer,
        "label": label,
        "label_int": encode_label(label),
        "attribution_mode": attribution_mode,
        "scores": scores,
    }


def main() -> None:
    args = parse_args()
    h_k_values = _parse_h_k_values(args.h_k)
    primary_h_k = h_k_values[0]
    h_weight_mode: Literal["shifted", "softmax"] = args.h_weight_mode

    records: list[dict[str, object]] = []
    raw_records = _iter_raw_records(args.scores_path)
    for line_index, raw_record in raw_records:
        parsed_record = load_attribution_record(
            raw_record,
            line_index=line_index,
            source_path=args.scores_path,
        )
        scores = parsed_record["scores"]
        if not isinstance(scores, list):
            raise TypeError("Internal error: parsed scores must be a list")
        features = compute_density_features(
            scores=scores,
            max_score_floor=args.max_score_floor,
            h_k=primary_h_k,
            h_weight_mode=h_weight_mode,
        )
        feature_dict = features.to_dict()
        for h_k in h_k_values:
            h_payload = compute_h_at_k(
                scores=scores,
                k=h_k,
                normalize=True,
                weight_mode=h_weight_mode,
            )
            feature_dict[f"h_at_k_{h_k}"] = float(h_payload["h_at_k"])
            feature_dict[f"h_at_k_norm_{h_k}"] = float(h_payload["h_at_k_normalized"])

        # Backward-compatible alias for legacy pipelines.
        feature_dict["entropy"] = float(feature_dict["entropy_top_k"])
        records.append(
            {
                "sample_id": parsed_record["sample_id"],
                "prompt": parsed_record["prompt"],
                "answer": parsed_record["answer"],
                "label": parsed_record["label"],
                "label_int": parsed_record["label_int"],
                "attribution_mode": parsed_record["attribution_mode"],
                **feature_dict,
            }
        )

    frame = pd.DataFrame.from_records(records)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_path, index=False)


if __name__ == "__main__":
    main()
