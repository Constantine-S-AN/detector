#!/usr/bin/env python3
"""Convert attribution outputs into model-ready features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from ads.detector.logistic import encode_label
from ads.features.density import compute_density_features


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores-path", type=Path, default=Path("artifacts/scores.jsonl"))
    parser.add_argument("--output-path", type=Path, default=Path("artifacts/features.csv"))
    parser.add_argument("--max-score-floor", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = _read_jsonl(args.scores_path)

    records: list[dict[str, object]] = []
    for row in rows:
        attribution = row["attribution"]
        if not isinstance(attribution, list):
            raise TypeError("Invalid attribution payload, expected list")
        scores = [float(item["score"]) for item in attribution]
        features = compute_density_features(scores=scores, max_score_floor=args.max_score_floor)
        feature_dict = features.to_dict()
        records.append(
            {
                "sample_id": row["sample_id"],
                "prompt": row["prompt"],
                "answer": row["answer"],
                "label": row["label"],
                "label_int": encode_label(str(row["label"])),
                **feature_dict,
            }
        )

    frame = pd.DataFrame.from_records(records)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_path, index=False)


if __name__ == "__main__":
    main()
