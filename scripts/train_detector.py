#!/usr/bin/env python3
"""Train logistic groundedness detector from extracted features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from ads.detector.logistic import LogisticDetector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-path", type=Path, default=Path("artifacts/features.csv"))
    parser.add_argument("--model-path", type=Path, default=Path("artifacts/models/logistic.joblib"))
    parser.add_argument("--split-path", type=Path, default=Path("artifacts/data/splits.json"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--decision-threshold", type=float, default=0.5)
    parser.add_argument("--score-threshold", type=float, default=None)
    parser.add_argument("--max-score-floor", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.features_path)

    train_frame, test_frame = train_test_split(
        frame,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=frame["label_int"],
    )

    detector = LogisticDetector(random_state=args.seed)
    detector.fit(train_frame, label_column="label_int")
    detector.save(args.model_path)

    thresholds: dict[str, float] = {"decision_threshold": args.decision_threshold}
    if args.score_threshold is not None:
        thresholds["score_threshold"] = args.score_threshold
    if args.max_score_floor is not None:
        thresholds["max_score_floor"] = args.max_score_floor

    split_payload = {
        "seed": args.seed,
        "test_size": args.test_size,
        "decision_threshold": args.decision_threshold,
        "thresholds": thresholds,
        "train_ids": train_frame["sample_id"].tolist(),
        "test_ids": test_frame["sample_id"].tolist(),
    }
    args.split_path.parent.mkdir(parents=True, exist_ok=True)
    args.split_path.write_text(
        json.dumps(split_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
