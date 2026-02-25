#!/usr/bin/env python3
"""Convert attribution outputs into model-ready features."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal

import pandas as pd

from ads.detector.logistic import encode_label
from ads.features.density import compute_density_features, compute_h_at_k
from ads.io import iter_jsonl


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


def main() -> None:
    args = parse_args()
    h_k_values = _parse_h_k_values(args.h_k)
    primary_h_k = h_k_values[0]
    h_weight_mode: Literal["shifted", "softmax"] = args.h_weight_mode

    records: list[dict[str, object]] = []
    for row in iter_jsonl(args.scores_path):
        attribution = row["attribution"]
        if not isinstance(attribution, list):
            raise TypeError("Invalid attribution payload, expected list")
        scores = [float(item["score"]) for item in attribution]
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
