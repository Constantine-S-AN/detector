#!/usr/bin/env python3
"""Export static demo assets for the Next.js frontend."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from ads.io import read_jsonl


def _load_metrics(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_thresholds(
    predictions_frame: pd.DataFrame, metrics: dict[str, Any]
) -> dict[str, float]:
    metric_thresholds = metrics.get("thresholds", {})
    if isinstance(metric_thresholds, dict):
        decision = metric_thresholds.get("decision_threshold")
        score = metric_thresholds.get("score_threshold")
        floor = metric_thresholds.get("max_score_floor")
        if all(value is not None for value in (decision, score, floor)):
            return {
                "decision_threshold": float(decision),
                "score_threshold": float(score),
                "max_score_floor": float(floor),
            }

    if (
        not predictions_frame.empty
        and "decision_threshold" in predictions_frame
        and "score_threshold" in predictions_frame
        and "max_score_floor" in predictions_frame
    ):
        first = predictions_frame.iloc[0]
        return {
            "decision_threshold": float(first["decision_threshold"]),
            "score_threshold": float(first["score_threshold"]),
            "max_score_floor": float(first["max_score_floor"]),
        }

    return {
        "decision_threshold": 0.5,
        "score_threshold": 0.55,
        "max_score_floor": 0.05,
    }


def _build_summary(predictions_frame: pd.DataFrame) -> dict[str, float | int]:
    """Build compact summary stats for analysis dashboard cards."""
    frame = predictions_frame.copy()
    frame["label_int"] = frame["label_int"].astype(int)
    frame["predicted_label"] = frame["predicted_label"].astype(int)

    faithful_mask = frame["label_int"] == 1
    hallucinated_mask = frame["label_int"] == 0

    true_positive = int(((frame["label_int"] == 1) & (frame["predicted_label"] == 1)).sum())
    true_negative = int(((frame["label_int"] == 0) & (frame["predicted_label"] == 0)).sum())
    false_positive = int(((frame["label_int"] == 0) & (frame["predicted_label"] == 1)).sum())
    false_negative = int(((frame["label_int"] == 1) & (frame["predicted_label"] == 0)).sum())

    return {
        "num_samples": int(frame.shape[0]),
        "num_faithful": int(faithful_mask.sum()),
        "num_hallucinated": int(hallucinated_mask.sum()),
        "mean_score_faithful": float(frame.loc[faithful_mask, "groundedness_score"].mean()),
        "mean_score_hallucinated": float(frame.loc[hallucinated_mask, "groundedness_score"].mean()),
        "tp": true_positive,
        "tn": true_negative,
        "fp": false_positive,
        "fn": false_negative,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores-path", type=Path, default=Path("artifacts/scores.jsonl"))
    parser.add_argument("--features-path", type=Path, default=Path("artifacts/features.csv"))
    parser.add_argument(
        "--predictions-path", type=Path, default=Path("artifacts/predictions_all.csv")
    )
    parser.add_argument("--metrics-path", type=Path, default=Path("artifacts/metrics.json"))
    parser.add_argument("--plots-dir", type=Path, default=Path("artifacts/plots"))
    parser.add_argument("--output-dir", type=Path, default=Path("site/public/demo"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scores_rows = read_jsonl(args.scores_path)
    features_frame = pd.read_csv(args.features_path)
    predictions_frame = pd.read_csv(args.predictions_path)
    metrics = _load_metrics(args.metrics_path)
    thresholds = _resolve_thresholds(predictions_frame, metrics)
    summary = _build_summary(predictions_frame)

    features_map = {
        row["sample_id"]: row.drop(labels=["sample_id", "prompt", "answer", "label"]).to_dict()
        for _, row in features_frame.iterrows()
    }
    prediction_map = {row["sample_id"]: row.to_dict() for _, row in predictions_frame.iterrows()}

    output_dir = args.output_dir
    examples_dir = output_dir / "examples"
    plots_output_dir = output_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    examples_dir.mkdir(parents=True, exist_ok=True)
    plots_output_dir.mkdir(parents=True, exist_ok=True)

    if args.plots_dir.exists():
        for plot_file in args.plots_dir.glob("*"):
            if plot_file.is_file():
                shutil.copy2(plot_file, plots_output_dir / plot_file.name)

    index_rows: list[dict[str, Any]] = []
    for row in scores_rows:
        sample_id = str(row["sample_id"])
        prediction = prediction_map[sample_id]
        detail_payload = {
            "sample_id": sample_id,
            "label": row["label"],
            "prompt": row["prompt"],
            "answer": row["answer"],
            "prediction": {
                "groundedness_score": prediction["groundedness_score"],
                "predicted_label": int(prediction["predicted_label"]),
                "confidence": prediction["confidence"],
                "abstain_flag": bool(prediction["abstain_flag"]),
            },
            "features": features_map[sample_id],
            "top_influential": row["attribution"][:20],
            "plot_refs": {
                "roc": "/demo/plots/roc.svg",
                "pr": "/demo/plots/pr.svg",
                "calib": "/demo/plots/calib.svg",
                "abstain_curve": "/demo/plots/abstain_curve.svg",
                "hist_faithful": "/demo/plots/hist_faithful.svg",
                "hist_hallucinated": "/demo/plots/hist_hallucinated.svg",
            },
        }
        detail_path = examples_dir / f"{sample_id}.json"
        detail_path.write_text(
            json.dumps(detail_payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        index_rows.append(
            {
                "sample_id": sample_id,
                "label": row["label"],
                "prompt_preview": str(row["prompt"])[:96],
                "groundedness_score": float(prediction["groundedness_score"]),
                "predicted_label": int(prediction["predicted_label"]),
                "abstain_flag": bool(prediction["abstain_flag"]),
                "detail_path": f"/demo/examples/{sample_id}.json",
            }
        )

    (output_dir / "index.json").write_text(
        json.dumps(
            {
                "examples": index_rows,
                "count": len(index_rows),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    (output_dir / "analysis.json").write_text(
        json.dumps(
            {
                "metrics": metrics,
                "thresholds": thresholds,
                "summary": summary,
                "plot_refs": {
                    "roc": "/demo/plots/roc.svg",
                    "pr": "/demo/plots/pr.svg",
                    "calib": "/demo/plots/calib.svg",
                    "abstain_curve": "/demo/plots/abstain_curve.svg",
                    "hist_faithful": "/demo/plots/hist_faithful.svg",
                    "hist_hallucinated": "/demo/plots/hist_hallucinated.svg",
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
