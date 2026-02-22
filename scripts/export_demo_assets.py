#!/usr/bin/env python3
"""Export static demo assets for the Next.js frontend."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _load_metrics(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


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
    scores_rows = _read_jsonl(args.scores_path)
    features_frame = pd.read_csv(args.features_path)
    predictions_frame = pd.read_csv(args.predictions_path)
    metrics = _load_metrics(args.metrics_path)

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
                "plot_refs": {
                    "roc": "/demo/plots/roc.svg",
                    "pr": "/demo/plots/pr.svg",
                    "calib": "/demo/plots/calib.svg",
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
