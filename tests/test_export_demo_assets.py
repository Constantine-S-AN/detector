"""Tests for demo asset export script behaviors."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def _write_scores(path: Path) -> None:
    row = {
        "sample_id": "sample-000",
        "prompt": "p",
        "answer": "a",
        "label": "faithful",
        "attribution": [
            {
                "train_id": "train-1",
                "score": 1.0,
                "text": "t1",
                "meta": {"rank": 1, "mode": "peaked"},
            }
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")


def test_export_demo_assets_uses_output_dir_public_base(tmp_path: Path) -> None:
    artifacts_dir = tmp_path / "artifacts"
    scores_path = artifacts_dir / "scores.jsonl"
    features_path = artifacts_dir / "features.csv"
    predictions_path = artifacts_dir / "predictions_all.csv"
    metrics_path = artifacts_dir / "metrics.json"
    plots_dir = artifacts_dir / "plots"
    output_dir = tmp_path / "site" / "public" / "demo-stress"

    _write_scores(scores_path)
    pd.DataFrame(
        [
            {
                "sample_id": "sample-000",
                "prompt": "p",
                "answer": "a",
                "label": "faithful",
                "entropy_top_k": 0.1,
                "top1_share": 0.9,
                "top5_share": 1.0,
                "peakiness_ratio": 0.9,
                "gini": 0.8,
                "max_score": 1.0,
                "effective_k": 1.2,
                "abstain_flag": False,
                "top_k": 1,
            }
        ]
    ).to_csv(features_path, index=False)
    pd.DataFrame(
        [
            {
                "sample_id": "sample-000",
                "prompt": "p",
                "answer": "a",
                "label": "faithful",
                "label_int": 1,
                "groundedness_score": 0.9,
                "predicted_label": 1,
                "confidence": 0.9,
                "abstain_flag": False,
                "decision_threshold": 0.5,
                "score_threshold": 0.55,
                "max_score_floor": 0.05,
            }
        ]
    ).to_csv(predictions_path, index=False)
    metrics_path.write_text(
        json.dumps(
            {
                "roc_auc": 1.0,
                "thresholds": {
                    "decision_threshold": 0.5,
                    "score_threshold": 0.55,
                    "max_score_floor": 0.05,
                },
            }
        ),
        encoding="utf-8",
    )
    plots_dir.mkdir(parents=True, exist_ok=True)
    (plots_dir / "roc.svg").write_text("<svg/>", encoding="utf-8")

    script_path = Path("scripts/export_demo_assets.py").resolve()
    subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--scores-path",
            str(scores_path),
            "--features-path",
            str(features_path),
            "--predictions-path",
            str(predictions_path),
            "--metrics-path",
            str(metrics_path),
            "--plots-dir",
            str(plots_dir),
            "--output-dir",
            str(output_dir),
        ],
        check=True,
    )

    index_payload = json.loads((output_dir / "index.json").read_text(encoding="utf-8"))
    analysis_payload = json.loads((output_dir / "analysis.json").read_text(encoding="utf-8"))
    detail_payload = json.loads(
        (output_dir / "examples" / "sample-000.json").read_text(encoding="utf-8")
    )

    assert index_payload["examples"][0]["detail_path"] == "/demo-stress/examples/sample-000.json"
    assert analysis_payload["plot_refs"]["roc"] == "/demo-stress/plots/roc.svg"
    assert detail_payload["plot_refs"]["roc"] == "/demo-stress/plots/roc.svg"
