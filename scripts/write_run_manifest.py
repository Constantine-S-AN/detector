#!/usr/bin/env python3
"""Write run-level manifest for reproducibility metadata."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_git_revision() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return completed.stdout.strip() or None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--output-path", type=Path, default=Path("artifacts/run_manifest.json"))
    parser.add_argument("--decision-threshold", type=float, default=0.5)
    parser.add_argument("--score-threshold", type=float, default=0.55)
    parser.add_argument("--max-score-floor", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifacts_dir = args.artifacts_dir
    data_manifest = _load_json(artifacts_dir / "data" / "manifest.json")
    splits = _load_json(artifacts_dir / "data" / "splits.json")
    metrics = _load_json(artifacts_dir / "metrics.json")
    metrics_thresholds = metrics.get("thresholds", {})
    splits_thresholds = splits.get("thresholds", {})
    if not isinstance(splits_thresholds, dict):
        splits_thresholds = {}

    thresholds = {
        "decision_threshold": float(
            metrics_thresholds.get(
                "decision_threshold",
                splits_thresholds.get(
                    "decision_threshold",
                    splits.get("decision_threshold", args.decision_threshold),
                ),
            )
        ),
        "score_threshold": float(
            metrics_thresholds.get(
                "score_threshold",
                splits_thresholds.get("score_threshold", args.score_threshold),
            )
        ),
        "max_score_floor": float(
            metrics_thresholds.get(
                "max_score_floor",
                splits_thresholds.get("max_score_floor", args.max_score_floor),
            )
        ),
    }
    build_dataset_command = (
        "scripts/build_stress_dataset.py"
        if data_manifest.get("stress_type")
        else "scripts/build_controlled_dataset.py"
    )

    payload: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "git_revision": _safe_git_revision(),
        "dataset": data_manifest,
        "metrics_snapshot": {
            "roc_auc": metrics.get("roc_auc"),
            "pr_auc": metrics.get("pr_auc"),
            "brier": metrics.get("brier"),
            "ece": metrics.get("ece"),
            "coverage": metrics.get("coverage"),
            "accuracy_when_answered": metrics.get("accuracy_when_answered"),
            "thresholds": thresholds,
        },
        "thresholds": thresholds,
        "command_params": {
            "build_features": {"max_score_floor": thresholds["max_score_floor"]},
            "train_detector": thresholds,
            "evaluate_detector": thresholds,
            "write_run_manifest": thresholds,
        },
        "plots": metrics.get("plots", {}),
        "commands": [
            build_dataset_command,
            "scripts/run_attribution.py",
            "scripts/build_features.py",
            "scripts/train_detector.py",
            "scripts/evaluate_detector.py",
            "scripts/export_demo_assets.py",
            "scripts/write_run_manifest.py",
        ],
    }

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
