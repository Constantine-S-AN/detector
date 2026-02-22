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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifacts_dir = args.artifacts_dir
    data_manifest = _load_json(artifacts_dir / "data" / "manifest.json")
    metrics = _load_json(artifacts_dir / "metrics.json")

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
        },
        "plots": metrics.get("plots", {}),
        "commands": [
            "scripts/build_controlled_dataset.py",
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
