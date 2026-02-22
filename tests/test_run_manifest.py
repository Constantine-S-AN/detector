"""Tests for run manifest generation script."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path


def test_write_run_manifest(tmp_path: Path) -> None:
    artifacts_dir = tmp_path / "artifacts"
    data_dir = artifacts_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    (data_dir / "manifest.json").write_text(
        json.dumps({"seed": 42, "num_samples": 40}),
        encoding="utf-8",
    )
    (artifacts_dir / "metrics.json").write_text(
        json.dumps({"roc_auc": 0.9, "plots": {"roc": "roc.svg"}}),
        encoding="utf-8",
    )

    script_path = Path("scripts/write_run_manifest.py").resolve()
    output_path = artifacts_dir / "run_manifest.json"

    subprocess.run(
        [
            "python3",
            str(script_path),
            "--artifacts-dir",
            str(artifacts_dir),
            "--output-path",
            str(output_path),
        ],
        check=True,
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["dataset"]["seed"] == 42
    assert payload["metrics_snapshot"]["roc_auc"] == 0.9
    assert payload["thresholds"]["decision_threshold"] == 0.5
    assert payload["thresholds"]["score_threshold"] == 0.55
    assert payload["thresholds"]["max_score_floor"] == 0.05
    assert payload["metrics_snapshot"]["thresholds"]["decision_threshold"] == 0.5
    assert "scripts/write_run_manifest.py" in payload["commands"]
