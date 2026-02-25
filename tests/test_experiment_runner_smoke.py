"""Smoke test for multi-seed experiment runner."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd


def test_experiment_runner_smoke(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    run_id = "smoke"

    subprocess.run(
        [
            sys.executable,
            "scripts/run_experiments.py",
            "--run-id",
            run_id,
            "--runs-root",
            str(runs_root),
            "--backends",
            "toy",
            "--ks",
            "5",
            "--seeds",
            "0,1",
            "--detectors",
            "logistic",
            "--num-samples",
            "12",
            "--train-size",
            "48",
            "--attribution-top-k",
            "10",
        ],
        check=True,
        text=True,
    )

    summary_csv = runs_root / run_id / "summary.csv"
    summary_md = runs_root / run_id / "summary.md"
    assert summary_csv.exists()
    assert summary_md.exists()

    frame = pd.read_csv(summary_csv)
    required_columns = {
        "backend",
        "k",
        "detector",
        "n_success",
        "n_skipped",
        "roc_auc_mean",
        "roc_auc_std",
        "pr_auc_mean",
        "pr_auc_std",
        "ece_mean",
        "ece_std",
        "brier_mean",
        "brier_std",
    }
    assert required_columns.issubset(set(frame.columns))
    assert frame.shape[0] >= 1
