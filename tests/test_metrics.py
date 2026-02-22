"""Tests for evaluation metrics bundle outputs."""

from __future__ import annotations

import numpy as np

from ads.eval.metrics import compute_metrics_bundle


def test_metrics_bundle_contains_abstain_curve() -> None:
    y_true = np.array([1, 1, 0, 0, 1, 0], dtype=int)
    y_score = np.array([0.92, 0.71, 0.33, 0.21, 0.65, 0.44], dtype=float)
    y_pred = (y_score >= 0.5).astype(int)
    abstain_flags = np.array([False, False, False, False, False, False])

    metrics = compute_metrics_bundle(
        y_true=y_true, y_score=y_score, y_pred=y_pred, abstain_flags=abstain_flags
    )

    abstain_curve = metrics["curves"]["abstain"]
    assert len(abstain_curve) > 0
    assert abstain_curve[0]["x"] <= abstain_curve[-1]["x"]
    assert all(0.0 <= point["x"] <= 1.0 for point in abstain_curve)
    assert all(0.0 <= point["y"] <= 1.0 for point in abstain_curve)
