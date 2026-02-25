"""Tests for bootstrap CI and calibration toggle behavior."""

from __future__ import annotations

import numpy as np

from ads.eval.calibration import calibrate_scores
from ads.eval.ci import bootstrap_ci
from ads.eval.metrics import metric_brier, metric_ece, metric_roc_auc


def test_bootstrap_ci_bounds_are_valid() -> None:
    y_true = np.array([1, 1, 0, 0, 1, 0, 1, 0], dtype=int)
    y_score = np.array([0.9, 0.8, 0.2, 0.1, 0.7, 0.4, 0.6, 0.3], dtype=float)

    ci = bootstrap_ci(metric_roc_auc, y_true, y_score, n=200, seed=0)

    assert set(ci.keys()) == {"mean", "lo", "hi"}
    assert ci["lo"] <= ci["mean"] <= ci["hi"]


def test_calibration_toggle_outputs_in_range() -> None:
    train_scores = np.array([0.1, 0.2, 0.8, 0.9, 0.7, 0.3], dtype=float)
    train_labels = np.array([0, 0, 1, 1, 1, 0], dtype=int)
    test_scores = np.array([0.15, 0.35, 0.65, 0.85], dtype=float)

    for method in ("platt", "isotonic"):
        calibrated = calibrate_scores(train_scores, train_labels, test_scores, method)
        assert calibrated.shape == test_scores.shape
        assert np.all(calibrated >= 0.0)
        assert np.all(calibrated <= 1.0)

    # Smoke metric computations on calibrated outputs.
    y_true = np.array([0, 0, 1, 1], dtype=int)
    platt_scores = calibrate_scores(train_scores, train_labels, test_scores, "platt")
    assert metric_ece(y_true, platt_scores) >= 0.0
    assert metric_brier(y_true, platt_scores) >= 0.0
