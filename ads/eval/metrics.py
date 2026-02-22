"""Evaluation metrics for groundedness detectors."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def _safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    if np.unique(y_true).size < 2:
        return None
    return float(roc_auc_score(y_true, y_score))


def _safe_pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    if np.unique(y_true).size < 2:
        return None
    return float(average_precision_score(y_true, y_score))


def _expected_calibration_error(y_true: np.ndarray, y_score: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for idx in range(n_bins):
        left = bins[idx]
        right = bins[idx + 1]
        mask = (y_score >= left) & (y_score < right)
        if idx == n_bins - 1:
            mask = (y_score >= left) & (y_score <= right)
        if not np.any(mask):
            continue
        bin_confidence = float(np.mean(y_score[mask]))
        bin_accuracy = float(np.mean(y_true[mask]))
        bin_weight = float(np.mean(mask))
        ece += abs(bin_accuracy - bin_confidence) * bin_weight
    return float(ece)


def _abstain_coverage_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    abstain_flags: np.ndarray,
) -> dict[str, float]:
    answered_mask = ~abstain_flags
    coverage = float(np.mean(answered_mask))
    if np.any(answered_mask):
        accuracy = float(np.mean((y_pred[answered_mask] == y_true[answered_mask]).astype(float)))
    else:
        accuracy = 0.0
    return {
        "coverage": coverage,
        "accuracy_when_answered": accuracy,
    }


def _curve_points(x_values: np.ndarray, y_values: np.ndarray) -> list[dict[str, float]]:
    return [{"x": float(x), "y": float(y)} for x, y in zip(x_values, y_values, strict=False)]


def compute_metrics_bundle(
    y_true: np.ndarray,
    y_score: np.ndarray,
    y_pred: np.ndarray,
    abstain_flags: np.ndarray,
) -> dict[str, Any]:
    """Compute scalar metrics and curve points for analysis/reporting."""
    y_true_int = y_true.astype(int)
    y_pred_int = y_pred.astype(int)
    abstain_bool = abstain_flags.astype(bool)

    roc_auc = _safe_auc(y_true_int, y_score)
    pr_auc = _safe_pr_auc(y_true_int, y_score)
    brier = float(brier_score_loss(y_true_int, y_score))
    ece = _expected_calibration_error(y_true_int, y_score, n_bins=10)
    abstain_metrics = _abstain_coverage_accuracy(y_true_int, y_pred_int, abstain_bool)

    curve_payload: dict[str, list[dict[str, float]]] = {
        "roc": [],
        "pr": [],
        "calibration": [],
    }
    if np.unique(y_true_int).size >= 2:
        fpr, tpr, _ = roc_curve(y_true_int, y_score)
        precision, recall, _ = precision_recall_curve(y_true_int, y_score)
        curve_payload["roc"] = _curve_points(fpr, tpr)
        curve_payload["pr"] = _curve_points(recall, precision)

    bins = np.linspace(0.0, 1.0, 11)
    for idx in range(10):
        left = bins[idx]
        right = bins[idx + 1]
        mask = (y_score >= left) & (y_score < right)
        if idx == 9:
            mask = (y_score >= left) & (y_score <= right)
        if not np.any(mask):
            continue
        curve_payload["calibration"].append(
            {
                "x": float(np.mean(y_score[mask])),
                "y": float(np.mean(y_true_int[mask])),
            }
        )

    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "brier": brier,
        "ece": ece,
        "coverage": abstain_metrics["coverage"],
        "accuracy_when_answered": abstain_metrics["accuracy_when_answered"],
        "curves": curve_payload,
    }
