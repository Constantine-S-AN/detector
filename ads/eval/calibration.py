"""Calibration helpers for detector scores."""

from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


def calibrate_scores(
    train_scores: np.ndarray,
    train_labels: np.ndarray,
    target_scores: np.ndarray,
    method: str,
    *,
    random_state: int = 42,
) -> np.ndarray:
    """Calibrate scores using train split and apply on target scores."""
    method_norm = method.strip().lower()
    if method_norm == "none":
        return np.asarray(target_scores, dtype=float)

    x_train = np.asarray(train_scores, dtype=float).reshape(-1, 1)
    y_train = np.asarray(train_labels, dtype=int)
    x_target = np.asarray(target_scores, dtype=float).reshape(-1, 1)

    if method_norm == "platt":
        model = LogisticRegression(max_iter=1000, random_state=random_state)
        model.fit(x_train, y_train)
        return model.predict_proba(x_target)[:, 1].astype(float)

    if method_norm == "isotonic":
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(x_train.ravel(), y_train.astype(float))
        return np.asarray(iso.predict(x_target.ravel()), dtype=float)

    raise ValueError(f"Unsupported calibration method: {method}")
