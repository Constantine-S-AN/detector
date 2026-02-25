"""Bootstrap confidence interval helpers for evaluation metrics."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


MetricFn = Callable[[np.ndarray, np.ndarray], float | None]


def bootstrap_ci(
    metric_fn: MetricFn,
    y_true: np.ndarray,
    y_score: np.ndarray,
    n: int = 1000,
    seed: int = 0,
    alpha: float = 0.95,
) -> dict[str, float]:
    """Estimate metric mean and percentile CI via bootstrap resampling."""
    y_true_arr = np.asarray(y_true)
    y_score_arr = np.asarray(y_score)
    if y_true_arr.shape[0] != y_score_arr.shape[0]:
        raise ValueError("y_true and y_score must have same length")
    if y_true_arr.shape[0] == 0:
        return {"mean": 0.0, "lo": 0.0, "hi": 0.0}

    rng = np.random.default_rng(seed)
    values: list[float] = []
    n_points = y_true_arr.shape[0]

    for _ in range(n):
        indices = rng.integers(0, n_points, size=n_points)
        sample_true = y_true_arr[indices]
        sample_score = y_score_arr[indices]
        value = metric_fn(sample_true, sample_score)
        if value is None or not np.isfinite(value):
            continue
        values.append(float(value))

    if not values:
        return {"mean": 0.0, "lo": 0.0, "hi": 0.0}

    value_array = np.asarray(values, dtype=np.float64)
    lower_q = (1.0 - alpha) / 2.0
    upper_q = 1.0 - lower_q
    return {
        "mean": float(np.mean(value_array)),
        "lo": float(np.quantile(value_array, lower_q)),
        "hi": float(np.quantile(value_array, upper_q)),
    }
