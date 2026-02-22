"""Tests for threshold/logistic detector behavior."""

from __future__ import annotations

import pandas as pd

from ads.detector.logistic import LogisticDetector
from ads.detector.threshold import ThresholdDetector
from ads.features.density import compute_density_features


def test_threshold_detector_flags_abstain() -> None:
    detector = ThresholdDetector(score_threshold=0.5, score_floor=0.1)
    features = compute_density_features([0.01, 0.02, 0.03], max_score_floor=0.1)
    output = detector.predict(features)
    assert output.abstain_flag is True


def test_logistic_detector_fit_and_predict() -> None:
    frame = pd.DataFrame(
        [
            {
                "entropy_top_k": 0.2,
                "top1_share": 0.8,
                "top5_share": 0.95,
                "gini": 0.7,
                "max_score": 3.0,
                "effective_k": 2.0,
                "label": 1,
            },
            {
                "entropy_top_k": 0.85,
                "top1_share": 0.09,
                "top5_share": 0.4,
                "gini": 0.15,
                "max_score": 0.8,
                "effective_k": 9.5,
                "label": 0,
            },
            {
                "entropy_top_k": 0.3,
                "top1_share": 0.6,
                "top5_share": 0.85,
                "gini": 0.6,
                "max_score": 2.4,
                "effective_k": 3.0,
                "label": 1,
            },
            {
                "entropy_top_k": 0.9,
                "top1_share": 0.07,
                "top5_share": 0.35,
                "gini": 0.1,
                "max_score": 0.6,
                "effective_k": 10.2,
                "label": 0,
            },
        ]
    )
    detector = LogisticDetector(random_state=42)
    detector.fit(frame, label_column="label")

    features = compute_density_features([3.5, 0.5, 0.3, 0.1], max_score_floor=0.05)
    output = detector.predict_output(features)
    assert 0.0 <= output.groundedness_score <= 1.0
