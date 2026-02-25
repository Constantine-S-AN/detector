"""Tests for stress-analysis grouping in evaluate_detector."""

from __future__ import annotations

import pandas as pd

from scripts.evaluate_detector import build_stress_analysis


def test_build_stress_analysis_groups_distributed_truth_rows() -> None:
    frame = pd.DataFrame(
        [
            {
                "sample_id": "s-1",
                "label_int": 1,
                "predicted_label": 1,
                "groundedness_score": 0.9,
                "abstain_flag": False,
                "attribution_mode": "peaked",
            },
            {
                "sample_id": "s-2",
                "label_int": 0,
                "predicted_label": 0,
                "groundedness_score": 0.2,
                "abstain_flag": False,
                "attribution_mode": "diffuse",
            },
            {
                "sample_id": "s-3",
                "label_int": 1,
                "predicted_label": 0,
                "groundedness_score": 0.45,
                "abstain_flag": False,
                "attribution_mode": "distributed",
            },
            {
                "sample_id": "s-4",
                "label_int": 0,
                "predicted_label": 1,
                "groundedness_score": 0.62,
                "abstain_flag": True,
                "attribution_mode": "distributed_truth",
            },
        ]
    )

    analysis = build_stress_analysis(frame)
    assert "normal" in analysis
    assert "distributed_truth" in analysis
    assert analysis["normal"]["count"] == 2
    assert analysis["distributed_truth"]["count"] == 2
    assert 0.0 <= float(analysis["distributed_truth"]["coverage"]) <= 1.0
