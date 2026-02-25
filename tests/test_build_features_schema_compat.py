"""Schema compatibility tests for scripts/build_features.py."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _run_build_features(scores_path: Path, output_path: Path) -> pd.DataFrame:
    script_path = Path("scripts/build_features.py").resolve()
    subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--scores-path",
            str(scores_path),
            "--output-path",
            str(output_path),
        ],
        check=True,
    )
    return pd.read_csv(output_path)


def _write_jsonl(path: Path, rows: list[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n"
    path.write_text(payload, encoding="utf-8")


def _base_items() -> list[dict[str, object]]:
    return [
        {
            "train_id": "train-001",
            "score": 1.2,
            "text": "alpha evidence",
            "meta": {"rank": 1, "mode": "distributed"},
        },
        {
            "train_id": "train-002",
            "score": 0.4,
            "text": "beta evidence",
            "meta": {"rank": 2, "mode": "distributed"},
        },
        {
            "train_id": "train-003",
            "score": 0.2,
            "text": "gamma evidence",
            "meta": {"rank": 3, "mode": "distributed"},
        },
    ]


def test_build_features_supports_wrapper_items_and_legacy_and_items_only(tmp_path: Path) -> None:
    items = _base_items()

    wrapper_scores = tmp_path / "wrapper_scores.jsonl"
    items_only_scores = tmp_path / "items_only_scores.jsonl"
    legacy_scores = tmp_path / "legacy_scores.jsonl"

    _write_jsonl(
        wrapper_scores,
        [
            {
                "sample_id": "sample-wrapper",
                "prompt": "p",
                "answer": "a",
                "label": "faithful",
                "attribution_mode": "distributed",
                "items": items,
                "k_requested": 3,
            }
        ],
    )
    _write_jsonl(items_only_scores, [items])
    _write_jsonl(
        legacy_scores,
        [
            {
                "sample_id": "sample-legacy",
                "prompt": "p",
                "answer": "a",
                "label": "faithful",
                "attribution": items,
                "top_k": 3,
            }
        ],
    )

    frame_wrapper = _run_build_features(wrapper_scores, tmp_path / "features_wrapper.csv")
    frame_items_only = _run_build_features(items_only_scores, tmp_path / "features_items_only.csv")
    frame_legacy = _run_build_features(legacy_scores, tmp_path / "features_legacy.csv")

    for frame in (frame_wrapper, frame_items_only, frame_legacy):
        assert "attribution_mode" in frame.columns
        for column in (
            "entropy_top_k",
            "peakiness_ratio",
            "effective_k",
            "h_at_k_20",
            "h_at_k_norm_20",
        ):
            value = float(frame.iloc[0][column])
            assert np.isfinite(value)

    excluded_columns = {"sample_id", "prompt", "answer", "label", "label_int", "attribution_mode"}
    feature_columns = [column for column in frame_wrapper.columns if column not in excluded_columns]
    for column in feature_columns:
        wrapper_value = float(frame_wrapper.iloc[0][column])
        items_only_value = float(frame_items_only.iloc[0][column])
        legacy_value = float(frame_legacy.iloc[0][column])
        assert np.isclose(wrapper_value, items_only_value, atol=1e-9)
        assert np.isclose(wrapper_value, legacy_value, atol=1e-9)


def test_build_features_preserves_attribution_mode_with_fallback_priority(tmp_path: Path) -> None:
    items = _base_items()
    scores_path = tmp_path / "mixed_scores.jsonl"
    _write_jsonl(
        scores_path,
        [
            {
                "sample_id": "s-1",
                "prompt": "p1",
                "answer": "a1",
                "label": "faithful",
                "attribution_mode": "distributed_truth",
                "sample_meta": {"attribution_mode": "peaked"},
                "items": items,
            },
            {
                "sample_id": "s-2",
                "prompt": "p2",
                "answer": "a2",
                "label": "hallucinated",
                "sample_meta": {"attribution_mode": "peaked"},
                "items": items,
            },
            {
                "sample_id": "s-3",
                "prompt": "p3",
                "answer": "a3",
                "label": "hallucinated",
                "attribution": [
                    {
                        "train_id": "train-101",
                        "score": 0.3,
                        "text": "delta",
                        "meta": {"mode": "diffuse"},
                    }
                ],
            },
            {
                "sample_id": "s-4",
                "prompt": "p4",
                "answer": "a4",
                "label": "hallucinated",
                "items": [
                    {
                        "train_id": "train-102",
                        "score": 0.25,
                        "text": "epsilon",
                    }
                ],
            },
        ],
    )

    frame = _run_build_features(scores_path, tmp_path / "features.csv")
    mode_map = {str(row["sample_id"]): str(row["attribution_mode"]) for _, row in frame.iterrows()}
    assert mode_map["s-1"] == "distributed_truth"
    assert mode_map["s-2"] == "peaked"
    assert mode_map["s-3"] == "diffuse"
    assert mode_map["s-4"] == "unknown"
