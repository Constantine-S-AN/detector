#!/usr/bin/env python3
"""Evaluate trained detector and render metrics/plots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from ads.detector.logistic import FEATURE_COLUMNS, LogisticDetector
from ads.eval.metrics import compute_metrics_bundle
from ads.eval.plots import render_evaluation_plots


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-path", type=Path, default=Path("artifacts/features.csv"))
    parser.add_argument("--model-path", type=Path, default=Path("artifacts/models/logistic.joblib"))
    parser.add_argument("--split-path", type=Path, default=Path("artifacts/data/splits.json"))
    parser.add_argument("--metrics-path", type=Path, default=Path("artifacts/metrics.json"))
    parser.add_argument("--plot-dir", type=Path, default=Path("artifacts/plots"))
    parser.add_argument(
        "--predictions-path",
        type=Path,
        default=Path("artifacts/predictions_all.csv"),
    )
    parser.add_argument(
        "--test-predictions-path",
        type=Path,
        default=Path("artifacts/predictions_test.csv"),
    )
    parser.add_argument("--ablation-path", type=Path, default=Path("artifacts/ablation.csv"))
    parser.add_argument("--decision-threshold", type=float, default=0.5)
    parser.add_argument("--score-threshold", type=float, default=0.55)
    parser.add_argument("--max-score-floor", type=float, default=0.05)
    return parser.parse_args()


def _load_splits(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_binary_auc(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    if np.unique(y_true).size < 2:
        return None
    return float(roc_auc_score(y_true, y_score))


def _safe_binary_pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    if np.unique(y_true).size < 2:
        return None
    return float(average_precision_score(y_true, y_score))


def _compute_single_feature_ablation(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    *,
    seed: int,
) -> list[dict[str, float | str | None]]:
    """Train one-feature logistic baselines and report test-set AUCs."""
    y_train = train_frame["label_int"].to_numpy(dtype=int)
    y_test = test_frame["label_int"].to_numpy(dtype=int)
    rows: list[dict[str, float | str | None]] = []

    for feature_name in FEATURE_COLUMNS:
        scaler = StandardScaler()
        model = LogisticRegression(max_iter=2000, random_state=seed)

        x_train = train_frame[[feature_name]].to_numpy(dtype=float)
        x_test = test_frame[[feature_name]].to_numpy(dtype=float)
        x_train_scaled = scaler.fit_transform(x_train)
        x_test_scaled = scaler.transform(x_test)

        model.fit(x_train_scaled, y_train)
        score = model.predict_proba(x_test_scaled)[:, 1]
        rows.append(
            {
                "feature": feature_name,
                "roc_auc": _safe_binary_auc(y_test, score),
                "pr_auc": _safe_binary_pr_auc(y_test, score),
            }
        )
    rows.sort(
        key=lambda item: float(item["roc_auc"]) if item["roc_auc"] is not None else -1.0,
        reverse=True,
    )
    return rows


def _normalize_mode_bucket(value: object) -> str:
    normalized = str(value).strip().lower()
    if normalized in {"distributed", "distributed_truth"}:
        return "distributed_truth"
    return "normal"


def build_stress_analysis(frame: pd.DataFrame) -> dict[str, dict[str, float | int | None]]:
    """Summarize prediction behavior by attribution mode buckets."""
    if frame.empty:
        return {}

    if "attribution_mode" in frame.columns:
        mode_series = frame["attribution_mode"].fillna("unknown").astype(str)
    else:
        mode_series = pd.Series(["unknown"] * len(frame), index=frame.index, dtype="object")

    enriched_frame = frame.copy()
    enriched_frame["mode_bucket"] = mode_series.map(_normalize_mode_bucket)

    result: dict[str, dict[str, float | int | None]] = {}
    for bucket_name, bucket_frame in enriched_frame.groupby("mode_bucket"):
        y_true = bucket_frame["label_int"].to_numpy(dtype=int)
        y_score = bucket_frame["groundedness_score"].to_numpy(dtype=float)
        y_pred = bucket_frame["predicted_label"].to_numpy(dtype=int)
        coverage = 1.0 - float(bucket_frame["abstain_flag"].astype(bool).mean())
        accuracy = float((y_true == y_pred).mean())
        result[str(bucket_name)] = {
            "count": int(len(bucket_frame)),
            "coverage": float(coverage),
            "accuracy": accuracy,
            "mean_score": float(bucket_frame["groundedness_score"].mean()),
            "roc_auc": _safe_binary_auc(y_true, y_score),
            "pr_auc": _safe_binary_pr_auc(y_true, y_score),
        }
    return result


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.features_path)
    split_payload = _load_splits(args.split_path)
    test_ids = set(split_payload["test_ids"])

    detector = LogisticDetector.load(args.model_path)
    scores_all = detector.predict_score_frame(frame)
    predictions_all = (scores_all >= args.decision_threshold).astype(int)
    confidence_all = np.maximum(scores_all, 1.0 - scores_all)
    abstain_all = frame["abstain_flag"].astype(bool).to_numpy()
    predictions_all = np.where(abstain_all, 0, predictions_all)

    all_frame = frame.copy()
    all_frame["groundedness_score"] = scores_all
    all_frame["predicted_label"] = predictions_all
    all_frame["confidence"] = confidence_all
    all_frame["abstain_flag"] = abstain_all
    all_frame["decision_threshold"] = float(args.decision_threshold)
    all_frame["score_threshold"] = float(args.score_threshold)
    all_frame["max_score_floor"] = float(args.max_score_floor)
    args.predictions_path.parent.mkdir(parents=True, exist_ok=True)
    all_frame.to_csv(args.predictions_path, index=False)

    test_frame = all_frame[all_frame["sample_id"].isin(test_ids)].copy()
    train_frame = all_frame[~all_frame["sample_id"].isin(test_ids)].copy()
    args.test_predictions_path.parent.mkdir(parents=True, exist_ok=True)
    test_frame.to_csv(args.test_predictions_path, index=False)

    metrics = compute_metrics_bundle(
        y_true=test_frame["label_int"].to_numpy(dtype=int),
        y_score=test_frame["groundedness_score"].to_numpy(dtype=float),
        y_pred=test_frame["predicted_label"].to_numpy(dtype=int),
        abstain_flags=test_frame["abstain_flag"].to_numpy(dtype=bool),
        decision_threshold=float(args.decision_threshold),
    )
    metrics["thresholds"] = {
        "decision_threshold": float(args.decision_threshold),
        "score_threshold": float(args.score_threshold),
        "max_score_floor": float(args.max_score_floor),
    }
    metrics["plots"] = render_evaluation_plots(
        metrics=metrics,
        predictions_frame=test_frame,
        output_dir=args.plot_dir,
    )
    metrics["ablation"] = _compute_single_feature_ablation(
        train_frame=train_frame,
        test_frame=test_frame,
        seed=int(split_payload.get("seed", 42)),
    )
    metrics["stress_analysis"] = build_stress_analysis(test_frame)
    args.ablation_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metrics["ablation"]).to_csv(args.ablation_path, index=False)

    args.metrics_path.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
