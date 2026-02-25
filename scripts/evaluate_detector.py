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
from ads.eval.calibration import calibrate_scores
from ads.eval.ci import bootstrap_ci
from ads.eval.metrics import compute_metrics_bundle
from ads.eval.metrics import metric_brier, metric_ece, metric_pr_auc, metric_roc_auc
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
    parser.add_argument(
        "--calibration",
        type=str,
        choices=("none", "platt", "isotonic"),
        default="none",
    )
    parser.add_argument("--ci-bootstrap-n", type=int, default=1000)
    parser.add_argument("--ci-seed", type=int, default=0)
    parser.add_argument("--summary-md-path", type=Path, default=Path("artifacts/summary.md"))
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


def _subset_breakdown(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true_i = y_true.astype(int)
    y_pred_i = y_pred.astype(int)

    tp = int(np.sum((y_true_i == 1) & (y_pred_i == 1)))
    fp = int(np.sum((y_true_i == 0) & (y_pred_i == 1)))
    tn = int(np.sum((y_true_i == 0) & (y_pred_i == 0)))
    fn = int(np.sum((y_true_i == 1) & (y_pred_i == 0)))

    precision = float(tp / max(tp + fp, 1))
    recall = float(tp / max(tp + fn, 1))
    fpr = float(fp / max(fp + tn, 1))
    return {
        "precision": precision,
        "recall": recall,
        "fpr": fpr,
        "tp": float(tp),
        "fp": float(fp),
        "tn": float(tn),
        "fn": float(fn),
    }


def _calibration_metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    return {
        "ece": metric_ece(y_true.astype(int), y_score.astype(float)),
        "brier": metric_brier(y_true.astype(int), y_score.astype(float)),
    }


def _write_summary_md(path: Path, metrics: dict[str, Any]) -> None:
    stress = metrics.get("stress_analysis", {})
    normal = stress.get("normal", {})
    dist = stress.get("distributed_truth", {})

    lines = ["# Evaluation Summary", ""]
    lines.append("## Headline")
    lines.append("")
    lines.append(
        f"- ROC-AUC: {float(metrics.get('roc_auc', 0.0) or 0.0):.4f}  "
        f"PR-AUC: {float(metrics.get('pr_auc', 0.0) or 0.0):.4f}"
    )
    lines.append(
        f"- Calibration ({metrics.get('calibration', {}).get('method', 'none')}): "
        f"ECE={float(metrics.get('ece', 0.0) or 0.0):.4f}, "
        f"Brier={float(metrics.get('brier', 0.0) or 0.0):.4f}"
    )

    lines.append("")
    lines.append("## Distributed-truth stress analysis")
    lines.append("")
    if normal and dist:
        lines.append(
            f"- Normal subset FPR={float(normal.get('fpr', 0.0)):.4f}, "
            f"Distributed-truth FPR={float(dist.get('fpr', 0.0)):.4f}."
        )
        lines.append(
            "- 误报来源解释：distributed-truth 样本的真实证据通常分散在多个训练项上，"
            "导致单峰度下降、entropy 上升，阈值/线性分类器更易把其误判为 hallucination。"
        )
    else:
        lines.append("- 当前评估集没有可分辨的 distributed_truth 子集或样本不足。")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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

    # Pre-calibration metrics on test split.
    metrics_pre = compute_metrics_bundle(
        y_true=test_frame["label_int"].to_numpy(dtype=int),
        y_score=test_frame["groundedness_score"].to_numpy(dtype=float),
        y_pred=test_frame["predicted_label"].to_numpy(dtype=int),
        abstain_flags=test_frame["abstain_flag"].to_numpy(dtype=bool),
        decision_threshold=float(args.decision_threshold),
    )

    # Optional calibration (fit on train split, apply to test split).
    calibration_method = str(args.calibration)
    calibrated_test_scores = test_frame["groundedness_score"].to_numpy(dtype=float)
    if calibration_method != "none":
        calibrated_test_scores = calibrate_scores(
            train_scores=train_frame["groundedness_score"].to_numpy(dtype=float),
            train_labels=train_frame["label_int"].to_numpy(dtype=int),
            target_scores=calibrated_test_scores,
            method=calibration_method,
            random_state=int(split_payload.get("seed", 42)),
        )

    calibrated_test_pred = (calibrated_test_scores >= args.decision_threshold).astype(int)
    calibrated_test_pred = np.where(
        test_frame["abstain_flag"].to_numpy(dtype=bool),
        0,
        calibrated_test_pred,
    )

    metrics = compute_metrics_bundle(
        y_true=test_frame["label_int"].to_numpy(dtype=int),
        y_score=calibrated_test_scores,
        y_pred=calibrated_test_pred.astype(int),
        abstain_flags=test_frame["abstain_flag"].to_numpy(dtype=bool),
        decision_threshold=float(args.decision_threshold),
    )

    # Keep both pre and post calibration summaries.
    pre_calib_scalar = _calibration_metrics(
        y_true=test_frame["label_int"].to_numpy(dtype=int),
        y_score=test_frame["groundedness_score"].to_numpy(dtype=float),
    )
    post_calib_scalar = _calibration_metrics(
        y_true=test_frame["label_int"].to_numpy(dtype=int),
        y_score=calibrated_test_scores,
    )

    metrics["calibration"] = {
        "method": calibration_method,
        "pre": pre_calib_scalar,
        "post": post_calib_scalar,
        "curves_pre": metrics_pre.get("curves", {}).get("calibration", []),
        "curves_post": metrics.get("curves", {}).get("calibration", []),
    }

    # Update test frame with post-calibration predictions for plots and stress analysis.
    test_frame["groundedness_score"] = calibrated_test_scores
    test_frame["predicted_label"] = calibrated_test_pred.astype(int)

    # Bootstrap CIs on post-calibration test scores.
    y_true_test = test_frame["label_int"].to_numpy(dtype=int)
    y_score_test = test_frame["groundedness_score"].to_numpy(dtype=float)
    metrics["roc_auc_ci"] = bootstrap_ci(
        metric_roc_auc,
        y_true_test,
        y_score_test,
        n=int(args.ci_bootstrap_n),
        seed=int(args.ci_seed),
    )
    metrics["pr_auc_ci"] = bootstrap_ci(
        metric_pr_auc,
        y_true_test,
        y_score_test,
        n=int(args.ci_bootstrap_n),
        seed=int(args.ci_seed) + 1,
    )
    metrics["ece_ci"] = bootstrap_ci(
        lambda yt, ys: metric_ece(yt, ys),
        y_true_test,
        y_score_test,
        n=int(args.ci_bootstrap_n),
        seed=int(args.ci_seed) + 2,
    )
    metrics["brier_ci"] = bootstrap_ci(
        metric_brier,
        y_true_test,
        y_score_test,
        n=int(args.ci_bootstrap_n),
        seed=int(args.ci_seed) + 3,
    )

    # Stress subset analysis: distributed_truth vs normal (if attribution_mode exists).
    if "attribution_mode" in test_frame.columns:
        distributed_mask = test_frame["attribution_mode"].astype(str).eq("distributed")
        normal_mask = ~distributed_mask
    else:
        distributed_mask = np.zeros(test_frame.shape[0], dtype=bool)
        normal_mask = np.ones(test_frame.shape[0], dtype=bool)

    stress_payload: dict[str, Any] = {}
    if np.any(normal_mask):
        normal_true = test_frame.loc[normal_mask, "label_int"].to_numpy(dtype=int)
        normal_pred = test_frame.loc[normal_mask, "predicted_label"].to_numpy(dtype=int)
        stress_payload["normal"] = _subset_breakdown(normal_true, normal_pred)
    if np.any(distributed_mask):
        dist_true = test_frame.loc[distributed_mask, "label_int"].to_numpy(dtype=int)
        dist_pred = test_frame.loc[distributed_mask, "predicted_label"].to_numpy(dtype=int)
        stress_payload["distributed_truth"] = _subset_breakdown(dist_true, dist_pred)
    metrics["stress_analysis"] = stress_payload

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
    args.ablation_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metrics["ablation"]).to_csv(args.ablation_path, index=False)

    args.metrics_path.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_summary_md(args.summary_md_path, metrics)


if __name__ == "__main__":
    main()
