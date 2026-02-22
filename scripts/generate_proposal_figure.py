#!/usr/bin/env python3
"""Generate professional proposal-oriented figures for README showcase."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

BASELINE_COLOR = "#1d6f98"
STRESS_COLOR = "#d17a22"
GRID_COLOR = "#d9e2ec"
TEXT_DARK = "#0f172a"
TEXT_MUTED = "#334155"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _metric_rows(
    baseline: dict[str, Any], stress: dict[str, Any]
) -> list[tuple[str, float, float, str]]:
    return [
        ("ROC-AUC", float(baseline["roc_auc"]), float(stress["roc_auc"]), "higher"),
        ("PR-AUC", float(baseline["pr_auc"]), float(stress["pr_auc"]), "higher"),
        (
            "Answered Acc.",
            float(baseline["accuracy_when_answered"]),
            float(stress["accuracy_when_answered"]),
            "higher",
        ),
        ("ECE", float(baseline["ece"]), float(stress["ece"]), "lower"),
        ("Brier", float(baseline["brier"]), float(stress["brier"]), "lower"),
    ]


def _read_threshold_block(manifest: dict[str, Any]) -> dict[str, float]:
    thresholds = manifest.get("thresholds", {})
    if not isinstance(thresholds, dict):
        return {}
    return {
        key: float(value) for key, value in thresholds.items() if isinstance(value, (int, float))
    }


def _format_thresholds(tag: str, values: dict[str, float]) -> str:
    return (
        f"{tag}: decision={values.get('decision_threshold', 0.5):.2f}, "
        f"score={values.get('score_threshold', 0.55):.2f}, "
        f"floor={values.get('max_score_floor', 0.05):.2f}"
    )


def _style_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#cbd5e1")
    ax.spines["bottom"].set_color("#cbd5e1")
    ax.tick_params(colors=TEXT_MUTED)


def _draw_metric_dumbbell(ax: plt.Axes, rows: list[tuple[str, float, float, str]]) -> None:
    labels = [item[0] for item in rows]
    y = np.arange(len(rows))
    baseline_values = np.array([item[1] for item in rows], dtype=float)
    stress_values = np.array([item[2] for item in rows], dtype=float)

    for idx in range(len(rows)):
        ax.plot(
            [baseline_values[idx], stress_values[idx]],
            [y[idx], y[idx]],
            color="#94a3b8",
            lw=3.0,
            alpha=0.8,
            solid_capstyle="round",
            zorder=1,
        )

    ax.scatter(baseline_values, y, color=BASELINE_COLOR, s=90, zorder=2)
    ax.scatter(stress_values, y, color=STRESS_COLOR, s=90, zorder=2)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=11, color=TEXT_DARK)
    ax.set_xlim(0.0, 1.02)
    ax.set_xlabel("Metric value", fontsize=11, color=TEXT_MUTED)
    ax.grid(axis="x", color=GRID_COLOR, alpha=0.7)
    ax.invert_yaxis()
    _style_axes(ax)

    for idx, (_, baseline_value, stress_value, direction) in enumerate(rows):
        delta = stress_value - baseline_value
        improved = delta > 0 if direction == "higher" else delta < 0
        delta_color = "#15803d" if improved else "#b91c1c"
        ax.text(
            baseline_value + 0.012,
            idx + 0.18,
            f"B {baseline_value:.3f}",
            fontsize=9,
            color=BASELINE_COLOR,
            va="center",
        )
        ax.text(
            stress_value + 0.012,
            idx - 0.18,
            f"S {stress_value:.3f}",
            fontsize=9,
            color=STRESS_COLOR,
            va="center",
        )
        ax.text(
            1.015,
            idx,
            f"{delta:+.3f}",
            fontsize=10,
            weight="bold",
            color=delta_color,
            va="center",
            ha="right",
        )

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Baseline",
            markerfacecolor=BASELINE_COLOR,
            markersize=9,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Stress (distributed-truth)",
            markerfacecolor=STRESS_COLOR,
            markersize=9,
        ),
    ]
    ax.legend(handles=legend_elements, loc="lower right", frameon=False, fontsize=10)


def _section_text(ax: plt.Axes, title: str, lines: list[str], y_top: float) -> float:
    ax.text(
        0.03,
        y_top,
        title,
        fontsize=12,
        weight="bold",
        color=TEXT_DARK,
        va="top",
        ha="left",
    )
    y = y_top - 0.08
    for line in lines:
        ax.text(0.05, y, f"- {line}", fontsize=10, color=TEXT_MUTED, va="top", ha="left")
        y -= 0.08
    return y - 0.04


def _render_overview_figure(
    baseline: dict[str, Any],
    stress: dict[str, Any],
    baseline_manifest: dict[str, Any],
    stress_manifest: dict[str, Any],
    output_path: Path,
) -> None:
    rows = _metric_rows(baseline, stress)
    baseline_thresholds = _read_threshold_block(baseline_manifest)
    stress_thresholds = _read_threshold_block(stress_manifest)

    fig = plt.figure(figsize=(15.5, 8.8), facecolor="#f8fafc")
    spec = fig.add_gridspec(2, 3, height_ratios=[0.18, 0.82], width_ratios=[1.15, 1.6, 1.15])

    ax_header = fig.add_subplot(spec[0, :])
    ax_left = fig.add_subplot(spec[1, 0])
    ax_metrics = fig.add_subplot(spec[1, 1])
    ax_right = fig.add_subplot(spec[1, 2])

    for ax in (ax_header, ax_left, ax_right):
        ax.set_axis_off()

    ax_header.text(
        0.5,
        0.72,
        "ADS Research Proposal: Method, Evidence, and Failure Boundary",
        ha="center",
        va="center",
        fontsize=21,
        weight="bold",
        color=TEXT_DARK,
    )
    ax_header.text(
        0.5,
        0.28,
        (
            "A professional summary for proposal review: controlled performance, "
            "mechanistic interpretability, and distributed-truth stress degradation."
        ),
        ha="center",
        va="center",
        fontsize=11.5,
        color=TEXT_MUTED,
    )

    left_y = _section_text(
        ax_left,
        "Research Objective",
        [
            "Detect grounded vs hallucinated outputs via attribution-density geometry.",
            "Keep decisions auditable with top influential training evidence.",
            "Quantify uncertainty drift instead of relying on opaque confidence alone.",
        ],
        y_top=0.96,
    )
    _section_text(
        ax_left,
        "Method Snapshot",
        [
            "Attribution backend -> top-k influence scores.",
            "Density features: entropy, top shares, peakiness_ratio, gini.",
            "Logistic detector + abstain-aware fallback mechanism.",
        ],
        y_top=left_y,
    )

    _draw_metric_dumbbell(ax_metrics, rows)
    ax_metrics.set_title(
        "Baseline vs Stress (Dumbbell Comparison)",
        fontsize=13,
        weight="bold",
        color=TEXT_DARK,
        pad=12,
    )

    right_y = _section_text(
        ax_right,
        "Boundary Condition",
        [
            "Distributed-truth answers weaken top-1 concentration.",
            "Stress mode degrades ROC/PR and worsens calibration.",
            "This motivates robustness-focused next-stage research.",
        ],
        y_top=0.96,
    )
    _section_text(
        ax_right,
        "Reproducibility Signals",
        [
            "Deterministic pipeline with run manifests.",
            _format_thresholds("baseline", baseline_thresholds),
            _format_thresholds("stress", stress_thresholds),
        ],
        y_top=right_y,
    )

    fig.text(
        0.5,
        0.018,
        "Delta labels at right show stress-minus-baseline effect size per metric.",
        ha="center",
        fontsize=10,
        color="#475569",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _render_comparison_figure(
    baseline: dict[str, Any], stress: dict[str, Any], output_path: Path
) -> None:
    rows = _metric_rows(baseline, stress)

    fig, ax = plt.subplots(figsize=(12.8, 6.8), facecolor="#f8fafc")
    _draw_metric_dumbbell(ax, rows)
    ax.set_title(
        "Controlled vs Distributed-Truth Stress: Core Metric Shift",
        fontsize=15,
        weight="bold",
        color=TEXT_DARK,
        pad=14,
    )
    fig.text(
        0.5,
        0.025,
        "Higher-is-better metrics drop under stress; lower-is-better metrics increase.",
        ha="center",
        fontsize=10,
        color="#475569",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _render_controlled_figure(
    baseline_metrics: dict[str, Any],
    predictions_frame: pd.DataFrame,
    output_path: Path,
) -> None:
    frame = predictions_frame.copy()
    frame["label_int"] = frame["label_int"].astype(int)
    frame["predicted_label"] = frame["predicted_label"].astype(int)
    frame["groundedness_score"] = frame["groundedness_score"].astype(float)

    tp = int(((frame["label_int"] == 1) & (frame["predicted_label"] == 1)).sum())
    tn = int(((frame["label_int"] == 0) & (frame["predicted_label"] == 0)).sum())
    fp = int(((frame["label_int"] == 0) & (frame["predicted_label"] == 1)).sum())
    fn = int(((frame["label_int"] == 1) & (frame["predicted_label"] == 0)).sum())

    mean_faithful = float(frame.loc[frame["label_int"] == 1, "groundedness_score"].mean())
    mean_hallu = float(frame.loc[frame["label_int"] == 0, "groundedness_score"].mean())

    ablation = baseline_metrics.get("ablation", [])
    ablation_rows: list[tuple[str, float]] = []
    if isinstance(ablation, list):
        for item in ablation:
            if isinstance(item, dict) and item.get("roc_auc") is not None:
                ablation_rows.append((str(item["feature"]), float(item["roc_auc"])))
    ablation_rows = sorted(ablation_rows, key=lambda row: row[1], reverse=True)[:6]

    fig = plt.figure(figsize=(15.5, 8.8), facecolor="#f8fafc")
    spec = fig.add_gridspec(3, 3, height_ratios=[0.16, 0.44, 0.40], hspace=0.30, wspace=0.28)
    ax_header = fig.add_subplot(spec[0, :])
    ax_metrics = fig.add_subplot(spec[1, 0])
    ax_cm = fig.add_subplot(spec[1, 1])
    ax_mean = fig.add_subplot(spec[1, 2])
    ax_ablation = fig.add_subplot(spec[2, :])

    ax_header.set_axis_off()
    ax_header.text(
        0.5,
        0.66,
        "Controlled-Setting Evidence (Professional Summary)",
        ha="center",
        va="center",
        fontsize=20,
        weight="bold",
        color=TEXT_DARK,
    )
    ax_header.text(
        0.5,
        0.24,
        "This panel highlights clean separation in the controlled dataset and interpretable feature signals.",
        ha="center",
        va="center",
        fontsize=11,
        color=TEXT_MUTED,
    )

    ax_metrics.set_axis_off()
    metric_lines = [
        f"ROC-AUC: {float(baseline_metrics['roc_auc']):.4f}",
        f"PR-AUC: {float(baseline_metrics['pr_auc']):.4f}",
        f"ECE: {float(baseline_metrics['ece']):.4f}",
        f"Brier: {float(baseline_metrics['brier']):.4f}",
        f"Coverage: {float(baseline_metrics['coverage']):.4f}",
        f"Answered Accuracy: {float(baseline_metrics['accuracy_when_answered']):.4f}",
    ]
    _section_text(
        ax_metrics,
        "Key Metrics",
        metric_lines,
        y_top=0.95,
    )

    cm = np.array([[tp, fp], [fn, tn]], dtype=float)
    cm_img = ax_cm.imshow(cm, cmap="Blues")
    ax_cm.set_title("Confusion Matrix", fontsize=12, weight="bold", color=TEXT_DARK)
    ax_cm.set_xticks([0, 1], labels=["Pred=Faithful", "Pred=Hallu"])
    ax_cm.set_yticks([0, 1], labels=["True=Faithful", "True=Hallu"])
    for i in range(2):
        for j in range(2):
            ax_cm.text(
                j,
                i,
                f"{int(cm[i, j])}",
                ha="center",
                va="center",
                color=TEXT_DARK,
                fontsize=13,
                weight="bold",
            )
    _style_axes(ax_cm)
    fig.colorbar(cm_img, ax=ax_cm, fraction=0.046, pad=0.04)

    mean_labels = ["Faithful mean", "Hallucinated mean"]
    mean_values = [mean_faithful, mean_hallu]
    colors = [BASELINE_COLOR, STRESS_COLOR]
    ax_mean.bar(mean_labels, mean_values, color=colors, width=0.58)
    ax_mean.set_ylim(0, 1.02)
    ax_mean.set_title("Mean Groundedness Score", fontsize=12, weight="bold", color=TEXT_DARK)
    ax_mean.grid(axis="y", color=GRID_COLOR, alpha=0.7)
    _style_axes(ax_mean)
    for idx, value in enumerate(mean_values):
        ax_mean.text(idx, value + 0.02, f"{value:.4f}", ha="center", fontsize=10, color=TEXT_DARK)

    if ablation_rows:
        features = [item[0] for item in ablation_rows]
        values = [item[1] for item in ablation_rows]
        y_pos = np.arange(len(features))
        ax_ablation.barh(y_pos, values, color="#2563eb", alpha=0.88)
        ax_ablation.set_yticks(y_pos, labels=features)
        ax_ablation.set_xlim(0.0, 1.05)
        ax_ablation.invert_yaxis()
        ax_ablation.set_title(
            "Top Single-Feature ROC-AUC (Ablation)",
            fontsize=12,
            weight="bold",
            color=TEXT_DARK,
        )
        ax_ablation.grid(axis="x", color=GRID_COLOR, alpha=0.7)
        _style_axes(ax_ablation)
        for idx, value in enumerate(values):
            ax_ablation.text(value + 0.01, idx, f"{value:.3f}", va="center", fontsize=10)
    else:
        ax_ablation.set_axis_off()
        ax_ablation.text(
            0.5, 0.5, "Ablation data unavailable", ha="center", va="center", fontsize=12
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-metrics", type=Path, default=Path("artifacts/metrics.json"))
    parser.add_argument(
        "--stress-metrics", type=Path, default=Path("artifacts_stress/metrics.json")
    )
    parser.add_argument(
        "--baseline-manifest", type=Path, default=Path("artifacts/run_manifest.json")
    )
    parser.add_argument(
        "--stress-manifest",
        type=Path,
        default=Path("artifacts_stress/run_manifest.json"),
    )
    parser.add_argument(
        "--baseline-predictions",
        type=Path,
        default=Path("artifacts/predictions_all.csv"),
    )
    parser.add_argument(
        "--comparison-output",
        type=Path,
        default=Path("docs/images/proposal-baseline-vs-stress.png"),
    )
    parser.add_argument(
        "--overview-output",
        type=Path,
        default=Path("docs/images/proposal-overview.png"),
    )
    parser.add_argument(
        "--controlled-output",
        type=Path,
        default=Path("docs/images/proposal-controlled-evidence.png"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baseline = _load_json(args.baseline_metrics)
    stress = _load_json(args.stress_metrics)
    baseline_manifest = _load_json(args.baseline_manifest)
    stress_manifest = _load_json(args.stress_manifest)
    predictions_frame = pd.read_csv(args.baseline_predictions)

    _render_overview_figure(
        baseline=baseline,
        stress=stress,
        baseline_manifest=baseline_manifest,
        stress_manifest=stress_manifest,
        output_path=args.overview_output,
    )
    _render_controlled_figure(
        baseline_metrics=baseline,
        predictions_frame=predictions_frame,
        output_path=args.controlled_output,
    )
    _render_comparison_figure(
        baseline=baseline,
        stress=stress,
        output_path=args.comparison_output,
    )


if __name__ == "__main__":
    main()
