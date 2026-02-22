#!/usr/bin/env python3
"""Generate proposal-oriented figures from baseline and stress metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-metrics", type=Path, default=Path("artifacts/metrics.json"))
    parser.add_argument(
        "--stress-metrics", type=Path, default=Path("artifacts_stress/metrics.json")
    )
    parser.add_argument(
        "--baseline-manifest",
        type=Path,
        default=Path("artifacts/run_manifest.json"),
    )
    parser.add_argument(
        "--stress-manifest",
        type=Path,
        default=Path("artifacts_stress/run_manifest.json"),
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
    return parser.parse_args()


def _bar_panel(
    ax: plt.Axes,
    labels: list[str],
    baseline_values: list[float],
    stress_values: list[float],
    title: str,
    better_hint: str,
) -> None:
    x = np.arange(len(labels))
    width = 0.36

    bars_baseline = ax.bar(
        x - width / 2,
        baseline_values,
        width,
        label="Baseline (Controlled)",
        color="#1f7aa8",
    )
    bars_stress = ax.bar(
        x + width / 2,
        stress_values,
        width,
        label="Stress (Distributed-Truth)",
        color="#d17a22",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0.0, 1.05)
    ax.set_title(f"{title} ({better_hint})", fontsize=12, weight="bold")
    ax.grid(axis="y", alpha=0.2)

    for bars in (bars_baseline, bars_stress):
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.02,
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )


def _read_threshold_block(manifest: dict[str, Any]) -> dict[str, float]:
    thresholds = manifest.get("thresholds", {})
    if not isinstance(thresholds, dict):
        return {}
    return {
        key: float(value) for key, value in thresholds.items() if isinstance(value, (int, float))
    }


def _text_panel(ax: plt.Axes, title: str, lines: list[str], face_color: str) -> None:
    ax.set_facecolor(face_color)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0.04, 0.9, title, fontsize=12, weight="bold", color="#0f172a", va="top")
    y = 0.78
    for line in lines:
        ax.text(0.05, y, f"- {line}", fontsize=10, color="#1f2937", va="top")
        y -= 0.14


def _render_comparison_figure(
    baseline: dict[str, Any], stress: dict[str, Any], output_path: Path
) -> None:
    higher_better_labels = ["ROC-AUC", "PR-AUC", "Answered Acc."]
    higher_baseline = [
        float(baseline["roc_auc"]),
        float(baseline["pr_auc"]),
        float(baseline["accuracy_when_answered"]),
    ]
    higher_stress = [
        float(stress["roc_auc"]),
        float(stress["pr_auc"]),
        float(stress["accuracy_when_answered"]),
    ]

    lower_better_labels = ["ECE", "Brier"]
    lower_baseline = [
        float(baseline["ece"]),
        float(baseline["brier"]),
    ]
    lower_stress = [
        float(stress["ece"]),
        float(stress["brier"]),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 7.5), constrained_layout=True)
    _bar_panel(
        axes[0],
        higher_better_labels,
        higher_baseline,
        higher_stress,
        title="Quality Metrics",
        better_hint="higher is better",
    )
    _bar_panel(
        axes[1],
        lower_better_labels,
        lower_baseline,
        lower_stress,
        title="Calibration/Error Metrics",
        better_hint="lower is better",
    )

    fig.suptitle(
        "ADS Research Proposal Evidence: Controlled Setting vs Distributed-Truth Stress",
        fontsize=15,
        weight="bold",
    )
    fig.text(
        0.5,
        0.01,
        (
            "Takeaway: ADS separates faithful/hallucinated well on controlled data, but quality degrades"
            " under distributed-truth style attribution patterns, motivating robustness-focused follow-up work."
        ),
        ha="center",
        fontsize=10,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _render_overview_figure(
    baseline: dict[str, Any],
    stress: dict[str, Any],
    baseline_manifest: dict[str, Any],
    stress_manifest: dict[str, Any],
    output_path: Path,
) -> None:
    fig = plt.figure(figsize=(14.5, 8.8), constrained_layout=True)
    spec = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.35, 1.0], height_ratios=[1.0, 1.0])

    ax_context = fig.add_subplot(spec[0, 0])
    ax_metrics = fig.add_subplot(spec[:, 1])
    ax_limit = fig.add_subplot(spec[0, 2])
    ax_method = fig.add_subplot(spec[1, 0])
    ax_repro = fig.add_subplot(spec[1, 2])

    _text_panel(
        ax_context,
        "Research Objective",
        [
            "Detect grounded vs hallucinated outputs using attribution-density geometry.",
            "Keep decisions auditable via top influential training evidence.",
            "Expose when confidence estimates drift from reality.",
        ],
        face_color="#e8f3f7",
    )

    _text_panel(
        ax_method,
        "Method Snapshot",
        [
            "Attribution backend -> top-k influence scores.",
            "Density features: entropy, top shares, peakiness_ratio, gini.",
            "Logistic + threshold fallback for abstention and robustness.",
        ],
        face_color="#edf7ee",
    )

    _text_panel(
        ax_limit,
        "Boundary Condition",
        [
            "Distributed-truth answers weaken top-1 concentration.",
            "Stress mode shows ROC/PR drop and calibration degradation.",
            "Proposal focus: robustness under diffuse but valid evidence.",
        ],
        face_color="#fff4e8",
    )

    base_thresholds = _read_threshold_block(baseline_manifest)
    stress_thresholds = _read_threshold_block(stress_manifest)
    threshold_text = (
        f"baseline={base_thresholds.get('decision_threshold', 0.5):.2f}/"
        f"{base_thresholds.get('score_threshold', 0.55):.2f}/"
        f"{base_thresholds.get('max_score_floor', 0.05):.2f}"
    )
    stress_threshold_text = (
        f"stress={stress_thresholds.get('decision_threshold', 0.5):.2f}/"
        f"{stress_thresholds.get('score_threshold', 0.55):.2f}/"
        f"{stress_thresholds.get('max_score_floor', 0.05):.2f}"
    )
    _text_panel(
        ax_repro,
        "Reproducibility Signals",
        [
            "Deterministic pipelines with run manifests.",
            f"Thresholds (decision/score/floor): {threshold_text}",
            f"Thresholds (decision/score/floor): {stress_threshold_text}",
        ],
        face_color="#f2f1fb",
    )

    metric_rows = [
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

    ax_metrics.set_title("Baseline vs Stress Metrics", fontsize=13, weight="bold", pad=12)
    y = np.arange(len(metric_rows))
    baseline_values = [item[1] for item in metric_rows]
    stress_values = [item[2] for item in metric_rows]
    ax_metrics.barh(y + 0.18, baseline_values, height=0.32, color="#1f7aa8", label="Baseline")
    ax_metrics.barh(
        y - 0.18,
        stress_values,
        height=0.32,
        color="#d17a22",
        label="Stress (distributed-truth)",
    )
    ax_metrics.set_yticks(y, labels=[item[0] for item in metric_rows], fontsize=10)
    ax_metrics.set_xlim(0.0, 1.05)
    ax_metrics.grid(axis="x", alpha=0.2)
    ax_metrics.invert_yaxis()
    ax_metrics.legend(loc="lower right", fontsize=9, frameon=True)
    ax_metrics.set_xlabel("Metric value", fontsize=10)

    for idx, (_, baseline_value, stress_value, direction) in enumerate(metric_rows):
        delta = stress_value - baseline_value
        improved = delta > 0 if direction == "higher" else delta < 0
        delta_color = "#15803d" if improved else "#b91c1c"
        ax_metrics.text(
            1.01,
            idx,
            f"{delta:+.3f}",
            va="center",
            ha="right",
            fontsize=10,
            color=delta_color,
            weight="bold",
        )

    fig.suptitle(
        "ADS Research Proposal: Method, Evidence, and Failure Boundary",
        fontsize=18,
        weight="bold",
    )
    fig.text(
        0.5,
        0.012,
        (
            "Interpretation: strong controlled performance + transparent evidence tracing are promising,"
            " but distributed-truth degradation motivates the next research milestone."
        ),
        ha="center",
        fontsize=10,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    baseline = _load_json(args.baseline_metrics)
    stress = _load_json(args.stress_metrics)
    baseline_manifest = _load_json(args.baseline_manifest)
    stress_manifest = _load_json(args.stress_manifest)

    _render_comparison_figure(
        baseline=baseline,
        stress=stress,
        output_path=args.comparison_output,
    )
    _render_overview_figure(
        baseline=baseline,
        stress=stress,
        baseline_manifest=baseline_manifest,
        stress_manifest=stress_manifest,
        output_path=args.overview_output,
    )


if __name__ == "__main__":
    main()
