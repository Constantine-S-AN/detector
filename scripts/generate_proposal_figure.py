#!/usr/bin/env python3
"""Generate proposal-oriented baseline vs stress comparison figure."""

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
        "--output-path",
        type=Path,
        default=Path("docs/images/proposal-baseline-vs-stress.png"),
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


def main() -> None:
    args = parse_args()
    baseline = _load_json(args.baseline_metrics)
    stress = _load_json(args.stress_metrics)

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

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_path, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
