"""Plotting functions for detector evaluation outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure


def _save_dual(fig: Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.png", dpi=160, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)


def render_evaluation_plots(
    metrics: dict[str, Any],
    predictions_frame: pd.DataFrame,
    output_dir: str | Path,
) -> dict[str, str]:
    """Render curves and score histograms for detector analysis."""
    plot_dir = Path(output_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    roc_points = metrics.get("curves", {}).get("roc", [])
    roc_fig, roc_ax = plt.subplots(figsize=(5.2, 4.0))
    if roc_points:
        roc_x = [point["x"] for point in roc_points]
        roc_y = [point["y"] for point in roc_points]
        roc_ax.plot(roc_x, roc_y, linewidth=2)
    roc_ax.plot([0, 1], [0, 1], linestyle="--")
    roc_ax.set_title("ROC Curve")
    roc_ax.set_xlabel("False Positive Rate")
    roc_ax.set_ylabel("True Positive Rate")
    _save_dual(roc_fig, plot_dir, "roc")

    pr_points = metrics.get("curves", {}).get("pr", [])
    pr_fig, pr_ax = plt.subplots(figsize=(5.2, 4.0))
    if pr_points:
        pr_x = [point["x"] for point in pr_points]
        pr_y = [point["y"] for point in pr_points]
        pr_ax.plot(pr_x, pr_y, linewidth=2)
    pr_ax.set_title("Precision-Recall Curve")
    pr_ax.set_xlabel("Recall")
    pr_ax.set_ylabel("Precision")
    _save_dual(pr_fig, plot_dir, "pr")

    calib_points = metrics.get("curves", {}).get("calibration", [])
    calib_fig, calib_ax = plt.subplots(figsize=(5.2, 4.0))
    if calib_points:
        calib_x = [point["x"] for point in calib_points]
        calib_y = [point["y"] for point in calib_points]
        calib_ax.plot(calib_x, calib_y, marker="o")
    calib_ax.plot([0, 1], [0, 1], linestyle="--")
    calib_ax.set_title("Calibration")
    calib_ax.set_xlabel("Predicted Probability")
    calib_ax.set_ylabel("Observed Positive Rate")
    _save_dual(calib_fig, plot_dir, "calib")

    abstain_points = metrics.get("curves", {}).get("abstain", [])
    abstain_fig, abstain_ax = plt.subplots(figsize=(5.2, 4.0))
    if abstain_points:
        abstain_x = [point["x"] for point in abstain_points]
        abstain_y = [point["y"] for point in abstain_points]
        abstain_ax.plot(abstain_x, abstain_y, marker="o")
    abstain_ax.set_title("Coverage vs Accuracy")
    abstain_ax.set_xlabel("Coverage")
    abstain_ax.set_ylabel("Accuracy (answered only)")
    abstain_ax.set_xlim(0, 1)
    abstain_ax.set_ylim(0, 1)
    _save_dual(abstain_fig, plot_dir, "abstain_curve")

    faithful_scores = predictions_frame.loc[
        predictions_frame["label_int"] == 1, "groundedness_score"
    ].to_numpy(dtype=float)
    hallucinated_scores = predictions_frame.loc[
        predictions_frame["label_int"] == 0, "groundedness_score"
    ].to_numpy(dtype=float)

    faithful_fig, faithful_ax = plt.subplots(figsize=(5.2, 4.0))
    faithful_ax.hist(faithful_scores, bins=min(10, max(3, faithful_scores.size)))
    faithful_ax.set_title("Faithful Score Histogram")
    faithful_ax.set_xlabel("Groundedness Score")
    faithful_ax.set_ylabel("Count")
    _save_dual(faithful_fig, plot_dir, "hist_faithful")

    hallucinated_fig, hallucinated_ax = plt.subplots(figsize=(5.2, 4.0))
    hallucinated_ax.hist(hallucinated_scores, bins=min(10, max(3, hallucinated_scores.size)))
    hallucinated_ax.set_title("Hallucinated Score Histogram")
    hallucinated_ax.set_xlabel("Groundedness Score")
    hallucinated_ax.set_ylabel("Count")
    _save_dual(hallucinated_fig, plot_dir, "hist_hallucinated")

    return {
        "roc": "roc.svg",
        "pr": "pr.svg",
        "calib": "calib.svg",
        "abstain_curve": "abstain_curve.svg",
        "hist_faithful": "hist_faithful.svg",
        "hist_hallucinated": "hist_hallucinated.svg",
    }
