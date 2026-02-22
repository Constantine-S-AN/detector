"""Density-style features derived from attribution scores."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import NDArray

from ads.attribution.base import AttributionItem


@dataclass(slots=True)
class DensityFeatures:
    """Feature bundle used by groundedness detectors."""

    entropy_top_k: float
    top1_share: float
    top5_share: float
    peakiness_ratio: float
    gini: float
    max_score: float
    effective_k: float
    abstain_flag: bool
    top_k: int

    def to_dict(self) -> dict[str, float | bool | int]:
        """Convert to a dictionary for dataframe/JSON export."""
        return asdict(self)


def normalize_scores(scores: Sequence[float] | NDArray[np.float64]) -> NDArray[np.float64]:
    """Normalize scores to a valid probability simplex."""
    raw_scores: NDArray[np.float64] = np.asarray(scores, dtype=np.float64)
    if raw_scores.size == 0:
        return raw_scores
    clipped_scores = np.maximum(raw_scores, 0.0)
    total = float(np.sum(clipped_scores))
    if total <= 0.0:
        return np.full(clipped_scores.shape, fill_value=1.0 / clipped_scores.size, dtype=np.float64)
    return clipped_scores / total


def _compute_gini(probabilities: NDArray[np.float64]) -> float:
    """Compute Gini coefficient from probabilities."""
    if probabilities.size == 0:
        return 0.0
    sorted_prob = np.sort(probabilities)
    cumulative = np.cumsum(sorted_prob)
    n_points = sorted_prob.size
    gini_value = (n_points + 1 - 2 * np.sum(cumulative) / cumulative[-1]) / n_points
    return float(gini_value)


def compute_density_features(
    scores: Sequence[float], max_score_floor: float = 0.05
) -> DensityFeatures:
    """Build density descriptors from attribution scores."""
    if not scores:
        return DensityFeatures(
            entropy_top_k=1.0,
            top1_share=0.0,
            top5_share=0.0,
            peakiness_ratio=0.0,
            gini=0.0,
            max_score=0.0,
            effective_k=0.0,
            abstain_flag=True,
            top_k=0,
        )

    raw_scores: NDArray[np.float64] = np.asarray(scores, dtype=np.float64)
    probabilities = normalize_scores(raw_scores)
    sorted_probabilities = np.sort(probabilities)[::-1]

    entropy_raw = -np.sum(sorted_probabilities * np.log(np.clip(sorted_probabilities, 1e-12, 1.0)))
    entropy_normalizer = np.log(sorted_probabilities.size) if sorted_probabilities.size > 1 else 1.0
    entropy_top_k = float(entropy_raw / entropy_normalizer) if entropy_normalizer > 0 else 0.0

    top1_share = float(sorted_probabilities[0])
    top5_share = float(np.sum(sorted_probabilities[: min(5, sorted_probabilities.size)]))
    peakiness_ratio = float(top1_share / max(top5_share, 1e-12))
    max_score = float(np.max(raw_scores))
    effective_k = float(np.exp(entropy_raw))

    return DensityFeatures(
        entropy_top_k=entropy_top_k,
        top1_share=top1_share,
        top5_share=top5_share,
        peakiness_ratio=peakiness_ratio,
        gini=_compute_gini(sorted_probabilities),
        max_score=max_score,
        effective_k=effective_k,
        abstain_flag=max_score < max_score_floor,
        top_k=int(sorted_probabilities.size),
    )


def features_from_attributions(
    attributions: Sequence[AttributionItem], max_score_floor: float = 0.05
) -> DensityFeatures:
    """Compute density features from attribution objects."""
    return compute_density_features(
        scores=[item.score for item in attributions],
        max_score_floor=max_score_floor,
    )
