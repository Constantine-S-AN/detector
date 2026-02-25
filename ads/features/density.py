"""Density-style features derived from attribution scores."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Literal

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
    peakiness_ratio_score: float
    peakiness_ratio_prob: float
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


def _probabilities_from_top_k(
    top_k_scores: NDArray[np.float64],
    *,
    weight_mode: Literal["shifted", "softmax"] = "shifted",
    eps: float = 1e-12,
) -> NDArray[np.float64]:
    """Convert sorted top-k scores into a valid probability vector."""
    if top_k_scores.size == 0:
        return np.asarray([], dtype=np.float64)

    if weight_mode == "softmax":
        stabilized = top_k_scores - np.max(top_k_scores)
        exp_scores = np.exp(stabilized)
        total = float(np.sum(exp_scores))
        if total <= 0.0:
            return np.full(top_k_scores.shape, fill_value=1.0 / top_k_scores.size, dtype=np.float64)
        return exp_scores / total

    shifted = top_k_scores - np.min(top_k_scores)
    shifted = shifted + float(eps)
    total = float(np.sum(shifted))
    if total <= 0.0:
        return np.full(top_k_scores.shape, fill_value=1.0 / top_k_scores.size, dtype=np.float64)
    return shifted / total


def compute_h_at_k(
    scores: Sequence[float],
    k: int,
    *,
    normalize: bool = True,
    weight_mode: Literal["shifted", "softmax"] = "shifted",
    eps: float = 1e-12,
) -> dict[str, float | int]:
    """Compute entropy H@K on top-k influence scores.

    Steps:
    1. Sort scores descending.
    2. Truncate to top-k.
    3. Convert scores to probabilities.
    4. Compute `H@K = -sum(p_i log p_i)`.
    """
    if k <= 0:
        raise ValueError("k must be positive")

    raw_scores: NDArray[np.float64] = np.asarray(scores, dtype=np.float64)
    if raw_scores.size == 0:
        return {
            "h_at_k": 0.0,
            "h_at_k_normalized": 0.0,
            "k_requested": int(k),
            "k_effective": 0,
        }

    sorted_scores = np.sort(raw_scores)[::-1]
    k_effective = int(min(k, sorted_scores.size))
    top_k_scores = sorted_scores[:k_effective]
    probabilities = _probabilities_from_top_k(top_k_scores, weight_mode=weight_mode, eps=eps)

    safe_probabilities = np.clip(probabilities, eps, 1.0)
    entropy = float(-np.sum(probabilities * np.log(safe_probabilities)))

    entropy_normalized = entropy
    if normalize and k_effective > 1:
        entropy_normalized = float(entropy / np.log(float(k_effective)))

    return {
        "h_at_k": entropy,
        "h_at_k_normalized": entropy_normalized,
        "k_requested": int(k),
        "k_effective": k_effective,
    }


def compute_density_features(
    scores: Sequence[float],
    max_score_floor: float = 0.05,
    *,
    h_k: int = 20,
    h_weight_mode: Literal["shifted", "softmax"] = "shifted",
    h_eps: float = 1e-12,
) -> DensityFeatures:
    """Build density descriptors from attribution scores.

    Notes:
    - `entropy_top_k` is normalized H@K (defaults to K=20).
    - `max_score_floor` is used to set abstain behavior when max influence is too small.
    - `peakiness_ratio` remains as backward-compatible alias of `peakiness_ratio_score`.
    """
    if not scores:
        return DensityFeatures(
            entropy_top_k=1.0,
            top1_share=0.0,
            top5_share=0.0,
            peakiness_ratio=0.0,
            peakiness_ratio_score=0.0,
            peakiness_ratio_prob=0.0,
            gini=0.0,
            max_score=0.0,
            effective_k=0.0,
            abstain_flag=True,
            top_k=0,
        )

    raw_scores: NDArray[np.float64] = np.asarray(scores, dtype=np.float64)
    sorted_scores = np.sort(raw_scores)[::-1]

    entropy_payload = compute_h_at_k(
        scores=raw_scores,
        k=h_k,
        normalize=True,
        weight_mode=h_weight_mode,
        eps=h_eps,
    )

    probabilities = normalize_scores(raw_scores)
    sorted_probabilities = np.sort(probabilities)[::-1]
    score_probabilities = _probabilities_from_top_k(sorted_scores, weight_mode="softmax", eps=h_eps)

    entropy_raw = float(entropy_payload["h_at_k"])
    entropy_top_k = float(entropy_payload["h_at_k_normalized"])

    top1_share = float(sorted_probabilities[0])
    top5_share = float(np.sum(sorted_probabilities[: min(5, sorted_probabilities.size)]))
    top1_score = float(sorted_scores[0])
    sum_top5_score = float(np.sum(sorted_scores[: min(5, sorted_scores.size)]))
    peakiness_ratio_score = float(top1_score / max(sum_top5_score, h_eps))
    p1_softmax = float(score_probabilities[0])
    sum_top5_softmax = float(np.sum(score_probabilities[: min(5, score_probabilities.size)]))
    peakiness_ratio_prob = float(p1_softmax / max(sum_top5_softmax, h_eps))

    peakiness_ratio = peakiness_ratio_score
    max_score = float(np.max(raw_scores))
    effective_k = float(np.exp(entropy_raw))

    return DensityFeatures(
        entropy_top_k=entropy_top_k,
        top1_share=top1_share,
        top5_share=top5_share,
        peakiness_ratio=peakiness_ratio,
        peakiness_ratio_score=peakiness_ratio_score,
        peakiness_ratio_prob=peakiness_ratio_prob,
        gini=_compute_gini(sorted_probabilities),
        max_score=max_score,
        effective_k=effective_k,
        abstain_flag=max_score < max_score_floor,
        top_k=int(sorted_probabilities.size),
    )


def features_from_attributions(
    attributions: Sequence[AttributionItem],
    max_score_floor: float = 0.05,
    *,
    h_k: int = 20,
    h_weight_mode: Literal["shifted", "softmax"] = "shifted",
    h_eps: float = 1e-12,
) -> DensityFeatures:
    """Compute density features from attribution objects."""
    return compute_density_features(
        scores=[item.score for item in attributions],
        max_score_floor=max_score_floor,
        h_k=h_k,
        h_weight_mode=h_weight_mode,
        h_eps=h_eps,
    )
