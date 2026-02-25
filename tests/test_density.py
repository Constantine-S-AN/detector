"""Tests for density feature extraction."""

from __future__ import annotations

import math
import numpy as np

from ads.features.density import compute_density_features, compute_h_at_k, normalize_scores


def test_normalize_scores_sums_to_one() -> None:
    probabilities = normalize_scores([3.0, 1.0, 1.0])
    assert abs(float(probabilities.sum()) - 1.0) < 1e-9


def test_density_abstain_floor() -> None:
    features = compute_density_features([0.01, 0.01, 0.02], max_score_floor=0.05)
    assert features.abstain_flag is True


def test_density_peaked_has_lower_entropy() -> None:
    peaked = compute_density_features([4.0, 0.3, 0.2, 0.1])
    diffuse = compute_density_features([1.0, 1.0, 1.0, 1.0])
    assert peaked.entropy_top_k < diffuse.entropy_top_k


def test_normalize_scores_boundary_cases() -> None:
    empty = normalize_scores([])
    assert empty.size == 0

    all_zero = normalize_scores([0.0, 0.0, 0.0])
    assert all_zero.tolist() == [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]

    all_negative = normalize_scores([-1.0, -2.0])
    assert all_negative.tolist() == [0.5, 0.5]

    mixed = normalize_scores([-1.0, 2.0, 3.0])
    assert mixed.tolist() == [0.0, 0.4, 0.6]


def test_compute_h_at_k_peaked_vs_uniform() -> None:
    peaked = compute_h_at_k([10.0, 0.0, 0.0, 0.0, 0.0], k=5, weight_mode="shifted")
    uniform = compute_h_at_k([1.0, 1.0, 1.0, 1.0, 1.0], k=5, weight_mode="shifted")

    assert abs(float(peaked["h_at_k"]) - 0.0) < 1e-9
    assert abs(float(peaked["h_at_k_normalized"]) - 0.0) < 1e-9
    assert abs(float(uniform["h_at_k"]) - math.log(5.0)) < 1e-9
    assert abs(float(uniform["h_at_k_normalized"]) - 1.0) < 1e-9


def test_compute_h_at_k_boundary_cases() -> None:
    payload = compute_h_at_k([3.0, 2.0, 1.0], k=10)
    assert payload["k_requested"] == 10
    assert payload["k_effective"] == 3

    empty = compute_h_at_k([], k=5)
    assert empty["k_effective"] == 0
    assert float(empty["h_at_k"]) == 0.0
    assert float(empty["h_at_k_normalized"]) == 0.0


def test_compute_h_at_k_shifted_handles_negative_scores() -> None:
    payload = compute_h_at_k([-3.0, -2.0, -1.0], k=3, weight_mode="shifted")
    assert np.isfinite(float(payload["h_at_k"]))
    assert np.isfinite(float(payload["h_at_k_normalized"]))


def test_compute_h_at_k_shifted_is_scale_invariant() -> None:
    scores = [5.0, 2.0, 0.2, -0.1, -0.4]
    scaled = [item * 3.0 for item in scores]

    base = compute_h_at_k(scores, k=5, weight_mode="shifted")
    scaled_payload = compute_h_at_k(scaled, k=5, weight_mode="shifted")

    assert abs(float(base["h_at_k_normalized"]) - float(scaled_payload["h_at_k_normalized"])) < 1e-9


def test_density_features_exports_peakiness_versions() -> None:
    features = compute_density_features([3.0, 1.0, 0.5, 0.25], h_k=3)
    assert features.peakiness_ratio == features.peakiness_ratio_score
    assert features.peakiness_ratio_prob > 0.0
