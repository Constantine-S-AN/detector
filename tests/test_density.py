"""Tests for density feature extraction."""

from __future__ import annotations

from ads.features.density import compute_density_features, normalize_scores


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
