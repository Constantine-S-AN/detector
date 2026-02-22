"""Tests for synthetic toy attribution backend."""

from __future__ import annotations

from ads.attribution.toy_backend import ToyAttributionBackend
from ads.features.density import compute_density_features


def _train_rows() -> list[dict[str, object]]:
    return [{"train_id": f"train-{idx}", "text": f"sample {idx}"} for idx in range(50)]


def test_toy_backend_peaked_concentrates_mass() -> None:
    backend = ToyAttributionBackend(train_items=_train_rows(), seed=13, mode="peaked")
    items = backend.compute(prompt="p", answer="a", top_k=20)
    scores = [item.score for item in items]
    features = compute_density_features(scores)
    assert features.top1_share > 0.20
    assert features.peakiness_ratio > 0.35


def test_toy_backend_diffuse_spreads_mass() -> None:
    backend = ToyAttributionBackend(train_items=_train_rows(), seed=13, mode="diffuse")
    items = backend.compute(prompt="p", answer="a", top_k=20)
    scores = [item.score for item in items]
    features = compute_density_features(scores)
    assert features.top1_share < 0.10
    assert 0.12 <= features.peakiness_ratio <= 0.30


def test_toy_backend_peaked_has_higher_peakiness_ratio_than_diffuse() -> None:
    peaked_backend = ToyAttributionBackend(train_items=_train_rows(), seed=13, mode="peaked")
    diffuse_backend = ToyAttributionBackend(train_items=_train_rows(), seed=13, mode="diffuse")

    peaked_scores = [
        item.score for item in peaked_backend.compute(prompt="p", answer="a", top_k=20)
    ]
    diffuse_scores = [
        item.score for item in diffuse_backend.compute(prompt="p", answer="a", top_k=20)
    ]

    peaked_features = compute_density_features(peaked_scores)
    diffuse_features = compute_density_features(diffuse_scores)
    assert peaked_features.peakiness_ratio > diffuse_features.peakiness_ratio


def test_toy_backend_distributed_mode_is_intermediate() -> None:
    peaked_backend = ToyAttributionBackend(train_items=_train_rows(), seed=13, mode="peaked")
    diffuse_backend = ToyAttributionBackend(train_items=_train_rows(), seed=13, mode="diffuse")
    distributed_backend = ToyAttributionBackend(
        train_items=_train_rows(),
        seed=13,
        mode="distributed",
    )

    peaked_items = peaked_backend.compute(prompt="p", answer="a", top_k=20)
    diffuse_items = diffuse_backend.compute(prompt="p", answer="a", top_k=20)
    distributed_items = distributed_backend.compute(prompt="p", answer="a", top_k=20)

    peaked_features = compute_density_features([item.score for item in peaked_items])
    diffuse_features = compute_density_features([item.score for item in diffuse_items])
    distributed_features = compute_density_features([item.score for item in distributed_items])

    assert all(item.meta["mode"] == "distributed" for item in distributed_items)
    assert (
        diffuse_features.top1_share < distributed_features.top1_share < peaked_features.top1_share
    )
    assert (
        diffuse_features.peakiness_ratio
        < distributed_features.peakiness_ratio
        < peaked_features.peakiness_ratio
    )
