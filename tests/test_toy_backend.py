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


def test_toy_backend_diffuse_spreads_mass() -> None:
    backend = ToyAttributionBackend(train_items=_train_rows(), seed=13, mode="diffuse")
    items = backend.compute(prompt="p", answer="a", top_k=20)
    scores = [item.score for item in items]
    features = compute_density_features(scores)
    assert features.top1_share < 0.10
