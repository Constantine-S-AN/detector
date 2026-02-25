"""Smoke tests for minimal runnable DDA backend."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ads.attribution.dda_backend import DDARealBackend, DDATfidfProxyBackend


def _write_train_corpus(path: Path, n_rows: int = 10) -> None:
    rows = [
        {
            "train_id": f"train-{idx}",
            "text": f"Tokyo city fact {idx}. Capital of Japan and population context {idx}.",
            "source": "synthetic_train_pool",
        }
        for idx in range(n_rows)
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n")


def test_dda_backend_compute_smoke(tmp_path: Path) -> None:
    train_path = tmp_path / "train_corpus.jsonl"
    cache_dir = tmp_path / "cache"
    _write_train_corpus(train_path, n_rows=10)

    backend = DDATfidfProxyBackend(
        train_corpus_path=train_path,
        seed=42,
        alpha=0.2,
        cache_dir=cache_dir,
        model_id="dda_tfidf_smoke",
    )
    items = backend.compute(
        prompt="Give one factual statement about Tokyo",
        answer="Tokyo is the capital city of Japan.",
        top_k=20,
        sample_meta={"sample_id": "sample-001"},
    )

    assert len(items) == 10
    scores = [item.score for item in items]
    assert scores == sorted(scores, reverse=True)
    assert all(np.isfinite(score) for score in scores)

    first = items[0]
    assert isinstance(first.train_id, str)
    assert isinstance(first.score, float)
    assert isinstance(first.rank, int)
    assert isinstance(first.text, str)
    assert isinstance(first.source, str)
    assert isinstance(first.meta, dict)


def test_dda_backend_cache_hit(tmp_path: Path) -> None:
    train_path = tmp_path / "train_corpus.jsonl"
    cache_dir = tmp_path / "cache"
    _write_train_corpus(train_path, n_rows=10)

    backend = DDATfidfProxyBackend(
        train_corpus_path=train_path,
        seed=42,
        alpha=0.2,
        cache_dir=cache_dir,
        model_id="dda_tfidf_smoke",
    )

    first = backend.compute(
        prompt="What is Tokyo?",
        answer="Tokyo is the capital city of Japan.",
        top_k=5,
        sample_meta={"sample_id": "cache-sample"},
    )
    second = backend.compute(
        prompt="What is Tokyo?",
        answer="Tokyo is the capital city of Japan.",
        top_k=5,
        sample_meta={"sample_id": "cache-sample"},
    )

    assert [item.to_dict() for item in first] == [item.to_dict() for item in second]
    assert any(cache_dir.glob("*.json"))


def test_dda_real_backend_smoke_optional_dep(tmp_path: Path) -> None:
    pytest.importorskip("sentence_transformers")

    train_path = tmp_path / "train_corpus.jsonl"
    cache_dir = tmp_path / "cache-real"
    _write_train_corpus(train_path, n_rows=10)

    backend = DDARealBackend(
        train_corpus_path=train_path,
        seed=42,
        alpha=0.2,
        cache_dir=cache_dir,
        model_id="sentence-transformers/all-MiniLM-L6-v2",
        device="cpu",
    )
    items = backend.compute(
        prompt="Give one factual statement about Tokyo",
        answer="Tokyo is the capital city of Japan.",
        top_k=5,
        sample_meta={"sample_id": "sample-real"},
    )

    assert len(items) == 5
    scores = [item.score for item in items]
    assert scores == sorted(scores, reverse=True)
    assert all(np.isfinite(score) for score in scores)
    for item in items:
        payload = item.to_dict()
        assert isinstance(payload["train_id"], str)
        assert isinstance(payload["score"], float)
        assert isinstance(payload["rank"], int)
        assert isinstance(payload["text"], str)
        assert isinstance(payload["source"], str)
        assert isinstance(payload["meta"], dict)
