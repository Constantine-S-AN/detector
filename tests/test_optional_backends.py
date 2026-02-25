"""Smoke tests for optional attribution backend adapters."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ads.attribution import create_backend
from ads.attribution.cea_backend import CEABackend
from ads.attribution.dda_backend import DDATfidfProxyBackend
from ads.attribution.trak_backend import TRAKBackend


def _write_train_corpus(path: Path) -> None:
    rows = [{"train_id": f"train-{idx}", "text": f"sample text {idx}"} for idx in range(10)]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n")


def test_cea_backend_placeholder() -> None:
    backend = CEABackend(train_corpus_path="artifacts/data/train_corpus.jsonl")
    with pytest.raises(NotImplementedError):
        backend.compute(prompt="p", answer="a", top_k=5)


def test_dda_backend_smoke(tmp_path: Path) -> None:
    train_path = tmp_path / "train_corpus.jsonl"
    _write_train_corpus(train_path)

    backend = DDATfidfProxyBackend(train_corpus_path=train_path, cache_dir=tmp_path / "cache")
    items = backend.compute(prompt="p", answer="a", top_k=5, sample_meta={"sample_id": "s-1"})

    assert len(items) == 5


def test_create_backend_dda_alias_warns_deprecated(tmp_path: Path) -> None:
    train_path = tmp_path / "train_corpus.jsonl"
    _write_train_corpus(train_path)

    with pytest.warns(DeprecationWarning, match="backend=dda"):
        backend = create_backend("dda", train_corpus_path=train_path, seed=42)
    items = backend.compute(prompt="p", answer="a", top_k=3, sample_meta={"sample_id": "alias"})
    assert len(items) == 3


def test_trak_backend_fails_gracefully() -> None:
    backend = TRAKBackend(train_corpus_path="artifacts/data/train_corpus.jsonl")
    try:
        backend.compute(prompt="p", answer="a", top_k=5)
    except RuntimeError:
        assert True
    except NotImplementedError:
        assert True
    else:
        pytest.fail("Expected RuntimeError or NotImplementedError from TRAK backend placeholder")


def test_create_backend_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="Unsupported backend"):
        create_backend(
            "not-a-backend",
            train_corpus_path="artifacts/data/train_corpus.jsonl",
        )
