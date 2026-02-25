"""Smoke tests for optional attribution backend adapters."""

from __future__ import annotations

import pytest

from ads.attribution import canonicalize_backend_name, create_backend
from ads.attribution.cea_backend import CEABackend
from ads.attribution.dda_backend import DDABackend
from ads.attribution.trak_backend import TRAKBackend


def test_cea_backend_placeholder() -> None:
    backend = CEABackend(train_corpus_path="artifacts/data/train_corpus.jsonl")
    with pytest.raises(NotImplementedError):
        backend.compute(prompt="p", answer="a", top_k=5)


def test_dda_backend_placeholder() -> None:
    backend = DDABackend(train_corpus_path="artifacts/data/train_corpus.jsonl")
    with pytest.raises(NotImplementedError):
        backend.compute(prompt="p", answer="a", top_k=5)


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


def test_create_backend_accepts_new_dda_variants_and_alias() -> None:
    tfidf_backend = create_backend(
        "dda_tfidf_proxy",
        train_corpus_path="artifacts/data/train_corpus.jsonl",
    )
    real_backend = create_backend(
        "dda_real",
        train_corpus_path="artifacts/data/train_corpus.jsonl",
    )
    with pytest.warns(DeprecationWarning):
        aliased_backend = create_backend(
            "dda",
            train_corpus_path="artifacts/data/train_corpus.jsonl",
        )

    assert isinstance(tfidf_backend, DDABackend)
    assert isinstance(real_backend, DDABackend)
    assert isinstance(aliased_backend, DDABackend)
    assert tfidf_backend.variant == "tfidf_proxy"
    assert real_backend.variant == "real"
    assert aliased_backend.variant == "tfidf_proxy"


def test_backend_alias_resolution_for_dda() -> None:
    with pytest.warns(DeprecationWarning):
        resolved = canonicalize_backend_name("dda")
    assert resolved == "dda_tfidf_proxy"
