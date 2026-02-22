"""Smoke tests for optional attribution backend adapters."""

from __future__ import annotations

import pytest

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
