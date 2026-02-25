"""Schema validation tests for attribution outputs across backends."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ads.attribution import create_backend


@pytest.fixture()
def train_corpus_path(tmp_path: Path) -> Path:
    rows = [
        {
            "train_id": f"train-{idx}",
            "text": f"training snippet {idx}",
            "source": "synthetic",
        }
        for idx in range(12)
    ]
    path = tmp_path / "train_corpus.jsonl"
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n")
    return path


def _assert_item_schema(item: dict[str, object]) -> None:
    assert "train_id" in item and isinstance(item["train_id"], str)
    assert "score" in item and isinstance(item["score"], float)
    assert "rank" in item and isinstance(item["rank"], int)
    assert "text" in item and isinstance(item["text"], str)
    assert "source" in item and isinstance(item["source"], str)
    assert "meta" in item and isinstance(item["meta"], dict)


def test_toy_backend_output_schema(train_corpus_path: Path) -> None:
    backend = create_backend("toy", train_corpus_path=train_corpus_path, seed=42)
    items = backend.compute(
        prompt="p",
        answer="a",
        top_k=5,
        sample_meta={"label": "faithful", "attribution_mode": "peaked"},
    )
    assert len(items) == 5
    for item in items:
        _assert_item_schema(item.to_dict())


def test_dda_backend_output_schema(train_corpus_path: Path, tmp_path: Path) -> None:
    backend = create_backend(
        "dda",
        train_corpus_path=train_corpus_path,
        seed=42,
        dda_cache_dir=tmp_path / "cache",
    )
    items = backend.compute(
        prompt="Tell me a fact",
        answer="This is one grounded fact",
        top_k=5,
        sample_meta={"sample_id": "x-1"},
    )
    assert len(items) == 5
    for item in items:
        _assert_item_schema(item.to_dict())


def test_optional_backends_keep_schema_signature(train_corpus_path: Path) -> None:
    for backend_name in ("trak", "cea"):
        backend = create_backend(backend_name, train_corpus_path=train_corpus_path, seed=42)
        with pytest.raises((RuntimeError, NotImplementedError)):
            backend.compute(
                prompt="p",
                answer="a",
                top_k=3,
                sample_meta={"sample_id": "z"},
                attribution_mode="peaked",
            )


def test_dda_real_backend_schema_optional_dep(train_corpus_path: Path, tmp_path: Path) -> None:
    pytest.importorskip("sentence_transformers")
    backend = create_backend(
        "dda_real",
        train_corpus_path=train_corpus_path,
        seed=42,
        dda_cache_dir=tmp_path / "cache",
        dda_model_id="sentence-transformers/all-MiniLM-L6-v2",
        dda_device="cpu",
    )
    items = backend.compute(
        prompt="Tell me a fact",
        answer="This is one grounded fact",
        top_k=5,
        sample_meta={"sample_id": "real-1"},
    )
    assert len(items) == 5
    for item in items:
        _assert_item_schema(item.to_dict())
