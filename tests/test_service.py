"""Tests for shared scan service."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ads.service import clear_runtime_caches, scan_sample


def _write_train_corpus(path: Path) -> None:
    rows = [{"train_id": f"train-{idx}", "text": f"training text {idx}"} for idx in range(30)]
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_scan_sample_threshold_mode(tmp_path: Path) -> None:
    train_path = tmp_path / "train_corpus.jsonl"
    _write_train_corpus(train_path)

    result = scan_sample(
        prompt="Tell me about the Pacific Ocean.",
        answer="This is speculative and uncertain.",
        backend_name="toy",
        top_k=10,
        seed=42,
        train_corpus_path=train_path,
        method="threshold",
    )

    assert "features" in result
    assert "prediction" in result
    assert "thresholds" in result
    assert len(result["top_influential"]) == 10
    assert result["requested_detector"] == "threshold"
    assert result["detector"] == "threshold"
    assert result["fallback_reason"] is None


def test_scan_sample_logistic_missing_with_fallback(tmp_path: Path) -> None:
    train_path = tmp_path / "train_corpus.jsonl"
    _write_train_corpus(train_path)

    result = scan_sample(
        prompt="Give one grounded claim.",
        answer="This answer is speculative and uncertain.",
        backend_name="toy",
        top_k=10,
        seed=42,
        train_corpus_path=train_path,
        method="logistic",
        model_path=tmp_path / "missing.joblib",
        allow_fallback=True,
    )

    assert result["requested_detector"] == "logistic"
    assert result["detector"] == "threshold"
    assert result["fallback_reason"] == "logistic_model_missing"


def test_scan_sample_logistic_missing_without_fallback_raises(tmp_path: Path) -> None:
    train_path = tmp_path / "train_corpus.jsonl"
    _write_train_corpus(train_path)
    model_path = tmp_path / "missing.joblib"

    with pytest.raises(ValueError, match=str(model_path)):
        scan_sample(
            prompt="Give one grounded claim.",
            answer="This answer is speculative and uncertain.",
            backend_name="toy",
            top_k=10,
            seed=42,
            train_corpus_path=train_path,
            method="logistic",
            model_path=model_path,
            allow_fallback=False,
        )


def test_clear_runtime_caches_is_safe() -> None:
    clear_runtime_caches()
    clear_runtime_caches()
