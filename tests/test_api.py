"""Integration-style tests for ADS FastAPI endpoints."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from ads.api import app
from ads.service import clear_runtime_caches


def _write_train_corpus(base_dir: Path) -> None:
    corpus_dir = base_dir / "artifacts" / "data"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        {"train_id": f"train-{index}", "text": f"reference text {index}"} for index in range(50)
    ]
    (corpus_dir / "train_corpus.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_health_endpoint() -> None:
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_scan_and_cache_clear_endpoints(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    _write_train_corpus(tmp_path)
    clear_runtime_caches()

    client = TestClient(app)
    scan_response = client.post(
        "/scan",
        json={
            "prompt": "Give me one grounded fact about Tokyo.",
            "answer": "According to the provided sources, Tokyo is the capital city of Japan.",
            "method": "threshold",
            "top_k": 12,
            "backend": "toy",
            "score_threshold": 0.61,
            "max_score_floor": 0.03,
        },
    )
    assert scan_response.status_code == 200
    payload = scan_response.json()
    assert payload["requested_detector"] == "threshold"
    assert payload["detector"] == "threshold"
    assert payload["fallback_reason"] is None
    assert payload["prediction"]["groundedness_score"] >= 0.0
    assert len(payload["top_influential"]) == 12
    first_item = payload["top_influential"][0]
    assert set(first_item.keys()) == {"train_id", "score", "text", "meta"}
    assert payload["thresholds"]["score_threshold"] == 0.61
    assert payload["thresholds"]["max_score_floor"] == 0.03

    clear_response = client.post("/runtime/cache/clear")
    assert clear_response.status_code == 200
    assert clear_response.json() == {"status": "cleared"}


def test_scan_request_validation() -> None:
    client = TestClient(app)
    response = client.post(
        "/scan",
        json={
            "prompt": "p",
            "answer": "a",
            "top_k": 0,
            "backend": "toy",
            "method": "threshold",
        },
    )
    assert response.status_code == 422


def test_scan_logistic_missing_is_strict_by_default(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    _write_train_corpus(tmp_path)
    clear_runtime_caches()

    client = TestClient(app)
    response = client.post(
        "/scan",
        json={
            "prompt": "Give one grounded fact about Tokyo.",
            "answer": "According to sources, Tokyo is in Japan.",
            "backend": "toy",
            "method": "logistic",
            "top_k": 10,
        },
    )
    assert response.status_code == 400
    payload = response.json()
    assert payload["code"] == "MODEL_MISSING"
    assert "logistic model missing:" in payload["error"]
