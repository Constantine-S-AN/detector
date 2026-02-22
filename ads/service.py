"""Shared scan service used by CLI and API modes."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from ads.attribution import BackendName, create_backend
from ads.attribution.base import AttributionBackend
from ads.detector.logistic import LogisticDetector
from ads.detector.threshold import ThresholdDetector
from ads.features.density import features_from_attributions

DetectorMethod = Literal["threshold", "logistic"]


def _resolve_with_mtime(path: str | Path) -> tuple[str, int]:
    """Resolve a filesystem path and attach mtime for cache invalidation."""
    resolved = Path(path).resolve()
    if resolved.exists():
        return str(resolved), resolved.stat().st_mtime_ns
    return str(resolved), -1


@lru_cache(maxsize=32)
def _load_backend_cached(
    backend_name: BackendName,
    train_corpus_key: str,
    train_corpus_mtime_ns: int,
    seed: int,
) -> AttributionBackend:
    """Load and cache attribution backend instances."""
    del train_corpus_mtime_ns
    return create_backend(
        backend_name,
        train_corpus_path=train_corpus_key,
        seed=seed,
    )


@lru_cache(maxsize=8)
def _load_logistic_cached(model_key: str, model_mtime_ns: int) -> LogisticDetector:
    """Load and cache logistic detector instances."""
    del model_mtime_ns
    return LogisticDetector.load(model_key)


def clear_runtime_caches() -> None:
    """Clear in-process backend/model caches."""
    _load_backend_cached.cache_clear()
    _load_logistic_cached.cache_clear()


def scan_sample(
    prompt: str,
    answer: str,
    *,
    backend_name: BackendName = "toy",
    top_k: int = 20,
    seed: int = 42,
    train_corpus_path: str | Path = "artifacts/data/train_corpus.jsonl",
    method: DetectorMethod = "logistic",
    model_path: str | Path = "artifacts/models/logistic.joblib",
    allow_fallback: bool = True,
    max_score_floor: float = 0.05,
    score_threshold: float = 0.55,
    decision_threshold: float = 0.5,
) -> dict[str, Any]:
    """Run attribution + feature extraction + detector inference."""
    train_corpus_key, train_corpus_mtime_ns = _resolve_with_mtime(train_corpus_path)
    backend = _load_backend_cached(
        backend_name,
        train_corpus_key,
        train_corpus_mtime_ns,
        seed,
    )
    attributions = backend.compute(prompt=prompt, answer=answer, top_k=top_k)
    features = features_from_attributions(attributions, max_score_floor=max_score_floor)

    model_key, model_mtime_ns = _resolve_with_mtime(model_path)
    fallback_reason: str | None = None
    if method == "logistic" and Path(model_key).exists():
        detector = _load_logistic_cached(model_key, model_mtime_ns)
        output = detector.predict_output(features, threshold=decision_threshold)
        detector_used = "logistic"
    elif method == "logistic" and not allow_fallback:
        raise ValueError(f"logistic model missing: {model_key}")
    else:
        if method == "logistic" and not Path(model_key).exists():
            fallback_reason = "logistic_model_missing"
        threshold_detector = ThresholdDetector(
            score_threshold=score_threshold, score_floor=max_score_floor
        )
        output = threshold_detector.predict(features)
        detector_used = "threshold"

    return {
        "prompt": prompt,
        "answer": answer,
        "requested_detector": method,
        "detector": detector_used,
        "fallback_reason": fallback_reason,
        "features": features.to_dict(),
        "prediction": output.to_dict(),
        "thresholds": {
            "decision_threshold": decision_threshold,
            "score_threshold": score_threshold,
            "max_score_floor": max_score_floor,
        },
        "top_influential": [item.to_dict() for item in attributions],
    }
