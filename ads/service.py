"""Shared scan service used by CLI and API modes."""

from __future__ import annotations

from functools import lru_cache
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal

from ads.attribution import (
    BackendInputName,
    BackendName,
    canonicalize_backend_name,
    create_backend,
)
from ads.attribution.base import AttributionBackend, AttributionItem
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


def _hash_text(value: str) -> str:
    """Build a short deterministic hash for an evidence snippet."""
    return sha256(value.encode("utf-8")).hexdigest()[:16]


def _snippet_text(value: str, max_chars: int) -> str:
    """Build a compact single-line snippet with a strict character cap."""
    normalized_text = " ".join(value.split())
    if len(normalized_text) <= max_chars:
        return normalized_text
    if max_chars <= 3:
        return normalized_text[:max_chars]
    return normalized_text[: max_chars - 3].rstrip() + "..."


def _normalized_non_negative(scores: list[float]) -> list[float]:
    """Normalize scores into non-negative shares."""
    if not scores:
        return []
    clipped = [max(float(score), 0.0) for score in scores]
    total = float(sum(clipped))
    if total <= 0.0:
        uniform = 1.0 / float(len(clipped))
        return [uniform for _ in clipped]
    return [value / total for value in clipped]


def _build_evidence_payload(
    attributions: list[AttributionItem],
    *,
    redact: bool,
    redaction_snippet_chars: int,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Build redacted top-influential payload and summary evidence fields."""
    snippet_chars = max(8, int(redaction_snippet_chars))
    sorted_attributions = sorted(attributions, key=lambda item: float(item.score), reverse=True)
    score_shares = _normalized_non_negative([float(item.score) for item in sorted_attributions])

    top_influential_payload: list[dict[str, Any]] = []
    highlights: list[dict[str, Any]] = []
    for index, item in enumerate(sorted_attributions):
        full_text = str(item.text)
        snippet = _snippet_text(full_text, snippet_chars)
        text_hash = _hash_text(full_text)
        top_influential_payload.append(
            {
                "train_id": str(item.train_id),
                "score": float(item.score),
                "text": snippet if redact else full_text,
                "text_hash": text_hash,
                "meta": dict(item.meta),
            }
        )
        if index < 3:
            highlights.append(
                {
                    "train_id": str(item.train_id),
                    "score": float(item.score),
                    "score_share": float(score_shares[index]),
                    "snippet": snippet,
                    "text_hash": text_hash,
                }
            )

    evidence_summary = {
        "item_count": len(sorted_attributions),
        "top1_share": float(score_shares[0]) if score_shares else 0.0,
        "top5_share": float(sum(score_shares[:5])),
        "top10_share": float(sum(score_shares[:10])),
        "highlights": highlights,
    }
    redaction_payload = {
        "enabled": bool(redact),
        "snippet_chars": snippet_chars,
        "hash_algorithm": "sha256",
        "full_text_included": not redact,
    }
    return top_influential_payload, evidence_summary, redaction_payload


def scan_sample(
    prompt: str,
    answer: str,
    *,
    backend_name: BackendInputName = "toy",
    top_k: int = 20,
    seed: int = 42,
    train_corpus_path: str | Path = "artifacts/data/train_corpus.jsonl",
    method: DetectorMethod = "logistic",
    model_path: str | Path = "artifacts/models/logistic.joblib",
    allow_fallback: bool = True,
    max_score_floor: float = 0.05,
    score_threshold: float = 0.55,
    decision_threshold: float = 0.5,
    redact: bool = True,
    redaction_snippet_chars: int = 96,
) -> dict[str, Any]:
    """Run attribution + feature extraction + detector inference."""
    canonical_backend_name = canonicalize_backend_name(backend_name)
    train_corpus_key, train_corpus_mtime_ns = _resolve_with_mtime(train_corpus_path)
    backend = _load_backend_cached(
        canonical_backend_name,
        train_corpus_key,
        train_corpus_mtime_ns,
        seed,
    )
    attributions = backend.compute(prompt=prompt, answer=answer, top_k=top_k)
    features = features_from_attributions(attributions, max_score_floor=max_score_floor)
    top_influential_payload, evidence_summary, redaction_payload = _build_evidence_payload(
        attributions,
        redact=redact,
        redaction_snippet_chars=redaction_snippet_chars,
    )

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
        "top_influential": top_influential_payload,
        "evidence_summary": evidence_summary,
        "redaction": redaction_payload,
    }
