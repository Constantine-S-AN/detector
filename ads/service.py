"""Shared scan service used by CLI and API modes."""

from __future__ import annotations

from functools import lru_cache
import hashlib
from pathlib import Path
from typing import Any, Literal

from ads.attribution import BackendName, create_backend
from ads.attribution.base import AttributionBackend
from ads.detector.logistic import LogisticDetector
from ads.detector.threshold import ThresholdDetector
from ads.features.density import compute_h_at_k, features_from_attributions

DetectorMethod = Literal["threshold", "logistic"]


def _shifted_top_k_probabilities(scores: list[float], k: int, eps: float = 1e-12) -> list[float]:
    if not scores:
        return []
    k_effective = min(int(k), len(scores))
    top_scores = sorted(scores, reverse=True)[:k_effective]
    min_score = min(top_scores)
    shifted = [float(score - min_score + eps) for score in top_scores]
    total = sum(shifted)
    if total <= 0.0:
        return [1.0 / k_effective for _ in range(k_effective)]
    return [value / total for value in shifted]


def _redact_meta(meta: dict[str, Any], redact: bool) -> dict[str, Any]:
    if not redact:
        return dict(meta)
    blocked_tokens = ("path", "raw", "full", "source_text", "content")
    sanitized: dict[str, Any] = {}
    for key, value in meta.items():
        lowered = str(key).lower()
        if any(token in lowered for token in blocked_tokens):
            continue
        sanitized[str(key)] = value
    return sanitized


def _snippet_payload(
    text: str,
    *,
    redact: bool,
    snippet_max_len: int,
    allow_full_text: bool,
) -> tuple[str, str | None]:
    if allow_full_text and not redact:
        return text, None

    snippet = text[:snippet_max_len]
    if len(text) > snippet_max_len:
        snippet += "…"
    text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
    if redact:
        return snippet, text_hash
    return snippet, None


def _build_evidence_summary(
    attributions: list[dict[str, Any]],
    *,
    top_k: int,
    h_k: int,
    max_score_floor: float,
    entropy_top_k: float,
    peakiness_ratio_score: float,
    max_score: float,
    abstain_flag: bool,
    redact: bool,
    snippet_max_len: int,
    allow_full_text: bool,
    preview_n: int = 5,
) -> dict[str, Any]:
    scores = [float(item["score"]) for item in attributions]
    probabilities = _shifted_top_k_probabilities(scores, k=top_k)
    h_payload = compute_h_at_k(scores=scores, k=h_k, normalize=False, weight_mode="shifted")

    if max_score < max_score_floor:
        abstain_reason = "max_score_below_floor"
    elif abstain_flag:
        abstain_reason = "detector_abstained"
    else:
        abstain_reason = None

    preview: list[dict[str, Any]] = []
    for item in attributions[:preview_n]:
        text = str(item.get("text", ""))
        snippet, text_hash = _snippet_payload(
            text,
            redact=redact,
            snippet_max_len=snippet_max_len,
            allow_full_text=allow_full_text,
        )
        row: dict[str, Any] = {
            "train_id": str(item.get("train_id", "")),
            "rank": int(item.get("rank", 0)),
            "score": float(item.get("score", 0.0)),
        }
        if snippet:
            row["snippet"] = snippet
        if text_hash is not None:
            row["text_hash"] = text_hash
        preview.append(row)

    return {
        "top1_mass": float(probabilities[0]) if probabilities else 0.0,
        "top5_mass": float(sum(probabilities[: min(5, len(probabilities))])) if probabilities else 0.0,
        "h_at_k": float(h_payload["h_at_k"]),
        "h_k": int(h_payload["k_effective"]),
        "entropy_top_k_normalized": float(entropy_top_k),
        "peakiness_ratio_score": float(peakiness_ratio_score),
        "max_score": float(max_score),
        "abstain_flag": bool(abstain_flag),
        "abstain_reason": abstain_reason,
        "top_items_preview": preview,
    }


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
    redact: bool = True,
    snippet_max_len: int = 160,
    allow_full_text: bool = False,
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

    top_influential: list[dict[str, Any]] = []
    for item in attributions:
        item_payload = item.to_dict()
        snippet, text_hash = _snippet_payload(
            str(item_payload.get("text", "")),
            redact=redact,
            snippet_max_len=snippet_max_len,
            allow_full_text=allow_full_text,
        )
        item_payload["text"] = snippet
        meta_payload = _redact_meta(dict(item_payload.get("meta", {})), redact=redact)
        if text_hash is not None:
            meta_payload["text_hash"] = text_hash
        item_payload["meta"] = meta_payload
        top_influential.append(item_payload)

    evidence_summary = _build_evidence_summary(
        top_influential,
        top_k=top_k,
        h_k=top_k,
        max_score_floor=max_score_floor,
        entropy_top_k=float(features.entropy_top_k),
        peakiness_ratio_score=float(features.peakiness_ratio_score),
        max_score=float(features.max_score),
        abstain_flag=bool(output.abstain_flag),
        redact=redact,
        snippet_max_len=snippet_max_len,
        allow_full_text=allow_full_text,
        preview_n=5,
    )

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
        "evidence_summary": evidence_summary,
        "redaction": {
            "enabled": bool(redact),
            "snippet_max_len": int(snippet_max_len),
            "allow_full_text": bool(allow_full_text),
        },
        "top_influential": top_influential,
    }
