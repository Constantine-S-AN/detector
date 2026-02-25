"""Attribution backend registry."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, cast
from warnings import warn

from ads.attribution.base import AttributionBackend
from ads.attribution.toy_backend import ToyAttributionBackend, ToyMode

BackendName = Literal["toy", "trak", "cea", "dda_tfidf_proxy", "dda_real"]
BackendInputName = Literal["toy", "trak", "cea", "dda_tfidf_proxy", "dda_real", "dda"]

BACKEND_NAMES: tuple[BackendName, ...] = (
    "toy",
    "trak",
    "cea",
    "dda_tfidf_proxy",
    "dda_real",
)
BACKEND_INPUT_NAMES: tuple[BackendInputName, ...] = (
    "toy",
    "trak",
    "cea",
    "dda_tfidf_proxy",
    "dda_real",
    "dda",
)

_BACKEND_ALIAS_MAP: dict[str, BackendName] = {"dda": "dda_tfidf_proxy"}


def canonicalize_backend_name(name: str) -> BackendName:
    """Resolve legacy aliases into canonical backend names."""
    normalized = name.strip().lower()
    if normalized in BACKEND_NAMES:
        return cast(BackendName, normalized)
    if normalized in _BACKEND_ALIAS_MAP:
        warn(
            "backend='dda' is deprecated and maps to 'dda_tfidf_proxy'. "
            "Please update clients to the canonical backend name.",
            category=DeprecationWarning,
            stacklevel=2,
        )
        return _BACKEND_ALIAS_MAP[normalized]
    expected = ", ".join(BACKEND_INPUT_NAMES)
    raise ValueError(f"Unsupported backend '{name}'. Expected one of: {expected}.")


def create_backend(
    name: str,
    train_corpus_path: str | Path,
    seed: int = 42,
    mode: ToyMode = "auto",
) -> AttributionBackend:
    """Build a backend instance by name."""
    canonical_name = canonicalize_backend_name(name)

    if canonical_name == "toy":
        return ToyAttributionBackend.from_jsonl(train_corpus_path, seed=seed, mode=mode)
    if canonical_name == "trak":
        from ads.attribution.trak_backend import TRAKBackend

        return TRAKBackend(train_corpus_path=train_corpus_path, seed=seed)
    if canonical_name == "cea":
        from ads.attribution.cea_backend import CEABackend

        return CEABackend(train_corpus_path=train_corpus_path, seed=seed)
    if canonical_name in {"dda_tfidf_proxy", "dda_real"}:
        from ads.attribution.dda_backend import DDABackend

        variant = "tfidf_proxy" if canonical_name == "dda_tfidf_proxy" else "real"
        return DDABackend(train_corpus_path=train_corpus_path, seed=seed, variant=variant)
    expected = ", ".join(BACKEND_INPUT_NAMES)
    raise ValueError(f"Unsupported backend '{name}'. Expected one of: {expected}.")


__all__ = [
    "BACKEND_NAMES",
    "BACKEND_INPUT_NAMES",
    "BackendName",
    "BackendInputName",
    "AttributionBackend",
    "ToyMode",
    "canonicalize_backend_name",
    "create_backend",
    "ToyAttributionBackend",
]
