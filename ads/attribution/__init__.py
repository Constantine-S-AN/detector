"""Attribution backend registry."""

from __future__ import annotations

from pathlib import Path
import warnings
from typing import Literal

from ads.attribution.base import AttributionBackend
from ads.attribution.toy_backend import ToyAttributionBackend, ToyMode

BackendName = Literal["toy", "trak", "cea", "dda", "dda_tfidf_proxy", "dda_real"]
BACKEND_NAMES: tuple[BackendName, ...] = (
    "toy",
    "trak",
    "cea",
    "dda",
    "dda_tfidf_proxy",
    "dda_real",
)


def create_backend(
    name: str,
    train_corpus_path: str | Path,
    seed: int = 42,
    mode: ToyMode = "auto",
    dda_alpha: float = 0.35,
    dda_min_score: float = 0.0,
    dda_cache_dir: str | Path = ".cache/ads/dda",
    dda_model_id: str = "dda_tfidf_v1",
    dda_ckpt: str | Path | None = None,
    dda_device: str = "cpu",
) -> AttributionBackend:
    """Build a backend instance by name."""
    if name == "toy":
        return ToyAttributionBackend.from_jsonl(train_corpus_path, seed=seed, mode=mode)
    if name == "trak":
        from ads.attribution.trak_backend import TRAKBackend

        return TRAKBackend(train_corpus_path=train_corpus_path, seed=seed)
    if name == "cea":
        from ads.attribution.cea_backend import CEABackend

        return CEABackend(train_corpus_path=train_corpus_path, seed=seed)
    if name == "dda":
        warnings.warn(
            "`backend=dda` is deprecated and maps to `dda_tfidf_proxy`. "
            "Use `dda_tfidf_proxy` or `dda_real`.",
            DeprecationWarning,
            stacklevel=2,
        )
        from ads.attribution.dda_backend import DDATfidfProxyBackend

        return DDATfidfProxyBackend(
            train_corpus_path=train_corpus_path,
            seed=seed,
            alpha=dda_alpha,
            min_score=dda_min_score,
            cache_dir=dda_cache_dir,
            model_id=dda_model_id,
        )
    if name == "dda_tfidf_proxy":
        from ads.attribution.dda_backend import DDATfidfProxyBackend

        return DDATfidfProxyBackend(
            train_corpus_path=train_corpus_path,
            seed=seed,
            alpha=dda_alpha,
            min_score=dda_min_score,
            cache_dir=dda_cache_dir,
            model_id=dda_model_id,
        )
    if name == "dda_real":
        from ads.attribution.dda_backend import DDARealBackend

        return DDARealBackend(
            train_corpus_path=train_corpus_path,
            seed=seed,
            alpha=dda_alpha,
            min_score=dda_min_score,
            cache_dir=dda_cache_dir,
            model_id=dda_model_id,
            ckpt=dda_ckpt,
            device=dda_device,
        )
    expected = ", ".join(BACKEND_NAMES)
    raise ValueError(f"Unsupported backend '{name}'. Expected one of: {expected}.")


__all__ = [
    "BACKEND_NAMES",
    "BackendName",
    "AttributionBackend",
    "ToyMode",
    "create_backend",
    "ToyAttributionBackend",
]
