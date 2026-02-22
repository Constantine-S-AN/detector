"""Attribution backend registry."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from ads.attribution.base import AttributionBackend
from ads.attribution.toy_backend import ToyAttributionBackend, ToyMode

BackendName = Literal["toy", "trak", "cea", "dda"]


def create_backend(
    name: BackendName,
    train_corpus_path: str | Path,
    seed: int = 42,
    mode: ToyMode = "auto",
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
    from ads.attribution.dda_backend import DDABackend

    return DDABackend(train_corpus_path=train_corpus_path, seed=seed)


__all__ = [
    "BackendName",
    "AttributionBackend",
    "ToyMode",
    "create_backend",
    "ToyAttributionBackend",
]
