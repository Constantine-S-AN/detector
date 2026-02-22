"""Experimental DDA backend adapter (best-effort)."""

from __future__ import annotations

from pathlib import Path

from ads.attribution.base import AttributionBackend, AttributionItem


class DDABackend(AttributionBackend):
    """Experimental wrapper for Data Distribution Attribution methods."""

    name = "dda"

    def __init__(self, train_corpus_path: str | Path, seed: int = 42) -> None:
        """Store adapter configuration for experimental DDA methods."""
        self._train_corpus_path = Path(train_corpus_path)
        self._seed = seed

    def compute(self, prompt: str, answer: str, top_k: int) -> list[AttributionItem]:
        """Compute influences using DDA extensions when available."""
        raise NotImplementedError(
            "TODO: DDA backend is experimental; plug in method implementation before use."
        )
