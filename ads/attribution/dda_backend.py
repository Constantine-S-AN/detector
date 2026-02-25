"""Experimental DDA backend adapter (best-effort)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from ads.attribution.base import AttributionBackend, AttributionItem


class DDABackend(AttributionBackend):
    """Experimental wrapper for Data Distribution Attribution methods."""

    name = "dda_tfidf_proxy"

    def __init__(
        self,
        train_corpus_path: str | Path,
        seed: int = 42,
        variant: Literal["tfidf_proxy", "real"] = "tfidf_proxy",
    ) -> None:
        """Store adapter configuration for experimental DDA methods."""
        self._train_corpus_path = Path(train_corpus_path)
        self._seed = seed
        self._variant = variant
        self.name = "dda_tfidf_proxy" if variant == "tfidf_proxy" else "dda_real"

    @property
    def variant(self) -> Literal["tfidf_proxy", "real"]:
        """Return concrete DDA implementation variant."""
        return self._variant

    def compute(
        self,
        prompt: str,
        answer: str,
        top_k: int,
        *,
        sample_meta: dict[str, Any] | None = None,
        attribution_mode: str | None = None,
    ) -> list[AttributionItem]:
        """Compute influences using DDA extensions when available."""
        del prompt, answer, top_k, sample_meta, attribution_mode
        raise NotImplementedError(
            f"DDA backend '{self.name}' is experimental; plug in method implementation before use."
        )
