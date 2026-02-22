"""Optional CEA backend adapter (best-effort)."""

from __future__ import annotations

from pathlib import Path

from ads.attribution.base import AttributionBackend, AttributionItem


class CEABackend(AttributionBackend):
    """Adapter shell for CEA attribution backends."""

    name = "cea"

    def __init__(self, train_corpus_path: str | Path, seed: int = 42) -> None:
        """Store adapter configuration for optional CEA execution."""
        self._train_corpus_path = Path(train_corpus_path)
        self._seed = seed

    def compute(self, prompt: str, answer: str, top_k: int) -> list[AttributionItem]:
        """Compute influences with CEA if user wires a local implementation."""
        raise NotImplementedError(
            "TODO: CEA backend is optional; provide implementation and dependencies locally."
        )
