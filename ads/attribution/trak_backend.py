"""Optional TRAK backend adapter (best-effort)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ads.attribution.base import AttributionBackend, AttributionItem


class TRAKBackend(AttributionBackend):
    """Adapter shell for integrating TRAK without blocking demo pipeline."""

    name = "trak"

    def __init__(self, train_corpus_path: str | Path, seed: int = 42) -> None:
        """Create backend and probe optional runtime dependencies."""
        self._train_corpus_path = Path(train_corpus_path)
        self._seed = seed
        self._import_error: Exception | None = None
        try:
            import torch  # noqa: F401
        except Exception as exc:  # pragma: no cover - optional dependency
            self._import_error = exc

    def compute(
        self,
        prompt: str,
        answer: str,
        top_k: int,
        *,
        sample_meta: dict[str, Any] | None = None,
        attribution_mode: str | None = None,
    ) -> list[AttributionItem]:
        """Compute influence scores via TRAK integration (not implemented by default)."""
        del sample_meta, attribution_mode
        if self._import_error is not None:
            raise RuntimeError(
                "TRAK backend dependencies are missing. Install extras with `pip install .[trak]`"
            ) from self._import_error
        raise NotImplementedError(
            "TODO: TRAK integration is project-specific; wire model checkpoints and cache here."
        )
