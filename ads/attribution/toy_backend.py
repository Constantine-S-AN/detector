"""Reproducible synthetic attribution backend for CI and demos."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np

from ads.attribution.base import AttributionBackend, AttributionItem

ToyMode = Literal["auto", "peaked", "diffuse"]
_DIFFUSE_HINTS = (
    "speculative",
    "uncertain",
    "fabricated",
    "hallucinated",
    "guess",
    "cannot verify",
)


class ToyAttributionBackend(AttributionBackend):
    """Synthetic backend that emits peaked or diffuse influence distributions."""

    name = "toy"

    def __init__(
        self,
        train_items: Sequence[dict[str, Any]],
        seed: int = 42,
        mode: ToyMode = "auto",
    ) -> None:
        """Initialize a deterministic synthetic backend."""
        if not train_items:
            raise ValueError("ToyAttributionBackend requires a non-empty train_items corpus.")
        self._train_items = list(train_items)
        self._seed = seed
        self._mode = mode

    @classmethod
    def from_jsonl(
        cls,
        path: str | Path,
        seed: int = 42,
        mode: ToyMode = "auto",
    ) -> ToyAttributionBackend:
        """Load training corpus records from JSONL."""
        corpus_path = Path(path)
        rows = [
            json.loads(line)
            for line in corpus_path.read_text(encoding="utf-8").splitlines()
            if line
        ]
        return cls(train_items=rows, seed=seed, mode=mode)

    def compute(self, prompt: str, answer: str, top_k: int) -> list[AttributionItem]:
        """Return deterministic top-k synthetic influences for the given sample."""
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        sample_size = min(top_k, len(self._train_items))
        rng = np.random.default_rng(self._seed_from_text(prompt=prompt, answer=answer))
        backend_mode = self._resolve_mode(answer)

        indices = rng.choice(len(self._train_items), size=sample_size, replace=False)
        raw_scores = self._generate_scores(rng=rng, size=sample_size, mode=backend_mode)

        ranked_pairs = sorted(
            zip(indices, raw_scores, strict=False),
            key=lambda pair: pair[1],
            reverse=True,
        )
        results: list[AttributionItem] = []
        for rank, (corpus_index, score) in enumerate(ranked_pairs, start=1):
            train_item = self._train_items[int(corpus_index)]
            results.append(
                AttributionItem(
                    train_id=str(train_item["train_id"]),
                    score=float(score),
                    text=str(train_item["text"]),
                    meta={"rank": rank, "mode": backend_mode},
                )
            )
        return results

    def _resolve_mode(self, answer: str) -> Literal["peaked", "diffuse"]:
        if self._mode == "peaked":
            return "peaked"
        if self._mode == "diffuse":
            return "diffuse"
        answer_lower = answer.lower()
        if any(token in answer_lower for token in _DIFFUSE_HINTS):
            return "diffuse"
        return "peaked"

    def _generate_scores(
        self,
        rng: np.random.Generator,
        size: int,
        mode: Literal["peaked", "diffuse"],
    ) -> np.ndarray:
        if mode == "diffuse":
            return rng.uniform(0.8, 1.2, size=size)

        peaked_scores = rng.uniform(0.02, 0.2, size=size)
        dominant_index = int(rng.integers(0, size))
        peaked_scores[dominant_index] += float(rng.uniform(2.8, 4.5))

        if size > 2:
            secondary_count = min(2, size - 1)
            secondary_pool = np.array([idx for idx in range(size) if idx != dominant_index])
            secondary_indices = rng.choice(secondary_pool, size=secondary_count, replace=False)
            peaked_scores[secondary_indices] += rng.uniform(0.5, 1.1, size=secondary_count)
        return peaked_scores

    def _seed_from_text(self, prompt: str, answer: str) -> int:
        payload = f"{self._seed}::{prompt}::{answer}".encode()
        digest = hashlib.sha256(payload).hexdigest()
        return int(digest[:16], 16)
