"""Core interfaces for attribution backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class AttributionItem:
    """A scored training sample returned by an attribution backend."""

    train_id: str
    score: float
    rank: int
    text: str = ""
    source: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert this item into a JSON-serializable dictionary."""
        return asdict(self)


class AttributionBackend(ABC):
    """Abstract attribution backend interface."""

    name: str = "base"

    @abstractmethod
    def compute(
        self,
        prompt: str,
        answer: str,
        top_k: int,
        *,
        sample_meta: dict[str, Any] | None = None,
        attribution_mode: str | None = None,
    ) -> list[AttributionItem]:
        """Compute top-k training influences for a prompt/answer pair."""


@dataclass(slots=True)
class AttributionResult:
    """Stable attribution output schema persisted by batch jobs."""

    sample_id: str
    backend: str
    k_requested: int
    k_effective: int
    items: list[AttributionItem]
    backend_meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert this result into a JSON-serializable dictionary."""
        return {
            "sample_id": self.sample_id,
            "backend": self.backend,
            "k_requested": self.k_requested,
            "k_effective": self.k_effective,
            "items": [item.to_dict() for item in self.items],
            "backend_meta": dict(self.backend_meta),
        }
