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
    text: str
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert this item into a JSON-serializable dictionary."""
        return asdict(self)


class AttributionBackend(ABC):
    """Abstract attribution backend interface."""

    name: str = "base"

    @abstractmethod
    def compute(self, prompt: str, answer: str, top_k: int) -> list[AttributionItem]:
        """Compute top-k training influences for a prompt/answer pair."""
