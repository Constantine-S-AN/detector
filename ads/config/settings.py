"""Typed settings for ADS pipelines."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ArtifactPaths(BaseModel):
    """Filesystem locations for generated artifacts."""

    root: str = "artifacts"
    data: str = "artifacts/data"
    models: str = "artifacts/models"
    plots: str = "artifacts/plots"
    report: str = "artifacts/report"


class ADSConfig(BaseModel):
    """Top-level configuration for reproducible ADS runs."""

    seed: int = 42
    top_k: int = 20
    max_score_floor: float = 0.05
    artifact_paths: ArtifactPaths = Field(default_factory=ArtifactPaths)
