"""YAML config loader for ADS."""

from __future__ import annotations

from pathlib import Path

import yaml

from ads.config.settings import ADSConfig


def load_config(path: str | Path) -> ADSConfig:
    """Load ADS configuration from a YAML file."""
    config_path = Path(path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return ADSConfig.model_validate(payload or {})
