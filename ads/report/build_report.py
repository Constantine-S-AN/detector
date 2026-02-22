"""Build a machine-readable report index from generated artifacts."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def build_report_index(artifacts_dir: str | Path = "artifacts") -> dict[str, Any]:
    """Create report index JSON with metrics and available plots."""
    root = Path(artifacts_dir)
    metrics_path = root / "metrics.json"
    plots_dir = root / "plots"
    output_path = root / "report" / "index.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    metrics = {}
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    plots = sorted(file.name for file in plots_dir.glob("*.svg"))
    payload: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "metrics": metrics,
        "plots": plots,
        "artifacts_root": str(root),
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


if __name__ == "__main__":
    build_report_index()
