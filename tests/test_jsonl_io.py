"""Tests for shared JSONL streaming helpers."""

from __future__ import annotations

from pathlib import Path

from ads.io import iter_jsonl, read_jsonl, write_jsonl


def test_jsonl_write_then_read_roundtrip(tmp_path: Path) -> None:
    rows = [
        {"sample_id": "s-1", "score": 0.9},
        {"sample_id": "s-2", "score": 0.1, "meta": {"mode": "diffuse"}},
    ]
    path = tmp_path / "rows.jsonl"

    write_jsonl(path, rows)

    text = path.read_text(encoding="utf-8")
    assert text.endswith("\n")
    assert text.count("\n") == len(rows)
    assert read_jsonl(path) == rows
    assert list(iter_jsonl(path)) == rows
