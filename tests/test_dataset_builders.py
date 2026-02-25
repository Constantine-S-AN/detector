"""Tests for dataset builders writing explicit attribution_mode."""

from __future__ import annotations

from scripts.build_controlled_dataset import build_demo_samples
from scripts.build_stress_dataset import build_stress_samples


def test_controlled_dataset_assigns_mode_from_label() -> None:
    rows = build_demo_samples(num_samples=20, seed=42)
    for row in rows:
        label = str(row["label"])
        mode = str(row["attribution_mode"])
        if label == "faithful":
            assert mode == "peaked"
        elif label == "hallucinated":
            assert mode == "diffuse"
        else:
            raise AssertionError(f"Unexpected label: {label}")


def test_stress_dataset_assigns_distributed_mode() -> None:
    rows = build_stress_samples(num_samples=20, seed=42)
    for row in rows:
        assert row["attribution_mode"] == "distributed"


def test_controlled_attribution_mode_stays_constant_for_each_label() -> None:
    rows = build_demo_samples(num_samples=40, seed=7)

    faithful_modes = {
        str(row["attribution_mode"]) for row in rows if str(row["label"]) == "faithful"
    }
    hallucinated_modes = {
        str(row["attribution_mode"]) for row in rows if str(row["label"]) == "hallucinated"
    }

    assert faithful_modes == {"peaked"}
    assert hallucinated_modes == {"diffuse"}
