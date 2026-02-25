#!/usr/bin/env python3
"""Generate a deterministic controlled dataset for ADS demos."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ads.io import write_jsonl

FAITHFUL_TEMPLATES = [
    "According to the provided sources, {fact}.",
    "The reference notes that {fact}; this is directly grounded in the training corpus.",
    "Grounded summary: {fact}.",
]

HALLUCINATED_TEMPLATES = [
    "This is speculative and uncertain: {claim}.",
    "I cannot verify this and it may be fabricated: {claim}.",
    "Hallucinated guess: {claim}; treat it as uncertain.",
]

FACT_BANK = [
    ("the Pacific Ocean is the largest ocean on Earth", "the Pacific Ocean is 2 km wide"),
    ("Mount Everest is Earth's highest mountain above sea level", "Mount Everest is underwater"),
    ("Tokyo is the capital city of Japan", "Tokyo is the capital of Australia"),
    ("water boils near 100C at sea level", "water boils at 40C at sea level"),
    ("Python was created by Guido van Rossum", "Python was created in 1890"),
    ("the heart has four chambers", "the heart has nine chambers"),
    ("the Great Wall is in China", "the Great Wall is in Brazil"),
    ("Mars is known as the red planet", "Mars is a moon of Jupiter"),
    ("light travels faster than sound", "sound travels faster than light"),
    ("the Nile is one of the longest rivers", "the Nile is a desert plateau"),
]

PROMPTS = [
    "Answer the user question with factual grounding: {topic}",
    "Provide a concise explanation: {topic}",
    "What is the correct statement about: {topic}?",
    "Give a grounded answer only: {topic}",
]


def build_train_corpus(train_size: int, seed: int) -> list[dict[str, object]]:
    """Create synthetic train corpus entries referenced by attribution results."""
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for idx in range(train_size):
        topic_idx = int(rng.integers(0, len(FACT_BANK)))
        fact, wrong = FACT_BANK[topic_idx]
        rows.append(
            {
                "train_id": f"train-{idx:04d}",
                "text": f"Reference {idx}: {fact}. Counter-example: {wrong}.",
                "meta": {"topic_index": topic_idx, "source": "synthetic"},
            }
        )
    return rows


def build_demo_samples(num_samples: int, seed: int) -> list[dict[str, object]]:
    """Create balanced faithful/hallucinated evaluation samples."""
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for idx in range(num_samples):
        fact, claim = FACT_BANK[idx % len(FACT_BANK)]
        topic = fact.split(" is ")[0]
        prompt_template = PROMPTS[int(rng.integers(0, len(PROMPTS)))]
        prompt = prompt_template.format(topic=topic)

        is_faithful = idx % 2 == 0
        if is_faithful:
            answer_template = FAITHFUL_TEMPLATES[int(rng.integers(0, len(FAITHFUL_TEMPLATES)))]
            answer = answer_template.format(fact=fact)
            label = "faithful"
            attribution_mode = "peaked"
        else:
            answer_template = HALLUCINATED_TEMPLATES[
                int(rng.integers(0, len(HALLUCINATED_TEMPLATES)))
            ]
            answer = answer_template.format(claim=claim)
            label = "hallucinated"
            attribution_mode = "diffuse"

        rows.append(
            {
                "sample_id": f"sample-{idx:03d}",
                "prompt": prompt,
                "answer": answer,
                "label": label,
                "attribution_mode": attribution_mode,
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/data"))
    parser.add_argument("--num-samples", type=int, default=40)
    parser.add_argument("--train-size", type=int, default=240)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_rows = build_train_corpus(train_size=args.train_size, seed=args.seed)
    sample_rows = build_demo_samples(num_samples=args.num_samples, seed=args.seed)

    write_jsonl(args.output_dir / "train_corpus.jsonl", train_rows)
    write_jsonl(args.output_dir / "demo_samples.jsonl", sample_rows)

    manifest = {
        "seed": args.seed,
        "num_samples": args.num_samples,
        "train_size": args.train_size,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
