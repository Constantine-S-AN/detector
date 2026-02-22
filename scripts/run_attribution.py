#!/usr/bin/env python3
"""Batch attribution runner for ADS datasets."""

from __future__ import annotations

import argparse
from pathlib import Path

from ads.attribution import create_backend
from ads.io import read_jsonl, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-path", type=Path, default=Path("artifacts/data/demo_samples.jsonl")
    )
    parser.add_argument(
        "--train-corpus-path", type=Path, default=Path("artifacts/data/train_corpus.jsonl")
    )
    parser.add_argument("--output-path", type=Path, default=Path("artifacts/scores.jsonl"))
    parser.add_argument("--backend", type=str, default="toy")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.dataset_path)
    backend = create_backend(
        args.backend, train_corpus_path=args.train_corpus_path, seed=args.seed, mode="auto"
    )

    output_rows: list[dict[str, object]] = []
    for row in rows:
        prompt = str(row["prompt"])
        answer = str(row["answer"])
        attribution = backend.compute(prompt=prompt, answer=answer, top_k=args.top_k)
        output_rows.append(
            {
                "sample_id": row["sample_id"],
                "prompt": prompt,
                "answer": answer,
                "label": row["label"],
                "backend": args.backend,
                "top_k": args.top_k,
                "attribution": [item.to_dict() for item in attribution],
            }
        )

    write_jsonl(args.output_path, output_rows)


if __name__ == "__main__":
    main()
