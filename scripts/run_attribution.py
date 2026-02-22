#!/usr/bin/env python3
"""Batch attribution runner for ADS datasets."""

from __future__ import annotations

import argparse
from collections.abc import Iterator
from pathlib import Path

from ads.attribution import create_backend
from ads.io import iter_jsonl, write_jsonl


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
    parser.add_argument(
        "--toy-mode",
        type=str,
        choices=("auto", "peaked", "diffuse", "distributed"),
        default="auto",
    )
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _iter_output_rows(args: argparse.Namespace) -> Iterator[dict[str, object]]:
    if args.backend == "toy":
        backend = create_backend(
            args.backend,
            train_corpus_path=args.train_corpus_path,
            seed=args.seed,
            mode=args.toy_mode,
        )
    else:
        backend = create_backend(
            args.backend,
            train_corpus_path=args.train_corpus_path,
            seed=args.seed,
        )

    for row in iter_jsonl(args.dataset_path):
        prompt = str(row["prompt"])
        answer = str(row["answer"])
        attribution = backend.compute(prompt=prompt, answer=answer, top_k=args.top_k)
        yield {
            "sample_id": row["sample_id"],
            "prompt": prompt,
            "answer": answer,
            "label": row["label"],
            "backend": args.backend,
            "top_k": args.top_k,
            "attribution": [item.to_dict() for item in attribution],
        }


def main() -> None:
    args = parse_args()
    write_jsonl(args.output_path, _iter_output_rows(args))


if __name__ == "__main__":
    main()
