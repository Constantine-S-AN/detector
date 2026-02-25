#!/usr/bin/env python3
"""Batch attribution runner for ADS datasets."""

from __future__ import annotations

import argparse
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from ads.attribution import BACKEND_NAMES, create_backend
from ads.attribution.base import AttributionItem, AttributionResult
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
    parser.add_argument("--backend", type=str, choices=BACKEND_NAMES, default="toy")
    parser.add_argument(
        "--toy-mode",
        type=str,
        choices=("auto", "peaked", "diffuse", "distributed"),
        default="auto",
    )
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dda-alpha", type=float, default=0.35)
    parser.add_argument("--dda-min-score", type=float, default=0.0)
    parser.add_argument("--dda-cache-dir", type=Path, default=Path(".cache/ads/dda"))
    parser.add_argument("--dda-model-id", type=str, default="dda_tfidf_v1")
    parser.add_argument("--dda-ckpt", type=str, default=None)
    parser.add_argument("--dda-device", type=str, default="cpu")
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
            dda_alpha=args.dda_alpha,
            dda_min_score=args.dda_min_score,
            dda_cache_dir=args.dda_cache_dir,
            dda_model_id=args.dda_model_id,
            dda_ckpt=args.dda_ckpt,
            dda_device=args.dda_device,
        )

    for row in iter_jsonl(args.dataset_path):
        prompt = str(row["prompt"])
        answer = str(row["answer"])
        row_meta: dict[str, object] = {
            "sample_id": row.get("sample_id"),
            "label": row.get("label"),
            "attribution_mode": row.get("attribution_mode"),
        }
        requested_mode = row.get("attribution_mode")
        attribution = backend.compute(
            prompt=prompt,
            answer=answer,
            top_k=args.top_k,
            sample_meta=row_meta,
            attribution_mode=str(requested_mode) if requested_mode is not None else None,
        )
        payload = _result_payload(
            sample_id=str(row["sample_id"]),
            backend_name=args.backend,
            requested_k=args.top_k,
            items=attribution,
            prompt=prompt,
            answer=answer,
            label=str(row["label"]),
            attribution_mode=str(requested_mode) if requested_mode is not None else None,
        )
        yield payload


def _result_payload(
    *,
    sample_id: str,
    backend_name: str,
    requested_k: int,
    items: list[AttributionItem],
    prompt: str,
    answer: str,
    label: str,
    attribution_mode: str | None,
) -> dict[str, Any]:
    result = AttributionResult(
        sample_id=sample_id,
        backend=backend_name,
        k_requested=int(requested_k),
        k_effective=len(items),
        items=items,
        backend_meta={
            "score_semantics": "larger score means more influential",
            "score_direction": "descending",
        },
    )
    payload = result.to_dict()
    # Backward compatibility fields used by downstream scripts/tests.
    payload.update(
        {
            "prompt": prompt,
            "answer": answer,
            "label": label,
            "attribution_mode": attribution_mode,
            "top_k": int(requested_k),
            "attribution": payload["items"],
        }
    )
    return payload


def main() -> None:
    args = parse_args()
    write_jsonl(args.output_path, _iter_output_rows(args))


if __name__ == "__main__":
    main()
