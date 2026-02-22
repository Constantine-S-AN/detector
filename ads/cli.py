"""CLI entrypoint for ADS workflows."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Literal

import typer

from ads.report.build_report import build_report_index
from ads.service import scan_sample

app = typer.Typer(help="Attribution Density Scanner CLI")


@app.command()
def scan(
    prompt: Annotated[str, typer.Option(help="User prompt")],
    answer: Annotated[str, typer.Option(help="Model answer")],
    top_k: Annotated[int, typer.Option(help="Number of top influential samples")] = 20,
    backend: Annotated[
        Literal["toy", "trak", "cea", "dda"],
        typer.Option(help="Attribution backend"),
    ] = "toy",
    train_corpus_path: Annotated[
        Path,
        typer.Option(help="Path to training corpus JSONL"),
    ] = Path("artifacts/data/train_corpus.jsonl"),
    method: Annotated[
        Literal["logistic", "threshold"],
        typer.Option(help="Detector method"),
    ] = "logistic",
    model_path: Annotated[
        Path,
        typer.Option(help="Path to saved logistic detector"),
    ] = Path("artifacts/models/logistic.joblib"),
    seed: Annotated[int, typer.Option(help="Random seed")] = 42,
) -> None:
    """Run a single scan and print JSON output."""
    result = scan_sample(
        prompt=prompt,
        answer=answer,
        backend_name=backend,
        top_k=top_k,
        seed=seed,
        train_corpus_path=train_corpus_path,
        method=method,
        model_path=model_path,
    )
    typer.echo(json.dumps(result, ensure_ascii=False, indent=2))


@app.command("build-report")
def build_report(
    artifacts_dir: Annotated[Path, typer.Option(help="Artifacts directory")] = Path("artifacts"),
) -> None:
    """Build report index from artifacts."""
    payload = build_report_index(artifacts_dir=artifacts_dir)
    typer.echo(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    app()
