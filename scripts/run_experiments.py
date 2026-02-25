#!/usr/bin/env python3
"""Run multi-seed attribution-density experiments and aggregate results."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class ExperimentRow:
    backend: str
    k: int
    seed: int
    detector: str
    status: str
    roc_auc: float | None
    pr_auc: float | None
    ece: float | None
    brier: float | None
    run_dir: str
    metrics_path: str | None
    note: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument("--backends", type=str, default="dda_tfidf_proxy,toy,dda_real")
    parser.add_argument("--ks", type=str, default="5,10,20")
    parser.add_argument("--seeds", type=str, default="0,1,2,3,4")
    parser.add_argument("--detectors", type=str, default="logistic")
    parser.add_argument("--num-samples", type=int, default=40)
    parser.add_argument("--train-size", type=int, default=240)
    parser.add_argument("--attribution-top-k", type=int, default=20)
    parser.add_argument("--decision-threshold", type=float, default=0.5)
    parser.add_argument("--score-threshold", type=float, default=0.55)
    parser.add_argument("--max-score-floor", type=float, default=0.05)
    parser.add_argument("--dda-alpha", type=float, default=0.35)
    parser.add_argument("--dda-min-score", type=float, default=0.0)
    parser.add_argument(
        "--dda-model-id", type=str, default="sentence-transformers/all-MiniLM-L6-v2"
    )
    parser.add_argument("--dda-device", type=str, default="cpu")
    parser.add_argument("--dda-ckpt", type=str, default=None)
    parser.add_argument("--dda-cache-dir", type=Path, default=Path(".cache/ads/dda"))
    return parser.parse_args()


def _parse_csv_ints(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def _parse_csv_strs(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _run(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=cwd, text=True, capture_output=True, check=True)


def _is_optional_backend_failure(backend: str, error_text: str) -> bool:
    if backend not in {"dda_real", "trak", "cea"}:
        return False
    lowered = error_text.lower()
    optional_markers = (
        "dependencies are missing",
        "modulenotfounderror",
        "pip install -e .[dda]",
        "sentence_transformers",
        "notimplementederror",
    )
    return any(marker in lowered for marker in optional_markers)


def _metrics_values(path: Path) -> tuple[float | None, float | None, float | None, float | None]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return (
        payload.get("roc_auc"),
        payload.get("pr_auc"),
        payload.get("ece"),
        payload.get("brier"),
    )


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mean = _mean(values)
    return (sum((value - mean) ** 2 for value in values) / len(values)) ** 0.5


def _write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        headers = [
            "backend",
            "k",
            "detector",
            "n_success",
            "n_skipped",
            "roc_auc_mean",
            "roc_auc_std",
            "pr_auc_mean",
            "pr_auc_std",
            "ece_mean",
            "ece_std",
            "brier_mean",
            "brier_std",
        ]
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=headers)
            writer.writeheader()
        return

    headers = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def _write_summary_md(path: Path, run_id: str, aggregated_rows: list[dict[str, Any]]) -> None:
    lines = [f"# Experiment Summary ({run_id})", ""]
    if not aggregated_rows:
        lines.append("No successful runs.")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    lines.append("## Aggregated metrics")
    lines.append("")
    lines.append(
        "| backend | K | detector | n_success | n_skipped | ROC-AUC mean±std | PR-AUC mean±std | ECE mean±std | Brier mean±std |"
    )
    lines.append("|---|---:|---|---:|---:|---|---|---|---|")
    for row in aggregated_rows:
        lines.append(
            "| {backend} | {k} | {detector} | {n_success} | {n_skipped} | "
            "{roc_auc_mean:.4f}±{roc_auc_std:.4f} | {pr_auc_mean:.4f}±{pr_auc_std:.4f} | "
            "{ece_mean:.4f}±{ece_std:.4f} | {brier_mean:.4f}±{brier_std:.4f} |".format(**row)
        )

    best = max(aggregated_rows, key=lambda item: item["roc_auc_mean"])
    worst_ece = max(aggregated_rows, key=lambda item: item["ece_mean"])
    lines.append("")
    lines.append("## Key takeaways")
    lines.append("")
    lines.append(
        f"- Best ROC-AUC: `{best['backend']}` K={best['k']} detector={best['detector']} "
        f"({best['roc_auc_mean']:.4f}±{best['roc_auc_std']:.4f})."
    )
    lines.append(
        f"- Worst calibration (ECE): `{worst_ece['backend']}` K={worst_ece['k']} detector={worst_ece['detector']} "
        f"({worst_ece['ece_mean']:.4f}±{worst_ece['ece_std']:.4f})."
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("exp-%Y%m%d-%H%M%S")

    backends = _parse_csv_strs(args.backends)
    ks = _parse_csv_ints(args.ks)
    seeds = _parse_csv_ints(args.seeds)
    detectors = _parse_csv_strs(args.detectors)

    rows: list[ExperimentRow] = []

    repo_root = Path.cwd()
    run_root = args.runs_root / run_id

    for backend in backends:
        for k_value in ks:
            for seed in seeds:
                for detector in detectors:
                    if detector != "logistic":
                        rows.append(
                            ExperimentRow(
                                backend=backend,
                                k=k_value,
                                seed=seed,
                                detector=detector,
                                status="SKIPPED",
                                roc_auc=None,
                                pr_auc=None,
                                ece=None,
                                brier=None,
                                run_dir=str(run_root / backend / f"k_{k_value}" / f"seed_{seed}"),
                                metrics_path=None,
                                note="detector_not_supported_yet",
                            )
                        )
                        continue

                    work_dir = run_root / backend / f"k_{k_value}" / f"seed_{seed}" / detector
                    data_dir = work_dir / "data"
                    model_dir = work_dir / "models"
                    plots_dir = work_dir / "plots"
                    work_dir.mkdir(parents=True, exist_ok=True)
                    data_dir.mkdir(parents=True, exist_ok=True)
                    model_dir.mkdir(parents=True, exist_ok=True)
                    plots_dir.mkdir(parents=True, exist_ok=True)

                    dataset_path = data_dir / "demo_samples.jsonl"
                    train_path = data_dir / "train_corpus.jsonl"
                    scores_path = work_dir / "scores.jsonl"
                    features_path = work_dir / "features.csv"
                    model_path = model_dir / "logistic.joblib"
                    split_path = data_dir / "splits.json"
                    metrics_path = work_dir / "metrics.json"
                    pred_path = work_dir / "predictions_all.csv"
                    test_pred_path = work_dir / "predictions_test.csv"
                    ablation_path = work_dir / "ablation.csv"

                    try:
                        _run(
                            [
                                sys.executable,
                                "scripts/build_controlled_dataset.py",
                                "--output-dir",
                                str(data_dir),
                                "--num-samples",
                                str(args.num_samples),
                                "--train-size",
                                str(args.train_size),
                                "--seed",
                                str(seed),
                            ],
                            cwd=repo_root,
                        )
                        _run(
                            [
                                sys.executable,
                                "scripts/run_attribution.py",
                                "--dataset-path",
                                str(dataset_path),
                                "--train-corpus-path",
                                str(train_path),
                                "--output-path",
                                str(scores_path),
                                "--backend",
                                backend,
                                "--top-k",
                                str(args.attribution_top_k),
                                "--seed",
                                str(seed),
                                "--dda-alpha",
                                str(args.dda_alpha),
                                "--dda-min-score",
                                str(args.dda_min_score),
                                "--dda-cache-dir",
                                str(args.dda_cache_dir),
                                "--dda-model-id",
                                str(args.dda_model_id),
                                "--dda-device",
                                str(args.dda_device),
                            ]
                            + (["--dda-ckpt", str(args.dda_ckpt)] if args.dda_ckpt else []),
                            cwd=repo_root,
                        )
                        _run(
                            [
                                sys.executable,
                                "scripts/build_features.py",
                                "--scores-path",
                                str(scores_path),
                                "--output-path",
                                str(features_path),
                                "--h-k",
                                str(k_value),
                                "--max-score-floor",
                                str(args.max_score_floor),
                            ],
                            cwd=repo_root,
                        )
                        _run(
                            [
                                sys.executable,
                                "scripts/train_detector.py",
                                "--features-path",
                                str(features_path),
                                "--model-path",
                                str(model_path),
                                "--split-path",
                                str(split_path),
                                "--seed",
                                str(seed),
                                "--decision-threshold",
                                str(args.decision_threshold),
                                "--score-threshold",
                                str(args.score_threshold),
                                "--max-score-floor",
                                str(args.max_score_floor),
                            ],
                            cwd=repo_root,
                        )
                        _run(
                            [
                                sys.executable,
                                "scripts/evaluate_detector.py",
                                "--features-path",
                                str(features_path),
                                "--model-path",
                                str(model_path),
                                "--split-path",
                                str(split_path),
                                "--metrics-path",
                                str(metrics_path),
                                "--plot-dir",
                                str(plots_dir),
                                "--predictions-path",
                                str(pred_path),
                                "--test-predictions-path",
                                str(test_pred_path),
                                "--ablation-path",
                                str(ablation_path),
                                "--decision-threshold",
                                str(args.decision_threshold),
                                "--score-threshold",
                                str(args.score_threshold),
                                "--max-score-floor",
                                str(args.max_score_floor),
                            ],
                            cwd=repo_root,
                        )

                        roc_auc, pr_auc, ece, brier = _metrics_values(metrics_path)
                        rows.append(
                            ExperimentRow(
                                backend=backend,
                                k=k_value,
                                seed=seed,
                                detector=detector,
                                status="OK",
                                roc_auc=roc_auc,
                                pr_auc=pr_auc,
                                ece=ece,
                                brier=brier,
                                run_dir=str(work_dir),
                                metrics_path=str(metrics_path),
                                note=None,
                            )
                        )
                    except subprocess.CalledProcessError as exc:
                        error_text = (exc.stdout or "") + "\n" + (exc.stderr or "")
                        status = (
                            "SKIPPED"
                            if _is_optional_backend_failure(backend, error_text)
                            else "FAILED"
                        )
                        rows.append(
                            ExperimentRow(
                                backend=backend,
                                k=k_value,
                                seed=seed,
                                detector=detector,
                                status=status,
                                roc_auc=None,
                                pr_auc=None,
                                ece=None,
                                brier=None,
                                run_dir=str(work_dir),
                                metrics_path=None,
                                note=(error_text[-400:] if error_text else "subprocess_failed"),
                            )
                        )
                        if status == "FAILED":
                            # keep running remaining configs to maximize coverage
                            continue

    run_root.mkdir(parents=True, exist_ok=True)
    details_path = run_root / "results_detailed.csv"
    with details_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "backend",
                "k",
                "seed",
                "detector",
                "status",
                "roc_auc",
                "pr_auc",
                "ece",
                "brier",
                "run_dir",
                "metrics_path",
                "note",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "backend": row.backend,
                    "k": row.k,
                    "seed": row.seed,
                    "detector": row.detector,
                    "status": row.status,
                    "roc_auc": row.roc_auc,
                    "pr_auc": row.pr_auc,
                    "ece": row.ece,
                    "brier": row.brier,
                    "run_dir": row.run_dir,
                    "metrics_path": row.metrics_path,
                    "note": row.note,
                }
            )

    grouped: dict[tuple[str, int, str], list[ExperimentRow]] = {}
    for row in rows:
        grouped.setdefault((row.backend, row.k, row.detector), []).append(row)

    aggregated_rows: list[dict[str, Any]] = []
    for (backend, k_value, detector), group_rows in sorted(grouped.items()):
        ok_rows = [row for row in group_rows if row.status == "OK"]
        skipped_rows = [row for row in group_rows if row.status == "SKIPPED"]
        if ok_rows:
            roc_values = [float(row.roc_auc) for row in ok_rows if row.roc_auc is not None]
            pr_values = [float(row.pr_auc) for row in ok_rows if row.pr_auc is not None]
            ece_values = [float(row.ece) for row in ok_rows if row.ece is not None]
            brier_values = [float(row.brier) for row in ok_rows if row.brier is not None]
            aggregated_rows.append(
                {
                    "backend": backend,
                    "k": k_value,
                    "detector": detector,
                    "n_success": len(ok_rows),
                    "n_skipped": len(skipped_rows),
                    "roc_auc_mean": _mean(roc_values) if roc_values else 0.0,
                    "roc_auc_std": _std(roc_values) if roc_values else 0.0,
                    "pr_auc_mean": _mean(pr_values) if pr_values else 0.0,
                    "pr_auc_std": _std(pr_values) if pr_values else 0.0,
                    "ece_mean": _mean(ece_values) if ece_values else 0.0,
                    "ece_std": _std(ece_values) if ece_values else 0.0,
                    "brier_mean": _mean(brier_values) if brier_values else 0.0,
                    "brier_std": _std(brier_values) if brier_values else 0.0,
                }
            )
        else:
            aggregated_rows.append(
                {
                    "backend": backend,
                    "k": k_value,
                    "detector": detector,
                    "n_success": 0,
                    "n_skipped": len(skipped_rows),
                    "roc_auc_mean": 0.0,
                    "roc_auc_std": 0.0,
                    "pr_auc_mean": 0.0,
                    "pr_auc_std": 0.0,
                    "ece_mean": 0.0,
                    "ece_std": 0.0,
                    "brier_mean": 0.0,
                    "brier_std": 0.0,
                }
            )

    summary_csv = run_root / "summary.csv"
    _write_summary_csv(summary_csv, aggregated_rows)
    _write_summary_md(run_root / "summary.md", run_id, aggregated_rows)


if __name__ == "__main__":
    main()
