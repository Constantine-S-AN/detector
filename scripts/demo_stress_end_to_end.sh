#!/usr/bin/env bash
set -euo pipefail

if [ -x ".venv/bin/python" ]; then
  PYTHON_BIN=".venv/bin/python"
else
  PYTHON_BIN="python3"
fi

export PYTHONPATH="${PWD}:${PYTHONPATH:-}"

ARTIFACTS_DIR="${ARTIFACTS_DIR:-artifacts_stress}"
DEMO_OUTPUT_DIR="${DEMO_OUTPUT_DIR:-site/public/demo-stress}"
DECISION_THRESHOLD="${DECISION_THRESHOLD:-0.5}"
SCORE_THRESHOLD="${SCORE_THRESHOLD:-0.55}"
MAX_SCORE_FLOOR="${MAX_SCORE_FLOOR:-0.05}"

mkdir -p "$ARTIFACTS_DIR/data" "$ARTIFACTS_DIR/models" "$ARTIFACTS_DIR/plots" "$ARTIFACTS_DIR/report" "$DEMO_OUTPUT_DIR"

"$PYTHON_BIN" scripts/build_stress_dataset.py --output-dir "$ARTIFACTS_DIR/data" --seed 42 --num-samples 40 --train-size 240
"$PYTHON_BIN" scripts/run_attribution.py --dataset-path "$ARTIFACTS_DIR/data/demo_samples_stress.jsonl" --train-corpus-path "$ARTIFACTS_DIR/data/train_corpus.jsonl" --output-path "$ARTIFACTS_DIR/scores.jsonl" --backend toy --toy-mode distributed --seed 42 --top-k 20
"$PYTHON_BIN" scripts/build_features.py --scores-path "$ARTIFACTS_DIR/scores.jsonl" --output-path "$ARTIFACTS_DIR/features.csv" --max-score-floor "$MAX_SCORE_FLOOR"
"$PYTHON_BIN" scripts/train_detector.py --features-path "$ARTIFACTS_DIR/features.csv" --model-path "$ARTIFACTS_DIR/models/logistic.joblib" --split-path "$ARTIFACTS_DIR/data/splits.json" --seed 42 --test-size 0.3 --decision-threshold "$DECISION_THRESHOLD"
"$PYTHON_BIN" scripts/evaluate_detector.py --features-path "$ARTIFACTS_DIR/features.csv" --model-path "$ARTIFACTS_DIR/models/logistic.joblib" --split-path "$ARTIFACTS_DIR/data/splits.json" --metrics-path "$ARTIFACTS_DIR/metrics.json" --plot-dir "$ARTIFACTS_DIR/plots" --predictions-path "$ARTIFACTS_DIR/predictions_all.csv" --test-predictions-path "$ARTIFACTS_DIR/predictions_test.csv" --ablation-path "$ARTIFACTS_DIR/ablation.csv" --decision-threshold "$DECISION_THRESHOLD" --score-threshold "$SCORE_THRESHOLD" --max-score-floor "$MAX_SCORE_FLOOR"
"$PYTHON_BIN" scripts/export_demo_assets.py --scores-path "$ARTIFACTS_DIR/scores.jsonl" --features-path "$ARTIFACTS_DIR/features.csv" --predictions-path "$ARTIFACTS_DIR/predictions_all.csv" --metrics-path "$ARTIFACTS_DIR/metrics.json" --plots-dir "$ARTIFACTS_DIR/plots" --output-dir "$DEMO_OUTPUT_DIR"
"$PYTHON_BIN" scripts/write_run_manifest.py --artifacts-dir "$ARTIFACTS_DIR" --output-path "$ARTIFACTS_DIR/run_manifest.json" --decision-threshold "$DECISION_THRESHOLD" --score-threshold "$SCORE_THRESHOLD" --max-score-floor "$MAX_SCORE_FLOOR"
"$PYTHON_BIN" -m ads.cli build-report --artifacts-dir "$ARTIFACTS_DIR"

echo "ADS stress demo artifacts generated under $ARTIFACTS_DIR and $DEMO_OUTPUT_DIR."
