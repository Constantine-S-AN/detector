#!/usr/bin/env bash
set -euo pipefail

if [ -x ".venv/bin/python" ]; then
  PYTHON_BIN=".venv/bin/python"
else
  PYTHON_BIN="python3"
fi

export PYTHONPATH="${PWD}:${PYTHONPATH:-}"

mkdir -p artifacts/data artifacts/models artifacts/plots artifacts/report site/public/demo

DECISION_THRESHOLD="${DECISION_THRESHOLD:-0.5}"
SCORE_THRESHOLD="${SCORE_THRESHOLD:-0.55}"
MAX_SCORE_FLOOR="${MAX_SCORE_FLOOR:-0.05}"

"$PYTHON_BIN" scripts/build_controlled_dataset.py --seed 42 --num-samples 40 --train-size 240
"$PYTHON_BIN" scripts/run_attribution.py --backend toy --seed 42 --top-k 20
"$PYTHON_BIN" scripts/build_features.py --max-score-floor "$MAX_SCORE_FLOOR"
"$PYTHON_BIN" scripts/train_detector.py --seed 42 --test-size 0.3 --decision-threshold "$DECISION_THRESHOLD" --score-threshold "$SCORE_THRESHOLD" --max-score-floor "$MAX_SCORE_FLOOR"
"$PYTHON_BIN" scripts/evaluate_detector.py --decision-threshold "$DECISION_THRESHOLD" --score-threshold "$SCORE_THRESHOLD" --max-score-floor "$MAX_SCORE_FLOOR"
"$PYTHON_BIN" scripts/export_demo_assets.py
"$PYTHON_BIN" scripts/write_run_manifest.py --decision-threshold "$DECISION_THRESHOLD" --score-threshold "$SCORE_THRESHOLD" --max-score-floor "$MAX_SCORE_FLOOR"
"$PYTHON_BIN" -m ads.report.build_report

echo "ADS demo artifacts generated under artifacts/ and site/public/demo/."
