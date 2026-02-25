# Attribution Density Scanner (ADS)

[![CI](https://github.com/Constantine-S-AN/detector/actions/workflows/ci.yml/badge.svg)](https://github.com/Constantine-S-AN/detector/actions/workflows/ci.yml)
[![Deploy Pages](https://github.com/Constantine-S-AN/detector/actions/workflows/pages.yml/badge.svg)](https://github.com/Constantine-S-AN/detector/actions/workflows/pages.yml)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

ADS is a groundedness detector focused on explainability.
It uses attribution density over training data to distinguish `faithful` from `hallucinated` outputs.

## Why ADS

- Replaces single-score confidence with influence-distribution geometry to reduce black-box decisions.
- Supports both FULL mode (local API) and STATIC mode (pure static Pages assets).
- Fully reproducible end to end: fixed random seeds, versioned artifacts, and exported visuals/reports.

## 1-Minute Quickstart

```bash
make setup
make demo
make build-site
```

Output directories:

- `artifacts/`: data, models, metrics, plots, reports
- `site/public/demo/`: static demo assets for the frontend
- `site/out/`: Next.js static export output
- `artifacts/run_manifest.json`: run metadata manifest

## Architecture

![ADS Architecture](site/public/ads-architecture.svg)

Pipeline:

1. Attribution backend (`toy` / `trak` / `cea` / `dda`)
2. Density features (`entropy` / `top-share` / `gini` / `effective_k`)
3. Detector (`threshold` + `logistic`)
4. Evaluation & plots (`ROC` / `PR` / `Calibration` / `abstain` / `hist`)
5. Static export for portfolio and GitHub Pages

## Density Features Definitions

- **H@K (top-K influence entropy)**
  - Sort influence scores in descending order and truncate to top-K.
  - Default `weight_mode="shifted"`: `s'_i = (s_i - min(topK)) + eps`, then normalize `p_i = s'_i / Σ s'_j`.
  - Compute `H@K = -Σ_i p_i log p_i` (natural log).
  - The primary default metric is normalized: `H@K_norm = H@K / log(k_effective)` with `k_effective=min(K,n)`.

- **Peakiness ratios (two variants)**
  - `peakiness_ratio_score = top1_score / sum_top5_score` (default in main results; also keeps backward-compatible alias `peakiness_ratio`).
  - `peakiness_ratio_prob = p1 / sum_top5_p` (where `p` is generated from `softmax(topK scores)` for comparison).

- **Defaults**
  - Default `K=20` (configurable via `scripts/build_features.py --h-k`, including comma-separated multi-K values).
  - Default `weight_mode="shifted"`, `eps=1e-12`.

## Proposal-Focused Demo Visuals

![Proposal Overview](docs/images/proposal-overview.png)

The figure above is designed for a proposal landing page, combining the research question, method pipeline, reproducibility signals, and stress-test degradation evidence in one visual.

1. Controlled setting (evidence that the method works on controlled data)

![Proposal Controlled Evidence](docs/images/proposal-controlled-evidence.png)

2. Mechanism evidence (single-sample explainability: attribution peak + top-influential evidence chain)

![Proposal Explainability](docs/images/proposal-scan-explainability.png)

3. Boundary condition (core motivation of the proposal: performance degrades under distributed-truth)

![Baseline vs Stress](docs/images/proposal-baseline-vs-stress.png)

Reproduce the figures above (including overview):

```bash
make demo
make demo-stress
python scripts/generate_proposal_figure.py
```

## Benchmark Snapshot (Toy Controlled Set)

- Dataset: `n=40` (`20 faithful / 20 hallucinated`)
- ROC-AUC: `1.0000`
- PR-AUC: `1.0000`
- ECE: `0.0159`
- Brier: `0.000319`
- Coverage: `1.0000`
- Answered Accuracy: `1.0000`

Metric source: `artifacts/metrics.json` (refreshed after each `make demo` run).

## Modes

### STATIC (GitHub Pages ready)

- Frontend reads `site/public/demo/index.json` and `site/public/demo/examples/*.json`.
- No backend dependency; can be deployed directly to GitHub Pages.

### FULL (Local API)

1. Start the API
   ```bash
   make serve-api
   ```
2. Configure frontend API base URL
   ```bash
   NEXT_PUBLIC_API_BASE=http://127.0.0.1:8000
   ```
3. Open `/scan` for real-time scanning

## Common Commands

```bash
make setup        # install python/node dependencies
make format       # black/isort + prettier
make lint         # ruff/black/isort/mypy + next lint
make test         # pytest
make demo         # end-to-end pipeline + static demo assets
make demo-stress  # distributed-truth stress pipeline (toy distributed mode)
make export-demo  # export only frontend demo assets
make build-site   # next static build
make serve-api    # run FastAPI for FULL mode
```

## Development & Testing

- Dependency management uses `pyproject.toml` (PEP 621) + optional extras.
- Install development/test dependencies:

```bash
pip install -e .[dev]
```

- Run the full test suite:

```bash
PYTHONPATH=. pytest -q
```

Note: API tests depend on `httpx` (already included in `[project.optional-dependencies].dev`); if unavailable, tests in `tests/test_api.py` are skipped automatically.

## End-to-End Script

```bash
bash scripts/demo_end_to_end.sh
```

Default execution order:

1. `build_controlled_dataset.py`
2. `run_attribution.py`
3. `build_features.py`
4. `train_detector.py`
5. `evaluate_detector.py`
6. `export_demo_assets.py`
7. `write_run_manifest.py`
8. `ads.report.build_report`

## API Example

```bash
curl -X POST http://127.0.0.1:8000/scan \
  -H 'Content-Type: application/json' \
  -d '{
    "prompt": "Provide one grounded fact about Tokyo",
    "answer": "According to the provided sources, Tokyo is the capital city of Japan.",
    "method": "logistic",
    "backend": "toy",
    "allow_fallback": false,
    "top_k": 20,
    "decision_threshold": 0.50,
    "score_threshold": 0.55,
    "max_score_floor": 0.05
  }'
```

`/scan` is strict by default (`allow_fallback=false`): when `method=logistic` and the model is missing, it returns `400` (`code=MODEL_MISSING`) rather than silently falling back. For best-effort behavior, explicitly set `allow_fallback=true`.

## Optional Backends

- `ads/attribution/trak_backend.py`
- `ads/attribution/cea_backend.py`
- `ads/attribution/dda_backend.py` (experimental)

These plugins are integrated in best-effort mode and do not block `make demo`.

> Note: the `toy` backend is only for sanity checks and CI demos, and should not be used for research conclusions.
> The toy distribution pattern is driven by `attribution_mode` (dataset field/request parameter), not by answer wording.

## Reproducibility Notes

- Fixed random seed (default `42`)
- Persisted intermediate outputs: `scores.jsonl`, `features.csv`, `predictions_*.csv`
- Persisted plots/assets: PNG/SVG + frontend JSON
- `run_manifest.json` records key configs, metric snapshots, and command chain

## CI/CD

- `.github/workflows/ci.yml`: lint + test on PRs and `main`
- `.github/workflows/pages.yml`: build and deploy Pages from `main`

## Limitations & Future Work

- Under distributed-truth settings, correct answers may still produce diffuse attribution.
- Real LLM attribution can be expensive; caching and approximate retrieval should be optimized next.
- `TRAK/CEA/DDA` are currently interface-level integrations; benchmark and empirical reports are still needed.
- Stress demo (toy distributed mode) reproduces this boundary condition:
  ```bash
  make demo-stress
  ```
  Outputs are generated in `artifacts_stress/` and `site/public/demo-stress/`. Expected pattern: lower `top1_share/peakiness_ratio`, worse `ROC-AUC/PR-AUC`, and increased false-positive risk.

## Citation

See `CITATION.cff`.

## License

MIT (`LICENSE`).
