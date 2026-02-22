PYTHON ?= python3
VENV ?= .venv
PIP := $(VENV)/bin/pip
PY := $(VENV)/bin/python
SITE_DIR := site

.PHONY: setup format lint test demo build-site serve-api export-demo clean

setup:
	@if [ ! -d "$(VENV)" ]; then $(PYTHON) -m venv $(VENV); fi
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev]"
	cd $(SITE_DIR) && corepack enable && pnpm install

format:
	$(VENV)/bin/black ads scripts tests
	$(VENV)/bin/isort ads scripts tests
	cd $(SITE_DIR) && pnpm format

lint:
	$(VENV)/bin/ruff check ads scripts tests
	$(VENV)/bin/black --check ads scripts tests
	$(VENV)/bin/isort --check-only ads scripts tests
	$(VENV)/bin/mypy ads
	cd $(SITE_DIR) && pnpm lint

test:
	$(PY) -m pytest

demo:
	bash scripts/demo_end_to_end.sh

build-site:
	cd $(SITE_DIR) && pnpm build

serve-api:
	$(VENV)/bin/uvicorn ads.api:app --host 127.0.0.1 --port 8000 --reload

export-demo:
	$(PY) scripts/export_demo_assets.py

clean:
	rm -rf artifacts site/out
