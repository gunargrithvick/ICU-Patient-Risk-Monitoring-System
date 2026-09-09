# Shortcuts for the things you actually run. Every target here is a command you could type
# by hand - nothing is hidden, and `make -n <target>` prints it if you want to see first.
#
# `make help` lists everything.

PY ?= python
PIP := $(PY) -m pip
RUFF := $(PY) -m ruff
PYTEST := $(PY) -m pytest

# The clinical text in this codebase is UTF-8 (SpO₂, ≥, °C). Windows consoles default to
# cp1252 and the first alert line would raise UnicodeEncodeError, so it is set for every
# target rather than remembered per command.
export PYTHONIOENCODING := utf-8

.DEFAULT_GOAL := help
.PHONY: help install dev lint fmt fmt-check check test test-fast cov etl etl-synthetic \
        train bootstrap serve dash tick info docker-build docker-test up down logs \
        clean clean-artifacts

help: ## List the available targets
	@grep -hE '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

# ------------------------------------------------------------------------------ install

install: ## Install the package (runtime dependencies only)
	$(PIP) install -e .

dev: ## Install with the test and lint toolchain
	$(PIP) install -e ".[dev]"

# --------------------------------------------------------------------------------- checks

lint: ## Ruff, including import order
	$(RUFF) check src tests

fmt: ## Reformat in place
	$(RUFF) format src tests
	$(RUFF) check src tests --fix

fmt-check: ## Fail if anything is unformatted (this is what CI runs)
	$(RUFF) format --check src tests

check: lint fmt-check test ## Everything CI runs, in the same order

test: ## The full suite
	$(PYTEST) tests/

test-fast: ## Skip anything marked slow
	$(PYTEST) tests/ -m "not slow"

cov: ## Suite with a coverage report
	$(PYTEST) tests/ --cov --cov-report=term-missing --cov-report=html
	@echo "HTML report: htmlcov/index.html"

# ------------------------------------------------------------------------ data and model

etl: ## Build the window dataset from data/raw (PhysioNet extract)
	$(PY) -m icu_monitor etl

etl-synthetic: ## Build a stand-in dataset with the simulator - no download needed
	$(PY) -m icu_monitor etl --synthetic --stays 900

train: ## Train, evaluate, and write the artefact + model card
	$(PY) -m icu_monitor train

bootstrap: etl-synthetic train ## Dataset + model from nothing, for a fresh clone

# -------------------------------------------------------------------------------- running

serve: ## FastAPI on ICU_API_PORT (default 8000); /docs for the schema
	$(PY) -m icu_monitor serve

dash: ## The Streamlit dashboard on :8501
	$(PY) -m icu_monitor dashboard

tick: ## Advance the ward and print it - no browser, no server
	$(PY) -m icu_monitor tick

info: ## Resolved settings and per-component readiness
	$(PY) -m icu_monitor info

# --------------------------------------------------------------------------------- docker

docker-build: ## Build the image
	docker build -t icu-monitor:2.0.0 .

docker-test: ## Lint + suite inside the image, on Linux Python
	docker compose --profile ci run --rm test

up: ## API on :8000 and dashboard on :8501, both bound to localhost
	docker compose up -d
	@echo "dashboard  http://localhost:8501"
	@echo "api docs   http://localhost:8000/docs"

down: ## Stop both, keep the data volumes
	docker compose down

logs: ## Follow both services
	docker compose logs -f

# --------------------------------------------------------------------------------- tidying

clean: ## Remove caches and build output
	$(PY) -c "import pathlib,shutil; [shutil.rmtree(p, ignore_errors=True) for p in ['build','dist','htmlcov','.pytest_cache','.ruff_cache','.mypy_cache']]"
	$(PY) -c "import pathlib,shutil; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('__pycache__')]"
	$(PY) -c "import pathlib,shutil; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('*.egg-info')]"
	$(PY) -c "import pathlib; [p.unlink(missing_ok=True) for p in pathlib.Path('.').glob('.coverage*')]"

clean-artifacts: ## Also delete the trained model and processed data (both rebuildable)
	$(PY) -c "import shutil; shutil.rmtree('artifacts', ignore_errors=True); shutil.rmtree('data/processed', ignore_errors=True)"
