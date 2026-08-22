SHELL := /bin/bash
PY ?= uv run python

# Knobs shared by the pipeline targets (override on the command line).
DATA ?=
BACKBONE ?= xgboost
K ?= 70
M ?= 2200
ARGS ?=

define require_data
	$(if $(DATA),,$(error DATA is required, e.g. make $@ DATA=/abs/path/dataset.csv))
endef

.PHONY: sync lint fix test demo rffs baselines conformal explain comparison \
	paper-rffs paper-no-prs paper-analysis

# ---- Setup and development -------------------------------------------------

sync:
	uv sync --all-groups

lint:
	uv run ruff format --check .
	uv run ruff check --no-fix .

fix:
	uv run ruff format .
	uv run ruff check --fix .

test:
	uv run pytest

# ---- Demo: the full pipeline on the bundled public dataset -----------------
# UCI Superconductivity (CC BY 4.0) ships in public-datasets/, so this works
# right after `make sync`. Results land in results/demo/.

DEMO_DATA := public-datasets/superconductivity/xopa/superconductivity_xopa.csv
DEMO_OUT := results/demo
DEMO_CONFIG := configs/superconductivity.yaml

demo:
	$(PY) scripts/run_rffs.py --data $(DEMO_DATA) --config $(DEMO_CONFIG) --out $(DEMO_OUT) \
		--operators tree mrmr lasso --grid 70 50 30
	latest=$$(ls -td $(DEMO_OUT)/*_rffs_xgboost | head -1); \
	$(PY) scripts/run_conformal.py --data $(DEMO_DATA) --config $(DEMO_CONFIG) --out $(DEMO_OUT) \
		--features-file $$latest/rffs_summary.json --m 500 && \
	$(PY) scripts/run_explain.py --data $(DEMO_DATA) --config $(DEMO_CONFIG) --out $(DEMO_OUT) \
		--features-file $$latest/rffs_summary.json && \
	$(PY) scripts/run_baselines.py --data $(DEMO_DATA) --config $(DEMO_CONFIG) --out $(DEMO_OUT) \
		--k 30 --boruta-max-iter 30 && \
	$(PY) scripts/run_model_comparison.py --data $(DEMO_DATA) --config $(DEMO_CONFIG) --out $(DEMO_OUT) \
		--feature-sets $$latest/rffs_summary.json

# ---- Pipeline stages on an industrial dataset ------------------------------
# Every target requires DATA=/abs/path/dataset.csv (see the README for the
# expected schema). Extra script flags go in ARGS, e.g.
#   make conformal DATA=... ARGS="--features-file <run>/rffs_summary.json"

rffs:
	$(call require_data)
	$(PY) scripts/run_rffs.py --data $(DATA) --backbone $(BACKBONE) $(ARGS)

baselines:
	$(call require_data)
	$(PY) scripts/run_baselines.py --data $(DATA) --backbone $(BACKBONE) --k $(K) $(ARGS)

conformal:
	$(call require_data)
	$(PY) scripts/run_conformal.py --data $(DATA) --backbone $(BACKBONE) --m $(M) $(ARGS)

explain:
	$(call require_data)
	$(PY) scripts/run_explain.py --data $(DATA) --backbone $(BACKBONE) $(ARGS)

comparison:
	$(call require_data)
	$(PY) scripts/run_model_comparison.py --data $(DATA) $(ARGS)

# ---- Paper campaign (authors only) -----------------------------------------
# Requires the proprietary dataset, which is never stored in this repository.
# --paper validates the dataset identity (hash, shapes) before anything runs;
# the regime window is the record range analyzed conditionally in the paper.

BACKBONES ?= xgboost random_forest lightgbm mlp
REGIME ?= 22500 28499

paper-rffs:
	$(call require_data)
	for b in $(BACKBONES); do \
		$(PY) scripts/run_rffs.py --data $(DATA) --backbone $$b --paper $(ARGS) || exit 1; \
	done

paper-no-prs:
	$(call require_data)
	$(PY) scripts/run_rffs.py --data $(DATA) --backbone xgboost --paper \
		--exclude-categories PRS --label rffs_noprs $(ARGS)

paper-analysis:
	$(call require_data)
	$(if $(RFFS_SUMMARY),,$(error RFFS_SUMMARY is required: path to the winning run's rffs_summary.json))
	$(PY) scripts/run_baselines.py --data $(DATA) --k $(K) --paper
	$(PY) scripts/run_conformal.py --data $(DATA) --m $(M) --paper \
		--features-file $(RFFS_SUMMARY) --slice $(REGIME)
	$(PY) scripts/run_explain.py --data $(DATA) --paper \
		--features-file $(RFFS_SUMMARY) --highlight $(REGIME)
	$(PY) scripts/run_model_comparison.py --data $(DATA) --paper \
		--feature-sets $(RFFS_SUMMARY)
