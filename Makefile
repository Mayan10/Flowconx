# FlowCon-X reproduction targets.
#
#   make setup        install the package and dev dependencies
#   make data         build both canonical datasets from data/raw/*.zip
#   make audit        shortcut and leakage audit on both datasets
#   make repro-small  reduced end-to-end pipeline, under 30 minutes on one GPU
#   make repro-full   every experiment behind every number in the paper
#   make paper        regenerate all tables and figures from results/
#   make test         lint, type check, unit tests, leakage tests
#
# Every target writes into results/ and splits/; neither is ever hand-edited.

PYTHON ?= python3
SEEDS_SMALL ?= 0 1 2
SEEDS_FULL ?= 0 1 2 3 4 5 6 7 8 9
DATASETS ?= cesnet_quic22 fiveg_traffic

.PHONY: setup data audit test lint typecheck repro-small repro-full paper clean-results help

help:
	@grep -E '^#   ' Makefile | sed 's/^#   //'

setup:
	$(PYTHON) -m pip install -e .
	$(PYTHON) -m pip install -r requirements-dev.txt

# ---------------------------------------------------------------- data
# Streams from the zip archives. Never expands them: they are 3.2 GB and
# 21 GB, and the pipeline is designed to run on a machine that cannot hold
# the expanded corpus.
data:
	$(PYTHON) -m flowconx.data.prepare --source all --seed 42

data-checksums:
	$(PYTHON) -m flowconx.data.prepare --source all --seed 42 --no-checksum

# ---------------------------------------------------------------- audit
audit:
	@for d in $(DATASETS); do \
		echo "=== audit: $$d ==="; \
		$(PYTHON) -m flowconx.audit.run_audit --csv data/processed/$$d.csv --seed 42 --rare-class-mode drop; \
	done

# ---------------------------------------------------------------- tests
lint:
	ruff check flowconx scripts tests

typecheck:
	mypy

test: lint typecheck
	$(PYTHON) -m pytest tests/ -q -m "not slow"

test-all: lint
	$(PYTHON) -m pytest tests/ -q

# ---------------------------------------------------------------- runs
# repro-small: one dataset, one split protocol, three seeds, short schedule.
# Sized to finish inside 30 minutes on a single GPU.
repro-small:
	@for s in $(SEEDS_SMALL); do \
		$(PYTHON) -m flowconx.run --config configs/cesnet_main.yaml --seed $$s \
			--set train.epochs=8 --set train.stage1_epochs=4 --overwrite; \
	done
	$(PYTHON) -m flowconx.analysis.aggregate --results results --out results/aggregate.json

# repro-full: the headline table, every ablation, every evaluation mode.
repro-full: data audit
	$(PYTHON) scripts/run_all_experiments.py --seeds $(SEEDS_FULL)
	$(PYTHON) -m flowconx.analysis.significance --results results --out results/significance.json
	$(PYTHON) -m flowconx.analysis.aggregate --results results --out results/aggregate.json
	$(MAKE) paper

paper:
	$(PYTHON) scripts/make_paper_assets.py --results results --out paper

clean-results:
	@echo "This deletes every committed number. Ctrl-C now if that is not what you meant."
	@sleep 5
	rm -rf results/*/ splits/*/
