.PHONY: install install-dev install-all lint test smoke run-all figures download train-command

install:
	python -m pip install -e .

install-dev:
	python -m pip install -e ".[dev]"

install-all:
	python -m pip install -e ".[all]"

lint:
	ruff check .

test:
	pytest

smoke:
	bash scripts/run_smoke_pipeline.sh

run-all:
	instella-reasoning run-all --config configs/pipeline/full.yaml

figures:
	instella-reasoning plots --scores outputs/atlas_run/scores.jsonl \
		--contamination outputs/atlas_run/contamination.jsonl \
		--output-dir outputs/atlas_run/figures

download:
	bash scripts/download_data.sh

train-command:
	instella-reasoning train-command --config configs/training/amd_base_upstream.yaml
