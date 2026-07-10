.PHONY: install install-dev lint test smoke train-command

install:
	python -m pip install -e .

install-dev:
	python -m pip install -e ".[dev]"

lint:
	ruff check .

test:
	pytest

smoke:
	bash scripts/run_smoke_pipeline.sh

train-command:
	instella-reasoning train-command --config configs/training/amd_base_upstream.yaml
