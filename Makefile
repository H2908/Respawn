.PHONY: install dev test lint data reproduce clean

install:
	pip install -e .

dev:
	pip install -e ".[dev]"

test:
	pytest tests/ -v --tb=short

lint:
	ruff check respawn/ recoverybench/ experiments/ examples/ tests/
	ruff format --check respawn/ recoverybench/ experiments/ examples/ tests/

format:
	ruff format respawn/ recoverybench/ experiments/ examples/ tests/
	ruff check --fix respawn/ recoverybench/ experiments/ examples/ tests/

data:
	pip install -e ".[bench]"
	python experiments/whoandwhen.py --output data/whoandwhen_traces.jsonl
	python experiments/run_recoverybench.py --generate-only --output data/scenarios.jsonl

reproduce:
	pip install -e ".[bench]"
	python experiments/run_recoverybench.py \
		--scenarios data/scenarios.jsonl \
		--output results/ \
		--figures docs/images/

clean:
	rm -rf dist/ build/ *.egg-info .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
