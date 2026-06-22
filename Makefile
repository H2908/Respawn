.PHONY: install dev test lint figures reproduce data clean

install:
	pip install -e .

dev:
	pip install -e ".[dev]"

test:
	pytest

lint:
	ruff check .

# Regenerate the synthetic figure + sweeps
figures:
	python experiments/run_recoverybench.py

# Clone the external Who&When dataset (licensed separately by its authors)
data:
	git clone --depth 1 https://github.com/ag2ai/Agents_Failure_Attribution.git

# Reproduce every number and figure in the README
reproduce: figures
	@if [ -d "Agents_Failure_Attribution/Who&When" ]; then \
		python experiments/whoandwhen.py --data "Agents_Failure_Attribution/Who&When"; \
	else \
		echo ">> Who&When not found. Run 'make data' first, then 'make reproduce'."; \
	fi

clean:
	rm -rf __pycache__ */__pycache__ .pytest_cache .ruff_cache *.egg-info build dist