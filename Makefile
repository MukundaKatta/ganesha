.PHONY: test lint typecheck clean install dev

install:
	pip install -e .

dev:
	pip install -e ".[dev]"

test:
	PYTHONPATH=src python3 -m pytest tests/ -v --tb=short

test-cov:
	PYTHONPATH=src python3 -m pytest tests/ -v --tb=short --cov=ganesha --cov-report=term-missing

lint:
	ruff check src/ tests/

typecheck:
	mypy src/ganesha/

clean:
	rm -rf build/ dist/ *.egg-info src/*.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
