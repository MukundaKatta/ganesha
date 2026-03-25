# Contributing to Ganesha

Thank you for your interest in contributing to Ganesha!

## Development Setup

```bash
# Clone the repository
git clone https://github.com/MukundaKatta/ganesha.git
cd ganesha

# Install in development mode
pip install -e ".[dev]"

# Run tests
make test

# Run linter
make lint

# Run type checker
make typecheck
```

## Running Tests

```bash
# All tests
make test

# With coverage
make test-cov

# Specific test file
PYTHONPATH=src python3 -m pytest tests/test_core.py -v
```

## Code Style

- We use [Ruff](https://github.com/astral-sh/ruff) for linting.
- Type hints are expected for all public APIs.
- Target Python 3.9+ compatibility.

## Pull Request Process

1. Fork the repository and create a feature branch.
2. Write tests for any new functionality.
3. Ensure all tests pass (`make test`).
4. Ensure linting passes (`make lint`).
5. Submit a pull request with a clear description.

## Project Structure

- `src/ganesha/` — Library source code.
- `tests/` — Test suite.
- `docs/` — Documentation.
