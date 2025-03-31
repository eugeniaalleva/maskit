.PHONY: install reinstall dev test format lint clean

# Editable install
install:
	pip install -e .


# Dev install with extras
dev:
	pip install -e .[dev]

# Reinstall the editable package (force re-reading dependencies)
devreinstall:
	pip uninstall -y maskit || true
	pip install -e .[dev]

# Run tests
test:
	pytest

# Format code using black
format:
	black src test

# Lint with flake8 (optional)
lint:
	flake8 src test

# Clean build artifacts
clean:
	rm -rf build dist *.egg-info .pytest_cache __pycache__ .mypy_cache