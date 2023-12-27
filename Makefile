# installs package and its dependencies
install:
	pip install -e .

# installs package, its dependencies and development dependencies
install_dev:
	pip install -e .[dev]

# runs tests
test:
	pytest --cov=itb tests/

# runs code linting
lint:
	black src/ tests/
	flake8 src/ tests/
	isort src/ tests/
