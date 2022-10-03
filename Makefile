test:
	pytest --cov=itb tests/

lint:
	black src/ tests/
	flake8 src/ tests/
	isort src/ tests/
