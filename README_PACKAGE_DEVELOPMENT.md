# Setup development env

```bash
python3.10 -m venv venv
source venv/bin/activate
pip install -e '.[dev]'
```

# Package development

This section of readme is responsible for descriptions of methods for development of
the package and contain the most important commands to run.

## Generate dist version

```bash
# build package
python -m build
```

## install package on local machine

```
pip install -e .
```
## Test deploy
### Validate the printing information on PYPI using *twine*
```bash
twine check dist/*
```

### Upload package to testpypi using *twine*
```bash
twine upload -r testpypi dist/*
```

## Deploy
### Upload package to pypi using *twine*
```bash
twine upload -r pypi dist/*
```

## Installing extra dependencies from the PYPI when testing installation on test PYPI

```bash
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple itb"
```
