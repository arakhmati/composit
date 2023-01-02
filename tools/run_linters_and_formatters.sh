#!/usr/bin/env sh

set -e

# Formatters
black --check persistent_numpy tests/unit_tests

# Type Checkers
mypy --config-file mypy.ini persistent_numpy

# Linters
MYPYPATH=/dev/null flake8 --config .flake8 --mypy-config mypy.ini persistent_numpy tests/unit_tests
pylint --rcfile .pylintrc persistent_numpy tests/unit_tests
