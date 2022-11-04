#!/usr/bin/env sh

set -e

# Formatters
black --check networkx_persistent tests/unit_tests

# Type Checkers
mypy --config-file mypy.ini networkx_persistent

# Linters
MYPYPATH=/dev/null flake8 --config .flake8 --mypy-config mypy.ini networkx_persistent tests/unit_tests
pylint --rcfile .pylintrc networkx_persistent tests/unit_tests
