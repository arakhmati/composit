#!/usr/bin/env sh

set -e

# Formatters
black --check composit tests/unit_tests

# Type Checkers
mypy --config-file mypy.ini composit

# Linters
MYPYPATH=/dev/null flake8 --config .flake8 --mypy-config mypy.ini composit tests/unit_tests
pylint --rcfile .pylintrc composit tests/unit_tests
