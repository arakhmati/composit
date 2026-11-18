#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# Python 3.11 and Poetry must already be installed; do not modify system Python.
python3.11 --version
poetry --version
bash install_native_deps.sh "$@"
poetry env use python3.11
poetry install
