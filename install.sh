#!/usr/bin/env bash

set -e
set -x

sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.11

curl -sSL https://install.python-poetry.org | python3.11 -
export PATH=$HOME/.local/bin:$PATH
poetry shell
poetry install

bash install_native_deps.sh
