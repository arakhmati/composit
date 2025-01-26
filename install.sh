#!/usr/bin/env bash

set -e
set -x

sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt -y update
sudo apt -y install python3.11

curl -sSL https://install.python-poetry.org | python3.11 -
export PATH=$HOME/.local/bin:$PATH
poetry install

bash install_native_deps.sh
