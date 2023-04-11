#!/usr/bin/env bash

# TODO: this should be part of a build system

set -e
set -x

mkdir vendor -p
cd vendor
if [ ! -d "immer" ] ; then
  git clone https://github.com/arximboldi/immer.git
fi
cd immer
git checkout 9dad616455aee3cf847ec349c6b5d98ca90b403b
cd ..
cd ..
