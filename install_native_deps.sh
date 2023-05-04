#!/usr/bin/env bash

# TODO: this should be part of a build system

set -e
set -x


mode="prod"

while getopts m: flag
do
    case "${flag}" in
        m) mode=${OPTARG};;
    esac
done

mkdir vendor -p
cd vendor
if [ ! -d "immer" ] ; then
  git clone https://github.com/arximboldi/immer.git
fi
cd immer
git checkout 9dad616455aee3cf847ec349c6b5d98ca90b403b

cd ..

if [ ! -d "tracy" ] && [ $mode = "dev" ] ; then
  sudo apt install -y libglfw3-dev libfreetype6-dev libcapstone-dev libdbus-1-dev libxkbcommon-dev libwayland-dev libglvnd-dev
  git clone https://github.com/wolfpld/tracy
  cd tracy/profiler/build/unix
  LEGACY=1 make release
fi
cd ..


cd ..
