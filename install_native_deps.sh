#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

mode="prod"
opencl=0
while getopts "m:o" flag; do
    case "$flag" in
        m) mode="$OPTARG" ;;
        o) opencl=1 ;;
        *) exit 2 ;;
    esac
done

sudo apt-get update
sudo apt-get install -y build-essential git graphviz
# CPU setup needs no GPU driver. Install a vendor driver separately if needed.
if [ "$opencl" = 1 ]; then
    sudo apt-get install -y ocl-icd-opencl-dev
fi
mkdir -p vendor
if [ ! -d vendor/immer ]; then
    git clone https://github.com/arximboldi/immer.git vendor/immer
fi
git -C vendor/immer checkout 9dad616455aee3cf847ec349c6b5d98ca90b403b

if [ "$mode" = dev ]; then
    sudo apt-get install -y libglfw3-dev libfreetype6-dev libcapstone-dev libdbus-1-dev libxkbcommon-dev libwayland-dev libglvnd-dev
    if [ ! -d vendor/tracy ]; then
        git clone https://github.com/wolfpld/tracy vendor/tracy
    fi
    (cd vendor/tracy/profiler/build/unix && LEGACY=1 make release)
fi
