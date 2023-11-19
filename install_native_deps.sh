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

sudo apt -y update
sudo apt install opencl-headers ocl-icd-libopencl1

mkdir neo
cd neo
wget https://github.com/intel/intel-graphics-compiler/releases/download/igc-1.0.14828.8/intel-igc-core_1.0.14828.8_amd64.deb
wget https://github.com/intel/intel-graphics-compiler/releases/download/igc-1.0.14828.8/intel-igc-opencl_1.0.14828.8_amd64.deb
wget https://github.com/intel/compute-runtime/releases/download/23.30.26918.9/intel-level-zero-gpu-dbgsym_1.3.26918.9_amd64.ddeb
wget https://github.com/intel/compute-runtime/releases/download/23.30.26918.9/intel-level-zero-gpu_1.3.26918.9_amd64.deb
wget https://github.com/intel/compute-runtime/releases/download/23.30.26918.9/intel-opencl-icd-dbgsym_23.30.26918.9_amd64.ddeb
wget https://github.com/intel/compute-runtime/releases/download/23.30.26918.9/intel-opencl-icd_23.30.26918.9_amd64.deb
wget https://github.com/intel/compute-runtime/releases/download/23.30.26918.9/libigdgmm12_22.3.0_amd64.deb
sudo dpkg -i *.deb
cd ..
rm -rf neo

sudo ln -s /usr/lib/x86_64-linux-gnu/libOpenCL.so.1 /usr/lib/libOpenCL.so

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
