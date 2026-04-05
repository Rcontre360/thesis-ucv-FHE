#!/usr/bin/env bash
set -e
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SDK_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
EXTERNAL_DIR="${SDK_DIR}/external"
INSTALL_PREFIX="${EXTERNAL_DIR}/install"
CMAKE="${SDK_DIR}/.venv/bin/cmake"

mkdir -p "${INSTALL_PREFIX}"

# Ensure HEonGPU submodule is present
cd "${SDK_DIR}"
if [ ! -d "external/HEonGPU/src" ]; then
    echo "Initializing HEonGPU submodule..."
    git submodule update --init --recursive
fi

# Build HEonGPU
echo "Building HEonGPU..."
cd "${EXTERNAL_DIR}/HEonGPU"

$CMAKE -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES=86 \
    -DCMAKE_INSTALL_PREFIX="${INSTALL_PREFIX}" \
    -DTHRUST_INCLUDE_DIR="$(dirname "$(which nvcc)")/../targets/x86_64-linux/include"

$CMAKE --build build -j"$(nproc)"
$CMAKE --install build

echo "------------------------------------------------"
echo "HEonGPU built and installed to ${INSTALL_PREFIX}"
echo "------------------------------------------------"
