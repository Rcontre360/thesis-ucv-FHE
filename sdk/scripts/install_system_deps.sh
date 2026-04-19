#!/usr/bin/env bash
set -e
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SDK_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
EXTERNAL_DIR="${SDK_DIR}/external"
INSTALL_PREFIX="${SDK_DIR}/build/heongpu"
CMAKE="$(command -v cmake)" || { echo "cmake not found in PATH. Install cmake >= 3.30 (e.g. pip install cmake)"; exit 1; }
CMAKE_VERSION="$("${CMAKE}" --version | head -1 | awk '{print $3}')"
CMAKE_MAJOR="$(echo "${CMAKE_VERSION}" | cut -d. -f1)"
CMAKE_MINOR="$(echo "${CMAKE_VERSION}" | cut -d. -f2)"
if [ "${CMAKE_MAJOR}" -lt 3 ] || { [ "${CMAKE_MAJOR}" -eq 3 ] && [ "${CMAKE_MINOR}" -lt 30 ]; }; then
    echo "cmake ${CMAKE_VERSION} is too old. HEonGPU requires cmake >= 3.30."
    echo "Install a newer version, e.g.: pip install cmake"
    exit 1
fi

mkdir -p "${INSTALL_PREFIX}"

if [ -n "$(ls -d "${INSTALL_PREFIX}/lib/cmake/HEonGPU"* 2>/dev/null)" ]; then
    echo "HEonGPU already installed at ${INSTALL_PREFIX}, skipping."
    exit 0
fi

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
    -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH:-86}" \
    -DCMAKE_INSTALL_PREFIX="${INSTALL_PREFIX}" \
    -DTHRUST_INCLUDE_DIR="$(dirname "$(which nvcc)")/../targets/x86_64-linux/include" \
    -DCMAKE_CUDA_FLAGS="--pre-include cstdint"

$CMAKE --build build -j"${BUILD_JOBS:-8}"
$CMAKE --install build

echo "------------------------------------------------"
echo "HEonGPU built and installed to ${INSTALL_PREFIX}"
echo "------------------------------------------------"
