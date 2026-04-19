#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SDK_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
CMAKE="$(command -v cmake)" || { echo "cmake not found in PATH"; exit 1; }
PYTHON="${PYTHON:-$(command -v python3)}" || { echo "python3 not found in PATH"; exit 1; }
if ! "${PYTHON}" -m pybind11 --cmakedir &>/dev/null; then
    echo "pybind11 not found in the active Python environment ($(${PYTHON} --version))."
    echo "Activate an environment that has pybind11 installed, e.g.:"
    echo "  pip install pybind11"
    exit 1
fi
PYBIND11_DIR="$("${PYTHON}" -m pybind11 --cmakedir)"

CUDA_ARCH="${CUDA_ARCH:-86}"
BUILD_JOBS="${BUILD_JOBS:-12}"

echo "Building fhe-sdk Python bindings..."
$CMAKE -S "${SDK_DIR}" -B "${SDK_DIR}/build" \
    -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH}" \
    -DCMAKE_PREFIX_PATH="${SDK_DIR}/build/heongpu" \
    -Dpybind11_DIR="${PYBIND11_DIR}"

$CMAKE --build "${SDK_DIR}/build" -j"${BUILD_JOBS}"

echo "--------------------------------------------"
echo "fhe-sdk built. Bindings at build/src/backend/"
echo "--------------------------------------------"
