#!/usr/bin/env bash

set -e
set -x

# 1. Setup paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SDK_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
EXTERNAL_DIR="${SDK_DIR}/external"
INSTALL_PREFIX="${EXTERNAL_DIR}/install"

mkdir -p "${EXTERNAL_DIR}"
mkdir -p "${INSTALL_PREFIX}"

# 2. Ensure FIDESlib is present
cd "${SDK_DIR}"
if [ ! -d "external/FIDESlib" ]; then
    echo "FIDESlib not found."
    exit 1
fi

# 3. Build patched OpenFHE (if not already done)
if [ ! -f "${INSTALL_PREFIX}/lib/libOPENFHEpke.so" ]; then
    echo "Building patched OpenFHE..."
    cd "${EXTERNAL_DIR}/FIDESlib/deps"
    ./build.sh "${INSTALL_PREFIX}"
else
    echo "OpenFHE already installed, skipping."
fi

# 4. Apply minimal source fixes to FIDESlib
cd "${EXTERNAL_DIR}/FIDESlib"

# Fix uint63_t typo in Context.cu
sed -i 's/uint63_t/uint64_t/g' src/CKKS/Context.cu

# Exclude multi-GPU source (not needed for single-GPU, has nvcc 12.8 parse issues)
if ! grep -q 'REMOVE_ITEM SOURCE_FILES_CUDA.*LimbPartitionMGPU' CMakeLists.txt; then
    sed -i '/file(GLOB_RECURSE SOURCE_FILES_CUDA \${SOURCE_DIR}\/\*.cu)/a list(REMOVE_ITEM SOURCE_FILES_CUDA "${SOURCE_DIR}/CKKS/LimbPartitionMGPU.cu")' CMakeLists.txt
fi

# 5. Build FIDESlib
echo "Building FIDESlib..."
rm -rf build
mkdir -p build
cd build

cmake -DCMAKE_BUILD_TYPE=Release \
      -DOPENFHE_INSTALL_PREFIX="${INSTALL_PREFIX}" \
      -DFIDESLIB_INSTALL_PREFIX="${INSTALL_PREFIX}" \
      -DFIDESLIB_INSTALL_OPENFHE=OFF \
      -DFIDESLIB_COMPILE_TESTS=OFF \
      -DFIDESLIB_COMPILE_BENCHMARKS=OFF \
      -DFIDESLIB_ARCH="86-real" \
      ..

make -j$(nproc)
cmake --build . --target install -j

echo "------------------------------------------------"
echo "FIDESlib built and installed to ${INSTALL_PREFIX}"
echo "------------------------------------------------"
