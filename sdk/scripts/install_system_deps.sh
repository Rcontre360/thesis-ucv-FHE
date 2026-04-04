#!/usr/bin/env bash

set -e
set -x

# 1. Setup paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SDK_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
EXTERNAL_DIR="${SDK_DIR}/external"
INSTALL_PREFIX="${EXTERNAL_DIR}/install"
VENV_DIR="${SDK_DIR}/.venv"

mkdir -p "${EXTERNAL_DIR}"
mkdir -p "${INSTALL_PREFIX}"

# 2. Add FIDESlib as submodule if not already there
cd "${SDK_DIR}"
if [ ! -d "external/FIDESlib" ]; then
    echo "Adding FIDESlib as git submodule..."
    # Using the GitHub URL found in README
    git submodule add https://github.com/CAPS-UMU/FIDESlib.git external/FIDESlib
else
    echo "FIDESlib directory already exists. Ensuring submodules are up to date..."
    git submodule update --init --recursive
fi

# 3. Build patched OpenFHE (required by FIDESlib)
echo "Building patched OpenFHE..."
cd "${EXTERNAL_DIR}/FIDESlib/deps"
# The script build.sh takes the installation prefix as the first argument
# It handles cloning openfhe-src and applying the patch.
./build.sh "${INSTALL_PREFIX}"

# 4. Build FIDESlib
echo "Building FIDESlib..."
cd "${EXTERNAL_DIR}/FIDESlib"
mkdir -p build
cd build
cmake -DCMAKE_BUILD_TYPE=Release \
      -DOPENFHE_INSTALL_PREFIX="${INSTALL_PREFIX}" \
      -DFIDESLIB_INSTALL_PREFIX="${INSTALL_PREFIX}" \
      -DFIDESLIB_INSTALL_OPENFHE=OFF \
      -DFIDESLIB_COMPILE_TESTS=OFF \
      -DFIDESLIB_COMPILE_BENCHMARKS=OFF ..
make -j$(nproc)
make install

# 5. Build fideslib-python bindings
# NOTE: Instead of manual build, we now use 'pip install -e .'
# However, we need to point CMake to our install prefix for OpenFHE and FIDESlib
echo "System dependencies installed in ${INSTALL_PREFIX}."
echo "Now run 'pip install -e .' from the sdk/ directory to install the SDK and build bindings."
echo "Remember to set CMAKE_PREFIX_PATH=\"${INSTALL_PREFIX}\" if pip cannot find the libraries."
