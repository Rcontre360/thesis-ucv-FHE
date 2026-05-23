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

# Ensure HEonGPU submodule is present (when running from a git checkout — pip
# sdists already ship the source so this is skipped automatically).
cd "${SDK_DIR}"
if [ ! -d "external/HEonGPU/src" ]; then
    if [ -d ".git" ]; then
        echo "Initializing HEonGPU submodule..."
        git submodule update --init --recursive
    else
        echo "HEonGPU source missing and no .git available — sdist may be incomplete."
        exit 1
    fi
fi

# HEonGPU's thirdparty CMakeLists runs thirdparty/build.sh, which itself does a
# `git submodule update --init --recursive` for its nested submodules. That
# fails inside a pip sdist (no .git). Replace it with a no-op when the nested
# submodule source is already present (it always is when shipped via sdist).
THIRDPARTY_BUILD_SH="${EXTERNAL_DIR}/HEonGPU/thirdparty/build.sh"
if [ -f "${EXTERNAL_DIR}/HEonGPU/thirdparty/GPU-FFT/CMakeLists.txt" ]; then
    cat > "${THIRDPARTY_BUILD_SH}" <<'EOF'
#!/usr/bin/env bash
# Patched by fhe-sdk install: nested-submodule source is pre-populated.
exit 0
EOF
    chmod +x "${THIRDPARTY_BUILD_SH}"
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

$CMAKE --build build -j"${BUILD_JOBS:-$(nproc 2>/dev/null || echo 8)}"
$CMAKE --install build

echo "------------------------------------------------"
echo "HEonGPU built and installed to ${INSTALL_PREFIX}"
echo "------------------------------------------------"
