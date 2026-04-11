#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SDK_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON="${PYTHON:-$(command -v python3)}" || { echo "python3 not found in PATH"; exit 1; }

# Build HEonGPU and the Python bindings using the existing self-contained script.
PYTHON="${PYTHON}" bash "${SCRIPT_DIR}/build.sh"

# Install _backend.so into the venv's site-packages, replicating what
# pip would do as its final step — without re-running compiler discovery.
SITE_PACKAGES="$("${PYTHON}" -c "import sysconfig; print(sysconfig.get_path('platlib'))")"
cmake --install "${SDK_DIR}/build" \
    --prefix "${SITE_PACKAGES}" \
    --component python_modules

# HEonGPU shared libs are not installed system-wide; point the linker to them.
export LD_LIBRARY_PATH="${SDK_DIR}/build/heongpu/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

"${PYTHON}" -m pytest "${SDK_DIR}/tests/" -v
