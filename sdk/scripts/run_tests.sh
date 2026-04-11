#!/usr/bin/env bash
# Build the SDK (HEonGPU + Python bindings) and run the test suite.
#
# Usage:
#   ./scripts/run_tests.sh              # build + test
#   ./scripts/run_tests.sh --no-build   # skip build, only run tests
#   ./scripts/run_tests.sh -k "encode"  # pass extra args to pytest
#
# Environment variables:
#   CUDA_ARCH   GPU compute capability (default: 86 for RTX 30/40 series)
#   PYTHON      Python interpreter to use (default: auto-detected from .env)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SDK_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ---------------------------------------------------------------------------
# Python interpreter — prefer the project venv
# ---------------------------------------------------------------------------
if [ -z "${PYTHON:-}" ]; then
    if [ -x "${SDK_DIR}/.env/bin/python" ]; then
        PYTHON="${SDK_DIR}/.env/bin/python"
    else
        PYTHON="$(command -v python3)"
    fi
fi

echo "Using Python: ${PYTHON} ($(${PYTHON} --version))"

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
SKIP_BUILD=0
PYTEST_ARGS=()
for arg in "$@"; do
    if [ "${arg}" = "--no-build" ]; then
        SKIP_BUILD=1
    else
        PYTEST_ARGS+=("${arg}")
    fi
done

# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
if [ "${SKIP_BUILD}" -eq 0 ]; then
    echo ""
    echo "==> Building HEonGPU and Python bindings..."
    bash "${SCRIPT_DIR}/build.sh"
fi

# ---------------------------------------------------------------------------
# Environment for the test run
# ---------------------------------------------------------------------------
BACKEND_DIR="${SDK_DIR}/build/src/backend"
HEONGPU_LIB="${SDK_DIR}/build/heongpu/lib"

if [ ! -d "${BACKEND_DIR}" ]; then
    echo ""
    echo "ERROR: ${BACKEND_DIR} does not exist."
    echo "       Run without --no-build or run scripts/build.sh first."
    exit 1
fi

export PYTHONPATH="${SDK_DIR}/src:${BACKEND_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export LD_LIBRARY_PATH="${HEONGPU_LIB}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

# ---------------------------------------------------------------------------
# Ensure pytest is installed
# ---------------------------------------------------------------------------
if ! "${PYTHON}" -m pytest --version &>/dev/null; then
    echo "Installing pytest into the active environment..."
    "${PYTHON}" -m pip install pytest --quiet
fi

# ---------------------------------------------------------------------------
# Run tests
# ---------------------------------------------------------------------------
echo ""
echo "==> Running tests (PYTHONPATH includes ${BACKEND_DIR})..."
echo ""

cd "${SDK_DIR}"
"${PYTHON}" -m pytest tests/ -v "${PYTEST_ARGS[@]}"
