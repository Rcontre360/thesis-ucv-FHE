#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SDK_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON="${PYTHON:-$(command -v python3)}" || { echo "python3 not found in PATH"; exit 1; }

"${PYTHON}" -m pip install "${SDK_DIR}"

export LD_LIBRARY_PATH="${SDK_DIR}/build/heongpu/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

"${PYTHON}" -m pytest "${SDK_DIR}/tests/" -v
