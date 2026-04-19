#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SDK_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON="${PYTHON:-$(command -v python3)}" || { echo "python3 not found in PATH"; exit 1; }

if [ -z "$1" ]; then
    echo "Usage: $0 <example_name>"
    echo "Available examples:"
    for f in "${SDK_DIR}/examples/"*.py; do
        basename "$f" .py
    done
    exit 1
fi

EXAMPLE="${SDK_DIR}/examples/${1}.py"
if [ ! -f "$EXAMPLE" ]; then
    echo "Error: example '${1}' not found at ${EXAMPLE}"
    exit 1
fi

INSTALL_LIB="${SDK_DIR}/build/heongpu/lib"

export LD_LIBRARY_PATH="${INSTALL_LIB}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

exec "${PYTHON}" "$EXAMPLE"
