#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SDK_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON="${PYTHON:-$(command -v python3)}" || { echo "python3 not found in PATH"; exit 1; }

if [ -z "$1" ]; then
    echo "Usage: $0 <example_name>"
    echo "Available examples:"
    while IFS= read -r f; do
        # Print path relative to examples/, without .py
        rel="${f#${SDK_DIR}/examples/}"
        echo "  ${rel%.py}"
    done < <(find "${SDK_DIR}/examples" -name "*.py" ! -name "network.py" | sort)
    exit 1
fi

EXAMPLE="${SDK_DIR}/examples/${1}.py"
if [ ! -f "$EXAMPLE" ]; then
    echo "Error: example '${1}' not found at ${EXAMPLE}"
    exit 1
fi

exec "${PYTHON}" "$EXAMPLE"
