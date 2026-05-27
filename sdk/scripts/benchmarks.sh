#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SDK_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [ -x "${SDK_DIR}/.env/bin/python" ]; then
    PYTHON="${SDK_DIR}/.env/bin/python"
else
    PYTHON="${PYTHON:-$(command -v python3)}" || { echo "python3 not found in PATH"; exit 1; }
fi

if [ -z "$1" ]; then
    echo "Usage: $0 <benchmark_name>"
    echo "Trains the case network once and runs every backend in its own"
    echo "subprocess, writing bench/<benchmark_name>/results.csv."
    echo "Available benchmarks:"
    for d in "${SDK_DIR}"/bench/*/; do
        name="$(basename "${d}")"
        [ -f "${d}bench.py" ] && echo "  ${name}"
    done
    exit 1
fi

BENCH="${SDK_DIR}/bench/${1}/bench.py"
if [ ! -f "$BENCH" ]; then
    echo "Error: benchmark '${1}' not found at ${BENCH}"
    exit 1
fi

exec "${PYTHON}" "$BENCH"
