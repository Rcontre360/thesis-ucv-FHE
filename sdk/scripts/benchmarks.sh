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
    echo "subprocess (per-library venv from bench/envs.toml), writing"
    echo "bench/<benchmark_name>/results.csv."
    echo "Available benchmarks:"
    for d in "${SDK_DIR}"/bench/*/; do
        name="$(basename "${d}")"
        [ -d "${d}processes" ] && echo "  ${name}"
    done
    exit 1
fi

if [ ! -d "${SDK_DIR}/bench/${1}/processes" ]; then
    echo "Error: benchmark '${1}' not found at bench/${1}/processes"
    exit 1
fi

cd "${SDK_DIR}"
exec "${PYTHON}" -m bench "$1"
