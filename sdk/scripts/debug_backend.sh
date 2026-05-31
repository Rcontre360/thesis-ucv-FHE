#!/usr/bin/env bash
# Run a single bench backend directly with live stdout/stderr (no subprocess capture),
# so SIGKILL / crash output is visible. The full `benchmarks.sh` orchestrator wraps
# each backend in capture_output=True, which can swallow the real cause.
#
# Usage: scripts/debug_backend.sh <case> <backend>
# Example: scripts/debug_backend.sh mlp run_orion
#
# Counts default to 1/1 for fast iteration; override with BENCH_LATENCY_N / BENCH_ACCURACY_N.
set -e

SDK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CASE="${1:?usage: $0 <case> <backend>}"
BACKEND="${2:?usage: $0 <case> <backend>}"
BID="${BACKEND#run_}"

# Pick the venv that matches bench/config.toml's [interpreters] table.
case "${BID}" in
    orion) PY="${SDK_DIR}/.env-orion/bin/python" ;;
    *)     PY="${SDK_DIR}/.env/bin/python" ;;
esac

ARTIFACTS="${SDK_DIR}/bench/${CASE}/artifacts/inputs.npz"
if [ ! -f "${ARTIFACTS}" ]; then
    echo "[debug] artifacts missing — running train first" >&2
    PYTHONPATH="${SDK_DIR}" BENCH_CASE="${CASE}" \
        "${SDK_DIR}/.env/bin/python" -m bench "${CASE}" train
fi

export PYTHONPATH="${SDK_DIR}:${PYTHONPATH}"
export BENCH_CASE="${CASE}"
export BENCH_LATENCY_N="${BENCH_LATENCY_N:-1}"
export BENCH_ACCURACY_N="${BENCH_ACCURACY_N:-1}"

echo "[debug] ${CASE}/${BACKEND}  python=${PY}  latency=${BENCH_LATENCY_N}  accuracy=${BENCH_ACCURACY_N}"
echo "----------------------------------------"
exec "${PY}" -m bench "${CASE}" "${BACKEND}"
