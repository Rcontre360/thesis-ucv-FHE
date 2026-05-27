#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SDK_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Orion (and the SDK) require Python in [3.9, 3.13); the system "python3" may be
# newer (e.g. a 3.14 from Homebrew), so prefer an explicit compatible version.
if [ -z "${BASE_PYTHON:-}" ]; then
    for cand in python3.12 python3.11 python3.10 python3.9; do
        if command -v "$cand" >/dev/null 2>&1; then BASE_PYTHON="$(command -v "$cand")"; break; fi
    done
fi
if [ -z "${BASE_PYTHON:-}" ] && [ -x "${SDK_DIR}/.env/bin/python" ]; then
    BASE_PYTHON="${SDK_DIR}/.env/bin/python"
fi
[ -n "${BASE_PYTHON:-}" ] || { echo "no Python 3.9-3.12 found; set BASE_PYTHON"; exit 1; }
echo "Using base python: ${BASE_PYTHON} ($(${BASE_PYTHON} --version 2>&1))"

NAME="$1"
if [ -z "$NAME" ]; then
    echo "Usage: $0 <library>"
    echo "Creates the per-library venv declared in bench/envs.toml and installs"
    echo "that library's dependencies."
    echo "Known libraries:"
    echo "  orion   -> .env-orion  (unconstrained torch so dynamo works on py3.12)"
    exit 1
fi

case "$NAME" in
  orion)
    VENV="${SDK_DIR}/.env-orion"
    ORION_SRC="${SDK_DIR}/temp/orion"
    if [ ! -d "$ORION_SRC" ]; then
        echo "Cloning Orion into ${ORION_SRC} ..."
        git clone https://github.com/baahl-nyu/orion.git "$ORION_SRC"
    fi
    [ -d "$VENV" ] || "$BASE_PYTHON" -m venv "$VENV"
    "$VENV/bin/pip" install -U pip wheel
    # Unconstrained on purpose: in its own env Orion may pull whatever torch it
    # needs (>=2.4 supports torch.compile/dynamo on Python 3.12).
    "$VENV/bin/pip" install -e "$ORION_SRC"
    # Extra deps the shared bench utils import (measure.py).
    "$VENV/bin/pip" install pynvml
    ;;
  *)
    echo "Unknown library '${NAME}'. Known: orion"
    exit 1
    ;;
esac

echo "----------------------------------------"
echo "Created ${VENV}"
"$VENV/bin/python" -c "import torch, numpy, pynvml, orion; print('ok: torch', torch.__version__)"
