#!/usr/bin/env bash
set -e
SDK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec env PYTHONPATH="${SDK_DIR}:${PYTHONPATH}" python -m bench "$@"
