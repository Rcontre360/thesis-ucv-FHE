#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SDK_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "Cleaning build artifacts..."
rm -rf "${SDK_DIR}/build"

echo "Cleaning HEonGPU build (third-party cache)..."
rm -rf "${SDK_DIR}/external/HEonGPU/build"

echo "Done."
