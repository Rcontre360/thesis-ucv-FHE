#!/usr/bin/env bash
# Publish the current `pyproject.toml` version of fhe-sdk to PyPI.
#
# Workflow:
#   1) bump `version = "x.y.z"` in pyproject.toml yourself
#   2) make any source changes you want shipped
#   3) ./scripts/publish.sh
#
# This does NOT check that the version was bumped — PyPI will reject a
# duplicate version with a 400 error, which is your "did you forget?" signal.
#
# Expects:
#   - `.env/` virtualenv at the SDK root (created by your usual workflow)
#   - `~/.pypirc` with valid PyPI credentials
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SDK_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${SDK_DIR}"

if [ ! -f ".env/bin/activate" ]; then
    echo "error: ${SDK_DIR}/.env not found — create the venv first." >&2
    exit 1
fi
# shellcheck source=/dev/null
source .env/bin/activate

pip install --quiet --upgrade build twine

rm -rf dist/ build/
python -m build --sdist
twine upload dist/*.tar.gz

echo
echo "done — view at https://pypi.org/project/fhe-sdk/"
