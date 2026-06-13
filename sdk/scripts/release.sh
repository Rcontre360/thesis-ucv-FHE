#!/usr/bin/env bash
# Release fhe-sdk: test, bump the version, publish to PyPI, then tag + push.
#
# Usage:
#   ./scripts/release.sh 0.2.0 "BSGS cached baby steps"   # run after last commit
#
# Order (each step aborts the release on failure):
#   1) run the test suite
#   2) only if tests pass: sed the version into pyproject.toml + commit it
#   3) build the sdist (carries the bumped version — so it must follow the sed)
#   4) twine upload
#   5) git tag v<version>
#   6) git push (the bump commit) + push the tag
#
# The tag is created here, last — never create it by hand. Because the version
# bump and tag happen only after tests pass and the upload succeeds, a failure
# leaves no wrong-versioned commit pushed and no dangling tag.
#
# Expects:
#   - `.env/` virtualenv at the SDK root
#   - `~/.pypirc` with valid PyPI credentials
set -euo pipefail

if [ $# -ne 2 ]; then
    echo "usage: $0 <version> <message>   e.g. $0 0.2.0 \"BSGS cached baby steps\"" >&2
    exit 2
fi
VERSION="${1#v}"               # accept "0.2.0" or "v0.2.0"
TAG="v${VERSION}"
MESSAGE="$2"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SDK_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${SDK_DIR}"

if [ ! -f ".env/bin/activate" ]; then
    echo "error: ${SDK_DIR}/.env not found — create the venv first." >&2
    exit 1
fi
# shellcheck source=/dev/null
source .env/bin/activate

# 1) tests — abort the release if anything fails.
python -m pytest tests/

# 2) bump the version only now that tests passed, and commit it so the tag
#    points at the corrected pyproject.toml.
sed -i -E 's/^version = ".*"/version = "'"${VERSION}"'"/' pyproject.toml
git add pyproject.toml
git commit -m "release: ${VERSION}"

# 3) build the sdist (now carries ${VERSION}).
pip install --quiet --upgrade build twine
rm -rf dist/ build/
python -m build --sdist

# 4) publish.
twine upload dist/*.tar.gz

# 5) + 6) tag last (annotated, with the message you passed), then push the
#         commit and the tag.
git tag -a "${TAG}" -m "${MESSAGE}"
git push
git push origin "${TAG}"

echo
echo "released ${TAG} — https://pypi.org/project/fhe-sdk/${VERSION}/"
