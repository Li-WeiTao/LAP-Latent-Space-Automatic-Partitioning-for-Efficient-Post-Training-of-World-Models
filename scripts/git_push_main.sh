#!/usr/bin/env bash
# IME-safe push helper: avoids typing "origin" / "main" in the terminal.
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

export HTTPS_PROXY="${HTTPS_PROXY:-http://127.0.0.1:17891}"
export HTTP_PROXY="${HTTP_PROXY:-http://127.0.0.1:17891}"

remote="origin"
branch="main"

echo "Pushing $(git rev-parse --short HEAD) -> ${remote}/${branch}"
git push "${remote}" "${branch}"

echo
echo "Remote tip:"
git fetch "${remote}" "${branch}"
git log --oneline "${remote}/${branch}" -3
