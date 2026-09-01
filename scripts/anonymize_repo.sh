#!/usr/bin/env bash
# Produce an anonymous copy of the repository for double-blind submission.
#
#   ./scripts/anonymize_repo.sh /tmp/flowconx-anonymous
#
# Writes a *copy*. It never modifies the working tree, because an anonymiser
# that runs in place is one accidental commit away from destroying authorship
# history.
#
# What it removes:
#   - the .git directory entirely (history carries names, emails and remotes)
#   - author names, institution and contact details from every text file
#   - remote URLs and anything pointing at the hosting account
#   - absolute paths containing a home directory name
#   - the LICENSE copyright line
#
# What it does NOT remove, deliberately:
#   - results/, splits/ and paper/ -- the artifact is the point
#   - dataset citations and DOIs -- those are third-party, not identifying
#
# Always read the diff before submitting. This script is a first pass, not a
# guarantee, and the reviewer-facing consequence of a missed identifier is
# a desk reject.

set -euo pipefail

DEST="${1:-}"
if [ -z "$DEST" ]; then
  echo "usage: $0 <destination-directory>" >&2
  exit 1
fi
if [ -e "$DEST" ]; then
  echo "error: $DEST already exists; refusing to overwrite" >&2
  exit 1
fi

SRC="$(cd "$(dirname "$0")/.." && pwd)"
echo "==> Copying $SRC to $DEST"
mkdir -p "$DEST"
# Excludes: history, caches, raw data, virtualenvs, checkpoints.
rsync -a \
  --exclude '.git' \
  --exclude '.idea' \
  --exclude '__pycache__' \
  --exclude '.pytest_cache' \
  --exclude '.ruff_cache' \
  --exclude '.mypy_cache' \
  --exclude '*.egg-info' \
  --exclude '.venv' \
  --exclude 'data/raw' \
  --exclude 'data/processed' \
  --exclude '*.pt' \
  --exclude '.DS_Store' \
  "$SRC/" "$DEST/"

# Identifying strings. Extend this list before every submission.
declare -a PATTERNS=(
  's|Mayan10|ANONYMOUS|g'
  's|mayan25sharma@gmail\.com|anonymous@example.com|g'
  's|Mayan Sharma|Anonymous Author|g'
  's|/Users/mayan|/path/to|g'
  's|github\.com/[A-Za-z0-9_.-]*/Flowconx|github.com/ANONYMOUS/flowconx|gI'
  's|Co-Authored-By:.*|Co-Authored-By: ANONYMOUS|g'
)

echo "==> Scrubbing identifying strings"
while IFS= read -r -d '' file; do
  for pattern in "${PATTERNS[@]}"; do
    if [ "$(uname)" = "Darwin" ]; then
      sed -i '' -E "$pattern" "$file" 2>/dev/null || true
    else
      sed -i -E "$pattern" "$file" 2>/dev/null || true
    fi
  done
done < <(find "$DEST" -type f \( -name '*.py' -o -name '*.md' -o -name '*.yaml' -o -name '*.yml' \
         -o -name '*.toml' -o -name '*.sh' -o -name '*.json' -o -name '*.tex' -o -name 'Makefile' \
         -o -name 'Dockerfile' -o -name 'LICENSE' \) -print0)

cat > "$DEST/LICENSE" <<'LIC'
MIT License

Copyright (c) 2026 Anonymous Author(s)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
LIC

echo "==> Residual scan (review every hit before submitting)"
# This script is excluded from its own scan: it necessarily contains the
# patterns it searches for, and flagging them every run trains you to ignore
# the output, which is how a real identifier gets through.
LEAKS=$(grep -rIniE 'mayan|sharma|@gmail|github\.com/[A-Za-z0-9_.-]+/' "$DEST" \
  --exclude-dir=.git \
  --exclude 'anonymize_repo.sh' 2>/dev/null \
  | grep -viE 'zenodo|kaggle|anonymous|example\.com' || true)
if [ -n "$LEAKS" ]; then
  echo "$LEAKS"
  echo
  echo "!! Residual matches above. Extend PATTERNS in this script and re-run."
  exit 2
fi

echo "==> Clean. Anonymous copy at $DEST"
echo "    Verify by hand before uploading:  grep -rIn 'your-name' $DEST"
