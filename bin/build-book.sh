#!/bin/bash
# Build FoxML documentation as a browsable HTML book using MkDocs Material.
#
# Usage:
#   bin/build-book.sh            # Build HTML site
#   bin/build-book.sh serve      # Build + launch dev server with live reload
#
# Output: book/site/index.html
#
# Dependencies (installed automatically):
#   pip: mkdocs-material
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# ── Ensure dependencies ──────────────────────────────────────────
# Use conda Python if available, fall back to system python
if command -v conda &>/dev/null; then
    PYTHON="$(conda info --base)/bin/python"
elif [[ -x /opt/anaconda/bin/python ]]; then
    PYTHON=/opt/anaconda/bin/python
else
    PYTHON=python
fi
PIP="$PYTHON -m pip"
MKDOCS="$PYTHON -m mkdocs"

if ! "$PYTHON" -c "import material" 2>/dev/null; then
    echo "Installing mkdocs-material..."
    $PIP install --quiet mkdocs-material
fi

# ── Create _book_src with only .md symlinks ──────────────────────
# MkDocs requires docs_dir to be a child of the config directory.
# We create _book_src/ containing symlinks to ONLY .md files,
# preserving the directory structure. This avoids issues with
# non-doc files, broken symlinks, and MkDocs trying to copy
# binary/code files as static assets.
SRC="$REPO_ROOT/_book_src"
rm -rf "$SRC"

echo "Linking markdown files into _book_src/..."

# Find all .md files and create symlinks preserving directory structure
while IFS= read -r -d '' mdfile; do
    # Get relative path (strip leading ./)
    rel="${mdfile#./}"
    target_dir="$SRC/$(dirname "$rel")"
    mkdir -p "$target_dir"
    ln -s "$REPO_ROOT/$rel" "$SRC/$rel"
done < <(find . -name '*.md' -type f \
    -not -path './.git/*' \
    -not -path './ARCHIVE/*' \
    -not -path './TRAINING/results/*' \
    -not -path './RESULTS/*' \
    -not -path './test_demo/*' \
    -not -path './test_raw/*' \
    -not -path './node_modules/*' \
    -not -path './__pycache__/*' \
    -not -path './.venv/*' \
    -not -path './catboost_info/*' \
    -not -path './DASHBOARD/dashboard/target/*' \
    -not -path './INTERNAL/docs/analysis/archive/*' \
    -not -path './site/*' \
    -not -path './book/*' \
    -not -path './_book_src/*' \
    -not -path '*/venv/*' \
    -not -path '*/dist-info/*' \
    -not -path './deploy/*' \
    -not -path './SCRIPTS/*' \
    -not -path './MCP_SERVERS/*' \
    -print0)

# Create visible symlinks for hidden .claude/ directories
if [[ -d "$REPO_ROOT/.claude/skills" ]]; then
    mkdir -p "$SRC/claude-skills"
    for f in "$REPO_ROOT/.claude/skills/"*.md; do
        [[ -f "$f" ]] && ln -s "$f" "$SRC/claude-skills/$(basename "$f")"
    done
fi
if [[ -d "$REPO_ROOT/.claude/plans" ]]; then
    mkdir -p "$SRC/claude-plans"
    for f in "$REPO_ROOT/.claude/plans/"*.md; do
        [[ -f "$f" ]] && ln -s "$f" "$SRC/claude-plans/$(basename "$f")"
    done
fi

FILE_COUNT=$(find "$SRC" -name '*.md' -type l | wc -l)
echo "Linked $FILE_COUNT markdown files."

# ── Build or serve ───────────────────────────────────────────────
if [[ "${1:-}" == "serve" ]]; then
    echo "Starting MkDocs dev server..."
    echo "Open: http://127.0.0.1:8000"
    $MKDOCS serve
else
    echo "Building HTML documentation..."
    $MKDOCS build --clean
    # Clean up build source directory
    rm -rf "$SRC"
    echo ""
    echo "HTML book built at: book/site/"
    echo "Open: file://$REPO_ROOT/book/site/index.html"
fi
