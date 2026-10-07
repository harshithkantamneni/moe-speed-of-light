#!/usr/bin/env bash
# Build an anonymised copy of the artifact for double-blind review: the committed tree of this repository (without the
# planning, review and application notes, and without earlier drafts of the paper), the committed tree of the gpu
# branch (job scripts with their registered predictions, and every result), and the gpu branch's commit times as a
# plain log (no author or committer names), since the archive carries no git history. Identifying strings are replaced,
# and the script fails if any remain.
#
#   scripts/anon_export.sh [OUT_DIR] [GPU_BRANCH_CHECKOUT]
#
# OUT_DIR defaults to ./anon_export; the result is OUT_DIR/moe-speed-of-light-anon.tar.gz.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${1:-$ROOT/anon_export}"
GPU="${2:-${MOSL_GPU:-/home/claude/gpu-branch}}"
NAME="moe-speed-of-light-anon"
DST="$OUT/$NAME"
# identifying strings and what replaces them (checked case-insensitively afterwards)
PATTERN='harshith|kantamneni|drogon4231|hk4231'

rm -rf "$DST" && mkdir -p "$DST/gpu-branch"

# 1. this repository at HEAD, minus notes and drafts that are not part of the artifact
git -C "$ROOT" archive --format=tar HEAD \
  -- . ':(exclude)apply' ':(exclude)reports' ':(exclude)research_notes' ':(exclude)paper/archive' \
       ':(exclude)paper/paper_v1_prereview.tex' ':(exclude)paper/paper_v2_prerewrite.tex' ':(exclude)paper/audit_paper.tex' \
  | tar -x -C "$DST"

# 2. the gpu branch at its tip: job scripts (with the registered predictions) and results
git -C "$GPU" archive --format=tar HEAD | tar -x -C "$DST/gpu-branch"

# 3. the gpu branch's history as commit hash, commit time and files changed, for checking registration times
git -C "$GPU" log --format='commit %H %cI' --name-only > "$DST/gpu-branch/REGISTRATION_LOG.txt"

# 4. replace identifying strings in every text file
grep -rlI -i -E "$PATTERN" "$DST" | while read -r f; do
  sed -i -E \
    -e 's#github\.com/harshithkantamneni/moe-speed-of-light#github.com/ANONYMOUS/moe-speed-of-light#g' \
    -e 's#harshithkantamneni@users\.noreply\.github\.com#anonymous@example.org#g' \
    -e 's#Harshith Kantamneni#Anonymous Author#g' \
    -e 's#harshithkantamneni#ANONYMOUS#g' \
    "$f"
done

# 5. fail if anything identifying is left, in text or in binary files (PDF metadata included)
if grep -rl -i -a -E "$PATTERN" "$DST"; then
  echo "identifying strings remain in the files above" >&2
  exit 1
fi

tar -C "$OUT" -czf "$OUT/$NAME.tar.gz" "$NAME"
echo "wrote $OUT/$NAME.tar.gz ($(du -sh "$OUT/$NAME.tar.gz" | cut -f1)); no identifying strings found"
