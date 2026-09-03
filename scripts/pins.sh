#!/usr/bin/env bash
# =============================================================================
#  Pinned upstream commits for the source repos the tasks depend on.
#  Sourced by scripts/clone_repos.sh, scripts/setup_python_envs.sh and every
#  tasks/*/setup.sh. Why pins exist, and how to bump them: docs/5-versions.md.
#
#  Override for an experiment:   CRAZYFLOW_REF=main bash tasks/hovering/setup.sh
#  (but the numbers and file/line citations in the lessons are only valid at
#  the pins).
# =============================================================================

# crazyflow 0.3.2 (2026-08-26). Same version the race venv gets from PyPI, so
# both venvs simulate the SAME drone physics.
CRAZYFLOW_REF="${CRAZYFLOW_REF:-dede875b685e29de5b65d3cbd29b481035f6fa30}"

# lsy_drone_racing 2026-08-29 ("Pin MuJoCo below 3.11", #134). This is the last
# commit BEFORE upstream changed the race to five gate passes
# (gate_order = [1, 2, 3, 4, 2], #135, 2026-09-03). The racing lessons, the
# vendored lsy_level2_race() reference and the 7.80 s baseline all describe the
# four-gate lap, so the course stays on this commit.
LSY_REF="${LSY_REF:-709dbc9dfa4c07ea13381959b2520f5d981dbff1}"

# pin_repo <dir> <url> <ref>
#   Clone <url> into <dir> if it is not there yet, then check out exactly <ref>.
#   Idempotent: an existing clone (even one made at HEAD by an older version of
#   these scripts) is moved to the pin. Local edits to TRACKED files block the
#   checkout on purpose -- commit or stash them first.
pin_repo() {
  local dir="$1" url="$2" ref="$3" name
  name="$(basename "$dir")"
  if [ ! -d "$dir/.git" ]; then
    echo "==> cloning $name"
    git clone --quiet "$url" "$dir" || { echo "ERROR: clone of $url failed"; return 1; }
  fi
  if [ "$(git -C "$dir" rev-parse HEAD 2>/dev/null)" != "$ref" ]; then
    # A full SHA can be fetched directly from GitHub; fall back to a full fetch.
    git -C "$dir" fetch --quiet origin "$ref" 2>/dev/null || git -C "$dir" fetch --quiet origin
    git -C "$dir" checkout --quiet "$ref" \
      || { echo "ERROR: could not check out $ref in $dir (uncommitted local edits?)"; return 1; }
  fi
  echo "==> $name @ $(git -C "$dir" log -1 --format='%h %ad %s' --date=short)"
}
