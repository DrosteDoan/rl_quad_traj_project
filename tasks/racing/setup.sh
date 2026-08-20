#!/usr/bin/env bash
# =============================================================================
#  One-time setup for the RACING task, run INSIDE the toolbox container.
#
#      docker compose exec dev bash tasks/racing/setup.sh
#
#  Installs:
#    main env  <- vendored crazy_track (training side) + editable crazyflow
#    race env  <- lsy_drone_racing (its own crazyflow) + CPU policy stack
#                 + crazy_track --no-deps (the Lesson-3 bridge needs both)
#
#  (If you also want the other tasks, run scripts/clone_repos.sh on the host +
#   scripts/setup_python_envs.sh in here instead — this is the racing-only path.)
# =============================================================================
set -euo pipefail

MAIN=/opt/venvs/main/bin/pip
RACE=/opt/venvs/race/bin/pip
REPOS="/workspace/repos"
CRAZY_TRACK="/workspace/tasks/racing/crazy_track"   # vendored (upstream is private)
LSY="$REPOS/lsy_drone_racing"

[ -x "$RACE" ] || { echo "ERROR: /opt/venvs/race missing — rebuild the image (the racing task needs it)"; exit 1; }

# Keep the JAX stack pinned (so installs can't upgrade jaxlib past the CUDA
# plugin). The toolbox image ships this constraints file.
CONSTRAINTS="-c /tmp/requirements/constraints.txt"
[ -f /tmp/requirements/constraints.txt ] || CONSTRAINTS=""

# --- crazyflow: the simulator crazy_track trains against (same one the
# --- hovering/circle tasks use). Clone if this container never had it.
mkdir -p "$REPOS"
if [ ! -d "$REPOS/crazyflow" ]; then
  echo "==> Cloning learnsyslab/crazyflow ..."
  git clone --depth 1 https://github.com/learnsyslab/crazyflow.git "$REPOS/crazyflow"
fi
echo "==> [main] crazyflow (editable)"
$MAIN install --no-cache-dir $CONSTRAINTS -e "$REPOS/crazyflow"

echo "==> [main] crazy_track (vendored, editable)"
$MAIN install --no-cache-dir $CONSTRAINTS -e "$CRAZY_TRACK"

# --- lsy_drone_racing: public, cloned (NOT vendored).
if [ ! -d "$LSY" ]; then
  echo "==> Cloning learnsyslab/lsy_drone_racing ..."
  git clone https://github.com/learnsyslab/lsy_drone_racing.git "$LSY"
fi

echo "==> [race] CPU policy stack (torch/SB3 — inference only, training is in main)"
$RACE install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu \
    "torch==2.8.0+cpu" "scipy>=1.11" "matplotlib>=3.8" "stable-baselines3>=2.3" "sb3-contrib>=2.9"

echo "==> [race] lsy_drone_racing (editable, brings its own crazyflow pin)"
$RACE install --no-cache-dir -e "$LSY"

echo "==> [race] crazy_track policy/trajectory code (--no-deps, on purpose)"
$RACE install --no-cache-dir --no-deps -e "$CRAZY_TRACK"

echo
echo "==> Verifying ..."
/opt/venvs/main/bin/python -c "import crazy_track; from crazy_track.envs.datt_env import DATTTrackingEnv; print('main:  crazy_track OK')"
/opt/venvs/race/bin/python -c "import lsy_drone_racing, crazyflow, crazy_track.controllers.datt; print('race:  lsy + bridge OK')"

echo
echo "Done. Now start at tasks/racing/lessons/01-the-tracking-problem.md"
echo "(or run the full check: bash scripts/smoke_test.sh)"
