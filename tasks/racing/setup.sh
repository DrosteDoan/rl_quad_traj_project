#!/usr/bin/env bash
# =============================================================================
#  One-time setup for the RACING task, run INSIDE the toolbox container.
#
#      docker compose exec dev bash tasks/racing/setup.sh
#
#  Installs:
#    main env  <- vendored crazy_track (training side) + editable crazyflow
#    race env  <- lsy_drone_racing + PyPI crazyflow (same pinned version) + CPU policy stack
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

# Constraints: jax, mujoco, mujoco-mjx, gymnasium, crazyflow, SB3 (docs/5-versions.md).
# Prefer the file in the MOUNTED REPO, so a `git pull` updates the pins without
# rebuilding the image; fall back to the copy baked into the image.
CONSTRAINTS_FILE=/workspace/requirements/constraints.txt
[ -f "$CONSTRAINTS_FILE" ] || CONSTRAINTS_FILE=/tmp/requirements/constraints.txt
CONSTRAINTS="-c $CONSTRAINTS_FILE"
[ -f "$CONSTRAINTS_FILE" ] || CONSTRAINTS=""

# Pinned commits + pin_repo helper (docs/5-versions.md). Re-running this script
# moves clones that were made at HEAD onto the pins.
source /workspace/scripts/pins.sh

# --- crazyflow: the simulator crazy_track trains against (same one the
# --- hovering/circle tasks use). Clone if this container never had it.
mkdir -p "$REPOS"
pin_repo "$REPOS/crazyflow" https://github.com/learnsyslab/crazyflow.git "$CRAZYFLOW_REF"
# Move the simulator stack of an OLDER image onto the pinned pair first: an
# unpinned image baked mujoco 3.12 + mujoco-mjx 3.12, and installing the race repo
# (mujoco<3.11) on top of that left mjx 3.12 next to mujoco 3.10 -- the
# "mismatching versions" students hit. Requesting both explicitly, under the
# constraints, makes pip move them TOGETHER.
if [ -n "$CONSTRAINTS" ]; then
  echo "==> [main] simulator stack at the pinned versions (mujoco, mujoco-mjx, gymnasium)"
  $MAIN install --no-cache-dir $CONSTRAINTS mujoco mujoco-mjx gymnasium
fi
echo "==> [main] crazyflow (editable)"
$MAIN install --no-cache-dir $CONSTRAINTS -e "$REPOS/crazyflow"

echo "==> [main] crazy_track (vendored, editable)"
$MAIN install --no-cache-dir $CONSTRAINTS -e "$CRAZY_TRACK"

# --- lsy_drone_racing: public, cloned (NOT vendored) at the pinned commit.
pin_repo "$LSY" https://github.com/learnsyslab/lsy_drone_racing.git "$LSY_REF"

# The race venv uses the SAME constraints as main (jax, mujoco, mujoco-mjx,
# gymnasium, crazyflow==0.3.2, SB3): same physics as training, same model format.
echo "==> [race] CPU policy stack (torch/SB3 — inference only, training is in main)"
$RACE install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu $CONSTRAINTS \
    "torch==2.8.0+cpu" "scipy>=1.11" "matplotlib>=3.8" "stable-baselines3>=2.3" "sb3-contrib>=2.9" \
    jax mujoco mujoco-mjx gymnasium     # explicit, so the constraints pin them here too

echo "==> [race] lsy_drone_racing (editable; pulls crazyflow 0.3.2 from PyPI)"
$RACE install --no-cache-dir $CONSTRAINTS -e "$LSY"

echo "==> [race] crazy_track policy/trajectory code (--no-deps, on purpose)"
$RACE install --no-cache-dir --no-deps -e "$CRAZY_TRACK"

echo
echo "==> Verifying ..."
/opt/venvs/main/bin/python -c "import crazy_track; from crazy_track.envs.datt_env import DATTTrackingEnv; print('main:  crazy_track OK')"
/opt/venvs/race/bin/python -c "import lsy_drone_racing, crazyflow, mujoco, crazy_track.controllers.datt; print('race:  lsy + bridge OK  (crazyflow', crazyflow.__version__, '| mujoco', mujoco.__version__ + ')')"
/opt/venvs/main/bin/python -c "import crazyflow, mujoco; print('main:  crazyflow', crazyflow.__version__, '| mujoco', mujoco.__version__)"

echo
echo "Done. Now start at tasks/racing/lessons/01-the-tracking-problem.md"
echo "(or run the full check: bash scripts/smoke_test.sh)"
