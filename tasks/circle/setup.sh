#!/usr/bin/env bash
# =============================================================================
#  One-time setup for the CIRCLE (trajectory-tracking) task, run INSIDE the
#  toolbox container. Installs the Crazyflow simulator into the shared `main`
#  Python environment.
#
#      docker compose exec dev bash tasks/circle/setup.sh
#
#  (If you also want the other tasks, run scripts/clone_repos.sh +
#   scripts/setup_python_envs.sh instead — this is the quick, circle-only path.)
#
#  Note: the extra libraries this task uses beyond hovering — casadi (for the MPC
#  baseline), matplotlib and scipy (for the benchmark plots) — are ALREADY in the
#  toolbox `main` venv (see rl_quad_control/requirements/main.txt), so there is
#  nothing else to install.
# =============================================================================
set -euo pipefail

PIP=/opt/venvs/main/bin/pip
PY=/opt/venvs/main/bin/python
REPOS="/workspace/repos"
CRAZYFLOW="$REPOS/crazyflow"

# Constraints: jax, mujoco, mujoco-mjx, gymnasium, crazyflow, SB3 (docs/5-versions.md).
# Prefer the file in the MOUNTED REPO, so a `git pull` updates the pins without
# rebuilding the image; fall back to the copy baked into the image.
CONSTRAINTS_FILE=/workspace/requirements/constraints.txt
[ -f "$CONSTRAINTS_FILE" ] || CONSTRAINTS_FILE=/tmp/requirements/constraints.txt
CONSTRAINTS="-c $CONSTRAINTS_FILE"
[ -f "$CONSTRAINTS_FILE" ] || CONSTRAINTS=""

# Pinned commit + pin_repo helper (docs/5-versions.md). Also moves an older
# clone that was made at HEAD onto the pin.
source /workspace/scripts/pins.sh
mkdir -p "$REPOS"
pin_repo "$CRAZYFLOW" https://github.com/learnsyslab/crazyflow.git "$CRAZYFLOW_REF"

echo "==> Installing crazyflow into the main venv (editable) ..."
$PIP install --no-cache-dir $CONSTRAINTS -e "$CRAZYFLOW"

echo "==> Verifying ..."
SCIPY_ARRAY_API=1 $PY -c "import crazyflow, casadi, matplotlib; from crazyflow.dynamics import Dynamics; print('crazyflow OK:', crazyflow.__file__)"

echo
echo "Done. Now try:"
echo "  cd /workspace/tasks/circle"
echo "  python -m k12_hover.check"
