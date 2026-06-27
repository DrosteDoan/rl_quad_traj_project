#!/usr/bin/env bash
# =============================================================================
#  One-time setup for the HOVERING task, run INSIDE the toolbox container.
#  Installs the Crazyflow simulator into the shared `main` Python environment.
#
#      docker compose exec dev bash tasks/hovering/setup.sh
#
#  (If you also want the other tasks, run scripts/clone_repos.sh +
#   scripts/setup_python_envs.sh instead — this is the quick, hovering-only path.)
# =============================================================================
set -euo pipefail

PIP=/opt/venvs/main/bin/pip
PY=/opt/venvs/main/bin/python
REPOS="/workspace/repos"
CRAZYFLOW="$REPOS/crazyflow"

# Keep the JAX stack pinned (so this install can't upgrade jaxlib past the CUDA
# plugin). The toolbox image ships this constraints file.
CONSTRAINTS="-c /tmp/requirements/constraints.txt"
[ -f /tmp/requirements/constraints.txt ] || CONSTRAINTS=""

mkdir -p "$REPOS"
if [ ! -d "$CRAZYFLOW" ]; then
  echo "==> Cloning learnsyslab/crazyflow ..."
  git clone --depth 1 https://github.com/learnsyslab/crazyflow.git "$CRAZYFLOW"
fi

echo "==> Installing crazyflow into the main venv (editable) ..."
$PIP install --no-cache-dir $CONSTRAINTS -e "$CRAZYFLOW"

echo "==> Verifying ..."
SCIPY_ARRAY_API=1 $PY -c "import crazyflow; from crazyflow.dynamics import Dynamics; print('crazyflow OK:', crazyflow.__file__)"

echo
echo "Done. Now try:"
echo "  cd /workspace/tasks/hovering"
echo "  python -m k12_hover.check"
