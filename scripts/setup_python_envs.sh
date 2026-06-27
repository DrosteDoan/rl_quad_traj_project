#!/usr/bin/env bash
# =============================================================================
#  Install the Python repos into the right venv. Run this INSIDE the container
#  (heavy wheels are already baked in, so this is fast).
#
#      docker compose exec dev bash scripts/setup_python_envs.sh
#
#  Three isolated environments (their dependency sets conflict on numpy):
#    main      (3.11) crazyflow, gym-pybullet-drones, lsy_drone_racing,
#                     RAPTOR_in_RotorPy            -> numpy >= 2
#    crazysim  (3.11) CrazySim cflib + cfclient    -> numpy < 1.25
#    datt      (3.10) DATT                         -> numpy 1.23 (via PYTHONPATH)
#
#  "Editable" (pip install -e) means the installed package points AT the source
#  in ./repos — edit a .py file and the change is live immediately, no reinstall.
# =============================================================================
set -uo pipefail   # not -e: a single repo problem shouldn't abort the rest

REPOS="/workspace/repos"
MAIN=/opt/venvs/main/bin/pip
CRAZYSIM=/opt/venvs/crazysim/bin/pip

einstall() { # einstall <pip> <path[extra]>
  local pip="$1" spec="$2"; shift 2
  local path="${spec%%\[*}"   # strip any [extra] to test the directory
  if [ -d "$path" ]; then
    echo "==> [$(basename "$(dirname "$(dirname "$pip")")")] pip install -e $spec"
    "$pip" install -e "$spec" "$@" || echo "WARN: install of $spec failed"
  else
    echo "==> [skip] $path not found (did you run clone_repos.sh?)"
  fi
}

echo "############ MAIN env (Python 3.12, numpy>=2) ############"
# Keep the JAX stack pinned so editable installs don't upgrade jaxlib past the
# CUDA plugin (see requirements/constraints.txt).
CONSTRAINTS="-c /tmp/requirements/constraints.txt"
[ -f /tmp/requirements/constraints.txt ] || CONSTRAINTS=""
einstall "$MAIN" "$REPOS/crazyflow" $CONSTRAINTS
einstall "$MAIN" "$REPOS/gym-pybullet-drones" $CONSTRAINTS
einstall "$MAIN" "$REPOS/lsy_drone_racing[sim]" $CONSTRAINTS   # [sim] pulls crazyflow, jax, warp-lang
einstall "$MAIN" "$REPOS/RAPTOR_in_RotorPy" $CONSTRAINTS

echo "############ CRAZYSIM env (Python 3.11, numpy<1.25) ############"
# CrazySim sub-libraries have no .git in the submodule, so setuptools-scm needs
# a pretend version (same trick as the upstream CrazySim Dockerfile).
export SETUPTOOLS_SCM_PRETEND_VERSION=1.0.0
einstall "$CRAZYSIM" "$REPOS/CrazySim/crazyflie-lib-python"
einstall "$CRAZYSIM" "$REPOS/CrazySim/crazyflie-clients-python"

echo "############ DATT env (Python 3.10) ############"
# DATT is NOT a pip package (no setup.py). Per its README it is used by putting
# the repo's PARENT folder on PYTHONPATH, then `import DATT.<module>`. We add a
# .pth file to the datt venv so `import DATT...` works from anywhere.
DATT_SITE=$(/opt/venvs/datt/bin/python -c "import site; print(site.getsitepackages()[0])" 2>/dev/null)
if [ -d "$REPOS/DATT" ] && [ -n "$DATT_SITE" ]; then
  echo "$REPOS" > "$DATT_SITE/datt_repos.pth"
  echo "==> [datt] added $REPOS to PYTHONPATH (import DATT.* now works)"
else
  echo "==> [skip] DATT not found or datt venv missing"
fi

echo
echo "Done. Quick check:"
echo "  /opt/venvs/main/bin/python    -c 'import crazyflow, gym_pybullet_drones; print(\"main OK\")'"
echo "  /opt/venvs/crazysim/bin/python -c 'import cflib; print(\"crazysim OK\")'"
echo "  /opt/venvs/datt/bin/python    -c 'import DATT, torch; print(\"datt OK\")'"
