#!/usr/bin/env bash
# =============================================================================
#  Install the Python repos into the right venv. Run this INSIDE the container
#  (heavy wheels are already baked in, so this is fast).
#
#      docker compose exec dev bash scripts/setup_python_envs.sh
#
#  Four isolated environments (their dependency sets conflict):
#    main      (3.12) crazyflow, gym-pybullet-drones, lsy_drone_racing,
#                     RAPTOR_in_RotorPy, crazy_track -> numpy >= 2
#    race      (3.12) lsy_drone_racing + ITS OWN crazyflow (tasks/racing)
#    crazysim  (3.11) CrazySim cflib + cfclient    -> numpy < 1.25
#    datt      (3.10) DATT                         -> numpy 1.23 (via PYTHONPATH)
#
#  Why `race` exists: crazy_track trains against the crazyflow mounted in
#  ./repos (editable, in `main`), while lsy_drone_racing pins its own crazyflow
#  from PyPI. Installing one on top of the other can silently downgrade the
#  simulator underneath a policy you already trained — a bug that presents as
#  "my policy mysteriously got worse". The Lesson-3 race bridge needs BOTH, so
#  crazy_track also goes into `race`, but with --no-deps: its policy and
#  trajectory code is pure Python (numpy/scipy/torch) and never imports
#  crazyflow at module level, so the race env keeps its own simulator untouched.
#
#  "Editable" (pip install -e) means the installed package points AT the source
#  in ./repos — edit a .py file and the change is live immediately, no reinstall.
# =============================================================================
set -uo pipefail   # not -e: a single repo problem shouldn't abort the rest

REPOS="/workspace/repos"
MAIN=/opt/venvs/main/bin/pip
CRAZYSIM=/opt/venvs/crazysim/bin/pip
# Pinned commits + pin_repo helper. Re-running this script moves an older,
# unpinned clone (made at HEAD) onto the pins -- that is the fix for the
# "mismatching versions" problem described in docs/5-versions.md.
source /workspace/scripts/pins.sh

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
# CUDA plugin (see requirements/constraints.txt) — and pin torch to the build
# the image baked (cu126 on the GPU image, +cpu on the CPU one). Without the
# torch pin, an editable install can drag in the current default PyPI torch,
# which now ships CUDA-13: too new for the drivers this image targets, and its
# nvidia-*-cu13 libraries shadow JAX's cu12 cudnn, breaking JAX on GPU with
# "Could not create cudnn handle" (measured, 2026-08-20).
RUNTIME_CONSTRAINTS=/tmp/runtime-constraints.txt
# Prefer the constraints in the MOUNTED REPO (a `git pull` updates the pins
# without rebuilding the image); fall back to the copy baked into the image.
CONSTRAINTS_FILE=/workspace/requirements/constraints.txt
[ -f "$CONSTRAINTS_FILE" ] || CONSTRAINTS_FILE=/tmp/requirements/constraints.txt
cat "$CONSTRAINTS_FILE" > "$RUNTIME_CONSTRAINTS" 2>/dev/null || : > "$RUNTIME_CONSTRAINTS"
/opt/venvs/main/bin/python -c "import torch; print(f'torch=={torch.__version__}')" >> "$RUNTIME_CONSTRAINTS" \
  || echo "WARN: could not read baked torch version (torch unpinned)"
CONSTRAINTS="-c $RUNTIME_CONSTRAINTS"
# Move an older image's simulator stack onto the pinned mujoco/mujoco-mjx pair
# BEFORE anything else (see tasks/racing/setup.sh for the story).
if [ -s "$RUNTIME_CONSTRAINTS" ]; then
  echo "==> [main] simulator stack at the pinned versions (mujoco, mujoco-mjx, gymnasium)"
  "$MAIN" install $CONSTRAINTS mujoco mujoco-mjx gymnasium || echo "WARN: could not re-pin mujoco/mujoco-mjx"
fi
[ -d "$REPOS/crazyflow" ] && pin_repo "$REPOS/crazyflow" https://github.com/learnsyslab/crazyflow.git "$CRAZYFLOW_REF"
[ -d "$REPOS/lsy_drone_racing" ] && pin_repo "$REPOS/lsy_drone_racing" https://github.com/learnsyslab/lsy_drone_racing.git "$LSY_REF"
einstall "$MAIN" "$REPOS/crazyflow" $CONSTRAINTS
# gym-pybullet-drones declares torch ^2.13 (conflicts with the pin above), but
# every runtime dependency it needs is already baked into the image via
# requirements/main.txt — so install just the package itself.
einstall "$MAIN" "$REPOS/gym-pybullet-drones" $CONSTRAINTS --no-deps
einstall "$MAIN" "$REPOS/lsy_drone_racing" $CONSTRAINTS   # its deps (crazyflow, jax, mujoco, warp-lang) are pinned by the constraints
einstall "$MAIN" "$REPOS/RAPTOR_in_RotorPy" $CONSTRAINTS
# crazy_track (tasks/racing, training side). VENDORED inside this repo (its
# upstream is private — see tasks/racing/crazy_track/VENDORED.md). Base deps
# only — numpy/scipy/matplotlib/pyyaml, torch, SB3, sb3-contrib and tensorboard
# are already baked into the image, and crazyflow is the editable install
# above. Do NOT use its [sim] extra here: that would pull the PyPI crazyflow
# over the editable one.
CRAZY_TRACK=/workspace/tasks/racing/crazy_track
einstall "$MAIN" "$CRAZY_TRACK" $CONSTRAINTS

echo "############ RACE env (Python 3.12, tasks/racing) ############"
RACE=/opt/venvs/race/bin/pip
if [ -x "$RACE" ]; then
  # CPU torch FIRST, so lsy's editable install below finds torch already
  # satisfied instead of downloading the multi-GB default CUDA build. The race
  # env only runs policy inference at 50 Hz — CPU is plenty (training happens
  # in `main`, where torch has CUDA).
  # The SAME constraints file as main (jax, mujoco, mujoco-mjx, gymnasium,
  # crazyflow==0.3.2, SB3): the race venv must simulate the same physics that the
  # policy was trained on, and load the same SB3 model format.
  RACE_CONSTRAINTS="-c $CONSTRAINTS_FILE"
  [ -f "$CONSTRAINTS_FILE" ] || RACE_CONSTRAINTS=""
  "$RACE" install --extra-index-url https://download.pytorch.org/whl/cpu $RACE_CONSTRAINTS \
      "torch==2.8.0+cpu" "scipy>=1.11" "matplotlib>=3.8" \
      "stable-baselines3>=2.3" "sb3-contrib>=2.9" \
      jax mujoco mujoco-mjx gymnasium \
      || echo "WARN: race env policy stack install failed"
  # lsy_drone_racing, editable, WITH deps — brings crazyflow from PyPI (pinned 0.3.2).
  einstall "$RACE" "$REPOS/lsy_drone_racing" $RACE_CONSTRAINTS
  # crazy_track policy/trajectory code, WITHOUT deps (see header note).
  einstall "$RACE" "$CRAZY_TRACK" --no-deps
else
  echo "==> [skip] race venv not found (old image? rebuild to get /opt/venvs/race)"
fi

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
