#!/usr/bin/env bash
# =============================================================================
#  Clone every source repository into ./repos  (run this ON THE HOST, once).
#
#  We build everything FROM SOURCE so you can edit the code and see the effect.
#  Re-running this script is safe: existing repos are left alone (it skips them).
#
#  Usage:   bash scripts/clone_repos.sh
# =============================================================================
# No `set -e`: this script is idempotent and re-runnable. If one repo or an
# optional submodule fails (e.g. a flaky network), we warn and keep going so the
# other repos still get cloned. Just run the script again to fill in any gaps.
set -uo pipefail

# Resolve project root (the folder that contains this script's parent).
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPOS="$ROOT/repos"
# Pinned commits for the simulator repos + the pin_repo helper (docs/5-versions.md).
source "$ROOT/scripts/pins.sh"
mkdir -p "$REPOS"
cd "$REPOS"

# clone <url> <dir> [extra git-clone args...]
clone() {
  local url="$1" dir="$2"; shift 2
  if [ -d "$dir/.git" ]; then
    echo "==> [skip] $dir already cloned"
  else
    echo "==> cloning $dir"
    git clone "$@" "$url" "$dir"
  fi
}

# 1) crazyflow — MAIN simulator (MuJoCo / JAX). PINNED: the numbers in the
#    lessons and the vendored crazy_track code were validated at this commit
#    (docs/5-versions.md).
pin_repo "$REPOS/crazyflow" https://github.com/learnsyslab/crazyflow.git "$CRAZYFLOW_REF"

# 2) gym-pybullet-drones — RL example
clone https://github.com/learnsyslab/gym-pybullet-drones.git gym-pybullet-drones

# 2b) lsy_drone_racing — drone-racing task built on crazyflow (main env, and
#     the `race` env for the tasks/racing lessons). PINNED: upstream changed the
#     track to five gate passes on 2026-09-03; the course races the four-gate lap.
pin_repo "$REPOS/lsy_drone_racing" https://github.com/learnsyslab/lsy_drone_racing.git "$LSY_REF"

# NOTE on crazy_track (tasks/racing): its upstream repository is PRIVATE, so it
# is NOT cloned here. The pinned source is vendored inside this repo at
# tasks/racing/crazy_track/ (see the VENDORED.md there) and installed from that
# path by scripts/setup_python_envs.sh.

# 3) learning-to-fly — C++ RLtools example.
#    Only the rl_tools submodule tree is needed to build the RL trainer. The
#    'controller' submodule is real-drone firmware (a large recursive tree:
#    crazyflie firmware, CMSIS, FreeRTOS, ...) and is OPTIONAL, so we don't pull
#    it by default. Set CLONE_LTF_CONTROLLER=1 if you really need the firmware.
clone https://github.com/arplaboratory/learning-to-fly.git learning-to-fly
if [ -d learning-to-fly/.git ]; then
  echo "==> learning-to-fly: init rl_tools submodule (skipping optional firmware controller)"
  # Non-recursive on purpose: rl_tools' own deep submodules (embedded_platforms,
  # firmware, CMSIS, ...) are huge and not needed for the desktop trainer. The
  # specific header-only externals are initialised by scripts/build_cpp.sh.
  git -C learning-to-fly submodule update --init -- external/rl_tools \
    || echo "WARN: rl_tools submodule init had problems — re-run the script"
  if [ "${CLONE_LTF_CONTROLLER:-0}" = "1" ]; then
    echo "==> learning-to-fly: also cloning optional firmware controller (large)"
    git -C learning-to-fly submodule update --init --recursive -- controller \
      || echo "WARN: controller (firmware) submodule failed — it is optional"
  fi
fi

# 4) raptor — rl-tools example.
#    Only the 'rl-tools' submodule is needed to build. raptor's 'data' and
#    'media' submodules point at upstream repos that currently 404, so we skip
#    them (they are example assets, not required for the build).
clone https://github.com/rl-tools/raptor.git raptor
if [ -d raptor/.git ]; then
  echo "==> raptor: init rl-tools submodule (non-recursive; skipping broken data/media)"
  # NOTE: raptor's 'data' and 'media' submodules 404 upstream, and its headline
  # build targets (foundation_policy_*) expect 'data'. So raptor is included as a
  # code reference; a full from-source training build is not currently
  # reproducible from upstream. rl-tools headers are enough to read/experiment.
  git -C raptor submodule update --init -- rl-tools \
    || echo "WARN: raptor rl-tools submodule init had problems — re-run the script"
fi

# 5) DATT — example (legacy stack, datt venv)
clone https://github.com/KevinHuang8/DATT.git DATT

# 6) RAPTOR_in_RotorPy — example
clone https://github.com/Sheng-Cheng/RAPTOR_in_RotorPy.git RAPTOR_in_RotorPy

# 7) CrazySim — SITL simulator. The firmware/lib/clients come from the
#    maintainer's `sitl-release` branches (see CrazySim README).
clone https://github.com/gtfactslab/CrazySim.git CrazySim --recursive
if [ ! -d "CrazySim/crazyflie-firmware/.git" ]; then
  echo "==> cloning CrazySim SITL firmware (sitl-release)"
  rm -rf CrazySim/crazyflie-firmware
  git clone --recurse-submodules -b sitl-release \
      https://github.com/llanesc/crazyflie-firmware CrazySim/crazyflie-firmware
fi
clone https://github.com/llanesc/crazyflie-lib-python.git     CrazySim/crazyflie-lib-python    -b sitl-release
clone https://github.com/llanesc/crazyflie-clients-python.git CrazySim/crazyflie-clients-python -b sitl-release

echo
echo "All repositories are in: $REPOS"
echo "Next: start the container, then run scripts/setup_python_envs.sh"
