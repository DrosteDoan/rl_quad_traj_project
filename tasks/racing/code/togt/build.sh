#!/usr/bin/env bash
# =============================================================================
#  Build the TOGT-Planner driver for racing Lesson 6. Run INSIDE the container:
#
#      bash tasks/racing/code/togt/build.sh
#
#  Nothing is installed system-wide and no sudo is needed:
#    1. github.com/FSC-Lab/TOGT-Planner is cloned into repos/TOGT-Planner at the
#       commit pinned in scripts/pins.sh (TOGT_REF) -- pin_repo moves an older
#       clone onto the pin, like the other repos.
#    2. TOGT needs cmake >= 3.25. Ubuntu 22.04's apt cmake is 3.22, so if the
#       system cmake is missing or too old a cmake wheel goes into the main venv.
#    3. Eigen 3.4.0 (header-only): the system copy if there is one, otherwise a
#       tarball into repos/eigen-3.4.0 whose CMake config is generated in
#       repos/eigen-build.
#    4. togt_race.cpp is configured + built against TOGT's library. TOGT's own
#       CMake fetches RapidJSON at configure time, so this step needs network.
#
#  Result:  repos/togt-build/togt_race        (repos/ is git-ignored)
#  Re-run after editing togt_race.cpp: the build is incremental.
#
#  Overrides (for a checkout that is not mounted at /workspace):
#      WS=<repo root>  MAIN_VENV=<venv>  CMAKE=<cmake binary>  EIGEN3_DIR=<dir with Eigen3Config.cmake>
# =============================================================================
set -euo pipefail
WS="${WS:-/workspace}"
MAIN_VENV="${MAIN_VENV:-/opt/venvs/main}"
HERE="$(cd "$(dirname "$0")" && pwd)"
REPOS="$WS/repos"
TOGT_DIR="${TOGT_DIR:-$REPOS/TOGT-Planner}"
BUILD_DIR="${BUILD_DIR:-$REPOS/togt-build}"

# shellcheck source=/dev/null
source "$WS/scripts/pins.sh"
mkdir -p "$REPOS"
pin_repo "$TOGT_DIR" https://github.com/FSC-Lab/TOGT-Planner.git "$TOGT_REF"

# --- cmake >= 3.25 -----------------------------------------------------------
version_ge() { [ "$(printf '%s\n%s\n' "$2" "$1" | sort -V | head -n1)" = "$2" ]; }
CMAKE="${CMAKE:-}"
if [ -z "$CMAKE" ]; then
  if command -v cmake >/dev/null 2>&1 \
     && version_ge "$(cmake --version | head -1 | awk '{print $3}')" 3.25; then
    CMAKE=cmake
  else
    echo "==> system cmake is missing or older than 3.25: installing a cmake wheel into $MAIN_VENV"
    "$MAIN_VENV/bin/pip" install --no-cache-dir -q "cmake>=3.25,<4"
    CMAKE="$MAIN_VENV/bin/cmake"
  fi
fi
echo "==> cmake: $CMAKE ($("$CMAKE" --version | head -1))"

# --- Eigen -------------------------------------------------------------------
extra=()
if [ -n "${EIGEN3_DIR:-}" ]; then
  extra+=(-DEigen3_DIR="$EIGEN3_DIR")
elif [ -d /usr/include/eigen3 ] || [ -d /usr/local/include/eigen3 ]; then
  echo "==> Eigen: system headers"
else
  EIGEN_SRC="$REPOS/eigen-3.4.0"
  EIGEN_BUILD="$REPOS/eigen-build"
  if [ ! -f "$EIGEN_BUILD/Eigen3Config.cmake" ]; then
    if [ ! -d "$EIGEN_SRC" ]; then
      echo "==> Eigen: downloading 3.4.0 (header-only) into $EIGEN_SRC"
      curl -sL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz | tar -xz -C "$REPOS"
    fi
    "$CMAKE" -S "$EIGEN_SRC" -B "$EIGEN_BUILD" -DBUILD_TESTING=OFF -DEIGEN_BUILD_DOC=OFF >/dev/null
  fi
  echo "==> Eigen: $EIGEN_BUILD"
  extra+=(-DEigen3_DIR="$EIGEN_BUILD")
fi

# --- configure + build ---------------------------------------------------------
# CMAKE_FIND_USE_PACKAGE_REGISTRY=OFF: RapidJSON's configure export()s itself into
# the user package registry, after which every NEW build tree would find that
# stale config and fail (measured upstream).
"$CMAKE" -S "$HERE" -B "$BUILD_DIR" -DTOGT_DIR="$TOGT_DIR" -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_FIND_USE_PACKAGE_REGISTRY=OFF ${extra[@]+"${extra[@]}"}
"$CMAKE" --build "$BUILD_DIR" --target togt_race -j"$(nproc)"

echo
echo "built: $BUILD_DIR/togt_race"
echo "next:  python tasks/racing/code/togt_plan.py --track raw      (Lesson 6 section 2)"
