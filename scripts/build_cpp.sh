#!/usr/bin/env bash
# =============================================================================
#  Build the C++ repositories FROM SOURCE. Run INSIDE the container.
#
#      docker compose exec dev bash scripts/build_cpp.sh
#
#  Build output goes into <repo>/build inside the mounted workspace, so you can
#  edit the C++ source on the host and just re-run this script to rebuild.
# =============================================================================
# Not -e: if one repo fails to build we still want the others attempted.
set -uo pipefail
REPOS="/workspace/repos"
NPROC="$(nproc)"

# ---- learning-to-fly --------------------------------------------------------
if [ -d "$REPOS/learning-to-fly" ]; then
  echo "############ building learning-to-fly ############"
  cd "$REPOS/learning-to-fly"
  # rl_tools is header-only; init it non-recursively, then only the small
  # header-only externals the desktop trainer needs (NOT the huge
  # embedded_platforms / firmware tree).
  git submodule update --init -- external/rl_tools
  ( cd external/rl_tools && git submodule update --init -- \
      external/cli11 external/highfive external/json external/tensorboard \
      tests/lib/googletest ) || echo "WARN: rl_tools externals init had issues"
  # fetch the small UI deps if the helper exists
  [ -x src/ui/get_dependencies.sh ] && ( cd src/ui && ./get_dependencies.sh ) || true
  cmake -S . -B build -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DRL_TOOLS_BACKEND_ENABLE_MKL:BOOL=OFF \
      -DRL_TOOLS_DISABLE_CPU_SPECIFIC_OPTIMIZATIONS:BOOL=ON \
    && cmake --build build -j"$NPROC" \
    && echo "==> learning-to-fly binaries are in $REPOS/learning-to-fly/build" \
    || echo "WARN: learning-to-fly build failed — read the FIRST error above."
fi

# ---- raptor (rl-tools) ------------------------------------------------------
if [ -d "$REPOS/raptor" ]; then
  echo "############ raptor (rl-tools) ############"
  cd "$REPOS/raptor"
  git submodule update --init -- rl-tools
  ( cd rl-tools && git submodule update --init -- \
      external/cli11 external/highfive external/json external/tensorboard ) \
      || echo "WARN: rl-tools externals init had issues"
  # raptor's CMake project IS the rl-tools submodule (see raptor README), so we
  # point cmake at rl-tools, not the repo root.
  cmake -S rl-tools -B build -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DRL_TOOLS_BACKEND_ENABLE_MKL=OFF \
      -DRL_TOOLS_ENABLE_HDF5=ON -DRL_TOOLS_ENABLE_JSON=ON -DRL_TOOLS_ENABLE_TENSORBOARD=ON \
    && echo "==> raptor/rl-tools configured in $REPOS/raptor/build" \
    || echo "WARN: raptor/rl-tools configure failed — read the FIRST error above."
  echo "NOTE: raptor's headline 'foundation_policy_*' training targets need its"
  echo "      'data' submodule, which currently 404s upstream, so we don't build"
  echo "      them here. rl-tools is configured for you to build other targets, e.g.:"
  echo "        cmake --build $REPOS/raptor/build --target <target> -j$NPROC"
fi

# ---- CrazySim SITL firmware -------------------------------------------------
if [ -d "$REPOS/CrazySim/crazyflie-firmware/sitl_make" ]; then
  echo "############ building CrazySim SITL firmware ############"
  # The firmware's version.c generator calls `python` — make sure it exists
  # (the image ships python-is-python3, this is just a belt-and-suspenders).
  command -v python >/dev/null 2>&1 || sudo ln -sf /usr/bin/python3 /usr/local/bin/python
  cd "$REPOS/CrazySim/crazyflie-firmware"
  mkdir -p sitl_make/build && cd sitl_make/build
  cmake ..
  # Build the 'cf2' SITL firmware (used by the MuJoCo backend). The 'crazysim_gz'
  # target is the Gazebo plugin and needs Gazebo — only build it (via 'make all')
  # if the image was built with INSTALL_GAZEBO=true.
  if command -v gz >/dev/null 2>&1; then BUILD_TARGET="all"; else BUILD_TARGET="cf2"; fi
  echo "==> building firmware target: $BUILD_TARGET"
  make "$BUILD_TARGET" -j"$NPROC" \
    && echo "==> CrazySim firmware built: $REPOS/CrazySim/crazyflie-firmware/sitl_make/build/cf2" \
    || echo "WARN: CrazySim firmware build failed — read the FIRST error above."
fi

echo
echo "C++ builds finished. If a build failed, read its FIRST error (scroll up)."
