#!/usr/bin/env bash
# Allow the Docker container to open windows on your (Linux) desktop.
# Run this ONCE per login on the HOST, before starting the container with the
# GUI override. Safe to re-run.
#
#   bash scripts/allow_gui.sh
#
# It grants local (non-network) X11 clients access. To undo: `xhost -local:`.
set -euo pipefail
if ! command -v xhost >/dev/null 2>&1; then
  echo "xhost not found — install x11-xserver-utils (Debian/Ubuntu) or your distro's equivalent." >&2
  exit 1
fi
xhost +local:
echo "OK. The container can now open GUI windows on DISPLAY=$DISPLAY."
