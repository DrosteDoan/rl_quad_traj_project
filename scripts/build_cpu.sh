#!/usr/bin/env bash
# Build the CPU-only variant of the toolbox image (no CUDA; smaller; runs anywhere).
# Tag: rl-quad-traj:cpu
#
#   bash scripts/build_cpu.sh
#
# Then use it by pointing compose at this image, e.g.:
#   IMAGE=rl-quad-traj:cpu docker compose up -d     # (if you templatise the image)
# or run it directly:
#   docker run --rm -it -v "$PWD":/workspace -w /workspace rl-quad-traj:cpu bash
set -euo pipefail
cd "$(dirname "$0")/.."

docker build \
  --build-arg BASE_IMAGE=ubuntu:22.04 \
  --build-arg MAIN_REQS=main-cpu.txt \
  --build-arg DATT_REQS=datt-cpu.txt \
  --build-arg MAIN_CONSTRAINTS=constraints-cpu.txt \
  --build-arg UID="$(id -u)" --build-arg GID="$(id -g)" \
  -t rl-quad-traj:cpu .

echo "Built rl-quad-traj:cpu"
