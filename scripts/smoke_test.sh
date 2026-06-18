#!/usr/bin/env bash
# =============================================================================
#  Quick "is everything alive?" check. Run INSIDE the container.
#      docker compose exec dev bash scripts/smoke_test.sh
# =============================================================================
set -uo pipefail

echo "===== GPU ====="
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi -L || echo "nvidia-smi present but no GPU visible"
else
  echo "No nvidia-smi -> running in CPU mode (normal on macOS)."
fi

echo; echo "===== MAIN env (Python 3.11) ====="
/opt/venvs/main/bin/python - <<'PY'
import importlib
for m in ["numpy", "jax", "mujoco", "gymnasium", "crazyflow",
          "gym_pybullet_drones", "pybullet", "torch"]:
    try:
        mod = importlib.import_module(m)
        print(f"  ok   {m:22s} {getattr(mod,'__version__','')}")
    except Exception as e:
        print(f"  FAIL {m:22s} {type(e).__name__}: {e}")
try:
    import jax
    print("  jax devices:", jax.devices())
except Exception as e:
    print("  jax devices: (error)", e)
PY

echo; echo "===== CRAZYSIM env (Python 3.11) ====="
/opt/venvs/crazysim/bin/python - <<'PY'
import importlib
for m in ["numpy", "cflib"]:
    try:
        mod = importlib.import_module(m)
        print(f"  ok   {m:18s} {getattr(mod,'__version__','')}")
    except Exception as e:
        print(f"  FAIL {m:18s} {type(e).__name__}: {e}")
PY

echo; echo "===== DATT env (Python 3.10) ====="
/opt/venvs/datt/bin/python - <<'PY'
import importlib
for m in ["numpy", "torch", "gym", "stable_baselines3", "DATT"]:
    try:
        mod = importlib.import_module(m)
        print(f"  ok   {m:18s} {getattr(mod,'__version__','')}")
    except Exception as e:
        print(f"  FAIL {m:18s} {type(e).__name__}: {e}")
try:
    import torch
    print("  torch CUDA available:", torch.cuda.is_available())
except Exception as e:
    print("  torch CUDA: (error)", e)
PY
