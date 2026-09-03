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

echo; echo "===== Pinned source repos (scripts/pins.sh) ====="
for r in crazyflow lsy_drone_racing; do
  d=/workspace/repos/$r
  if [ -d "$d/.git" ]; then
    echo "  $r @ $(git -C "$d" log -1 --format='%h %ad %s' --date=short)"
  else
    echo "  $r  (not cloned yet -- run scripts/clone_repos.sh or a tasks/*/setup.sh)"
  fi
done

echo; echo "===== MAIN env (Python 3.12) ====="
/opt/venvs/main/bin/python - <<'PY'
import importlib
for m in ["numpy", "scipy", "jax", "mujoco", "mujoco.mjx", "gymnasium", "crazyflow",
          "gym_pybullet_drones", "pybullet", "torch",
          "stable_baselines3", "sb3_contrib", "crazy_track"]:
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

echo; echo "===== RACING task (main env: training side) ====="
/opt/venvs/main/bin/python - <<'PY'
# The DATT tracking env is what Lesson 2 of tasks/racing trains. 56 numbers =
# 13 state + 3 unit-mass force + 10 x 4 look-ahead window (v5 observation).
try:
    from crazy_track.envs.datt_env import DATTTrackingEnv
    env = DATTTrackingEnv(num_envs=2, seed=0, v5=True)
    obs, _ = env.reset()
    ok = obs.shape == (2, 56)
    print(f"  {'ok  ' if ok else 'FAIL'} DATTTrackingEnv obs {obs.shape} (expect (2, 56))")
    from crazy_track.controllers.utils import MASS, THRUST_MAX
    print(f"  info drone weight {MASS * 9.81:.4f} N, max thrust {THRUST_MAX:.2f} N "
          f"(TWR {THRUST_MAX / (MASS * 9.81):.2f}) -- Lesson 1 section 3")
except Exception as e:
    print(f"  FAIL DATTTrackingEnv {type(e).__name__}: {e}")
PY

echo; echo "===== RACE env (Python 3.12, tasks/racing lesson 3) ====="
if [ -x /opt/venvs/race/bin/python ]; then
/opt/venvs/race/bin/python - <<'PY'
import importlib
for m in ["lsy_drone_racing", "crazyflow", "mujoco", "gymnasium", "jax", "torch",
          "stable_baselines3", "sb3_contrib"]:
    try:
        mod = importlib.import_module(m)
        print(f"  ok   {m:22s} {getattr(mod,'__version__','')}")
    except Exception as e:
        print(f"  FAIL {m:22s} {type(e).__name__}: {e}")
# Both venvs must run the SAME simulator version (docs/5-versions.md).
try:
    import subprocess, crazyflow
    main_v = subprocess.run(["/opt/venvs/main/bin/python", "-c", "import crazyflow; print(crazyflow.__version__)"],
                            capture_output=True, text=True).stdout.strip()
    same = main_v == crazyflow.__version__
    print(f"  {'ok  ' if same else 'FAIL'} crazyflow main={main_v} race={crazyflow.__version__} "
          f"({'same physics in both venvs' if same else 'DIFFERENT -- re-run tasks/racing/setup.sh'})")
except Exception as e:
    print(f"  FAIL crazyflow version compare {type(e).__name__}: {e}")
# The bridge imports crazy_track's policy code inside lsy's env — the
# load-bearing check for the two-venv split (crazy_track installed --no-deps
# must NOT have disturbed lsy's own crazyflow).
try:
    import crazy_track.controllers.datt  # noqa
    print("  ok   crazy_track.controllers.datt (policy importable in race env)")
except Exception as e:
    print(f"  FAIL crazy_track.controllers.datt {type(e).__name__}: {e}")
PY
else
  echo "  (race venv not present — rebuild the image to get /opt/venvs/race)"
fi

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
