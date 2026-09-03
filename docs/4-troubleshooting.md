# 4. Troubleshooting

Read the **first** error line, not the last — the first one is usually the real
cause. Below are the problems you're most likely to hit.

---

### "could not select device driver 'nvidia' with capabilities: [[gpu]]"
Docker can't find a GPU.
- **On a Mac:** expected — there is no NVIDIA GPU. Use the plain
  `docker compose up -d` (no `gpu.yml`). Everything runs on CPU.
- **On Windows/Linux:** your driver or the NVIDIA Container Toolkit isn't ready.
  Re-check [1-install-docker.md](1-install-docker.md) Step 1–4, make sure Docker
  Desktop is running, and reboot. Test with:
  `docker run --rm --gpus all nvidia/cuda:12.6.3-base-ubuntu22.04 nvidia-smi`

### `jax` / `torch` says CUDA is not available, but I have a GPU
1. You forgot the GPU override: start with
   `docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d`.
2. Confirm `nvidia-smi` works **inside** the container:
   `docker compose exec dev nvidia-smi`.
3. If `nvidia-smi` works but JAX doesn't see the GPU, your host driver may be
   older than the CUDA the wheel expects — update the NVIDIA driver.

### The build is extremely slow / freezes on a Mac
You're emulating x86 on Apple Silicon — it's slow by nature. Be patient on the
first build. For real training, use a machine with an NVIDIA GPU (Windows/Linux).

### "permission denied" on files, or new files owned by `root`
You didn't pass your user id. Stop the container and rebuild after:
```bash
export HOST_UID=$(id -u); export HOST_GID=$(id -g)
docker compose down
docker compose build       # (add the gpu.yml on a GPU machine)
```

### `pip install -e` fails for a CrazySim sub-library
It needs a git version it can't find. The setup script already sets
`SETUPTOOLS_SCM_PRETEND_VERSION=1.0.0`; if you install by hand, do the same:
```bash
SETUPTOOLS_SCM_PRETEND_VERSION=1.0.0 /opt/venvs/main/bin/pip install -e repos/CrazySim/crazyflie-lib-python
```

### `gym==0.21.0` (DATT) won't install
That ancient package only builds with old tooling. The **datt** env in the image
already pins `setuptools==65.5.0` + `wheel==0.38.4` before installing it. If you
recreate the env yourself, install those two first, then `gym`.

### A C++ build (learning-to-fly / raptor) fails
- Make sure submodules are present: inside the repo run
  `git submodule update --init --recursive`, then re-run `scripts/build_cpp.sh`.
- Read the first compiler error. A missing header usually means a missing
  submodule or system `-dev` package.

### A simulator window won't open / "cannot connect to display"
See the **graphics** section of [3-develop.md](3-develop.md). Easiest fix:
render headless to a file instead of opening a window.

### I broke my environment — how do I start clean?
Your code in `repos/` is safe (it's on your computer). Just rebuild the box:
```bash
docker compose down
docker compose build         # + gpu.yml on a GPU machine
docker compose up -d
docker compose exec dev bash
bash scripts/setup_python_envs.sh && bash scripts/build_cpp.sh
```

### How do I free up disk space?
```bash
docker system df       # see what's using space
docker system prune    # remove stopped containers / dangling images (safe)
```

### `pip` complains "mujoco-mjx 3.12.0 requires mujoco>=3.12.0.dev0, but you have mujoco 3.10.0" (or any mujoco / crazyflow "ResolutionImpossible")
Two packages that must move together got different versions: `mujoco` and
`mujoco-mjx` (the race repo needs `mujoco<3.11`, an unpinned image installed 3.12).
Since 2026-09 everything is pinned (`requirements/constraints*.txt`,
`scripts/pins.sh`, [5-versions.md](5-versions.md)). Fix:
```bash
git pull                                   # get the pinned scripts
bash tasks/racing/setup.sh                 # (or scripts/setup_python_envs.sh) — moves the clones to the pins and reinstalls
bash scripts/smoke_test.sh                 # every version line should read ok
```
No image rebuild is needed: the setup scripts read `requirements/constraints.txt`
from your checkout (mounted at `/workspace`) and move `mujoco` and `mujoco-mjx`
to the pinned pair together. Rebuild only if you also want the baked packages
refreshed.

### The MuJoCo window dies with `TypeError: mjv_moveCamera(): incompatible function arguments` as soon as the mouse moves over it
Your venv has **mujoco 3.11 or newer** next to gymnasium 1.3.0. MuJoCo 3.11.0
(27 July 2026) removed the `mjvScene` argument from `mjv_moveCamera`; gymnasium's
MuJoCo viewer (up to 1.3.0) still passes it, so the first cursor event over the
window raises this error (this is also why `lsy_drone_racing` pins `mujoco<3.11`).
Check and fix:
```bash
/opt/venvs/race/bin/pip freeze | grep -i mujoco     # must read mujoco==3.10.0 and mujoco-mjx==3.10.0
bash tasks/racing/setup.sh                           # re-pins BOTH venvs to mujoco / mujoco-mjx 3.10.0
```
The same applies to `--render` in the hovering and circle tasks (they use the same
viewer). Headless runs are unaffected: `scripts/sim.py ... --render False`,
`compare_models.py` and `evaluate.py` never open a window.

### Racing: every episode "fails" / the lap never finishes / success 0/20
Your `repos/lsy_drone_racing` is probably at upstream HEAD, where the track has
**five** gate passes (`gate_order = [1, 2, 3, 4, 2]`). The course races the
four-gate lap at a pinned commit. Check and fix:
```bash
git -C repos/lsy_drone_racing log -1 --oneline     # must show 709dbc9 "Pin MuJoCo below 3.11"
bash tasks/racing/setup.sh                          # re-pins the clone
```

### Racing: `ValueError: Incompatible shapes for broadcasting: shapes=[(13,), (1, 1, 4)]`
The race config is still in `state` control mode (13-number commands) while the
bridge sends 4-number attitude commands. Set `control_mode = "attitude"` under
`[env]` in `repos/lsy_drone_racing/config/level0.toml` (and `level1.toml`) —
Lesson 3 §4. The scaffold `race_bridge.py` now refuses to start in `state` mode
and prints exactly this advice.

### A lesson cites `file.py:NN` and the line does not match
The repo is not at the pinned commit (see above), or you edited the file. The
citations are exact at the pins in `scripts/pins.sh`.

---

Still stuck? Copy the **first 15 lines** of the error and the command you ran,
and ask whoever maintains this project (or paste it to an AI assistant). Always
say which machine you're on (Windows GPU / Linux GPU / Mac CPU).
