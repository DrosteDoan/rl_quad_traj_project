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

---

Still stuck? Copy the **first 15 lines** of the error and the command you ran,
and ask whoever maintains this project (or paste it to an AI assistant). Always
say which machine you're on (Windows GPU / Linux GPU / Mac CPU).
