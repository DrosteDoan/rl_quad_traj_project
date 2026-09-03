# 2. Build & run the environment

Do these steps in order. After the first time, starting up is just one command.

Throughout, `$` means "type this in your terminal" (don't type the `$`).
On Windows, run everything **inside the Ubuntu (WSL) terminal**, not PowerShell.

---

## Step 1 — Get the project and the source code

```bash
# go to where you keep projects, then:
cd rl_quad_control               # this folder (the one with the Dockerfile)

# download all 8 source repositories into ./repos
# (crazyflow and lsy_drone_racing are checked out at pinned commits — see 5-versions.md)
bash scripts/clone_repos.sh
```

This downloads crazyflow, lsy_drone_racing, CrazySim, gym-pybullet-drones,
learning-to-fly, raptor, DATT and RAPTOR_in_RotorPy into `repos/`. It can take a
few minutes.

---

## Step 2 — Build the Docker image (one-time, ~15–40 min)

This downloads CUDA, Python and all the heavy libraries **once**. Grab a snack.

First, tell Docker your user id so files you create stay owned by you:

```bash
export HOST_UID=$(id -u)
export HOST_GID=$(id -g)
```
(We use `HOST_UID`, not `UID`, because `UID` is read-only in the bash shell.)

### Windows (NVIDIA GPU) / Linux GPU box — build with GPU
```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml build
```

### macOS — build CPU-only
```bash
# Apple Silicon: force the amd64 platform (needed for CUDA libraries)
export DOCKER_DEFAULT_PLATFORM=linux/amd64
docker compose build
```
(Intel Mac: you can skip the `DOCKER_DEFAULT_PLATFORM` line.)

> Want the full Gazebo simulator for CrazySim too? Add
> `--build-arg INSTALL_GAZEBO=true` — but we use **MuJoCo** by default, so you
> usually don't need it. It makes the image much bigger.

---

## Step 3 — Start the container (leave it running)

### Windows / Linux GPU box
```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d
```

### macOS
```bash
docker compose up -d
```

`up -d` starts the box in the background and keeps it alive. You start it once
and keep using it all day.

---

## Step 4 — Go inside the container

```bash
docker compose exec dev bash
```

Your prompt changes — you are now **inside** the container, in `/workspace`
(which is this project folder). The main Python environment is already active.

> 💡 You'll do almost all your work from inside this shell. Open a second
> terminal tab and `docker compose exec dev bash` again whenever you want
> another shell into the same box.

---

## Step 5 — Finish setting up the code (one-time, inside the container)

```bash
# install the Python repos in "editable" mode (fast — libraries already there)
bash scripts/setup_python_envs.sh

# compile the C++ repos (learning-to-fly, raptor, CrazySim firmware)
bash scripts/build_cpp.sh
```

---

## Step 6 — Check everything works

```bash
bash scripts/smoke_test.sh
```

You should see `ok` next to the libraries.
- On a **GPU machine**: `jax devices:` lists a `cuda` device and
  `torch CUDA available: True`.
- On a **Mac**: it says CPU mode / `False` — that's expected and fine.

🎉 You're set up. Next: [3-develop.md](3-develop.md).

---

## Every day after this

You don't rebuild. You just:

```bash
# start the box (pick the line for your machine)
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d   # GPU
docker compose up -d                                                   # Mac

# jump in
docker compose exec dev bash
```

When you're done for the day (optional):
```bash
docker compose stop      # pause the box (fast to resume)
# or
docker compose down      # remove the box (your code in ./repos is safe)
```
