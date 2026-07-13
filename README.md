# rl_quad_traj_project

A project on **reinforcement learning to control a quadrotor so it tracks a
desired trajectory.** This repo gives you one ready-to-use **GPU Docker
development environment** that bundles all the simulators and example code, so
anyone — on **Windows**, **macOS**, or **Linux** — can build everything *from
source*, edit it, and see the effect.

> 👋 New to Docker? No problem — these guides assume no prior experience. Follow
> the docs in `docs/` in order and you'll be fine. Start with
> **[docs/1-install-docker.md](docs/1-install-docker.md)**.

---

## What's inside

The container is a **toolbox** (CUDA, compilers, Python). The actual code lives
in `repos/` on your computer and is **mounted** into the container, so you edit
it normally and rebuild from source.

| Repository | Role | Built in |
|---|---|---|
| [crazyflow](https://github.com/learnsyslab/crazyflow) | **Main simulator** (MuJoCo / JAX) | main env (Py 3.12) |
| [CrazySim](https://github.com/gtfactslab/CrazySim) | SITL simulator (MuJoCo, firmware-in-the-loop) | C++ firmware + **crazysim** env |
| [gym-pybullet-drones](https://github.com/learnsyslab/gym-pybullet-drones) | RL-with-quadrotor study example | main env |
| [lsy_drone_racing](https://github.com/learnsyslab/lsy_drone_racing) | Drone-racing task built on crazyflow | main env |
| [learning-to-fly](https://github.com/arplaboratory/learning-to-fly) | Development example (C++ / RLtools) | C++ (CMake) |
| [raptor](https://github.com/rl-tools/raptor) | Example (C++ / rl-tools) | C++ (CMake) |
| [DATT](https://github.com/KevinHuang8/DATT) | Example | **datt env** (Py 3.10) |
| [RAPTOR_in_RotorPy](https://github.com/Sheng-Cheng/RAPTOR_in_RotorPy) | Example (RotorPy + PyTorch) | main env |

### Three Python environments (on purpose)
Some repos pin dependency versions that *conflict* (mainly the NumPy generation),
so they can't share one environment. Inside the container, switch with
`activate-main` / `activate-crazysim` / `activate-datt` (main is active by default):

| Env | Python | Holds | Stack |
|---|---|---|---|
| **`main`** | 3.12 | crazyflow, gym-pybullet-drones, lsy_drone_racing, RAPTOR_in_RotorPy | NumPy 2, JAX, MuJoCo, PyTorch (cu126), gymnasium 1.2 |
| **`crazysim`** | 3.11 | CrazySim `cflib` + `cfclient` (talk to the SITL) | NumPy <1.25 (cflib's pin) |
| **`datt`** | 3.10 | DATT | NumPy 1.23, old `gym` 0.21, PyTorch 1.13 |

DATT isn't a pip package — it's used via `PYTHONPATH` (the setup script wires this
up so `import DATT.*` just works in the `datt` env).

---

## Tasks

Learning tasks live in **`tasks/`** as *content* (code + lessons + a README) that
runs inside this one shared container — you don't build a new image per task.

| Task | What it is |
|---|---|
| [`tasks/hovering`](tasks/hovering/README.md) | Learn RL by training a Crazyflie to hover (Stable-Baselines3 + crazyflow). Start here. |
| [`tasks/circle`](tasks/circle/README.md) | Follow a moving circular path (trajectory tracking), then benchmark the RL tracker vs PID, MPC & the onboard controller. Do `hovering` first. |

Each task's `README.md` tells you the one-time setup (e.g.
`bash tasks/hovering/setup.sh` to install crazyflow into the `main` env) and how
to run it.

## CPU-only image

The default image is GPU (CUDA). For a smaller image with no CUDA (any machine):

```bash
bash scripts/build_cpu.sh        # builds rl-quad-traj:cpu
# or, with compose:
docker compose -f docker-compose.yml -f docker-compose.cpu.yml up -d --build
```

## GUI windows (3-D viewers, Linux)

To see MuJoCo viewer windows from inside the container (e.g. `--render`), add the
GUI override and allow X11 access first:

```bash
bash scripts/allow_gui.sh        # once per login (runs `xhost +local:`)
docker compose -f docker-compose.yml -f docker-compose.gui.yml up -d   # add gpu file if GPU
```

It passes your `DISPLAY` through and switches MuJoCo to a windowed GL backend.
The number-only output (no windows) works everywhere without this.

---

## ⚠️ GPU support — read this first

| Machine | GPU in Docker? | Use it for |
|---|---|---|
| **Windows + NVIDIA GPU** | ✅ Yes (via WSL2) | everything, incl. training |
| **Linux + NVIDIA GPU** | ✅ Yes | everything, incl. big training |
| **macOS** (any) | ❌ **No** — runs on CPU | coding, simulation, small runs |

Docker on macOS **cannot reach an NVIDIA GPU** (Macs have none, and there's no
CUDA for macOS). On a Mac you do all development and small runs on the CPU (this
works fine); for **heavy training**, run the exact same commands on a machine
with an NVIDIA GPU (a Windows/Linux workstation or a remote GPU server).

---

## Quickstart (after installing Docker — see `docs/1`)

```bash
cd rl_quad_traj_project
export HOST_UID=$(id -u) HOST_GID=$(id -g)   # UID is read-only in bash; use HOST_UID

bash scripts/clone_repos.sh                 # download all source repos -> ./repos

# build the image (pick your machine):
docker compose -f docker-compose.yml -f docker-compose.gpu.yml build   # NVIDIA GPU
docker compose build                                                   # macOS / CPU

# start it and go inside:
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d   # GPU
#   (Mac: docker compose up -d)
docker compose exec dev bash

# one-time finish, inside the container:
bash scripts/setup_python_envs.sh           # editable-install Python repos
bash scripts/build_cpp.sh                    # compile C++ repos
bash scripts/smoke_test.sh                   # check it all imports / sees the GPU
```

Full, click-by-click instructions are in `docs/`.

---

## Documentation
1. [Install Docker (Windows, macOS & Linux)](docs/1-install-docker.md)
2. [Build & run the environment](docs/2-build-and-run.md)
3. [Develop: edit code & see the effect](docs/3-develop.md)
4. [Troubleshooting](docs/4-troubleshooting.md)

## Repository layout
```
rl_quad_traj_project/
├── Dockerfile                 # the toolbox image (CUDA 12 + 2 Python venvs + C++ tools)
├── docker-compose.yml         # base (CPU, works everywhere)
├── docker-compose.gpu.yml     # GPU override (NVIDIA machines)
├── requirements/              # pinned Python deps for each env
│   ├── main.txt   └── datt.txt
├── scripts/                   # clone_repos, setup_python_envs, build_cpp, smoke_test
├── docs/                      # the step-by-step guides above
└── repos/                     # (created by clone_repos.sh) all source code lives here
```
