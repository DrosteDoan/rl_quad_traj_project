# 5. Versions & pins — what is installed, and why it is pinned

Read this if `pip` complains about `mujoco`, `mujoco-mjx` or `crazyflow`, if a
lesson cites a line number that does not match, or before you upgrade anything.

---

## The short version

Every simulator-related package and both source repos are **pinned**. Two
files own the pins:

| File | Pins |
|---|---|
| `scripts/pins.sh` | the git commits `repos/crazyflow` and `repos/lsy_drone_racing` are checked out at (`CRAZYFLOW_REF`, `LSY_REF`) |
| `requirements/constraints.txt` (GPU image) / `constraints-cpu.txt` (CPU image) | the Python packages every `pip install` in this project is held to. At runtime the setup scripts read `constraints.txt` from the **mounted repo** first, so a `git pull` updates the pins without an image rebuild (a constraint on a package that is not requested, e.g. the CUDA plugin on the CPU image, has no effect) |

`bash scripts/smoke_test.sh` prints the versions that are actually installed
in each venv and the commit each repo is at. That output is the first thing to
paste when you report a problem.

## The versions

Both task venvs run **Python 3.12**. Verified resolution on 2026-09-03 (Linux,
`uv pip compile` of the image files; the same for the GPU and CPU images
unless noted):

| Package | `main` venv (training, hovering/circle/racing) | `race` venv (racing, Lesson 3+) | Why this version |
|---|---|---|---|
| crazyflow | **0.3.2**, editable from `repos/crazyflow` @ `dede875` | **0.3.2** from PyPI | the simulator; both venvs must simulate the same physics |
| mujoco | **3.10.0** | **3.10.0** | lsy_drone_racing requires `<3.11` (their #134): MuJoCo 3.11 removed the scene argument of `mjv_moveCamera`, and gymnasium 1.3.0's viewer still passes it — the window dies with `TypeError: mjv_moveCamera(): incompatible function arguments` on the first mouse move |
| mujoco-mjx | **3.10.0** | **3.10.0** | must be the same minor as mujoco (`mujoco-mjx X.Y` requires `mujoco>=X.Y`) |
| gymnasium | **1.3.0** | **1.3.0** | crazyflow ≥1.2 and lsy ≥1.2; 1.3.0 is what upstream's own lock uses |
| jax / jaxlib | **0.9.2** (+ `jax-cuda12-plugin` 0.9.2 on the GPU image) | **0.9.2** (CPU) | one JAX across all installs, matched to the CUDA plugin |
| numpy | 2.5.x | 2.5.x | crazyflow needs ≥ 2 |
| scipy | 1.18.x | 1.18.x | crazyflow needs ≥ 1.17 |
| torch | **2.8.0+cu126** (GPU image) / **2.8.0+cpu** (CPU image) | **2.8.0+cpu** | cu126 matches the RTX 2080-class host driver; lsy's `rl` extra pins 2.8.0 |
| stable-baselines3 | **2.9.0** | **2.9.0** | policies saved in `main` must load in `race` |
| sb3-contrib | **2.9.0** | **2.9.0** | RecurrentPPO (`--v7`) |
| flax | 0.12.6 | 0.12.6 | highest that accepts jax 0.9.2 |
| warp-lang | 1.17.x | 1.17.x | lsy dependency |
| lsy_drone_racing | editable, `repos/lsy_drone_racing` @ `709dbc9` (2026-08-29) | same | see below |
| crazy_track (vendored) | editable, `tasks/racing/crazy_track` | editable, `--no-deps` | pinned snapshot `1921aa3` (see its `VENDORED.md`) |

The other venvs (`crazysim`, `datt`) are unchanged and unrelated to the tasks.

## Why the pins exist (the 2026-09 "mismatching versions" incident)

1. The image used to install `mujoco>=3.3.0` and `mujoco-mjx>=3.3.0`, so a
   build in late August 2026 got **mujoco 3.12 / mjx 3.12**.
2. The setup scripts cloned `lsy_drone_racing` at upstream **HEAD**. On
   2026-08-29 upstream added `mujoco>=3.3.0,<3.11.0` (their PR #134, a
   gymnasium viewer problem with newer mujoco).
3. `tasks/racing/setup.sh` / `scripts/setup_python_envs.sh` then installed
   that lsy into `main`: pip downgraded `mujoco` to 3.10 **under** a 3.12
   `mujoco-mjx`, printing
   `mujoco-mjx 3.12.0 requires mujoco>=3.12.0.dev0, but you have mujoco 3.10.0`.
   Depending on the day the clone was made, students saw either that error in
   `main`, or a race venv on a different mujoco/jax than the venv their policy
   was trained in.
4. Independently, on 2026-09-03 upstream changed the race to **five gate
   passes** (`gate_order = [1, 2, 3, 4, 2]`, their #135). A bridge that flies
   the four-gate lap of Lessons 3–5 never "finishes" on that track: every
   episode is scored as a failure.

The fix is structural, not a one-off: mujoco/mjx/gymnasium/crazyflow/SB3 are
constrained everywhere, both repos are held at explicit commits, and the
setup scripts *move an existing clone to the pin* and *move an older image's
mujoco / mujoco-mjx pair to the pinned versions together* when re-run — so an
already-built image is repaired by `bash tasks/racing/setup.sh` (or
`scripts/setup_python_envs.sh`) alone.

## Which commit, and why exactly these

| Repo | Commit | Date | Why |
|---|---|---|---|
| `learnsyslab/crazyflow` | `dede875b685e29de5b65d3cbd29b481035f6fa30` (version 0.3.2) | 2026-08-26 | the release the race venv gets from PyPI, so `main` and `race` are identical; 0.3.1 → 0.3.2 only touched Gaussian-splat rendering |
| `learnsyslab/lsy_drone_racing` | `709dbc9dfa4c07ea13381959b2520f5d981dbff1` | 2026-08-29 | last commit before the five-pass track; includes the mujoco pin; the Lesson-3 citations (`controller.py:63`, `race_core.py:237`, `attitude_controller.py:130`) are exact here |

## Upgrading (for teachers)

Do it deliberately, all at once, and re-verify:

1. Change `CRAZYFLOW_REF` / `LSY_REF` in `scripts/pins.sh` and the pins in
   both `requirements/constraints*.txt` **and** `requirements/main*.txt`
   (mujoco and mujoco-mjx together).
2. Rebuild the image, `bash scripts/clone_repos.sh`, `bash scripts/setup_python_envs.sh`,
   `bash scripts/smoke_test.sh`.
3. Re-run the three task checks (`python -m k12_hover.check` in hovering and
   circle, the racing smoke section) and the vendored tests
   (`cd tasks/racing/crazy_track && python -m pytest tests -q`).
4. For a new `lsy_drone_racing` commit: diff `config/level0.toml` (gate poses,
   `gate_order`), re-check every `file.py:NN` citation in
   `tasks/racing/lessons/`, and re-measure the baseline lap in the racing
   README. If upstream's track changed, the vendored `lsy_level2_race()` and the
   Lesson-3 bridge design must change with it.
5. Retrain rather than reuse checkpoints across a simulator upgrade: the
   vendored `crazy_track` research code was developed on crazyflow 0.2.1, the
   racing lessons were validated on 0.3.x, and policies are only comparable
   within one version.

## Checking your own machine

```bash
bash scripts/smoke_test.sh
```

Expected: `crazyflow @ dede875`, `lsy_drone_racing @ 709dbc9`, every `ok`
line, and `crazyflow main=0.3.2 race=0.3.2 (same physics in both venvs)`.

## Verified (2026-09-03)

The pinned configuration was exercised on Linux (WSL Ubuntu 24.04, Python 3.12, CPU) with
`main` and `race` venvs created exactly as the Dockerfile creates them and the repo's own
`tasks/*/setup.sh` scripts:

- **Repair test.** A `main` venv deliberately broken like an old image (mujoco 3.12 +
  mujoco-mjx 3.12) with `repos/lsy_drone_racing` at upstream HEAD and an empty `race` venv was
  repaired by `bash tasks/racing/setup.sh` alone: mujoco and mujoco-mjx moved to 3.10.0 together,
  lsy moved to `709dbc9`, and both venvs ended on crazyflow 0.3.2 / mujoco 3.10.0 / gymnasium 1.3.0
  / jax 0.9.2 / SB3 2.9.0. `scripts/smoke_test.sh` reported every line `ok` and
  `crazyflow main=0.3.2 race=0.3.2 (same physics in both venvs)`.
- **Vendored tests:** `python -m pytest tasks/racing/crazy_track/tests -q` → 70 passed.
- **Lesson entry points on this stack:** hovering `check` / shipped model (0.2 cm) / 20 k-step
  training; circle `check` / hover brain on the period-8 circle (12.5 cm) / tracker on the period-4
  circle (2.7 cm) / `benchmark_traj` (RL 3.4 cm ≈ MPC 3.3 cm < PID 7.4 cm < State 11.9 cm);
  racing `ppo_train --v5` (obs 56) / figure-8 benchmark of a 4 M-step v5 policy (0.055 m slow,
  0.163 m fast) / `plot_trajectory` / Lesson 3 action-space check / the race bridge under
  `scripts/sim.py` and `compare_models.py`.
- **End to end:** a completed Lesson-3 bridge (instructor copy, not in this repo) driving a 4 M-step
  v5 policy lapped Level 0 **20/20** under `compare_models.py` on this stack (1.5 s takeoff, cruise
  1.5 m/s, 8.20 s). Faster references clip a gate frame — the speed–risk dial of Lessons 3–5.
