# 3. Developing: how to edit code and see the effect

The whole point of this setup: **you edit code on your computer with a normal
editor, and run it inside the container.** Edits are live instantly.

---

## The mental model

```
  YOUR COMPUTER                         THE CONTAINER
  ----------------                      -------------------------
  rl_quad_traj_project/   <─ mounted ─> /workspace/
    repos/crazyflow/        as the      /workspace/repos/crazyflow/
    repos/DATT/             same        /workspace/repos/DATT/
    ...                     folder      ...
  (edit here in VS Code)                (run here: python, cmake, ...)
```

`repos/` on your computer **is** `/workspace/repos/` in the container — the same
files. Save a file in your editor → it's already changed inside the container.

Because the Python repos were installed with `pip install -e` ("editable"),
changing a `.py` file needs **no reinstall** — just re-run your script.
For the **C++** repos, after editing you must **recompile**:
`bash scripts/build_cpp.sh`.

---

## Recommended editor: VS Code

1. Install **VS Code** + the **Dev Containers** extension (and on Windows, the
   **WSL** extension).
2. Open the `rl_quad_traj_project` folder.
3. *(Optional, nicest)* Click the blue corner button → **"Attach to Running
   Container"** → pick `rl_quad_traj`. Now VS Code's terminal and Python run
   *inside* the container, with full autocomplete against the installed
   libraries.
4. Point VS Code's Python interpreter to `/opt/venvs/main/bin/python`.

---

## Which Python environment do I use?

There are **four**, because their libraries conflict (mainly the NumPy
version, plus two different crazyflow pins). Inside the container:

| You're working on…                                          | Use this            | Activate with        |
|-------------------------------------------------------------|---------------------|----------------------|
| crazyflow, gym-pybullet-drones, lsy_drone_racing, RAPTOR_in_RotorPy, crazy_track training (tasks/racing) | **main** (3.12)     | `activate-main`      |
| Racing the LSY track (tasks/racing Lesson 3+ — PyPI crazyflow, same pinned version as main) | **race** (3.12)     | `activate-race`      |
| CrazySim Python (talking to the SITL via `cflib`/`cfclient`)| **crazysim** (3.11) | `activate-crazysim`  |
| DATT                                                        | **datt** (3.10)     | `activate-datt`      |

`activate-main` is on by default when you open a shell. After switching, type
`activate-main` to switch back (or open a new shell). The CrazySim *firmware*
itself is C and doesn't use any of these envs — you build it with
`scripts/build_cpp.sh`.

You can always be explicit instead of activating:
```bash
/opt/venvs/main/bin/python my_script.py
/opt/venvs/race/bin/python repos/lsy_drone_racing/scripts/sim.py --config level0.toml
/opt/venvs/crazysim/bin/python repos/CrazySim/flytest.py
/opt/venvs/datt/bin/python repos/DATT/main.py
```

---

## A typical loop (example with crazyflow)

```bash
docker compose exec dev bash          # get inside
cd repos/crazyflow
# ... edit a file in VS Code on your computer ...
python examples/your_example.py       # run it inside the container
```
Didn't like the result? Edit again, re-run. That's the loop.

### C++ example (learning-to-fly)
```bash
cd repos/learning-to-fly
# ... edit a .cpp / .h file ...
cd /workspace && bash scripts/build_cpp.sh   # recompile
./repos/learning-to-fly/build/src/<binary>   # run the rebuilt program
```

---

## Seeing graphics (simulator windows, plots)

Rendering a window from inside a container needs an "X server" on your computer.

- **Linux host:** before starting, run `xhost +local:docker` once per login.
- **Windows:** install **VcXsrv** (or use WSLg, which ships with recent Windows
  11 — GUIs often work with no setup). The compose file already passes
  `DISPLAY`.
- **Mac:** install **XQuartz**, open it, in its *Preferences → Security* tick
  *"Allow connections from network clients"*, then run
  `xhost + 127.0.0.1`. Set `DISPLAY=host.docker.internal:0`.

If graphics are painful, prefer **headless** rendering: MuJoCo can render to an
image/video file with `MUJOCO_GL=egl` (already set), and save plots to PNG
instead of showing windows. This always works, even over SSH.

---

## Where to put YOUR new code

- Quick experiments / scripts that tie repos together → make a top-level
  `experiments/` folder in the project (it's mounted, so it's editable + saved).
- Changes that belong to a repo (a new crazyflow controller, a DATT tweak) →
  edit inside that repo under `repos/`.

> Each source repo under `repos/` is its own git repository. Commit your changes
> there (e.g. `cd repos/crazyflow && git checkout -b my-feature`) so you don't
> lose them and can see exactly what you changed.

➡️ Stuck? [4-troubleshooting.md](4-troubleshooting.md)
