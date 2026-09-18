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

### Racing: `KeyError: 'n_gates_passed'` (or `'gate_sequence'`) on the first control step, or the bridge says "race observation has no [...]"
Your `repos/lsy_drone_racing` clone predates upstream's 20 July 2026 commit that
introduced these observation keys (older versions emit `target_gate` instead), so
it is far behind the pinned commit. Fix:
```bash
bash tasks/racing/setup.sh                          # moves the clone to the pin and reinstalls
git -C repos/lsy_drone_racing log -1 --oneline      # must show 709dbc9
```

### Racing Lesson 6: `bash tasks/racing/code/togt/build.sh` stops at cmake ("CMake 3.25 or higher is required") or at RapidJSON / Eigen
TOGT-Planner needs cmake ≥ 3.25 and Ubuntu 22.04's apt package is 3.22, so the build
script installs a cmake wheel into the main venv when the system one is too old (if you
set `CMAKE=` yourself, use `/opt/venvs/main/bin/cmake`). TOGT's own CMake downloads
RapidJSON at configure time and the script downloads Eigen 3.4.0 if the system has none,
so the first build needs network. A configure that fails after a partial run is usually
a stale tree: `rm -rf repos/togt-build` and re-run the script.

### Racing Lesson 6: the planner says `Config files not found!`
TOGT reads its YAML with a hand-rolled parser that keeps a trailing carriage return in
every value, so a CRLF checkout of `tasks/racing/crazy_track/configs/togt/` breaks every
path lookup. The repo's `.gitattributes` forces LF for those files; a clone made before it
existed needs a re-checkout after `git pull`:
```bash
git rm --cached -r -q tasks/racing/crazy_track/configs/togt && git checkout -- tasks/racing/crazy_track/configs/togt
file tasks/racing/crazy_track/configs/togt/lsy_level2_tube.yaml     # must not say "with CRLF line terminators"
```

### Racing Lesson 6: `Optimizatoin fails!` (sic), a segfault, or `never crosses gate N inside its opening`
`dynamicConstCheck` must stay `false` in `configs/togt/cf21b_togt/*/planning.yaml` — with it
on, L-BFGS never converges on this track's short pieces. A segfault comes from relative
parameter paths (call the planner through `togt_plan.py`, which absolutizes them) or from
prism gates (`length > 0` in a track yaml). A plan that misses a gate usually means the
`--margin` left no window, or `--mode aos`, whose refine stage drops gates on this track.

### Racing Lesson 6: `Controller execution time exceeded loop frequency by 0.0xx s` on every step
Expected for the MPC family: one ipopt solve takes 10–40 ms and the race loop has 20 ms.
The race is simulated step by step, so the lap is still valid. `race_bridge_mpc.py`
silences lsy's per-step warning and prints the measured solve time once per episode —
that number is why MPC is a reference point, not a deployable stack (Lesson 6 §6). The very
first solve of a run takes about a second (ipopt builds the problem and prints its banner);
later solves are warm-started.

### Racing Lesson 7: `compare_models.py --start hover` — the first lap ends on the gate-1 frame
`--start hover` writes `level0_hoverstart.toml` with the drone at rest at z = 1.0 m and the
rotors stopped; lsy spins them up from rest, and the drone sags about 0.5 m in the first
0.4 s. A plan that moves off at t = 0 reaches gate 1 before the drone has climbed back:
measured without a hold, one lap in five ended on the gate-1 frame. `race_bridge_mpc.py`
therefore holds the plan's first point for `RACE_SETTLE` seconds (default 1.0) before the
plan moves — keep it — and the printed time is the benchmark clock plus that hold
(`mpc_offsetfree`, lsy line, cruise 2.5: 5.62 s = 4.62 s + 1.0 s, against 4.641 s from
`race_eval.py --start hover`). If you drive your own Lesson-3 bridge with `--start hover`,
it has no such hold: `obs["pos"]` arrives at z ≈ 1.0 and your `_build_reference` must hold
there before it moves, or you will see the same frame contact. The benchmark harness never
shows this because its 1.5 s lead-in does the settling.

### Racing Lesson 7: `train_racing.py` is slow / how long does it take?
4 M steps at 16 envs is 977 PPO iterations of 4096 steps. Measured on the validation
machine (14-core CPU, no GPU): 2615 steps/s and 25.5 min when the run had the machine to
itself; 1300 steps/s and 51 min when two seeds ran at once; about 740 steps/s while an MPC
matrix ran alongside. Every iteration prints `fps` (the cumulative average since the start)
and `time_elapsed`, so the time left is `(4000000 - total_timesteps) / fps`. Lesson 2's
"~50 minutes" for the baseline is the same simulator and the same PPO — the racing
envelope changes the references, not the cost per step — so the two runs are equally long
on the same machine. Do not start more than about six simulations at once (each
`race_eval.py` MPC cell is one), and the line "An NVIDIA GPU may be present ... Falling
back to cpu" at the top of the log is expected on a machine without a CUDA jaxlib.

### Racing Lesson 8: a ground-start plan at Level 1 ends after 0.3–0.9 s with 0 gates

The drone never leaves the floor (or slides into pole 1 from the start pad). Level 1
draws the start pose up to 0.1 m away from the plan's first point, and a ground-start
plan holds that first point for 0.2–0.3 s while the rotors spin up — so the reference
asks for a 0.1 m rest-to-rest move in 0.2 s — a quintic's peak acceleration is 5.77·d/T²,
so about 14 m/s² — while the drone is still on the ground and cannot make it. Set `RACE_START_BLEND=1.0`: the reference then holds the
drone *where it actually is* and fades the offset out over one second. It is not needed
for the Lesson-6/7 plans (they start with a 1.5 s takeoff from the observed pose), and it
is a no-op only when the plan's first point is the start pose itself — the `ground`
tracks start 0.04 m higher, so the blend is doing something there even at Level 0.

### Racing Lesson 8: my shell mangles `mpcdev:att=sim,drag=0.495,fgain=1.0`

Quote the spec: `RACE_CONTROLLER='mpcdev:att=sim,drag=0.495,fgain=1.0'` (and the same for
`--controller`). Unquoted, some shells split on the commas or treat `=` specially, and the
spec parser then rejects a key it never received. An unknown key is always an error, on
purpose: a typo in a sweep is worth a crash, not a silent default.

### Racing Lesson 8: `No controller found in ...` or `Multiple controllers found in ...`

lsy's loader imports the file you name and expects **exactly one** `Controller` subclass in
it. Copy the *file* into `repos/lsy_drone_racing/lsy_drone_racing/control/`, not the folder,
and copy `race_bridge_mpc.py` and `race_probe.py` separately — they are one controller each.
If you extended a bridge by pasting a second class into it, move that class to its own file.

### Racing Lesson 8: the harness and the race disagree by 0.02 s for `mpcdev`

Both are right. The race reports the time at the *start* of the 20 ms step in which the last
gate is crossed (lsy's `sim.py`: `curr_time = i / freq`), so every lap is floored to a
multiple of 0.02 s. The corrected-model tracker crosses gate 4 about 10–30 ms earlier than
the vendored one on the same plan, which is often enough to fall into the previous step — so
its 5.30 s against 5.32 s is a rounding boundary, not 20 ms of speed. Compare plans by their
reference's last-gate time, and trackers by their success count.

### A lesson cites `file.py:NN` and the line does not match
The repo is not at the pinned commit (see above), or you edited the file. The
citations are exact at the pins in `scripts/pins.sh`.

---

Still stuck? Copy the **first 15 lines** of the error and the command you ran,
and ask whoever maintains this project (or paste it to an AI assistant). Always
say which machine you're on (Windows GPU / Linux GPU / Mac CPU).
