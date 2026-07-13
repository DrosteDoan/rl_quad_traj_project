# 🔵 Task: Follow a Circle — Trajectory Tracking & a Controller Benchmark

Go beyond hovering: teach the simulated Crazyflie 2.1 Brushless to **follow a
moving target around a circle**, then **benchmark** your reinforcement-learning
tracker against the classic controllers engineers already trust (PID, MPC, and the
drone's built-in onboard controller).

This is a **task** inside the shared `rl_quad_control` development environment. The
Docker image is the *infrastructure*; this folder is the *content* you work on.

> **Prerequisite:** do the **`hovering`** task first (its Lessons 1–6). This task
> builds directly on the hover environment, reward knobs, and trained hover brain.

```
        ·····●·····            the target (●) slides around the circle
      ··         ··            and the drone 🚁 has to keep up with it
     ·    🚁       ·
      ··         ··
        ···········
```

---

## 1–3. Container, install, check (same as hovering)

```bash
# from rl_quad_control/ — start the container (add the gpu override if you have one)
docker compose up -d
docker compose exec dev bash

# inside the container, once per container:
bash tasks/circle/setup.sh          # installs Crazyflow into the main venv
cd /workspace/tasks/circle
python -m k12_hover.check            # prints "Everything works! ✅"
```

`setup.sh` installs the same `learnsyslab/crazyflow` the hovering task uses. The
extra libraries this task needs — **casadi** (the MPC baseline), **matplotlib** and
**scipy** (benchmark plots) — are **already in the `main` venv**, so there is
nothing else to install.

---

## 4. The 3-minute taste

```bash
# (a) Reuse your HOVER brain to follow a slow circle — NO new training (Lesson 8)
python -m k12_hover.eval_traj --model models/hover_PPO.zip --period 8

# (b) Watch a trained TRACKER follow a fast circle (Lesson 9)
python -m k12_hover.eval_traj --model models/track_PPO.zip --n-samples 10 --period 4

# (c) Score RL vs PID vs MPC vs the onboard controller (Lesson 11) -> a PNG
python -m k12_hover.benchmark_traj --model models/track_PPO.zip --n-samples 10 --laps 3
```

Add `--render` (with the GUI override, see the hovering README) to watch in 3-D:
the grey dots are the circle and the red ball is the moving target.

> **About the included `models/track_PPO.zip`:** it was trained on the reference
> Crazyflow fork and is provided so you can try `eval`/`benchmark` immediately. For
> the best results in *this* toolbox, **train your own** (it only takes a few
> minutes on a GPU) — that is exactly what Lesson 9 walks you through:
> ```bash
> python -m k12_hover.train_traj_sb3 --jax-device gpu --num-envs 256 \
>     --timesteps 3000000 --n-samples 10 --period 4 --save-name track_PPO
> ```

---

## 5. The lessons (do these in order, after the hovering task)

Read the markdown in `lessons/` on your computer; run the commands in the container.

| # | Lesson | What you learn |
|---|--------|----------------|
| 8 | `lessons/08_follow_a_circle.md` | Reuse your hover brain to chase a moving target — no retraining |
| 9 | `lessons/09_train_a_tracker.md` | Add a *look-ahead* and train a real path-tracker |
| 10 | `lessons/10_meet_the_controllers.md` | How RL, PID, State & MPC each work, and where they live in the code |
| 11 | `lessons/11_run_the_benchmark.md` | Race all four on one circle; score with RMSE |
| 12 | `lessons/12_stress_test_and_future.md` | Push RL until it breaks; its limits and how to improve it |

(The lessons are numbered 8–12 because they continue the hovering task's 1–6 and
the root project's deploy lesson 7.)

---

## What's in this folder

```
tasks/circle/
├── README.md            ← you are here
├── setup.sh             ← installs Crazyflow into the main venv (run once)
├── k12_hover/           ← the code
│   ├── hover_env.py         ← the hover TASK (reused as the base)         [hovering L2]
│   ├── trajectory_env.py    ← the circle TASK: a moving target + look-ahead [Lessons 8-9]
│   ├── train_traj_sb3.py    ← train a circle-follower                     [Lesson 9]
│   ├── eval_traj.py         ← watch & score circle-following              [Lessons 8-9]
│   ├── controllers.py       ← classical baselines: PID + MPC              [Lesson 10]
│   ├── benchmark_traj.py    ← race RL vs PID/MPC/State (RMSE plot)        [Lesson 11]
│   ├── stress_test.py       ← sweep circle speed to find RL's limits      [Lesson 12]
│   ├── train_sb3.py eval_sb3.py hover_env.py wrappers.py sb3_adapter.py check.py
│   └── __init__.py          ← env factories (make_sb3_env / make_traj_sb3_env)
├── lessons/             ← Lessons 8-12
└── models/              ← hover_PPO.zip (reuse demo) + track_PPO.zip (reference)
```

---

## Notes for teachers

* **Same infrastructure as hovering.** No image rebuild is needed: casadi,
  matplotlib and scipy are already in `requirements/main.txt`. If you ever *do*
  add a package, edit that file (or the `Dockerfile`), rebuild, and share it back.
* **This task is ported to the `learnsyslab/crazyflow` API** (Python ≥ 3.12,
  `Dynamics`/`drone=`, 3-argument reset), like the hovering task — *not* the
  utiasDSL API used by the root `k12_RL_quad_traj/k12_hover/` project. The code is
  the same lessons, adapted to this fork.
* The MPC baseline uses **casadi + ipopt** (a point-mass teaching model). It is the
  slowest controller (~5 ms/step) but needs no compiled solver — unlike the
  acados-based MPC in `lsy_drone_racing`.
