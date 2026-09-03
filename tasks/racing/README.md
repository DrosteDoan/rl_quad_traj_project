# 🏁 Task: Racing — RL Trajectory Tracking for the LSY Drone Race

Train a **reinforcement-learning trajectory-tracking controller** (the DATT
design, [arXiv:2310.09053](https://arxiv.org/abs/2310.09053)) and race it on the
[LSY drone-racing](https://github.com/learnsyslab/lsy_drone_racing) track.

This is a **task** inside the shared `rl_quad_control` development environment.
The Docker image is the *infrastructure*; this folder is the *content* you work
on.

> **Prerequisite:** do the **`hovering`** and **`circle`** tasks first. There
> you taught a drone to hover, then to follow a slow circle. Here the target
> moves *fast*, and you get a stopwatch.

## What you will build

A neural-network controller that takes a **trajectory it is given** and flies it
as fast and accurately as possible — then an evaluation that scores it exactly
the way the LSY leaderboard does.

**Scope, stated up front:** this is about **tracking**, not **planning**.
Somebody hands you the path; your job is to follow it well. Lesson 3 explains
precisely which race levels that covers (0 and 1) and which it cannot (2 and 3).
Knowing the boundary of your own method is half of engineering.

## The number to beat

| | time | what it is |
|---|---|---|
| LSY leaderboard, all-time best | **3.394 s** | winter25 semester |
| LSY leaderboard, current best | **3.419 s** | at 100 % success rate |
| This course's verified baseline | **7.80 ± 0.004 s** | Level 0, 20/20 success — one 4 M-step v5 seed on a deliberately conservative reference (cruise 2.0 m/s, wide obstacle margins). Known-achievable with exactly the tools in these lessons; beating it is your job. |

The baseline's tracking is tight — its lap time lives almost entirely in the
*reference*, which is precisely the decomposition Lessons 3 §6 and 5 teach you
to measure.

## 1. Start the toolbox container (once)

From the **`rl_quad_control/`** folder (see its `README.md` / `docs/`):

```bash
docker compose up -d                 # CPU; add -f docker-compose.gpu.yml if GPU
docker compose exec dev bash
```

## 2. Install (once per container)

The racing task needs the `race` venv (built into the image) plus editable
installs. Either run the full `bash scripts/setup_python_envs.sh` (sets up every
task) or the racing-only quick path:

```bash
bash tasks/racing/setup.sh
```

## 3. Check it works

```bash
bash scripts/smoke_test.sh           # the RACING/RACE sections must be all ok
```

## The lessons (do these in order)

| # | Lesson | ⏱️ | You finish with |
|---|---|---|---|
| 1 | [The tracking problem](lessons/01-the-tracking-problem.md) | 45 min | Being able to say what the policy sees, does, and is paid for |
| 2 | [Train a DATT policy](lessons/02-train-datt.md) | 30 min + compute | Your own trained tracker + its learning curve |
| 2b | [Where reference trajectories come from](lessons/02b-where-references-come-from.md) | 45 min | You can explain the quintic-chaining generator you are about to use, and name the alternatives (min-snap, MINCO, time-optimal, …) |
| 3 | [Evaluate like the race](lessons/03-evaluate-like-the-race.md) | 90 min | A lap time on the LSY protocol |
| 4 | [Brainstorm: make it faster](lessons/04-brainstorm-faster-tracking.md) | open | A pre-registered experiment of *your own* |
| 5 | [See the trajectory, compare the models](lessons/05-plot-and-compare.md) | 60 min | Speed-profile plots of your reference & flown laps + a head-to-head model comparison |

Lessons 1–3 are guided (2b is a concept interlude — read, don't code).
**Lesson 4 is where you do research.** Lesson 5 gives you the plots and tables
to *show* what you did.

## What's in this folder

```
tasks/racing/
├── README.md            ← you are here
├── setup.sh             ← racing-only install (run once per container)
├── crazy_track/         ← VENDORED source of the tracking project (pinned!)
│   ├── VENDORED.md      ← why it is vendored and at which commit
│   ├── src/crazy_track/ ← envs, training, eval, controllers, trajectories
│   └── results/         ← your training runs land here (git-ignored)
├── code/
│   ├── race_bridge.py   ← SCAFFOLD — you complete _build_reference in Lesson 3
│   ├── plot_trajectory.py   ← trajectory + speed-profile figures  [Lesson 5]
│   └── compare_models.py    ← head-to-head model comparison       [Lesson 5]
└── lessons/             ← 01..05, plus interlude 2b — the actual course
```

The **LSY race environment** is *not* vendored — it is public and cloned into
`repos/lsy_drone_racing` by `scripts/clone_repos.sh`, then installed into its
own `race` venv (see `scripts/setup_python_envs.sh` for why two venvs).

## Rules of the game

From measured failures in the parent project — each cost real days.

1. **Three seeds minimum** before believing anything about a policy. Two of that
   project's headline conclusions inverted when the third seed arrived.
2. **One variable per experiment.** Change the reward *and* the network and you
   have learned about neither.
3. **Write the hypothesis down first**, including the number that would make you
   wrong. Deciding afterwards what counts as success is self-deception.
4. **Never command what physics forbids.** Thrust-to-weight is 1.88, rate limit
   15 rad/s. An infeasible reference cannot be tracked by any controller.
5. **Baseline first.** You cannot claim an improvement without a number to
   improve on.

## What you hand in

1. Lap time as **mean ± std over 3 training seeds × 20 race episodes**, with the
   success rate, next to the leaderboard number.
2. Your lap-time decomposition: takeoff / path / tracking loss.
3. The Lesson-5 figures: reference vs flown trajectory with speed profile, and
   the model-comparison table.
4. Every experiment you ran with hypothesis, threshold and verdict — **including
   the ones that failed**.
5. One paragraph on what you would do next.

## Notes for teachers

* **`crazy_track` is vendored, not cloned.** Its upstream repo is private;
  students only have access to `rl_quad_control`. The snapshot is pinned so the
  lessons' file/line citations stay exact — see `crazy_track/VENDORED.md`
  before bumping it.
* **Two venvs by design.** Training runs in `main` (editable crazyflow, GPU
  torch); the race runs in `race` (lsy_drone_racing + PyPI crazyflow, CPU
  torch). Both are held at the same crazyflow / mujoco versions by
  `requirements/constraints*.txt`; sharing one env can silently downgrade the
  simulator under a trained policy.
* **The race repo is pinned** (`scripts/pins.sh`, `LSY_REF`, 2026-08-29). On
  2026-09-03 upstream changed the track to five gate passes
  (`gate_order = [1, 2, 3, 4, 2]`); every lap time in these lessons, the
  vendored `lsy_level2_race()` and the 7.80 s baseline describe the four-gate
  lap. Moving the pin means re-deriving the reference track, the baseline and
  the file/line citations (Lesson 3). See `docs/5-versions.md`.
* **Do not commit a completed `_build_reference`.** It is the Lesson-3
  exercise; keep any reference solution in an instructor-only place.

## Credits

Built on [crazyflow](https://github.com/utiasDSL/crazyflow) (simulator),
[lsy_drone_racing](https://github.com/learnsyslab/lsy_drone_racing) (race
environment, LSY/TUM), and the DATT trajectory-tracking design
([arXiv:2310.09053](https://arxiv.org/abs/2310.09053)).
