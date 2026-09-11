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
| Lesson 6's model-based reference | **7.14 s**, 20/20 at Level 0 *and* Level 1 | offset-free MPC on the pole-safe closed-form line (`race_bridge_mpc.py`, `RACE_LINE=safe RACE_CRUISE=2.0`); 6.12 s at 18/20 (Level 0) / 16/20 (Level 1) on the original line at cruise 2.5. Not a policy — the precision a model-based optimiser reaches on this plan, on the leaderboard's own clock. The parent project's 3.59–4.46 s numbers are on a different clock (Lesson 6 §0). |
| Lesson 7's fastest ranked lap | **5.32 s**, 15/20 at Level 0 (8/20 at Level 1 — unranked) | plain MPC on the pole-aware TOGT tube (`togt_plan.py --track poles`, 0.85 × TWR, unstretched, through `race_bridge_mpc.py`). **Not a policy**, and ranked at Level 0 only. The ranked *policy* number of this course remains the baseline's **8.10 s** — v5 seed 0 on the Lesson-3 line at cruise 1.5: 8.099 ± 0.004 s, 20/20 at Level 0 and 8.086 ± 0.042 s, 17/20 at Level 1, re-measured 2026-09-11 through the MPC bridge (`RACE_TAKEOFF_Z=0.7 RACE_CRUISE=1.5`). Lesson 7's racing-envelope policies beat the MPC family in the harness under noise and rank nowhere in the race (gate-frame contacts, Lesson 7 §4). |

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
| 6 | [Plan faster, track tighter: the TOGT planner and MPC](lessons/06-togt-planner-and-mpc.md) | 3 h + compute | A (plan × tracker) benchmark on both clocks — the parent project's and the leaderboard's — with pre-registered verdicts |
| 7 | [The final benchmark: MPC versions vs RL trackers, under disturbance and noise](lessons/07-final-benchmark.md) | 3 h + compute | The conditions matrix (wind, gusts, payload, Lighthouse noise) on the leaderboard clock, a racing-envelope policy trained in one run, and the 20-episode race tables that say which tracker ranks — and why the fastest one does not |

Lessons 1–3 are guided (2b is a concept interlude — read, don't code).
**Lesson 4 is where you do research.** Lesson 5 gives you the plots and tables
to *show* what you did. Lesson 6 hands you the two levers Lesson 4 could only
name — a time-optimal planner and a model-predictive tracker — and the discipline
to compare them on the clock that counts. Lesson 7 is the final benchmark: the
same trackers under the world (wind, a payload, a real positioning system) and
the walls (gate frames, poles), and the honest reading of which one ranks.

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
│   ├── plot_trajectory.py   ← trajectory + speed-profile figures  [Lessons 5, 6]
│   ├── compare_models.py    ← head-to-head comparison on the race protocol [Lessons 5, 6]
│   ├── togt/                ← TOGT-Planner driver (C++) + build.sh, and the pole-aware track yaml [Lessons 6, 7]
│   ├── togt_plan.py         ← plan a lap (raw / tube / poles) and diagnose it [Lessons 6, 7]
│   ├── race_eval.py         ← one (plan, tracker) run on either clock, any condition [Lessons 6, 7]
│   ├── race_table.py        ← the (plan × tracker) table from those runs    [Lessons 6, 7]
│   ├── race_refs.py         ← ground-start wrapper + diagnostics (shared)   [Lesson 6]
│   ├── test_race_refs.py    ← checks for race_refs.py (pytest, main venv)   [Lessons 6, 7]
│   ├── race_bridge_mpc.py   ← COMPLETE bridge: MPC family in the LSY race   [Lessons 6, 7]
│   └── train_racing.py      ← one-run racing-envelope policy (Lesson 2's v5 recipe, faster references) [Lesson 7]
├── plans/               ← TOGT plans you generate (git-ignored)
└── lessons/             ← 01..07, plus interlude 2b — the actual course
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
* **Lesson 6 pins one more repo** (`TOGT_REF` in `scripts/pins.sh`, a C++ build
  made by `code/togt/build.sh`, not by `clone_repos.sh`) and the vendored
  snapshot moved to `58eed32` for it — `crazy_track/VENDORED.md` lists what the
  bump changed and confirms the Lesson 1–5 line citations. Every number the
  parent project reports for these controllers is on its *benchmark clock*
  (hover start); the lesson makes students measure on both clocks and never
  put a hover-start number next to the leaderboard.
* **Lesson 7 trains one more policy family and makes one track decision.**
  `code/train_racing.py` is Lesson 2's v5 recipe with a single change — the
  references are drawn from the racing envelope (|v| ≤ 5 m/s, |a| ≤ 15 m/s²)
  instead of the baseline's (3.5, 10) — so a student's run is comparable to
  the baseline in everything but that variable; ~25–50 min per 4 M-step seed
  on CPU (`docs/4-troubleshooting.md`). The poles stay in the race:
  `code/togt/lsy_level2_tube_poles.yaml` (`togt_plan.py --track poles`) is the
  race-legal tube, found by a via-placement search on 2026-09-11, and
  `compare_models.py --no-poles` is the documented fallback (the poles are
  relocated to the arena corners; the frames stay). The lesson's central
  result — the racing-envelope policies win the noise matrix and never rank in
  the race — depends on contacts, so never let a student rank a tracker from
  the harness tables alone.

## Credits

Built on [crazyflow](https://github.com/utiasDSL/crazyflow) (simulator),
[lsy_drone_racing](https://github.com/learnsyslab/lsy_drone_racing) (race
environment, LSY/TUM), and the DATT trajectory-tracking design
([arXiv:2310.09053](https://arxiv.org/abs/2310.09053)).
