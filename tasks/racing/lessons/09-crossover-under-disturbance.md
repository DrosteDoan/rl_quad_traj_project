# Lesson 9 — Where the crossover is: learned vs model-based racing under graded disturbance

**Status: in progress.** Tracks, contacts, the disturbance knobs, the driver, and calibration are built and
tested; the robust and contrast RL recipes are built and tested; nothing has been trained yet. This lesson
grows as each phase lands. The living design document, updated before this lesson text catches up to it, is
`tasks/racing/code/lesson9/METHODOLOGY.md` — read it first if a claim here and there disagree.

**Prerequisites:** Lesson 7 (the disturbance models, the conditions matrix), Lesson 8 (the corrected MPC
family, ground-start plans, the box contact model). Everything runs in the `rl_quad_traj` container.

---

> **The one big idea:** Lesson 7 asked whether a tracker is precise, robust, or survives contact, at one
> disturbance setting, on one track. This lesson asks a sharper question: as a disturbance grows from
> nothing to a calibrated ceiling, *where* does the ranking between model-based and learned trackers flip,
> and does that crossover point sit in the same place on fifteen different track geometries or wander with
> the course. The notice that governs every number this lesson produces: **the crossover regimes found here
> are not intended to represent universal thresholds for learned or model-based control** — they describe
> this course's tracks, this course's disturbance models, and these specific trackers.

## 0. The trackers

Five model-based, the Lesson 8 corrected family, plus two learned groups of three seeds each:

| column | what it is | spec |
|---|---|---|
| `M1` | the corrected MPC (Lesson 8's measured model) | `mpcdev:att=sim,drag=0.495,fgain=1.0` |
| `M1+ESO` | M1 + velocity ESO | `...,dist=eso` |
| `M1+L1` | M1 + L1 adaptation | `...,dist=l1` |
| `M1+mass` | M1 + thrust-scale (mass) adaptation | `...,mass=1` |
| `mppi_l1` | sampling-based MPC + L1, reconfigured to an 0.8 s preview horizon | `mppi_l1` |
| `robust_s0/1/2` | trained with domain randomization matched to this lesson's own disturbance ceilings | `robust:<path>` |
| `contrast_s0/1/2` | the same recipe, but with the *vendored*, unmatched disturbance ranges | `robust:<path>` |

The contrast group exists to answer one question honestly rather than just disclose it: does matching the
RL training ranges to the disturbance ceilings this lesson measures against buy anything beyond generic
domain-randomization robustness? Full reasoning in `METHODOLOGY.md` §5b.

## 1. Train the two RL groups

Both recipes start from the same racing-envelope policy Lesson 7 §2 trained (asymmetric actor-critic, PPO
settings unchanged), with two things equalised between them so the comparison to the MPC family is not
handicapped by an accident of the vendored recipe: an 0.8 s reference preview (matching the MPC family's
own horizon, not the vendored 0.6 s) and training at 100 Hz (matching the harness, so the policy's own L1
estimator is not shaped by one control rate and evaluated at another — `METHODOLOGY.md` limitation 14).

What differs between the two groups is exactly the disturbance training:

|  | force box | Lighthouse | mass |
|---|---|---|---|
| `robust` (`train_robust.py`) | x,y ±4.4 m/s², z −3.6..+1.8 (0.8× this lesson's frozen ceilings) | λ ~ U(0, 0.8) of this lesson's own axis, sizes + refresh interval coupled | fraction ~ U(0, 0.184), heavier only |
| `contrast` (`train_contrast.py`) | vendored ±3.5 m/s² box, unchanged | vendored noise scale ~ U(0, 1.5), uncoupled from the refresh rate | none |

Six seeds (0, 1, 2 for each group) — not five, not a lone seed per group. Lesson 7 §2 found a single
training seed can be a coin flip (`racing_s1`, 3/4 on every plan through no fault of the recipe); three
seeds per group, the same count on both sides, keeps that lottery from quietly favouring whichever group
happened to draw fewer seeds. Six also schedules cleanly: paired two at a time under a 2-concurrent-session
machine, three rounds use every slot, five would waste one.

Run each round's pair together and wait for both to finish before starting the next (about 50–100 minutes
per run on this machine, so roughly 2.5–5 hours end to end for all six; `--timesteps 8000000` because
training at 100 Hz halves the simulated seconds per environment step at a fixed step budget, so the budget
is doubled to keep the amount of simulated experience comparable to the vendored recipe's 4,000,000 at
50 Hz):

Every command below runs INSIDE the `rl_quad_traj` container, not on the host — `numpy`, `jax` and
`crazyflow` live only in the container's `/opt/venvs/main`, and running a bare `python ...` on the host
fails with `ModuleNotFoundError: No module named 'numpy'` (or worse, silently picks up an unrelated host
Python). Each command below is a complete, self-contained `docker exec`, safe to paste into a plain host
terminal as-is:

```bash
# round 1 -- run these two together, in separate terminals
docker exec -it rl_quad_traj bash -lc 'cd /workspace && export JAX_PLATFORMS=cpu MPLBACKEND=Agg && \
  /opt/venvs/main/bin/python tasks/racing/code/lesson9/train_robust.py \
  --timesteps 8000000 --seed 0 --reason "Lesson 9: robust seed 0"'
```
```bash
docker exec -it rl_quad_traj bash -lc 'cd /workspace && export JAX_PLATFORMS=cpu MPLBACKEND=Agg && \
  /opt/venvs/main/bin/python tasks/racing/code/lesson9/train_contrast.py \
  --timesteps 8000000 --seed 0 --reason "Lesson 9: contrast seed 0"'
```
```bash
# round 2
docker exec -it rl_quad_traj bash -lc 'cd /workspace && export JAX_PLATFORMS=cpu MPLBACKEND=Agg && \
  /opt/venvs/main/bin/python tasks/racing/code/lesson9/train_robust.py \
  --timesteps 8000000 --seed 1 --reason "Lesson 9: robust seed 1"'
```
```bash
docker exec -it rl_quad_traj bash -lc 'cd /workspace && export JAX_PLATFORMS=cpu MPLBACKEND=Agg && \
  /opt/venvs/main/bin/python tasks/racing/code/lesson9/train_contrast.py \
  --timesteps 8000000 --seed 1 --reason "Lesson 9: contrast seed 1"'
```
```bash
# round 3
docker exec -it rl_quad_traj bash -lc 'cd /workspace && export JAX_PLATFORMS=cpu MPLBACKEND=Agg && \
  /opt/venvs/main/bin/python tasks/racing/code/lesson9/train_robust.py \
  --timesteps 8000000 --seed 2 --reason "Lesson 9: robust seed 2"'
```
```bash
docker exec -it rl_quad_traj bash -lc 'cd /workspace && export JAX_PLATFORMS=cpu MPLBACKEND=Agg && \
  /opt/venvs/main/bin/python tasks/racing/code/lesson9/train_contrast.py \
  --timesteps 8000000 --seed 2 --reason "Lesson 9: contrast seed 2"'
```

`JAX_PLATFORMS=cpu` is not optional on this machine: without it, JAX processes on this WSL setup fall back
to a path that is measured over 10x slower and hangs at exit (found while parallelising the calibration
sweep, §5 below). Thread counts are deliberately left uncapped here, unlike the calibration launcher's
8-way parallel sweep — two long, internally-vectorized training runs should split the machine's cores
between them rather than each being pinned to one.

Each run prints its own log directory on startup
(`tasks/racing/crazy_track/results/<stamp>_racing-robust-train/` or `..._racing-contrast-train/`), with
`datt_ppo_final.zip` and a tensorboard log inside once it finishes. A model from either script is evaluated
through `robust:<path>` (`driver.py`), never `datt:<path>` — that spec loads through the vendored
controller, which assumes the vendored 0.6 s window and would silently mistrack the reference.

**Watch `rollout/ep_rew_mean` in tensorboard as it trains.** The vendored `SB3Adapter` these scripts build
on never reports it (a pre-existing gap — `_update_info_buffer` needs an `"episode"` key in `infos` that
the vendored adapter doesn't set; `METHODOLOGY.md` §5a has the full trace). `monitored_adapter.py` fixes
this without touching the vendored file or changing how the policy trains — confirmed that key is read only
for logging, never by the actual PPO update. Round 1 (seed 0 of each group) trained before this fix
existed, so it has to be read from optimizer-internal diagnostics alone (`train/value_loss`,
`train/explained_variance`) — both looked like ordinary, converging PPO runs, but flown directly on five
study tracks at nominal conditions, `robust_s0` completed none and `contrast_s0` completed one.

All three seeds of both groups, once trained, told a consistent story: 0–2 of 6 screening tracks completed,
RMSE 0.13–0.36 m against the MPC family's nominal 0.06–0.10 m, no clear improving trend across seeds — not a
bug (mechanics checked out at every stage) but not close to competitive either.

**A hyperparameter mismatch, found after rounds 1–3.** `gamma=0.98` and `n_steps=256` are both defined in
units of steps, and neither was rescaled when `freq` doubled from the vendored 50 to this recipe's 100 —
`gamma`'s effective discount horizon halved from 1.0 s to 0.5 s (shorter than the 0.8 s observation window
built for this recipe), and `n_steps`'s rollout window halved from 85% of an episode to 43%. Corrected
defaults are now CLI flags (`--gamma 0.99 --n-steps 512 --batch-size 2048`; add `--gamma 0.98 --n-steps 256
--batch-size 1024` to reproduce rounds 1–3 exactly). Before redoing all six seeds, screen with one pair at
seed 0 — the same seed already used, so this isolates the hyperparameter change alone:

```bash
docker exec -it rl_quad_traj bash -lc 'cd /workspace && export JAX_PLATFORMS=cpu MPLBACKEND=Agg && \
  /opt/venvs/main/bin/python tasks/racing/code/lesson9/train_robust.py \
  --seed 0 --reason "Lesson 9: hyperparameter screen, robust seed 0"'
```
```bash
docker exec -it rl_quad_traj bash -lc 'cd /workspace && export JAX_PLATFORMS=cpu MPLBACKEND=Agg && \
  /opt/venvs/main/bin/python tasks/racing/code/lesson9/train_contrast.py \
  --seed 0 --reason "Lesson 9: hyperparameter screen, contrast seed 0"'
```

(the corrected values are now the defaults, so no extra flags are needed for the screen itself). If this
closes the gap meaningfully, all six seeds get redone at the corrected settings. If it barely moves things,
the more likely bottleneck is structural: every training reference is a smooth, randomly-wandering curve in
open space (`ChainedPolyTrajectory.random`) — nothing in it ever asks the policy to thread a narrow,
oriented opening, which is the literal skill a 0.4 m gate with 3.5–5 cm of margin demands. `METHODOLOGY.md`
§5a has three design options for that, not yet decided.

## 2. Tracks, contacts, disturbances, calibration

*(the rest of this lesson is written as each phase's results land; see `METHODOLOGY.md` for the finished
design of the track library, the contact model, the five disturbance conditions and their frozen ceilings,
and the trial/grid protocol — all built and tested, none of it summarised here yet)*
