# Lesson 9 — Where the crossover is: learned vs model-based racing under graded disturbance

**Status: in progress.** Tracks, contacts, the disturbance knobs, the driver, and calibration are built and
tested; the RL recipe was rebuilt after the first training rounds (the gate-aware recipe, below) and its contrast group dropped; nothing has been swept yet. This lesson
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
| ~~`contrast_s0/1/2`~~ | ~~the same recipe, but with the *vendored*, unmatched disturbance ranges~~ — **dropped 2026-09-25** (§1) | — |

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

**Screen result: mixed, not a fix on its own.** `robust_s0_screen` improved in survival (7 → 10 of 24 gates,
0 → 1 completions) but got slightly *less* precise (RMSE 0.206 → 0.216 m); `contrast_s0_screen` regressed
(9 → 8 gates, 1 → 0 completions, RMSE 0.178 → 0.214 m). The hyperparameter mismatch was real but not the
dominant bottleneck. Three follow-up checks, each isolating one variable, converge on the same explanation:
(1) flying `robust_s0`/`contrast_s0` and M1 on the *same* tracks at the *same* margins showed M1 completing
with under 9.3 cm of deviation while the RL policies crashed 4–11× over the available margin — a controller
precision gap, not a path-generation one; (2) flying the same checkpoints inside their own training range
(λ = 0.3, 0.6, not just the nominal λ = 0 they were measured at) showed no improvement, ruling out a
train/eval distribution mismatch; (3) Lesson 7's own pre-existing benchmark data (`racing_s0`–`s2`, written
months before this lesson existed) shows the identical signature — nominal max deviation 0.38–0.54 m against
the MPC family's 0.10–0.35 m, and its own stated conclusion, "a 0.4–0.5 m deviation at 4–5 m/s is a frame."
The reference-distribution gap is the best-supported explanation on the table; `METHODOLOGY.md` §5a has the
full trace.

**A second, related gap: control authority.** The MPC family plans with the full ±1.0 rad roll/pitch bound
(`mpc_dev.py`'s `rp_max`); every RL policy in this course, robust and contrast alike, has trained under a
±0.7 rad cap inherited unchanged from the vendored recipe since Lesson 1 — an asymmetry the "equal inputs"
work of §5a's window/frequency fixes never touched, because it was never examined for the *action* side.
Cheaply checked first: M1 flown with `rp_max=0.7` (matching RL's own ceiling, zero code changes needed since
`rp_max` is already spec-configurable) still completed all three same-path tracks, just less precisely —
ruling the cap out as *sufficient* on its own to explain the RL crashes, but leaving open whether it costs
something on top of RL's larger existing tracking error. `full_authority_env.py`/`full_authority_policy.py`/
`train_full_authority.py` isolate that: one seed (0, locked, same RNG stream as `robust_s0`), everything
else held equal to `robust_s0_screen`, only the roll/pitch scale widened to match M1's:

```bash
docker exec -it rl_quad_traj bash -lc 'cd /workspace && export JAX_PLATFORMS=cpu MPLBACKEND=Agg && \
  /opt/venvs/main/bin/python tasks/racing/code/lesson9/train_full_authority.py \
  --reason "Lesson 9: control-authority screen, seed 0"'
```

Evaluated only through `robust_full:<path>` (`driver.py`) — `robust:<path>` would silently apply the wrong
(0.7×) scale to a model trained expecting the full range.

**Control-authority result: not a fix, ruled out.** The training curve plateaued in the exact same band as
`robust_s0_screen`'s own curve at the same step counts; the flown evaluation confirmed it — 8/24 gates, 0/6
completions, mean RMSE 0.2044, a wash against `robust_s0_screen`'s 10/24 gates, 1/6 completions, RMSE 0.2161.
Combined with a same-path M1 diagnostic (M1 capped to RL's own 0.7 rad ceiling still completed all three
tracks checked), control authority is ruled out as a contributing explanation.

**The locality finding.** Binning deviation from the reference into "near a gate" (within 0.3 s of a
crossing) vs "between gates," on the same three tracks: M1 shows no gate-locality at either authority level
(precise everywhere). Both RL groups show the opposite, consistently, on every track — mean deviation near a
gate is 34–148% higher than between gates, and 4 of 6 RL failure cases are clearly *diverging* in the final second before impact (one is flat, one converging), while M1 is flat or actively converging in the same window. This is not "RL fails by a
comparable amount that happens to land at a gate" — its error specifically grows worst exactly where the
geometry punishes it most.

**The viability screen.** Checking whether training even has a contact model surfaced a third gap: it
doesn't — the only crash conditions are hitting the floor or drifting more than 2 m off the reference, with
no gate geometry in the training scene at all. Rather than commit to the full reference-distribution redesign
(and the overfitting question that comes with it) before knowing whether it's worth the budget, the smaller,
prior question is whether the recipe can pass a viability gate at all: one seed, trained entirely on the 3
real dev-track references with a genuine gate-contact consequence added to `step()` (`contact.py`'s exact box
test, the same one the eval harness uses), evaluated on the existing 6-track screen at one number:
completions at λ = 0.

**Version 1: 0 of 6.** The first build re-origined each dev track so every episode started already airborne
(these are ground-start plans, and training's floor crash, `pos[:, 2] < 0.05`, would trip on a literal ground
start). It trained cleanly — reward 8 → 282, episode length 48 → 516 of 700, still climbing at 8M steps, with
`explained_variance` at 0.85–0.95 and no sign of instability; a mid-run look at the curve did not support
raising the learning rate — and then scored **0/6 completions, 0/24 gates, every failure a gate-1 contact at
t = 1.2–1.7 s**, on all six tracks including `level2`. Failures that uniform are not generic imprecision or
a too-small track pool, which would scatter across gates. The eval harness always flies a real ground start;
v1 never trained on one.

That is a train/eval gap, and it is a gap only a learned controller can have. M1 has no training distribution
to be out of: it re-solves from an explicit dynamics model at every step, so a ground launch is another
initial condition rather than a situation needing prior exposure — and it flies this exact launch on tracks
4, 93 and 387 with under 9 cm of deviation, through the same `driver.py` code path v1 failed in.

**Version 2** removes the gap at its source instead of routing around it. The references are
`driver.build_traj(track)` — the same `GroundStartTrajectory` eval constructs (hold, climb-out, gate 1 at its
normal lead-time) — and the training floor crash moves to `pos[:, 2] < -0.3`, which is `driver.py`'s own
divergence check rather than a new number. The simulator was never the obstacle (every controller flies this
launch in the harness); the obstacle was a crash rule written when nothing trained near the ground.
`episode_time` is 8.0 s to cover the longest full ground-start duration (7.196 s). v1's two candidate causes —
the missing launch, and gate 1's truncated lead-time — are removed together, so a v2 pass will not say which
mattered; the question here is viability, not attribution.

**Version 2 result: 0 of 6 again** (2/24 gates; `level2` and study 387 now clear gate 1 and hit gate 2, four
tracks still hit gate 1). Training was healthy (reward 60 → 334, episode length 155 → 521 of 800). The
informative check was flying it on its *own* three training tracks: it completes only one. On the other two it
misses gate 1 with no contact, swerving 0.4–0.8 m sideways from about 0.2 s before the gate and recovering
after it, while altitude tracks throughout and the launch is fine. Deterministic and stochastic actions give
identical outcomes, so this is not an evaluation-mode artefact. The missing ground start was at most a minor
factor.

Training ended an episode on contact, gross divergence or the floor — never on a missed gate. That makes the
two failures very unequal: contact forfeits the rest of the episode (on the order of 200 reward at gate 1, a
rough estimate) on top of its penalty, while a 0.8 m swerve costs about 0.3 in total. A policy that cannot
reliably hold a 3.5–5 cm margin is pushed to go around the gate instead of through it. (The tracking reward
does pull toward the gate; it is soft and saturating, and it loses to a cliff that costs two orders of
magnitude more.) This is inferred, not proven — "won't thread" and "can't hold the line" produce the same
trace, and the fix below cannot tell them apart; it only removes the cheap way out.

**Version 3** ends the episode on *any* missed gate, using `driver.py`'s own rule (a crossing counts inside
`HALF_OPENING` and within ±1.0 s of the gate's clock; a miss is that clock + 1.0 s), so training and eval share
one definition of failure. Replayed over six real flown paths it agrees with `driver.py` on every one — M1's
completions never trigger a false miss, and the v2 policy's misses fire at exactly gate time + 1.0 s.
Finishing all four gates does not end the episode (that would forfeit the reward for succeeding). Penalties are
−3.5 for contact or a miss and the vendored −5.0 for the floor or gross divergence; the terminal constant is
the smaller lever, since the forfeited return dominates it, so −3.5 is chosen for scale rather than expected to
change behaviour on its own. A callback now writes `gate/*` scalars to tensorboard each rollout — how episodes
ended, gates passed per episode, and `gate/full_lap_frac`, the training-time completion rate — so the next run
can show whether the policy is threading, which v2's aggregate curves could not.

```bash
docker exec -it rl_quad_traj bash -lc 'cd /workspace && export JAX_PLATFORMS=cpu MPLBACKEND=Agg && \
  /opt/venvs/main/bin/python tasks/racing/code/lesson9/train_gate_aware.py \
  --reason "Lesson 9: gate-aware viability screen v3 (missed-gate termination), seed 0"'
```

**v3 result: it learns its training tracks, and does not transfer.** 122 minutes; the new `gate/*` scalars showed
gates passed per episode rising 0 → 2.9, misses falling 0.49 → 0.00, and a pooled training full-lap rate of 0.535
(142 episodes). Flown, it completes 2 of its own 3 training tracks but only 1 of 6 on the selection set — level
with the open-space baseline (1/6, 10 vs 11 gates). Misses vanishing while contacts remain says the cheap way out
is closed and what is left is failing to hold the line (inferred from the two fractions, not from a trace).

**Version 4** tackles the two suspects together: a larger pool and longer training. `gen_pool.py` generated 111
training tracks and 22 validation tracks from fixed seed windows (about 0.25 s per candidate per worker, about 4 minutes on 8 workers),
with the same generator and filters as the study tracks; the training pool is the 3 dev tracks plus those 111, and
the validation pool is never trained on. From here RL decisions are judged on validation, not study tracks. Before
v4 existed, the validation baselines at λ = 0 were: M1 22/22 completions (RMSE 0.0525, no re-tuning), open-space RL
2/22, v3 3/22. The bar for v4 is written down in advance in `METHODOLOGY.md`: 11 or more of 22 is viable, 7–10 is
promising, 6 or fewer means neither the pool nor the steps fixed transfer. It trains 16M steps (estimated at ~4.3 hours; it took 5.7) and saves a checkpoint every 2M so one run also shows whether longer training helps transfer or only
fits the pool better.

```bash
python tasks/racing/code/lesson9/gen_pool.py   # already run: 111 train + 22 val tracks (skip)
docker exec -it rl_quad_traj bash -lc 'cd /workspace && export JAX_PLATFORMS=cpu MPLBACKEND=Agg && \
  /opt/venvs/main/bin/python tasks/racing/code/lesson9/train_gate_aware.py \
  --reason "Lesson 9: gate-aware v4 (114-track pool, 16M steps), seed 0"'
```

Evaluate each checkpoint on validation with
`eval_pool.py --role val --member label=robust:<path> --m1` (inside the container).

**v4 result: it passes the viability bar, and only that.** The final (16M) checkpoint completes **13 of 22**
validation tracks at λ = 0 (68/88 gates, RMSE 0.187), against the pre-registered line of 11: viable. For scale, M1
completes 22/22 (RMSE 0.0525), open-space RL 2/22, and v3 3/22. At matched 8M steps the 114-track pool completed
11/22 against the 3-track pool's 3/22, so the pool moved transfer; steps mattered too (4M = 3/22, 6M = 10/22).
The curve then plateaued in a 10–16 band (12M 14, 14M 16, 16M 13); the 14M checkpoint scored higher but the bar
named the final one, so 13 is the number. Viable means roughly 60% of M1, not competitive: it threads gates from
a path about 3.5× looser (RMSE 0.187 vs 0.0525, median max deviation 0.365 m vs 0.085 m). One seed, λ = 0 only.

**The λ > 0 preview, and what happened to the contrast group.** On the 22 validation tracks, v4 against M1 and M1+L1
at λ = 0.25–1.0 shows a condition-specific ordering: v4 overtakes M1 in `wind_const`, `payload`, `lighthouse` and
`combined` at some λ, never in `wind_gust`, and M1+L1 (the strongest MPC member) beats it in every steady-force
condition. On tracking error it is worse than M1 everywhere, but flat in λ where M1's grows: the completion crossovers
come from v4 not degrading, not from it tracking better. v4 was trained on ranges matched to 0.8× the very ceilings the
study measures against, so its robustness cannot be told apart from "trained on the exam". The contrast group — the same
recipe on the vendored, unmatched ranges — was meant to separate that. It did not learn the gate-aware task (0 of 22
validation tracks after 16M steps, most likely because its vendored sensor never gives an easy episode to learn from), and
every rescue either changed its curriculum or answered a narrower question, so **the control group was dropped on
2026-09-25** as infeasible. The confound stays and is stated: the RL line is "trained on the study's own ceilings".

**Held-out conditions instead.** The test moves to the other side: fly v4 and the MPC family on disturbances neither was
developed against — an upward force beyond v4's training limit (`lift`, the payload mirrored), a lighter drone (`light`),
a wind that steps on mid-lap (`step_wind`), and a frozen position stream (`blackout`) — and ask whether v4's standing
survives against each condition's nearest in-distribution analogue. The design, the difference-in-differences measure and
the decision rule are written into `METHODOLOGY.md` section 5c before any flight; `heldout_conditions.py` computes them.

```bash
docker exec -it rl_quad_traj bash -lc 'cd /workspace && /opt/venvs/main/bin/python \
  tasks/racing/code/lesson9/heldout_conditions.py \
  --member v4=robust:/workspace/tasks/racing/code/lesson9/saved/gate_aware_v4/ckpt/datt_ppo_final.zip'
```

No training is involved; it takes roughly 20 minutes on 8 workers.

**Result (2026-09-26).** By the pre-registered rule the verdict is *generalises* (2 overfit / 5 generalises / 1 mixed of 8
cells): moving from the trained condition to the unseen one, v4 mostly loses no more than the MPC members do. Two cells are
clear exceptions — upward force (v4 collapses to 8% by λ=0.75, right past its training limit, while payload holds 85%) and a
lighter drone at λ=1 (v4 69% against ≥86% for both MPC members). A post-hoc bootstrap over the 22 tracks resolves only those
two cells, and the count drops to "mixed" if the threshold were 15 points instead of the pre-registered 20. So: v4's
robustness is bounded by its training range, and is not shown to be exam-specific elsewhere — with one seed and n=22.
Absolute standing is unchanged: M1+L1 still beats v4 in raw completions under steady and stepped wind at high λ.

**Three seeds.** The same recipe trained with seeds 1 and 2 completes 13/22 and 6/22 validation tracks at λ=0 (seed 0: 13/22), so the
RL result is a spread, not a single line, and is reported per seed. Held-out verdicts: seeds 0 and 1 generalise by the pre-registered
rule, seed 2 is mixed (4 of 8 cells show the overfit signature). The lighter drone at λ=1 is an overfit cell in all three seeds; step wind
and blackout transfer in all three. Seed 2 has only 6 viable tracks, so its numbers are noisy.

## 2. Tracks, contacts, disturbances, calibration

*(the rest of this lesson is written as each phase's results land; see `METHODOLOGY.md` for the finished
design of the track library, the contact model, the five disturbance conditions and their frozen ceilings,
and the trial/grid protocol — all built and tested, none of it summarised here yet)*
