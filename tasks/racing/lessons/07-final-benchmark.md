# Lesson 7 — The final benchmark: MPC versions vs RL trackers, under disturbance and noise

⏱️ ~3 hours of reading and running, plus a few hours of compute in the
background (three training seeds, the conditions matrix, the race protocol —
start each before you read the section that uses it). **You finish when** you
have (a) the conditions matrix of §3 on the leaderboard clock with a verdict on
each pre-registered hypothesis, (b) the 20-episode race tables of §4 for at
least one plan, and (c) one sentence, defended by a row of each, on why the
tracker that wins the matrix does not rank in the race.

> **The one big idea:** Lesson 6 found that *precision* wins a nominal lap. This
> lesson adds the two things the leaderboard has and the benchmark harness has
> not — the **world** (wind, a payload, a real positioning system) and the
> **walls** (gate frames, poles) — and measures three kinds of tracker against
> both: the MPC family, the course's baseline policy, and a policy trained in
> one run on a racing envelope. The answer has three corners, not two:
> **precision** (the MPC family wins nominal laps), **robustness** (the policies
> win under sensing noise), and **contacts** (which rank none of the fast
> trackers in the real race). A lap time is only a lap time if it survived all
> three.

**Prerequisites:** Lesson 6 (all of it — the two clocks, the planner, the MPC
family, the bridge), Lesson 2 (training), Lesson 3 (the protocol), Lesson 4
(pre-registration). Everything runs on CPU.

---

## 0. What this lesson compares, and on which clock

Eight trackers, three arenas.

The trackers — four model-based, four learned (these are the column names of
every table below and the labels `race_table.py` prints):

| column | what it is | from |
|---|---|---|
| `mpc` | plain MPC | Lesson 6 §3 |
| `mpc_offsetfree` | MPC + velocity ESO (offset-free) | Lesson 6 §3 |
| `mpc_l1` | MPC + L1 adaptation (the hybrid) | Lesson 6 §3 |
| `mppi_l1` | sampling-based MPC + L1 | Lesson 6 §3 |
| `v5_s0` | the course baseline: Lesson 2's `--v5`, 4 M steps, seed 0 | Lesson 2 |
| `racing_s0`, `racing_s1`, `racing_s2` | the same policy trained on a racing envelope, one run per seed | §2 |

The arenas:

| | harness, benchmark clock | harness, leaderboard clock | the real race |
|---|---|---|---|
| tool | `race_eval.py --start hover` | `race_eval.py --start ground` | `compare_models.py --bridge race_bridge_mpc.py` |
| drone at t = 0 | at rest in the air, at the plan's first point | on the ground at the race start | on the ground (default) — or, new in this lesson, at rest in the air (`--start hover`, §1) |
| disturbances, sensing | `--disturbance`, `--sensor` (§3) | same | none beyond Level 0/1's own randomisation |
| contacts | none | none | a gate frame or a pole ends the episode |
| episodes | one deterministic run (three seeds where the condition is random) | same | 20; mean over successes; ≥ 50 % to rank |

Only the third column ranks. The first two are where you learn *why* a tracker
ranks or not — and Lesson 6's rule holds: a number is only comparable to a
number on the same clock.

## 1. The two clocks inside the race

Lesson 6 §0 separated the benchmark clock (hover start, motion onset → last
gate) from the leaderboard's (ground start, t = 0 → last gate), and §4 measured
the takeoff tax in the harness: exactly 1.5 s wherever the takeoff is benign.
What was missing is the benchmark clock *inside the race* — with the race's
contacts and its 50 Hz loop — so that a parent-project number and a race number
can sit side by side on the same clock.

`compare_models.py --start hover` does it. It writes `level0_hoverstart.toml`
next to the level config: identical to `level0.toml` except that
`[[env.track.drones]] pos` puts the drone at rest at z = 1.0 m. Gates, poles,
randomisation, loop rate and contacts are unchanged. The bridge sees the start
height and skips its takeoff leg (`RACE_START=auto`; force one or the other
with `ground` / `hover`).

```bash
cd /workspace
cp tasks/racing/code/race_bridge_mpc.py repos/lsy_drone_racing/lsy_drone_racing/control/
RACE_CRUISE=2.5 /opt/venvs/race/bin/python tasks/racing/code/compare_models.py \
    --bridge race_bridge_mpc.py --start hover --episodes 5 --config level0.toml eso=mpc_offsetfree
```

Two things you will see, both measured (Level 0, `mpc_offsetfree`, the `lsy`
line at cruise 2.5):

1. **The drone sags.** lsy spins the rotors up from rest; released at 1.0 m the
   drone drops about 0.5 m in the first 0.4 s, and a plan that moves off at
   t = 0 reaches gate 1 before the drone has climbed back. Without a hold, one
   lap in five ended on the gate-1 frame. So the bridge holds the plan's first
   point for `RACE_SETTLE` seconds first (default 1.0 s); with the hold, five
   laps of five.
2. **The reported time is the benchmark clock plus the hold:** 5.62 s =
   4.62 s + 1.0 s (logged run: four of five episodes at 5.62–5.64 s, one gate-1
   frame contact — the hold shrinks the sag's toll, it does not remove the
   Level-0 noise). The harness's benchmark clock for the same (plan, tracker)
   is 4.641 s (`race_eval.py --start hover`; §2's table). The 0.02 s is the
   50 Hz loop — the same 0.02 s Lesson 6 §5 found between 6.12 s (race, ground
   start) and 6.141 s (harness, ground start).

`RACE_SETTLE` is a knob, not a law: lower it and watch the gate-1 offset grow.
The benchmark harness never showed the sag because its 1.5 s lead-in was doing
the settling — the same lead-in whose absence made the estimator variants fail
the ground clock in Lesson 6 §4.

> 🔑 Three clocks, one drone: benchmark (hover, harness) → hover inside the race
> (+ settle hold, + contacts) → leaderboard (ground, + takeoff). Every number in
> this lesson names its clock: §2 carries both harness clocks, §3 is the
> leaderboard clock in the harness, §4 is the leaderboard itself.

## 2. A racing tracker in one run

**What the parent project needed.** Its best race tracker (the "RL s0" of
Lesson 6 §4: 4.085 s on the f0.85 ×1.05 tube plan, benchmark clock) is a *body-rate*
policy reached through four acrobatic training recipes in sequence — acro2 →
acro3 → acro4 → 4.1 → 4.2 — each a lesson learned on the previous one, all on
the `force_torque` interface. The race does not accept that interface: only
`state` and `attitude` commands are legal (Lesson 3 §3). So the parent
project's racer cannot be entered, and its recipe cannot be copied.

**The single change.** Open `tasks/racing/code/train_racing.py` — 109 lines,
and the docstring is the specification. `RacingTrackingEnv` subclasses
Lesson 2's v5 environment and overrides one method, `_sample_traj`: the
reference trajectories the policy trains on are drawn from a *racing envelope*,

```
baseline (ppo_train --v5):   |v| up to U(0.5, 3.5) m/s,   |a| up to U(1, 10) m/s^2
train_racing.py:             |v| up to U(1.0, 5.0) m/s,   |a| up to U(3, 15) m/s^2
```

— the envelope the parent project's acrobatic policies already train on
(`datt_env.py`, `_sample_traj`), which the attitude-mode policy never saw. Same
policy class (the asymmetric actor-critic), same 56-number observation, same
PPO settings, same force perturbation and noise randomisation, same chained
quintics (Lesson 2b). The result loads in `race_bridge.py`, in `race_eval.py`
(`datt:<zip>`) and in the MPC bridge unchanged. The tube plans of Lesson 6 peak
at 4.7–5.05 m/s: the baseline was tracking them from *outside* its training
box — Lesson 4 §3 (B1) asked you to check exactly this.

Train three seeds (main venv, from `/workspace`; ~25 minutes each alone on a
14-core CPU, ~50 minutes when two run at once — `docs/4-troubleshooting.md`):

```bash
for S in 0 1 2; do
  python tasks/racing/code/train_racing.py --timesteps 4000000 --seed $S --reason "L7: racing envelope, seed $S"
done
```

Each run lands in `tasks/racing/crazy_track/results/<stamp>_racing-train/datt_ppo_final.zip`.
Then Lesson 6's matrix, both clocks, with your seeds as columns (`--label` names
the column):

```bash
for S in 0 1 2; do
  R=tasks/racing/crazy_track/results/<your-run-s$S>_racing-train/datt_ppo_final.zip
  for START in hover ground; do
    for C in 2.5 3.0; do
      python tasks/racing/code/race_eval.py --plan closed-form --cruise $C --controller datt:$R --label racing_s$S --start $START --reason "L7 nominal"
    done
    for P in tube_f0.85 tube_f0.95; do for X in 1.00 1.05 1.15; do
      python tasks/racing/code/race_eval.py --plan tasks/racing/plans/$P.csv --time-scale $X --controller datt:$R --label racing_s$S --start $START --reason "L7 nominal"
    done; done
    python tasks/racing/code/race_eval.py --plan tasks/racing/plans/raw_f0.95.csv --controller datt:$R --label racing_s$S --start $START --reason "L7 nominal"
  done
done
python tasks/racing/code/race_table.py
```

**Verified on this course's stack** (crazyflow 0.3.2, mujoco 3.10.0, casadi
3.8.0, 2026-09-11; single runs; cell = race time in s (max deviation in m), or
gates/4 when the lap was not completed; the five Lesson-6 columns are Lesson 6
§4's). Benchmark clock first:

| plan | `mpc` | `mpc_offsetfree` | `mpc_l1` | `mppi_l1` | `v5_s0` | `racing_s0` | `racing_s1` | `racing_s2` |
|---|---|---|---|---|---|---|---|---|
| closed-form, cruise 2.5 | 4.657 (0.19) | 4.641 (0.13) | 4.627 (0.21) | 4.627 (0.20) | 4.700 (0.45) | 4.656 (0.38) | 3/4 (0.34) | 4.687 (0.45) |
| closed-form, cruise 3.0 | 4.521 (0.20) | 4.478 (0.25) | 2/4 (1.38) | 4.501 (0.54) | 4.574 (0.48) | 4.534 (0.38) | 3/4 (0.36) | 4.552 (0.47) |
| tube f0.85 ×1.00 | 3.887 (0.46) | 3.876 (0.23) | 3.855 (0.83) | 3/4 (0.37) | 3.961 (0.45) | 3.926 (0.38) | 3/4 (0.46) | 3.911 (0.48) |
| tube f0.85 ×1.05 | 4.077 (0.55) | 4.071 (0.13) | 4.048 (0.58) | 2/4 (0.52) | 4.138 (0.40) | 4.117 (0.38) | 3/4 (0.44) | 4.103 (0.44) |
| tube f0.95 ×1.00 | **3.593** (0.30) | 3.603 (0.35) | 2/4 (0.65) | 2/4 (1.73) | 3/4 (0.52) | **3.648** (0.44) | 3/4 (0.49) | **3.639** (0.54) |
| tube f0.95 ×1.05 | 3.780 (0.24) | 3.775 (0.26) | 3.744 (0.40) | 2/4 (0.69) | 3.844 (0.47) | 3.811 (0.38) | 3/4 (0.47) | 3.796 (0.50) |
| tube f0.95 ×1.15 | 4.118 (0.16) | 4.107 (0.10) | 4.092 (0.13) | 4.092 (0.14) | 4.174 (0.39) | 4.156 (0.38) | 3/4 (0.43) | 4.142 (0.44) |
| raw ×1.00 | 2/4 (0.23) | 1/4 (0.42) | 1/4 (0.27) | 3/4 (0.17) | 1/4 (0.47) | 1/4 (0.43) | 1/4 (0.33) | 1/4 (0.48) |

The same 64 runs on the **leaderboard clock** (`--start ground`, 1.5 s takeoff):

| plan | `mpc` | `mpc_offsetfree` | `mpc_l1` | `mppi_l1` | `v5_s0` | `racing_s0` | `racing_s1` | `racing_s2` |
|---|---|---|---|---|---|---|---|---|
| closed-form, cruise 2.5 | 6.157 (0.19) | 6.141 (0.13) | 3/4 (0.30) | 6.115 (0.27) | 6.200 (0.45) | 6.156 (0.38) | 3/4 (0.34) | 6.187 (0.45) |
| closed-form, cruise 3.0 | 6.021 (0.20) | 3/4 (1.74) | 3/4 (1.05) | 5.992 (0.20) | 6.074 (0.48) | 6.034 (0.38) | 3/4 (0.36) | 6.052 (0.47) |
| tube f0.85 ×1.00 | 5.387 (0.46) | 5.377 (0.23) | 3/4 (0.86) | 5.382 (0.39) | 5.461 (0.45) | 5.426 (0.38) | 3/4 (0.46) | 5.411 (0.48) |
| tube f0.85 ×1.05 | 5.577 (0.55) | 5.570 (0.12) | 5.544 (0.56) | 5.557 (0.15) | 5.638 (0.40) | 5.617 (0.38) | 3/4 (0.44) | 5.603 (0.45) |
| tube f0.95 ×1.00 | 5.093 (0.30) | 5.106 (0.35) | 1/4 (2.77) | 2/4 (1.94) | 3/4 (0.52) | 5.148 (0.44) | 3/4 (0.49) | 5.139 (0.54) |
| tube f0.95 ×1.05 | 5.280 (0.24) | 5.274 (0.26) | 5.245 (0.41) | 3/4 (1.62) | 5.344 (0.47) | 5.311 (0.38) | 3/4 (0.47) | 5.296 (0.50) |
| tube f0.95 ×1.15 | 5.618 (0.16) | 5.607 (0.10) | 5.592 (0.13) | 5.592 (0.15) | 5.674 (0.39) | 5.656 (0.38) | 3/4 (0.43) | 5.642 (0.44) |
| raw ×1.00 | 2/4 (0.23) | 2/4 (0.21) | 2/4 (0.16) | 2/4 (0.21) | 1/4 (0.47) | 1/4 (0.43) | 1/4 (0.33) | 1/4 (0.48) |

Read it in this order:

1. **Two of three seeds fly the unstretched f0.95 plan** — the plan the
   baseline (3/4), the parent project's best acro policy (3/4, Lesson 6 §4)
   and the L1 hybrid (2/4) could not: `racing_s0` 3.648 s and `racing_s2`
   3.639 s against plain MPC's 3.593 s. That is 0.05 s behind the MPC and 0.2 s
   ahead of the baseline's best completed tube lap (3.844 s at ×1.05) — the
   stretch the baseline needed is gone. On every stretched plan the two seeds
   sit 0.02–0.04 s behind the MPC and 0.02–0.05 s ahead of the baseline.
2. **They are not more precise.** Their max deviation (0.38–0.54 m) is the
   baseline's (0.39–0.52 m), not the plain and offset-free MPC's (0.10–0.35 m
   on the f0.95 rows).
   The envelope bought *completing the plan at speed*, not precision. Which of
   the two a gate frame rewards, you will find out in §4.
3. **Seed 1 is the lottery.** Same code, same command line, `--seed 1`: 3/4 on
   every tube and closed-form plan, on both clocks (1/4 on the raw plan, like
   every other tracker), with a deviation envelope no larger than its
   siblings' (0.34–0.49 m). One gate, every time. Lesson 2 §5 in one column —
   a single-seed policy claim is a coin flip, and this is the other face of the
   coin. Which gate it misses, and by how much, is in each run's directory
   (`rollout.png`); look before you retrain.
4. **The ground clock adds 1.500 s and nothing else** for the policies, as for
   the MPC family in Lesson 6 §4: no new failures. The policies carry no
   estimator at all — which, as Lesson 8 §5 shows, is the whole of the
   difference: what the MPC's estimator loses on this clock is not the takeoff
   but its own model error.

**Caveats, before you believe any of it:** 4 M steps and one recipe — no
hyper-parameter search, no curriculum, no second variable; three training
seeds, one of which fails; one evaluation run per nominal cell; a harness
without contacts. The parent project's route to its racer took four recipes;
this one took one because the vendored environment already contained the
envelope. What the environment does *not* contain is the race — §4.

## 3. The conditions, the hypotheses, the matrix

**The conditions.** Five, all from the parent project's figure-8 disturbance
benchmark (`tasks/racing/crazy_track/src/crazy_track/disturbances.py` and
`sensors.py`); `race_eval.py --disturbance` and `--sensor` apply them to a race
lap on either clock. Parameters as the code applies them; the m/s² column is
force ÷ 43.4 g:

| `--disturbance` / `--sensor` | what the code applies | m/s² on the drone | what it mimics |
|---|---|---|---|
| `wind_const` | 0.11 N along +x, constant, at the centre of mass | 2.54 (26 % of weight) | steady wind drag — 72 % of the ±3.5 m/s² per-axis force the policies train under |
| `wind_gust` | 0.08 N mean along +x, plus 0.08 N · sin(2π · 0.7 t) along [1, 0.3, 0], plus Ornstein–Uhlenbeck turbulence (σ ≈ 0.04 N, τ = 0.5 s) on all three axes; seeded | 1.84 mean; ±1.84 (x) / ±0.55 (y) at 0.7 Hz; ±0.92 per axis of turbulence | the partially periodic flow of DATT's fan set-up |
| `payload` | −0.0981 N in z, constant (a 10 g payload); mass and inertia unchanged | 2.26 downward (23 % of weight) | a payload as a thrust bias — beyond the ±1.75 m/s² vertical range the policies train under |
| `lighthouse` (`--sensor`) | position at 34 ± 18 Hz with zero-order hold (interval clipped to 8–100 Hz), 0.7 mm jitter, a 1.5 cm/axis bias drawn once per episode (≈ 2.6 cm in 3D), velocity + N(0, 0.03 m/s), attitude + N(0, 0.5°), gyro + N(0, 0.02 rad/s), one control step of latency on everything but the gyro; seeded | — | the Bitcraze Lighthouse (LH2) deck with the onboard EKF (Taffanel et al., 2021) |
| `wind_const` + `lighthouse` | both at once | | the parent project's *deployment* cell |

Left out on purpose: `ground`, the in-ground-effect thrust gain. It is
0.073 m/s² at z = 0.08 m and 0.0009 m/s² at gate height (0.7 m); the benchmark
had to fly 8 cm above the floor to see it at all, and a race lap is inside its
11 cm reach only for the first fraction of the takeoff.

Two facts about the noisy cells decide how you run them. The gust realisation
and the sensor bias are *seeded* (`--seed`), so one draw is one sample: the
parent project uses 10 evaluation seeds for every gust and Lighthouse cell;
the verified tables below use three, and print those cells as
`mean ± std (k/3)` — over the completed seeds, k completed of three, or `0/3`.
And every Lighthouse measurement upstream is at ≤ 3 m/s: at the tube plan's
speed a 34 Hz hold is 0.14 m of travel per sample, outside anything the sensor
model was validated against. Treat the Lighthouse rows as what the *model*
says, not what the lab says.

**Pre-register** (Lesson 4 §1). These were written before the matrix ran; copy
them, and add the number that would make each one wrong for you:

| # | hypothesis | right if | wrong if |
|---|---|---|---|
| H1 | **precision vs robustness**: the MPC family wins the nominal cells, the policies win the Lighthouse cells | on each plan the fastest completed nominal cell is an MPC variant, *and* under `lighthouse` more policy columns than MPC-family columns complete 3/3 | an MPC variant completes 3/3 under Lighthouse on the tube plan while a policy does not |
| H2 | the racing envelope also buys **gust** robustness | `racing_s0` and `racing_s2` each complete more `wind_gust` seeds than `v5_s0` on both plans | either seed completes fewer than the baseline on either plan |
| H3 | the L1 hybrid **never ranks** | `mpc_l1` is unranked (< 10/20) on every race row and completes no Lighthouse cell | it ranks anywhere, or completes 3/3 under Lighthouse |
| Kill | a single-run racing policy must **rank in the race** or it is not a racing tracker | any racing seed ≥ 10/20 on any 20-episode row of §4 | 0/20 everywhere |

**Run the matrix** (main venv; leaderboard clock; the two plans that bracket the
speed range — the closed-form line at cruise 2.5 and the ×1.05 tube plan that
every tracker completes nominally; an MPC cell takes 20–30 s, a policy cell
about 5 s; every `datt:` spec needs its own `--label`, or `race_table.py` files
both policies under one `datt` column and keeps only the latest run):

```bash
B=tasks/racing/crazy_track/results/<your-baseline>_datt-train/datt_ppo_final.zip
R=tasks/racing/crazy_track/results/<your-run>_racing-train/datt_ppo_final.zip
run() { python tasks/racing/code/race_eval.py --start ground --reason "L7 matrix" "$@"; }
cf="--plan closed-form --cruise 2.5"
tube="--plan tasks/racing/plans/tube_f0.95.csv --time-scale 1.05"
for C in mpc mpc_offsetfree mpc_l1 mppi_l1 "datt:$B --label v5_mine" "datt:$R --label racing_mine"; do
  for PLAN in "$cf" "$tube"; do
    run $PLAN --controller $C --disturbance wind_const
    run $PLAN --controller $C --disturbance payload
    for S in 0 1 2; do
      run $PLAN --controller $C --disturbance wind_gust --seed $S
      run $PLAN --controller $C --sensor lighthouse --seed $S
      run $PLAN --controller $C --disturbance wind_const --sensor lighthouse --seed $S
    done
  done
done
python tasks/racing/code/race_table.py --start ground
```

**Verified (2026-09-11)** — leaderboard clock in the harness; single-run cells as
in §2, seeded cells as `mean ± std (completed/3)`.

Closed-form line, cruise 2.5:

| condition | `mpc` | `mpc_offsetfree` | `mpc_l1` | `mppi_l1` | `v5_s0` | `racing_s0` | `racing_s1` | `racing_s2` |
|---|---|---|---|---|---|---|---|---|
| nominal | 6.157 (0.19) | 6.141 (0.13) | 3/4 (0.30) | **6.115** (0.27) | 6.200 (0.45) | 6.156 (0.38) | 3/4 (0.34) | 6.187 (0.45) |
| `wind_const` | 6.073 (0.35) | 6.140 (0.14) | 3/4 (0.53) | 6.135 (0.26) | 6.196 (0.39) | 3/4 (0.33) | 3/4 (0.36) | 6.207 (0.40) |
| `payload` | 6.151 (0.24) | 6.131 (0.12) | 3/4 (0.57) | 6.119 (0.14) | 6.212 (0.44) | 6.164 (0.47) | 6.129 (0.39) | 6.187 (0.46) |
| `wind_gust` | **6.103 ± 0.007 (3/3)** | 6.115 ± 0.002 (2/3) | 6.108 (1/3) | 6.095 ± 0.006 (2/3) | 6.188 ± 0.007 (2/3) | 6.141 (1/3) | 0/3 | **6.197 ± 0.014 (3/3)** |
| `lighthouse` | **6.128 ± 0.009 (3/3)** | 6.094 ± 0.001 (2/3) | 0/3 | **6.102 ± 0.017 (3/3)** | **6.183 ± 0.002 (3/3)** | **6.124 ± 0.008 (3/3)** | 0/3 | **6.167 ± 0.002 (3/3)** |
| `wind_const` + `lighthouse` | 6.048 (1/3) | 6.101 ± 0.004 (2/3) | 6.085 (1/3) | 6.129 ± 0.002 (2/3) | **6.163 ± 0.008 (3/3)** | 6.111 ± 0.011 (2/3) | 0/3 | **6.180 ± 0.006 (3/3)** |

Tube plan f0.95 ×1.05 (top speed 5.05 m/s ÷ 1.05 ≈ 4.8 m/s):

| condition | `mpc` | `mpc_offsetfree` | `mpc_l1` | `mppi_l1` | `v5_s0` | `racing_s0` | `racing_s1` | `racing_s2` |
|---|---|---|---|---|---|---|---|---|
| nominal | 5.280 (0.24) | 5.274 (0.26) | **5.245** (0.41) | 3/4 (1.62) | 5.344 (0.47) | 5.311 (0.38) | 3/4 (0.47) | 5.296 (0.50) |
| `wind_const` | 2/4 (1.29) | 2/4 (1.46) | 3/4 (0.87) | 5.272 (0.58) | 5.319 (0.40) | 5.311 (0.52) | 3/4 (0.49) | 5.299 (0.57) |
| `payload` | 1/4 (0.59) | 3/4 (0.40) | 1/4 (0.68) | 1/4 (3.04) | 5.329 (0.47) | 5.301 (0.50) | 5.245 (0.50) | 5.286 (0.54) |
| `wind_gust` | 0/3 | 0/3 | 0/3 | 0/3 | 0/3 | **5.338 ± 0.005 (3/3)** | 5.327 (1/3) | **5.318 ± 0.010 (3/3)** |
| `lighthouse` | 0/3 | 0/3 | 0/3 | 0/3 | **5.343 ± 0.009 (3/3)** | **5.311 ± 0.007 (3/3)** | 0/3 | **5.306 ± 0.010 (3/3)** |
| `wind_const` + `lighthouse` | 0/3 | 0/3 | 0/3 | 0/3 | 5.318 (2/3) | 5.290 (1/3) | 0/3 | **5.287 ± 0.000 (3/3)** |

**Which tracker is best where** — read the tables cell by cell, then check these
against them:

- **Nominal: the MPC family**, on both plans — by 0.04–0.05 s over the best
  racing seed and 0.09–0.10 s over the baseline — and on the tube plan the
  hybrid is the fastest single cell, 5.245 s. Precision is what a nominal lap
  pays for (Lesson 6 §6).
- **Lighthouse: the policies.** On the tube plan no MPC variant completes a
  single seed (0/3 in four columns) while `racing_s0`, `racing_s2` and the
  baseline complete every seed, within 0.01 s of their nominal times. On the
  slower closed-form line the family mostly survives (`mpc` 3/3, `mppi_l1` 3/3,
  `mpc_offsetfree` 2/3, the hybrid 0/3): the noise penalty grows with speed.
- **Gusts: the racing seeds on the fast plan** (`racing_s0` 3/3, `racing_s2`
  3/3; the baseline 0/3; every MPC variant 0/3) — and plain MPC on the slow one
  (3/3). H2's "both plans" fails on the closed-form line: `racing_s0` 1/3,
  `racing_s2` 3/3, the baseline 2/3 — two seeds of the same recipe further
  apart than either is from the baseline.
- **Payload: every policy** on the tube plan (four of four, seed 1 included),
  no MPC variant (the best, `mpc_offsetfree`, 3/4); on the closed-form line
  everyone but the hybrid.
- **Steady wind: `mppi_l1`** is the only model-based tracker that completes the
  tube plan (5.272 s); `racing_s0`, `racing_s2` and the baseline complete it
  too. On the closed-form line the family completes (the hybrid excepted) and
  `racing_s0`, `racing_s1` lose a gate.
- **The deployment cell (wind + Lighthouse): `racing_s2`**, 3/3 on both plans;
  the baseline 3/3 and 2/3; `racing_s0` 2/3 and 1/3; the MPC family 0/3 on the
  tube plan and 1–2/3 on the line.

**Why** — each sentence traces to a measurement of the parent project (its
figure-8 disturbance benchmark, RMSE in metres) or to a row above:

1. **Plain MPC has no state for a standing force.** Under `wind_const` it
   re-predicts the same biased trajectory every step — its figure-8 wind cell
   is the worst in the pool (0.196 m). On the closed-form line its feedback
   absorbs 2.5 m/s² (6.073 s, 4/4); on the tube plan, planned at 0.95 of the
   thrust limit, it does not (2/4, a 1.29 m excursion).
2. **The estimator variants do not rescue the plan they were built for.** On
   the figure-8 the offset-free MPC ties the deployment cell (0.057 m), yet
   here it does *not* rescue the tube plan under wind (2/4) or payload (3/4),
   although both are the constant forces it was built for. Lesson 6 §4 blamed
   the 1.5 s soft-start ramp opening during the climb; Lesson 8 §5 tested that
   and found it innocent (gating the ramp on lift-off: 0/10 either way). The
   estimate the optimiser receives is dominated by the **prediction model's own
   error** — an attitude gain of 0.73 against the simulator's 0.94 — learned in
   one turn as a world-frame force and applied in the next. An estimator is only
   as good as the model it corrects.
3. **Noise enters an optimiser as a phantom disturbance.** On the figure-8 the
   MPC family's Lighthouse failures are ipopt transients on a noisy,
   zero-order-held position, not the latency (plain MPC 0.136 ± 0.057 m over
   10 seeds, range 0.060–0.230); at 4.8 m/s each 34 Hz hold is 0.14 m of travel
   and the transient is a missed gate. The L1 hybrid is the worst case: its
   fast law (≈ 25 rad/s effective) passes the noise straight into the horizon —
   the bandwidth dilemma the parent project measured on ADRC (low bandwidth
   wins the noise cells, high bandwidth wins the gusts), on the L1 axis.
4. **The policies were trained for this.** v5 puts noisy observations in front
   of the actor with a per-episode noise scale drawn from U(0, 1.5) times the
   Lighthouse model, a privileged critic, and a ±3.5 m/s² per-axis random force
   every episode with an L1 estimate of it in the observation. On the figure-8
   that made v5 the policy that ties the deployment cell (0.059 ± 0.002 m over
   three training seeds); on a race lap it makes the same policy complete under
   Lighthouse where every optimiser fails. Nothing was added for this lesson —
   the racing envelope changed the references, not the robustness recipe.
5. **What the envelope bought is speed under noise, not a new mechanism.**
   Under Lighthouse the racing seeds and the baseline all complete, at 5.31 /
   5.31 / 5.34 s; under gusts on the fast plan only the racing seeds do. And
   the parent project's crossover warning applies: the best-precision
   checkpoint was not the most robust one, twice. Select against the race
   criterion (§4), not against the matrix.

**Verdicts:** H1 ✓ (nominal: `mppi_l1` 6.115 s and `mpc_l1` 5.245 s are the
fastest cells; Lighthouse on the tube: three policy columns at 3/3, the MPC
family none). H2 half — ✓ on the tube plan (3/3, 3/3 against 0/3), ✗ on the
closed-form line (1/3, 3/3 against 2/3). H3 ✓ (0/3 under Lighthouse on both
plans; unranked on every race row of Lesson 6 §5: 0/20, 0/20, 8/20, 4/20) —
with Lesson 6's twist intact: nominally it is the fastest tracker on the tube
plan. Kill: §4.

## 4. The same trackers in the real race

Lesson 6 ended on two facts: every TOGT tube plan ends on a pole in the race
(the corridors of gates 1, 3 and 4 stand on poles 1, 3 and 4), and the best
ranked number of the course, 7.14 s, spent about a second going around poles
the planner never heard of. This lesson tried both ways out — tell the planner
about the poles, and take the poles away — and raced five trackers on each.

### 4.1 A pole-aware tube

`tasks/racing/code/togt/lsy_level2_tube_poles.yaml` (read its header) keeps
upstream's tube where it is harmless and replaces the three corridors that
stood on poles by a `SingleBall` via *beside* each pole — south-east of pole 1,
south of poles 3 and 4, radius 0.15 m — so that the time-optimal path has to
pass the pole with room to spare. It was found by a search over four strategies
(shorter corridors, laterally offset corridors, ball vias, free placement) with
`togt_plan.py --track-yaml`, `race_eval.py` and one race episode per candidate;
the runner-up (free placement) tied its race time and lost on minimum
clearance (0.18 m).

```bash
python tasks/racing/code/togt_plan.py --track poles --thrust-frac 0.85     # -> tasks/racing/plans/poles_f0.85.csv
python tasks/racing/code/togt_plan.py --track poles --thrust-frac 0.95
```

What the diagnostics say (upstream's tube in brackets):

| | 0.85 × TWR | 0.95 × TWR |
|---|---|---|
| planned lap (motion onset → last gate) | **3.830 s** (3.867) | 3.527 s (3.566) |
| top speed | 4.47 m/s | 5.03 m/s |
| crossing angles G1..G4 | 10 / 5 / 16 / 17° (2–7°) | 11 / 7 / 15 / 17° |
| gate-plane crossings through a frame zone | none (one: gate 3 at 0.41 m) | none |
| clearance from the pole surfaces, poles 1..4 | 0.21 / 0.33 / 0.30 / 0.20 m (0.06 / 0.31 / 0.02 / 0.03) | 0.22 / 0.32 / 0.30 / 0.20 m |

Two things to notice. The pole-aware plan is 0.04 s *shorter* than upstream's
tube — a via is a weaker constraint than a corridor — and its crossings are
more oblique, up to 17° at gate 4: a pole 0.5 m in front of a gate on its axis
forbids a long straight approach, so the pole-4 clearance and the gate-4 angle
trade off against each other.

Then race it, one episode each, with the harness twin next to it
(`race_eval.py --start ground`; Level 0):

| plan | `mpc` harness | `mpc` race | `mpc_offsetfree` harness | `mpc_offsetfree` race |
|---|---|---|---|---|
| 0.85 × TWR, ×1.00 | 5.333 (0.319) | **4/4, 5.32 s** | 5.332 (0.261) | 2/4 (3.80 s) |
| 0.85 × TWR, ×1.05 | 5.529 (0.365) | 2/4 (3.78 s) | 5.522 (0.273) | 2/4 (4.54 s) |
| 0.85 × TWR, ×1.15 | 5.915 (0.153) | 4/4, 5.90 s | 5.912 (0.080) | 4/4, 5.90 s |
| 0.95 × TWR, ×1.00 | 5.031 (0.219) | 2/4 (4.26 s) | 3/4 (0.348) | 2/4 (4.24 s) |
| 0.95 × TWR, ×1.05 | 5.204 (0.519) | 1/4 (3.16 s) | not run | not run |

The plan clears every pole by ≥ 0.20 m on paper, and the race still ends on
pole 1 for most (thrust, stretch, tracker) combinations. The logged replay of
the ×1.05 `mpc` episode ends at t = 3.82 s, 0.083 m from pole 1's surface on
the gate-2 → gate-3 leg, where the *reference* passes 0.253 m from it: the
tracker cuts 0.17 m inside the plan on that leg, into the ≈ 0.10 m at which lsy
registers a contact (Lesson 6 §5). Plain MPC at 0.85 × TWR, unstretched, does
not cut there, and that combination is the plan of the protocol below; the ESO
variant needs ×1.15.

> 🔑 Reference clearance is necessary, not sufficient. What touches the pole is
> the *flown* path. Check clearance on the flown CSV (`RACE_LOG_DIR`,
> `plot_trajectory.py --flown`, `race_refs.obstacle_clearance`) before you
> believe a plan — and remember that a Level-0 episode is not deterministic
> (Lesson 6 §5): one race episode per candidate is a screen, the 20-episode
> protocol is the verdict.

### 4.2 The protocol — five trackers, two levels, three plans

The five that matter: `mpc`, `mpc_offsetfree`, the baseline `v5_s0`, and the
two racing seeds that fly the fast plans in the harness (`racing_s0`,
`racing_s2`). Race venv, from `/workspace`; the MPC bridge takes a policy as
`datt:<zip>`, and `--no-poles` writes `level0_nopoles.toml` with the four
poles moved to the arena corners — lsy expects the obstacle list to keep its
length — while the gate frames stay:

```bash
cp tasks/racing/code/race_bridge_mpc.py repos/lsy_drone_racing/lsy_drone_racing/control/
B=/workspace/tasks/racing/crazy_track/results/<your-baseline>_datt-train/datt_ppo_final.zip
R0=/workspace/tasks/racing/crazy_track/results/<your-run-s0>_racing-train/datt_ppo_final.zip
R2=/workspace/tasks/racing/crazy_track/results/<your-run-s2>_racing-train/datt_ppo_final.zip
T="mpc=mpc eso=mpc_offsetfree v5_s0=datt:$B racing_s0=datt:$R0 racing_s2=datt:$R2"
for L in level0 level1; do
  # A: the pole-aware tube, with the poles
  RACE_PLAN=/workspace/tasks/racing/plans/poles_f0.85.csv /opt/venvs/race/bin/python tasks/racing/code/compare_models.py \
      --bridge race_bridge_mpc.py --episodes 20 --config $L.toml --out-dir tasks/racing/figures/A_$L $T
  # C: the same plan, the poles moved out of the way
  RACE_PLAN=/workspace/tasks/racing/plans/poles_f0.85.csv /opt/venvs/race/bin/python tasks/racing/code/compare_models.py \
      --bridge race_bridge_mpc.py --episodes 20 --config $L.toml --no-poles --out-dir tasks/racing/figures/C_$L $T
  # B: upstream's fast tube, the poles moved out of the way
  RACE_PLAN=/workspace/tasks/racing/plans/tube_f0.95.csv RACE_TIME_SCALE=1.05 /opt/venvs/race/bin/python tasks/racing/code/compare_models.py \
      --bridge race_bridge_mpc.py --episodes 20 --config $L.toml --no-poles --out-dir tasks/racing/figures/B_$L $T
done
```

**Verified (2026-09-11; 20 episodes per cell; mean ± std over the successful
episodes, successes/20; below 50 % unranked):**

| plan | poles | level | `mpc` | `mpc_offsetfree` | `v5_s0` | `racing_s0` | `racing_s2` |
|---|---|---|---|---|---|---|---|
| A: pole-aware tube, 0.85 × TWR, ×1.00 | yes | 0 | **5.320 ± 0.000 s, 15/20** | 0/20 | 0/20 | 0/20 | 0/20 |
| A | yes | 1 | 5.320 ± 0.000 s, 8/20 (unranked) | 0/20 | 0/20 | 0/20 | 0/20 |
| C: the same plan | no | 0 | **5.320 ± 0.000 s, 13/20** | 0/20 | 0/20 | 0/20 | 0/20 |
| C | no | 1 | **5.320 ± 0.000 s, 11/20** | 0/20 | 0/20 | 0/20 | 0/20 |
| B: upstream's tube f0.95 ×1.05 | no | 0 | 0/20 | 0/20 | 0/20 | 0/20 | 0/20 |
| B | no | 1 | 0/20 | 5.280 s, 1/20 (unranked) | 0/20 | 0/20 | 0/20 |

And the slow lines, with the poles, all six trackers — the lines Lesson 6
ranked its MPCs on (Lesson-6 MPC cells quoted; `RACE_TAKEOFF_Z=0.7
RACE_CRUISE=1.5` is the Lesson-3 baseline's geometry, the line starting at
gate 1's height):

| line | level | `mpc` | `mpc_offsetfree` | `v5_s0` | `racing_s0` | `racing_s1` | `racing_s2` |
|---|---|---|---|---|---|---|---|
| `lsy` line from z 0.7, cruise 1.5 | 0 | 0/20 | 0/20 | **8.099 ± 0.004 s, 20/20** | 0/20 | 0/20 | 0/20 |
| same | 1 | 0/20 | 0/20 | **8.086 ± 0.042 s, 17/20** | 0/20 | 0/20 | 0/20 |
| `lsy` line from z 0.7, cruise 2.0 | 0 | 0/20 | 0/20 | 0/20 | 0/20 | 0/20 | 0/20 |
| same | 1 | 0/20 | 0/20 | 6.840 s, 1/20 | 0/20 | 0/20 | 0/20 |
| `safe` line, cruise 2.0 | 0 / 1 | 7.180 / 7.182 s, 20/20 (Lesson 6) | **7.140 / 7.139 s, 20/20** (Lesson 6) | 0/20 | 0/20 | 0/20 | 0/20 |
| `lsy` line, cruise 2.5 | 0 / 1 | 0/20 (Lesson 6) | 6.122 s 18/20 / 6.124 s 16/20 (Lesson 6) | 0/20 | 0/20 | 0/20 | 0/20 |

Read the two tables together:

1. **The pole-aware plan works — for plain MPC.** 5.32 s at 15/20 on Level 0 is
   the fastest ranked lap of this course, 1.8 s under Lesson 6's 7.14 s, and
   every one of its 15 successes reads 5.32 s: the failures are contacts, not
   slow laps. Level 1 takes it to 8/20 (unranked) with the poles and 11/20
   without — and the term that does it is the **mass**, not the start pose:
   Lesson 8 §5 logs the sampled mass per episode and finds the start offset
   uncorrelated with the outcome (r = −0.01 / +0.05) while the heaviest drones
   cross gate 2 0.10–0.25 m low (a thrust-scale adaptation on a corrected model
   takes this same cell to 18/20). Note the reversal since
   Lesson 6: on the `safe` line the ESO variant was 0.04 s faster and equally
   ranked; on the fast plan it never completes — §3's mechanism 2, now with
   contacts.
2. **The poles are not what stops the policies.** Protocol C removes them and
   changes nothing for `mpc_offsetfree`, `v5_s0`, `racing_s0`, `racing_s2`:
   0/20 stays 0/20. What stops them is a gate frame — the replays below.
3. **Upstream's fast tube fails even without poles** (Protocol B: 0/20 for
   everyone at Level 0, one ESO lap at Level 1). A logged replay of the ESO
   variant passes gates 1–3 with 1–9 cm of offset and ends 0.12 m short of
   gate 4, 0.24 m off its opening and 0.24 m low: a **gate-4 frame** contact at
   4.8 m/s. Lesson 6 §2's warning about this plan — a second crossing of gate
   3's plane 0.41 m from its centre — is a separate risk that this episode
   happened to survive. The harness has no contacts — which is why the same
   plan "works" in every harness table of Lessons 6 and 7, at 5.28 s.
4. **On the closed-form lines the racing policies are pole-limited**, not
   gate-limited: in `RACE_LOG_DIR` replays they pass gate 1 with 5–8 cm of
   offset and then touch pole 2 at (1.0, 0.25) on the leg from gate 1 to the
   first via, where the reference passes 0.19 m from the pole and the policy
   deviates about 0.2 m towards it. The baseline at cruise 1.5 clears it by
   being slower and less faithful — the mechanism Lesson 6 §5 found for plain
   MPC on pole 4, on another pole.
5. **The Kill criterion fired.** No racing seed ranks on any row: 0/20 on the
   pole-aware tube with and without poles, on upstream's tube, and on every
   closed-form line at every cruise. The fastest and most noise-robust trackers
   of §3 are not racing trackers on the leaderboard's own protocol.

**The replays** (Level 0, the pole-aware plan with the poles, one logged
episode each — `RACE_LOG_DIR`, then `plot_trajectory.py --flown`; gate offsets
in the gate plane; distances at the end of the episode):

| tracker | gate 1 | gate 2 | episode ends | what it hit | max deviation |
|---|---|---|---|---|---|
| `racing_s2` | pass, 0.079 m | pass, 0.092 m | t = 4.52 s, 0.24 m from gate 3's centre | **gate-3 frame** | 0.53 m at t = 2.80 s |
| `mpc_offsetfree` | pass, 0.063 m | pass, 0.044 m | t = 3.80 s, 0.10 m from pole 1's surface, on the gate-2 → gate-3 leg | **pole 1** (it cuts inside the plan; plain MPC does not) | 0.30 m at t = 3.80 s |
| `v5_s0` | — | — | t = 2.30 s, 0.39 m from gate 1's centre | **gate-1 frame**, on the approach | 0.33 m at t = 2.30 s |

A gate's opening is 0.4 m wide and its frame 0.72 m: 0.24 m from the centre is
inside the frame, 0.39 m is its outer edge. The racing policy's 0.53 m of
deviation on a plan whose top speed is 4.47 m/s is harmless in the harness and
a frame in the race; the baseline's 0.33 m reaches gate 1 in the wrong place;
the ESO variant, precise at the gates (4–6 cm), leaves the plan between them.

### 4.3 Which number, then

On the leaderboard's own clock and protocol, this course's ranked results are:

| | plan / tracker | Level 0 | Level 1 |
|---|---|---|---|
| fastest ranked lap | pole-aware TOGT tube (0.85 × TWR, unstretched), **plain MPC** — not a policy | **5.320 s, 15/20** | 8/20, unranked |
| ranked at both levels, model-based | pole-safe closed-form line, cruise 2.0, **offset-free MPC** (Lesson 6) | 7.140 s, 20/20 | 7.139 s, 20/20 |
| ranked policy | Lesson-3 line (z 0.7, cruise 1.5), **the baseline v5 policy** | 8.099 s, 20/20 | 8.086 s, 17/20 |

The racing-envelope policies — the fastest learned trackers of §2 and the most
noise-robust trackers of §3 — do not appear. That is the lesson's central
result, and it is not a failure of the measurement.

## 5. Read the result honestly

Five mechanisms, each one a sentence you can defend with a row:

1. **Precision wins nominal laps; noise robustness wins noisy ones; contacts
   decide the race.** The MPC family is fastest in every nominal harness cell
   (§2); the policies are the only completions under Lighthouse and gusts on
   the fast plan (§3); plain MPC is the only ranked tracker on the fast plan in
   the race, at 15/20 (§4). No tracker in this lesson holds two corners.
2. **A 0.4–0.5 m deviation at 4–5 m/s is a frame.** The racing seeds fly the
   unstretched f0.95 plan in the harness with 0.44–0.54 m of deviation and
   finish; in the race the same deviation meets gate 3 (0.24 m from the centre)
   or pole 2. The harness never told you, because it has no walls: read the
   *max deviation* column against the 0.2 m half-opening, not against
   "completed".
3. **An estimator is a liability when it is correcting the wrong model**
   (Lesson 6 §4, twice more here): the ESO variant loses the tube plan under
   wind and payload in the harness and cuts inside the plan at pole 1 in the
   race; the L1 hybrid never ranks. Lesson 8 §5 re-measures this with the ramp
   gated on lift-off and on a corrected model, and names the mechanism: an
   additive disturbance estimate is the wrong *structure* for a model error that
   scales with tilt and thrust.
4. **Reference clearance is not flown clearance.** 0.25 m on the plan became
   0.083 m on the flown path for a tracker that cut the leg; the race registers
   a contact at ≈ 0.10 m. Check the flown CSV.
5. **The single-run envelope closed the harness gap to the model-based trackers
   (0.05 s on the unstretched plan) and none of the race gap.** Speed is a
   training-distribution question; ranking is a precision-and-contacts
   question. The parent project's phased recipe reached a body-rate racer the
   race does not accept; this one reached an attitude-mode tracker the race
   does not rank. Both are results.

**Caveats you must write down with any number from this lesson:** single
deterministic runs on the harness, and three seeds where the condition is
random (the parent project uses ten); three training seeds, one of which fails
every nominal plan; one recipe, no hyper-parameter search; the harness has no
contacts and no randomisation; the Lighthouse model is validated at ≤ 3 m/s
and applied here at 4.8; the MPC is not real-time (20–30 ms per 20 ms step,
Lesson 6 §5), so 5.32 s is a laptop's number, not a drone's; Level 0 keeps a
0.001 rad action noise (Lesson 6 §5), which is why a plan flown within a few
centimetres of the contact distance at pole 1 (§4.1) is 15/20 and not 20/20 —
every failure a contact, never a slow lap; and the 8.10 s baseline row is `lsy_level2_race()`'s line
through `race_bridge_mpc.py` from z 0.7 at cruise 1.5 — the Lesson-3 geometry,
not necessarily your own bridge's reference.

---

## 🛠️ Before you move on

1. **The verdict table.** H1–H3 and the Kill with your thresholds, your
   numbers, your verdicts — the pre-registered wording of §3, not a re-reading
   of it after the tables.
2. **Rerun the matrix with your own seeds.** Your three `train_racing.py`
   seeds in the columns, and evaluation seeds 3, 4 and 5 in the noisy cells
   (one `--seed` per run; the verified tables used 0–2; three new draws tell you whether `racing_s0`'s 1/3 gust
   cell on the closed-form line is the draw or the policy):
   ```bash
   for S in 3 4 5; do
     python tasks/racing/code/race_eval.py --plan closed-form --cruise 2.5 --controller datt:$R --label racing_mine --start ground --disturbance wind_gust --seed $S --reason "L7 my seeds"
   done
   python tasks/racing/code/race_table.py --start ground --cond wind_gust
   ```
3. **One variable.** Three follow-ups that each change one flag of
   `train_racing.py`, each with a prediction written down first:
   `--no-perturb` (do the payload and wind cells come from the force
   perturbation or from the envelope?), `--vel-max 4` (does a narrower envelope
   trade the unstretched f0.95 plan for a smaller deviation — and does a smaller
   deviation rank?), `--seg-min 0.6` (more direction changes per second: does it
   help the 17° crossing at gate 4?). Three seeds each, the §3 matrix, and at
   least the Protocol-A row.
4. **The race-legal tube, your way.** Either beat `lsy_level2_tube_poles.yaml`:
   copy it, move a via or shrink a ball, plan it with `togt_plan.py --track-yaml
   <your file>`, race it (the search that found it scored every candidate on
   planned lap × race episode × minimum *flown* clearance — the runner-up lost
   on the last one). Or take the other road, `compare_models.py --no-poles`,
   and find the fastest plan that ranks a *policy* without poles. Protocol C
   says the frames stop it first — which gate, at which deviation, and what
   would the plan have to give up?
5. **What you would do next, and why** — one paragraph. The evidence points
   three ways: precision (train against the flown deviation, not only the
   reference), contacts (a planner that knows the frames and the flown
   clearance, not only the gates), or the model the tracker predicts with.
   Write your answer before you read Lesson 8, which takes the third road and
   measures the other two on the way — including the fourth idea that used to
   stand here, a soft-start gated on motion onset: it gains nothing (§5 there).
   Pick one, say which corner of §5's triangle it moves, and name the number
   that would tell you it did.

**Back to:** [Lesson 4 — Brainstorm: make it faster](04-brainstorm-faster-tracking.md)
— with the whole board measured: plan, tracker, clock, world, walls.
