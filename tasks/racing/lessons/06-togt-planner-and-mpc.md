# Lesson 6 — Plan faster, track tighter: the TOGT planner and MPC

⏱️ ~3 hours of reading and running, plus about an hour of compute in the
background. **You finish when** you have (a) the (plan × tracker) table of §4
on *both* clocks, (b) a verdict on each of the three pre-registered hypotheses,
and (c) one number from the real race environment (§5) next to its benchmark
twin, with the difference explained.

> **The one big idea:** lap time = **plan** + **tracking**. Lesson 4 could only
> *name* those two levers; this lesson hands you both — a time-optimal planner
> and a model-predictive tracker — and lets you measure two things the parent
> project found the hard way: a better plan only pays when the tracker can cash
> it, and "better tracker" means *precise*, not *adaptive*, on a nominal track.
> Plus one rule of measurement you will never again get wrong: **a lap time is
> only comparable to another lap time on the same clock.**

**Prerequisites:** Lesson 2 (a trained policy of yours), 2b (the planner
landscape), 3 (the bridge and the race protocol), 5 (the plotting tools).
Everything here runs on CPU; the MPC is slow (§3), so start the long runs
before you read on.

---

## 0. Two clocks — read this twice

The parent project's race numbers and the leaderboard's are **not on the same
clock**:

| | benchmark clock (crazy_track's `freestyle_eval` / `togt_race_eval`) | leaderboard clock (lsy `scripts/sim.py`, `evaluate.py`) |
|---|---|---|
| drone at t = 0 | at rest **in the air**, at the plan's first point (−1.5, 0.75, 1.0) | **on the ground** at (−1.5, 0.75, 0.01) — `[[env.track.drones]]` in every level toml |
| reference before the lap | holds still for a 1.5 s lead-in | none: the clock is already running |
| `race_time` | motion onset → last-gate crossing | t = 0 → last-gate crossing, **takeoff included** (Lesson 3 §3) |
| episodes | one deterministic rollout | 20 randomised episodes; mean over successes; ≥ 50 % must succeed |
| what ends a lap early | nothing — gates and poles have no collision geometry there | contact with a gate frame or an obstacle pole |
| controller rate | 100 Hz | 50 Hz |

Every race number in the parent project's reports — 4.464 s (RL tracker on the
closed-form plan), 4.085 s (RL tracker on a TOGT tube plan), 3.592 s (MPC on
the same plan) — is on the **benchmark clock**. Never put one of them next to
3.394 s. The parent project knew (`lsy_level2_race()`'s docstring,
`tasks/racing/crazy_track/src/crazy_track/trajectories/freestyle.py:409`, says
so in one sentence); knowing is not measuring. This lesson's tools make the
distinction operational:

- `race_eval.py --start hover` reproduces the benchmark clock exactly;
- `race_eval.py --start ground` prepends a takeoff from the race start pose
  and starts the clock at t = 0 (`race_refs.py: GroundStartTrajectory`;
  `plot_trajectory.py --ground-start` draws it). Same physics harness, so the
  *difference* between the two numbers is the takeoff tax and nothing else;
- §5 runs the same stack inside the real race environment, whose clock *is* the
  leaderboard's, with its randomisation and its collisions.

## 1. What a time-optimal planner does — and why its plans are untrackable

Lesson 2b's landscape ended with the time-optimal methods that beat human
pilots. [TOGT-Planner](https://github.com/FSC-Lab/TOGT-Planner) (Qin et al.,
ICRA 2024) is one you can run in seconds: each gate becomes a *corridor* (a
rectangle the path must cross), the path between corridors is a chain of
polynomial pieces (the MINCO representation — the same family as our quintics,
optimised jointly in shape and timing), and the optimiser (L-BFGS) minimises
**time** subject to penalties on thrust, body rate and tilt. Its parameter sets
and our track files live in `tasks/racing/crazy_track/configs/togt/`; read its
`README.md` — every quirk in it cost the parent project an hour.

On our four gates the answer is a **2.59 s** plan against the 4.49 s floor of
the closed-form line (Lesson 2b §2): 42 % shorter. Three things to know before
you touch it:

1. **It spends every margin it is given.** The crossings sit exactly at the
   edge of the allowed window (0.13 m of a 0.13 m window — `togt_plan.py`
   prints the offsets), the thrust sits at its bound, and the crossings are
   *oblique*: gate 1 at 45°, gate 3 at **86°** to the gate normal. That single
   number is the whole story of "untrackable": a tracker's error along the
   path is harmless, an error *in the gate plane* is a miss, and
   lag × in-plane speed lands in the gate plane. With ~0.2 m of tracking error
   the raw plan scores 0–1 of 4 gates for every tracker the parent project
   owns, and stretching it ×2 in time does not help — the geometry is wrong,
   not the speed.
2. **The tube trick.** `configs/togt/lsy_level2_tube.yaml` brackets each gate
   with an entry and an exit corridor 0.6 m along its normal (±0.05 m
   windows). The optimiser now *has* to cross within a few degrees of the
   normal. Planned laps: 3.57 s at 0.95 × TWR, 3.87 s at 0.85 × TWR; crossings
   at 2–7°. A second planner-side knob is the **time stretch** (`--time-scale`,
   `SampledRaceTrajectory` in `src/crazy_track/trajectories/sampled.py:25`):
   same path, velocities ÷ s, accelerations ÷ s² — a feasible plan stays
   feasible, and it is the cleanest "how much slower must it be" dial.
3. **The scale trick** — why `configs/togt/cf21b_togt/quad.yaml` says the
   drone weighs 1 kg. TOGT's thrust penalty is evaluated in absolute newtons;
   for a 43 g drone it is ~530× too weak and the solver ignores the thrust
   bounds. Multiplying mass, inertia and thrust bounds by k = 23.05 leaves
   every acceleration, body rate and trajectory unchanged (thrust = m·(a + g))
   and puts the penalty where upstream tuned it. Read the comment header of
   that file; it is a nice example of dimensional reasoning doing real work.

And one thing the parent project's evaluators never checked, which you will
in §2: **TOGT knows the gates, not the poles.** Its benchmark harness has no
collision geometry; the race has.

## 2. Build it, plan a lap, read the diagnostics

Build the planner once (in the container; ~2 minutes, no sudo — it clones
`repos/TOGT-Planner` at the commit pinned in `scripts/pins.sh`, fetches a cmake
wheel and Eigen if the image's are too old or missing, and compiles the small
driver in `tasks/racing/code/togt/`):

```bash
bash tasks/racing/code/togt/build.sh
```

Then plan the raw track (main venv, from `/workspace`):

```bash
python tasks/racing/code/togt_plan.py --track raw
```

The plan lands in `tasks/racing/plans/raw_f0.95.csv`, and the script prints
what decides whether anything can fly it. Read every line:

```
plan: tasks/racing/plans/raw_f0.95.csv
  planned lap (motion onset -> last gate)  2.592 s     incl. the stop at the end  3.13 s
  top speed 4.31 m/s   thrust demand 7.7..17.5 m/s^2 (limit 17.5)   min z 0.76 m
  intended crossings inside the opening: NO  (offsets 0.130 0.130 0.131 0.002 m, bound 0.13)
  crossing angle to the gate normal:  G1   45 deg (2.3 m/s in-plane)   G2   12 deg   G3   86 deg (2.4 m/s in-plane)   G4    8 deg
  [!] the path crosses a gate PLANE through the FRAME zone (0.13-0.43 m from the centre): ...
  obstacle clearance (pole surface):  pole1 0.02 m  pole2 0.14 m  pole3 0.49 m  pole4 0.08 m   [!] < 0.10 m at pole [1, 4]
```

- *thrust demand 17.5 of 17.5* and *offsets 0.130 of 0.130*: every margin
  spent, as promised.
- *G3 86°, 2.4 m/s in-plane*: the drone slides through gate 3 almost parallel
  to its plane. Legal for a zero-thickness corridor; hopeless for a tracker.
- *obstacle clearance 0.02 m at pole 1*: the path shaves the pole in front of
  gate 1. Nothing in the benchmark harness would notice.

Now the tube track, at two thrust headrooms:

```bash
python tasks/racing/code/togt_plan.py --track tube --thrust-frac 0.85
python tasks/racing/code/togt_plan.py --track tube --thrust-frac 0.95
```

Crossings drop to 2–7°, the planned laps are 3.87 s and 3.57 s, top speeds
4.7 and 5.05 m/s. Two warnings stay, and they matter in §5: the path
crosses gate 3's plane a second time 0.41 m from its centre (just outside the
frame, inside the 7 cm safety margin the feasibility check uses), and the
clearance from poles 1, 3 and 4 is 2–7 cm — the entry/exit corridors of gates
1, 3 and 4 sit *on top of* the poles the track designers put 0.5–0.7 m out on
those gate axes. Draw one and look:

```bash
python tasks/racing/code/plot_trajectory.py --plan tasks/racing/plans/tube_f0.95.csv
python tasks/racing/code/plot_trajectory.py --plan tasks/racing/plans/tube_f0.95.csv --time-scale 1.05 --ground-start
```

The second command is the same plan on the leaderboard clock: a takeoff leg
from the ground and no lead-in. Its speed profile is what §4's `--start
ground` rows fly.

> 🔑 The planner is a *tool that optimises exactly what you asked*. Every
> number in that printout is the answer to "what did I forget to ask for?"
> Crossing geometry, obstacle clearance and tracker headroom are all
> planner-side constraints — and cost nothing at run time once added.

## 3. Three MPCs: none, ESO, L1

Open `tasks/racing/crazy_track/src/crazy_track/controllers/mpc.py`. It is
150 lines and you can read all of it.

**The controller** (`MPCController`, line 11). Prediction model: crazyflow's
*identified* attitude model `so_rpy` — state `[pos, rpy, vel, rpy-rates]`,
input `[roll, pitch, yaw, thrust]`, the same interface the race accepts. It
predicts 20 steps of 0.04 s (a 0.8 s horizon) with Euler integration,
penalises position error (line 87), velocity error, control effort and control
*changes*, bounds the inputs (thrust on line 95), and solves the nonlinear
program with ipopt (line 97), warm-started from the previous solution. Nothing
is learned; everything is the model plus a cost you can read.

**The three variants** — the docstring at lines 20–33 is the reference:

| spec | `disturbance=` | what enters the prediction model |
|---|---|---|
| `mpc` | `"none"` | nothing: under a steady wind it re-predicts the same biased trajectory every step |
| `mpc_offsetfree` | `"eso"` | a constant *disturbance acceleration* estimated by a velocity **extended state observer** (bandwidth 7 rad/s) — the classic offset-free MPC |
| `mpc_l1` | `"l1"` | the **same injection point**, but the estimate comes from the L1 piecewise-constant adaptation law your DATT policy and MPPI+L1 use (a_s = −5, 4 Hz low-pass) |

Both estimators live in `_estimate` (lines 127–142) and both run on the same
`so_rpy` thrust map the MPC predicts with (`_model_acc`, line 117) — so the
estimator cannot "discover" the thrust-calibration bias the model already
encodes and count it twice. Everything else (horizon, weights, the 1.5 s
soft-start ramp on lines 156–163) is identical, so **the only variable between
`mpc_offsetfree` and `mpc_l1` is the adaptation law**, and between those and
`mpc` it is whether there is one at all. That is what a clean comparison looks
like; it is the design discipline of Lesson 4 applied to a controller.

**What they cost.** One ipopt solve took the parent project 34 ms per 10 ms
control step; you will measure it yourself in §5. The MPC family is therefore
a *reference point* — the precision a model-based optimiser can reach — not a
controller you could fly on the real drone today. (A real-time-iteration or
acados implementation would be the next step; that is beyond this course.)

The fourth reference is `mppi_l1`: a sampling-based predictive controller with
the same L1 law (`controllers/mppi_l1.py`, `_l1_update` at line 79). All four
are registered by name in `eval/lissajous_benchmark.py:26` (`make_controller`),
which is why `race_eval.py` and the bridge of §5 accept them as plain strings.

## 4. The benchmark: pre-register, run, tabulate — on both clocks

**Pre-register first** (Lesson 4 §1). The parent project wrote these down
before its matrix ran; copy them into your report and add the number that
would make each one *wrong* for you:

| # | hypothesis | right if |
|---|---|---|
| H1 | MPC tracks the closed-form plan at least as well as your RL policy | 4/4 gates and a race time ≤ the policy's, at the same cruise |
| H2 | MPC precision lets the **unstretched** tube plan through (the RL tracker needed ×1.05–1.15) | any MPC variant 4/4 at `--time-scale 1.00` |
| H3 | the adaptive layer pays at speed | `mpc_l1` ≤ `mpc_offsetfree` < `mpc` in max deviation, or L1 completes a plan plain MPC fails |
| Kill | the hybrid must beat the RL tracker or it is not worth its complexity | best MPC-family time < the RL tracker's best |

**Run the matrix** (main venv). One command per cell; an MPC cell takes about
half a minute, so put this in a shell loop and go read §5 while it runs
(`--label` names the column; use it to tell your seeds apart):

```bash
M=tasks/racing/crazy_track/results/<your-run>_datt-train/datt_ppo_final.zip
for C in mpc mpc_offsetfree mpc_l1 datt:$M; do
  for START in hover ground; do
    python tasks/racing/code/race_eval.py --plan closed-form --cruise 3.0 --controller $C --start $START --reason "L6 matrix"
    for P in tube_f0.85 tube_f0.95; do for S in 1.00 1.05 1.15; do
      python tasks/racing/code/race_eval.py --plan tasks/racing/plans/$P.csv --time-scale $S --controller $C --start $START --reason "L6 matrix"
    done; done
    python tasks/racing/code/race_eval.py --plan tasks/racing/plans/raw_f0.95.csv --controller $C --start $START --reason "L6 matrix"
  done
done
python tasks/racing/code/race_table.py
```

Each run prints a `RESULT` line (gates, race time, time to gate 1, max
deviation) and writes a directory under `tasks/racing/crazy_track/results/`
with your `--reason`, the metrics and a `rollout.png`. `race_table.py` folds
them into one table per clock.

**What the parent project measured** (benchmark clock; `mpc_l1` is the hybrid;
"dev" = max deviation from the reference in metres; RL s0 = its best acro
policy, which is *not* the attitude-mode policy you trained — expect yours to
be less precise):

| plan | `mpc` | `mpc_offsetfree` | `mpc_l1` | `mppi_l1` | RL s0 |
|---|---|---|---|---|---|
| closed-form, cruise 3.0 | 4.522 (0.20) | 3/4 (0.84) | 4.485 (0.74) | 4.481 (0.38) | **4.464** |
| tube f0.85 ×1.00 | 3.890 (0.38) | 3.877 (0.23) | **3.857** (0.45) | 3.880 (0.46) | 3/4 |
| tube f0.85 ×1.05 | 4.077 (0.17) | 4.067 (0.10) | 4.046 (0.39) | 4.053 (0.27) | 4.085 |
| tube f0.95 ×1.00 | 3.596 (0.30) | **3.592** (0.32) | **1/4** (1.23) | 3/4 (0.58) | 3/4 |
| tube f0.95 ×1.05 | 3.786 (0.28) | 3.772 (0.21) | 3.744 (0.53) | 3/4 (0.85) | 3/4 |
| tube f0.95 ×1.15 | 4.119 (0.16) | 4.109 (0.10) | 4.096 (0.13) | 4.097 (0.16) | 4.123 |
| raw ×1.00 | 2/4 | 1/4 | 1/4 | 2/4 | 1/4 |

Verdicts upstream: H1 half (MPC completes but is 0.06 s *slower* — it tracks
faithfully where the RL policy corner-cut), H2 ✓, H3 **refuted** (the L1
hybrid is the only MPC variant that fails the fast plan, and deviates most in
8 of its 10 completed rows), Kill passed for the family, not for the hybrid.

**Verified on this course's stack** (crazyflow 0.3.2, mujoco 3.10.0, casadi 3.8.0,
2026-09-10; single runs; "course v5 policy" = the 4 M-step attitude-mode v5 seed the
lessons ship as their baseline, Lesson 2). Benchmark clock first:

| plan | `mpc` | `mpc_offsetfree` | `mpc_l1` | `mppi_l1` | course v5 policy |
|---|---|---|---|---|---|
| closed-form, cruise 2.5 | 4.657 (0.19) | 4.641 (0.13) | 4.627 (0.21) | 4.627 (0.20) | 4.700 (0.45) |
| closed-form, cruise 3.0 | 4.521 (0.20) | 4.478 (0.25) | 2/4 (1.38) | 4.501 (0.54) | 4.574 (0.48) |
| tube f0.85 ×1.00 | 3.887 (0.46) | 3.876 (0.23) | 3.855 (0.83) | 3/4 (0.37) | 3.961 (0.45) |
| tube f0.85 ×1.05 | 4.077 (0.55) | 4.071 (0.13) | 4.048 (0.58) | 2/4 (0.52) | 4.138 (0.40) |
| tube f0.95 ×1.00 | 3.593 (0.30) | 3.603 (0.35) | 2/4 (0.65) | 2/4 (1.73) | 3/4 (0.52) |
| tube f0.95 ×1.05 | 3.780 (0.24) | 3.775 (0.26) | 3.744 (0.40) | 2/4 (0.69) | 3.844 (0.47) |
| tube f0.95 ×1.15 | 4.118 (0.16) | 4.107 (0.10) | 4.092 (0.13) | 4.092 (0.14) | 4.174 (0.39) |
| raw ×1.00 | 2/4 (0.23) | 1/4 (0.42) | 1/4 (0.27) | 3/4 (0.17) | 1/4 (0.47) |

The same 40 runs on the **leaderboard clock** (`--start ground`, 1.5 s takeoff):

| plan | `mpc` | `mpc_offsetfree` | `mpc_l1` | `mppi_l1` | course v5 policy |
|---|---|---|---|---|---|
| closed-form, cruise 2.5 | 6.157 (0.19) | 6.141 (0.13) | 3/4 (0.30) | 6.115 (0.27) | 6.200 (0.45) |
| closed-form, cruise 3.0 | 6.021 (0.20) | 3/4 (1.74) | 3/4 (1.05) | 5.992 (0.20) | 6.074 (0.48) |
| tube f0.85 ×1.00 | 5.387 (0.46) | 5.377 (0.23) | 3/4 (0.86) | 5.382 (0.39) | 5.461 (0.45) |
| tube f0.85 ×1.05 | 5.577 (0.55) | 5.570 (0.12) | 5.544 (0.56) | 5.557 (0.15) | 5.638 (0.40) |
| tube f0.95 ×1.00 | 5.093 (0.30) | 5.106 (0.35) | 1/4 (2.77) | 2/4 (1.94) | 3/4 (0.52) |
| tube f0.95 ×1.05 | 5.280 (0.24) | 5.274 (0.26) | 5.245 (0.41) | 3/4 (1.62) | 5.344 (0.47) |
| tube f0.95 ×1.15 | 5.618 (0.16) | 5.607 (0.10) | 5.592 (0.13) | 5.592 (0.15) | 5.674 (0.39) |
| raw ×1.00 | 2/4 (0.23) | 2/4 (0.21) | 2/4 (0.16) | 2/4 (0.21) | 1/4 (0.47) |

Three things these two tables say that the upstream table could not:

- **The takeoff tax is exactly 1.5 s wherever the takeoff is benign** — every completed
  ground-clock cell is its hover-clock twin + 1.500 ± 0.005 s, because the takeoff ends
  at rest at the plan's first point and the plan is unchanged. What the ground clock
  *adds* is not a number but a failure mode: see the next point.
- **The estimator variants lose laps on the ground clock that they complete on the
  hover clock** (`mpc_offsetfree` on the closed-form plan at cruise 3.0: 4.478 s in the
  air, 3/4 with a 1.74 m excursion from the ground; `mpc_l1` on tube f0.85 ×1.00 and
  the closed-form plans). The obvious suspect is the 1.5 s soft-start ramp in
  `mpc.py:156–163`: it was written for a hover start, where the lead-in gives the
  estimator 1.5 s of stationary flight to converge, and on the leaderboard clock those
  same 1.5 s coincide with the takeoff. That reading stood until Lesson 8 §5 tested it
  directly — gating the ramp on lift-off, shortening it, removing it: 0/10 either way,
  and the same estimator on a *corrected* model is still worse than no estimator at all.
  What the ESO and the L1 law actually learn is the **prediction model's own error**
  (`mpc.py` predicts with an attitude gain of 0.73 where this simulator's is 0.94), as a
  world-frame "disturbance" discovered in one turn and applied in the next. The ramp is
  a symptom; the model is the cause. Keep the observation, and read §5's mechanism 2
  knowing that the axis it names is not the one that decides.
- **The course's v5 policy is a precision tracker too, ~0.07–0.10 s behind the MPC on
  every plan it completes**, and it fails the unstretched f0.95 plan the way the parent
  project's best acro policy did (3/4). Your own seeds go in that column.


**Reading the table** — do this with your own numbers, not the ones above:

1. Read the raw-plan row first. Nothing flies it, at 1–6 cm of precision or at
   20 cm. Geometry, not speed (§1).
2. On the tube plans, which tracker needs the least time stretch to complete?
   That is *precision*, and it converts directly into lap time: the parent
   project's RL tracker needed 5–15 % of stretch on every tube plan; the MPC
   flew them unstretched.
3. On the closed-form plan the ranking may invert: a tracker that cuts corners
   beats one that follows the line faithfully. Which is "better" depends on how
   much of the racing line the planner already found.
4. Now compare the two clocks row by row. The ground-clock number minus the
   hover-clock number is the takeoff tax (plus the lead-in's absence). Does the
   *ranking* change? If two trackers swap places between clocks, look at their
   `t_gate1`: the takeoff is where a model-mismatch shows first.

## 5. The same stack in the real race

The benchmark harness has no randomisation and no collisions, and its clock
starts in the air. `race_bridge_mpc.py` puts any of these controllers into
lsy's environment with the SAME reference object `race_eval.py --start ground`
flew (a takeoff from the actual start pose, then the closed-form line or a TOGT
plan), so the only things that change are the ones the race adds: its 50 Hz
loop, its Level-1 randomisation (mass ±11.5 %, inertia, start pose, action
noise) and its contacts.

```bash
cp tasks/racing/code/race_bridge_mpc.py repos/lsy_drone_racing/lsy_drone_racing/control/
cd repos/lsy_drone_racing
RACE_CONTROLLER=mpc_offsetfree RACE_CRUISE=2.5 RACE_VERBOSE=1 \
  /opt/venvs/race/bin/python scripts/sim.py --config level0.toml --controller race_bridge_mpc.py --render False
```

(`control_mode = "attitude"` in the level toml, as in Lesson 3 §4.) The bridge
prints its plan once, then the race's `Flight time` and `Gates passed`, then one
line you will not see anywhere else: the controller's solve time per 20 ms
step. Write that number down.

Then the tube plan through the same bridge:

```bash
RACE_CONTROLLER=mpc_offsetfree RACE_PLAN=/workspace/tasks/racing/plans/tube_f0.95.csv RACE_TIME_SCALE=1.05 \
  /opt/venvs/race/bin/python scripts/sim.py --config level0.toml --controller race_bridge_mpc.py --render False
```

**What happens (verified 2026-09-10, Level 0, one episode each, `RACE_TAKEOFF_T=1.5`):**

| reference through the bridge | `mpc` | `mpc_offsetfree` | `mpc_l1` |
|---|---|---|---|
| closed-form line, cruise 2.0 | 3/4 (6.54 s) | 3/4 | 1/4 |
| closed-form line, cruise 2.5 | 3/4 (5.90 s) | **4/4, 6.12 s** | 0/4 |
| closed-form line, cruise 3.0 | 3/4 | 2/4 | 2/4 |
| tube f0.85 ×1.00 / ×1.05 / ×1.15 | 2/4, 2/4, 3/4 | 2/4, 2/4, 3/4 | 3/4, 1/4, 3/4 |
| tube f0.95 ×1.00 / ×1.05 / ×1.15 | 3/4, 1/4, 3/4 | 2/4, 2/4, 3/4 | 1/4, 3/4, 2/4 |
| raw ×1.00 | 0/4 | — | — |

Compare that with §4, where the same references and the same controllers completed
almost everything on either clock. The benchmark harness and the race differ in three
things; each shows up in this table:

1. **Contacts.** Every tube plan ends early — the parent project's corridors stand on
   poles 1, 3 and 4 (§2), and the race ends an episode on contact. So does the
   closed-form line at cruise 2.5 for plain MPC: it passed gates 1–3 with 2–4 cm of
   offset and then touched pole 4, which the lsy line passes 0.03–0.08 m from on the
   *reference* (`plot_trajectory.py --bridge ... --flown` and `race_refs.obstacle_clearance`
   on the flown CSV show it). The drone's collision body in lsy is a 0.07 m box: a faithful
   tracker on a line that shaves a pole *will* touch it; the Lesson-3 policy at cruise
   1.5 got away with it by being slower and less faithful.
2. **The estimators lose the line early** (as in §4's ground-clock table): `mpc_l1`
   leaves it inside the first two seconds at every cruise, and `mpc_offsetfree` is the
   most precise variant at cruise 2.5 but not at 3.0. The soft-start ramp opening during
   the climb is the tempting explanation; §4's bullet says why it is the wrong one
   (Lesson 8 §5 gated the ramp on lift-off and gained nothing).
3. **50 Hz instead of 100 Hz**, and about 20–30 ms of solve time per 20 ms step — the
   number the bridge prints. The race is simulated step by step, so this costs wall time,
   not laps; on a real drone it would cost the lap.

So the first real-race number for this stack is `mpc_offsetfree` on the closed-form line
at cruise 2.5: **6.12 s, 4/4** — against 4.478 s for the same controller and line on the
benchmark clock, and 6.141 s on `race_eval.py --start ground`. The 1.5 s is the takeoff
(§0); the 0.02 s is the 50 Hz loop. Read those three numbers together once and you will
never confuse the clocks again.

**A pole-safe line.** `race_refs.closed_form_line(..., line="safe")` (`RACE_LINE=safe`)
keeps the same gates and hairpin, takes gate 3 at half cruise and adds three vias that pass
poles 3 and 4 with ≥ 0.14 m to spare on the reference (found by a small search with
`obstacle_clearance`, at the cost of ~0.8 s of planned lap; no via placement clears both
poles above cruise ~2.7). Single episodes: cruise 2.0 → `mpc` 7.18 s and `mpc_offsetfree`
7.14 s, both 4/4, `mpc_l1` 2/4; cruise 2.5 → `mpc` 6.94 s 4/4, the estimator variants
2/4. That is the trade the track designers built in: the poles cost a faithful tracker
about a second, unless the *planner* knows about them.

**The 20-episode protocol, verified 2026-09-10** (mean ± std over successful episodes,
successes out of 20; a row below 50 % is unranked):

| line (through `race_bridge_mpc.py`) | level | `mpc` | `mpc_offsetfree` | `mpc_l1` |
|---|---|---|---|---|
| closed-form `lsy`, cruise 2.5 | 0 | 0/20 | **6.122 ± 0.006 s, 18/20** | 0/20 |
| closed-form `lsy`, cruise 2.5 | 1 | 0/20 | **6.124 ± 0.016 s, 16/20** | 0/20 |
| closed-form `safe`, cruise 2.0 | 0 | 7.180 ± 0.000 s, 20/20 | **7.140 ± 0.000 s, 20/20** | 7.147 s, 8/20 (unranked) |
| closed-form `safe`, cruise 2.0 | 1 | 7.182 ± 0.006 s, 20/20 | **7.139 ± 0.013 s, 20/20** | 7.145 s, 4/20 (unranked) |

- **Level 0 is not quite deterministic**: `level0.toml` keeps a 0.001-rad action noise
  and a small dynamics disturbance (`[env.disturbances.*]`), invisible on the Lesson-3
  policy (spread 0.000) and decisive for a lap that passes 8 cm from a pole — hence
  18/20, not 20/20, on the `lsy` line. The `safe` line's 20/20 with a 0.000 spread is
  what a real margin looks like.
- **H4, verdict:** at Level 1 (mass ±11.5 %) the ESO variant keeps 20/20 and stays
  0.04 s faster than plain MPC, but plain MPC keeps 20/20 too — the mass error is
  absorbed by the MPC's own feedback (a 0.5 g mass error at hover is a 5 % thrust
  error, well inside its control authority). The prediction "the estimator recovers
  what it cost" is **not supported** at this randomisation level; the L1 hybrid is
  unranked on every row for the reason §4's bullet now gives: its estimate is learning
  the prediction model's error rather than a disturbance. Write down what would make H4
  right — a heavier payload, wind — and notice that Level 1 does not contain it. (Lesson 8
  §5 finds the randomisation that *does* defeat plain MPC on a fast plan, and it is
  multiplicative: the mass, through the thrust map, not an additive force.)
- **Against the leaderboard, on the leaderboard's clock:** 7.14 s at 100 % success is
  the best fully-ranked number of this course so far (the Lesson-3 policy baseline is
  7.80 s), still 3.7 s from the record — and the whole gap is *plan*: the pole-safe line
  spends 1.5 s taking off and ~1 s going around poles the time-optimal planner never
  heard of (§2). That is the next lever, and it is planner-side.


And score like the leaderboard, all variants under one protocol
(`compare_models.py` from Lesson 5, now with `--bridge`; a spec instead of a
`.zip` is passed to the bridge as `RACE_CONTROLLER`; the other knobs are
inherited from your shell so that one comparison changes one thing):

```bash
cd /workspace
RACE_CRUISE=2.5 /opt/venvs/race/bin/python tasks/racing/code/compare_models.py \
    --bridge race_bridge_mpc.py --episodes 20 --config level0.toml mpc=mpc eso=mpc_offsetfree l1=mpc_l1
RACE_CRUISE=2.5 /opt/venvs/race/bin/python tasks/racing/code/compare_models.py \
    --bridge race_bridge_mpc.py --episodes 20 --config level1.toml mpc=mpc eso=mpc_offsetfree l1=mpc_l1
# the pole-safe line, one notch slower, same three trackers, both levels
RACE_LINE=safe RACE_CRUISE=2.0 /opt/venvs/race/bin/python tasks/racing/code/compare_models.py \
    --bridge race_bridge_mpc.py --episodes 20 --config level0.toml --out-dir tasks/racing/figures/safe mpc=mpc eso=mpc_offsetfree l1=mpc_l1
RACE_LINE=safe RACE_CRUISE=2.0 /opt/venvs/race/bin/python tasks/racing/code/compare_models.py \
    --bridge race_bridge_mpc.py --episodes 20 --config level1.toml --out-dir tasks/racing/figures/safe mpc=mpc eso=mpc_offsetfree l1=mpc_l1
```

Level 0 randomises nothing (mass, gates, start) but keeps a small action noise, so
its 20 episodes are near-copies of the single episode above; the spread you see is
that noise meeting a marginal clearance. The table exists to make the *protocol*
identical, not to add information.

**The benchmark clock inside the race.** `compare_models.py --start hover` writes a
`level0_hoverstart.toml` next to the level config with the drone placed at rest at
hover height, and the bridges then skip their takeoff leg (`RACE_START=auto`). Two
things you will see: the reported time is the benchmark clock's plus a settle hold
(`RACE_SETTLE`, default 1.0 s), and the hold is not optional — lsy spins the rotors up
from rest, so a drone released at 1.0 m sags about 0.5 m in the first 0.4 s and a fast
plan reaches gate 1 before it has recovered (measured: without the hold one lap in five
ends on the gate-1 frame; with it, four of five logged episodes at 5.62 s = 4.62 s + 1.0 s). The
benchmark harness never showed this because its 1.5 s lead-in was doing the settling.

Level 1 is the hypothesis §4 could not test: on a nominal track there is
nothing to adapt to, but Level 1 randomises the mass by ±0.005 kg — 11.5 % of
the airframe, which a fixed-model MPC feels as a thrust error at every step.
Pre-register it: *H4 — at Level 1 the estimator variants recover what they cost
at Level 0* (right if their success rate or mean time beats plain `mpc` there
while not beating it at Level 0).

## 6. Read the result honestly

Four mechanisms, each one a sentence you should be able to defend with a row
of your table:

1. **What the RL tracker was missing is precision, and MPC has it.** Its 1–6 cm
   gate offsets fly the unstretched plans; the policy's ~20 cm needed 5–15 % of
   stretch. That is how a 42 % plan-side gain became 19.5 % of realised lap
   time for the MPC and 8.5 % for the policy (benchmark clock).
2. **Adaptation is a liability where there is nothing to adapt to.** On a
   nominal track the estimator's residual is the model's own bias during
   aggressive transients — not a constant force. A fast estimator (L1, 25 rad/s
   effective) feeds it forward as a phantom disturbance over the whole 0.8 s
   horizon; the slower ESO (7 rad/s) attenuates it. Paper 1's bandwidth dilemma
   for ADRC, on the model-mismatch axis. The hybrid's *purpose* — wind, payload,
   mass error — is exactly what Level 0 lacks and Level 1 has (H4).
3. **Corner-cutting versus faithfulness.** On the closed-form plan the RL
   policy beats every MPC by cutting the reference; on the tube plans, where
   the reference *is* the fast line, faithfulness wins by half a second.
4. **Crossing geometry and obstacle clearance are planner-side constraints.**
   Even a perfect tracker cannot fly a crossing at 86°, and the benchmark
   harness will happily fly through a pole. The race will not.

**Caveats you must write down with any number from this lesson:** single
deterministic runs on the benchmark clock (the classical-stack convention);
one MPPI seed; the MPC is not real-time (20–35 ms per solve against a 10–20 ms
step, so a real drone could not run it as is); the interfaces differ (attitude
for MPC, body rates for the parent project's acro policy); the benchmark clock
is not the leaderboard's (§0); and a single-seed policy column says nothing
about policies in general (Lesson 2 §5 — three seeds).

---

## 🛠️ Before you move on

1. **The verdict table.** H1–H4 with your thresholds, your numbers, your
   verdicts — including the ones that went the other way.
2. **The takeoff tax, measured three ways:** ground-clock minus hover-clock
   from §4, `t_gate1` from the same runs, and the race's own `Flight time`
   minus the benchmark number. Do the three agree? If not, which term is
   missing from which clock?
3. **Make the tube plan race-legal.** Copy `configs/togt/lsy_level2_tube.yaml`
   next to it, shorten or angle the corridors that sit on poles 1, 3 and 4 (or
   replace them by `SingleBall` vias beside the poles), plan your file with
   `togt_plan.py --track-yaml <your.yaml>`, and find the fastest plan whose
   clearance is above 0.12 m everywhere. Race it through the bridge. What did
   the poles cost, in seconds? (Lesson 7 §4 shows one answer,
   `togt_plan.py --track poles` — do yours before you read it; `--track` also takes
   `raw`, `tube` and, after Lesson 8, `ground` and `ground-b08g3`.)
4. **One variable.** `mpc_offsetfree_w3` and `mpc_offsetfree_w15` change only
   the ESO bandwidth; `mpc_l1_c2` only the L1 cutoff. Pick the axis §6's
   mechanism 2 predicts and test it on the f0.95 plan.
5. **What you would do next, and why** — one paragraph. The parent project's
   answer was "train the tracker on TOGT-like references"; yours may differ.

**Back to:** [Lesson 4 — Brainstorm: make it faster](04-brainstorm-faster-tracking.md)
— with two more levers on the board and a second clock in your report.
