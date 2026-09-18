# Lesson 8 — Closing the gap: diagnose, identify, replan, adapt

⏱️ ~4 hours of reading and running. The compute is small — the probes take about a
minute, a plan a few seconds, a 20-episode MPC cell 75–105 s on an idle machine — so
roughly 15 minutes in all, and you can start each cell as you reach its section rather
than the night before. **You finish when** you have
(a) the failure anatomy of one 20-episode cell of yours (which feature ends each
episode, when, and whether it is a lag or a cut), (b) the identification table of §2
from your own probes next to the model the MPC predicts with, (c) a 20-episode cell
at *each* level on the ground-start plan with the unmodified vendored MPC, and (d) a
verdict on each of the six pre-registered hypotheses of §0 — including the three that
went the other way.

> **The one big idea:** Lesson 7 left this course at 5.32 s and the leaderboard at
> 3.39 s. This lesson closes 1.2 s of that gap without changing a line of the
> controller, and it does so in an order you should keep for the rest of your career:
> **diagnose** (read the flown paths before touching a gain — the failures were not
> where the plan assumed), **identify** (measure the simulator the model predicts —
> its attitude gain is 0.94, the model says 0.73), **replan** (the takeoff was 2.3 s of
> the lap; a ground-start plan with the gates pre-shifted against the tracker's
> systematic cut flies at 4.12 s), and only then **adapt** (Level 1 loses to the mass,
> not to the start pose; a thrust-scale adaptation on the corrected model takes the
> adopted plan from 6/20 to 18/20). And one result you will meet again: a 40-cell
> tuning sweep in which every weight, horizon and lookahead change is *worse* than the
> default is not a failed experiment — it is the measurement that the cost was already
> at a local optimum and the error was the model.

**Prerequisites:** Lessons 6 and 7 (all of it: the two clocks, the planner, the MPC
family, the bridge, the protocol), Lesson 3 (the leaderboard's rules), Lesson 4
(pre-registration). Everything runs on CPU.

**Receipts.** Every number in this lesson was measured on 2026-09-17/18 in the
instructor's experiment archive, `results/2026-09-17_mpc-gap/` (outside this repo; one
folder per work package — `diagnose/`, `physics/`, `planner/`, `mpcgap/`, `controller/`,
`level1/`, `protocol/`, `audit/` — each with a `NOTES.md` and its logs). The parentheses
`(receipt: …)` name the file a number comes from, the way Lesson 7 names its tables;
where a number is a 5-, 10- or 15-episode *screen* rather than a 20-episode protocol
cell, the text says so. An adversarial audit re-derived all 28 protocol cells from the
per-episode logs (§7).

---

## 0. Where the 5.32 s goes, and six hypotheses

Lesson 7's fastest ranked lap — plain MPC on the pole-aware plan, 1.5 s takeoff — reads
5.32 s in every one of its 15 successes. A lap whose successes all read the same number
is a lap the *reference* sets; the controller only decides whether the episode survives.
So decompose the reference before anything else (gate times on the leaderboard clock:
G1 2.317, G2 3.213, G3 4.416, G4 5.330 s; receipt: `diagnose/logs/reference_anatomy.txt`):

| term | s | what it is |
|---|---|---|
| rest-to-rest quintic takeoff, ground → hover point (−1.5, 0.75, 1.0) | 1.500 | `race_refs.GroundStartTrajectory`, `RACE_TAKEOFF_T` |
| hover point → gate 1, *from rest again* (the plan's `initState` is at rest) | 0.817 | the anatomy of `poles_f0.85.csv` |
| gate 1 → gate 2 | 0.897 | |
| gate 2 → gate 3 (through the G2-exit hairpin; 2.11 m/s at gate 3) | 1.202 | |
| gate 3 → B3 → B4 → gate 4 (the pole-4 detour; 3.27 m/s at gate 4) | 0.915 | |
| **lap on the leaderboard clock** | **5.331 → reads 5.32** | the race clock floors to its 20 ms step |

Three facts shaped everything that follows. **The drone stops twice before gate 1**: the
takeoff is rest-to-rest and the plan starts from rest at the hover point, so the first
2.32 s of the lap are two rest-to-rest manoeuvres. **The plan runs at 0.85 × TWR** because
the 0.95 plan ends on contacts (Lesson 7 §4.1). And **hardware is not a differentiator**:
everyone races the same cf21B_500 (TWR 1.88) in the same simulator, at the same 50 Hz,
and the simulator does not penalise the solve time. What is *ours* is the takeoff, the
plan, the model the MPC predicts with, its cost, and the Level-1 adaptation.

**Pre-register** (Lesson 4 §1). These were written down before any experiment ran
(`reports/2026-09-17_mpc-leaderboard-gap-plan.md` §1); copy them, and add the number
that would make each one wrong for you:

| id | hypothesis | right if | kill |
|---|---|---|---|
| H1 takeoff | a TOGT plan whose `initState` is the ground pose, flown straight into gate 1, replaces takeoff + first leg (2.32 s) by ≈ 1.2–1.5 s | ground-clock lap ≤ 4.5 s with plain MPC at 0.85 × TWR, ≥ 10/20 at Level 0 | the plan lifts the drone faster than the rotors spin up, or the first leg becomes untrackable (a contact at gate 1 or pole 1) |
| H2 end state | an `endState` further past gate 4, or a non-zero end velocity, lets the drone cross gate 4 at speed | ≥ 0.1 s off the plan, no new gate-4 contacts | the deceleration already starts after gate 4 |
| H3 model | the MPC's forward-Euler step of 0.04 s on attitude dynamics with ω_n = 13.7 rad/s (ω_n·dt = 0.55) and its under-weighted position cost cause the 0.2–0.5 m deviation | the harness deviation halves on the same plan; the 0.95 × TWR plan becomes race-legal | RK4 and tuned weights do not move the deviation → the error is model mismatch, not discretisation |
| H4 lookahead | sampling the reference at t + τ (τ ≈ one solve + one loop), or a reference-acceleration feed-forward, removes a pure lag | the along-path lag of the diagnosis vanishes | the deviation is lateral (a cut), not a lag |
| H5 Level 1 | plain MPC's Level-1 losses come from the mass randomisation; an estimator gated on lift-off, or a thrust-scale adaptation on the vertical residual, restores ≥ 10/20 | Level-1 success ≥ 15/20 at the same lap time | the failures do not correlate with mass → the start pose at marginal clearance is the cause |
| H6 via radius | slightly larger via balls (0.15 → 0.18–0.20 m) buy Level-1 success for ≤ 0.1 s | | |

The plan also carried an *assumption*, not a hypothesis: that the 5/20 Level-0 failures
are pole contacts at marginal clearance (0.20–0.21 m on the reference at poles 1 and 4;
0.083 m flown in Lesson 7's replay). §1 tests the assumption first. It is wrong.

## 1. Diagnose before you tune: read the flown paths

Everything §1 needs is already on your disk after Lesson 7: the flown paths of the
protocol-A cell (`RACE_LOG_DIR`, `flown_epNN.csv` at 50 Hz, 20 per level) and its
`comparison.csv`. No simulation was run for this section (receipt: `diagnose/NOTES.md`,
`diagnose/logs/`; the tables below are `failure_summary.csv`, `failure_verdicts.txt`,
`episodes.csv`, `deviation_profile.csv`, `start_correlation.csv`, `takeoff.csv`,
`pole_clearance.csv`). Protocol C (the same plan, poles moved to the corners) was read
the same way, as a control.

**The contact model you need to read a flown path.** lsy ends the episode on any MuJoCo
contact between the drone's collision *box* (0.07 × 0.07 × 0.02 m half-extents; the race
enables it with `use_box_collision(sim, True)` in `race_core.py`) and a gate's four frame
boxes (0.20–0.36 m from the centre, 0.01 m half-thickness) or a pole (a 0.015 m capsule).
So a level drone touches a gate *post* when its in-plane horizontal offset reaches
0.101 m at the two 45° gates (G1, G2) and 0.13 m at G3/G4, a *rail* when its vertical
offset reaches 0.18 m, and a pole when the surface distance is 0.07–0.10 m. Those three
numbers turn a flown CSV into a verdict.

**What ended the 33 failures of 80 logged episodes** (protocols A and C, both levels;
every failure attributable to exactly one feature, the runner-up ≥ 0.13 m away):

| feature | where / when | L0 (5 fail) | L1 (12) | C-L0 (7) | C-L1 (9) | all |
|---|---|---|---|---|---|---|
| **gate 1's left post, on the gate-2 → gate-3 *return* leg** (t = 3.98–4.06 s, 2.8–2.9 m/s) | 3 | 8 | 1 | 5 | **17** |
| gate 2's inside post / bottom rail, at the intended crossing (t = 3.20–3.22 s) | 1 | 1 | 4 | 0 | 6 |
| gate 1's left post at the intended crossing (t = 2.30 s, 3.8 m/s) | 1 | 2 | 1 | 1 | 5 |
| gate 3's inside post, at the intended crossing (t = 4.46–4.52 s) | 0 | 1 | 1 | 3 | 5 |
| **any pole** | 0 | 0 | 0 | 0 | **0** |

Not one failure touches a pole. Protocol C, which moves the poles away, fails as often
(7/20, 9/20) with the same features. The assumption of §0 was wrong for this cell, and
every plan lever aimed at pole clearance (H6) was aimed at the wrong thing.

**The dominant failure is a lag, the rest is a cut.** Out of the G2-exit hairpin — where
the plan's speed bottoms at 1.61 m/s (t = 3.48 s) and its thrust demand peaks at
15.4 m/s² (t = 3.50 s) — the drone falls **0.29–0.44 m behind** the reference (an
equivalent lag of 0.08–0.13 s) and 0.15–0.28 m *outside* the line at 2.8–2.9 m/s; the
reference itself passes only 0.159 m (box gap) in front of gate 1's post at t = 3.96 s.
That 0.4 s window holds 14 of the 15 Level-0 deviation maxima and 17 of the 33 failures.
Everywhere else the error is a **cut**: the successes fly 0.12–0.15 m inside both hairpins
(the G1 → G2 U-turn at 2.8–3.4 s and the G3 → G4 corner at 4.6–5.0 s) and 0.04 m inside
at every gate crossing. TOGT already places every crossing 0.050 m inside the turn (the
edge of its ±0.05 window); the tracker adds 0.03–0.05 m; the flown gate-1 offset is
0.087 m against a post at 0.101 m. **The whole population crosses gate 1 with about 1 cm
of margin**, and the five gate-1 failures at the intended crossing are its tail. Pole 4 is
flown at 0.133 m where the plan passes at 0.203 (contact ≈ 0.10): the same cut, seen at a
pole — no contact in 80 episodes, but the plan's margin is mostly gone.

Mean over the 15 Level-0 successes, along the lap (dev / along-path / lateral / vertical /
equivalent lag; receipt: `deviation_profile.csv`):

| t (s) | phase | dev | along | lat | dz | lag |
|---|---|---|---|---|---|---|
| 0.2–1.4 | takeoff | 0.005–0.02 | 0 | 0.005 | 0 | 0 |
| 1.6–1.8 | leaving the hover point | 0.04–0.07 | +0.04…+0.07 (ahead) | 0.01–0.02 | −0.03 | −0.10…−0.03 |
| 2.2–2.6 | gate 1 | 0.08–0.13 | −0.03…−0.12 | 0.05–0.07 (inside) | +0.03 | 0.005–0.05 |
| 2.8–3.4 | G1 → G2 hairpin, gate 2 | 0.14–0.15 | −0.04…+0.03 | **0.14–0.15 (inside)** | **−0.06…−0.10** | −0.01…+0.05 |
| 3.6 | the G2-exit hairpin (speed minimum) | 0.066 | +0.02 | 0.06 | −0.06 | −0.008 |
| 3.8–4.0 | accelerating out of it, past gate 1's post | 0.25–0.34 | **−0.19…−0.31 (behind)** | 0.14–0.17 (outside) | −0.04…−0.08 | **+0.056…+0.094** |
| 4.2–4.4 | gate 3 | 0.17–0.25 | −0.17…−0.24 | 0.05–0.07 | 0…+0.02 | +0.07…+0.10 |
| 4.6–5.0 | G3 → B3 → B4 → G4 | 0.15–0.16 | −0.11…+0.06 | **0.10–0.16 (inside)** | −0.03…−0.04 | −0.03…+0.08 |

Two more rows of this profile matter later. The drone is 0.06–0.10 m *low* on the climb
into gate 2 and 0.03 m low in cruise even at Level 0 with the nominal mass (a constant
hover-thrust offset — §2 finds its cause). And the takeoff is tracked to 2 cm: lift-off
at 0.26–0.28 s, z(1.5 s) = 0.998–1.002 m, but the drone never actually stops at the hover
point (0.10–0.20 m/s at 1.5 s, already 0.11–0.17 s *ahead* of the reference: the 0.8 s
horizon sees the plan leave). The 1.5 s rest-to-rest climb wastes time and creates no
tracking problem — the only thing a shorter start must respect is the 0.27 s before the
rotors lift the drone (receipt: `takeoff.csv`; §2 measures it).

**Level 1 has the same features 2.4× as often** (8/20 of the dominant class against
3/20), and the start pose is *irrelevant*: the correlation of the start offset with
success is −0.007 (x) / +0.053 (y), and the horizontal deviation at t = 1.5 s is
0.009–0.023 m whatever the draw — the takeoff quintic, built from the observed start,
absorbs it (receipt: `start_correlation.csv`, `takeoff.csv`). What does correlate is the
mass, read through its proxies: the failures fly 0.04–0.12 m low in cruise (successes
0.01–0.06), and the lag at the gate-2 crossing is 0.030 s for the failures against 0.017 s
for the successes (r = −0.76 with success) — 7 of the 8 Level-1 failures of the dominant
class crossed gate 2 ≥ 0.023 s late, all 8 successes ≤ 0.022 s. A drone 11.5 % heavier has
11 % less thrust-to-weight; the plan's 15.4 m/s² at the hairpin exit is 0.93 of what such a
drone has, so it climbs late into gate 2 and cannot accelerate out of the hairpin as
demanded. H5's kill criterion ("the failures do not correlate with mass") did *not* fire —
but the plan's own assumption ("start pose at marginal clearance") did. §5 logs the mass
itself and confirms both.

> 🔑 Read the flown paths before you touch a gain. Half an hour with 80 CSVs told the
> experiment (a) the failures are frames, not poles, so pole clearance is not the lever;
> (b) the dominant error is a lag born in a 0.4 s window of peak thrust demand, and the
> rest is a systematic inside cut with 1 cm of margin at gate 1; (c) the start pose does
> nothing at Level 1 and the mass does everything. Every later decision — which plan
> shifts to make, what the model must predict better, what to adapt — came from this
> table, not from a tuning run. Lesson 8's `race_runner.py` (§5) prints these features
> per episode (the `EP` line: verdict, gate crossings with lag and in-plane offset, cruise
> height, pole-4 clearance) so that you never rebuild them by hand; for Lesson 7's logs use
> `plot_trajectory.py --flown` and `race_refs.obstacle_clearance`.

## 2. Identify the simulator: what the model did not know

The MPC predicts with crazyflow's *identified* attitude model `so_rpy` (Lesson 6 §3). Nobody
had checked it against the race simulator's Mellinger loop and rotor model. Lesson 8 ships
a probe controller that flies *scripted* attitude commands in the real race environment
(Level 0, `level0.toml` unmodified, its action noise and dynamics disturbance on) and logs
every 50 Hz sample, and a script that fits the logs. One lsy `Controller` subclass per
file, installed by copy — copy the *file*, not the folder (`docs/4-troubleshooting.md`):

```bash
cp tasks/racing/code/race_bridge_mpc.py tasks/racing/code/race_probe.py repos/lsy_drone_racing/lsy_drone_racing/control/
cd repos/lsy_drone_racing
# (a) the fastest climb: full thrust from rest on the floor, level attitude (3 repeats)
PROBE=climb PROBE_THRUST=0.8 /opt/venvs/race/bin/python scripts/sim.py --config level0.toml --controller race_probe.py --n_runs 3 --render False
# (b) hover thrust: does m*g lift the drone? does the MPC's hover (0.4395 N)?
PROBE=hold PROBE_THRUST=0.4256 /opt/venvs/race/bin/python scripts/sim.py --config level0.toml --controller race_probe.py --n_runs 3 --render False
PROBE=hold PROBE_THRUST=0.4395 /opt/venvs/race/bin/python scripts/sim.py --config level0.toml --controller race_probe.py --n_runs 3 --render False
# (c) open-loop roll and pitch steps at hover — 0.3 rad, then 0.15 rad: `att=sim` is the mean
#     over BOTH, `att=sim03` the 0.3 rad set alone, and you want to be able to tell them apart
PROBE=hoverstep  /opt/venvs/race/bin/python scripts/sim.py --config level0.toml --controller race_probe.py --n_runs 3 --render False
PROBE=hoverstep PROBE_STEP=0.15 PROBE_TAG=a015 /opt/venvs/race/bin/python scripts/sim.py --config level0.toml --controller race_probe.py --n_runs 3 --render False
PROBE=thruststep /opt/venvs/race/bin/python scripts/sim.py --config level0.toml --controller race_probe.py --n_runs 3 --render False
PROBE=lateral    /opt/venvs/race/bin/python scripts/sim.py --config level0.toml --controller race_probe.py --n_runs 3 --render False
cd /workspace
python tasks/racing/code/probe_fit.py tasks/racing/figures/probes        # main venv: the fits and the table below
# (PROBE_LOG_DIR is relative to the REPO ROOT, not to the clone you ran the probes from;
#  every episode prints the file it wrote, so check one before you fit.)
```

`PROBE=climb|hold|hoverstep|thruststep|lateral` selects the script; `PROBE_THRUST`,
`PROBE_STEP` (step size, 0.3 rad), `PROBE_FSTEP` (the thrust step's level, 0.6 N),
`PROBE_PITCH` (0.5 rad), `PROBE_TILT` (a pitch held from t = 0 during a climb),
`PROBE_T`, `PROBE_TAG` and `PROBE_LOG_DIR` (default `tasks/racing/figures/probes`) are
the knobs; the docstring of `race_probe.py` has the exact sequences. Each `sim.py` call
costs ≈ 30 s of environment creation plus 1–10 s per episode; the probe itself has no
solver. `probe_fit.py` fits an ARX(2,1) model to every step response, overlays the
so_rpy model (continuous, and forward-Euler at the MPC's 40 ms) on the *measured command
sequences*, regresses the climbs for the thrust gain and the drag, and reads the lift-off
times. What it found (3 repeats each; receipts: `physics/NOTES.md`,
`physics/logs/analysis/summary.txt`, `attitude_models.csv`, `probe_results.csv`):

| quantity | race simulator, measured | so_rpy, the model the MPC predicts with | the MPC's forward Euler at 40 ms of it |
|---|---|---|---|
| attitude step 0.3 rad: dc gain (at 0.5 s) | **0.94** (1.00 by 0.8 s) | 0.73 | 0.73 |
| poles (s⁻¹) / ω_n / ζ | −11.6, −29.2 / 18.4 / **1.11** | −6.4 ± 12.2j / 13.7 / 0.47 | equivalent ζ **0.20** |
| rise 10–90 % / overshoot | **0.233 s / 1 %** | 0.114 s / 19 % | 0.080 s / **56 %** |
| open-loop RMSE on the measured commands (rad) | the fit: 0.019 exact, 0.022 RK4-40 ms, 0.043 Euler-40 ms | 0.064 | **0.111** (max 0.25) |
| thrust gain (hover 0.4225 ± 0.003 N; m·g = 0.4256) | **1.00** | 0.968 (hover 0.4395 N: **+3.3 %**) | |
| linear drag (s⁻¹) | **0.495 xy / 0.544 z** (crazyflow's `drag_matrix`; fitted z 0.537) | none | |
| thrust lag / pure delays | τ = 57 ms; ≤ 1 sample (20 ms) | none | |
| rotor spin-up from rest at full thrust | nothing leaves the floor before **0.14 s** (0.18 s at 0.6 N, 0.25 s at 0.5 N); z = 0.7 m at 0.55 s, 1.0 m at 0.64 s | | |

In the MPC's own convention ddθ = a·θ + b·θ̇ + c·u, the fitted roll/pitch parameters are
**(−338.1, −40.8, 318.3)** — the mean over all six step episodes, which is what `att=sim`
carries — against so_rpy's (−189.0, −12.8, 138.1). The ratio c/(−a) is the dc gain — 0.94
against 0.73 — and −b/(2√(−a)) the damping. **Mind which fit you are quoting:** the table
above, and `att=sim`, are the mean over *both* step sizes. Run only the 0.3 rad probe and you
fit (−282.9, −35.4, 266.3) instead — the spec `att=sim03`, ωₙ 16.8, ζ 1.05 — which is a
different drone on paper and the same one in the air (both are within 0.02 rad of the
simulator, against so_rpy's 0.064). `probe_fit.py` names whichever preset your own fit is
nearer to and prints the exact `ka`/`kb`/`kc` override, so you never fly parameters you did
not measure. With both probes run, expect it to say `att=sim`, about 0.1 % away. Read the table's last column once more: the model is not only wrong in gain
and shape, its Euler discretisation at 40 ms makes it *worse* (a 56 % phantom overshoot
where the simulator has 1 %); with the *fitted* parameters, Euler at 40 ms is still
acceptable (0.043 rad) and RK4 is as good as exact (0.022 vs 0.019). The parameters are
the first-order error, the integrator the second — the answer to H3's kill criterion,
before a single closed-loop run.

Four consequences, each used later:

1. **The MPC under-commands tilt by ≈ 30 % in every turn.** With a 0.73 gain in the model
   it asks for less roll than the manoeuvre needs and gets 0.94 of *that*; the inside cut
   and the hairpin lag of §1 are this model error, not a controller-gain problem (§4 proves
   it: the model fix, with the vendored weights, removes both).
2. **Thrust and drag.** The model's hover is 3.3 % too high — the 0.03 m constant offset
   of §1 in the other direction is the closed loop fighting it — and 2 m/s² of drag at
   4 m/s is missing from the prediction. A plan at 0.95 × TWR asks 17.5 m/s² where the
   rotors give 18.4 minus 2.0 of drag at top speed: **physically infeasible**, which is why
   every 0.95 plan of Lessons 6–8 ends on a contact. 0.85 (15.4 + 2.0 = 17.4) keeps a 5 %
   margin; **0.88–0.90 × TWR is the ceiling** unless the planner models drag.
3. **A ground start needs a 0.15–0.20 s hold and no vertical phase.** Nothing leaves the
   floor before 0.14 s; thrust reaches 90 % of the command at 0.19 s; and the attitude loop
   works *while the drone sits on the floor*: a pitch of 0.3 rad commanded from t = 0 is
   0.19 rad at lift-off and 0.27 rad at 0.30 s, and the tilted full-thrust climb reaches
   1.0 m only 37 ms after the level one while already 1.0 m downrange at 2.9 m/s (receipt:
   `physics/logs/batch_tilt.log`). So the plan may accelerate towards gate 1 from its first
   sample after the hold; §3's 0.2 s hold is this number.
4. **No lookahead beyond one step is justified** (delays ≤ 20 ms): H4's "pure lag" has no
   physical carrier except the drag, which is a model term, not a delay.

**Caveats you write next to these numbers:** the fit describes *this simulator* (crazyflow
0.3.2's Mellinger loop and rotor model), not a Crazyflie; the ARX model covers the first
0.5 s of a step (the slow tail from 0.94 to 1.00 is outside it); yaw was not probed; and
`drag = 0.495` is crazyflow's parameter (`drones/params.toml`), quoted, not fitted — the
fitted value is the z axis's 0.537.

## 3. Replan from the ground

H1 says the two rest-to-rest manoeuvres before gate 1 are the largest lever. Lesson 8
extends `togt_plan.py` with the flags a ground start needs — `--init-pos X Y Z` and
`--bound-z LO HI` (the parameter files' `boundZ` starts at 0.15 m; a track whose
`initState` is below that gets `0.0 3.0` automatically), `--end-pos`, `--end-vel`,
`--via-radius` for H2/H6, and `--hold T` (default 0.2) for the diagnostics — and three
tracks next to the pole-aware one, each one edit apart from the last: `ground`
(`togt/lsy_level2_ground.yaml`: `initState` at (−1.5, 0.75, 0.05), everything else the
pole-aware track), `ground-b08` (its gates 1 and 2 pre-shifted — you meet it below) and
`ground-b08g3` (the headline track, gate 3 shifted as well). Read the three headers: the
diff between them *is* this section. Plan them (main venv, the driver from Lesson 6 §2):

```bash
python tasks/racing/code/togt_plan.py --track ground --thrust-frac 0.85            # -> tasks/racing/plans/ground_f0.85.csv
python tasks/racing/code/togt_plan.py --track ground-b08 --thrust-frac 0.85        # -> tasks/racing/plans/ground-b08_f0.85.csv
python tasks/racing/code/togt_plan.py --track ground-b08g3 --thrust-frac 0.90      # -> tasks/racing/plans/ground-b08g3_f0.90.csv
```

The diagnostics gain three lines: the ground-clock lap with the hold (`planned lap +
T`, valid because the plan's first point *is* the start pose, so `GroundStartTrajectory`
degenerates to a pure hold), z / vz / thrust over the first 0.5 s, and the closest passage
of every pole with its time, height and speed. In the archive the two plans are
`ground_f0.85.csv` and `ground-b08g3_f0.90.csv` (the archive calls them `gz05_f085` and
`gz05_b08_g3_f090`, `gz05` being its name for an `initState` at z 0.05; receipts:
`planner/logs/plan_b1.out`, `plan_b11.out`).

**The ground start is worth ≈ 1 s on paper.** The unbiased plan reaches gate 1 in
**0.97 s** (0.98 from z 0.01) instead of 2.32; the other legs are unchanged (0.94 / 1.20 /
0.92 s against the adopted plan's 0.90 / 1.20 / 0.91), so the planned lap is 4.02 s at
0.85 × TWR (top speed 4.55 m/s, thrust 15.5 of 17.5 m/s², every crossing 0.050 m off
centre, clearance 0.25 / 0.32 / 0.30 / 0.20 m) and 3.69 s at 0.95. The plan lifts 0.09 m in
its first 0.2 s, passes pole 1 at z 0.65 m *beside* it — and the harness tracks it as well
as it tracks the takeoff: 4.319 s on the ground clock with a 0.3 s hold, max deviation
0.209 m against 0.319 for the adopted plan (receipt: `planner/logs/h_b1.out`). H1's
kill criterion about the lift-off did not fire.

**And the vendored MPC cannot fly it.** Level 0, plain `mpc` through the unmodified
bridge, hold 0.3: **0/5 at every fraction and every start height** (`gz01/gz05/gz10` ×
0.85/0.95), and 0/5 or 1/5 with the plan stretched ×1.10 / ×1.20 — always gate 1's frame at
1.16–1.26 s, always the same place: the flown crossing sits **0.13–0.16 m sideways** from
the centre while the vertical error is ≤ 0.02 m (receipts: `planner/logs/r_gz0?_f0?5_h03.log`,
`analysis_g1.log`). The reference crosses at −0.05 (TOGT's window edge), the tracker adds
0.10 m of the same cut at 4 m/s, and 0.13 m is a contact. Slowing the plan barely moves it
(−0.15 → −0.13 → −0.12 m for −15 % of speed): a structural bias of the tracker in a turn,
the §1 cut at a faster crossing. H1's *other* kill criterion — "the first leg becomes
untrackable: contact at gate 1" — fired, but the cause is the tracker, not the lift-off.
(Only the z-0.05 start survived a stretch: `ground_f0.85` ×1.10 at 5/5, 4.72 s — the first
ranked-looking ground start, −0.60 s, and the wrong road.)

**Three planner-side answers, screened** (5 episodes each unless stated; receipts:
`planner/logs/RANKING.md` — 51 rows — and the `r_<plan>_h<hold>.log` it names):

1. *An entry corridor 0.3 m before gate 1* (`g1in30`) does what it is built for — the
   flown gate-1 offset drops from −0.15 to −0.02…−0.05 m — and the lap dies elsewhere (2/5:
   gate 1's frame *corner* on the return leg, gate 3).
2. *Tighter gate windows* (`--margin 0.36`, ±0.02 m) centre every crossing and, because
   every turn then starts later, shrink the cut everywhere: **5/5 at 4.62 s** unstretched
   (`gz01_m36_f085`) — for +0.29 s of planned lap.
3. **Pre-compensation.** If the tracker's cut is systematic, shift the *reference* against
   it: gate 1 and its corridors +0.08 m along gate 1's +y (against the measured −0.10 m
   cut), gate 2 and its corridors −0.08 m along gate 2's +y *and 0.06 m up* (against its
   +0.13…+0.18 m cut and the 0.1 m climb lag of §1). That is `b08`. The reference's own
   gate-2 crossing then sits at (−0.13, +0.11) from the true centre — exactly the planner's
   0.13 m bound (0.20 − the 0.07 m box); the next size, `b10`, puts it at 0.15 and the
   planner's own check calls it a FRAME crossing. The flown crossings land within 0.02 m
   (gate 1) and 0.06 m (gate 2) of the true centres: **`ground-b08_f0.85`, hold 0.3: 5/5 at
   4.38 s**; hold 0.25: 5/5 at 4.34; **hold 0.2: 15/15 at 4.28 s** (receipts:
   `r_gz05_b08_f085_h03.log`, `_h025.log`, `_h02.log`, `_h02_n10.log`).

> 🔑 **The hold is a pure additive term**: 0.2 / 0.25 / 0.3 s → 4.28 / 4.34 / 4.38 s at
> equal success. §2's 0.15–0.20 s spin-up is all the plan needs, and the rotors and the
> attitude loop are already working during it. A longer hold buys nothing but time — and
> Lesson 6 §5's settle hold (1.0 s at a hover start) was the same physics with a 0.5 m
> sag on top.

**0.90 × TWR and the gate-3 shift.** At 0.90 the planned lap drops by 0.17 s
(`ground-b08_f0.90`: 3.925 s; `gz01_b08_f090`: **15/15 at 4.24 s** with hold 0.3), and the
0.90 family loses ≈ 1 episode in 10 — 51/55 pooled over five plans, *every* failure a
gate-3 frame with the flown crossing at +0.11/+0.12 m on the inside of the hairpin exit
(the successes cross at +0.07…+0.10, the 0.85 plans at +0.04…+0.08): the reference sits at
+0.05 and the drone lags it by ≈ 0.07 s there. So shift gate 3 with its entry corridor
−0.05 m along its +y (`g3`; it costs nothing in lap time, and pulls the pole-3 reference
clearance from 0.30 to 0.28 m). The flown gate-3 crossing moves to +0.02…+0.07, and the
**headline plan is born: `ground-b08g3_f0.90` — planned 3.924 s, gate 1 at 0.966 s, top speed
3.86 m/s, thrust 16.5 of 17.5 m/s², crossings 0.03 / 0.13 / 0.03 / 0.05 m off centre,
clearance 0.24 / 0.24 / 0.28 / 0.20 m, and 4.124 s on the ground clock with a 0.2 s hold,
which the race reads as 4.12** (receipt: `planner/logs/plan_b11.out`). Screens: hold 0.3
5/5 then 9/10 (14/15 at 4.22 s); **hold 0.2: 5/5 at 4.12 s** (receipts:
`r_gz05_b08_g3_f090_h03.log`, `_h03_n10.log`, `_h02.log`). Its pole-4 twins with the via
moved 0.05 m away (`p5`: flown pole-4 clearance 0.15–0.17 m instead of 0.11–0.13) cost
+0.04–0.10 s: `gz05_b06_p5_g3_f090` 15/15 at 4.22, `gz05_b08_p5_g3_f090` 15/15 at 4.26
(archive variants: not in this repo, but `--via-radius` and a copied track file rebuild one
in a minute — and §7 says why you might want to).
§6 races the headline at 20 episodes and both levels.

**Dead ends, all raced or planned, so that you do not repeat them:**

| lever | what happened | verdict |
|---|---|---|
| end state further past gate 4, or crossing gate 4 at 3 m/s (`--end-pos`, `--end-vel`, `--coast`) | +0.003 / +0.007 / −0.002 s: the last 0.9 s are shaped by the pole-4 detour, not by the stop | **H2 refuted** |
| larger via balls, r 0.15 → 0.18 / 0.20 (`--via-radius`) | −0.054 / −0.093 s on paper, and the optimiser cuts *closer* to the poles (pole 4: 0.203 → 0.183 → 0.170 m) — a larger ball is a larger region to cut through | **H6 works the other way round**: to buy clearance, move the via centre (`p5`), at a cost in time |
| a climb via above the start | +0.019 s; the free plan already climbs | dead |
| entry corridors (`g1in30`, `g2in80`) | 2/5–4/5: the gate-1 → gate-2 climb gets steeper and gate 2 is hit low | dead |
| tight windows on top of `b08` (`m34`, `m36`) | +0.20 / +0.30 s for nothing (5/5, 4/5) | dead |
| 0.95 × TWR, any track | the harness cuts 0.35–0.53 m and passes pole 4 at 0.10–0.11 m; the race: 0/5 everywhere (§2 said why) | infeasible without a drag-aware planner |
| time-stretching an unbiased plan | ×1.10: 5/5 at 4.72 s (gz05 only), 0/5 for gz01 up to ×1.20 | the cut is not a speed effect |

## 4. Fix the model, not the gains

§2 measured what the model gets wrong; H3 predicted that fixing the integrator and the
weights halves the deviation. Lesson 8 ships `tasks/racing/code/mpc_dev.py`: the vendored
`MPCController` re-implemented with switches, **bit-exact at the defaults** (max |Δu| = 0
over 40 steps of the same state sequence at 50 and 100 Hz, for the plain, ESO and L1
variants; `test_race_refs.py` checks 10 steps of it in the main venv; receipt:
`mpcgap/logs/check_dev.log`, `scaffold2/check_dev2.log`). The vendored `mpc.py` stays as
identified upstream; the corrected model lives here. Spec grammar, accepted by
`race_eval.py --controller`, the bridge's `RACE_CONTROLLER` and `race_runner.py`:

```
mpcdev                       the vendored controller, exactly
mpcdev:key=val,key=val,...   unknown keys raise
```

| key | default | meaning |
|---|---|---|
| `int` | `euler` | discretisation over `dtp`: `euler` (vendored) or `rk4` |
| `H`, `dtp` | 20, 0.04 | horizon steps and prediction step (0.8 s) |
| `wp`, `wv`, `wu`, `wf`, `wdu`, `wt` | 1, 0.05, 0.02, 0.02, 0.1, 0 | position, velocity, roll/pitch, thrust-from-hover, input-rate, terminal weights |
| `tau` | 0 | lookahead [s]: the reference sampled at t + τ + dtp·k (H4) |
| `wa`, `accff` | 0, auto | reference-acceleration feed-forward (H4) |
| `dist`, `eso_w`, `l1_as`, `l1_c` | none, 7, −5, 4 | the vendored ESO / L1 (`mpc_offsetfree` = `dist=eso`, `mpc_l1` = `dist=l1`) |
| `ramp`, `ramp_t`, `liftoff_dz` | time, 1.5, 0.03 | the estimator's soft-start: from t = 0 (vendored), from lift-off, or none |
| `mass`, `mass_gain`, `mass_min`, `mass_max`, `mass_delay` | 0, 2, 0.8, 1.25, 0.3 | a thrust-scale adaptation k on the vertical residual, once airborne (§5) |
| `rp_max`, `yaw_max`, `iter`, `tol` | 1.0, 0.3, 60, 1e-4 | bounds and ipopt settings |
| **`att`** | `sorpy` | roll/pitch (a, b, c): `sorpy` = crazyflow's (−189.0, −12.8, 138.1), **`sim`** = §2's six-episode fit (−338.1, −40.8, 318.3), `sim03` = the 0.3-rad-step fit
(−282.9, −35.4, 266.3); `ka`, `kb`, `kc` override any of them |
| **`drag`**, `dragz` | 0, = drag | linear drag in the translational model (the simulator: 0.495 xy, 0.544 z) |
| **`fgain`** | 0.968 | thrust gain; `fgain=1.0` also recomputes the hover thrust (0.4256 N instead of 0.4395) |
| `flag` | 0 | a first-order thrust lag as a 13th state (measured τ = 0.057 s) |

The named specs of this lesson: **M1 = `mpcdev:att=sim,drag=0.495,fgain=1.0`** (the
measured model — §2's three first-order corrections, with the vendored weights, horizon
and Euler step) and **M1+mass = `mpcdev:att=sim,drag=0.495,fgain=1.0,mass=1`** (§5). Quote
the spec in your shell if it complains about the commas (`docs/4-troubleshooting.md`). Run
the two harness twins yourself (main venv, one deterministic episode each, ≈ 30 s; the
`SOLVE` line is new):

```bash
python tasks/racing/code/race_eval.py --plan tasks/racing/plans/ground_f0.85.csv --controller mpcdev --start ground --takeoff-t 0.3 --reason "L8 model"
python tasks/racing/code/race_eval.py --plan tasks/racing/plans/ground_f0.85.csv --controller mpcdev:att=sim,drag=0.495,fgain=1.0 --start ground --takeoff-t 0.3 --reason "L8 model"
```

**The model is the whole gain** (receipts: `controller/logs/h_p0_m{0,1}.log`,
`h_p2_m{0,1}.log` — one deterministic harness episode each, no contacts, no noise;
`controller/logs/r_p2_m1*.log`, `r_p1_m{0,1}.log`, `r_p0_m1.log`; `protocol/logs/`):

| quantity (Level 0) | vendored `mpc` | M1 | where |
|---|---|---|---|
| harness max deviation, adopted plan, takeoff 1.5 | 0.319 m (rmse 0.130) | **0.100 m** (0.056) — **× 3.2** smaller | `h_p0_m{0,1}` |
| harness max deviation, unbiased ground start, hold 0.3 | 0.209 m | **0.096 m** | `h_p2_m{0,1}` |
| harness gate offsets G1..G4, unbiased plan | 0.116 / 0.112 / 0.079 / 0.058 | 0.080 / 0.039 / 0.091 / 0.085 | harness |
| harness gate-3 crossing, adopted plan (plan 4.42 s) | 4.496 s — the §1 lag | **4.427 s** | harness |
| flown pole-4 clearance (plan 0.20) | 0.13 m | **0.19–0.20 m** | harness and race |
| race, unbiased `ground_f0.85`, hold 0.3 | **0/5** (the gate-1 cut of §3) | **13/15 at 4.295 s**; protocol F1 **18/20 at 4.296 ± 0.009** | `r_p2_m1`, `r_p2_m1_x10`; `gz05_f085_M1_h03_L0` |
| race, `ground-b08_f0.85`, hold 0.3 | 5/5 at 4.380 (dev 0.22–0.40) | 5/5 at 4.368 (dev 0.10) | `r_p1_m{0,1}` (screens) |
| race, adopted plan, takeoff 1.5 | 15/20 at 5.32 (Lesson 7) | **20/20 at 5.30** (protocol G1) | `poles_f085_M1_t15_L0` |
| solve time per 20 ms step, idle machine | 14–19 ms | 15–19 ms | protocol headers |

Read the race rows against §1 and §3. The unbiased ground-start plan, which the vendored
MPC could not pass gate 1 on, is *legal* with the corrected model: flown gate 1 at
−0.07/−0.08 (reference −0.05; the vendored tracker −0.13…−0.16), gate 2 at +0.02 with no
climb lag, the whole lap within 0.10 m of the reference. On the adopted plan the §1 lag
is gone — the harness gate-3 crossing is 7 ms late instead of 76 — and so is the 0.04 m cut
at gate 1 (flown −0.05/−0.06 = the reference's −0.05). No weight was touched.

**The 40-cell sweep — every other knob is neutral or worse** (harness, unbiased plan P2
and adopted plan P0, one deterministic episode each; receipt: `controller/logs/harness_table.md`
and the `h_p{2,0}_<tag>.log` it names; race screens in `controller/logs/r_*.log`):

| variant, on M1 unless stated | P2 max dev / rmse (m) | P0 max dev | where it fails | verdict |
|---|---|---|---|---|
| **M1** | **0.096 / 0.056** | **0.100** | gate 3 at 0.09 (the residual) | the baseline |
| `wv=0.1` | 0.088 / 0.052 | — | −1 cm at every gate, −2 cm at pole 1 | best screen cell; the race: 4/5 with a 0.21 m transient — noisier, not more precise |
| `int=rk4` | 0.093 / 0.037 | 0.096 | rmse only; solve 99–106 ms per step under load (23 ms alone) | nothing for 4–7× the compute; race 2/5 |
| `H=30` / `wt=5` / `dragz=0.544` / `att=sim03` / `drag=0.6` | 0.095–0.100 | 0.091–0.104 | within 5 mm of M1 | inert (H = 30 at 2× the solve time) |
| `flag=0.057` | 0.107 / 0.061 | 0.106 | gate 2 −28 mm, pole 1 +3 cm, gate 4 +8 mm | a different shape, not a better one; race 4/5 |
| `tau=0.02` | = M1, every crossing exactly 20 ms earlier | = M1 | — | **a pure time shift**: no residual lag to remove (H4) |
| `wdu=0.03` / `wu=0.01` / `rp_max=0.8` | 0.193 / 0.127 / 0.172 | 0.111 / — / 0.168 | pole 1 (first-leg turn at 4.5 m/s), gate 3 | worse |
| `wp=3` / `wp=5` | 0.188 / 0.287 | 0.176 / 0.383 | gate 3 at 0.19 (P0), pole 3 at **0.01** (`wp=5`, P0) | monotonically worse: 1 → 3 → 5 gives 0.10 → 0.18 → 0.38 |
| `wv=0.2` / `wv=0.5` | 0.258 / 0.352 | 0.163 / 0.384 | pole 1 at 0.10 / 0.06 | worse |
| `wa=0.05` (acceleration feed-forward) | 0.189 / 0.121 | 0.199 | gate 1 at 0.12–0.14, crossed 0.04 s early | runs ahead of the plan; worse |
| `iter=150,tol=1e-6` | = M1 | | ipopt's 60-iteration cap is not binding | inert |
| vendored `mpcdev` (= `mpc`) | 0.209 / 0.123 | 0.319 | pole 4 at 0.13 | the control |

Every gain increase makes the closed loop *worse*, with the damage concentrated at pole 1
(the first-leg turn) and the hairpin: with the 40 ms Euler prediction the vendored cost is
at a local optimum on both models. H3 was half right — the *parameters* are the cause
(0.319 → 0.100 m), the integrator is second-order (RK4 −4 mm at 4–7× the cost), and every
weight change loses. H4 is refuted in its lag form: the vendored "lag" was drag and the
tilt gain, i.e. model mismatch, and a lookahead on the corrected model is a time shift.

> 🔑 Fix the model, not the gains. Forty cells of weights, horizons, bounds and
> feed-forwards, screened in the harness and the top ones raced, and not one beats the
> default cost on the corrected model; the three numbers §2 measured in an hour of probes
> cut the deviation × 3.2. Write the sweep down anyway — "the cost is at a local optimum"
> is a result you can only claim after you looked.

**Where M1 stops** — and why it is *not* the headline tracker:

1. **The 0.90 × TWR plans.** M1 fails where the vendored MPC succeeds: `gz01_b08_f090`
   **0/5** on pole 1 at t = 2.76 s with a 0.21–0.27 m deviation at 3.0–3.3 m/s (the vendored
   MPC: 15/15); the unbiased `gz05_f090` 1/5 (receipts: `r_p3a_m1.log`, `r_p3b_m1.log`; 10
   episodes, never raced at 20). The window is the acceleration out of the G2-exit hairpin,
   where the plan demands 16.3 of the 17.5 m/s² and drag takes the rest of the headroom
   (§2): the tighter-tracking model follows the demand into the margin, the sluggish
   vendored one stays wide. Precision is not safety.
2. **The gate-3 cut on every 0.85 ground-start plan.** M1's residual failure is a 3–4 cm
   inside cut at the gate-3 crossing (flown +0.08…+0.10 on the plan's +0.05/+0.06; contact
   at 0.10) that no cost knob moves — **2 to 8 losses in 20 in every protocol cell** with
   it (§6: A3–A6, B1, B2, B5, F1–F3), for any mass. The `b08`/`g3` pre-compensations were
   built against the *vendored* tracker's cut (M1 crosses gate 2 7 cm on the *other* side
   of the b08 shift); a pre-compensated plan is tracker-specific, and M1 would need its own.
3. **Its −0.02 s is not a plan gain.** M1's laps read 0.02 s faster than the vendored MPC's
   on the same plan (5.30 vs 5.32; 4.26 vs 4.28) because it crosses gate 4 10–30 ms *early*
   (harness: 5.317 s against the plan's 5.33) and the race clock floors to the 20 ms step
   (audit finding D3, §7). Same plan, same last-gate time on the reference, an earlier
   crossing on the floor clock.

## 5. Level 1: the mass, the estimators, the start

§1 read the mass through proxies. Lesson 8's `race_runner.py` reproduces lsy's `sim.py`
loop for any bridge and logs, per episode, what `sim.py` does not: the **sampled mass**
(it lives in `env.unwrapped.data.sim_data.params.mass` — the `Sim` object's own copy stays
nominal, and the controller never sees it), the start pose, the outcome, the failure
verdict, the gate crossings with lag and in-plane offset against the bridge's own
reference, the lift-off time, the cruise height, the pole-4 clearance, the solve time and
the thrust scale k (receipt: `level1/NOTES.md` §0, `level1/logs/inspect_params.log`). Race
venv, from `/workspace`; the same `RACE_*` knobs as the bridge; `--out-dir` defaults to
`tasks/racing/figures/lesson8` and gets `<label>.csv`, `<label>.log` and
`<label>_flown/flown_epNN.csv`:

```bash
cp tasks/racing/code/race_bridge_mpc.py repos/lsy_drone_racing/lsy_drone_racing/control/
RACE_PLAN=tasks/racing/plans/poles_f0.85.csv RACE_TAKEOFF_T=1.5 RACE_CONTROLLER=mpc \
  /opt/venvs/race/bin/python tasks/racing/code/race_runner.py --config level1.toml --n 20 --label adopted_mpc_L1 --out-dir tasks/racing/figures/lesson8
```

Every episode prints lsy's three stats lines and one `EP` line (`mass=0.03959 (-8.7%)
start=(...) -> gates=4 t=4.12 OK ... t_lift=0.38 dz_cruise=+0.004 ... | G1@1.19 lag+0.022
y-0.013 z+0.008 (ref y+0.05 z+0.00); G2@... ; G3@...`), and the run ends with
`<label>: k/n laps, mean lap X s (± std), light half a/b, heavy half c/d`. It works at
Level 0 too (every row at m/m₀ = 1.000). Adopted plan, takeoff 1.5 s, `level1.toml`
unmodified except `control_mode = "attitude"` (md5 `cf21841c…`; receipts:
`level1/logs/<label>.{log,csv}`, `level1/logs/summary.log`):

**What the mass does, measured** (vendored `mpcdev` = `mpc`, 20 episodes, m/m₀ from
0.89 to 1.11; receipt: `level1/logs/base_l1_mpcdev.csv`):

| feature | correlation with m/m₀ | light → heavy |
|---|---|---|
| mean height offset in cruise | **−0.94** | +0.04 → −0.07 m (the hover-thrust model offset scales with the mass) |
| minimum height offset | −0.96 | |
| gate-2 crossing height | −0.81 | +0.015 → −0.123 m (the bottom rail is at −0.18) |
| lag at gate 2 | +0.92 | 0.004 → 0.034 s |
| lift-off time | +0.90 | 0.28 → 0.36 s (a heavy drone waits longer for thrust > weight) |
| success | −0.25 | light half 4/8, heavy half 2/12 |
| success vs \|start offset\| / z₀ / \|roll₀\| + \|pitch₀\| | −0.02 / +0.07 / −0.18 | the start pose is irrelevant, with the mass now *measured* |

The heavy half is lost on the vertical channel (gate 2 crossed 0.10–0.25 m low; 5 of the 8
heaviest end on its frame), the rest on §1's Level-0 modes. **The variants** (10 episodes
each unless stated; every success 5.32 s with the vendored model, 5.30 with M1 — §4.3;
receipt: `level1/NOTES.md` §3.9 and the logs it names):

| spec (Level 1, adopted plan) | k/n | light half | heavy half | failure modes | log |
|---|---|---|---|---|---|
| vendored `mpcdev` (= `mpc`) | 6/20 (Lesson 7: 8/20 — both genuine, a ±2 spread) | 4/8 | 2/12 | 6× gate 2 low (heavy), 5× gate 1, 3× gate 3 | `base_l1_mpcdev` |
| `mpcdev:mass=1` (adaptation on the vendored model) | 7/10 | 2/2 | 5/8 | 3× gate 1's post on the return leg; k pumped *down* to 0.82–0.87 by the unmodelled thrust lag | `v_mass1` |
| M1 | 8/10; protocol **11/20** | 8/8 (20 ep.) | 3/12 (20 ep.) | gate 3: the heavy drones exit the hairpin 3–5 cm low and wide | `v_sim`; `protocol/logs/poles_f085_M1_t15_L1` |
| **M1 + `mass=1`** | 9/10; protocol **18/20 at 5.302 s** | 12/12 | 6/8 | 2× gate 3 | `v_sim_mass1`; `final_l1_sim_mass1` |
| M1 + full model (`dragz=0.544`, `flag=0.057`) + `mass=1` | 8/10 | 3/5 | 5/5 | 2× gate 3 (*light* drones); k within 0.01 of m₀/m | `v_full_mass1` |
| `dist=eso,ramp=liftoff,ramp_t=0.5` (vendored model) | **0/10** | 0/4 | 0/6 | 5× pole 1, 3× gate 3, 2× gate 1 | `v_eso_liftoff_r05` |
| `dist=eso,ramp=liftoff` (ramp 1.5 s) | **0/10** | 0/7 | 0/3 | 4× gate 3, 4× gate 1, pole 1 | `v_eso_liftoff` |
| `dist=eso` (= `mpc_offsetfree`, ramp on time) | 1/10 | 1/7 | 0/3 | 3× pole 1, 3× gate 1, 3× gate 3 | `v_eso_vendored` |
| `dist=l1,ramp=liftoff` | 2/10 | 1/2 | 1/8 | gate 1, gate 2, pole 1, gate 3 — two each; flies low everywhere | `v_l1_liftoff` |
| M1 + `dist=eso,ramp=liftoff` | 2/10 | 2/2 | **0/8** | 5× gate 3, 2× pole 1, a time-out | `v_sim_eso_liftoff` |
| M1 + z-only ESO (`dmask=z`, archive-only key) | 6/10 (its estimator-free twin: 8/10) | 5/7 | 1/3 | 4× gate 3 | `v_sim_esoz` |

Read the estimator rows with §2 in hand. The lift-off-gated ESO **fixes the vertical
channel** (gate-2 height −0.016 m against −0.061 for the baseline's failures) and **loses
the horizontal one everywhere** (deviation 0.29–0.71 m against 0.10–0.58): with the
vendored model it learns the so_rpy attitude error — gain 0.73 where the simulator gives
0.94 — as a lateral "disturbance" during one turn and applies it in the next. The ramp's
length and its gating change nothing (0/10 either way). On the corrected model it learns
the *mass-proportional thrust error* as a constant world-frame acceleration instead, and
loses the heavy half 0/8. An additive estimate is the wrong *structure* for an error that
scales with the thrust; the multiplicative thrust-scale adaptation is the matching one (k
tracks m₀/m with r = −0.97, converged before gate 1; receipt: `v_sim_mass1_flown/ktrace_epNN.csv`).
That is what Lessons 6 and 7 saw as "the estimators lose laps on the ground clock" — it
was never the takeoff, it was the model (both lessons now say so). The one exception the
audit insisted on: the z-only ESO on M1 sits at 6/10, above the vendored baseline and two
laps below its estimator-free twin — 10-episode noise, not a lever.

**Two honest costs.** The adaptation *loses* Level-0 laps against M1 alone on the same
plan — 15/20 against 20/20, all five at gate 3 (receipt: `final_l0_sim_mass1`; k drifts to
1.04–1.18 during the gate-4 deceleration, the unmodelled thrust lag, and moves the gate-3
crossing across its edge) — while matching the vendored 15/20. And on the ground-start
plans it is neutral at Level 1 and harmful at Level 0 (§6: F3 12/20 against F1 18/20).
H5 is confirmed *for the adaptation on the adopted plan* and refuted for the estimators.

**The start blend.** A ground-start plan at Level 1 needs one more thing. The race draws
the start ±0.1 m while the plan's first point is fixed; the bridge's hold is a rest-to-rest
quintic from the *observed* start to that point, and a rest-to-rest quintic's peak
acceleration is 5.77·d/T²: over 0.3 s that is 5.3 m/s² for the 0.083 m draw that failed and
9 m/s² at the ±0.14 m corner of the draw — asked of a drone whose rotors are still spinning
up. It tilts on the floor, slides 2 cm
and the episode ends at 0.28 s with 0 gates (receipt: `mpcgap/logs/scaffold2/s2d_race_l1_gz05b08_blend0.log`,
`blend_first.log`). `RACE_START_BLEND=T` (new in the bridge; `race_refs.StartBlendTrajectory`)
holds the drone *where it is* during the hold and fades the offset out with a quintic over
T = 1.0 s: peak 0.85 m/s² for a 0.14 m draw, ≤ 5 mm of shift left at pole 1, < 1 mm at
gate 1 — it cannot move the reference into anything. With it the vendored MPC laps
`ground-b08_f0.85` at Level 1 **10/10 at 4.38 s** (T = 0.6: 9/10; receipts:
`g_b08_mpcdev_b10`, `g_b08_mpcdev_b06`). At Level 0 it is a no-op only for plans whose
first point *is* the start pose; for the z-0.05 plans it replaces the vendored 4 cm climb
during the hold by a 4 cm fade — no measurable effect (audit D1). Default 0, the Lesson-6
behaviour; set it for every ground-start cell at Level 1.

## 6. The protocol: 28 cells, two levels, one headline

Now the leaderboard's own protocol (Lesson 3 §3 and §5: 20 episodes, mean ± std over the
successes, ≥ 10/20 to rank, `level0.toml` / `level1.toml` unmodified except
`control_mode = "attitude"` — md5 `035ad88b…` / `cf21841c…`, proven from the pinned commit
by the audit). The four cells you run yourself (race venv, from `/workspace`; the bridge
installed as in §5):

```bash
RACE_PLAN=tasks/racing/plans/ground-b08g3_f0.90.csv RACE_TAKEOFF_T=0.2 RACE_START_BLEND=1.0 RACE_CONTROLLER=mpc \
  /opt/venvs/race/bin/python tasks/racing/code/race_runner.py --config level0.toml --n 20 --label headline_L0 --out-dir tasks/racing/figures/lesson8
RACE_PLAN=tasks/racing/plans/ground-b08g3_f0.90.csv RACE_TAKEOFF_T=0.2 RACE_START_BLEND=1.0 RACE_CONTROLLER=mpc \
  /opt/venvs/race/bin/python tasks/racing/code/race_runner.py --config level1.toml --n 20 --label headline_L1 --out-dir tasks/racing/figures/lesson8
RACE_CONTROLLER=mpcdev:att=sim,drag=0.495,fgain=1.0,mass=1 RACE_TAKEOFF_T=1.5 RACE_PLAN=tasks/racing/plans/poles_f0.85.csv \
  /opt/venvs/race/bin/python tasks/racing/code/race_runner.py --config level1.toml --n 20 --label adopted_M1mass_L1 --out-dir tasks/racing/figures/lesson8
RACE_CONTROLLER=mpcdev:att=sim,drag=0.495,fgain=1.0 RACE_TAKEOFF_T=1.5 RACE_PLAN=tasks/racing/plans/poles_f0.85.csv \
  /opt/venvs/race/bin/python tasks/racing/code/race_runner.py --config level0.toml --n 20 --label adopted_M1_L0 --out-dir tasks/racing/figures/lesson8
```

A 20-episode MPC cell takes 75–105 s on an idle 14-core machine. Expect **4.12 s, 20/20**
and **4.12 s, 16–18/20** for the first two, **5.30 s, 18/20** and **5.30 s, 20/20** for the
last two.

**Verified (2026-09-18; 20 episodes per cell; mean ± std over the successful episodes;
below 50 % unranked; M1 = `mpcdev:att=sim,drag=0.495,fgain=1.0`, "+mass" = `,mass=1`;
hold = `RACE_TAKEOFF_T`; every ground-start cell with `RACE_START_BLEND=1.0`; `*` = a
Level-1 cell of §5 cited, not re-run. Three of the plans are yours to regenerate —
`ground_f0.85`, `ground-b08_f0.85` and `ground-b08g3_f0.90` come from `--track ground`,
`ground-b08` and `ground-b08g3` at the thrust fraction in the name. The two `gz*` rows are
archive variants kept for the comparison they make (a pole-4 via moved 0.05 m out, and the
same pre-compensation from a z-0.01 start); they are not in this repo. Receipts: `protocol/logs/master_table.md` and one
`protocol/logs/<plan>_<spec>_<hold>_<level>.{log,csv}` per cell):**

| cell | plan | hold | tracker | level | k/20 | lap (s) | failures | solve ms |
|---|---|---|---|---|---|---|---|---|
| **C1** | ground-b08g3_f0.90 | 0.2 | vendored `mpc` | 0 | **20/20** | **4.120 ± 0.000** | — | 17 |
| C1r | same (replicate) | 0.2 | vendored `mpc` | 0 | **20/20** | 4.120 | — | 14 |
| **C2** | ground-b08g3_f0.90 | 0.2 | vendored `mpc` | 1 | **18/20** | **4.120 ± 0.000** | gate 1 (+7.9 % mass, 0.35 m low), gate 2 (+11.5 %, 0.32 m low) | 17 |
| C2r | same (replicate) | 0.2 | vendored `mpc` | 1 | **16/20** | 4.120 | 2× gate 2 (+10.2 / +9.9 %), gate 3 (+4.7 %), pole 1 (a floor slide: +7.7 %, a 0.11 m start draw, never lifted) | 18 |
| C4 | ground-b08g3_f0.90 | 0.3 | vendored `mpc` | 0 | 20/20 | 4.220 | — | 14 |
| C3 | ground-b08g3_f0.90 | 0.3 | vendored `mpc` | 1 | 18/20 | 4.220 | gate 3 (+9.2 %), gate 2 (+11.5 %) | 17 |
| D1 | gz05_b06_p5_g3_f090 | 0.3 | vendored `mpc` | 0 | 19/20 | 4.220 | gate 3 | 18 |
| D2 | gz05_b06_p5_g3_f090 | 0.3 | vendored `mpc` | 1 | 15/20 | 4.220 | 4× gate 3, gate 1 | 17 |
| E2 | gz01_b08_f090 | 0.3 | vendored `mpc` | 0 | 19/20 | 4.240 | gate 3 | 18 |
| E1 | gz01_b08_f090 | 0.3 | vendored `mpc` | 1 | 15/20 | 4.239 ± 0.005 | 3× gate 3, gate 1, gate 2 | 17 |
| A1 | ground-b08_f0.85 | 0.2 | vendored `mpc` | 0 | 19/20 | 4.280 | gate 1 | 19 |
| A2 | ground-b08_f0.85 | 0.2 | vendored `mpc` | 1 | 17/20 | 4.282 ± 0.007 | 3× gate 2 (three of the five heaviest draws, +8.4…+9.4 %; the heaviest of all completed) | 18 |
| A3 | ground-b08_f0.85 | 0.2 | M1 | 0 | 17/20 | 4.260 | 3× gate 3 | 19 |
| A4 | ground-b08_f0.85 | 0.2 | M1 | 1 | 14/20 | 4.267 ± 0.010 | 6× gate 3 | 18 |
| A5 | ground-b08_f0.85 | 0.2 | M1 + mass | 0 | 14/20 | 4.261 ± 0.005 | 6× gate 3 | 16 |
| A6 | ground-b08_f0.85 | 0.2 | M1 + mass | 1 | 16/20 | 4.260 | 4× gate 3 | 16 |
| B3* | ground-b08_f0.85 | 0.3 | vendored `mpc` | 0 | 19/20 | 4.380 | gate 1 | 18 |
| B4* | ground-b08_f0.85 | 0.3 | vendored `mpc` | 1 | 19/20 | 4.380 | gate 1 (a light drone) | 26 |
| B1 | ground-b08_f0.85 | 0.3 | M1 | 0 | 15/20 | 4.361 ± 0.005 | 5× gate 3 | 18 |
| B5 | ground-b08_f0.85 | 0.3 | M1 | 1 | 16/20 | 4.366 ± 0.010 | 4× gate 3 | 16 |
| B2 | ground-b08_f0.85 | 0.3 | M1 + mass | 1 | 14/20 | 4.361 ± 0.005 | 6× gate 3 | 18 |
| F1 | ground_f0.85 (unbiased) | 0.3 | M1 | 0 | 18/20 | 4.296 ± 0.009 | 2× gate 3 | 16 |
| F3 | ground_f0.85 (unbiased) | 0.3 | M1 + mass | 0 | 12/20 | 4.297 ± 0.008 | 8× gate 3 | 16 |
| F2 | ground_f0.85 (unbiased) | 0.3 | M1 + mass | 1 | 12/20 | 4.293 ± 0.010 | 8× gate 3 | 16 |
| G1 | poles_f0.85 (adopted) | takeoff 1.5 | M1 | 0 | **20/20** | 5.300 | — | 15 |
| G2 | poles_f0.85 (adopted) | takeoff 1.5 | M1 | 1 | 11/20 | 5.302 ± 0.006 | 9× gate 3 (heavy) | 16 |
| G3* | poles_f0.85 (adopted) | takeoff 1.5 | M1 + mass | 0 | 15/20 | 5.301 ± 0.005 | 5× gate 3 | 34 |
| G4* | poles_f0.85 (adopted) | takeoff 1.5 | M1 + mass | 1 | **18/20** | 5.302 ± 0.006 | 2× gate 3 | 37 |
| ref | poles_f0.85 (adopted) | takeoff 1.5 | vendored `mpc` | 0 / 1 | 15/20 / 8/20 (§5: 6/20) | 5.320 | Lesson 7 protocol A | 14–23 |

**Read them together** — Lesson 3's rules, applied:

1. **The plan sets the lap, the vendored tracker only decides survival.** Std 0.000 in every
   vendored *Level-0* cell: each success reads the reference's last-gate time floored to 20 ms.
   At Level 1 the start draw and the mass move it by one step in a few episodes (A2 4.282 ±
   0.007, E1 4.239 ± 0.005) — still the plan's time, read on either side of a 20 ms boundary. The
   headline — `ground-b08g3_f0.90`, hold 0.2, `RACE_START_BLEND=1.0`, the **unmodified
   vendored MPC**: **4.120 s at 40/40 (Level 0) and 34/40 = 85 % (Level 1; light half
   22/22, heavy half 12/18)** — is 1.20 s under Lesson 7's 5.32 s *at higher success at both
   levels*, and 0.73 s from the record. The second-best is the same plan at hold 0.3
   (4.220 s, 20/20 / 18/20): the hold's +0.10 s and nothing else (C1/C4, C2/C3, A1/B3).
2. **What Level 1 still loses with the winner** is the heaviest ~10 % of the draws
   (m/m₀ ≥ 1.08) crossing gates 1–2 0.25–0.35 m low on the 0.90 plan — its climb asks 15.9 of
   17.5 m/s² with no margin for +10 % of mass — plus one floor slide at a large start draw.
   A 0.3 s hold does *not* recover them (C3 loses the same two drones for +0.10 s); the 0.85
   plan does (B4*: 19/20), for +0.26 s.
3. **The corrected model tracks every plan 3× tighter and ranks lower on every ground-start
   plan** (A3 vs A1, B1 vs B3*, F1 vs A1): its gate-3 crossing sits at +0.08…+0.10 on the
   frame's edge for any mass (§4). Its value is the adopted plan: 20/20 at 5.30 (G1) — the
   first clean Level-0 cell on that plan — and, with the adaptation, 18/20 at Level 1 (G4*).
4. **The adaptation pays only there** (G2 → G4*: 11 → 18/20) and costs Level-0 laps on the
   same plan (G1 → G3*: 20 → 15/20) and everywhere else (F1 → F3: 18 → 12/20).
5. **Solve time is 14–19 ms per 20 ms step for every tracker** on an idle machine, with
   per-episode maxima of 41–216 ms; the synchronous simulator does not penalise a late step,
   a real Crazyflie would fly on stale commands (§7).

**Verdicts:** H1 ✓ with a twist (the planned 0.98 s + a 0.2 s hold replace 2.32 s; the
unbiased plan needs the corrected model, 18/20 at 4.30 s; the pre-compensated one flies
with the vendored MPC, 40/40 at 4.12 s). H2 ✗ (≤ 0.01 s). H3 half — the *parameters*
(gain 0.73 → 0.94, ζ 0.47 → 1.11, drag, thrust gain) are the cause, 0.319 → 0.100 m; the
integrator is second-order; every weight change is worse. H4 ✗ (τ is a pure time shift on
the corrected model; the vendored lag was model mismatch). H5 ✓ for the adaptation on the
adopted plan (6–8/20 → 18/20), ✗ for the estimators (every variant below its
estimator-free twin). H6 ✗ (larger balls cut closer; moving the via centre is what buys
clearance, +0.04 s). And the unplanned lever — pre-compensating the gate references
against the tracker's systematic cut — turned out to be the largest single one with the
vendored MPC: `b08` 0/5 → 15/15, `g3` at 0.90 × TWR 4.12 s at 40/40.

## 7. What the audit corrected, and what is weak

An adversarial audit re-derived all 28 cells from the per-episode CSV, log and flown-path
files — k, n, mean, std, min/max, failure counts, mass halves, solve times, the config md5s,
the tracker md5s, every counted gate pass inside the 0.20 m opening, `flight_time ==
(steps − 1)/50` in all 560 episodes — with no discrepancy in any count or lap (receipt:
`audit/NOTES.md`, `audit/logs/rederived_table.csv`, `rederive.log`). Eight things it changed
in the *narrative*, folded into §3–§6 above, so that you see what an audit is for:

- `RACE_START_BLEND=1.0` is *not* a no-op at Level 0 for the z-0.05 plans (|d| = 0.04 m);
- M1's −0.02 s is an early gate-4 crossing on the floor clock, not a plan gain;
- `b08` also raises gate 2 by 0.06 m, and its gate-2 reference sits at the planner's bound;
- the harness gain is × 3.2, not "halved";
- the estimator statement has an exception (the z-only ESO on M1 at 6/10);
- the 62 planner race screens carry no config md5 in their headers (bracketed by the
  md5-carrying headers before and after them, and the claim re-run in the protocol);
- the count of over-budget solve steps is unlogged (the bridge filters lsy's warning);
- Lesson 7's Level-1 baseline reads 8/20 and §5's re-measurement 6/20 — both genuine.

One finding was **withdrawn after verification**, and it is the one to learn from: the
audit read the drone's collision geometry from the model XML (a 0.086 m sphere active, the
box inactive) and concluded that every margin in this lesson is overstated. But
`race_core.py` calls crazyflow's `use_box_collision(sim, True)`, which enables the
0.07 × 0.07 × 0.02 m box and disables the sphere (`crazyflow/sim/sim.py`), so the box model
of §1, the planner and the runner stands. Check the *code path*, not only the file.

**What is weak, and what was not run:**

- The plan ranking (51 rows) rests on 5–15-episode screens; only the 28 protocol cells are
  20-episode results. `ground-b08_f0.90` with hold 0.2 — the archive's rank 1, the *same 4.12 s
  without the `g3` shift* — was raced only 5/5 and never at 20; the `g3` shift was chosen on
  hold-0.3 screens (4/5 vs 5/5). "M1 fails the 0.90 plans" rests on 10 episodes; every
  estimator verdict on 10.
- Every harness number (deviation, gate offsets, clearances, the whole §4 sweep) is one
  deterministic episode without contacts, noise or disturbance — a screen, never a verdict.
- Level-1 heavy halves are n = 5–13 per cell; the headline's heavy half is 12/18 (Wilson
  95 % CI 44–84 %), its pooled 34/40 is 71–93 %; no third replicate.
- Solve times are 14–19 ms on an idle machine, with maxima up to 216 ms (326 ms in a cited
  cell under load); the count of late steps is unlogged.
- The measured model is a fit to *this simulator*, and the pre-compensated plans are tuned
  to the *vendored* tracker's cut: any change of tracker, mass model or track invalidates
  the shifts (§4's M1 rows show it).
- The 0.95 × TWR plans were raced only unbiased (0/5 everywhere); Level 2 was never raced
  and cannot be with this stack (the bridge hard-codes the nominal obstacles and a fixed
  plan; Lesson 3 §1); and the leaderboard's own evaluation level and clock are not stated
  in the README it is quoted from — the comparison with 3.394 s is on the clock the README
  implies (ground start, t = 0), not on a documented protocol.
- The `FRAME_Gk?` verdicts (12 episodes in 9 cells) are a heuristic on the last flown
  sample; lsy does not log the contact partner.

## 8. What is left to the leaderboard

0.73 s. Not the takeoff any more (a 0.2 s hold plus the spin-up), not the controller's
compute, not the controller's gains — the **plan**: 0.90 × TWR with the hold gives 4.12 s,
and the raw time-optimal plan from hover was 2.59 s (Lesson 6 §1). The record entries
presumably fly something closer to it with a tracker that does not cut corners. In order
of expected return, all of it planner work:

1. Race `ground-b08_f0.90` (no `g3`) with hold 0.2 at 20 episodes, and a third Level-1
   replicate of the headline.
2. More climb margin on the first two legs for the heaviest Level-1 draws (a per-leg thrust
   fraction: 0.85 on the climb, 0.90 elsewhere), and the pole-4 via at +0.03 m (`p3`) for the
   0–4 cm of flown margin there.
3. A gate-3 pre-compensation built for the *corrected-model* tracker — its only failure
   mode — and only then the unbiased 0.85–0.90 plans with it: the tracker that flies the
   plan's clearances is the one that can take a tighter line.
4. A drag-aware planner (or the drag budget as a lower thrust fraction at speed) before
   any 0.95 × TWR plan is raced again.
5. Level 2 needs replanning within the 0.7 m sensor range; out of this bridge's reach by
   construction.

**What not to do (measured):** cost/horizon tuning of the MPC (40 cells), RK4 (nothing for
4–7× the compute), additive disturbance estimators for Level 1 (every variant below its
estimator-free twin), larger via balls, tighter windows or entry corridors on top of the
pre-compensation, the end state, and a longer hold.

---

## 🛠️ Before you move on

1. **The verdict table.** H1–H6 with your thresholds, your numbers, your verdicts, in the
   pre-registered wording of §0 — and the unplanned lever, named as such.
2. **The 4.12 s without the `g3` shift, at 20.** `--track ground-b08 --thrust-frac 0.90` is
   the headline track with gate 3 left where it is — the two files differ by that one shift,
   so planning both at 0.90 prices it (3.925 s against 3.924 s: the shift is free in *plan*
   time). Race `ground-b08_f0.90` with hold 0.2 at both levels, 20 episodes. The archive has
   it at 5/5 at Level 0 and nothing at Level 1, and it chose the `g3` shift on hold-0.3
   screens (§7). Does the shift buy laps, or did a 5-episode screen choose for you?
3. **A gate-3 pre-compensation for the corrected model.** M1's only failure mode on the
   0.85 ground-start plans is a +0.03…+0.05 m cut at gate 3 (§4). Shift gate 3 and its entry
   corridor against it in a copy of `togt/lsy_level2_ground.yaml`, plan at 0.85, and race
   `mpcdev:att=sim,drag=0.495,fgain=1.0` on it at 20. Prediction first: F1 is 18/20 at
   4.30 s; name the number that would make the plan worth keeping.
4. **A per-leg thrust fraction.** The planner has one thrust bound; the heavy Level-1 draws
   lose the *climb* (§6, reading 2). Find a way to give the first two legs 0.85 and the rest
   0.90 — two plans spliced at gate 2, or a per-piece bound in `togt/togt_race.cpp` — and race
   the result at Level 1. Report the heavy half separately.
5. **The third Level-1 replicate.** 34/40 has a 95 % interval of 71–93 %. Run `headline_L1`
   once more and pool; then say, with the interval, whether the headline ranks at Level 1
   the way Lesson 3 §5 means it.
6. **Identify the yaw axis.** §2 probed roll and pitch; the so_rpy yaw gain (1.44) was never
   checked because the race reference has no yaw. Add a `yawstep` probe to `race_probe.py`
   (one subclass per file — extend it, do not copy it), fit it with `probe_fit.py`, and say
   whether the `yaw_max = 0.3` bound of the MPC ever binds on the headline plan.
7. **Count the late steps.** The bridge filters lsy's per-step warning, so nobody knows how
   many of the 14–19 ms solves exceed the 20 ms budget. `MPCDev.solve_ms` holds every call:
   count the steps above 20 ms per episode in `race_runner.py`'s `EP` line, run the headline
   cell, and write one sentence on what those steps would cost a real Crazyflie.
8. **What you would do next, and why** — one paragraph. The evidence points at the plan
   (§8). If you disagree, say which corner of Lesson 7 §5's triangle your idea moves and
   name the number that would tell you it did.

**Back to:** [Lesson 4 — Brainstorm: make it faster](04-brainstorm-faster-tracking.md)
— with the method in the right order: diagnose, identify, replan, adapt.
