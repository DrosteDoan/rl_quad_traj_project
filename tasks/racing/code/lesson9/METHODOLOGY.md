# Lesson 9 methodology: learned vs model-based racing under graded disturbance

Status: design agreed 2026-09-21, build in progress (see the phase list at the end). This file is the
reference for `tasks/racing/lessons/09-*.md`; when the two disagree, fix whichever is wrong and say why.

## 1. Question and claim

**Question.** Under what perturbation conditions does the relative performance of learned and
model-based quadrotor racing controllers cross over, and how consistently do those crossover regimes
persist across track geometries?

Model-based controllers are often effective where the dynamics are accurately modelled. Learned
controllers may be less sensitive to model mismatch and to disturbances the model does not represent.
As a disturbance grows, the relative performance of the two may change, giving a crossover regime.
Repeating the experiment over many track geometries tests whether the crossover regimes persist across
task configurations or depend strongly on geometry.

**Notice.** The crossover regimes identified in this study are not intended to represent universal
thresholds for learned or model-based control.

## 2. Design at a glance

- **Tracks:** 15 random gate layouts, each with a time-optimal plan checked for flyability without
  reference to any controller. Three more (the dev tracks) are used only for calibration.
- **Disturbances:** five conditions, each applied at a coefficient lambda in [0, 1], the fraction of
  that disturbance's own range.
- **Members:** 4 model-based controllers and 3 robust RL policies, each plotted as its own line.
- **Trials:** 25 seeds for random disturbances, a repeat check for deterministic ones.
- **Outcome:** completion (4 gates in order, no contact), plus lap time, RMSE and other per-lap measures.
- **Deliverable:** per-track charts of completion against lambda, stored with one row per lap so any
  statistic can be computed later.

## 3. Tracks

- **Generation.** lsy's level-3 generator (`build_random_track_fn`, `lsy_drone_racing/envs/randomize.py`).
  Fixed: the level-2 start (-1.5, 0.75), gate heights 0.7 / 1.2 / 0.7 / 1.2 m, gate order 1-2-3-4.
  Varying: gate x, y and yaw. The generator's poles are ignored (TOGT does not model them). Layouts are
  generated up front and saved as JSON.
- **Planner.** TOGT tube plans that start on the floor at z = 0.05 with a 0.2 s hold (Lesson 8
  ground-start style), `--thrust-frac 0.80` (revised from 0.85, 2026-09-22 -- see limitation 12: the 0.85
  plans left too little thrust headroom for disturbance, and the collapse of the MPC family under
  calibration was partly an artifact of the plan, not the controllers). The track file is the tube corridors
  of upstream's
  `lsy_level2_tube.yaml` generalised to any gate, started on the floor. It has NO pole vias (upstream's
  `--track ground` carries the pole-avoidance balls B1, B3, B4, which make no sense without poles). The
  hand-tuned gate pre-shifts of Lesson 8 (`b08`, `b08g3`) are not used: they are tuned to one tracker on
  one track.
- **Flyability, judged from the plan and never from controller outcomes:**
  - structural filter: TOGT solves; the plan loads (every gate crossed inside its opening); no
    gate-plane crossing through a frame zone (a proxy); crossing angle (<= 25 deg) and arena limits
    (0.10 m inside lsy's +-2.5 x +-1.5 m) met;
  - exact contact test on the reference (`contact.py`, phase 2): the level drone box flying the plan
    must not touch a gate frame, and its **frame margin must be >= 3.0 cm** (user decision, 2026-09-22).
    Frame margin = how far the drone box may grow, per half-extent, before the path touches a frame;
    roughly how far a tracker may stray from the plan before lsy would end the race. The anchor is
    Lesson 8's level-2 ground plan (planned at 0.85, not this study's 0.80), which the corrected MPC flies
    at 18/20 at Level 0: 3.5 cm. Of the 15 tracks first accepted by the proxy alone (at 0.85), 3 touched a
    frame and 4 more had 0.9-1.6 cm;
  - yield is about 1.1 % of random layouts (15 of 1,319 at thrust-frac 0.80; 15 of 1,251 at 0.85 -- almost
    unchanged, because 90 % of rejects are arena excursions and frame crossings of the time-optimal PATH,
    which a thrust-headroom change barely touches); TOGT never fails; candidates are generated until enough
    pass; dev tracks use seeds >= 100000, disjoint from the study range;
  - speed stretch s = max(1, v_peak / 4.4 m/s) (thrust is already about 0.80 of the limit by
    construction);
  - take the first 15 seeds that pass, in seed order, and log every reject with its reason.
- **Baseline viability.** At lambda = 0 every member runs once on every track, as a report. A track is
  dropped only if no member is viable on it. A member is never dropped from a track for failing; the
  failure is reported, and non-viable (track, member) cells are excluded from any later crossover
  statistic.

## 4. Disturbances and lambda

lambda = 0 is exactly nominal. lambda = 1 is the ceiling.

- **Ceiling rule (user).** The maximum is the level at which at most one controller still persists
  (>= 50 % completion), calibrated on the dev tracks and then frozen. Starting values were 2x Lesson 7's, so
  that lambda = 0.5 reproduces Lesson 7's conditions as a regression check.
- **Frozen ceilings (recalibrated 2026-09-22 on the thrust-frac 0.80 dev tracks; `maxima.json`).** Calibrated on
  the 3 dev tracks (seeds 100023, 100082, 100092; 1,512 laps; `calibrate.py`) with the four MPC-family members,
  because the robust RL seeds train on these ceilings and did not exist yet. The ceiling is, in effect, the level
  at which the model-based family has collapsed to at most one survivor; the RL family will probably outlast it,
  and lambda = 1 is not "where RL fails". In units of Lesson 7's value: `wind_const` 1.5 (0.165 N, 3.8 m/s^2),
  `payload` 2.0 (20 g, 4.5 m/s^2), `wind_gust` 1.5 (mean 0.12 N, amplitude 0.12 N, turbulence sigma 0.06 N),
  `lighthouse` 1.0 (Lesson 7's model). **Identical to the first (thrust-frac 0.85) calibration** despite the
  extra thrust headroom -- evidence that the 3.0 cm frame margin, not thrust, is the binding limit at these
  levels; see limitation 12. Samples are small (3 laps per level for the deterministic conditions). At lambda = 0
  (nominal): M1 2/3, M1+ESO 2/3, M1+L1 3/3, mppi_l1 2/3 (M1 now fails one dev track nominally; a geometry effect
  of the new track 100092, not a regression).
- **Delay-only check (re-run on the 0.80 dev tracks).** The fixed 1-step latency alone, all error sizes zero:
  M1 2/3 (lambda = 0: 2/3, no loss), M1+ESO 3/3 (2/3, gains one -- noise), M1+L1 2/3 (3/3, loses one),
  **mppi_l1 0/3 (2/3, loses both)**. mppi_l1's tight margins (Phase 4's diagnosis: effective sample size ~1)
  make it sensitive to the delay alone; the user kept the latency as decided, so this is recorded, not acted on.

- **Conditions.**

| condition | what lambda scales |
|---|---|
| `wind_const` | steady force along +x |
| `payload` | downward force in z |
| `wind_gust` | mean push, sinusoid amplitude and turbulence sigma together; frequency (0.7 Hz) and turbulence time constant (0.5 s) fixed |
| `lighthouse` | size errors and update interval together; lambda = 0 refreshes every control step (a perfect sensor); the 1-step (10 ms) latency is fixed whenever Lighthouse is on, so there is a jump at 0+ |
| combined | gust + payload + Lighthouse at the same lambda (not `wind_const`, which stacks with the gust mean) |

- **Seeding.** Random parts are unit-normal streams scaled by lambda: one seed is the same draw at every
  lambda and for every member (common random numbers).
- **Scaling rule.** Every disturbance's ceiling is `scale x` Lesson 7's value (starting scale 2, so
  Lesson 7's condition is lambda = 0.5; phase 5 calibrates the scales and freezes them in `maxima.json`).
  Sizes are `lambda * scale * nominal`. Code: `knobs.py`, tested in `test_knobs.py`: at Lesson 7's value the
  constant forces and the gust are identical to `crazy_track.disturbances`, and the Lighthouse sensor has the
  vendored distributions (per-channel noise streams, so that one seed is the same draw at every lambda).
- **Lighthouse refresh rate is a staircase at the start.** The refresh interval is
  `dt + lam_eff * (T - dt)` with T the drawn nominal interval, but the loop is discrete (100 Hz): any interval
  above one step makes the period two steps. Measured mean refresh rate (10 seeds, scale 2):
  100 Hz at lambda = 0, **48 Hz at lambda = 0.05**, 36.5 Hz at 0.2, 21.8 Hz at 0.5 (Lesson 7's model; its mean
  interval is ~45 ms, not the 29 ms that "34 Hz" suggests), 13 Hz at 1.0. So Lighthouse has two discrete
  steps at lambda = 0+ (latency and the rate); the rest is smooth. Physical values at the grid points are in
  `knobs.describe`.
- **Latency check.** A delay-only run per member on the dev tracks confirms the fixed latency does not
  break anything by itself; if it does, go back to the user.

## 5. Members

- **Model-based (4), the Lesson 8 corrected-model family:** M1 = `mpcdev:att=sim,drag=0.495,fgain=1.0`;
  M1+ESO (`dist=eso`); M1+L1 (`dist=l1`); `mppi_l1` unchanged.
- **Learned (3), "robust RL":** the racing-envelope recipe, training seeds 0, 1, 2; perturbation box =
  0.8 x the frozen maxima; sensor noise level per episode from lambda in U(0, 0.8). All three are used,
  no re-rolling.
- **Not in the study:** `v5_s0`, `racing_s0`, `racing_s1`, `racing_s2`. No family mean: every member is
  its own line.

## 6. Trials and grid

- Stochastic conditions (gust, Lighthouse, combined): 25 seeds per (track, lambda).
- Deterministic conditions (`wind_const`, `payload`): first repeat one cell to check determinism;
  members whose repeats differ (expected: `mppi_l1`) get 3 seeds, the rest 1 run.
- Grid: lambda 0 to 1 in steps of 0.1; add 0.05 midpoints wherever a member's pooled completion changes
  by >= 0.2 across an interval.

## 7. Measurement

- **Clock:** the ground clock (Lesson 7 section 3).
- **How a lap runs (`driver.py`):** the vendored `rollout` loop plus three additions and nothing else: the swept
  contact test on every step (the lap ends at the first contact), an end at the last-gate crossing (lsy ends
  the race there) or when a gate is missed (its clock + 1.0 s passes without a crossing inside the opening),
  and a reused Sim with the external force zeroed and the ipopt initial guess zeroed at each lap start (so a
  lap does not depend on the laps before it). Verified: the flown path is bit-identical to the vendored
  harness's (0.00e+00 m) nominal and at Lesson 7's wind and gust, for M1 and for a policy.
- **Completion:** all 4 gates passed in order with no contact at any time. The contact check is lsy's
  drone box (0.07 x 0.07 x 0.02 m half-extents) against the gate frame boxes (0.20-0.36 m from the
  centre), swept along each step and yaw-aware; validated by reproducing Lesson 8's thresholds (0.101 m,
  0.13 m, 0.18 m). A run ends at the first contact; `t_impact` and the object are recorded.
- **RMSE:** over the racing segment only, from the end of the 0.2 s hold to the last-gate crossing.
  Secondary; censored on failed laps.
- **Stored, not plotted:** lap time (completers), max deviation, gates passed, time to gate 1, solve
  time, ipopt failures.

## 8. Outputs and what is deferred

- **Now:** one folder per track, five charts (one per condition): completion against lambda, one line
  per member, warm colours for MPC and cool for RL, a marker at lambda = 0.8.
- **Deferred:** the crossover statistic. Pairwise was rejected. Candidate: "leading family" (best member
  per family per track and lambda, margin 0.2, shown as the share of tracks led by each family against
  lambda). **Freeze the statistic before the study sweep** (Lesson 4 section 1 pre-registration), not
  after seeing study data.

## 9. Compute (measured 2026-09-22 with `driver.py`, one study track, M1 lap of 4.1 s, ground clock)

One M1 lap takes 8-12 s (mean solve 19-24 ms), M1+ESO / M1+L1 the same, `mppi_l1` 2-5 s, a policy 2-3 s. That is
about 35 s per (track, lambda, seed) cell over the 4 MPC-family members and 3 policies, less at high lambda
where laps stop at the first contact. Coarse grid: about 150-250 core-hours, about a day on 8-9 workers.
Two settings decide whether that holds, both measured:
  * `JAX_PLATFORMS=cpu` and single-thread limits (`OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=1`,
    `XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"`). With them, 4 concurrent
    workers run at 11-13 s per lap, the same as one alone. Without them the same 4 workers took 154-192 s per
    lap (15x slower), and the processes hung at exit in the WSL GPU driver (JAX probes CUDA);
  * memory: a worker peaks at about 1.2 GB, so about 8-9 workers in 15 GB (not 16).
Training the 3 robust seeds: 25-50 minutes each. A ground plan plans in 0.4 s; a candidate track costs about
1.3 s (planning, verdict, exact contact test).

## 10. Limitations to state in Lesson 9

1. Gate-only geometry with a fixed start, heights and gate order.
2. The corrected MPC model is a fit to this same simulator; its "accurate model" favours MPC.
3. The RL family is trained on this study's range (favours RL); with no contrast group, a crossover
   cannot be attributed to the training edge versus something intrinsic.
4. No parametric mass mismatch (Lesson 8 section 5 shows it is what defeats MPC in the race); out of
   scope because the training environment does not randomise mass.
5. Gust time structure is not in training; training runs at 50 Hz, the harness at 100 Hz.
6. Ceilings are outcome-calibrated on dev tracks.
7. Disturbances enter through crazy_track's force model; contacts are checked after the fact from the
   flown path, not by the physics engine.
8. The Lighthouse model was validated up to about 3 m/s and is applied at about 4.5 m/s.
9. The MPC solves in about 32 ms per 10 ms step: not real-time.
10. Three RL seeds and 15 tracks.
11. **Track population.** The 15 study tracks (thrust-frac 0.80) are the layouts whose time-optimal tube
    plan is contact-free with a reference frame margin >= 3.0 cm, stays in the arena and crosses gates
    within 25 deg: 15 of 1,319 candidates (1.1 %). This is selection on the plan, never on a controller
    outcome, so the comparison between members is paired and unaffected; what is limited is how far
    "consistent across geometries" reaches. The restriction is strong and essentially unchanged from the
    0.85 track set: against all 1,319 candidates the accepted set sits at the 6th-19th percentile in mean
    exit turn (71 vs 115 deg, 6th), mean turn at gates (78 vs 116 deg, 8th), path length (7.1 vs 8.5 m,
    19th), mean gate gap (1.6 vs 2.1 m, 15th), peak speed (4.6 vs 5.3 m/s, 18th) and planned lap (4.1 vs
    4.6 s, 11th): **compact layouts with gentle turns**. About three quarters of that shift comes from
    requiring a contact-free reference (not optional: a reference that touches a frame cannot be flown),
    the rest from the 3.0 cm margin. The likely direction is to UNDERSTATE geometry dependence, because
    hairpin-heavy and spread-out layouts are the ones removed. Stratified selection was considered and
    declined; it cannot restore a missing tail. After the sweep, plot the per-track crossover lambda
    against exit turn and plan length: flat is mild evidence of robustness, a trend is a warning about the
    untested layouts. State the scope of the claim accordingly: consistent across compact, gentle-turn
    layouts.
    The dev tracks (calibration only) use seeds >= 100000, disjoint from the study range.

12. **Plan aggressiveness sets where the model-based family collapses.** The plans are time-optimal at 0.85 of the
    thrust limit, so a constant force of about 4 m/s^2 (the dev plans' mean headroom is 3.8 m/s^2, from peak
    demands of 13.0-15.6 against 18.4) cannot be flown at the plan's fastest moments by any tracker, and the
    tube leaves only 3.5-5 cm of frame margin, so 72 % of failed calibration laps are frame contacts (26 %
    missed gates, 2 % divergence) and the median RMSE of completed laps (0.086 m) is only 3.5 cm below that of
    contact laps (0.121 m). Measured on the dev tracks by flying the same paths slower (more headroom, the
    frame margin unchanged): at 1.3x the payload collapse moved out from 3.4-4.5 to beyond 4.5 m/s^2 (all three
    MPC variants 100 % at 4.5) and ESO/L1's wind collapse from 3.8-5.1 to 5.1; plain M1 under constant wind did
    NOT improve (RMSE 0.22 -> 0.24), because a nominal MPC has no disturbance state and holds a steady offset.
    The ceilings, and any crossover, are therefore properties of this plan family at this aggressiveness, not
    of the controllers alone. A lower thrust fraction would raise them.

## 11. How the initial input became this framework

| proposed | now |
|---|---|
| disturbance on/off | lambda as a fraction of range; maxima frozen by the ceiling rule |
| 50 trials per condition, few tracks | 25 seeds x 15 tracks |
| "all controllers complete at least one lap" | structural filter, a stretch to a common speed cap, and a lambda = 0 viability report (no selection on outcomes) |
| random gates, then TOGT with tube | lsy generator saved as a track library, then ground-start tube plans |
| poles kept (sphere-clearance idea) | poles removed; completion is box-contact-free by lsy's own box model |
| RMSE, completion, lap time | same, RMSE over the racing segment only |
| Lesson 7's controllers | Lesson 8 corrected MPC family vs 3 robust RL seeds |
| family means / pairwise crossover | every member its own line; statistic deferred |

Corrections along the way: the thrust-headroom "ceiling" claim was retracted (Lesson 7's policies
finished with roughly a third of the headroom the disturbance needed); the optimistic lap-time guess was
replaced by the measured 24 s.

## 12. Build phases (each ends with a confirmation from the user)

1. Track library (`gen_tracks.py`, `track_lib.py`)
2. Contact check
3. Lambda knobs
4. Driver and storage
5. Dev calibration (frozen maxima are approved before training)
6. Robust RL training
7. Study sweep
8. Charts
9. `lessons/09-*.md`
