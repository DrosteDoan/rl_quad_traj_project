# Lesson 9 methodology: learned vs model-based racing under graded disturbance

Status (audited and updated 2026-09-25): design agreed 2026-09-21. Tracks, the contact model, the disturbance knobs, the
driver and calibration are built and frozen. The RL recipe was rebuilt after the first open-space training rounds; the
gate-aware recipe (section 5c) passed its pre-registered viability bar on validation tracks (robust group, seed 0,
saved), the gate-aware contrast group is built but not trained, and the study sweep has not started. This file is the
reference for `tasks/racing/lessons/09-*.md`; when the two disagree, fix whichever is wrong and say why.

## 0. Where the study stands (2026-09-25; read this first)

**The question** (section 1): under what perturbation conditions does the relative performance of learned and model-based racing
controllers cross over, and how consistently across track geometries? Everything else in this file serves that question.

**What the study is:** the MPC family (5 members) and the RL policy, flown on the study tracks under six conditions at graded
strengths lambda in [0, 1]; for each condition and track, find where the ordering flips. Nothing from that sweep has been run.

**Done.** Tracks, the contact model, the lambda knobs, the driver, and the frozen ceilings (sections 3-4, 7). The MPC family needs
nothing more: M1 completes 22/22 unseen validation tracks at Lesson 8's parameters. The first RL recipe could not thread gates
(0-2 of 6 tracks); the cause was that training had no gates or contact, and the rebuilt gate-aware recipe passes its bar (13/22
against M1's 22/22 at lambda = 0) and is saved (section 5c). A development preview on validation tracks (not a result) shows the
ordering depends on the condition: the strongest MPC member (M1+L1) beats RL under steady forces, RL is ahead where sensing noise or
the combined condition dominates, and RL tracks looser but does not degrade with lambda.

**Not done.** (1) The sweep itself, on the 10 study tracks no RL decision touched (limitation 15). (2) More than one RL seed (the
design says three). (3) The confound that RL was trained on ranges matched to the study's own ceilings, so an RL advantage may be
"trained on the exam" rather than learned control. The contrast group was meant to separate that; the fully vendored version did not learn (0 of 22 validation tracks after 16M steps), so the confound is UNANSWERED, not answered "no".

**Side threads (parked; not needed for the sweep):** the force-only ablation, held-out-condition and held-out-geometry tests
(section 5c), and any further contrast variants. They exist only to help interpret the RL line.

**Where things are:** design sections 1-4; members 5; RL history 5a-5c (long, chronological); protocol 6-9; limitations 10;
errata 13.

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

- **Tracks:** 15 random gate layouts (the study tracks), each with a time-optimal plan checked for flyability
  without reference to any controller. Three dev tracks calibrate the MPC ceilings and are also part of the RL
  training pool. Two further pools come from fixed seed windows and are never study tracks: `train` (111 tracks,
  RL training) and `val` (22 tracks, where RL design decisions are judged). RL-vs-MPC headline numbers use only
  the 10 study tracks no RL decision touched (limitation 15).
- **Disturbances:** six conditions (`wind_const`, `payload`, `wind_gust`, `lighthouse`, `mass_mult`, and `combined`),
  each applied at a coefficient lambda in [0, 1], the fraction of that disturbance's own range.
- **Members:** 5 model-based controllers and 3 robust RL policies, each plotted as its own line, plus a
  3-seed contrast group (section 5b) shown thin/muted on the same charts, isolating whether matched
  disturbance-training buys anything beyond generic robustness training.
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
- **RL pools (2026-09-25).** `gen_pool.py` generated two further pools from FIXED seed windows chosen in advance,
  keeping every accepted track (no truncation to a target count): `train`, seeds 200000-205999, 111 tracks of 6,000
  candidates (1.85 %), and `val`, seeds 300000-301999, 22 of 2,000 (1.10 %). Same generator and filters as the
  study tracks; reject-reason mix and margins match (every margin >= 3.0 cm). Disjoint from the study range
  (0-1,318) and the dev range (100000+), enforced by a test. The 1.85 % against the study's 1.14 % is unexplained
  beyond window-to-window variation.
- **Baseline viability.** At lambda = 0 every member runs once on every track, as a report. A track is
  dropped only if no member is viable on it. A member is never dropped from a track for failing; the
  failure is reported, and non-viable (track, member) cells are excluded from any later crossover
  statistic.

## 4. Disturbances and lambda

lambda = 0 is exactly nominal. lambda = 1 is the ceiling.

- **Ceiling rule (user).** The maximum is the level at which at most one controller still persists
  (>= 50 % completion), calibrated on the dev tracks and then frozen. Starting values were 2x Lesson 7's, so
  that lambda = 0.5 reproduces Lesson 7's conditions as a regression check.
- **Frozen ceilings (`maxima.json`), current as of the third calibration pass (2026-09-22).** In units of the
  condition's reference value (Lesson 7 for the first four, Lesson 8's Level-1 extreme for `mass_mult`):

  | condition | ceiling | physical value at lambda = 1 |
  |---|---|---|
  | `wind_const` | 1.5x | 0.165 N, 3.8 m/s^2 |
  | `payload` | 2.0x | 20 g, 4.5 m/s^2 |
  | `wind_gust` | 1.5x | mean 0.12 N, amplitude 0.12 N, turbulence sigma 0.06 N |
  | `lighthouse` | **1.5x** (moved from 1.0x -- see below) | Lesson 7's model x 1.5 |
  | `mass_mult` | 2.0x | +23 % mass (Lesson 8's own Level-1 extreme, +11.5 %, sits at lambda = 0.5) |

  Calibrated on the 3 dev tracks (seeds 100023, 100082, 100092; `calibrate.py`), with the MPC-family roster at
  the time of each pass, because the robust RL seeds train on these ceilings and did not exist yet -- the
  ceiling is, in effect, the level at which the MODEL-BASED family has collapsed to at most one survivor;
  the RL family will probably outlast it, and lambda = 1 is not "where RL fails".

  **Calibration history, kept for the record:**
  1. *First pass* (thrust-frac 0.85 dev tracks; 4 members: M1, M1+ESO, M1+L1, mppi_l1 at its vendored 0.5 s
     horizon): `wind_const` 1.5, `payload` 2.0, `wind_gust` 1.5, `lighthouse` 1.0.
  2. *Second pass* (recalibrated after the thrust-frac restart to 0.80; same 4 members): identical numbers --
     evidence the 3.0 cm frame margin, not thrust headroom, is the binding limit (limitation 12). `mass_mult`
     added as a 5th condition with a 5th member, M1+mass (`mass=1`): ceiling 2.0x.
  3. *Third pass* (2026-09-22, this one): mppi_l1 reconfigured to an 0.8 s preview horizon (`horizon=40,
     dt_plan=0.02`, was `horizon=25, dt_plan=0.02` = 0.5 s -- user decision, section 5 note) to match the M1
     family, which forced every condition's mppi_l1 rows to be re-flown; this pass ALSO backfilled M1+mass
     for the four originally-frozen conditions, since it had never been flown there (only `mass_mult` and the
     delay-only check included it before). **Four ceilings held; `lighthouse` moved from 1.0x to 1.5x.**
     Traced to source: at lambda_eff = 1.0, M1+mass newly completes 60 % of lighthouse laps (second only to
     M1's 67 %), pushing the "how many members persist" count from 1 to 2 -- mppi_l1's own lighthouse numbers
     got WORSE under the longer horizon (33 % -> 20 % at lambda_eff = 1.0), which if anything argued for
     LOWERING the ceiling, not raising it. So the shift is attributable to M1+mass entering the roster, not
     to the mppi_l1 reconfiguration, even though both happened in the same recalibration pass. `wind_const`,
     `payload`, `wind_gust` and `mass_mult` were unaffected: no other condition had two members newly cross
     50 % at the same level.
- **Delay-only check (current roster).** The fixed 1-step latency alone, all error sizes zero, vs. lambda = 0:
  M1 2/3 (2/3, no loss), M1+ESO 3/3 (2/3, gains one -- noise), M1+L1 2/3 (3/3, loses one), M1+mass 2/3 (2/3, no
  loss), mppi_l1 2/3 (2/3, no loss under the new 0.8 s horizon -- the vendored 0.5 s horizon lost both). mppi_l1's
  general fragility (Phase 4's diagnosis: effective sample size ~1) is unrelated to this specific channel; the
  user kept the latency as decided, so this stays recorded, not acted on.
- **`mass_mult`-specific finding, still true under the current roster:** M1+L1, not M1+mass, is the most robust
  member across most of the range (100 % completion through lambda_eff = 1.5), while M1+mass tracks plain M1
  closely until the highest levels, where it pulls ahead (67 % at lambda_eff = 2.0 vs <= 33 % for everyone
  else). A plausible reading: a heavier drone's extra sag looks, to first order, like the additive downward
  force L1 already estimates well (it ties the deployment cell for `payload` too), while `mass=1`'s specific
  benefit only shows up once the mismatch is severe.
- **Conditions.**

| condition | what lambda scales |
|---|---|
| `wind_const` | steady force along +x |
| `payload` | downward force in z |
| `wind_gust` | mean push, sinusoid amplitude and turbulence sigma together; frequency (0.7 Hz) and turbulence time constant (0.5 s) fixed |
| `lighthouse` | size errors and update interval together; lambda = 0 refreshes every control step (a perfect sensor); the 1-step (10 ms) latency is fixed whenever Lighthouse is on, so there is a jump at 0+ |
| `mass_mult` | the drone's TRUE flying mass, heavier only (multiplicative model error, Lesson 8 section 5's finding: this, not an additive force, is what defeats MPC in the race); calibrated separately (see below); NOT part of `combined` (user decision, 2026-09-22: a multiplicative model error is a different kind of thing from additive/sensing effects stacking) |
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

- **Model-based (5), the Lesson 8 corrected-model family:** M1 = `mpcdev:att=sim,drag=0.495,fgain=1.0`;
  M1+ESO (`dist=eso`); M1+L1 (`dist=l1`); M1+mass (`mass=1`, added 2026-09-22 alongside the `mass_mult`
  condition); `mppi_l1` **reconfigured 2026-09-22** to an 0.8 s preview horizon (`horizon=40, dt_plan=0.02`
  in `driver.py`'s `make_race_controller`, bypassing the vendored spec table since it has no comma-syntax
  override; the vendored default is `horizon=25, dt_plan=0.02` = 0.5 s). `dt_plan`, the class's own tuned
  rollout-integration step (the docstring's 2026-07-22 sweep), is left untouched -- only how far it looks
  changes, matching M1's 20 x 0.04 s = 0.8 s exactly in total horizon (mppi keeps its finer 0.02 s sampling,
  40 points instead of M1's 20).
- **Learned (3), "robust RL":** the racing-envelope recipe, training seeds 0, 1, 2, with a per-episode
  domain-randomization box matched to the frozen ceilings (section 5a). All three are used, no re-rolling.
  **State (2026-09-25):** trained in three rounds on the open-space recipe (section 5a; 0-2 of 6 screen tracks completed), then
  rebuilt as the gate-aware recipe (section 5c). Gate-aware robust seed 0 is trained and saved (`saved/gate_aware_v4/`);
  seeds 1 and 2 are not trained.
- **Contrast group (3), "contrast RL"** (added 2026-09-22, user request): the SAME racing envelope, PPO
  settings, preview window and frequency as the robust group, but the VENDORED, unmatched disturbance
  training (section 5b) -- isolates whether matching the domain-randomization ranges to the study's frozen
  ceilings buys anything beyond generic robustness training. Training seeds 0, 1, 2 (matching the robust
  group's count on purpose -- section 5b). Shown thin/muted on the study's charts, not folded into either
  family's story: its only job is to say whether the robust family's crossover location moves relative to
  it, or stays where it is. **State (2026-09-25):** open-space contrast seeds 0-2 were trained in the three rounds (section 5a)
  and are superseded; the gate-aware contrast environment is built and tested (section 5c), not trained.
- **Not in the study as first-class members:** `v5_s0`, `racing_s0`, `racing_s1`, `racing_s2` (the course's
  own Lesson 2/7 checkpoints). `v5_s0` was considered as a free contrast and rejected: it is trained on a
  narrower reference envelope (|v| up to 3.5 m/s) than this study's tracks need (stretched to a 4.4 m/s
  cap; Lesson 7 already measured it failing the unstretched fast tube plan, 3/4), so it would conflate
  "wrong reference envelope" with "unmatched disturbance training" -- not a clean isolation of the one
  variable the contrast group exists to test. No family mean for any of the three RL-family lines
  (robust, contrast): every member is its own line.

**Do the MPC family's Lesson 8 parameters transfer to these tracks? (2026-09-24, measured, lambda = 0, nominal
only).** The user asked whether the MPC parameters must be adjusted again, since Lesson 8 built them for the
same task but on a different track and plan. Two kinds of parameter: the physical ones (`att=sim` attitude
gains, `drag=0.495`, `fgain=1.0`) are system identification of the drone in this simulator, so they are
track-independent by construction, and `test_driver.py` confirms M1 reproduces Lesson 8's level-2 harness
result through `driver.py`. The tunable ones (cost weights, horizon H=20 at dt 0.04 = 0.8 s) were swept on one
plan in Lesson 8 section 4, which found the cost at a local optimum -- the x3.2 came from the model, not the
weights -- so transfer to other tracks is empirical. Evidence with NO re-tuning: M1 completes all 5 study
screen tracks (4, 93, 387, 25, 504; margins 3.8-4.6 cm; rmse 0.044-0.063, max deviation 7-10 cm) and 2 of 3
dev tracks; the family on study 25 and 504 completes 10/10 (mppi_l1 with roughly twice M1's rmse, 0.094-0.113).
The one failure is dev 100092, the tightest track (3.05 cm margin): M1, M1+ESO, M1+mass and mppi_l1 all fail it
at lambda=0 (calibration CSVs, 6/6 conditions each), only M1+L1 completes it -- a margin-limited knife edge
(M1's max deviation ~9 cm exceeds a 3 cm margin), not a parameter mismatch. Exception: `mppi_l1`'s parameters
are not Lesson 8's -- vendored, tuned on a Lissajous benchmark, with only its horizon changed here (0.5 -> 0.8 s
for equal preview) -- so it has the weakest provenance in the family. The 10 untouched study tracks (747, 757,
834, 837, 965, 989, 1089, 1145, 1250, 1318) have not been flown by any MPC member; the lambda=0 viability pilot
does that, and its role is to drop non-viable tracks, not to tune. Recommendation given to the user, not yet
agreed: keep the Lesson 8 M1 family frozen (developed on level 2, disjoint from every Lesson 9 track -- a
cleaner separation than the RL side has), and re-tune only on a rule declared in advance, on dev tracks, with
the same budget for every member.
Update 2026-09-25: M1 at Lesson 8's parameters completes 22/22 of the `val` tracks at lambda = 0 with no re-tuning (88/88
gates, mean RMSE 0.0525, median max deviation 0.085 m) -- a far larger unseen sample than the 8 tracks above. The
freeze recommendation is still not agreed by the user.

## 5a. The robust RL training recipe (`robust_env.py`, `lighthouse_batch.py`, `robust_policy.py`,
`train_robust.py` -- code written and mechanics-tested 2026-09-22, `test_robust_env.py`, 9 tests; trained in three
open-space rounds 2026-09-22/23 (below); superseded as the study recipe by section 5c)

Same policy, same environment, same PPO settings as `train_racing.py` (Lesson 7 section 2) -- one change,
the per-episode domain-randomization ranges, each set to 0.8 x its condition's frozen ceiling so the training
box sits inside the study's own range with room to spare at the top.

**Three channels, matching the study's five conditions minus `combined` (which is not separately trained --
see below):**

| channel | covers | training range | mechanism |
|---|---|---|---|
| force | `wind_const`, `payload`, `wind_gust` | x, y: +-4.4 m/s^2 (0.8 x 1.5 x Lesson 7's 2.54 m/s^2 `wind_const` value); z: -3.6 to +1.8 m/s^2 (0.8 x 2.0 x `payload`'s downward value, small upward margin) | `_sample_perturb`, unchanged: one constant force vector drawn per episode, `self.rng.uniform` |
| Lighthouse | `lighthouse` | per-episode lambda in U(0, 0.8), scaling size errors and the update interval together, the same mapping as `knobs.ScaledLighthouse` | a new `LighthouseSensorBatch` variant (the vendored one draws a noise SCALE in U(0, 1.5), not a lambda with a coupled interval; needs its own subclass) |
| mass | `mass_mult` | per-episode fraction in U(0, 0.8 x 2.0 x 0.1153) = U(0, 0.184), i.e. up to +18.4 % heavier, one-sided | new: `sim.data.params.mass` overridden after `sim.reset(mask)`, mirroring `driver.py`'s pattern (verified: `Sim.reset()` restores `default_data`, so the override must be re-applied every episode, exactly like the external force already is) |

None of the three channels represents the gust's time-varying structure (0.7 Hz swing, OU turbulence) --
the force channel is a single constant per episode, as in the vendored recipe; this is an existing
limitation of the base recipe, not new here (see the limitations list).

**Training-progress logging (`monitored_adapter.py`, added 2026-09-22 after round 1's seed-0 pair finished
with no `rollout/ep_rew_mean` in either log).** The vendored `SB3Adapter` (`crazy_track/training/ppo_train.py`,
used unmodified by `train_racing.py` too) never sets `infos[i]["episode"]`, so SB3's own
`_update_info_buffer` (`stable_baselines3/common/base_class.py`) never has anything to add to
`ep_info_buffer`, and `OnPolicyAlgorithm.train()` skips the `rollout/ep_rew_mean` / `ep_len_mean` log lines
entirely -- structurally, every run, not intermittently; a pre-existing gap in the vendored recipe, unnoticed
until now because Lesson 7's policies were validated by flying them afterward, not by watching training
curves live. `MonitoredSB3Adapter(SB3Adapter)` tracks per-env cumulative reward and length and populates the
missing key on `done`; confirmed by inspecting `on_policy_algorithm.py` and `ppo.py` that nothing in the
actual PPO update (GAE, the policy/value loss, clipping) reads `ep_info_buffer` -- it is purely a logging
sink, so wrapping the adapter changes what gets logged, not how the policy trains. `train_robust.py` and
`train_contrast.py` both use it from round 2 on (2 tests, `test_monitored_adapter.py`); round 1's seed-0 pair
was trained before this existed and has no `ep_rew_mean` history, only the optimizer-internal diagnostics
below.

**Round 1 (seed 0 of each group), read without `ep_rew_mean`.** Both trained for about 65 minutes (`robust_s0`: 65.3 min, and the run-directory name, the file times and the tensorboard
events all agree, because all three use the container's clock, UTC). An earlier note here blamed a 7-hour `RunLogger`
clock artifact; that was a misdiagnosis: the host runs at UTC+7, so host-side `ls` timestamps differ from the
container's by exactly that offset, and comparing across the two produced the apparent discrepancy (corrected 2026-09-25). Optimizer diagnostics look healthy: `train/value_loss` 27.6 -> 2.06,
`train/explained_variance` -0.07 -> 0.96, `train/std` (action noise) 1.00 -> 0.148 for `robust_s0` -- the
critic converges and the policy commits to a narrower behaviour, the ordinary shape of a PPO run that is
doing something. Flown directly (5 study tracks + the level2 reference track, nominal, single lap each):
`robust_s0` completes 0 of 6, `contrast_s0` completes 1 of 6 (track 25, 4.355 s), both with RMSE 0.15-0.26 m
against the MPC family's nominal 0.06-0.10 m -- not enough precision for the 3.5-5 cm frame margins these
tracks allow.

**Round 2 (seed 1 of each group), with `ep_rew_mean` available.** `rollout/ep_rew_mean` rose ~30 -> ~275 and
`rollout/ep_len_mean` ~85-97 -> ~457-469 (of a 600-step max) for both -- a direct survival-time improvement,
not only an optimizer-internal one. Flown on the SAME 6 tracks as round 1: `robust_s1` completes 1 of 6
(track 504, up from 0/6), `contrast_s1` completes 2 of 6 (tracks 25 and 504, up from 1/6); gate counts rose
on 4 of 6 tracks for `robust_s1` and 4 of 6 for `contrast_s1`; RMSE narrowed slightly to 0.17-0.22 m, still
well above the MPC family's nominal range.

**Reading rounds 1 and 2 together.** Seed 1 is a real, modest improvement over seed 0 for both groups -- not
noise-level. But both seeds of both groups now show the SAME shape: a handful of gates cleared, occasional
full completions on the easier tracks (25, 504), RMSE stuck around 0.17-0.26 m. With one seed that is
plausibly bad luck (Lesson 7's `racing_s1` again); with two seeds of two independently-varying recipes
showing the same pattern, it starts to look less like pure seed variance and more like 8,000,000 steps not
being enough for a task substantially harder than the vendored recipe solved (wider domain randomization,
freq=100 and the 0.8 s window landing together) -- not a bug: mechanics have checked out at every stage
(loadable, correct shapes, correct routing, monotone improvement in both `ep_rew_mean` and flown gate
counts). Not yet decisive with two of three seeds. If seed 2 (round 3) shows the same pattern, that is the
point to stop before the study sweep and investigate -- more timesteps, a hyperparameter check, or a closer
look at the reward -- rather than continuing on policies this far from the MPC family's precision.

**Hyperparameter fix, screening now (2026-09-23, user: "pay more attention to [PPO hyperparameters and the
training reference distribution]").** `gamma` and `n_steps` are defined in units of steps, and neither was
rescaled when `freq` doubled from the vendored 50 to this recipe's 100 -- "same PPO settings" was numerically
true but not true in effect. `gamma=0.98`'s effective discount horizon is `1/(1-gamma)` steps = 50 steps;
at freq=50 that is 1.0 s, at freq=100 (unchanged gamma) it is 0.5 s -- HALF the vendored recipe's effective
foresight, and shorter than the 0.8 s observation window built for this recipe (section 5a): the policy can
SEE 0.8 s ahead but was only credited for consequences up to about 0.5 s ahead. `n_steps=256` covered 5.12 s
/ 85% of a 300-step (6 s @ 50 Hz) episode in the vendored recipe; unchanged at freq=100 it covers only 2.56 s
/ 43% of the now-600-step episode, so GAE sees less of a typical episode's arc per update. Corrected
defaults, added as CLI flags on both `train_robust.py` and `train_contrast.py` (not hardcoded, so the
original setting stays reproducible): `--gamma 0.99` (restores ~1.0 s), `--n-steps 512` (restores ~5.12 s /
85%), `--batch-size 2048` (scaled with `n_steps` to keep the same 4-minibatch-per-epoch structure). Rounds
1-3 all used the uncorrected 0.98 / 256 / 1024.

**The training reference distribution never asks for gate-threading (2026-09-23, the other half of the user's steer; acted on from 2026-09-23, see section 5c).** Read `ChainedPolyTrajectory.random` directly: every reference the policy
has ever trained on is built from knot positions drawn uniformly in an OPEN [-1, 1] m cube around the
previous knot, independent random velocity and acceleration at each knot, joined by smooth quintics -- no
pinch point anywhere, nothing narrower than the open cube itself. The policy has never practiced converging
precisely onto a point-like target while moving fast then peeling away, which is exactly what a 0.4 m gate
opening with 3.5-5 cm of margin demands. This is structural, not a parameter, and needs a design decision
before any code changes it -- three options on the table, increasing in effort: (1) force a subset of knots
per episode to be gate-like (narrow range, constrained heading), (2) mix in real TOGT-planned track
references for some fraction of episodes instead of purely open-space curves, (3) leave it as a documented
limitation. Deferred until the hyperparameter screen (below) reports back, since it isolates whether the
frequency-scaling mismatch alone explains rounds 1-3's weak flown performance before committing to a
bigger design.

**Screening plan.** One seed pair (seed 0, reused deliberately -- same environment/reference RNG stream as
the existing `robust_s0`/`contrast_s0`, so the hyperparameter change is the ONLY thing that differs from a
direct comparison) at the corrected defaults, before committing to redoing all 6. If it meaningfully closes
the gap to the MPC family's nominal precision, the frequency-scaling mismatch was the dominant issue and all
six seeds get redone at the corrected settings. If it barely moves the six-track screen numbers, that
points at the reference-distribution question above as the more likely bottleneck, and one of its three
options needs designing before more compute is spent. 4 new tests confirm the CLI wiring
(`test_train_scripts_cli.py`): both scripts' `build_parser()` exposes `--gamma`/`--n-steps`/`--batch-size`
with the corrected defaults, the round 1-3 override reproduces the original numbers exactly, and the
4-minibatch structure holds in both settings.

**Screen result (2026-09-23): mixed, does not meet the "meaningfully closes the gap" bar.** One pair at
seed 0, corrected defaults, six-track screen: `robust_s0_screen` improved in survival (7/24 -> 10/24 gates,
0 -> 1 completions) but got *slightly less precise* (mean RMSE 0.206 -> 0.216 m); `contrast_s0_screen`
regressed (9 -> 8 gates, 1 -> 0 completions, RMSE 0.178 -> 0.214 m, ~20% worse). Neither direction is the
clear win that would justify redoing all six seeds at the corrected settings before addressing the
reference-distribution question below. The corrected defaults stay as the CLI default regardless (nothing
about the frequency-scaling fix is wrong, it is simply not sufficient alone), but the standing conclusion is
that the reference-distribution gap is the more likely dominant bottleneck, not the hyperparameter mismatch.

**Path-generation error vs controller error (2026-09-23, the user's question: are RL failures actually
TOGT/track-construction failures in disguise, the same way rejected candidate layouts crash into a frame
they weren't aiming for?).** Directly checked, not inferred. First: this is a different failure mode from
the frame-crossing rejects in section 3 -- those are candidate layouts TOGT itself never made it past the
Phase 1/2 filter and the exact contact test (section 3), so they never reach evaluation; every track actually
flown here has a reference proven contact-free at >=3.0 cm clearance everywhere, by construction. So the
question narrows to: given a verified-safe reference, is the RL failure margin-driven (an aggressive but
technically-clear path with too little room for any controller) or controller-driven (large tracking error
regardless of the path)?

Flew `robust_s0`/`contrast_s0` (round-1 checkpoints) and `M1` on the SAME three tracks (4, 93, 387;
`frame_margin_m` 4.4/3.9/4.6 cm) at `wind_const` lambda=0, logging deviation from reference at the moment of
contact and the max deviation before contact:

```
track    4  M1  COMPLETED  dev_max=8.9 cm   |  robust_s0  dev_at_impact=18.6 cm  |  contrast_s0  dev_at_impact=23.4 cm
track   93  M1  COMPLETED  dev_max=9.0 cm   |  robust_s0  dev_at_impact=31.4 cm  |  contrast_s0  dev_at_impact=49.2 cm
track  387  M1  COMPLETED  dev_max=9.3 cm   |  robust_s0  dev_at_impact=44.7 cm  |  contrast_s0  dev_at_impact=22.4 cm
```

Same path, same margin, same nominal conditions -- only the controller varies. M1 clears all three with
margin to spare (max deviation 8.9-9.3 cm, itself larger than the nominal frame-margin number, which is a
worst-case bound along the path, not a uniform corridor -- what matters is deviation specifically at the
gate-crossing moments, which M1 controls tightly and the RL policies do not). Both RL groups crash on all
three, deviating 2.1-5.5x M1's worst-case deviation and 4.2-12.6x the track's margin at the point of impact (from the table above; corrected 2026-09-25 -- an earlier '3-10x' and '4-11x' did not follow from these numbers) -- not a marginal near-miss a slightly wider corridor would fix. **Conclusion: this is overwhelmingly
a controller precision problem, not a path-generation problem.** The margin figure (3.5-5 cm, tuned
implicitly to the MPC family's demonstrated precision) reflects a precision standard these RL checkpoints
have not reached, which is a different claim from "the path is defective."

**Ruling out a train/eval distribution mismatch as the explanation (2026-09-23, the user's hypothesis that
this could instead be miscoded parameters).** Audited the eval-time observation pipeline directly against
training: `robust_policy.py`'s `RobustPolicyController` imports `WINDOW_DT` from `robust_env.py` (single
source, cannot silently drift) and its manually-reconstructed observation vector
(`[ref-pos, vel, quat, rel_window, L1 sigma]`) matches the vendored env's `_base_obs()` part order and math
exactly -- no coding bug found there. Did find two real, previously-unverified asymmetries: the force-
perturbation channel is drawn unconditionally every training episode from a nonzero-width box (no "off"
state; `--no-perturb` was never used in any of the six runs), and `LambdaLighthouseSensorBatch.measure()`
imposes a fixed one-control-step delay even at lambda=0 -- so neither policy has ever trained on a literally
undisturbed episode, while nominal eval (lambda=0) applies exactly zero force. Tested whether this explains
the large nominal-condition deviations by flying the same round-1 checkpoints at lambda=0.3 and 0.6 (inside
their own training range) on the same three tracks: no meaningful improvement. `robust_s0` stays in the same
27-45 cm max-deviation band across lambda in {0, 0.3, 0.6}; `contrast_s0` gets *worse* as lambda increases
(60 -> 90 -> 136 cm max deviation on track 93; corrected 2026-09-25, an earlier note gave 23 cm, which is another track's value). If nominal eval were out-of-distribution in a way that
mattered, moving into the trained range should have closed the gap; it does not. **This rules out a
train/eval mismatch as the dominant explanation** and leaves the reference-distribution gap above (no
gate-threading anywhere in `ChainedPolyTrajectory.random`) as the best-supported explanation, now backed by
three independent lines of evidence: the reward-curve plateau, the same-path/same-margin M1 comparison, and
this lambda-sweep showing the failure is flat-to-worsening regardless of disturbance level (i.e., not a
robustness gap -- a baseline precision gap present with or without disturbance).

**Cross-checked against Lesson 7's own pre-existing benchmark data (2026-09-23, user asked to verify against
the original racing seeds before trusting new measurements).** `racing_s0/s1/s2` and `v5_s0` -- the original
vendored-recipe RL seeds, evaluated through a completely different pipeline (`race_eval.py`/
`compare_models.py`, not this lesson's `driver.py`) on entirely different tracks (closed-form/tube/pole-aware
plans, not the Lesson 9 study tracks), written months before any Lesson 9 code existed -- show the identical
signature. Lesson 7 section 2: at NOMINAL conditions, harness clock, "They are not more precise. Their max
deviation (0.38-0.54 m) is the baseline's (0.39-0.52 m), not the plain and offset-free MPC's (0.10-0.35 m on
the f0.95 rows)" -- the same RL-vs-MPC precision gap, present before any of this lesson's freq/window/DR
changes were written. Lesson 7 section 4.2's replay table: `racing_s2` ended on the gate-3 frame at 0.53 m
max deviation, `v5_s0` on the gate-1 frame at 0.33 m -- and Lesson 7's own stated conclusion (section 5,
point 2) is "A 0.4-0.5 m deviation at 4-5 m/s is a frame," the same magnitude and the same mechanism reported
above for `robust_s0`/`contrast_s0` (23.5-60 cm max deviation before impact). This rules out "something
Lesson 9 specifically introduced" even more thoroughly than the code/observation-pipeline audit above: the
pattern reproduces across two lessons, two harnesses, two independent sets of trained checkpoints, and two
track geometries, so the root cause (training references that never demand narrow-passage precision)
predates Lesson 9 entirely -- Lesson 9 measured it with an exact contact model and calibrated margins instead
of a rule of thumb, but did not create it.

**Actuator saturation and peak speed, sampled the same way Lessons 6-8 did (2026-09-23, user asked to check
the same quantities).** Extended `driver.py`'s `Flyer.fly()` with an optional per-step action log (behind the
existing `keep_path` flag, additive, all 6 pre-existing `test_driver.py` tests still pass) and flew
`M1`/`robust_s0`/`contrast_s0` on the same three tracks, logging commanded thrust and roll/pitch against
`THRUST_MIN`/`THRUST_MAX`/`RPY_MAX` (`crazy_track.controllers.utils`) and flown peak speed against each
track's planned `v_peak`. Thrust: no controller runs into a chronic ceiling -- mean commanded thrust sits
around 11-12 m/s^2 of an 18.44 m/s^2 limit for all three; RL hits the high-thrust ceiling more often than M1
(5.8-16.1% of steps vs 0.7-1.9%), reading as "correcting harder," not "starved of thrust." Attitude: both RL
policies hit EXACTLY 0.700 rad on max pitch on every track -- `robust_policy.py` caps physical roll/pitch at
`RPY_MAX * 0.7`, a convention copied verbatim from the vendored `DATTPolicyController` (inherited from Lesson
2/7, not introduced here), and the share of steps within 2% of that ceiling is 1.3-25.1% across the six RL flights (median 9.5%); M1, with the full `RPY_MAX=1.0` available, reaches 1.000 on pitch on tracks 4 and 93 and on roll on track 387 (5.4-8.4% of steps within 2% of its own ceiling), so it also uses its full range but is not pinned at a self-imposed 70% cap (corrected 2026-09-25: an earlier '7.6-25.1%' mixed in M1's track-4 figure). Peak speed is confounded by episode length (RL
crashes at 150-350 steps, before the faster later section of the lap M1 flies through at 400+) and is not
read as an independent speed deficit. Net conclusion: this does not surface a new independent root cause --
it adds texture to the established one. The RL policies are fighting larger tracking errors (more thrust,
routinely maxing attitude authority to correct) rather than tracking with margin to spare the way M1 does;
the 70% pitch cap is a real, previously-unmeasured, non-Lesson-9-specific constraint, best read as a symptom
of chasing bigger errors rather than a separate cause.

## 5b. The contrast-group training recipe (`contrast_env.py`, `train_contrast.py` -- code written and
mechanics-tested 2026-09-22, `test_contrast_env.py`, 5 tests; trained in the three open-space rounds; the gate-aware
contrast environment of section 5c is built but not trained)

**Why this group exists.** The robust recipe (section 5a) matches its domain-randomization ranges to the
study's frozen ceilings precisely -- three channels, each at 0.8x the same numbers the study measures
against. That precision raises a fair objection: a crossover near the training edge could reflect something
about learned control, or it could just mean the policy was trained on the exam. Limitation 3 names this
gap; this group is how the study answers it instead of only disclosing it (user request, 2026-09-22).

**What is held equal, and why.** `ContrastTrackingEnv(RacingTrackingEnv)` keeps the SAME racing envelope
(`_sample_traj`, unchanged), the SAME PPO settings, and the SAME preview-window and frequency fixes as the
robust recipe (0.8 s / freq=100). Those three are held equal ON PURPOSE, not varied: they correct an
asymmetry between the MPC and RL families that has nothing to do with disturbance-training matching
(limitation 14), so leaving them at the vendored defaults here would confound the ONE comparison this group
exists to make. `contrast_env.py`'s class body is two lines on top of `RacingTrackingEnv` for exactly this
reason -- everything that should differ from the robust recipe is left untouched at the vendored default
(`RacingTrackingEnv._sample_perturb` already delegates to `DATTTrackingEnv`'s +-3.5 m/s^2 box when
`perturb=True`; `DATTTrackingEnv.__init__` already builds the vendored `LighthouseSensorBatch`, a per-episode
noise SCALE in U(0, 1.5) uncoupled from the update rate, when `noisy_sensor` is set), and there is no mass
channel at all -- the vendored recipe never had one, and this group does not gain one either.

**Seeds: 3 (0, 1, 2), not 2.** The user's reasoning: 6 seeds total (3 robust + 3 contrast) split into 3
concurrent-training pairs uses every slot under the 2-concurrent-session hardware limit (section 12); 5
would waste one (measured later: two concurrent runs each take about twice as long, so pairing uses the slots without shortening the total). It is also better science on its own terms -- Lesson 7's `racing_s1` showed a single bad
seed is not a bad recipe, and a 3-vs-2 seed-count asymmetry between the robust and contrast groups would
make that harder to read cleanly for whichever side had fewer.

**Evaluation.** A contrast-trained model is evaluated through the SAME `robust_policy.RobustPolicyController`
a robust model uses (`driver.py`'s `robust:<path>` spec) -- that wrapper's only job is the 0.8 s window at
the harness's own frequency, correct for either group; nothing about it assumes matched disturbance
training. `datt:<path>` (the vendored 0.6 s window) is wrong for both.

**Chart treatment (agreed, not yet built -- phase 8).** Thin/muted lines, their own colour family, distinct
from both the model-based and robust-RL lines; never folded into either group's mean. Used only to check
whether the robust family's crossover location moves relative to this baseline, or stays where it is.

## 5c. The RL viability investigation and the gate-aware recipe (2026-09-23 to 2026-09-25)

Results that followed the first training rounds (section 5a): the control-authority tests, the locality
finding, the four gate-aware versions and their pre-registered bar, the validation baselines, the lambda > 0
preview, the saved recipe, and the build of the gate-aware contrast group. Moved here from limitation 14 in
the 2026-09-25 audit; wording is unchanged except where a correction is marked, and every correction is
listed in section 13 (errata).

**Resolved into two tests, one already run (2026-09-23).** The user proposed two ways to close this gap:
equalise the MPC family DOWN to RL's 0.7 rad ceiling (cheap: `mpc_dev.py`'s `rp_max` is already a
spec-string-configurable hard OCP bound, default 1.0 -- `mpcdev:...,rp_max=0.7` needs no code change), or
give RL the MPC family's full authority via one experimental seed (expensive: requires retraining, since
training and eval apply the same 0.7 scale -- widening only eval would just feed a wider range to a
network that never learned to use it). Ran the cheap one first: M1 with `rp_max=0.7` on the same three
same-path tracks (section 5a's impact-deviation comparison) still completed all three -- precision
degraded (worst case, track 93: RMSE 0.053->0.109, max deviation 0.090->0.286 m, itself 7x the track's
own 3.9 cm margin, and it STILL threaded every gate) but nothing close to a crash. This rules out the
0.7 rad cap as SUFFICIENT on its own to explain the RL crashes -- if it were, capping M1 the same way
should have crashed it too. It does not rule out the cap costing something ON TOP of RL's existing
(much larger) tracking-precision gap, which only an actual RL run under full authority can show. Built
the mechanics for that second test (`full_authority_env.py`: `FullAuthorityTrackingEnv(RobustTrackingEnv)`,
overrides ONLY `_denorm_action`'s scale to `RPY_MAX*1.0`; `full_authority_policy.py`:
`FullAuthorityPolicyController`, the matching eval-time wrapper, deliberately not a `RobustPolicyController`
reuse for the same silent-mismatch reason `RobustPolicyController` itself exists; `train_full_authority.py`:
one seed only (0, locked, reusing `robust_s0`/`robust_s0_screen`'s RNG stream -- isolates authority as the
only new variable against the most current seed-0 baseline); `driver.py` routes a new `robust_full:<path>`
spec to the matching controller, `robust:<path>` on one of these models would silently apply the wrong
scale). 6 new tests (`test_full_authority.py`), all pass; full lesson9 suite re-run end to end (all 9
files) and confirmed at 55 tests, all passing.

**Result (2026-09-23): not a fix, closed out cleanly.** Trained one seed (0, corrected hyperparameters,
`rpy_scale=1.0`, 8M timesteps). The training curve itself already signalled this before evaluation: from
~4M steps on, `ep_rew_mean`/`ep_len_mean` plateaued in the SAME band as `robust_s0_screen`'s own curve at
the same step counts (270-290 / 480-530) -- if widened authority were meaningfully helping, this curve
should have broken past that ceiling, and it did not. Flown evaluation on the same 6-track screen set
(level2 + study 4/25/93/387/504) confirmed it: `full_authority_s0` 8/24 gates, 0/6 completions, mean RMSE
0.2044 vs `robust_s0_screen`'s 10/24 gates, 1/6 completions, mean RMSE 0.2161 -- a wash per-track (better on 2 tracks -- level2 and 93 -- worse on 4, including the loss of track 504, the one track `robust_s0_screen` actually completed), not a clean
improvement in either direction. Combined with the earlier M1-capped-at-0.7 diagnostic (0.7 rad was
sufficient for M1 to complete all three same-path tracks), this rules control authority OUT as a
contributing explanation, cleanly rather than left open: it was checked, not assumed. Between this,
the lambda-sweep (rules out train/eval distribution mismatch), and the same-path M1 comparison (rules out
path-generation error), the reference-distribution gap is now the only substantial untested explanation
remaining -- nothing tried so far (more timesteps, corrected hyperparameters, more control authority) has
touched the specific skill of converging tightly on a narrow, oriented target.

**Locational finding (2026-09-23, user's follow-up): the RL error is not just comparable-and-unlucky, it
is specifically worse at the gates.** M1's own capped-authority result (0.7 rad, max deviation 0.286 m on
track 93, still completed) raised a sharp question: if the RL family's deviation magnitude is comparable,
does it also land where M1's does (between gates, harmless) or specifically AT the crossings (fatal)?
Checked directly: binned per-step deviation from the reference into "near a gate" (within 0.3 s of a
`gate_times` crossing) vs "between gates," on the same three tracks, for M1 at both authority levels and
`robust_s0`/`contrast_s0`. M1 shows NO gate-locality at either authority level -- near-gate and
between-gates deviation are statistically the same (e.g. track 4: 0.058/0.089 near vs 0.051/0.081 far;
track 387: 0.058/0.091 vs 0.057/0.093) -- it does not need a "gate mode" because it is precise everywhere.
Both RL groups show the OPPOSITE pattern, consistently, on every one of 3 tracks: mean deviation near a
gate is 34-148% HIGHER than between gates (robust_s0: track 4 +34%, track 93 +48%, track 387 +148%;
contrast_s0: +52%, +43%, +82%). The approach trend sharpens this further: fitting a linear slope to
deviation over the final 1 s before each failure, 4 of 6 RL cases are clearly DIVERGING into the crash (slope +0.13 to +0.47 m/s, e.g. contrast_s0 track 93: +0.470, robust_s0 track 387: +0.323), one is flat (+0.010) and one converging (-0.155) (corrected 2026-09-25: an earlier '5 of 6' counted the flat one), while M1 is flat-to-slightly-
positive everywhere and, on its own hardest case (track 93 capped to 0.7 rad), actively CONVERGING
(-0.228 m/s) in the same window. Conclusion: this is not "RL fails by a generic amount that happens to
coincide with a gate" -- its error specifically GROWS worst exactly where the geometry punishes it most,
and is often still growing at the moment of impact rather than plateaued. This sharpens, not just
supports, the reference-distribution explanation: the reward (`exp(-2*err) - 0.02*||action||`, applied
uniformly every timestep in `datt_env.py`'s `step()`) gives no extra weight to error near a narrow
feature, and nothing in `ChainedPolyTrajectory.random` ever creates one to weight. Whichever of the three
reference-distribution options gets chosen, this finding argues it needs to specifically create moments
where being off-target has a sharp, LOCAL cost -- not just narrower open-space curves on average.

**The viability screen (2026-09-23): the smaller question first.** The user reframed the goal before
committing further budget: not "does this close the whole study's gap" (the full research question), but
"can the recipe pass a viability gate at all" -- a strictly smaller, prior question. Checked whether
training has ANY contact model before designing a fix: it does not -- `datt_env.py`'s `step()` crash
condition is `(pos[:, 2] < 0.05) | (err > 2.0)` only, and the `Sim` training constructs carries no gate
geometry at all. So neither reference-distribution option, alone, reproduces what the locality finding
calls for (a moment where being off-target is FATAL, not just costlier) -- a training-time analog of
`contact.py`'s test is a required third ingredient, not a separate later step.

Built `gate_aware_env.py`/`train_gate_aware.py`: `GateAwareTrackingEnv(RobustTrackingEnv)`, one seed,
trains 100% of episodes on the 3 real dev-track references (not mixed with `ChainedPolyTrajectory.random`
-- deliberately the sharpest possible test of "can it learn this at all") with a genuine gate-contact
crash added to `step()` (`contact.py`'s `pose_contacts`, already batch-shaped across parallel envs,
grouped by which of the 3 dev tracks each env is currently flying; same `-5.0` penalty the existing
floor/divergence crash already gets). Two problems found and fixed while building this, both by reading
source rather than assuming:

1. Episode length: the 3 dev tracks' own durations (5.8-7.2 s) exceed the vendored 6.0 s `episode_time`
   default, which would have truncated an episode before gate 4. `SampledRaceTrajectory.pos(t)` safely
   HOLDS at the final point past its own duration (confirmed in source), so this needed only widening
   `episode_time` to 7.0 s, not per-episode-length engineering.
2. A more fundamental one: these are GROUND-START plans (`RACE_START` z=0.01, the plan CSV's own t=0 at
   z=0.05 -- Lesson 8's ground-start convention), while `_set_states` (vendored) always starts an episode
   from REST at `self._traj[i].pos(0.0)`, and training's crash condition is `pos[:, 2] < 0.05` -- placing
   the drone at the plan's own start would crash it on step one, every time (eval never hits this because
   its divergence threshold is `pos[2] < -0.3`, not `< 0.05`; training was simply never built for a
   ground start). v1 fixed it with `TimeShiftedTrajectory` (SUPERSEDED by v2 below, which fixes the
   floor rule instead), which re-origined each dev track at the first moment
   its OWN climb clears z > 0.5 m (found by scanning each plan's climb profile: 0.28-0.38 s before gate 1
   on all three, since the climb on these ground-start plans is late and steep). Accepted trade-off, not
   engineered around further for this screen: gate 1 gets noticeably less lead-time than gates 2-4 as a
   result, which is fine for a viability check whose question is "can it thread a real gate with a real
   contact consequence at all" -- gates 2-4 give a cleaner read on that anyway.

7 new tests (`test_gate_aware.py`), including a wiring check that forces a known-overlap pose into the
real Sim (the centre of a gate's own contact box) and confirms `step()` reports it as a crashed, `-5.0`
episode -- not just that `contact.pose_contacts` works in isolation (already covered by `test_contact.py`).
All pass. Evaluated through the existing `robust:<path>` spec unchanged (authority stays at 0.7 rad, the
already-ruled-out variable, held fixed on purpose). v1 was trained; its result and the v2 that replaces
it follow.

**v1 result (2026-09-24): 0/6, and the shape of the failure is the finding.** Trained one seed, 8M
steps. The curve was healthy and still climbing at the end (`ep_rew_mean` 8.5 -> 282.0, `ep_len_mean`
48 -> 516 of a 700-step maximum, `explained_variance` 0.85-0.95 from ~160K steps on, `approx_kl`
0.004-0.007, `clip_fraction` 0.02-0.07). Mid-run the user found convergence disappointing and proposed a
higher learning rate; the diagnostics did not support it (still improving at 23% of budget, no
instability, and a bigger step on a landscape with a -5.0 cliff is a standard way to destabilise PPO), so
it was left alone -- and the run kept improving. The raw reward (~60 at the time) looked small next to
the ~270-290 of the open-space runs only because the two reward structures are not comparable: episodes
now end on a hard contact cliff, and cumulative reward is capped by how long an episode survives.
Flown on the 6-track screen at lambda=0: **0/6 completions, 0/24 gates, mean RMSE 0.3128**, and every
one of the six is a gate-1 contact (`level2` t=1.18 s, study 4 t=1.19, 25 t=1.46, 93 t=1.30, 387 t=1.44,
504 t=1.71). A pool too small to generalise, or generic imprecision, would scatter failures across gates;
a wall at gate 1 on every track is structural.

Two candidate causes, not separated by that data: (1) the eval harness always flies a real ground start
(`GroundStartTrajectory`: 0.2 s hold, quintic climb from z=0.01, then the plan) and v1 never trained on
one -- it re-origined every episode airborne; (2) gate 1 got 0.28-0.38 s of lead-time in training against
the plan's real ~1 s+. The user took (1) as the working hypothesis and asked whether it should then hit
the MPC family too. It should not, and structurally so: M1 has no training distribution to be out of. It
re-solves from an explicit dynamics model at every step, so a ground launch is another initial condition,
not a situation needing prior exposure (ground-effect thrust gain was checked in Lesson 7 and left out as
negligible, so the model is not altitude-dependent in any way that matters here). It is also already
measured: M1 flew this exact `GroundStartTrajectory` path through `driver.py` on tracks 4/93/387 with
under 9 cm of deviation. "Never saw this regime" is a failure only a learned controller can have.

**v2 (2026-09-24): remove the gap at its source.** `gate_aware_env.py` rewritten; `TimeShiftedTrajectory`
deleted. Three changes, each reusing something already validated rather than a new number:
* references are `driver.build_traj(track)` for the 3 dev tracks -- the same `GroundStartTrajectory` eval
  constructs, with gate 1 at its normal lead-time (this removes cause 2 as well as cause 1);
* the training floor crash moves from the vendored `pos[:, 2] < 0.05` to `pos[:, 2] < FLOOR_Z`,
  FLOOR_Z = -0.3, which is `driver.py`'s own divergence check (`Flyer.fly`: `pos[2] < -0.3`). 0.05 was only
  ever safe because every other training env starts well clear of the ground; a z=0.01 start trips it on
  step one. -0.3 sits below the physical floor, so it catches only a fall through it, as in eval -- no
  time-windowed exemption to get wrong. `err > 2.0` still catches "gave up and sat on the floor" once
  the reference moves away. Chosen over a windowed exemption because it is simpler and is eval's
  convention, unconditional over the whole lap, validated throughout this project;
* `episode_time` 8.0 s (was 7.0 for v1, 6.0 vendored): the full ground-start durations are 5.795 / 7.196 /
  6.423 s, and the longest must not be truncated before gate 4.
The simulator was never the obstacle -- every controller already flies this launch in the harness; the
obstacle was a crash rule written when nothing trained near the ground. 10 tests (`test_gate_aware.py`),
all pass: the reference is bit-identical to what `driver.build_traj` returns, starts at `RACE_START`
(z<0.05), and a ground start does NOT crash on step one (the test that would have failed under the
vendored threshold); FLOOR_Z is pinned to the literal in `driver.py`'s source so the two cannot drift;
gate 1 keeps a >1 s lead-time on all three tracks; the forced-contact wiring test still passes.
Residual, stated plainly: v2 removes both candidate causes at once, so a pass will not say which one
mattered; and it still trains on only 3 tracks, so a pass on the 6 held-out screen tracks is evidence of
transfer while a fail may be either cause or a pool too small -- the overfitting question raised
earlier is unchanged and is answered by the held-out evaluation itself.

**v2 result (2026-09-24): 0/6 again, 2/24 gates -- and the training-track check says why it is not simple
overfitting.** Training was healthy (`ep_rew_mean` 60 -> 334, `ep_len_mean` 155 -> 521 of 800,
`explained_variance` 0.94-0.99, still climbing slowly). Flown on the 6 held-out screen tracks at
lambda=0: 0/6 completions, 2/24 gates, mean RMSE 0.2273. Four tracks still contact gate 1 (t=1.3-1.5 s);
`level2` and study 387 now clear gate 1 and hit gate 2 (t=2.23, 2.53 s). So the missing ground start
was at most a minor contributor: a real launch and a normal gate-1 lead-time moved 2 of 6 tracks by one gate.
Then flown on its OWN 3 training tracks, same harness, lambda=0: 1/3 completes (100023, 4/4 gates);
100082 and 100092 both `missed_gate_1` with NO contact, max deviation 0.79 and 0.86 m. Deterministic vs
stochastic actions make no difference (4 stochastic samples per track: identical outcomes), so this is not
an eval-mode artefact. Training survival of ~5.2 s on average is therefore not evidence of threading.

Cause, partly verified: `GateAwareTrackingEnv.step()` ends an episode only on contact, `err > 2.0`, or
the floor -- there is no termination and no dedicated penalty for missing a gate, only the tracking-error term (the eval harness ends the lap when the target gate's
clock + 1.0 s passes without a crossing inside the opening). Certain, from the code. Inferred, supported
by the no-contact 0.8 m misses: the contact cliff teaches "stay away from the frame", and flying wide of
the gate is a cheaper way to do that than threading it -- at 0.8 m error the tracking reward is still
exp(-2*0.8) = 0.2 per step, positive, versus -5.0 for a touch. Not verified: that this is what the policy
actually does (would need the in-training-env gate-crossing rate), and it does not by itself explain the
four held-out gate-1 CONTACTS, which are the opposite failure. Candidate fix, not built: end the episode
on a missed gate using `driver.py`'s own crossing test (target gate's clock + 1.0 s without a crossing
inside `RaceGate.HALF_OPENING`), so training and eval share one definition of failure. This would be the
third ~2 h run in this line; the viability question's budget is the user's call, not assumed here.

**Trace, and a correction (2026-09-24).** The user asked whether a segmented path does not already force
the controller to a position at some point. It does -- the tracking reward pulls toward the gate -- and
"missing a gate costs nothing" above was an overstatement: it costs the tracking term. Deviation traces
(harness, lambda=0) on the two failing training tracks: launch is fine (dev <= 0.14 m through t=0.5 s,
altitude tracked throughout); 100082 goes 0.11 m (t=1.0) -> 0.40 -> 0.67 -> 0.79 m (t=1.75) -> 0.25 m
(t=2.25), gate 1 at 1.43 s; 100092 goes 0.10 -> 0.44 -> 0.84 m (t=1.5) -> 0.39 m (t=2.0), gate 1 at 1.23 s.
The excursion is lateral (xy), begins ~0.2 s before the gate and recovers after it; the passing track
100023 has 0.095 m at t=1.25 and no swerve. Consistent with skirting, and equally with "cannot hold the
line at speed" (same trace, different fix) -- not separated. The sharper statement of the asymmetry:
contact forfeits the rest of the episode (order 300 remaining steps at 0.5-0.7 reward, ~200, a rough
estimate not a measurement) on top of the -5.0, while a 0.8 m swerve costs ~0.3 in total, so contact is
~2 orders costlier than skirting and a policy that cannot reliably hold a 3.5-5 cm margin is pushed to swerve.
Terminating on a missed gate would make skirting forfeit the episode too, so only threading keeps the
return; it cannot distinguish 'will not' from 'cannot'.

**v3 (2026-09-24): terminate any miss; user asked for milder penalties.** User: "reward hacking, huh.
then terminate any misses, and i think the penalties should be kept a bit more modest rather than -5.0,
maybe -3.5 or so, but do what you think is best." Built in `gate_aware_env.py`:
* `gate_progress(prev, cur, t, dt, gates, gate_times, idx)` is `driver.py`'s target-gate rule for one env
  and one step (crossing = gate-frame x changes sign, inside `RaceGate.HALF_OPENING`, within +-1.0 s of the
  gate's clock; miss = clock + 1.0 s without one). Chosen over terminating at the moment of an
  outside-the-opening plane crossing: that would flag failures earlier but is not eval's definition, and an
  extended gate plane can be crossed elsewhere by a legitimate path; the timeout rule has no such false
  positives. Cost: up to 1 s between the miss and its penalty (0.99^100 = 0.37 discount) -- acceptable.
* Finishing all gates does NOT end the episode: terminating on success would forfeit the remaining reward
  for succeeding. It runs to truncation (8 s, longest lap 7.2 s).
* Penalty split: gate contact and a miss = -3.5 (GATE_PENALTY); floor/gross divergence keep the vendored
  -5.0 (FLOOR_PENALTY). Stated plainly: the terminal constant is the smaller lever -- the forfeited
  remaining return (~200, rough) dominates it, so -5.0 -> -3.5 is a scale choice (gentler on the value
  function) and is not expected to change behaviour by itself. The change that matters is the miss
  termination.
* Per-rollout logging: the env keeps cumulative `stats` (episodes, contact, miss, floor_div, trunc,
  gates_sum, full_laps) and a callback in `train_gate_aware.py` writes `gate/passed_per_ep`,
  `gate/frac_{contact,miss,floor_div,trunc}` and `gate/full_lap_frac` to tensorboard. Added because v2's
  aggregate curves could not distinguish threading from surviving; `gate/full_lap_frac` is the
  training-time completion rate.
Verified, not assumed: replaying six real flown paths (M1 and the v2 policy on the three dev tracks)
through `gate_progress` agrees with `driver.py`'s gates-passed and fail reason on all six -- M1's
completions (100023, 100082) never trigger a false miss, its 100092 lap ends in a gate-4 contact
(idx=3, no miss), and the v2 policy's misses fire at exactly gate time + 1.0 s (1.43 -> 2.43 s,
1.23 -> 2.23 s). 20 tests in `test_gate_aware.py` (10 new): the rule's window edges, a late crossing
inside the window counting and one outside it not, an outside-the-opening crossing neither advancing nor
missing early, a finished lap never missing, the source constants pinned to `driver.py`, the env ending a
miss at -3.5 and counting it as a miss (not contact/floor), no early termination, divergence keeping -5.0,
and the stats summary. Residual, unchanged: this cannot separate "won't thread" from "can't hold the
line", and it still trains on 3 tracks. Note 100092 is the tightest dev track (3.05 cm margin) and M1
itself does not complete it. (v3 was then trained: result below.)

**v3 result (2026-09-25): it learns the task on its training tracks; on unseen tracks it is at the
open-space baseline, not above it.** 122 min wall-clock, 8M steps. Training-side, from the new `gate/*`
scalars: `gate/passed_per_ep` 0.00 -> 0.05 -> 0.26 -> 1.07 -> 1.40 -> 2.11 -> 2.11 -> 2.71 -> 2.88 (at
0/1/2/3/4/5/6/7/8M), `gate/frac_miss` 0.49 (1M) -> 0.00 (8M), `gate/frac_contact` 0.41 at the end, and
`gate/full_lap_frac` 0.26 / 0.21 / 0.29 / 0.53 at 5 / 6 / 7 / 8M (pooled over the last 10 rollouts: 0.535 of 142
episodes; the final rollout alone is only 17 episodes, so single-rollout values are noisy). Still rising at
8M, `explained_variance` 0.95-0.99, `ep_len_mean` 580 of 800. Misses vanishing while contacts remain is
what the reward argument predicted: the cheap way out (swerving wide) is closed and what is left is
failing to hold the line (inferred from the two fractions, not from a trace).
Flown at lambda=0 in the harness: on its OWN 3 training tracks 2/3 complete, 10/12 gates (v2: 1/3);
on the SELECTION SET (level2 + study 4/25/93/387/504, limitation 15) **1/6 completions, 11/24 gates, mean
RMSE 0.2640** (only study 4 completes; level2 fails at gate 2, 25 at gate 1, 93 at gate 2, 387 at gate 4,
504 at gate 1). Failures no longer wall at gate 1. Against the same six tracks:
    open-space `robust_s0_screen`   10/24 gates  1/6  RMSE 0.2161
    control-authority screen         8/24         0/6  RMSE 0.2044
    gate-aware v1 (airborne start)   0/24         0/6  RMSE 0.3128
    gate-aware v2 (ground start)     2/24         0/6  RMSE 0.2273
    gate-aware v3 (miss-terminated) 11/24         1/6  RMSE 0.2640
So v3 matches the open-space baseline on completions and gates and is looser on RMSE; by the pre-stated
bar (completions at lambda=0 on the screen) it did NOT pass. Training-side full-lap rate 0.535 against
held-out 1/6 is suggestive of a transfer gap (3 training tracks), not conclusive: n=6 gives 1/6 a very wide
interval, and the training rate is stochastic-policy-with-disturbances while the flight is deterministic at
lambda=0. Precision did not improve: rmse 0.22-0.37 m (M1 0.04-0.06) and max deviation 0.35-0.68 m on the
selection set, 0.52-0.64 m even on completed training laps -- it now threads gates from a loose path.
The 10 untouched study tracks were NOT flown (limitation 15): the recipe is still being iterated.
Not done, options for the user: a larger disjoint training-track pool (the original option 2; generation
yield ~1%, seeds must avoid study 4..1318 and dev 100000+), longer training (curves still rising), or
stopping the viability line here.

**v4 (2026-09-25): a larger pool and longer training, together.** User: "i think 1 and 2 can be tackled at the
same time." Built:
* `gen_pool.py` (parallel, 8 workers; `gen_tracks.py` gained `train`/`val` roles and per-window manifests)
  generated two pools from FIXED seed windows chosen in advance, keeping every accepted track (no truncation to
  a target count): `train` seeds 200000..205999 -> **111 tracks of 6000 candidates (1.85 %)**, `val` seeds
  300000..301999 -> **22 of 2000 (1.10 %)**. Same generator and filters as the study tracks, so the training
  distribution matches the study distribution by construction while every evaluated instance is held out;
  reject-reason mix identical to study/dev (arena, ref_contact, frame_crossing, frame_margin), every margin
  >= 3.0 cm (train median 3.8 cm, max 6.9), laps 4.45 s median (max 6.48). The 1.85 % vs 1.14 % (study) yield
  is unexplained beyond window-to-window variation (train windows ran 4-11 passes per 400 candidates, val 1.10 %).
  Generation cost is about 0.25 s per candidate per worker: 8,000 candidates took about 4 minutes on 8 workers
      (container log times 18:07-18:11; an earlier draft said ~0.5 s and ~10 min, a miscalculation from a 6-candidate
      timing run that included startup) -- far cheaper than feared, which is why the pool is 114 tracks (3 dev + 111 train), not the 5-7 first discussed.
* `gate_aware_env.py` v4: the pool is discovered from `tracks/train/`, `EPISODE_TIME` is computed from the
  longest pool track (8.41 s -> 9.0; was 8.0 for the dev tracks), and the contact check moved into a pure
  `gate_contacts()` with a distance pre-filter (`pose_contacts` already ignores poses > `NEAR` from a gate;
  dropping them first avoids building a scipy rotation per track group per step). Equivalence to the
  unfiltered check is a test (4000 random poses incl. >100 real contacts, agree exactly); env step cost went from +2.07 ms over the robust env (4.66 vs 6.73 ms per 16-env step) to +1.37 ms (4.87 vs 6.24 ms; two separate benchmarks).
* `train_gate_aware.py` v4: 16M steps (v3's curves were still rising at 8M; estimated at ~4.3 h from ~15 ms per
      vector step, measured 5.70 h -- see the time correction below), a checkpoint every 2M steps (`ckpt/ppo_<steps>_steps.zip`) so ONE run shows
  whether longer training helps transfer or just fits the pool better, refuses to run if the train pool is
  empty, logs the pool. `eval_pool.py` flies checkpoints/M1 over a role's tracks in parallel at lambda=0.
* 24 tests in `test_gate_aware.py` at that point; new ones pin the pool: disjoint from study/val/dev, inside its fixed
  seed windows, contains no selection-set track (4/25/93/387/504) and no untouched study track, has >= 20
  train tracks, and every pool track is reachable by the sampler. One test was flaky and is fixed: the
  vendored `_set_states` adds +-0.05 m start noise, so "every env starts below z=0.05" was never guaranteed
  (it passed before by the luck of the RNG draw); it now asserts z0 < 0.07 and that at least one env starts
  below the vendored threshold, so the test still proves the point.

**Validation baselines, flown BEFORE v4 exists (lambda=0, 22 `val` tracks, no re-tuning of anything):**
    M1                       22/22 completions   88/88 gates   mean RMSE 0.0525   median max dev 0.085 m
    open-space robust_s0     2/22                33/88         0.2115             0.349 m
    gate-aware v3 (3 tracks) 3/22                42/88         0.2481             0.455 m
Three consequences. (a) M1 at Lesson 8's parameters completes every one of 22 tracks it was never tuned on,
the empirical answer to "must the MPC parameters be adjusted again": no. (b) v3's 3/22 confirms its 1/6 on the
selection set was not a fluke, and that a 3-track pool bought a small gain over open-space (42 vs 33 gates)
and no meaningful gain in completions (3 vs 2). (c) `val` gives n=22 instead of n=6 for judging transfer.

**Pre-registered bar for v4 (written before any v4 checkpoint has been flown; proposed by me, the user may
change it before the run finishes).** Headline = the FINAL (16M) checkpoint, deterministic, lambda=0, one seed,
on the 22 `val` tracks. Viable: >= 11/22 completions (half of M1's 22/22). Promising but not viable: 7-10/22
(at least +4 over v3's 3/22, roughly 2.5 binomial sd at n=22). No change: <= 6/22 -- pool size and steps did
not fix transfer, and the viability line is then a stop-or-rethink decision. All checkpoints (4/8/12/16M) are
reported as a curve, but picking a non-final checkpoint on `val` turns `val` into a selection set; the 10
untouched study tracks remain the clean set and are flown only once the recipe is frozen. One seed: a pass is
evidence the recipe can be viable, not that it is consistent (Lesson 7's `racing_s1` is the reminder), and a
frozen recipe would still need its seeds trained. (v4 was then trained: interim result below.)

**v4 interim (2026-09-25, run at 13.4M of 16M steps; NOT the headline).** The user asked whether it had
converged. Training side, from `gate/*` pooled by episode count over 2M-step windows: full-lap rate 0.17 (4-6M)
-> 0.21 -> 0.25 -> 0.30 (10-12M) -> 0.29 (12-13.4M, 2766 episodes, sd ~0.009, so no change), gates/episode 2.34
-> 2.33, contact 0.65, miss 0.04; reward and episode length flat since ~7M (~340, 470-540); explained variance
0.92-0.98. Mostly plateaued, possibly still creeping. The training full-lap rate (0.30 over 114 tracks) is not
comparable to v3's 0.535 (3 tracks). Transfer side, flown at lambda=0 on the 22 `val` tracks (allowed by the
pre-registration as a curve; it is not the headline and must not be used to stop or pick a checkpoint):
    ckpt   completions   gates   mean RMSE   median max dev
    2M     0/22          0/88    0.2673      0.516
    4M     3/22          43/88   0.2579      0.506
    6M     10/22         64/88   0.2577      0.526
    8M     11/22         65/88   0.2194      0.431
    10M    10/22         67/88   0.2059      0.330
    12M    14/22         74/88   0.2017      0.337
    (M1 22/22, 88/88, 0.0525, 0.085; open-space 2/22; v3 3/22)
Reading, with caveats. (a) The 12M checkpoint's 14/22 is above the pre-registered viability line (11/22), but
the headline is the 16M checkpoint, and at n=22 the binomial sd is ~2.3, so 10, 11, 10, 14 across 6-12M is
consistent with a plateau near 11-12 plus noise as well as a slow rise; gates (64 -> 65 -> 67 -> 74) and RMSE
(0.258 -> 0.202) drift the right way more steadily. (b) At matched 8M steps the 114-track pool completes 11/22
against the 3-track pool's 3/22 (v3, same recipe, same step count; one seed each, a difference of 8 tracks,
~3 sd): the pool, not just the steps, moved transfer. Within v4, steps also mattered early (4M 3/22 equals v3's
final; 6M 10/22). Both levers were needed, which is why doing them together was reasonable; this cannot
apportion the credit beyond that. (c) Precision is still far from M1: median max deviation 0.34 m vs 0.085 m
and RMSE 0.20 vs 0.0525; it completes 14/22 from a looser path. (d) Failures at 12M are not explained by track
margin (median 3.68 cm failed vs 3.76 cm completed), peak speed (4.55 vs 4.58 m/s) or stretch (1.04 both). Only 5
tracks were completed at >= 4 of the last 5 checkpoints; 4 were never completed at any checkpoint (300389,
300565, 300624, 300808), so which tracks pass shifts between checkpoints -- the policy sits on the edge of many
tracks. Not yet known: the 16M result, seed variance (one seed), and anything at lambda > 0.

**v4 headline (2026-09-25): the final 16M checkpoint on the 22 `val` tracks, lambda=0 -- 13/22 completions,
68/88 gates, mean RMSE 0.1867, median max deviation 0.365 m. Against the pre-registered bar (>= 11/22 viable):
VIABLE.** Same flights, same conditions as the baselines: M1 22/22, 88/88, 0.0525, 0.085 m; open-space 2/22;
v3 3/22. Full curve on `val` (completions / gates): 2M 0/0, 4M 3/43, 6M 10/64, 8M 11/65, 10M 10/67, 12M 14/74,
14M 16/77, 16M 13/68. The 12M row reproduced exactly (14/22, 74/88) when re-flown, so the flights are
deterministic. Reading, with caveats.
(a) The bar was met at the pre-registered checkpoint, not at the best one: 14M scored 16/22 and is NOT the
headline -- choosing it would turn `val` into a selection set. 16M < 14M is inside noise (binomial sd ~2.3 at
n=22); from 6M on the curve sits in a 10-16 band, mean of the last three checkpoints 14.3/22 (65 %). It
plateaued; it did not keep rising.
(b) "Viable" means what the bar meant -- roughly 60 % of M1 -- not competitive. It completes 13/22 from a looser
path: RMSE 0.187 vs 0.0525, median max deviation 0.365 vs 0.085 m. Of the 9 tracks the final misses
    (300253, 300267, 300389, 300522, 300565, 300624, 300637, 300808, 301381), 5 (300253, 300267, 300522, 300637, 301381)
    it completed at 12M or 14M and 4 (300389, 300565, 300624, 300808) it never completed at any checkpoint, so which tracks pass is unstable; 300253 fails before gate 1 at 16M and completes at 12M and 14M.
(c) One seed. The bar says a pass is evidence the recipe CAN be viable, not that it is consistent (Lesson 7's
`racing_s1` is the reminder); nothing here says how much a second seed would vary.
(d) lambda=0 only. The study's question is what happens as lambda grows, and nothing above touches lambda > 0.
M1 is already at 22/22 at lambda=0, so any crossover needs the MPC family to degrade faster than RL with
lambda; under the study's own rule, pairs are compared only where both are viable at lambda=0, i.e. on the
~13 of 22 tracks this seed completes.
(e) Not done, and needing the user: the recipe here is the matched-disturbance ("robust") variant only --
`GateAwareTrackingEnv` subclasses `RobustTrackingEnv` -- so the contrast group (vendored, unmatched disturbance
ranges) has no gate-aware version yet; the 6-seed design (3 robust + 3 contrast; robust seed 0 exists, so five runs remain) was estimated at ~4.5 h per run, ~27 h
    in total and ~14 h at two concurrent -- measured later: 5.70 h per run, ~28 h for the five, and pairing saves nothing (see the time correction below); the 10 untouched study tracks are still unflown by design, to be flown
only once the recipe is frozen; and the MPC-family freeze at Lesson 8 parameters remains un-agreed, though M1
completing 22/22 unseen val tracks is the strongest evidence for it so far.

**lambda > 0 preview (2026-09-25; development information on the 22 `val` tracks, NOT a study result).**
`preview_lambda.py`: the v4 final checkpoint, M1 and M1+L1, six conditions, lambda in {0.25, 0.5, 0.75, 1.0} of
each frozen ceiling, one seed for the deterministic conditions and three for wind_gust / lighthouse / combined,
lambda=0 flown once; 3,234 laps, full tables in `results/eval/val_lambda/summary.txt`. Two views: RAW
(completed laps / laps over all 22 tracks) and RETENTION (only tracks that member completed at lambda=0, since v4 is
viable on 13 of 22). Raw, v4 / M1 / M1+L1 by lambda (0.25 | 0.5 | 0.75 | 1.0), percent:
    wind_const   v4 64 59 45 23     M1 95 36 9 9        M1+L1 95 100 100 82
    payload      v4 68 59 64 59     M1 100 100 64 27    M1+L1 100 91 95 55
    wind_gust    v4 58 52 36 15     M1 94 74 41 21      M1+L1 97 94 76 44
    lighthouse   v4 71 71 48 26     M1 97 77 35 8       M1+L1 86 50 18 15
    mass_mult    v4 59 64 59 59     M1 95 95 82 64      M1+L1 100 86 73 59
    combined     v4 61 44 8 0       M1 86 8 0 0         M1+L1 85 14 0 0
Retention, v4 vs M1: wind_const 100/100/77/38 vs 95/36/9/9; payload 100/85/85/85 vs 100/100/64/27; wind_gust
92/77/54/21 vs 94/74/41/21; lighthouse 100/95/59/31 vs 97/77/35/8; mass_mult 100/92/85/85 vs 95/95/82/64;
combined 92/69/13/0 vs 86/8/0/0.
Reading, with caveats. (1) The ordering is condition-specific, which is what the research question asks
about: v4 overtakes M1 in wind_const (raw from lambda=0.5), payload (a tie at 0.75, ahead at 1.0), lighthouse (raw from 0.75,
retention from 0.5) and combined (0.5), and never in wind_gust or (raw, within the grid) mass_mult. (2) The
strongest MPC member matters: M1+L1 beats v4 everywhere in wind_const and wind_gust (steady and low-frequency
additive forces are what L1 is built for), ties it at payload lambda=1 (55 vs 59) and mass_mult lambda=1 (59
vs 59), and loses to it only where sensing noise or the combined condition dominates (lighthouse from 0.5,
combined at 0.5). (3) Everything collapses at high lambda in `combined` (v4 8 % at 0.75, 0 at 1.0). (4) v4 is
nearly flat in payload and mass_mult (59-68 %): there its failures are the precision failures it already has at
lambda=0, not disturbance failures.
THE CONFOUND THIS PREVIEW CANNOT REMOVE: v4 was trained with the disturbance ranges matched to 0.8x these
same frozen ceilings, and the ceilings were calibrated on the MPC family only. Any RL advantage above may be
matched training rather than learned control; separating them is what the contrast group (vendored, unmatched
ranges) is for, and it has no gate-aware version yet. Also: one RL seed; M1 and M1+L1 only (not ESO, mass,
mppi_l1); n=22 tracks gives roughly +-10 points on a raw rate; 3 seeds per stochastic cell are not independent
of the track.
An oddity, checked: v4 completes MORE tracks under lighthouse at lambda=0.25/0.5 (71 %) than at lambda=0 (59 %).
Hypothesis was the train/eval sensor mismatch (training always runs through a sensor with a 1-step delay; lambda=0
hands it a clean, zero-delay stream). Tested with the `latency_only` condition (the delay, no noise), v4 and M1 on
all 22 tracks: v4 14/22 (gates 69/88, RMSE 0.1863) vs 13/22 (68/88, 0.1867) at lambda=0 -- the delay explains
essentially none of it; M1 21/22 vs 22/22. Unexplained; consistent with v4 sitting on the edge of many tracks
(9 fail at lambda=0 but 8 of those 27 laps pass at lighthouse 0.25, while the 13 it completes keep 39/39), so any
perturbation flips edge tracks. Consequence: v4's lambda=0 count of 13/22 is a noisy point estimate of a policy
that completes roughly 59-71 % of these tracks, not a sharp 59 %.

**Tracking error under disturbance (2026-09-25, checking the user's claim that RL will always track worse than MPC).**
The lambda-preview files record `rmse_3d` for every lap, so the claim can be tested directly: `preview_lambda.py
--paired-rmse v4` compares v4 with each MPC member over the laps BOTH completed on the same (track, seed); table in
`results/eval/val_lambda/paired_rmse.txt`. Failed laps are excluded because their RMSE is censored, which selects
survivors: at high lambda the pairs left are the easy laps and n is small, so cells with n < 5 are anecdote.
Result: v4 tracks worse than M1 in every cell with n >= 5 (the share of pairs where v4 is tighter is 0.00 in 21 of
the 23 cells with data; the two exceptions have n = 1 and n = 4) and worse than M1+L1 (by median) in every cell with
n >= 5. But the two families respond to disturbance in opposite ways. v4's RMSE is about flat in lambda -- payload
0.190 / 0.188 / 0.181 at lambda 0.25 / 0.5 / 0.75, mass_mult 0.184-0.194, lighthouse 0.183-0.190 through 0.75 -- and
rises only modestly in wind_const (0.196 / 0.210 / 0.255 at lambda 0 / 0.25 / 0.5, n = 13, 14, 6) and wind_gust
(0.207 / 0.221 / 0.234 at 0.25 / 0.5 / 0.75, n = 38, 27, 8). M1's RMSE grows where it has no disturbance state:
wind_const 0.052 / 0.078 / 0.130 over the same three lambdas, payload 0.072 / 0.101 / 0.128 (n = 15, 13, 8),
mass_mult 0.052 -> 0.077 across the grid. M1+L1 stays tight (0.048-0.152 in cells with n >= 5, the top of that range
being wind_gust at lambda 1.0). So the gap to M1 narrows with lambda (payload: 2.6x at 0.25 -> 1.4x at 0.75, n = 8; the
n = 2 cell at 1.0 is 0.175 vs 0.164) without crossing on the data in hand, and the completion crossovers in the preview are
NOT tracking crossovers: v4 completes more laps at high lambda because its loose tracking does not degrade, not because
it tracks better. This supports the user's picture for this recipe -- RL trades precision for insensitivity -- with two
qualifications: the RL observation carries an L1 disturbance estimate too (limitation 14), so it is not non-adaptive,
and "always" is a statement about this recipe (uniform tracking reward, no precision shaping, 16M steps), not something
these data show for RL in general.


**Recipe saved; contrast group built (2026-09-25).** User: "this recipe is worth saving for now. we can only judge
if it's robust or overfitted with the contrast group." Two things, and one thing deliberately NOT done.
* SAVED, provisionally: `saved/gate_aware_v4/` (30 files, 2.8 MB, not git-ignored) -- copies of the code exactly
  as it trained (`gate_aware_env.py`, `train_gate_aware.py`, `robust_env.py`, `contact.py`, `driver.py`, ...,
  SHA-256 in `recipe.json`), every checkpoint (2M-16M) and the final (sha256[:16] d97b07ebfc6e33bf), the pool
  seeds (3 dev + 111 train, 22 val), both manifests, the run metadata, the val results and the lambda-preview
  tables. `recipe.json` states it is provisional: robust-vs-overfit is undecided until the contrast group exists.
  The working tree was dirty at save time (the whole Lesson 9 build is uncommitted); the copy in `code/` is the
  authoritative one, and nothing here has been committed to git.
* NOT done: the 10 untouched study tracks were not flown. "Save for now" is not a freeze of the study recipe,
  and they should be flown once, with robust and contrast evaluated together.
* BUILT: `GateAwareContrastEnv` = the same gate-aware recipe on `ContrastTrackingEnv` (vendored +-3.5 m/s^2 force
  box, vendored Lighthouse noise SCALE U(0, 1.5) uncoupled from the update rate, no mass channel). To make the
  two groups share the gate logic by construction rather than by a second copy of `step()`, that logic moved into
  `GateAwareMixin`; `GateAwareTrackingEnv = GateAwareMixin + RobustTrackingEnv` (the saved recipe) and
  `GateAwareContrastEnv = GateAwareMixin + ContrastTrackingEnv`. The refactor is tested against the SAVED code,
  loaded with `__file__` pointed at the live directory so its track-pool discovery sees the same `tracks/` (loading
  it from `saved/` would silently give it an empty train pool): bit-identical obs/reward/terminated/truncated and
  stats over 500 random-action steps (23 episode ends, all but 2 floor/divergence) AND over 1000 steps driven by the
  saved v4 policy itself (16 episode ends: 13 contacts, 1 miss, 2 completed laps, 28 gates passed) -- the second
  is the one that exercises the branches that moved. `train_gate_aware.py` now takes `--group {robust,contrast}`
  and seeds 0/1/2 (seed 0 of `robust` is the saved run); it refuses `--dr-lam-max` for the contrast group rather
  than ignore it. 30 tests in `test_gate_aware.py` (full lesson9 suite: 85 across 10 files, all pass).
* TIME CORRECTION: v4 took **5.70 h (5 h 42 min)** for 16M steps, not the ~4.3 h estimated from ~15 ms per vector step:
  run start 18:26:35 to the final checkpoint 00:08:34 by file times, 5.697 h by tensorboard, no pauses (largest gap
  between logged rollouts 18 s); the first 2M steps took 0.44 h and every later 2M a steady 0.71-0.82 h (cause of the
  early speed-up not investigated). The remaining runs of the 6-seed design (robust seeds 1, 2 and contrast seeds 0, 1, 2) are
  therefore ~28 h of machine time; two at once takes about twice as long each (Lesson 7), so pairing saves nothing.
HOW THE CONTRAST COMPARISON WILL BE READ, written before any contrast result exists. The question is whether v4's
lambda > 0 robustness is generic (it would survive unmatched training ranges) or an exam effect (trained on 0.8x
the very ceilings the study measures). Compare on `val`, robust vs contrast, at lambda = 0 first: the contrast recipe
must itself be viable (>= 11/22, the same bar). Then, per condition at lambda >= 0.5, RETENTION (tracks completed
at lambda=0 -> still completed): if robust exceeds contrast by >= 20 points (about 2 sd of a raw rate at n=22) in
at least 3 of the 6 conditions, the matched ranges buy something and the RL line must be labelled "trained on the
study's own ceilings"; if contrast is within 20 points almost everywhere, matching bought little and robustness
looks generic. Where the gap should appear if it is an exam effect: beyond the vendored coverage (force box
+-3.5 m/s^2, z halved to +-1.75, no mass channel) -- payload and mass_mult first, wind_const at lambda near 1
(1.5 x 2.54 = 3.8 m/s^2). A gap only there is the exam signature; a gap everywhere would be something else. A
single robust/contrast pair (one seed each) is a screen: differences inside the seed-to-seed spread cannot be
read, and Lesson 7's `racing_s1` says that spread can be large.

**Predictions recorded before the contrast run (2026-09-25).** The user's: matched training is not what makes RL
"robust" so much as a precision-for-invariance trade, and the contrast group "may perform well at lambda = 0 but will
crash very soon after" -- viable at lambda = 0 (possibly tighter than v4, since a narrower randomization box is the less
conservative one) and then collapsing at the first lambda steps. Mine, which differs: contrast should hold up wherever
the vendored ranges still cover the condition -- wind_const to about lambda 0.9 (1.5 x 2.54 = 3.8 m/s^2 against a
+-3.5 box), lighthouse (its noise scale reaches 1.5) -- and open a gap first where they do not: mass_mult, which has no
training channel at all, and payload beyond about lambda 0.4 (the vendored z range is halved to +-1.75 m/s^2 against
a 4.5 m/s^2 ceiling). What separates the two: if contrast's retention falls below 50 % at lambda = 0.25 in wind_const or
lighthouse -- inside its training coverage -- the user's "very soon" is right and range coverage is not the explanation;
the place to look is the sensor pipeline, because the vendored Lighthouse batch draws its noise scale independently of
the update interval, so lambda = 0's clean 100 Hz stream is outside contrast's training distribution in a way it is not
for v4 (whose interval shrinks toward one step as lambda -> 0). If contrast holds there and fails in mass_mult and
payload, coverage explains it. Weak prior evidence, from the superseded open-space recipe (one seed, three tracks,
wind_const only): contrast_s0 degraded with lambda on track 93 (max deviation 60 -> 90 -> 136 cm at lambda 0 / 0.3 /
0.6, inside its box) and barely moved on tracks 4 and 387, while robust_s0 stayed flat -- one track in the user's
direction, two not. Both predictions are scored against the rule already stated above (contrast viable at lambda = 0
first, then retention at lambda >= 0.5); the user's adds a stricter test at lambda = 0.25.

**Contrast run, mid-run look (2026-09-25, 7.2M of 16M steps, 1.13 h elapsed; NOT a result).** The user asked for a quick
read of the learning curve. Against v4 at the same step counts (pooled `gate/*` over 0.5M-step windows,
`gate_aware_env.py` stats): the contrast run sat on a plateau for its first ~4.5M steps -- gates per episode 0.00, full-lap
rate 0, reward flat at 61-65, episode length 147-154 steps (1.5 s), 94-95 % of episodes ending on the floor/divergence rule,
`explained_variance` 1.00 -- while v4 had left the same plateau by 0.5-1.5M (contact 29 % and 0.02 gates per episode at
1-1.5M; 1.0 gates per episode at 3-3.5M). The signature of that plateau is a policy whose reference simply outruns it: err
exceeds 2 m at about 1.5 s, and the value function fits a near-constant return perfectly. It is now leaving it: from ~5M,
floor/divergence falls 0.95 -> 0.56 (7.0-7.2M), contact rises 0 -> 0.22, misses 0.22, episode length 154 -> 178, reward
64 -> 84, gates per episode 0.05, KL 0.005 -> 0.011 and clip fraction 0.035 -> 0.116 (both still in a normal range). That
is roughly a 5M-step lag against v4 (v4 was at this stage at ~1.5-2M). One likely cause, supported by the code but not
tested: the vendored `LighthouseSensorBatch` draws its refresh interval from N(34 Hz, 18 Hz) clipped to 8-100 Hz
INDEPENDENTLY of the per-episode noise scale (`sensors.py`: `next_update = t + 1/hz`), so every contrast episode
has held, stale position at ~34 Hz on average and there is no easy-sensing episode to bootstrap from, whereas v4's coupled
sensor gives near-continuous updates at small lambda. One seed each, so seed luck is not excluded. Consequences to keep in
view: (1) the contrast group differs from the robust group in more than the matched force/mass ranges -- its sensing is
harder in EVERY episode -- which is what the vendored recipe is, but it means a weaker contrast result may reflect the
curriculum rather than the range coverage; (2) if the run is not viable at 16M (bar: >= 11/22 on val), lambda > 0
retention cannot be read, and the options are to report that, train longer with the extension disclosed, or add a second
contrast variant that keeps the vendored force box but couples the sensor (isolating the force range); (3) extrapolating
v4's pace after take-off (about 4.5M steps from 0.06 to 2.0 gates per episode) puts contrast near v4's plateau level at
~11-12M, leaving ~4M steps there -- an extrapolation, not a measurement. Its throughput so far is ~6.4M steps/h against
v4's ~2.8M steps/h because short episodes are cheap; it will slow as episodes lengthen (v4 took 0.44 h for its first 2M
and 0.71-0.82 h for each later 2M), so 2-3.5 more hours is a guess.

**Contrast run at 13.7M of 16M steps (2026-09-25, 2.78 h): it is not learning fast enough, and will not reach the bar.**
The user's read ("still not increasing fast enough") is right. Aligning the two runs at their own take-off (first 1M window
with floor/divergence < 0.75: contrast at 6M, v4 at 1M, a 5M-step lag), the contrast run 7M steps after take-off is at 0.13
gates per episode, full-lap rate 0.000, contact 0.49, floor/divergence 0.29; v4 at the same age was at 2.18, 0.284, 0.659,
0.011. Its gain has been ~0.02 gates per episode per 1M steps (0.02 / 0.04 / 0.06 / 0.09 / 0.11 / 0.11 / 0.13 / 0.13 at
0-7M after take-off) against v4's ~0.5 per 1M during its rise, roughly 20x slower, with ~2.3M steps left. A policy that
averages ~0.15 gates per episode and has never completed a lap in training cannot plausibly complete 11 of 22 validation
tracks, so the pre-registered viability bar (>= 11/22 at lambda = 0) will almost surely be missed; the final checkpoint will
still be flown on `val` to record it. Extending the run is not an answer: at that slope another 16M steps would reach
~0.4 gates per episode. What a miss does and does not say: the vendored recipe cannot learn the gate-aware task in this
budget on this seed -- informative, but it means the robust-vs-contrast lambda > 0 comparison cannot be read from it.
**Test of the sensing explanation, built and not yet run:** `GateAwareForceContrastEnv` (`--group contrast_force`) keeps
the vendored +-3.5 m/s^2 force box (z halved) and no mass channel but uses v4's own lambda-coupled Lighthouse sensor, so it
differs from `robust` in the force and mass channels ONLY; it holds the sensing curriculum equal to v4's, and it can say
nothing about the lighthouse channel (identical training by construction). Decision rule, written before any result: v4
left the plateau within ~1M steps (floor/divergence 0.79 at 0.5-1.0M, 0.33 at 1.0-1.5M) and had 0.10 / 0.49 / 1.01 gates per
episode at 2.0-2.5M / 2.5-3.0M / 3.0-3.5M. If the force-only contrast shows that take-off signature (floor/divergence below
0.5 and gates per episode above 0.3 by ~3M steps) the sensing curriculum is the cause of the fully vendored run's stall; if it
also stalls, the cause is elsewhere (the force or mass ranges, or seed luck) and the fully vendored result stands as
measured. Practical plan: launch the full 16M run and read `gate/*` at ~2-3M steps (~0.7-0.9 h, v4's pace), killing it if it
stalls, rather than spend a separate diagnostic run. The fully vendored contrast, if it finishes non-viable, is reported as
such and is not retried.

**Correction of framing (2026-09-25, the user's objection): `contrast_force` is an ablation, not a control group.** The question
the control group exists to answer is whether the RL policy is overfitted to the specific task at hand -- trained on the study's
own conditions -- and so skews the comparison. That is broader than range matching on two channels, and the force-only variant
holds the sensing channel identical to v4's, so it cannot answer it; calling it "the contrast group" or a replacement for it was
wrong. What it can do is narrower: say whether matching the FORCE and MASS ranges buys anything, given equal sensing
training. The fully vendored contrast remains the design that varies every channel, and its failure to learn (confounded with
its sensing curriculum, above) means the overfitting question is currently unanswered, not answered "no". Ways to answer it, in
increasing cost, each varying only the match between what was trained on and what is tested: (1) HELD-OUT TEST CONDITIONS, no
training: fly the saved v4 and the MPC family on disturbances v4 never saw and the calibration never targeted -- upward force
beyond its +1.8 m/s^2 training limit, a LIGHTER drone (v4 trained heavier-only), a force that steps mid-episode (training draws
one constant force per episode), a short Lighthouse blackout -- so neither family was developed against them, and read whether
v4's relative standing survives; (2) HELD-OUT GEOMETRY: tracks from outside the accepted family (a looser crossing-angle cap),
since the pool is drawn from the same generator and filters as the study tracks (limitation 11); (3) a viable fully unmatched
contrast run, which needs a learnability fix that itself changes the curriculum. Nothing here has been run; (1) and (2) need
new conditions or a new track pool, not new training.

**Fully vendored contrast, final (2026-09-25): NOT VIABLE.** 16.0M steps in 3.36 h (short episodes are cheap). Last 40 rollouts
in training: 0.122 gates per episode, full-lap rate 0.000, contact 0.507, floor/divergence 0.292. Flown at lambda = 0 on the 22
`val` tracks: **0/22 completions, 2/88 gates, mean RMSE 0.543, median max deviation 0.548 m** (bar: >= 11/22). One seed. It is
reported as measured -- the vendored recipe did not learn the gate-aware task in 16M steps, most likely because of its uncoupled
Lighthouse sensor (unproven) -- and is not retried; it cannot serve as the control group, so the "trained on the exam" question is
open. **Proposed replacement, not built: a range-scaled (dose-response) control.** The same gate-aware recipe and curriculum as v4
with ONE factor changed, the size of the disturbance training range across all three channels at once (force box, Lighthouse lambda
maximum, mass lambda maximum), e.g. 0.5x of v4's (lambda maximum 0.4 instead of 0.8, force box halved). It varies only how much of
the study's range is trained on, has no learnability confound (same structure and sensing as v4), and asks the question the contrast
group was created for: does the policy's robustness, and therefore any crossover, sit at its own training edge? Reading, before any
result: compare completion and retention against lambda per condition for v4 (edge 0.8) and the half-range policy (edge 0.4). If the
half-range curve falls away beyond ~0.4 where v4's does not, robustness is tied to the training edge and the RL line must be labelled
"trained on the study's own ceilings"; if the two overlap within noise (n = 22 tracks, roughly +-10 points) the crossover is not a
training-edge artifact. One seed each is a screen. It costs one run of about 5.7 h. It does NOT address task overfitting in general
(same tracks family, same objective); held-out conditions and geometry (above) are the tests for that.






## 6. Trials and grid

- Stochastic conditions (gust, Lighthouse, combined): 25 seeds per (track, lambda).
- Deterministic conditions (`wind_const`, `payload`, `mass_mult`): first repeat one cell to check determinism;
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

- **Now:** one folder per track, six charts (one per condition, including `combined`): completion against lambda, one line
  per member, warm colours for MPC and cool for RL, a marker at lambda = 0.8.
- **Deferred:** the crossover statistic. Pairwise was rejected. Candidate: "leading family" (best member
  per family per track and lambda, margin 0.2, shown as the share of tracks led by each family against
  lambda). **Freeze the statistic before the study sweep** (Lesson 4 section 1 pre-registration), not
  after seeing study data.

## 9. Compute (measured 2026-09-22 with `driver.py`, one study track, M1 lap of 4.1 s, ground clock)

One M1 lap takes 8-12 s (mean solve 19-24 ms), M1+ESO / M1+L1 / M1+mass about the same, `mppi_l1` 2-5 s, a
policy 2-3 s. That is about 40 s per (track, lambda, seed) cell over the 5 MPC-family members and 3 policies
(measured with 4 members before M1+mass was added; a 5th ipopt-solving member adds roughly another 8-12 s per
cell), less at high lambda where laps stop at the first contact. Coarse grid: about 175-290 core-hours, about
a day on 8-9 workers.
Two settings decide whether that holds, both measured:
  * `JAX_PLATFORMS=cpu` and single-thread limits (`OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=1`,
    `XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"`). With them, 4 concurrent
    workers run at 11-13 s per lap, the same as one alone. Without them the same 4 workers took 154-192 s per
    lap (15x slower), and the processes hung at exit in the WSL GPU driver (JAX probes CUDA);
  * memory: a worker peaks at about 1.2 GB, so about 8-9 workers in 15 GB (not 16).
Training (measured, container clock): open-space robust/contrast, 8M steps at 100 Hz, about 65 minutes alone (`robust_s0`: 65.3 min);
gate-aware v3 (3-track pool, 8M steps) 122 minutes; gate-aware v4 (114-track pool, 16M steps) **5.70 h** (5 h 42 min: run
start 18:26:35 to the final checkpoint 00:08:34 by file times, 5.697 h by tensorboard; no pauses, largest gap between
logged rollouts 18 s), of which the first 2M steps took 0.44 h and every later 2M a steady 0.71-0.82 h. Two concurrent runs
each take about twice as long (Lesson 7), so pairing saves nothing. A ground plan plans in 0.4 s; a candidate track cost
about 1.3 s in the first single-process measurement and about 0.25 s per candidate per worker in `gen_pool.py` (8,000
candidates in about 4 minutes on 8 workers; container log times 18:07-18:11). Evaluation: `eval_pool.py` (M1 plus two RL
members on the 22 val tracks) took 1-2 minutes on 8 workers; the lambda preview (3,234 laps, 3 members) took 61 minutes
on 8 workers (first to last lap write).

## 10. Limitations to state in Lesson 9

1. Gate-only geometry with a fixed start, heights and gate order.
2. The corrected MPC model is a fit to this same simulator; its "accurate model" favours MPC.
3. ~~The RL family is trained on this study's range, and with no contrast group a crossover cannot be
   attributed to the training edge versus something intrinsic~~ -- addressed 2026-09-22: a 3-seed contrast
   group (section 5b) trains the same recipe on the vendored, unmatched disturbance ranges, so the study
   can check whether the robust family's crossover location moves relative to it. The open-space contrast seeds were trained but that recipe is superseded; the gate-aware contrast group is built and not yet trained (section 5c), so this is still a designed answer, not a measured one.
4. ~~No parametric mass mismatch~~ -- superseded 2026-09-22: `mass_mult` (section 4) and M1+mass (section 5)
   were added at the user's request, specifically because Lesson 8 section 5 found this, not an additive
   force, is what defeats MPC in the race. The robust group's training includes matched mass domain randomization (section 5a). What remains true: mass is the only condition excluded from `combined`.
5. Gust time structure is not in training. The 50 Hz/100 Hz training/harness mismatch and its specific
   consequences (the L1 estimator's bandwidth, the preview window) are detailed in limitation 14.
6. Ceilings are outcome-calibrated on dev tracks.
7. Disturbances enter through crazy_track's force model; contacts are checked after the fact from the
   flown path, not by the physics engine.
8. The Lighthouse model was validated up to about 3 m/s and is applied at about 4.5 m/s.
9. The MPC solves in about 32 ms per 10 ms step: not real-time.
10. Three RL seeds per group (only seed 0 of the gate-aware robust group exists so far) and 15 study tracks, of which the
    RL-vs-MPC headline uses only the 10 that no RL decision touched (limitation 15); the MPC family alone can use all 15.
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
12. **Plan aggressiveness sets where the model-based family collapses.** The plans were time-optimal at 0.85 of the
    thrust limit when this was measured (the study now uses 0.80; see the last sentences and section 4), so a constant force of about 4 m/s^2 (the dev plans' mean headroom is 3.8 m/s^2, from peak
    demands of 13.0-15.6 against 18.4) cannot be flown at the plan's fastest moments by any tracker, and the
    tube leaves only 3.5-5 cm of frame margin, so 72 % of failed calibration laps are frame contacts (26 %
    missed gates, 2 % divergence) and the median RMSE of completed laps (0.086 m) is only 3.5 cm below that of
    contact laps (0.121 m). Measured on the dev tracks by flying the same paths slower (more headroom, the
    frame margin unchanged): at 1.3x the payload collapse moved out from 3.4-4.5 to beyond 4.5 m/s^2 (all three
    MPC variants 100 % at 4.5) and ESO/L1's wind collapse from 3.8-5.1 to 5.1; plain M1 under constant wind did
    NOT improve (RMSE 0.22 -> 0.24), because a nominal MPC has no disturbance state and holds a steady offset.
    The ceilings, and any crossover, are therefore properties of this plan family at this aggressiveness, not
    of the controllers alone (revisited, unchanged, after the thrust-frac 0.85 -> 0.80 restart: section 4's
    recalibration note). A lower thrust fraction would raise them.
13. **Independent per-episode training channels likely undertrain `combined`'s joint worst case** (section
    5a). Force, Lighthouse and mass severity are drawn from three separate RNG streams each episode, which
    is correct for the five solo conditions (each gets full-range exposure regardless of the others) but
    means the co-occurrence of two or three channels being simultaneously severe -- exactly what `combined`
    tests at evaluation, a single coupled point in the joint space -- is visited far less often in training
    than any one channel's own extreme (illustrative arithmetic: two independent top-quartile channels
    co-occur in about 6 % of episodes, not 25 %). This was a considered trade-off, not an oversight: a
    coupled/correlated training scheme was rejected because it would confound any `combined`-specific
    result between "training was coupled" and whatever else might explain it. Expect the RL family's
    `combined` line to be a weaker test of its own architecture's robustness than its solo lines are, and
    read any RL weakness specifically on `combined` with this in mind before attributing it to the
    controller.
14. **The MPC family and the RL family do not receive equal inputs, and the gap is now partly, but not fully,
    closed.** Both legitimately see the exact future reference (not a leak: any tracker with the course map
    would have this) but over different horizons: the M1 family and `mppi_l1` now both look 0.8 s ahead
    (section 5's mppi_l1 reconfiguration), while the vendored RL observation looks only 0.6 s ahead
    (`WINDOW=10` points at `WINDOW_DT=0.06` s) -- this was equalised to 0.8 s in the RL recipe (section 5a; built 2026-09-22 and used in every RL run since), which closes that gap. Separately, the deployed RL policy's own L1
    estimator (always active) is shaped by training at 50 Hz but integrates its adaptation law at 100 Hz in
    the harness, changing its effective bandwidth relative to what the policy learned to expect -- training at `freq=100` (section 5a, built) removes this specific mismatch, at roughly double the wall-clock cost for the same simulated experience unless `--timesteps` is also doubled (it was: 8M). Also worth stating plainly:
    RL's L1 channel is architecturally always on, while the MPC family's estimator is a per-member choice
    (M1 has none) -- not unfair given "every member its own line," but a reader comparing "RL vs plain M1"
    and "RL vs M1+L1" is answering two different questions, not one.

    **Control authority is unequal too, and this was never examined until the user asked (2026-09-23).**
    Every "equal inputs" item above is about what each controller can SEE; none is about what each is
    ALLOWED TO DO. `datt_env.py`'s `_denorm_action` (line 301, used at TRAINING time, unchanged by this
    lesson) clips physical roll/pitch to `RPY_MAX * 0.7` = 0.70 rad; `robust_policy.py` and the vendored
    `DATTPolicyController` both copy this exact formula at eval time, so training and eval agree -- this is
    not an eval-side throttle on a policy trained for more, the network has never been rewarded for wanting
    beyond 0.70 rad. M1 (and the rest of the MPC family) clips at the FULL `RPY_MAX` = 1.0 rad
    (`crazy_track/controllers/utils.py`). `git log` on `datt_env.py` shows the 0.7 factor present since the
    very first commit that added the racing task (`4ae7de8`) -- a vendored, upstream design choice, present
    since Lesson 1, with no stated rationale anywhere in the code or the lessons, and never revisited since.
    Measured impact (the actuator-saturation check above): both RL groups reach EXACTLY 0.700 rad max pitch on every one of the 3 tracks checked, with 1.3-25.1% of steps within 2% of that ceiling (median 9.5%), while M1 reaches its own (larger) 1.000 rad ceiling on all three (pitch on tracks 4 and 93, roll on 387; 5.4-8.4% of steps). This does not overturn the reference-distribution conclusion (a policy that never practiced
    gate-threading would still fail with more authority), but it means the comparison is not purely
    architecture-vs-architecture: one side has less control authority available by construction. Closing it
    is not a config change -- the network was trained around the 0.7 scale, so it requires retraining with a
    wider (candidate: matched to M1's 1.0 rad) cap, a new open decision, not yet made.

    The measurements that followed this limitation -- the two control-authority tests, the locality finding,
    the gate-aware viability screens v1-v4, the validation baselines, the lambda > 0 preview, the saved recipe
    and the build of the gate-aware contrast group -- are in section 5c (moved there 2026-09-25; they had been
    filed under this limitation). What this limitation itself states is above.

15. **The RL and MPC families were developed under different regimes, and the RL family's development touched
    study tracks (recorded 2026-09-24, at the user's request).** Two separate points.
    (a) *Development-time asymmetry is inherent.* MPC has no training phase and re-solves from an explicit model;
    the RL family needed several redesigns (hyperparameter rescaling, a control-authority screen, three
    gate-aware versions), and the gate-aware objective now encodes the evaluation criterion -- contact and a
    missed gate end the episode -- while M1 minimises tracking error only. That is "trained on the exam" at the
    level of the task objective, the same concern that motivated the contrast group for the disturbance ranges.
    It cannot be removed, only stated: the RL family's line means "RL trained on the task against a contact-aware
    objective", not "learned control" in general, and a result in which it still trails M1 at lambda = 0 (so no
    crossover appears) is a valid outcome of this study, not a failure of it. The principle that keeps the
    comparison fair is process symmetry, not identical treatment: each family is developed on a set disjoint
    from what it is evaluated on, then frozen. Deployment-time inputs (reference, measurements, preview window,
    control rate) are the ones held equal -- see limitation 14.
    (b) *A leak on the RL side, found and fixed by rule.* The 6-track RL screen is `level2` plus study tracks 4,
    25, 93, 387 and 504, and every RL design decision since round 1 was judged on it (hyperparameters, control
    authority, gate-aware v1-v3). The MPC family was calibrated on dev tracks and never selected on study
    tracks, so any final RL number on those five tracks is optimistic in a way M1's is not. Reporting rule:
    the final RL-vs-MPC numbers use only the 10 study tracks no RL decision has touched (747, 757, 834, 837,
    965, 989, 1089, 1145, 1250, 1318); tracks 4/25/93/387/504 and `level2` are the *selection set*, reported
    separately and labelled as such, never pooled into the headline. The dev tracks (100023, 100082, 100092)
    are shared development material for both families -- they set the MPC disturbance ceilings and are the
    gate-aware RL training tracks -- and are likewise excluded from the reported set.
    Update 2026-09-25: from v4, RL design decisions are judged on the `val` pool (22 tracks, seeds 300000-301999,
    generated by `gen_pool.py`, never trained on, not study tracks), so no further RL decision needs the study
    tracks. The selection set stays reported separately for continuity with v1-v3.
    Still open, not part of this limitation: whether the MPC family should be frozen at its Lesson 8 parameters
    (measured recommendation in section 5, before 5a; now also supported by M1 completing 22/22 val tracks with
    no re-tuning; still not agreed by the user).

## 11. How the initial input became this framework

| proposed | now |
|---|---|
| disturbance on/off | lambda as a fraction of range; maxima frozen by the ceiling rule |
| 50 trials per condition, few tracks | 25 seeds x 15 study tracks (10 for the RL-vs-MPC headline, limitation 15) |
| "all controllers complete at least one lap" | structural filter, a stretch to a common speed cap, and a lambda = 0 viability report (no selection on outcomes) |
| random gates, then TOGT with tube | lsy generator saved as a track library, then ground-start tube plans |
| poles kept (sphere-clearance idea) | poles removed; completion is box-contact-free by lsy's own box model |
| RMSE, completion, lap time | same, RMSE over the racing segment only |
| Lesson 7's controllers | Lesson 8 corrected MPC family vs the gate-aware RL groups (robust and contrast, 3 seeds each planned) |
| family means / pairwise crossover | every member its own line; statistic deferred |

Corrections along the way: the thrust-headroom "ceiling" claim was retracted (Lesson 7's policies
finished with roughly a third of the headroom the disturbance needed); the optimistic lap-time guess was
replaced by the measured 24 s.

## 12. Build phases (each ends with a confirmation from the user)

1. Track library (`gen_tracks.py`, `track_lib.py`) -- done; restarted 2026-09-22 at thrust-frac 0.80
2. Contact check (`contact.py`) -- done
3. Lambda knobs (`knobs.py`) -- done; extended 2026-09-22 with `mass_mult` and `mass_scale`
4. Driver and storage (`driver.py`) -- done; extended 2026-09-22 with the mass override, M1+mass, and the
   mppi_l1 0.8 s preview-horizon reconfiguration
5. Dev calibration (`calibrate.py`) -- done; three passes (section 4's history), all 5 ceilings frozen in
   `maxima.json`; `lighthouse` moved from 1.0x to 1.5x on the third pass (M1+mass entering the roster)
6. Robust RL training -- **code written 2026-09-22 (`robust_env.py`, `lighthouse_batch.py`,
   `robust_policy.py`, `train_robust.py`); verified by construction/reset/step (`test_robust_env.py`, 9 tests); then trained in three open-space rounds (section 5a)
   once the user began running them -- the 2026-09-22 "do not start training" instruction was theirs to lift, and it was; superseded as the
   study recipe by 6c.**
   `RobustTrackingEnv(RacingTrackingEnv)`: freq fixed at 100, `_t_offsets` overridden to 0.08 s spacing
   (WINDOW's point count, 10, and therefore the 56-number v5 observation shape, is UNCHANGED -- only the
   vendored `datt_env.py`'s WINDOW_DT=0.06 module constant is bypassed, never edited, because it is shared
   by every other lesson's trained policy); `_sample_perturb` overridden to the study-matched force box
   (x,y +-4.4 m/s^2, z -3.6..+1.8) and piggybacked to also call a new `_sample_mass` (its own RNG stream,
   `seed+2`, mirroring the Lighthouse sensor's `seed+1`); the vendored noise-scale `LighthouseSensorBatch` is
   swapped for `lighthouse_batch.LambdaLighthouseSensorBatch`, which draws a per-episode lambda in U(0, 0.8)
   coupling size errors and the update interval exactly as `knobs.ScaledLighthouse` does at evaluation
   (cross-checked against it directly: vel-noise std 0.0225 vs 0.0224, refresh-event count 1051 vs 1080 over
   4000 steps) -- with one deliberate deviation from the vendored *batch* class: the gyro channel is NOT
   delayed, matching the evaluation-time sensor and `knobs.ScaledLighthouse`, not the vendored batch class's
   own inconsistency between training and evaluation. `robust_policy.py`'s `RobustPolicyController` is a
   separate class from the vendored `DATTPolicyController` (which imports the shared `WINDOW_DT` and would
   silently feed a robust model the wrong window); `driver.py` routes a `robust:<path>` spec to it,
   `datt:<path>` stays on the vendored one. Tests confirm: the three channels are independent (pairwise
   correlation < 0.07 over 500 worlds), each stays within its designed range and is visibly exercised near
   its edges (not just its centre), a per-env episode-end reset resamples mass and Lighthouse too (not only
   the vendored hook's original force channel), and the whole env resets and steps without error.
6b. Contrast-group training -- **code written 2026-09-22 (`contrast_env.py`, `train_contrast.py`); tested
    (`test_contrast_env.py`, 5 tests); open-space contrast seeds 0-2 trained in the three rounds (section 5a), superseded by the
    gate-aware contrast group (6c).** `ContrastTrackingEnv(RacingTrackingEnv)`: the same freq=100 / 0.8 s window fix as the
    robust recipe, and NOTHING else changed -- the vendored +-3.5 m/s^2 force box and the vendored
    noise-scale `LighthouseSensorBatch` (not `LambdaLighthouseSensorBatch`), no mass channel. Tests confirm
    the window/frequency match `RobustTrackingEnv` exactly, the force box matches the vendored
    `PERTURB_ACC_MAX` (not the study-matched range), the vendored sensor class is in use, and no mass
    channel exists (`sim.data.params.mass` never diverges from `default_data`'s). Section 5b.
    **Hardware note (6 seeds total, 3 concurrent-training pairs):** 3 robust + 3 contrast seeds, the user's
    choice specifically so 6 seeds split into 3 pairs use every slot under the 2-concurrent-session limit (5
    seeds would waste one). That estimate (`--timesteps 8000000`, roughly 50-100 minutes per seed, 2.5-5 hours for all six) was for the
    open-space recipe. Measured later: two concurrent runs each take about twice as long, and a gate-aware run takes 5.70 h
    (v4, 16M steps), so the five remaining gate-aware runs are about 28 h of machine time.
6c. Gate-aware RL recipe (section 5c) -- `gate_aware_env.py` (`GateAwareMixin`, `GateAwareTrackingEnv`, `GateAwareContrastEnv`),
    `train_gate_aware.py` (`--group`, seeds 0-2), `gen_pool.py` (train/val pools), `eval_pool.py`, `preview_lambda.py`, the
    snapshot `saved/gate_aware_v4/`. Robust seed 0 (v4) trained and saved: 13/22 on val at lambda = 0 against a bar of 11.
    Robust seeds 1-2 and contrast seeds 0-2: not trained. 30 tests in `test_gate_aware.py`; full suite 85 across 10 files.
7. Study sweep -- not started (depends on 6c: robust seeds 1-2 and contrast seeds 0-2; the RL-vs-MPC headline uses the 10
   untouched study tracks, limitation 15)
8. Charts -- not started
9. `lessons/09-crossover-under-disturbance.md` -- started 2026-09-22: opening, the trackers table, and the
   training section (the two recipes, why the contrast group exists, the exact commands) and the v1-v4 gate-aware
   narrative (2026-09-23 to 09-25) are written; sections on tracks/contacts/disturbances/calibration/results are pending
   behind this file until their phases produce something to report. Indexed in `tasks/racing/README.md`'s
   lesson table, marked in progress.

## 13. Errata (audit of this file, 2026-09-25)

The user asked for the file to be re-checked for errors in earlier entries. Every claim below was checked against the
data or the code; "stated in chat" marks errors I also gave the user in a reply, not only in this file.

Wrong numbers or wording (corrected in place, each marked "corrected 2026-09-25" where it stands):
1. Same-path comparison, section 5a: RL deviation at impact is 2.1-5.5x M1's worst-case deviation and 4.2-12.6x the
   track margin, not "3-10x" and "4-11x" (from the table beside it). Stated in chat.
2. Lambda sweep, section 5a: contrast_s0's max deviation on track 93 was 60 -> 90 -> 136 cm, not 23 -> 90 -> 136 (23 cm
   is another track's value). Stated in chat.
3. Actuator saturation, sections 5a and 14: the share of steps within 2% of the 0.70 rad ceiling is 1.3-25.1% across the six RL
   flights (median 9.5%), not 7.6-25.1% (7.6% is M1's track-4 figure); and M1 reaches its 1.0 rad ceiling on all three tracks
   (pitch on 4 and 93, roll on 387), not "two of three". Stated in chat (the 7.6% figure).
4. Locational finding: 4 of 6 RL cases are clearly diverging into the crash (+0.13 to +0.47 m/s), one is flat (+0.010) and one
   converging (-0.155); "5 of 6 actively diverging" counted the flat one. Stated in chat, in softer wording.
5. Control-authority screen: it was better on 2 tracks and worse on 4 (including the lost completion on 504), not "worse on 3
   and lost one".
6. v3: `gate/full_lap_frac` at 5 / 6 / 7 / 8M was 0.26 / 0.21 / 0.29 / 0.53 (single rollouts); the entry dropped 0.26 and
   mislabelled the rest.
7. v4 headline: of the 9 tracks the final checkpoint misses, 5 it completed at 12M or 14M and 4 it never completed at any
   checkpoint; "mostly ones it completed earlier" overstated. Stated in chat.
8. Lambda preview: payload is a tie at lambda 0.75 (64 % vs 64 %), ahead of M1 only at 1.0; "0.75-1.0" overstated. Stated in chat.
9. M1 on the study screen tracks: max deviation 7-10 cm, not 7-11.
10. v2 "cause": "there is NO penalty for missing a gate" was an overstatement (the tracking term still applies); the
    entry already carried a correction below it, the original sentence now says so too.
Measurements that were wrong or estimated and are now measured:
11. v4 training time: 5.70 h (5 h 42 min), not the ~4.3 h estimated; the 6-seed cost is ~28 h for the five remaining runs
    (not 27 h for six, and not 14 h at two concurrent: pairing saves nothing). Cross-checked three ways (file times, tensorboard,
    checkpoint times); no pauses. Stated in chat.
12. Track-pool generation: ~0.25 s per candidate per worker and ~4 minutes for 8,000 candidates (log times 18:07-18:11), not
    ~0.5 s and ~10 minutes (a miscalculation from a 6-candidate run that included startup). Stated in chat.
13. Step-cost benchmark: the two benchmarks used slightly different robust baselines (4.66 and 4.87 ms); the entry read as if
    the cost moved from 4.87 to 6.24 ms.
14. Test counts: `test_gate_aware.py` has 30 tests (an entry said 33); the full suite is 85 across 10 files (an entry said 79;
    it was correct when written, before the contrast tests).
A misdiagnosis:
15. Section 5a recorded a "7-hour RunLogger clock artifact" for the first RL round. It does not reproduce: the run directory
    name, the file times and the tensorboard events agree (65.3 min for robust_s0), because all use the container's clock
    (UTC). The host is UTC+7, so host-side `ls` times differ from the container's by exactly that offset; comparing across the
    two produced the apparent discrepancy. (Recorded in an earlier session; stated to the user then.)
Stale statements updated: the file's status line; section 2's track and condition counts (six conditions including
`combined`; the train/val pools); section 5's "code written, not yet trained" for both RL groups; the headers of 5a and 5b
("NOT yet run", and a doubled "## 5b" title); section 6 (`mass_mult` is deterministic) and section 8 (six charts); section 9's
"25-50 minutes" training time; limitations 3, 4, 10, 12 and 14 (the 0.8 s window and freq=100 are built, not "PLANNED"; the plans
were at 0.85 when limitation 12 was measured); section 11's table; and section 12's phases 6, 6b, 7 and 9, which still said
"NO TRAINING HAS BEEN LAUNCHED". Structural: about 500 lines of results had been filed inside limitation 14; they are now
section 5c, and limitation 14 keeps only what it states plus a pointer.
Checked and found correct: the lambda-preview raw and retention tables against `results/eval/val_lambda/summary.txt` (all 36
rows), the val baselines, the v2 and v3 result numbers, the locality percentages (+34/+48/+148 and +52/+43/+82), the pool yields
and margins, EPISODE_TIME (8.41 s -> 9.0), the reject-reason mix, and the "M1 fails 100092 and only M1+L1 completes it" claim
(calibration CSVs).
