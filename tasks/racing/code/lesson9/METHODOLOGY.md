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
  **Code written, not yet trained** (2026-09-22): see section 5a.
- **Contrast group (3), "contrast RL"** (added 2026-09-22, user request): the SAME racing envelope, PPO
  settings, preview window and frequency as the robust group, but the VENDORED, unmatched disturbance
  training (section 5b) -- isolates whether matching the domain-randomization ranges to the study's frozen
  ceilings buys anything beyond generic robustness training. Training seeds 0, 1, 2 (matching the robust
  group's count on purpose -- section 5b). Shown thin/muted on the study's charts, not folded into either
  family's story: its only job is to say whether the robust family's crossover location moves relative to
  it, or stays where it is. **Code written, not yet trained.**
- **Not in the study as first-class members:** `v5_s0`, `racing_s0`, `racing_s1`, `racing_s2` (the course's
  own Lesson 2/7 checkpoints). `v5_s0` was considered as a free contrast and rejected: it is trained on a
  narrower reference envelope (|v| up to 3.5 m/s) than this study's tracks need (stretched to a 4.4 m/s
  cap; Lesson 7 already measured it failing the unstretched fast tube plan, 3/4), so it would conflate
  "wrong reference envelope" with "unmatched disturbance training" -- not a clean isolation of the one
  variable the contrast group exists to test. No family mean for any of the three RL-family lines
  (robust, contrast): every member is its own line.

## 5a. The robust RL training recipe (`robust_env.py`, `lighthouse_batch.py`, `robust_policy.py`,
`train_robust.py` -- code written and mechanics-tested 2026-09-22, `test_robust_env.py`, 9 tests; NOT yet run)

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

**Round 1 (seed 0 of each group), read without `ep_rew_mean`.** Both trained for about 65 minutes (the
tensorboard event timestamps, not the `RunLogger` directory-name timestamp -- the two disagree by about 7
hours in this environment, apparently a timezone mismatch between whatever clock `datetime.now()` reads at
directory-naming time and the container's filesystem clock; harmless but worth knowing before reading a
duration off a directory name again). Optimizer diagnostics look healthy: `train/value_loss` 27.6 -> 2.06,
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

**The training reference distribution never asks for gate-threading (2026-09-23, the other half of the
user's steer, not yet acted on).** Read `ChainedPolyTrajectory.random` directly: every reference the policy
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
three, deviating 3-10x more than M1's worst moment and 4-11x the track's actual margin at the point of
impact -- not a marginal near-miss a slightly wider corridor would fix. **Conclusion: this is overwhelmingly
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
(23 -> 90 -> 136 cm max deviation on track 93). If nominal eval were out-of-distribution in a way that
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
2/7, not introduced here), and both policies are pinned against that self-imposed ceiling 7.6-25.1% of the
flight; M1, with the full `RPY_MAX=1.0` available, reaches it on two of three tracks (1.000, 0.960) but is
not chronically pinned the way RL is against its 70% cap. Peak speed is confounded by episode length (RL
crashes at 150-350 steps, before the faster later section of the lap M1 flies through at 400+) and is not
read as an independent speed deficit. Net conclusion: this does not surface a new independent root cause --
it adds texture to the established one. The RL policies are fighting larger tracking errors (more thrust,
routinely maxing attitude authority to correct) rather than tracking with margin to spare the way M1 does;
the 70% pitch cap is a real, previously-unmeasured, non-Lesson-9-specific constraint, best read as a symptom
of chasing bigger errors rather than a separate cause.

## 5b. The contrast-group## 5b. The contrast-group training recipe (`contrast_env.py`, `train_contrast.py` -- code written and
mechanics-tested 2026-09-22, `test_contrast_env.py`, 5 tests; NOT yet run)

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
would waste one. It is also better science on its own terms -- Lesson 7's `racing_s1` showed a single bad
seed is not a bad recipe, and a 3-vs-2 seed-count asymmetry between the robust and contrast groups would
make that harder to read cleanly for whichever side had fewer.

**Evaluation.** A contrast-trained model is evaluated through the SAME `robust_policy.RobustPolicyController`
a robust model uses (`driver.py`'s `robust:<path>` spec) -- that wrapper's only job is the 0.8 s window at
the harness's own frequency, correct for either group; nothing about it assumes matched disturbance
training. `datt:<path>` (the vendored 0.6 s window) is wrong for both.

**Chart treatment (agreed, not yet built -- phase 8).** Thin/muted lines, their own colour family, distinct
from both the model-based and robust-RL lines; never folded into either group's mean. Used only to check
whether the robust family's crossover location moves relative to this baseline, or stays where it is.

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
Training the 3 robust seeds: 25-50 minutes each. A ground plan plans in 0.4 s; a candidate track costs about
1.3 s (planning, verdict, exact contact test).

## 10. Limitations to state in Lesson 9

1. Gate-only geometry with a fixed start, heights and gate order.
2. The corrected MPC model is a fit to this same simulator; its "accurate model" favours MPC.
3. ~~The RL family is trained on this study's range, and with no contrast group a crossover cannot be
   attributed to the training edge versus something intrinsic~~ -- addressed 2026-09-22: a 3-seed contrast
   group (section 5b) trains the same recipe on the vendored, unmatched disturbance ranges, so the study
   can check whether the robust family's crossover location moves relative to it. Not yet trained, so this
   remains a designed answer, not a measured one, until Phase 6 actually runs.
4. ~~No parametric mass mismatch~~ -- superseded 2026-09-22: `mass_mult` (section 4) and M1+mass (section 5)
   were added at the user's request, specifically because Lesson 8 section 5 found this, not an additive
   force, is what defeats MPC in the race. Training will include matched mass domain randomization
   (section 5a). What remains true: mass is the only condition excluded from `combined`.
5. Gust time structure is not in training. The 50 Hz/100 Hz training/harness mismatch and its specific
   consequences (the L1 estimator's bandwidth, the preview window) are detailed in limitation 14.
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
    (`WINDOW=10` points at `WINDOW_DT=0.06` s) -- equalising this to 0.8 s is PLANNED (section 5a) but not yet
    built, so it remains a live gap until Phase 6 delivers it. Separately, the deployed RL policy's own L1
    estimator (always active) is shaped by training at 50 Hz but integrates its adaptation law at 100 Hz in
    the harness, changing its effective bandwidth relative to what the policy learned to expect -- training at
    `freq=100` (section 5a, PLANNED) removes this specific mismatch, at roughly double the wall-clock training
    cost for the same simulated experience unless `--timesteps` is also doubled. Also worth stating plainly:
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
    Measured impact (the actuator-saturation check above): both RL groups pin at EXACTLY 0.700 rad on pitch
    7.6-25.1% of the flight, on every one of 3 tracks checked, while M1 reaches its own (larger) ceiling on
    2 of 3. This does not overturn the reference-distribution conclusion (a policy that never practiced
    gate-threading would still fail with more authority), but it means the comparison is not purely
    architecture-vs-architecture: one side has less control authority available by construction. Closing it
    is not a config change -- the network was trained around the 0.7 scale, so it requires retraining with a
    wider (candidate: matched to M1's 1.0 rad) cap, a new open decision, not yet made.

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
    0.2044 vs `robust_s0_screen`'s 10/24 gates, 1/6 completions, mean RMSE 0.2161 -- a wash per-track (better
    on 2 tracks, worse on 3, and it lost the one track `robust_s0_screen` actually completed), not a clean
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
    deviation over the final 1 s before each failure, 5 of 6 RL cases are actively DIVERGING into the crash
    (e.g. contrast_s0 track 93: +0.470 m/s, robust_s0 track 387: +0.323 m/s), while M1 is flat-to-slightly-
    positive everywhere and, on its own hardest case (track 93 capped to 0.7 rad), actively CONVERGING
    (-0.228 m/s) in the same window. Conclusion: this is not "RL fails by a generic amount that happens to
    coincide with a gate" -- its error specifically GROWS worst exactly where the geometry punishes it most,
    and is often still growing at the moment of impact rather than plateaued. This sharpens, not just
    supports, the reference-distribution explanation: the reward (`exp(-2*err) - 0.02*||action||`, applied
    uniformly every timestep in `datt_env.py`'s `step()`) gives no extra weight to error near a narrow
    feature, and nothing in `ChainedPolyTrajectory.random` ever creates one to weight. Whichever of the three
    reference-distribution options gets chosen, this finding argues it needs to specifically create moments
    where being off-target has a sharp, LOCAL cost -- not just narrower open-space curves on average.

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

1. Track library (`gen_tracks.py`, `track_lib.py`) -- done; restarted 2026-09-22 at thrust-frac 0.80
2. Contact check (`contact.py`) -- done
3. Lambda knobs (`knobs.py`) -- done; extended 2026-09-22 with `mass_mult` and `mass_scale`
4. Driver and storage (`driver.py`) -- done; extended 2026-09-22 with the mass override, M1+mass, and the
   mppi_l1 0.8 s preview-horizon reconfiguration
5. Dev calibration (`calibrate.py`) -- done; three passes (section 4's history), all 5 ceilings frozen in
   `maxima.json`; `lighthouse` moved from 1.0x to 1.5x on the third pass (M1+mass entering the roster)
6. Robust RL training -- **code written 2026-09-22 (`robust_env.py`, `lighthouse_batch.py`,
   `robust_policy.py`, `train_robust.py`); verified by construction/reset/step only (`test_robust_env.py`,
   9 tests), never by `.learn()`. NO TRAINING HAS BEEN LAUNCHED (user's explicit instruction).**
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
6b. Contrast-group training -- **code written 2026-09-22 (`contrast_env.py`, `train_contrast.py`); verified
    by construction/reset/step only (`test_contrast_env.py`, 5 tests), never by `.learn()`. NO TRAINING HAS
    BEEN LAUNCHED.** `ContrastTrackingEnv(RacingTrackingEnv)`: the same freq=100 / 0.8 s window fix as the
    robust recipe, and NOTHING else changed -- the vendored +-3.5 m/s^2 force box and the vendored
    noise-scale `LighthouseSensorBatch` (not `LambdaLighthouseSensorBatch`), no mass channel. Tests confirm
    the window/frequency match `RobustTrackingEnv` exactly, the force box matches the vendored
    `PERTURB_ACC_MAX` (not the study-matched range), the vendored sensor class is in use, and no mass
    channel exists (`sim.data.params.mass` never diverges from `default_data`'s). Section 5b.
    **Hardware note (6 seeds total, 3 concurrent-training pairs):** 3 robust + 3 contrast seeds, the user's
    choice specifically so 6 seeds split into 3 pairs use every slot under the 2-concurrent-session limit (5
    seeds would waste one). Each seed: `--timesteps 8000000` (double the vendored 4,000,000, compensating
    for freq=100 halving simulated experience per env-step), roughly 50-100 minutes. 3 pairs (each bounded
    by its slower member) x 50-100 minutes = roughly 150-300 minutes (2.5-5 hours) wall time for all 6 seeds.
7. Study sweep -- not started (depends on phases 6 and 6b)
8. Charts -- not started
9. `lessons/09-crossover-under-disturbance.md` -- started 2026-09-22: opening, the trackers table, and the
   full training section (the two recipes, why the contrast group exists, the 3-round/6-seed schedule, the
   exact commands) are written; sections on tracks/contacts/disturbances/calibration/results are pending
   behind this file until their phases produce something to report. Indexed in `tasks/racing/README.md`'s
   lesson table, marked in progress.
