# Lesson 9 — Where the crossover is: learned vs model-based racing under graded disturbance

⏱️ ~3 hours of reading and setting up; the compute is the expensive part. Calibrating the
ceilings is about 40 minutes on 8 workers, training one learned seed about 6 hours, and the full
sweep several days on 8 workers, so start each long run as soon as its section is ready and read on
while it flies. **You finish when** you have (a) your pre-registration written and dated before any
lap of the sweep, (b) your own track library and frozen ceilings, (c) a learned racer that clears
the viability bar you set for it in advance, (d) the sweep flown (or at least one track of it), and
(e) the crossover intensity λ\* of every channel and track computed with the rule you pre-registered,
next to whether a crossover was even possible there.

> **The one big idea:** Lesson 7 asked who wins at one disturbance setting. This lesson asks *where*
> the ranking between a learned and a model-based tracker changes as a disturbance grows from nothing
> to a calibrated ceiling, and whether that place stays put when the track changes. Answering it
> honestly is mostly design: tracks the learned policy was never developed on, ceilings calibrated
> before any policy exists, a comparison equalised on purpose, and a crossover rule fixed before the
> data can suggest one. And one check most comparisons skip: before asking who won, ask whether the
> losing side *could* have won. A crossover is impossible where the leader never falls far enough to
> be passed, and a missing crossover there says nothing.
>
> **Whatever you find describes these tracks, these disturbance models and these trackers.** It is
> not a universal threshold for learned or model-based control.

**Prerequisites:** Lesson 4 (pre-registration), Lesson 7 (the disturbance models and the λ grading),
Lesson 8 (the corrected MPC family M1, ground-start plans, the box contact model), and the TOGT
driver built (`bash tasks/racing/code/togt/build.sh`, Lesson 6). Everything runs inside the
`rl_quad_traj` container; the commands below are written to be pasted into a plain host terminal.

**Reference run.** The study this lesson guides you through was run in full, and its outputs are in
this repository so you can compare yours at every step: tracks and manifests in
`tasks/racing/code/lesson9/tracks/`, plans in `tasks/racing/plans/lesson9/`, the frozen ceilings in
`lesson9/maxima.json`, every lap under `lesson9/results/`, the trained policies in
`lesson9/policies/`, and the analysis in `lesson9/analysis/`.
Every runner below **skips laps that are already on disk**, so give your own runs their own `--tag`
(or work in a fresh copy of the repository) or you will fly nothing and read the reference instead.

---

## 0. The design

**Two families, each scored by its best member.**

| family | member | what it is | spec |
|---|---|---|---|
| model-based | `M1` | the corrected MPC of Lesson 8 | `mpcdev:att=sim,drag=0.495,fgain=1.0` |
| | `M1+ESO` | M1 + velocity extended-state observer | `…,dist=eso` |
| | `M1+L1` | M1 + L1 adaptation | `…,dist=l1` |
| | `M1+mass` | M1 + thrust-scale (mass) adaptation | `…,mass=1` |
| | `mppi_l1` | sampling-based MPC + L1, 0.8 s horizon | `mppi_l1` |
| learned | three seeds of one recipe | a gate-aware DATT policy (§4) | `robust:<zip>` (0.7 rad) or `robust_full:<zip>` (1.0 rad) |

The model-based family is fixed before you start and never re-tuned. That is a choice with a
consequence: your claims are about *this* family, not about what model-based control could do with
one more estimator.

**Six channels, each graded by λ ∈ [0, 1].** `wind_const`, `payload` (a steady downward force),
`mass_mult` (a heavier drone), `wind_gust`, `lighthouse` (degraded optical sensing) and `combined`
(gust + payload + sensing). λ = 0 is nominal; λ = 1 is a ceiling you calibrate in §3. Because the
ceiling is set by when the *model-based* family fails, λ measures difficulty for that family, not
physical severity, so report physical units next to it.

**Four track pools, kept apart on purpose.**

| role | seeds | used for | never used for |
|---|---|---|---|
| `dev` | 100000+ | calibrating the ceilings (§3) | training, reporting |
| `train` | 200000–205999 | training the learned policy (§4) | any decision you report |
| `val` | 300000–301999 | every learned design decision, previews, held-out tests (§4–§6) | the sweep |
| `study` | 0–1318 | the sweep (§7) | any design decision |

The point of the separation is that nothing you tune ever sees a track the sweep reports on. If you
screen an early design on a study track, that track is spent: drop it from the sweep. (The reference
run screened on five study tracks and swept the other ten; `study_sweep.py` excludes the five.)

## 1. Pre-register before you fly anything

Write these down, date them, and commit them before the first sweep lap. Each needs the number that
would make it wrong (Lesson 4).

1. **The crossover rule.** Score each family per cell (channel, λ, track) by its best member,
   S = max over members of the completion rate, and let Δ = S_RL − S_MPC. Decide:
   - the margin δ a lead must exceed (the reference run used 0.2);
   - the window: up to which λ a crossover counts (the edge of the learned training range is the
     natural choice, so every claim stays inside what the policy was trained on);
   - λ\*: the first λ at which Δ > δ *and after which Δ never falls below −δ again*, so that an
     isolated blip does not count.
2. **The power conditions.** Because S_RL ≤ 1, a crossover needs S_MPC < 1 − δ somewhere in the
   window. Call a track one that *allows* a crossover only if the best MPC member has that **room**
   and the learned family still completes half its laps there (**retention**). A missing crossover is
   evidence only on tracks that allow one.
3. **Viability.** What makes a member count on a track at all (the reference: at least half of its
   λ = 0 laps), and therefore which tracks enter the comparison.
4. **Your hypotheses**, with their kill numbers: which channels you expect to cross and why, whether
   precision and completion will order the controllers the same way, and whether the crossover
   location should move with the track.
5. **The learned viability bar** (§4) and **the held-out rule** (§6), both before their first flight.

Anything you decide *after* seeing data is allowed, but it is exploratory and has to be reported as
such. Keep a list with two columns, "fixed before" and "chosen after", and put it in your write-up.

## 2. Build the track library

Each candidate is a random lsy level-3 layout (start fixed on the floor, gates at 0.7/1.2/0.7/1.2 m,
only x, y and yaw random), planned as a TOGT ground-start tube. A layout is kept only on properties of
its *plan*, never on how any controller flies it: the plan must cross every gate inside its opening
at no more than 25° to the gate normal, stay in the arena, and keep at least 3 cm between the drone's
collision box and every gate frame.

```bash
docker exec -it rl_quad_traj bash -lc 'cd /workspace && export JAX_PLATFORMS=cpu && /opt/venvs/main/bin/python \
  tasks/racing/code/lesson9/gen_tracks.py --role study --n 15 --seed-start 0'
docker exec -it rl_quad_traj bash -lc 'cd /workspace && export JAX_PLATFORMS=cpu && /opt/venvs/main/bin/python \
  tasks/racing/code/lesson9/gen_tracks.py --role dev --n 3 --seed-start 100000'
docker exec -it rl_quad_traj bash -lc 'cd /workspace && export JAX_PLATFORMS=cpu && /opt/venvs/main/bin/python \
  tasks/racing/code/lesson9/check_references.py --role study'
docker exec -it rl_quad_traj bash -lc 'cd /workspace && export JAX_PLATFORMS=cpu && /opt/venvs/main/bin/python \
  tasks/racing/code/lesson9/gen_pool.py'          # the train and val pools, in parallel seed windows
```

You get `tracks/<role>/track_<seed>.json`, the plans in `tasks/racing/plans/lesson9/`, a manifest
with **every** candidate and why it was rejected, and a top-down overview in
`tasks/racing/figures/lesson9/tracks_<role>.png`. Check before moving on:

- **How hard does the filter select?** Compare the accepted tracks' peak speed and planned lap with
  all candidates' in `manifest_study.csv`. A strict filter keeps compact, gentle layouts; whatever you
  later conclude about geometry applies to that population only, so say which one it is.
- **Is every accepted reference contact-free with margin?** `tracks/reference_margins_study.csv`. The
  margin is how far a tracker may stray at a gate before the race ends; it is a few centimetres here.
- **Did you change a filter after looking at which tracks it accepts?** Then the change is a design
  decision on the study tracks, and it goes in your "chosen after" column.

## 3. Calibrate the ceilings on the dev tracks

The ceiling of a channel is the lowest scanned level from which at most one model-based member still
completes half of its laps, pooled over the dev tracks. Calibrate before any learned policy exists, on
dev tracks only, with the model-based family only.

```bash
docker exec -it rl_quad_traj bash -lc 'cd /workspace && export JAX_PLATFORMS=cpu && \
  /opt/venvs/main/bin/python tasks/racing/code/lesson9/calibrate.py run --workers 8'
docker exec -it rl_quad_traj bash -lc 'cd /workspace && export JAX_PLATFORMS=cpu && /opt/venvs/main/bin/python \
  tasks/racing/code/lesson9/calibrate.py report'
```

`report` prints the completion table per channel and member and writes `maxima_proposed.json`.
Freezing is a deliberate act: read the table, then copy it to `maxima.json` yourself. Once frozen, it
does not move again; if the roster or the plans change, recalibrate and record why.

Two things to notice in your table. First, the rule shapes what the model-based family can show below
the ceiling: by construction at least two members still finish half their laps one scan level below
it, so a window like λ ≤ 0.8 lies almost entirely where that family is known to succeed. Predict what
that implies for the channels the family models well before you see the sweep. Second, a ceiling is a
property of the plans as much as the controllers: the same paths flown slower move it.

## 4. Build a learned racer

A policy trained to track smooth open-space references does not race: nothing in that training asks
it to thread a 0.4 m opening with a few centimetres of margin, and it hits frames. The recipe in
`gate_aware_env.py` builds the race into training, and each of its choices closes a way the comparison
can quietly fail:

- **Real gates in training.** References are ground-start plans of the `train` pool, and an episode
  ends on gate-frame contact, using the same contact test as evaluation.
- **A missed gate also ends the episode.** Penalising contact alone teaches a policy to fly *around*
  the gate, which is cheaper than threading it.
- **A pool, not a handful of tracks.** A recipe that learns three training tracks can still fail on
  every unseen one; viability is judged on `val`, never on the tracks it trained on.
- **Domain randomisation tied to your ceilings.** Force, mass and sensing are drawn per episode up to
  a fixed fraction of the frozen ceilings (`robust_env.py`, `lighthouse_batch.py`). Check the ranges
  you actually train on against the ceilings in `maxima.json`, in physical units, and write them in
  your methods exactly as the code has them.

**Set the viability bar first.** Before flying any checkpoint, write down how many of the 22 `val`
tracks it must complete at λ = 0 to be worth sweeping (the reference: at least 11, half of what M1
completes), and that intermediate checkpoints are reported as a curve but not used to choose.

```bash
# train one seed (about 6 hours for 16M steps; a checkpoint every 2M)
docker exec -it rl_quad_traj bash -lc 'cd /workspace && export JAX_PLATFORMS=cpu && \
  /opt/venvs/main/bin/python tasks/racing/code/lesson9/train_gate_aware.py \
  --group robust --seed 0 --reason "Lesson 9: gate-aware recipe, 0.7 rad, seed 0"'

# fly it against M1 on the validation pool, in your own results folder
docker exec -it rl_quad_traj bash -lc 'cd /workspace && export JAX_PLATFORMS=cpu && \
  /opt/venvs/main/bin/python tasks/racing/code/lesson9/eval_pool.py --role val --m1 --tag my_val \
  --member s0=robust:/workspace/tasks/racing/crazy_track/results/<your run>/datt_ppo_final.zip'
```

Train **three seeds** (`--seed 0`, `1`, `2`) and keep all three, whatever they score: dropping the
weakest selects on the validation pool. Expect the spread between seeds of one recipe to be large,
possibly larger than the margin δ you chose; that is why the family is scored by its best member *and*
reported through its median seed and the seed you would deploy (the best on `val`).

**Choose the control authority, and say what it costs.** The MPC family was tuned with roll and pitch
bounded at 1.0 rad; the vendored learned recipe was built at 0.7 rad. `--group robust_full` trains the
same recipe at 1.0 rad (spec `robust_full:`). Each corner hands the cost to one side: both at 1.0 rad
is fair to the MPC, both at 0.7 rad caps the MPC below its tuning, each at its own bound is natural for
both but unequal. Pick your primary corner **before the sweep**, write down the rule that would make you
change it, and fly at least one other corner as a sensitivity line (the reference flew the 0.7 rad
recipe in full as its alternative).

## 5. Equalise what you can, and list what you cannot

Before the sweep, give both families the same measured state, the same reference preview span (0.8 s
here: the MPC horizon and the policy's ten preview points), the same control rate and the same
roll–pitch bound in the primary corner. Then write down every asymmetry that remains, with the side it
favours. Check at least these in the code: which family sees the gates in training; who observes a
disturbance estimate; preview resolution; yaw; whether body rate is observed; the estimators' soft
start; whether any member filters the measured state; and whether controller computation time is
simulated (here it is not: the simulation waits for every solve, which favours the slower MPC). Two
asymmetries that pull in opposite directions on the same channel leave its net effect unknown, and your
write-up should say so rather than pick one.

## 6. On the validation pool: preview, then held-out conditions

**Preview** the flown policy against M1 and M1+L1 on `val` at a coarse λ grid. This is development
information, not a result: it tells you whether a crossover is plausible at all and whether your grid
covers the interesting range.

```bash
docker exec -it rl_quad_traj bash -lc 'cd /workspace && export JAX_PLATFORMS=cpu && \
  /opt/venvs/main/bin/python tasks/racing/code/lesson9/preview_lambda.py --tag my_val_lambda \
  --member s0=robust:/workspace/tasks/racing/crazy_track/results/<your run>/datt_ppo_final.zip'
```

**Held-out conditions** ask whether the policy's standing survives conditions it never trained on.
`knobs.HELD_OUT` has four, each paired with its nearest trained analogue: an upward force (`lift`,
analogue `payload`), a lighter drone (`light`, analogue `mass_mult`), a wind that switches on mid-lap
(`step_wind`, analogue `wind_const`) and a frozen position stream (`blackout`, analogue `lighthouse`).
The measure is a difference of differences: how much *more* the policy loses than an MPC member when
moving from the analogue to the held-out condition. Write the per-seed rule before the first flight
(the reference: a cell is "overfit" if the policy loses at least 20 points more than both M1 and
M1+L1, "generalises" if it stays within 20 of both, and five cells of eight decide a seed's verdict).

```bash
docker exec -it rl_quad_traj bash -lc 'cd /workspace && export JAX_PLATFORMS=cpu && \
  /opt/venvs/main/bin/python tasks/racing/code/lesson9/heldout_conditions.py --tag my_val_heldout \
  --in-dir /workspace/tasks/racing/code/lesson9/results/eval/my_val_lambda \
  --member s0=robust:/workspace/tasks/racing/crazy_track/results/<your run>/datt_ppo_final.zip'
```

`--in-dir` points at *your* preview, which holds the analogue conditions. The summary prints the raw
table, the retention table, the differences of differences with a bootstrap interval over the tracks,
and the tally. Before you trust a verdict, work out how wide one cell's noise is: with 22 tracks, and
fewer for a seed that completes only a handful at λ = 0, a 20-point threshold is less than one standard
deviation. Read each condition per cell, not only through the one-word verdict.

## 7. Run the sweep

`study_sweep.py` flies every member on every untouched study track under every channel, in four
phases, one track and one channel at a time, each resumable from its own CSV:

- **`determinism`**: repeats one cell to find which members are deterministic. A deterministic member
  needs one lap per cell on the deterministic channels; a stochastic one (the sampling-based MPC) needs
  several. Do not skip this, or your seed counts are guesses.
- **`coarse`**: λ ∈ {0, 0.1, …, 1}, 25 noise seeds per cell on the stochastic channels, with common
  random numbers, so the same seed draws the same noise at every λ and for every controller.
- **`midpoints`**: adds λ + 0.05 wherever any member's completion changes by 0.2 or more between
  neighbours. Midpoints refine the curves; your crossover rule should use the coarse grid only.
- **`charts`**: one folder per track and channel, with the raw CSV and its completion charts.

```bash
M='--member fa0=robust_full:/workspace/tasks/racing/crazy_track/results/<run fa0>/ckpt/ppo_16000000_steps.zip \
   --member fa1=robust_full:/workspace/tasks/racing/crazy_track/results/<run fa1>/ckpt/ppo_16000000_steps.zip \
   --member fa2=robust_full:/workspace/tasks/racing/crazy_track/results/<run fa2>/ckpt/ppo_16000000_steps.zip \
   --sensitivity-member s0=robust:/workspace/tasks/racing/crazy_track/results/<run s0>/datt_ppo_final.zip \
   --sensitivity-member s1=robust:/workspace/tasks/racing/crazy_track/results/<run s1>/datt_ppo_final.zip \
   --sensitivity-member s2=robust:/workspace/tasks/racing/crazy_track/results/<run s2>/datt_ppo_final.zip'

docker exec -it rl_quad_traj bash -lc "cd /workspace && export JAX_PLATFORMS=cpu && \
  /opt/venvs/main/bin/python tasks/racing/code/lesson9/study_sweep.py --phase determinism $M"
docker exec -it rl_quad_traj bash -lc "cd /workspace && export JAX_PLATFORMS=cpu && \
  /opt/venvs/main/bin/python tasks/racing/code/lesson9/study_sweep.py --phase coarse $M \
  --track 1089 --condition lighthouse --workers 8"
# then midpoints and charts for the same track and channel; repeat for every track and channel
```

Replace each `<run …>` with your own run folders. To fly the reference policies instead, use
`/workspace/tasks/racing/code/lesson9/policies/<run>/…` (the top-level README and `lesson9/results/README.md`
say which folder is which). The sweep writes to `results/eval/study_sweep/`, where the reference laps already
are: to fly the whole sweep yourself, work in a fresh copy of the repository and delete that folder
there first. Budget it honestly: the coarse grid alone is tens of thousands of laps, most of them
solver time, so a full sweep on 8 workers takes days. Run it track by track and let the resumability
absorb interruptions.

**Spot-check before you trust anything, yours or the reference's.** Fly one cell into a scratch file
and compare it with the stored row; a deterministic member must reproduce it exactly:

```bash
docker exec -it rl_quad_traj bash -lc 'cd /workspace && export JAX_PLATFORMS=cpu && \
  /opt/venvs/main/bin/python tasks/racing/code/lesson9/driver.py --role study --track 1089 \
  --cond wind_const --lams 0.5 --only M1 --out /tmp/spot.csv && cat /tmp/spot.csv'
```

## 8. Analyse with the rule you wrote down

`tasks/racing/code/lesson9/analysis/` holds a small library (`lib.py`, `pairs.py`) that reads the
stored laps and implements the definitions of §1: completion tables, viability, family lines (best
member, median seed, median member) and λ\*. It reads only CSVs and needs nothing but `numpy`,
`pandas`, `scipy` and `matplotlib`. Compute in this order:

1. **Who enters.** The tracks where both families have a viable member. Everything after this counts
   only those, and says how many there are.
2. **Who could have won.** Room and retention per channel and track, *before* any crossover result.
   Report how many tracks allow a crossover next to every crossover count.
3. **Where it crossed.** λ\* per channel and track, inside the window and on the full grid:

   ```python
   from lib import *            # run from tasks/racing/code/lesson9/analysis
   for t in entering_tracks("lighthouse", FA):
       mpc, rl = family_lines("lighthouse", t, FA)       # best MPC member, best learned seed
       print(t, lam_star(rl - mpc, 0.8, 0.2), lam_star(rl - mpc, 1.0, 0.2))
   ```

4. **Why it crossed.** Every crossover is accompanied by an MPC fall, by construction, so that fall
   proves nothing on its own. Ask instead whether the learned family *rose*: compare its completion at
   λ\* with λ = 0 and with the first λ > 0, on unfiltered data (the viability filter makes a rise
   impossible to see).
5. **Precision separately from finishing.** On laps both families complete (`pairs.paired`), compare
   tracking error and lap time. A gated race is decided by finishing; check whether the two outcomes
   order the controllers the same way.
6. **How much to believe it.** Repeat the counts at other margins δ, for the median learned seed and
   against the median MPC member, and resample the seeds of each stochastic cell (`boot.py`). Remember
   that the unit of replication for a statement about tracks is the track, not the lap: tens of
   thousands of laps on six tracks are six observations.

`analysis.py` and `thesis_numbers.py` recompute every number the reference run reported and check each
against the published value; run them on the reference data to see the whole analysis end to end, or
read them as worked examples of each step above.

## 9. What a design like this cannot settle

Write these into your limitations, with your own numbers: a model-based family without a state filter
or delay compensation is not model-based control in general; an unsimulated computation time favours
the slower controller; three seeds per recipe cannot separate a recipe from its luck; a strict track
filter limits every geometry claim to the population it kept; ceilings move with plan aggressiveness;
and anything you decided after seeing the sweep is consistency with your account, not confirmation of
it. The only test that confirms is one whose rule was written before its data.

---

## 🛠️ Before you move on

1. **Find the power conditions in your calibration.** From your §3 table alone, predict on which
   channels the best MPC member will have no room below the window edge, and so on which channels no
   crossover is possible whatever the policy does. Check the prediction against the sweep.
2. **Spot-check the reference.** Fly two cells of the stored sweep yourself (one deterministic, one
   stochastic seed) and confirm they reproduce. Explain why the stochastic one only reproduces with
   the same seed.
3. **Argue the other authority corner.** Write the paragraph that makes the corner you did *not* pick
   the primary line, and say what would have to be true of the MPC's tuning for it to be right.
4. **Design the control group the reference could not build.** Matched training (the policy trained on
   the very perturbation models it is evaluated on) is a confound. Propose a control that separates
   "trained on matched ranges" from "learned generic robustness" and still learns to race, and name
   the number that would tell you it worked before you spend the compute.
5. **Recompute one λ\* by hand.** Take one track and channel, write out Δ on the coarse grid from the
   completion table, apply your rule, and check your answer against `lam_star`.

**Back to:** [Lesson 8 — Closing the gap](08-closing-the-gap.md)
