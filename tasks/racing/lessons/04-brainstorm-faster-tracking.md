# Lesson 4 — Brainstorm: make it faster

⏱️ Open-ended. This is the research part.
**You finish when** you have run at least one *pre-registered* experiment and
can defend its verdict — including if the verdict is "my idea did not work".

> **The one big idea:** the difference between research and guessing is not
> cleverness. It is **writing down what would prove you wrong, before you look
> at the result.** Everything below is organised to make that easy.

---

## 1. How to run an experiment here

Copy this into a file in your report folder *before* touching code:

```markdown
## Experiment: <name>              date, your name
HYPOTHESIS: <what happens, and the MECHANISM you think causes it>
MEASURE:    <the exact number, on the exact protocol>
THRESHOLD:  <the value that makes you right; and the value that makes you WRONG>
SEEDS:      3 training seeds x 20 race episodes
```

Then run it, then add one line: **VERDICT**, with the number.

Why this is not bureaucracy: without a threshold fixed in advance, *any* result
can be read as encouraging. With one, you get an actual answer — and the
experiments that fail become results instead of wasted weeks. In the parent
project the failed experiments are the most-cited part of the record.

Two rules that make verdicts trustworthy:

- **One variable per experiment.** Change the reward *and* the network and you
  have learned about neither.
- **Beat your noise floor.** You measured the three-seed spread in Lesson 2. An
  improvement smaller than that spread is not an improvement, it is weather.

## 2. Where the time actually is

Decompose first (you instrumented this in Lesson 3 §6):

```
lap time  =  takeoff  +  time along the path  +  time lost tracking badly
```

Every idea below is tagged with **which term it attacks**. If you cannot say
which term yours attacks, think for another ten minutes rather than start a
50-minute training run.

And keep the ceiling in view: the parent project measured a tracker that **beat
its own reference by cutting corners**. If your tracking error is already small,
the remaining time is in the *plan* — which this course does not optimise.
Proving that bound for your setup is a perfectly good result.

## 3. The idea board

Each entry: the idea, why it might work, **why it might not**, and the cheapest
experiment that decides.

### A. Observation design → *tracking error*

**A1. Longer or shorter look-ahead.** `WINDOW = 10`, `WINDOW_DT = 0.06`
(`datt_env.py:20`) → 0.6 s. At racing speed the drone covers far more ground in
0.6 s than at training speed.
- *Might work:* a longer horizon lets the policy set up for gates earlier.
- *Might not:* more inputs at fixed network capacity dilutes the useful ones,
  and information beyond the control horizon is not actionable.
- *Test:* `WINDOW_DT = 0.10` (1.0 s horizon), retrain, compare. **Cheap.**

**A2. Body-frame window.** `rel_win` is relative to the drone's *position* but
in *world* axes (`datt_env.py:233`).
- *Might work:* in body frame the same corner looks identical regardless of
  heading — one skill instead of many rotations of it.
- *Might not:* yaw is fixed at 0 here (`datt_env.py:302`), so the frames differ
  less than you would think. **Check before you build.**

**A3. Give it the reference *velocity*.** Currently the policy differences the
window to infer speed.
- *Might not:* it is redundant; PPO may already extract it.

### B. Training distribution → *tracking error* (highest yield)

**B1. Train at racing speeds.** This one has a receipt in the code already —
Lesson 1 §5.1, `datt_env.py:110`: policies trained on gentle references
**failed at 0.95 m RMSE**. The ranges were widened to `vel ≤ 3.5`, `acc ≤ 10`
for exactly that reason.

Now ask whether your *race* reference exceeds even that:

```bash
# in the race env, after building your reference in Lesson 3
python -c "
import numpy as np
from crazy_track.trajectories.freestyle import lsy_level2_race
traj = lsy_level2_race(cruise=3.0)
t = np.arange(0, traj.duration, 0.01)
v, a = traj.vel(t), traj.acc(t)
print('race max speed        %.2f m/s   (training draws up to 3.5)' % np.linalg.norm(v,axis=1).max())
print('race max acceleration %.2f m/s^2 (training draws up to 10)'  % np.linalg.norm(a,axis=1).max())"
```

**If the race is outside the training box, that is your bug** — and it is a
one-line fix at `datt_env.py:165`.

**B2. Train on race-like geometry.** Random polynomials are smooth; a race line
is tight turns through gates.
- *Might not:* **the big risk** — train on one track and you may get a policy
  that memorises it and generalises worse. Guard with a *held-out* track (there
  are others in `freestyle.py`). If it wins on the race track and loses
  everywhere else, you have overfitted, and that is a finding worth reporting.

**B3. Randomise mass to match Level 1** (±11.5 %, Lesson 3 §1).
- Clean pre-registered prediction: helps on Level 1, does nothing on Level 0.

### C. Architecture → *tracking error*

**C1. Recurrent policy (`--v7`).** A GRU actor with a feedforward privileged
critic, already implemented.
- *Might work:* memory lets it *infer* what it cannot see — such as the current
  mass, from how the drone responds. The natural fit for Level 1.
- *Might not:* recurrent PPO trains slower and is fiddlier.
- *Note:* the parent project has an ongoing study of this variant. Read its
  reports before starting so you do not repeat an existing measurement.

**C2. Frame stacking (`--v6`).** ⚠️ **Known negative result** — stacking 4
frames did *not* fix noise adaptation at 4 M steps. Do not spend a week
rediscovering it. If you try it, try it for a *different* reason, and say what
that reason is.

**C3. Bigger network.** Currently `net_arch=[64, 64]` (`ppo_train.py:133`).
- *Might not:* capacity is rarely the bottleneck and it slows everything. Try it
  **after** you have evidence — e.g. training reward plateaus below what a
  hand-tuned controller achieves.

### D. Reward → *tracking error*

**D1. Sharpen the error term.** `exp(-2·err)` (`datt_env.py:338`) is forgiving.
- *Might not:* too sharp and the gradient vanishes when the drone is far away —
  early training gets no signal at all. The parent project has an entire lesson
  on rewards that are flat exactly where the policy currently is.

**D2. Reward progress along the path instead of position error.** The classic
racing reward.
- ⚠️ This changes the problem from *tracking* to *racing*: a progress reward
  gives the policy permission to leave the reference. It may be faster, and it
  is no longer what this course scoped. **That is allowed — just say so in your
  report,** and expect gate-miss failures to rise.

### E. The interface → *all three terms*

**E1. `state` mode instead of `attitude` mode.** The race also accepts a 13-dim
full-state command that their onboard controller tracks at 500 Hz.
- *Might work:* their inner loop is 10× faster than your 50 Hz policy.
- *Might not:* you inherit their tuning and lose direct authority.
- *Test:* a config switch plus an output remap. Worth one afternoon.

**E2. Control rate.** The race env steps at 50 Hz. Check what rate your policy
was trained at, and whether it is being asked to run at a rate it never saw —
a real and easy-to-miss mismatch.

### F. Takeoff → *takeoff term*

Nobody optimises it, and it is inside the clock. Measure it (Lesson 3 🛠️ #2).
If it is 0.6 s of a 3.4 s lap, 30 % off it beats a heroic tracking gain.

## 4. What is already known (do not re-measure)

| finding | receipt |
|---|---|
| Gentle-reference training fails on fast references (0.95 m RMSE) | comment at `datt_env.py:110` |
| Asymmetric actor-critic (v5) beats the plain noisy-obs variant | parent project, 3 training seeds |
| Frame stacking (v6a) did **not** fix noise adaptation at 4 M steps | parent project reports |
| Single-seed policy conclusions invert; three seeds minimum | measured twice, both directions |
| A tracker can beat its own reference by corner-cutting | the 4.464 s speed-run analysis |
| Best-precision checkpoint ≠ most-robust checkpoint | measured twice, independently |

That last row deserves emphasis: the seed with the *best* precision was the
*worst* under disturbance, twice, in unrelated tests. **Select your final policy
against the leaderboard criterion** (mean time subject to ≥ 50 % success), not
against the prettiest training curve.

## 5. What a good final report contains

1. Your **decomposition** of lap time — where the time actually was.
2. A table of every experiment: hypothesis, threshold, verdict. **Including the
   failures.** A report with only successes is a report that hid something.
3. Your best number as **mean ± std over 3 seeds × 20 episodes**, beside the
   leaderboard number, with the protocol differences stated plainly.
4. One paragraph: **what you would do next, and why.** The most valuable thing
   you can hand the next student.

> 🎓 The point of this course is not to beat 3.394 s. It is that when somebody
> asks *"why is your controller faster?"* you have a measurement instead of a
> story. If you finish slower than the leaderboard but can prove **where** your
> time goes and **why** your idea did or did not help — you have done the
> assignment.

**Next:** [Lesson 5 — See the trajectory, compare the models](05-plot-and-compare.md)
gives you the plots (reference + speed profile, flown laps) and the comparison
harness to *show* what your experiments did.
