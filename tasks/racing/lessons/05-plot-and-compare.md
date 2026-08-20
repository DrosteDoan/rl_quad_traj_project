# Lesson 5 — See the trajectory, compare the models

⏱️ ~60 minutes. **You finish when** you have (a) a track map and speed profile
of your reference, (b) a flown lap drawn over it, and (c) a head-to-head table
of at least two policies on the LSY protocol.

> **The one big idea:** a trajectory is not a line, it is a **schedule** —
> where to be *and how fast to be going*. Two references can draw the same
> curve on paper and demand completely different thrust. Until you plot the
> speed profile next to the path, you are reasoning about half of your plan.
> And the same discipline applies to results: a lap time without its success
> rate, or a comparison without identical protocols, is half a number.

**Prerequisites:** Lesson 2 (you have trained policies) for §4; Lesson 3 (your
completed `_build_reference`) for §3–§4. §2 works immediately.

---

## 1. What a "speed profile" is, and why racers stare at it

Your reference `FreestyleTrajectory` answers three questions at every moment
`t`: `pos(t)`, `vel(t)`, `acc(t)`. The **speed profile** is `|vel(t)|` plotted
against time. It shows, at a glance:

- **where the time goes** — long slow valleys between gates cost more than a
  slightly lower top speed ever saves;
- **what the turns cost** — every hairpin is a dip; the *depth × width* of the
  dip is the price of that corner;
- **what physics will refuse** — thrust demand is `|acc + g|`, and the drone
  can produce at most `TWR·g ≈ 18.4 m/s²`. The reference builder time-scales
  each segment to stay under `0.95·TWR·g` (`freestyle.py`, `SEG_ACC_LIMIT`),
  so *asking* for a faster lap does not always *get* one — the builder quietly
  stretches the segment your cruise setting overreached on.

That last point is why "just raise `cruise`" stops working at some point: you
are no longer choosing the speed, the feasibility limit is.

## 2. Draw your reference (works right now)

In the **main** venv, from `/workspace`:

```bash
python tasks/racing/code/plot_trajectory.py --track lsy-level2-race --cruise 3.0
```

Two PNGs land in `tasks/racing/figures/`:

- **`..._map.png`** — top-down track, the path colored by speed (dark = slow,
  bright = fast), gates drawn with their openings and fly-through normals,
  obstacle poles as ×.
- **`..._profile.png`** — three panels on one time axis: speed, altitude, and
  thrust demand against the feasibility limit, with every gate crossing marked.

It also prints `feasibility_report(...)` — the same check your
`_build_reference` runs (Lesson 3 §4; remember `feasible` reads `False` on any
ground-start reference — check the thrust and gate components).

**Now do the experiment that makes the point.** Run it again at
`--cruise 1.5`, `3.0`, `4.5`, and put the three profile plots side by side:

1. Where does the extra cruise actually shorten the lap — and where does the
   profile barely change because the time-scaler refused?
2. Find the deepest speed valley. Which gate pair causes it? (Compare with the
   hairpin comment inside `lsy_level2_race()` —
   `tasks/racing/crazy_track/src/crazy_track/trajectories/freestyle.py:409`.)
3. At which cruise does `feasible` first go `False`, and which number in the
   report tripped?

> 🔑 This is the cheapest tuning loop in the whole course. A plot costs two
> seconds; a 20-episode evaluation costs minutes; a retrain costs an hour.
> **Exhaust the cheap loop first.**

## 3. Draw what the drone actually flew

The scaffold `race_bridge.py` has a flight recorder built in: set
`RACE_LOG_DIR` and every episode writes `flown_ep<N>.csv` (`t,x,y,z`). With
your completed bridge installed (Lesson 3 §4):

```bash
cd repos/lsy_drone_racing
RACE_LOG_DIR=/workspace/tasks/racing/figures/mylap \
  /opt/venvs/race/bin/python scripts/sim.py --config level0.toml
```

Then overlay it on the reference:

```bash
cd /workspace
python tasks/racing/code/plot_trajectory.py --track lsy-level2-race --cruise 3.0 \
    --flown tasks/racing/figures/mylap/flown_ep00.csv
```

The dashed grey line is reality. Read it like a coach:

- **Flown inside the reference on corners** → the tracker cuts corners. The
  parent project measured a policy *beating its own reference* this way. If
  the gates still pass, this is free speed — but it means your remaining lap
  time lives in the *plan*, not the tracker.
- **Flown lagging behind on straights** (same shape, arriving late) → the
  tracker can't hold the commanded speed; look at the thrust panel — you are
  probably near the feasibility ceiling.
- **Divergence right at takeoff** → your reference's ground start or takeoff
  leg is wrong (Lesson 3 §4 point 1). This is the most common first bug, and
  it is invisible in the lap time alone.

## 4. Compare models the honest way

You have three seeds from Lesson 2 (if not, go back — one seed is a coin
flip). Race them against each other on the exact leaderboard protocol:

```bash
/opt/venvs/race/bin/python tasks/racing/code/compare_models.py \
    --episodes 20 --config level0.toml \
    v5_s0=tasks/racing/crazy_track/results/<run-a>/datt_ppo_final.zip \
    v5_s1=tasks/racing/crazy_track/results/<run-b>/datt_ppo_final.zip \
    v5_s2=tasks/racing/crazy_track/results/<run-c>/datt_ppo_final.zip
```

Three design decisions in that script are the lesson — open it and find them:

1. **It calls lsy's own `simulate()`** (`from sim import simulate`), never a
   re-implementation. The moment you write your own loop, your numbers stop
   being comparable to the leaderboard, silently.
2. **Every model gets the identical config and episode count.** Comparing
   20 episodes of one model against 5 of another is how you fool yourself
   politely.
3. **The plot refuses to separate lap time from success rate** — dots and mean
   on the left, success on the right, the 50 % ranking cutoff drawn in. A
   3.2 s mean at 40 % success is not "almost leaderboard", it is *unranked*.

You get `comparison.png`, `comparison.csv`, and per-model flown-lap CSVs
(feed those back into §3 to see *why* the slow seed is slow).

### Reading the result

- The seed-to-seed spread you see here **is your noise floor** (Lesson 2 §5).
  When Lesson 4's experiment "improves" the mean by less than that spread,
  it improved nothing.
- Comparing a v5 against a v2 policy (Lesson 2's exercise 3)? Same command,
  two more `label=path` arguments. One variable at a time still applies — the
  two policies must differ in exactly the thing you claim to be testing.
- Put your best row **next to** 3.394 s / 3.419 s in your report, with the
  success rates. The gap is not embarrassing; an unexplained gap is.

---

## 🛠️ Before you move on

1. **The cruise sweep** (§2, three cruises). One paragraph: where does
   `lsy_level2_race` stop obeying your cruise setting and why?
2. **Diagnose one seed.** Take your slowest seed, overlay its flown lap on the
   reference, and name the failure mode from §3's list. Attach the plot.
3. **The comparison table** for your three seeds, pasted into your report,
   with one sentence stating whether the seeds agree within their own spread.
4. **Break it on purpose.** Run `compare_models.py` with `--episodes 3` and
   watch how far the 3-episode means scatter compared to your 20-episode
   ones. That scatter is what "n=3 episodes" buys — remember it whenever a
   paper (or a classmate) quotes one.

**You are done with the course.** What remains is Lesson 4's loop: hypothesis,
threshold, verdict — now with the plots to show what happened and the
comparison harness to prove it.
