# Lesson 2b — Where reference trajectories come from

⏱️ ~45 minutes of reading and looking at code. **You finish when** you can
explain how this course's references are generated, why they are smooth in
exactly the way they are, and name at least two other ways the field generates
racing trajectories.

> **The one big idea:** your policy is a *tracker* — it can only fly the plan
> it is handed, and it can never be better than that plan. So before you race
> (Lesson 3), you should understand where the plan comes from. You will **use**
> a generator, not write one; but using a tool you can't explain is how you end
> up optimizing the wrong half of the system.

**No new code in this lesson.** Everything here is read-and-verify, with the
plotting tool from Lesson 5 §2 available whenever you want to *see* what a
paragraph claims.

---

## 1. A trajectory is a function, not a drawing

The race needs, at every control tick, an answer to "where should I be, how
fast, accelerating how hard?" Formally the reference is three functions of
time:

```
pos(t), vel(t), acc(t)      for t in [0, duration]
```

Everything downstream consumes exactly this interface
(`tasks/racing/crazy_track/src/crazy_track/trajectories/base.py`): the DATT
policy samples its look-ahead window from it, `plot_trajectory.py` draws it,
and `feasibility_report` audits it.

Why does `acc(t)` matter so much for a quadrotor? Because of a beautiful
physical fact (the core of *differential flatness*): a quadrotor's thrust
vector must equal `m·(acc + g)`. **The acceleration of the path dictates the
attitude of the drone.** A path whose acceleration jumps demands an attitude
that jumps — which a physical vehicle cannot do. That single fact explains
most design choices below:

- Continuous `acc(t)` (called **C² continuity**) → continuous attitude
  reference → trackable.
- `|acc + g|` bounded by what the rotors can produce → the **feasibility
  limit** you keep meeting: thrust-to-weight 1.88 means no reference may
  demand more than `TWR·g ≈ 18.4 m/s²` of thrust acceleration, ever.

## 2. How THIS course generates references (read the real code)

Open `tasks/racing/crazy_track/src/crazy_track/trajectories/freestyle.py`.
The generator is **closed-form** — plain NumPy, no solver, no optimizer — and
it works in three steps.

### a) Describe the track as *ops* (`FreestyleTrajectory`, line 171)

You declare what the flight should do, in order:

```python
ops=[
    ("gate", g1, speed),     # cross the gate center along its normal, at speed
    ("via",  point, vel),    # free waypoint with a velocity (shape the racing line)
    ("hover", point, T),     # come to rest
    ("flip", ...),           # (freestyle only — not used in racing)
]
```

This is a tiny *domain language* for tracks. `lsy_level2_race()` (line 409) is
just such an ops list with the LSY gate poses pasted in from their
`config/level2.toml` — which is why the track ports 1:1 into the race
environment, where your bridge builds the gates from `obs["gates_pos"]`
instead. Its `("via", ...)` entries (lines 424 and 427) are the two swing-out
points of the racing line, and that list is exactly what you copy into your
bridge's `_build_reference` in Lesson 3 §4 and then edit.

### b) Connect the ops with quintic polynomials (line 209)

Between each pair of boundary states the generator fits, **per axis**, a
5th-order (quintic) polynomial — the lowest degree that can match position,
velocity *and* acceleration at both ends (`_quintic`,
`trajectories/chained_poly.py:88`; six boundary conditions, six coefficients).
Matching all three is what buys C² continuity across the whole chain — see §1
for why that is not a luxury.

> 🔁 **You have seen this before.** Lesson 1 §5: the *training* environment
> generates its random practice paths from the same C² chained-polynomial
> family (`datt_env.py:109` → `chained_poly.py`). Deployment references being
> drawn from the same smoothness family the policy trained on is not a
> coincidence — it is the contract that makes zero-shot tracking of a
> never-seen track work at all.

### c) Choose each segment's duration, then *time-scale* it to feasibility (lines 209–227)

How long should a segment take? The generator seeds a guess:

```
T = max( distance / cruise,          # fly at your cruise speed
         min_seg_T,                  # never teleport through short legs
         Δv / turn_budget )          # sharp direction changes need time
```

then applies the classical **time-scaling trick**: sample the quintic's
analytic peak thrust demand `max |acc + g|`; while it exceeds
`SEG_ACC_LIMIT = 17.2 m/s²` (line 37 — just under `0.95·TWR·g`), stretch
`T *= 1.15` and refit. Slowing a path down always reduces its accelerations,
so this converges — and it means **the generator quietly refuses cruise
settings physics can't cash**. You measured exactly this in Lesson 5 §2's
cruise sweep: past some cruise, the profile stops changing because the
time-scaler, not your argument, is choosing the speed.

### d) Audit the result (`feasibility_report`, line 446)

Before anything flies, the whole reference is checked analytically: peak
thrust demand vs the limit, minimum altitude, and — subtle and important —
that every crossing of a gate's *plane* goes either through the opening (with
margin) or fully clear of the frame. Recall from Lesson 3 §4 the one flag that
lies to racers: `feasible` also demands `min_z > 0.15 m`, which a mandatory
ground start can never satisfy — judge the components.

**That's the whole generator.** Declarative ops → quintic chaining → duration
law + time-scaling → analytic audit. Simple enough to read in one sitting,
strong enough to carry a 100 %-success race lap.

## 3. What the rest of the field does (the landscape)

Our closed-form chaining is one point in a well-mapped design space. You don't
need to implement any of these — you need to know they exist, what they
optimize, and what they cost, so that "improve the plan" (Lesson 4) is a
concrete option rather than a mystery.

| approach | idea in one line | what you gain / pay |
|---|---|---|
| **Quintic / spline chaining** (this course) | fix waypoint states, fit smooth polynomials, time-scale to feasibility | closed-form, instant, auditable / nowhere near time-optimal |
| **Minimum-snap** (Mellinger & Kumar, ICRA 2011) | optimize polynomial coefficients to minimize snap (4th derivative) through waypoints | the classic; smooth by construction, small QP / time allocation still heuristic |
| **MINCO / GCOPTER** (Wang et al., T-RO 2022, [arXiv:2103.00190](https://arxiv.org/abs/2103.00190)) | jointly optimize the *spatial shape* and the *time allocation*, constraints included | today's standard for aggressive flight; big lap-time gains / a real optimizer to tune, minutes not milliseconds |
| **Time-optimal CPC** (Foehn, Romero & Scaramuzza, *Science Robotics* 2021) | full-state optimal control: find the provably fastest trajectory through the gates | beat human champions' lap times / hours of offline compute per track |
| **Time-parameterization** (TOPP-RA — Pham & Pham, 2018) | keep a fixed geometric path, compute the fastest feasible speed profile along it | decouples "where" from "how fast"; our time-scaling is a crude cousin of this |
| **Online jerk-limited** (Ruckig — Berscheid & Kröger, 2021) | recompute a feasible trajectory to a moving target every millisecond | reactive, real-time / short-horizon, not a racing-line planner |
| **Learn the plan too** (Swift — Kaufmann et al., *Nature* 2023) | end-to-end RL: one network does planning *and* tracking from perception | champion-level racing / you lose the clean plan/track separation this course is built on — and with it, the ability to say *which* half failed |

Two things to notice in that table:

1. **The gap between row 1 and rows 3–4 is exactly your lap-time gap.** This
   course's verified baseline (7.80 s at 100 % success) tracks a *closed-form*
   reference; the 3.39 s leaderboard entries fly *optimized* ones. Your
   tracker is not 2× worse than theirs — your **plan** is slower, by design,
   and Lessons 3 §6 / 5 §3 gave you the measurements to prove it.
2. **The last row is the road not taken.** Swift shows you *can* collapse
   planning and tracking into one policy. This course deliberately keeps them
   separate — because when a combined system is slow, you cannot ask it which
   half to fix.

## 4. So can *you* swap in a better generator?

Yes — and it is a legitimate, well-scoped Lesson-4 experiment. Anything that
produces `pos(t) / vel(t) / acc(t)` (subclass `Trajectory`, or just wrap your
arrays) plugs into the same bridge, the same policy, the same protocol. Keep
the tracker fixed and change only the reference: that is one variable, and
`compare_models.py`'s protocol plus `plot_trajectory.py`'s overlay will tell
you honestly what your new plan bought — and whether the policy can still
track it (a time-optimal reference at this TWR is *hard*; expect your success
rate to object).

---

## 🛠️ Before you move on

1. **The attitude argument, in your own words.** Why must `acc(t)` be
   continuous for a quadrotor, and what — physically — would the drone be
   asked to do across a reference whose acceleration steps? (§1 has the
   pieces; one paragraph.)
2. **Find the refusal.** Read `connect()` (freestyle.py:209–227) and state in
   one sentence why raising `cruise` eventually stops shortening the lap.
   Then confirm it with the Lesson 5 §2 cruise sweep if you have not already.
3. **Read one landscape paper's abstract** (pick from §3) and answer: what
   does it optimize that our generator does not?
4. **Design on paper only:** if you had the time-optimal reference for this
   track, which of its properties would break *your current policy* first —
   and which lesson's tool would show it?

**Next:** [Lesson 3 — Evaluate like the race](03-evaluate-like-the-race.md),
where somebody hands the plan to your tracker with a stopwatch running.
