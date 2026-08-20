# Lesson 3 — Evaluate like the race

⏱️ ~90 minutes. **You finish when** you have a lap time on the LSY protocol and
can say honestly how it compares to the leaderboard.

> **The one big idea:** a number is meaningless until you know exactly how it
> was measured. Most of this lesson is not code — it is finding out what the
> leaderboard actually scores and making your evaluation match. Teams lose
> semesters to numbers that were never comparable.

---

## 1. Which level is *our* problem? (the scoping lesson)

The race has four levels. **Not all of them are tracking problems.**

| level | what is randomised | a **tracking** problem? |
|---|---|---|
| **0** | nothing (action/dynamics noise only) | ✅ **yes** — track perfectly known, so the path can be fixed offline |
| **1** | mass, inertia | ✅ **yes** — path still known; the *drone* changed and the controller must adapt |
| 2 | + gate & obstacle positions | ❌ no — the path must be **re-planned** when a gate moves |
| 3 | + full track layout | ❌ no — the path must be **planned online** |

**We work on Levels 0 and 1.** That is not dodging the hard part; it is the
boundary of the method. A pure tracker cannot repair a path that has become
wrong.

Look at the randomisation yourself —
`repos/lsy_drone_racing/config/level1.toml` (and `level2.toml` for contrast):

```toml
[env.randomizations.drone_mass]
fn = "uniform"
[env.randomizations.drone_mass.kwargs]
minval = -0.005
maxval = 0.005
```

±0.005 kg against a **43.38 g** airframe is **±11.5 %** of the drone's mass —
which directly changes the thrust needed to hover. A hand-tuned controller must
be retuned for it. Your DATT policy, trained with domain randomisation and
carrying an on-line disturbance estimate (the `l1.sigma_f` block from Lesson 1
§2), should absorb it. **That is now a claim you can test rather than believe.**

> 💡 Want Levels 2–3 later? The honest architecture is *your tracker + somebody's
> planner*. Keeping that interface clean is exactly why this course separates
> them.

## 2. Install the race environment

You already cloned it. Confirm both environments are healthy:

```bash
bash scripts/smoke_test.sh
```

The last three checks use the `race` env. If they fail, re-run
`bash scripts/setup_python_envs.sh` and read its header — it explains why the
race lives in a second environment and how the bridge still reaches both.

## 3. What the leaderboard actually measures

Read `repos/lsy_drone_racing/scripts/evaluate.py`. Four facts change how you work:

1. **20 episodes**, scored on the **mean time over the successful ones**.
2. **You must succeed in ≥ 10 of 20.** Below 50 % you are not ranked at all.
   Speed and risk are one coupled decision, not two.
3. **The clock starts with the drone on the ground** (`z = 0.01 m`) and stops at
   the last gate. **Takeoff is inside your lap time.**
4. **Only `state` and `attitude` control modes are legal.** Your DATT policy
   emits attitude commands, so it is legal as-is.

### ⚠️ A documentation trap — and how not to get caught

`repos/lsy_drone_racing/lsy_drone_racing/control/controller.py:63` says the
attitude command is:

```
A drone state command [x, y, z, vx, vy, vz, ax, ay, az, yaw, rrate, prate, yrate] in
absolute coordinates or an attitude command [thrust, roll, pitch, yaw] as a numpy array.
```

**The code disagrees with that docstring.** Their action space bounds the first
three entries by ±π/2 (angles) and the fourth by thrust limits
(`race_core.py:237`), and their own example controller ends with
(`control/attitude_controller.py:130`):

```python
        action = np.concatenate([euler_desired, [thrust_desired]], dtype=np.float32)
```

`euler_desired` first, thrust last. So the real order is
**`[roll, pitch, yaw, thrust]`** — which happens to be exactly what DATT already
outputs (Lesson 1 §3).

Do not take this on trust from *me* either. Verify:

```bash
/opt/venvs/race/bin/python -c "
from lsy_drone_racing.envs.race_core import build_action_space
s = build_action_space('attitude', 'cf21B_500'); print('low ', s.low); print('high', s.high)"
```

If the fourth entry is not bounded by ±π/2, the fourth entry is thrust.

> 🔑 **When docs and code disagree, the code is what runs.** Get this backwards
> and you command 0.4 N of roll and π/2 newtons of thrust: the drone flips
> instantly and you spend a day blaming your policy.

## 4. Build the bridge

Copy the scaffold into their control directory (their loader requires it to live
there, with exactly one `Controller` subclass per file):

```bash
cp tasks/racing/code/race_bridge.py repos/lsy_drone_racing/lsy_drone_racing/control/
```

Open it. Everything is written **except `_build_reference`**, which is your job
because it is the one real design decision. It must:

- **Start on the ground.** `obs["pos"]` at reset is the true start, `z ≈ 0.01 m`.
  The reference must start *there* and climb. Start it at hover height and step
  one asks the policy to teleport — you get a violent first second and lose the
  time anyway.
- **Route through the gates in order**, entering along each gate's normal.
  `lsy_level2_race()` in
  `tasks/racing/crazy_track/src/crazy_track/trajectories/freestyle.py` already routes
  these four gates (Level 0 and Level 2 share the same nominal gate poses);
  start there and prepend the takeoff.
- **Be feasible.** Run `feasibility_report(traj)` before you fly it and require
  the thrust demand under the limit and `gate_crossings_ok`. TWR is 1.88 — a
  reference demanding more acceleration than that is untrackable by *any*
  controller, and training longer will not help. (One subtlety you should
  notice rather than trip over: the combined `feasible` flag also requires
  `min_z > 0.15 m`, and your reference *must* start at `z ≈ 0.01 m` — so on a
  race reference that flag is always `False`. Check the components, and judge
  the takeoff leg by eye. When a tool's summary bit and your requirements
  disagree, read what the bit is actually made of — same habit as §3.)

Then point their config at it — `repos/lsy_drone_racing/config/level0.toml`:

```toml
[controller]
file = "race_bridge.py"

[env]
control_mode = "attitude"
```

Fly one episode and watch:

```bash
cd repos/lsy_drone_racing && /opt/venvs/race/bin/python scripts/sim.py --config level0.toml
```

## 5. Score it properly

```bash
cd repos/lsy_drone_racing && /opt/venvs/race/bin/python scripts/evaluate.py
```

⚠️ **Open `scripts/evaluate.py` before you trust it**: `config_file` is
hard-coded to `"level2.toml"` — a *planning* level (§1). For your Level-0/1
number, either change that line locally, or use Lesson 5's
`compare_models.py`, which runs the same 20-episode protocol through their
`simulate()` with the config you choose. (Same lesson as §3: read what the
script actually does, not what its name suggests.)

Their 20 episodes, printing mean successful time and success rate — the two
numbers that define your leaderboard position. Record every attempt:

| policy (seed) | level | mean time | success | notes |
|---|---|---|---|---|
| v5 s0 | 0 | ? | ?/20 | first legal number |

Repeat for seeds 1 and 2, and for Level 1. A single seed is a coin flip
(Lesson 2 §5), and you now have 20 random episodes stacked on top of the
training-seed variance — report both.

## 6. Read your result honestly

Against 3.394 s (all-time) and 3.419 s (current). Your first number will very
likely be slower, and the useful question is *which term is slow*:

```
lap time  =  takeoff  +  time along the path  +  time lost tracking badly
```

Instrument all three: the time the last gate is passed, the time gate 1 is
reached, and the RMSE between drone and reference. Then:

- **Tracking error small, lap slow** → your *reference* is the slow part. This
  is the expected outcome: the parent project measured a tracker that beat its
  own reference by corner-cutting. Since this course does not optimise the plan,
  this is a legitimate place to stop and **report a bound**.
- **Tracking error large** → the tracker is the problem, and Lesson 4 is for you.
- **Success below 50 %** → you are past the risk limit. Slow the reference until
  you clear it; an unranked fast lap scores nothing.

---

## 🛠️ Before you move on

1. **Verify the action order yourself** with the command in §3 and write down
   what you found. Never take an interface on faith again.
2. **Measure the takeoff tax.** Time from launch to gate 1. What fraction of a
   3.4 s lap is it? Is optimising it worth more than the same effort spent on
   tracking?
3. **Level 0 vs Level 1, same policy.** Does your policy lose time when mass is
   randomised? If it barely notices, you have just measured the value of domain
   randomisation — a result worth writing down.
4. **Break it on purpose.** Swap the action order to `[thrust, roll, pitch, yaw]`
   and watch one episode. Knowing what a wrong interface *looks like* will save
   you an afternoon someday.

**Next:** [Lesson 4 — Brainstorm: make it faster](04-brainstorm-faster-tracking.md)
