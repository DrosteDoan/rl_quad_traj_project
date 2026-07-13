# Lesson 8 — Follow a circle with the brain you already have

⏱️ ~30 minutes. No new training! Open
[`k12_hover/trajectory_env.py`](../k12_hover/trajectory_env.py) alongside this.

> **The one big idea:** a hover is a target that stands still. A *trajectory* is
> the **same target, sliding along a path**. So a drone that can hover can already
> chase a slowly-moving point — with **zero retraining**.

---

## 1. Why this works

Remember what your hover brain actually sees (Lesson 2). The first three numbers
of its observation are `rel_pos` — *the arrow pointing from the drone to the
target*. The brain learned one job: **"make that arrow shrink to zero."**

The brain never knew whether the target was standing still. It just chased the
arrow. So if we take the target and start walking it around a circle, the brain
keeps doing the only thing it knows: shrink the arrow, i.e. **catch up to wherever
the target is now.** That is trajectory following, for free.

`TrajectoryEnv` does exactly this. Instead of one fixed `target_pos`, it builds a
whole **circle of points** and, every step, hands the drone the arrow to *the point
the target has reached by now*:

```python
# trajectory_env.py (simplified)
theta = linspace(0, 2*pi, loop_len)          # angles around the circle
x = center_x + radius*cos(theta)             # the circle, as a table of points
y = center_y + radius*sin(theta)
# ...each step, pick the point for the current time and give the drone the arrow:
rel_pos = current_target - drone_position
```

`n_samples=0` means "no look-ahead" — the observation is the **exact same 13
numbers** the hover brain already understands. That is the whole trick.

---

## 2. Watch your hover brain fly a circle

Use the hover model you trained back in Lesson 3 (`models/hover_PPO.zip`):

```bash
conda activate crazyflow
cd k12_RL_quad_traj
# A gentle circle: 0.5 m radius, one lap every 8 seconds.
python -m k12_hover.eval_traj --model models/hover_PPO.zip --period 8 --render
```

You should see the drone loop around, chasing the **red ball** as it slides along
the grey circle. It works — and you didn't train anything new!

The script prints a **tracking error**: how far (in cm) the drone was from the
moving target. On a gentle circle a hover brain typically stays within ~10–15 cm.

---

## 3. Now break it (on purpose)

Turn the circle **faster** by making one lap take less time:

```bash
python -m k12_hover.eval_traj --model models/hover_PPO.zip --period 3     # fast
python -m k12_hover.eval_traj --model models/hover_PPO.zip --period 2     # faster!
```

Watch what happens: the drone starts to **lag behind** and **cut across the
corners** of the circle instead of following the curve. The tracking error grows.

Why? The hover brain only ever sees *where the target is right now*. It has **no
idea where the path goes next**, so it's always reacting a step late — like trying
to follow a friend by only ever looking at their footprints, never ahead at where
they're walking. At walking speed you keep up. At a run, you fall behind and start
short-cutting the turns.

> 💡 This is a real limitation, not a bug. It's the reason we'll build a smarter
> observation in Lesson 9.

---

## 4. 🛠️ Experiments

1. **Find the breaking point.** Start at `--period 8` and step down (7, 6, 5, …).
   At what lap time does the average error jump above ~20 cm? That's roughly where
   "just chase the point" stops being good enough.
2. **Bigger circle.** Try `--radius 1.0`. Does a bigger circle at the same period
   make tracking easier or harder? (Hint: bigger circle = the target moves faster.)
3. **A steadier brain.** If you trained `hover_steady.zip` in Lesson 4 (with a
   velocity penalty), compare it against `hover_PPO.zip` on the same circle. Does
   "holds still well" also mean "follows a moving target well"? Why might it not?

## ✅ Check your understanding

1. Why can a hover policy follow a circle without any retraining?
2. In your own words, why does it start cutting corners when the circle speeds up?
3. What single piece of information would help it stop cutting corners? (You're
   about to add exactly that.)

Next: **Lesson 9** — give the drone a look-ahead and train a real tracker.
