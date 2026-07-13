# Lesson 9 — Train a real tracker: give the drone a look-ahead

⏱️ ~1 hour (includes a training run). Open
[`k12_hover/trajectory_env.py`](../k12_hover/trajectory_env.py) and
[`k12_hover/train_traj_sb3.py`](../k12_hover/train_traj_sb3.py) alongside this.

> **The one big idea:** in Lesson 8 the drone could only see *where the target is
> now*. If we also let it see **where the path goes next**, it can steer into the
> turns *before* it reaches them — the way you look ahead down a road instead of
> staring at your own feet.

---

## 1. The look-ahead observation

We add a small block of numbers to the observation called `local_samples`: the
arrows to the **next few points on the path**, a fixed time apart.

```python
# trajectory_env.py — with n_samples > 0
idx = (steps + sample_offsets) % loop_len        # the next N points in time
future = trajectory[idx]                          # where the path will be
local_samples = future - drone_position           # arrows to those future points
```

Two knobs control it:

| Knob | Flag | Meaning |
|------|------|---------|
| how many points ahead | `--n-samples` | e.g. 10 points |
| how far apart in time | `--samples-dt` | e.g. 0.1 s → look 0 to ~0.9 s ahead |

So with `--n-samples 10 --samples-dt 0.1`, the drone sees the path for the next
~0.9 seconds. Now it can *anticipate* the curve instead of reacting late.

The observation grows from 13 numbers to `13 + 3 × n_samples`. Everything else —
the reward, the algorithm, the wrappers — is **identical** to hovering. We only
changed *what the drone sees*.

> This is the same design used by a full drone-racing RL project
> (`lsy_drone_racing`): sample the next N path points as **relative** future
> positions and feed them to the policy. We keep just that core idea and leave out
> its extras (frame-stacking, action-smoothness penalties) so the observation
> stays small and easy to read.

---

## 2. Train a tracker

Train a policy that *can* see ahead, on a circle too fast for the hover brain:

```bash
conda activate crazyflow
cd k12_RL_quad_traj
python -m k12_hover.train_traj_sb3 --jax-device gpu --num-envs 256 \
    --timesteps 3000000 --n-samples 10 --period 4 --save-name track_PPO
```

Watch `ep_rew_mean` climb, just like hovering. When it finishes, watch it fly
(**use the same `--n-samples` and `--period` you trained with**):

```bash
python -m k12_hover.eval_traj --model models/track_PPO.zip \
    --n-samples 10 --period 4 --render
```

---

## 3. The head-to-head (the payoff)

Put both brains on the **same fast circle** and compare tracking error:

```bash
# the reused hover brain (no look-ahead)
python -m k12_hover.eval_traj --model models/hover_PPO.zip --period 4
# the trained tracker (with look-ahead)
python -m k12_hover.eval_traj --model models/track_PPO.zip --n-samples 10 --period 4
```

The tracker should follow the curve noticeably tighter — it isn't cutting the
corners, because it can see them coming. **Same drone, same algorithm, same
reward — a better observation.**

> 💡 The lesson of Lessons 8 + 9 together: often the biggest wins in RL come not
> from a fancier algorithm, but from **giving the agent the right information**.

---

## 4. 🛠️ Experiments

1. **How far ahead is enough?** Train with `--n-samples 1`, `5`, and `10` (same
   `--period 4`). More look-ahead isn't always better — where do the gains stop?
2. **Reuse the hover knobs.** The reward knobs from Lesson 4 all still work here.
   Add `--velocity-coef 0.05` or `--tilt-action-coef 0.1` and see if the flight
   gets smoother (the tilt-action penalty also helps sim-to-real, see Lesson 7).
3. **A mini curriculum.** Train first on a slow circle (`--period 8`), then keep
   training the *same* model on a faster one (`--period 4`). Does starting easy and
   speeding up beat training on the fast circle from scratch?
4. **Sim-to-real.** Everything from Lesson 7 (export → deploy to CrazySim) applies
   to a tracker too. For deployment, train with `--physics first_principles`,
   `--yaw-coef`, and `--tilt-action-coef` set, exactly as for the deployable hover
   model.

## ✅ Check your understanding

1. What exactly is in the `local_samples` part of the observation?
2. Why must you pass the **same** `--n-samples` at eval time that you trained with?
3. The tracker and the hover brain use the same algorithm and reward. What is the
   one thing that makes the tracker better on a fast circle?
4. **Challenge:** how would you change `_build_trajectory` in `trajectory_env.py`
   to fly a figure-eight instead of a circle? (Hint: the reference
   `FigureEightEnv` uses `x = r·sin(t)`, `z = r/2·sin(2t) + 1`.)

🎉 You've gone from *hover* → *reuse for a circle* → *train a real path-tracker*.
That's the same ladder professional drone-RL projects climb.
