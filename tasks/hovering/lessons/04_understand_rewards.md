# Lesson 4 — The reward is the steering wheel

⏱️ ~40 minutes (includes a couple of short training runs).

> **The most important idea in RL:** the agent does *exactly* what earns
> reward — not what you *meant*. If the behavior is wrong, the reward is wrong.
> This is called **reward shaping**, and it is where most of the real work is.

---

## A cautionary tale

Our default reward only says "be close to the target":

```python
reward = exp(-distance)
```

A drone can earn lots of this reward by **zooming through** the target over and
over, like a kid running past the cookie jar grabbing one each lap. It's near
the target a lot... but it never *stops*. That's not hovering!

To get a real hover, we must also reward **holding still**. That's what the
extra knobs are for.

---

## The reward knobs

In `hover_env.py`, the reward is built from pieces, each with a coefficient (a
knob). You can set them from the command line:

| Knob | Flag | What turning it UP does |
|------|------|-------------------------|
| distance | `--distance-coef` | care more about being close |
| velocity | `--velocity-coef` | punish moving fast → **hold still** |
| spin | `--spin-coef` | punish spinning → stay calm |
| tilt | `--tilt-coef` | punish tipping → stay upright |

All penalties default to **0** (off), so by default only distance matters.

---

## Experiment A — Make it actually hover

First, train the "distance only" version and watch it:

```bash
python -m k12_hover.train_sb3 --jax-device gpu --timesteps 1000000 --save-name hover_distance_only
python -m k12_hover.eval_sb3 --model models/hover_distance_only.zip --render
```

Notice it may **drift or wobble** around the target instead of parking on it.

Now add a "hold still" penalty and train again:

```bash
python -m k12_hover.train_sb3 --jax-device gpu --timesteps 1000000 \
    --velocity-coef 0.1 --spin-coef 0.02 --save-name hover_steady
python -m k12_hover.eval_sb3 --model models/hover_steady.zip --render
```

Compare the two. The second should sit **much more calmly** on the target.

> 💡 You changed the drone's *personality* without touching the learning
> algorithm at all — only the reward.

---

## Experiment B — Push a knob too far

Set the velocity penalty very high:

```bash
python -m k12_hover.train_sb3 --jax-device gpu --timesteps 1000000 \
    --velocity-coef 2.0 --save-name hover_lazy
python -m k12_hover.eval_sb3 --model models/hover_lazy.zip
```

What might go wrong? If "moving is punished" outweighs "reach the target", the
laziest way to avoid the speed penalty is to **barely move** — even if that
means never closing the last gap to the target. This is **reward hacking**: the
agent satisfies your rules in a way you didn't intend.

The art of reward shaping is **balance**: enough "go to target" to make it fly
there, enough "hold still" to make it stop, but not so much that it freezes.

---

## How to compare runs fairly (TensorBoard)

Add `--tensorboard` to any training command, then open the graphs:

```bash
python -m k12_hover.train_sb3 --tensorboard --velocity-coef 0.1 --save-name run_a
python -m k12_hover.train_sb3 --tensorboard --velocity-coef 0.5 --save-name run_b
tensorboard --logdir models/tensorboard
```

Open the link it prints (usually http://localhost:6006) and compare the
`rollout/ep_rew_mean` curves side by side.

---

## ✅ Check your understanding

1. Why doesn't "reward = close to target" produce a calm hover by itself?
2. What is *reward hacking*? Give the example from Experiment B.
3. You want the drone to also point its nose north. Which part of the
   observation would the reward need to look at, and what would you add to
   `_reward` in `hover_env.py`?

Next: **Lesson 5** — swap the brain: PPO vs A2C vs SAC.
