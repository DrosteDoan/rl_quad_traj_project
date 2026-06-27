# Lesson 3 — Train your first hovering drone

⏱️ ~30 minutes (plus a few minutes of training time). This is the fun one. 🎉

We use **Stable-Baselines3 (SB3)**, a popular RL library that hides the hard
math so you can focus on the ideas. The same library is used in many drone
tutorials.

---

## Step 1 — Train

```bash
conda activate crazyflow
cd k12_RL_quad_traj
python -m k12_hover.train_sb3 --jax-device gpu --num-envs 128 --timesteps 2000000
```

No GPU? Use this instead (slower, still works):

```bash
python -m k12_hover.train_sb3 --num-envs 32 --timesteps 500000
```

That's it. The three lines that actually do the learning, hidden inside
`train_sb3.py`, are:

```python
env   = make_sb3_env(num_envs=128)     # build the drones + the hover task
model = PPO("MlpPolicy", env)          # create the brain (a neural network)
model.learn(total_timesteps=2_000_000) # practice 2 million steps
```

When it finishes, the trained brain is saved to `models/hover_PPO.zip`.

---

## Step 2 — Read the scoreboard

While training, SB3 prints tables. The single most important number is
**`ep_rew_mean`** — the average reward per episode. **You want it to go UP.**

```
| rollout/           |          |
|    ep_len_mean     | 210      |   ← episodes last longer (drone survives!)
|    ep_rew_mean     | 165      |   ← reward going UP = learning is working
| time/              |          |
|    fps             | 13000    |   ← steps per second (speed of training)
|    total_timesteps | 500000   |   ← how far along we are
```

A healthy training run looks like: `ep_rew_mean` starts low (single digits),
then climbs and flattens out. `ep_len_mean` grows toward the maximum (the drone
stops crashing early).

Some other rows, in plain language:

| Row | Meaning |
|-----|---------|
| `ep_len_mean` | how many steps the drone survives per episode |
| `explained_variance` | how well the "critic" predicts rewards (closer to 1 is better) |
| `entropy_loss` | how random the actions still are (shrinks as the agent gets confident) |
| `value_loss` | how wrong the critic's guesses are (usually shrinks) |

You don't need to understand all of them yet — just watch `ep_rew_mean` rise.

---

## Step 3 — Watch and score your drone

Score it over a few test flights (works on any machine, no graphics):

```bash
python -m k12_hover.eval_sb3 --model models/hover_PPO.zip
```

You'll see something like:

```
  Flight 1: reward =  220.50 | ended  3.1 cm from target
  ...
Average distance : 4.0 cm from target
Verdict          : great hover!
```

**Under ~10 cm from the target is a great hover.**

See it fly in 3D (needs a screen / display):

```bash
python -m k12_hover.eval_sb3 --model models/hover_PPO.zip --render
```

The red ball is the target. Watch the drone fly to it and hold still.

---

## What just happened? (PPO in 4 sentences)

1. The drone flew around using its current (at first, random) brain and recorded
   what it saw, did, and was rewarded.
2. A part of the brain called the **critic** learned to predict how much reward
   to expect from each situation.
3. Actions that did **better than expected** were made **more likely**; actions
   that did worse were made less likely.
4. Repeat two million times. That algorithm is **PPO** (Proximal Policy
   Optimization). The "proximal" part means it never changes the brain *too*
   much in one step, which keeps learning stable.

---

## 🛠️ Experiments

1. **Watch it learn live.** Run training and keep your eye on `ep_rew_mean` for
   the first minute. Roughly how many steps until it clearly starts rising?
2. **Train longer / shorter.** Try `--timesteps 200000` then `--timesteps 3000000`.
   How does the final "distance from target" in evaluation change?
3. **Fewer drones.** Train with `--num-envs 8` and the same timesteps. Is
   learning more or less steady? (Fewer drones = noisier experience.)

Next: **Lesson 4** — change the reward and change how the drone flies.
