# Lesson 5 — Different brains: PPO vs A2C vs SAC

⏱️ ~40 minutes. You'll run the *same task* with *different learning algorithms*
and compare them.

---

## One task, many algorithms

So far we used **PPO**. But there are many RL algorithms — different "brains"
that learn in different ways. With SB3, switching is a **one-word change**:

```bash
python -m k12_hover.train_sb3 --algo PPO
python -m k12_hover.train_sb3 --algo A2C
python -m k12_hover.train_sb3 --algo SAC --num-envs 8
```

That's the beauty of using a library: the hard parts are reusable, and you get
to experiment with the *ideas*.

---

## The two big families

### On-policy (PPO, A2C) — "learn only from your latest practice"

* They fly, collect fresh experience, learn from it, then **throw it away** and
  fly again.
* They love **many parallel drones** (`--num-envs 128`) because they need lots
  of fresh experience each round.
* Steady and reliable. PPO is the popular default for drones and robots.

| Algorithm | One-line personality |
|-----------|----------------------|
| **A2C** | The simple, classic on-policy method. Fast but a bit noisy. |
| **PPO** | A2C's careful cousin: never changes the brain too much at once. Very stable. |

### Off-policy (SAC, TD3, DDPG) — "keep a memory and reuse it"

* They store past experience in a big **replay memory** and learn from it again
  and again. This makes them **sample-efficient** (good at learning from fewer
  flights).
* They are happiest with **fewer parallel drones** (`--num-envs 8`).

| Algorithm | One-line personality |
|-----------|----------------------|
| **DDPG** | The original "reuse a memory" method for continuous control. |
| **TD3** | DDPG made more stable (fixes its over-confidence). |
| **SAC** | Adds a bit of deliberate randomness to explore better. A strong choice. |

---

## 🛠️ The bake-off

Train three drones (keep timesteps equal for a fair race):

```bash
python -m k12_hover.train_sb3 --jax-device gpu --algo PPO --num-envs 128 \
    --timesteps 1000000 --tensorboard --save-name race_PPO

python -m k12_hover.train_sb3 --jax-device gpu --algo A2C --num-envs 128 \
    --timesteps 1000000 --tensorboard --save-name race_A2C

python -m k12_hover.train_sb3 --jax-device gpu --algo SAC --num-envs 8 \
    --timesteps 1000000 --tensorboard --save-name race_SAC
```

Then score each:

```bash
python -m k12_hover.eval_sb3 --model models/race_PPO.zip
python -m k12_hover.eval_sb3 --model models/race_A2C.zip
python -m k12_hover.eval_sb3 --model models/race_SAC.zip
```

And compare the learning curves together:

```bash
tensorboard --logdir models/tensorboard
```

---

## Make a results table

Fill this in with your own numbers:

| Algorithm | Final distance (cm) | Looked smooth? | Notes |
|-----------|--------------------:|----------------|-------|
| PPO       |                     |                |       |
| A2C       |                     |                |       |
| SAC       |                     |                |       |

---

## ✅ Questions to discuss

1. Which algorithm reached a good hover **fastest** (fewest timesteps)?
2. Which gave the **smoothest** flight in the 3D viewer?
3. SAC used only 8 drones but PPO used 128. Why is that a *fair* comparison if
   we keep `--timesteps` the same? (Hint: timesteps counts total experience,
   not number of drones.)
4. There is no single "best" algorithm — it depends on the task. Based on your
   table, which would *you* pick for hovering, and why?

Next (optional, advanced): **Lesson 6** — open the hood and see PPO's actual code.
