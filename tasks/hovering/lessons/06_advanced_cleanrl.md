# Lesson 6 — Looking inside PPO (advanced)

⏱️ ~1 hour. This lesson is for students who finished Lessons 1–5 and want to see
what `model.learn()` actually *does*. It involves real (but readable) code.

> You do **not** need this lesson to train good drones. SB3 is enough. This is
> for the curious who want to peek under the hood.

Open [`k12_hover/train_cleanrl.py`](../k12_hover/train_cleanrl.py) and read it
top to bottom with this lesson.

---

## Why a second script?

SB3 hides PPO inside one line: `model.learn()`. That's great for getting work
done, but you can't *see* the algorithm. `train_cleanrl.py` is the **same PPO**,
written out by hand in ~200 lines, based on the well-known
[CleanRL](https://docs.cleanrl.dev/) style. Everything is visible.

It also talks to Crazyflow **directly on the GPU** (using `JaxToTorch` instead
of the SB3 adapter), so the drones (JAX) and the network (PyTorch) never leave
the GPU. That's why it can train thousands of drones at once.

```bash
python -m k12_hover.train_cleanrl --jax-device gpu --num-envs 512
```

---

## The five parts of PPO (find each in the code)

### 1. The Agent — two small networks

```python
self.critic     = ...   # input: observation → output: "how good is this state?"
self.actor_mean = ...   # input: observation → output: the average action to take
self.actor_logstd = ... # how much to randomly explore around that average
```

The **actor** decides what to do. The **critic** judges how good things are.
During exploration the actor samples from a bell curve (`Normal`) around its
chosen action — that randomness is how it *tries new things*.

### 2. The Rollout — go practice and write everything down

```python
for step in range(num_steps):
    action, logprob, _, value = agent.get_action_and_value(next_obs)
    next_obs, reward, terminated, truncated, _ = envs.step(action)
    # store obs, action, reward, value, done ...
```

Run the current policy for a fixed number of steps and record the whole
experience: what we saw, did, the reward, and the critic's value guess.

### 3. GAE — was each action better or worse than expected?

```python
delta = reward + gamma * next_value - value        # surprise at this step
advantage = delta + gamma * lambda * next_advantage # smoothed over the future
```

The **advantage** is the key signal: *positive* = that action did better than
the critic expected (do more of it); *negative* = worse (do less). `gamma` is
how much we care about the future; `lambda` smooths the estimate.

### 4. The PPO loss — improve, but don't lurch

```python
ratio = exp(new_logprob - old_logprob)             # how much the policy changed
pg_loss = max(-adv * ratio,
              -adv * clamp(ratio, 1-clip, 1+clip))  # the "proximal" clip
```

This nudges the policy to make good (positive-advantage) actions more likely —
**but** the `clamp` stops it from changing the policy more than `clip` (e.g.
20%) in one update. That single clamp is the whole idea behind the "P" (Proximal)
in PPO, and it's why PPO is so stable.

There are two more terms in the total loss:
* **value loss** — train the critic to predict returns accurately.
* **entropy bonus** — a small reward for staying a *little* random, so the agent
  keeps exploring instead of locking in too early.

### 5. The Update — repeat over the data a few times

```python
for epoch in range(update_epochs):
    for minibatch in shuffled(batch):
        ... compute loss ...
        loss.backward(); optimizer.step()
```

We reuse the collected rollout for several passes (`update_epochs`), in small
shuffled minibatches, then throw it away and collect fresh experience. (This
"reuse a few times then discard" is what makes PPO on-policy but efficient.)

---

## 🛠️ Experiments

1. **The clip knob.** Train with `--clip 0.05` (very cautious) and `--clip 0.5`
   (bold). Which learns faster? Which is more stable? Watch `ep_rew_mean`.
2. **The horizon knob.** Try `--gamma 0.90` vs `--gamma 0.999`. A higher gamma
   makes the drone care more about the far future. Does it help hovering?
3. **Compare to SB3.** Train SB3 PPO and CleanRL PPO with the same
   `--num-envs` and `--timesteps`. Do they reach a similar hover? (They should —
   it's the same algorithm.)
4. **Read the math, then explain it.** Pick the `pg_loss` lines and explain, in
   your own words, why the `clamp` keeps training stable.

---

## Where to go next

You now understand RL from the idea (Lesson 1) all the way down to the PPO
update (this lesson). The next steps in the bigger project:

* **Trajectory tracking** — change `HoverEnv` so the target *moves* along a path
  (a circle, a figure-8). The Crazyflow repo has a `FigureEightEnv` to study.
* **Harder drones** — add wind/disturbances and domain randomization so the
  policy is robust.
* **Sim-to-real** — run the trained brain in
  [CrazySim](https://github.com/gtfactslab/CrazySim) and, one day, on a real
  Crazyflie 2.1 Brushless.

Congratulations, pilot. 🚁
