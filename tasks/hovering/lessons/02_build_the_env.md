# Lesson 2 — How we describe the hover task (the Environment)

⏱️ ~30 minutes. Open [`k12_hover/hover_env.py`](../k12_hover/hover_env.py)
next to this lesson and read along.

---

## The environment answers three questions

To teach the drone, the computer needs three things spelled out. All three live
in `hover_env.py`, in the `HoverEnv` class.

### Question 1 — What does the drone SEE? (the observation)

A blindfolded pilot can't fly. We give the agent these numbers every step
(method `obs` in the code):

| What it sees | Numbers | Why it helps |
|--------------|---------|--------------|
| `rel_pos` | 3 | the arrow pointing from the drone to the target |
| `vel` | 3 | how fast it's moving (so it can slow down) |
| `quat` | 4 | which way it's tilted |
| `ang_vel` | 3 | how fast it's spinning |

That's **13 numbers**. Notice we give *relative* position (target minus
position), not the raw position. That way "I am 20 cm below the target" means
the same thing no matter where the target is. Smart observations make learning
easier.

### Question 2 — What can the drone DO? (the action)

The agent outputs **4 numbers** every step:

```
[ thrust ,  roll ,  pitch ,  yaw ]
   up?     tilt     tilt    spin
          left/right fwd/back
```

This is called **attitude control**. The agent does not control each of the 4
motors directly — it says "tilt this way and push this hard", and the drone's
built-in controller handles the motors. To keep things tidy, we wrap the env so
the agent can output numbers in the easy range **−1 to 1** (see
`NormalizeActions` in `wrappers.py`).

### Question 3 — How GOOD was that? (the reward)

This is the heart of RL. Look at the `_reward` method. The main idea is one line:

```python
distance = ||position − target||          # how far from the X
reward   = exp(-distance_coef * distance) # 1.0 on target, →0 far away
```

`exp(-distance)` is a smooth "hotter / colder" score: it is **1.0** when the
drone sits exactly on the target and shrinks toward **0** as it drifts away.
This gentle slope is what the agent climbs.

Then we add optional **penalties** (all off by default, you turn them on in
Lesson 4):

```python
reward -= velocity_coef * speed       # punish moving fast  → hold still
reward -= spin_coef     * spinning    # punish spinning
reward -= tilt_coef     * tilt        # punish tipping over
reward  = crash_penalty if crashed    # big minus for hitting the ground
```

---

## When does an episode end?

Look at `terminated` and the `max_episode_time` setting:

* **Crash** — the drone hits the ground (`z < 0`). Episode ends, big penalty.
* **Flew away** — it gets more than `bounds` meters (2 m) from the target. It's
  clearly lost; end the episode and start fresh.
* **Time up** — after `max_episode_time` seconds (5 s) we stop and reset. This
  isn't a failure, just the end of a practice run.

---

## Where does each episode start?

If the drone always started exactly on the target, it would learn nothing. So
`_reset_randomization` drops it at a **random spot** near the target with a
**small random push**. The agent must learn to recover from many situations,
not memorize one.

---

## Why is this so fast? (parallel drones)

Crazyflow runs the physics in **JAX**, which can simulate **hundreds of drones
at the same time** on a GPU. When you pass `num_envs=128`, you are flying 128
drones in parallel, each having its own adventure. More drones = more experience
per second = faster learning. This is Crazyflow's superpower.

---

## 🛠️ Try it yourself

Run this small experiment (it does NOT train; it just steps the env so you can
*see* the observation and reward):

```bash
cd /workspace/tasks/hovering          # inside the container; the main venv is already active
python - <<'PY'
import numpy as np
from k12_hover import make_sb3_env, target_distance

env = make_sb3_env(num_envs=1, device="cpu")
obs = env.reset()
print("The drone sees these 13 numbers:", np.round(obs[0], 2))
print("It starts", round(float(target_distance(obs)[0])*100), "cm from the target")

# Send a constant 'tilt up a little' action for a few steps.
action = np.array([[0.2, 0.0, 0.0, 0.0]], dtype=np.float32)
for step in range(5):
    obs, reward, done, info = env.step(action)
    print(f"step {step}: reward = {reward[0]:.3f}, distance = {target_distance(obs)[0]*100:.0f} cm")
env.close()
PY
```

**Questions:**

1. Which 3 of the 13 numbers are `rel_pos`? (Hint: the first three.)
2. Does the reward go up or down as the distance gets smaller?
3. Change the action to `[0.0, 0.0, 0.0, 0.0]` (no thrust). What happens to the
   distance, and why? (Think about gravity.)

Next: **Lesson 3** — actually train a drone.
