# Lesson 10 — Meet the controllers: five ways to fly a circle

⏱️ ~60 minutes (reading + short runs). This is the "big picture" lesson. Keep
[`k12_hover/controllers.py`](../k12_hover/controllers.py) open and read along.

> **What is a *controller*?** Anything that looks at the drone's situation and
> decides the command to send **this instant**. That's it. RL is one kind of
> controller. There are older, hand-built kinds too. In this lesson we meet **five**
> of them, understand how each one *thinks*, and see where each lives in the code.
> In [Lesson 11](11_run_the_benchmark.md) we race them.

Every controller in this project answers the **same three questions** every step:

```
   What do I see?   ->  the drone's position, velocity, tilt, and the target
   What do I want?  ->  to be where the target is (and moving with it)
   What do I send?  ->  an attitude command  [roll, pitch, yaw, thrust]
```

They only differ in **how** they turn "what I see" into "what I send". Two families:

| Family | Controllers | How it decides | Needs training? |
|--------|-------------|----------------|-----------------|
| **Learned** | RL-hover, RL-tracker | A neural network that *practiced* millions of times | yes (once) |
| **Classical** | PID, State, MPC | Physics + math, designed by hand | no |

---
---

## Part A — The learned controllers (Reinforcement Learning)

### A1. RL for hovering (recap of Lessons 1–6)

**💡 The idea.** We never told the drone *how* to hover. We gave it a **reward**
(high when it's near the target, low when far) and let it try millions of times.
The learning algorithm (**PPO**) slowly rewired a small neural network — the
**policy** — until it reliably earns high reward. The policy is a function:

```
   observation (13 numbers)  ->  [neural network]  ->  action (4 numbers)
   rel_pos, vel, quat, ang_vel                          roll, pitch, yaw, thrust
```

**🔍 In the code.** The *task* (see/do/reward) is
[`hover_env.py`](../k12_hover/hover_env.py); the *training loop* is
[`train_sb3.py`](../k12_hover/train_sb3.py) — three lines of Stable-Baselines3:

```python
model = PPO("MlpPolicy", env)
model.learn(total_timesteps=2_000_000)
model.save("models/hover_PPO.zip")
```

**What it actually learned:** to make the first three observation numbers
(`rel_pos = target − position`) shrink to zero. It has no concept of "hovering" —
only "make that arrow short." That simple fact is what makes the next part work.

### A2. RL for trajectory tracking (Lessons 8–9)

**💡 The idea.** A hover is a target that stands still; a **circle is a target that
moves**. Because the hover policy just chases the arrow, it can already follow a
*slow* circle with **no retraining** (Lesson 8). To follow a *fast* circle it needs
one more thing: a **look-ahead** — a peek at where the path goes next — so it can
steer into the turns instead of reacting late (Lesson 9).

**✏️ The observation grows** from 13 numbers to `13 + 3·N`:

```
   [ rel_pos, vel, quat, ang_vel ,  local_samples ]
     the usual 13 numbers          the next N path points, relative to the drone
```

**🔍 In the code.** [`trajectory_env.py`](../k12_hover/trajectory_env.py): the
target is a point marching around a precomputed circle (`current_target()`), and
`obs()` appends `local_samples`. Train it with
[`train_traj_sb3.py`](../k12_hover/train_traj_sb3.py). The trained tracker
(`models/track_PPO.zip`) follows a 0.79 m/s circle to about **2 cm**.

> **The catch (why we need this lesson):** RL only *knows* what it practiced. Train
> on a 0.79 m/s circle and it's brilliant there — but is it good *in general*? Is it
> better than a plain old PID? We can't know by admiring it. We must **measure it
> against the classics.** So let's meet them.

---
---

## Part B — The classical controllers

These use **physics**, not practice. They all need one shared trick, so let's do
that first.

### The shared trick: turning "desired push" into a tilt command

A quadrotor can only push along the direction it points. So every classical
controller first computes a **desired acceleration** `a_des` (which way and how
hard to push), then converts it to a tilt-and-thrust command. That conversion is
one small function, [`accel_to_attitude`](../k12_hover/controllers.py):

```python
pitch = arctan2(a_x, a_z)      # tilt nose down/up  -> accelerate in x
roll  = arctan2(-a_y, a_z)     # bank left/right    -> accelerate in y
thrust = mass * a_z / (cos(roll)·cos(pitch))   # push hard enough, allowing for tilt
```

Everything below just computes `a_des` in a different way.

---

### B1. PID — the workhorse of real drones

**💡 The idea.** Look at the **error** (how far you are from the target) and push
to erase it. PID mixes three views of that error:

| Term | Reads | In plain words | Fixes |
|------|-------|----------------|-------|
| **P** (proportional) | the error *now* | "far away → push hard" | being off-target |
| **I** (integral) | the error *piled up over time* | "always a little low → push a little extra forever" | steady droop |
| **D** (derivative) | how fast the error *changes* | "closing in fast → ease off" | overshoot / wobble |

**✏️ The math** (this is the whole controller):

```
   desired_accel = Kp·(position error) + Kd·(velocity error) + Ki·(piled-up error)
                 + gravity + the target's own acceleration (feed-forward)
```

`Kp, Kd, Ki` are the three **gains** — the knobs you tune. Too small and it lags;
too big and it shakes itself apart.

**🔍 In the code.** [`PIDController`](../k12_hover/controllers.py):

```python
class PIDController:
    def act(self, pos, vel, ref_pos, ref_vel, ref_acc):
        e_pos = ref_pos - pos
        e_vel = ref_vel - vel
        self._integral += e_pos * self.dt
        a_des = (self.kp*e_pos + self.kd*e_vel + self.ki*self._integral
                 + ref_acc + [0, 0, GRAVITY])
        return normalize_attitude(accel_to_attitude(a_des))
```

**Its blind spot:** PID only reacts to the error it feels *right now* — it never
looks ahead. On a curving path that means it's always a half-step behind.

**🛠️ Build-it-yourself:** open `controllers.py`, find the gains
`kp=6.0, kd=4.0, ki=1.0`. Predict what happens if you double `kp`, then try it in
the benchmark (Lesson 11).

---

### B2. State — let the drone's *built-in* controller do it

**💡 The idea.** A real Crazyflie already has a controller baked into its firmware
(a cascade of PID loops called **Mellinger**). Often the smartest move is: don't
write your own — just **tell the built-in one where to go** and let it handle the
motors. Crazyflow faithfully re-implements that firmware controller. You hand it a
**state setpoint** and it does the rest.

**✏️ The command** is a 13-number "here's the full state I want":

```
   [ x, y, z ,  vx, vy, vz ,  ax, ay, az ,  yaw ,  roll_rate, pitch_rate, yaw_rate ]
     where to be   how fast to move   (ignored)   which way    (ignored)
```

We fill in the target's position and velocity and let the onboard controller
figure out the tilt and thrust internally.

**🔍 In the code.** The gym environment *disables* state mode, so we drive the raw
simulator directly — see `run_state()` in
[`benchmark_traj.py`](../k12_hover/benchmark_traj.py):

```python
sim = Sim(..., control=Control.state)
cmd[0, 0, 0:3] = target_position      # where to go
cmd[0, 0, 3:6] = target_velocity      # how fast
sim.state_control(cmd); sim.step()
```

**Its quirk (a real-world lesson!):** Crazyflow gives this controller a
**deliberately slightly-wrong drone mass** (0.0393 kg instead of the true
0.0434 kg), copying how a real Crazyflie is a little miscalibrated from the
factory. Because it thinks the drone is lighter than it is, it doesn't push quite
hard enough, so it **sags below** the target. Watch for that in the benchmark — it's
a preview of the **sim-to-real gap** (the planned deploy lesson), and part of *why* tuning and
learning are worth the effort.

---

### B3. MPC — plan ahead, then act

**💡 The idea.** *Model-Predictive Control* is the "chess player" of controllers.
Every step it **imagines the next ~0.4 seconds**: using a simple model of the
drone, it asks *"if I push like this, then like that, where do I end up?"* and
**searches for the sequence of pushes that keeps it closest to the upcoming path**.
Then it does only the **first** push — and next step it re-plans from scratch. This
constant re-planning is called a **receding horizon**.

Like the RL tracker, MPC gets to **see the path ahead** — so it can anticipate the
turns. Unlike RL, it figures out the plan by *solving an optimization* live, every
single step, instead of recalling what it practiced.

**✏️ The math** — it minimizes a **cost** over the horizon of `N` steps:

```
   minimize   Σ  q_pos·(off the path)²  +  q_vel·(not matching target speed)²
              +  r_acc·(pushing too hard)²
   subject to the drone's motion:  next_pos = pos + vel·dt,  next_vel = vel + accel·dt
```

- `q_pos` big → hug the path.
- `q_vel` → **match the target's speed** (this one is subtle: penalize the
  *difference from the target's velocity*, not the velocity itself — otherwise MPC
  tries to stand still and lags badly behind a moving target).
- `r_acc` → don't thrash the controls.

**🔍 In the code.** [`MPCController`](../k12_hover/controllers.py) builds this
optimization once with **CasADi** (a math-optimization library) and solves it each
step:

```python
for k in range(N):
    opti.subject_to(P[:,k+1] == P[:,k] + V[:,k]*dt)     # the model of motion
    opti.subject_to(V[:,k+1] == V[:,k] + A[:,k]*dt)
    cost += q_pos*sumsqr(P[:,k]-ref[:,k])               # stay on the path
    cost += q_vel*sumsqr(V[:,k] - ref_vel)              # keep up with the target
    cost += r_acc*sumsqr(A[:,k])                         # be gentle
```

> **Note:** professional MPC (like in `lsy_drone_racing`) uses a heavier solver
> called *acados* and a full physics model. Ours is a **teaching version**: a
> point-mass model and the free CasADi solver. It's simpler, but the core idea —
> *predict, optimize, apply the first move, repeat* — is exactly the same.

**Its price:** all that planning costs **compute**. MPC solves a fresh optimization
every step (~5 ms), while the trained RL policy just runs a tiny network (~1 ms).
Keep an eye on that trade in the benchmark.

---
---

## Side-by-side: how each one "thinks"

| Controller | Its plan | Sees the path ahead? | Needs training? | Compute/step |
|------------|----------|:---:|:---:|:---:|
| **RL** | recall what practice taught | ✅ (look-ahead in obs) | ✅ once | low (~1 ms) |
| **PID** | erase the current error | ❌ | ❌ | tiny (~0.1 ms) |
| **State** | hand it to the firmware | ❌ | ❌ | tiny |
| **MPC** | solve for the best next moves | ✅ (optimizes over horizon) | ❌ | high (~5 ms) |

**The question the benchmark will answer:** does the *learned* controller actually
beat the hand-built ones — and is "seeing the path ahead" what really matters?

---

## ✅ Check your understanding

1. Every controller outputs the same 4 numbers. What are they?
2. What does the **I** term in PID fix that **P** alone cannot?
3. MPC and RL both "see the path ahead." How does each one *use* that peek
   differently at run time?
4. Why does the built-in **State** controller sag below the target? What real-world
   problem is that copying?
5. Which controller would you expect to be cheapest to run on a tiny drone
   computer? Which the most expensive?

Next: **[Lesson 11](11_run_the_benchmark.md)** — put all five on the same circle
and measure who wins.
