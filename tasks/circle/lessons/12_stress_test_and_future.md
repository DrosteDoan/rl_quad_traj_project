# Lesson 12 — Where does RL break? Stress test & the road ahead

⏱️ ~45 minutes. The most honest — and most scientific — question you can ask about
any controller is: **"where does it stop working?"** In this lesson we speed the
circle up until every controller fails, learn *why*, and map out how we'd make the
RL tracker better. Open [`k12_hover/stress_test.py`](../k12_hover/stress_test.py).

> **The one big idea:** a controller that works on the one case you tested is a
> *demo*. A controller you've pushed until it breaks — and understand *how* it
> breaks — is *engineering*. Finding the limits is the point, not a failure.

---

## 1. Run the stress test

Our RL tracker practiced on exactly one circle: 0.5 m radius, 4 s/lap = **0.79 m/s**.
The stress test keeps the radius fixed and makes the lap faster and faster, scoring
RL, PID and MPC at each speed.

```bash
conda activate crazyflow
cd k12_RL_quad_traj
python -m k12_hover.stress_test
```

It saves `models/stress_test.png` — RMSE vs speed, with a **red ✗** wherever the
drone lost the target (crashed or flew away).

---

## 2. What we found

| speed (m/s) | RL RMSE | PID RMSE | MPC RMSE | notes |
|---|---|---|---|---|
| 0.39 | 2.9 cm | 3.5 cm | **1.0 cm** | gentle — everyone's happy |
| 0.79 ⟵ *RL trained here* | 2.5 cm | 5.4 cm | 2.3 cm | RL & MPC tight |
| 1.05 | 5.8 cm | 7.5 cm | **3.6 cm** | RL starts to slip |
| 1.26 | 10.5 cm | 10.7 cm | **6.5 cm** | RL error 4× its best |
| 1.57 | 18.4 cm | 18.1 cm | 61 cm 💥 | **MPC's model breaks** |
| 2.09 | 31 cm | 39 cm | ✗ fail | very rough |
| 2.6+ | ✗ fail | ✗ fail | ✗ fail | **physically impossible** |

Three big lessons hide in this table.

### Lesson 2a — RL is only reliable near what it practiced

Look at RL's column: **rock-solid (~2–3 cm) up to its training speed, then the
error climbs steadily** — 5.8 cm, then 10, then 18, then it loses the drone around
2.6 m/s. This is *the* fundamental limitation of a learned controller:

> **RL knows what it practiced, and little else.** Far from its training conditions
> it is guessing. This is called a **distribution shift** — the world no longer
> looks like the world it trained in.

The good news: it degrades *gracefully* (a gentle climb, not an instant cliff), and
even out at 1.5 m/s — nearly twice its training speed — it's still flying, just
loosely. But you would never *trust* it there without retraining.

### Lesson 2b — MPC is only as good as its model

MPC is the **most accurate of all** at low speed (1.0 cm!) — but watch it **fall off
a cliff** at 1.57 m/s (jumping to 61 cm) and then fail. Why?

Our teaching MPC models the drone as a simple **point mass** and assumes it can be
pushed in any direction instantly. At low speed that's close enough. But fast
cornering needs **big tilts**, and a big tilt takes *time* to achieve and *wastes*
lift — physics our point-mass model completely ignores. So MPC confidently plans
moves the real drone can't execute, and diverges.

> **A model-based controller trusts its model.** When the model is wrong, the
> controller is wrong — often suddenly. (This is exactly why professional MPC uses
> a full physics model and a heavy solver like *acados*.)

### Lesson 2c — some limits belong to the drone, not the controller

Past ~2.2 m/s, **every** controller fails. That's not a bad controller — it's
**physics**. To hold a circle you must constantly accelerate toward its center
(*centripetal acceleration*), and that grows with speed squared:

```
   a_center = v² / radius
   the tilt needed:  tan(angle) = a_center / gravity
```

At 0.79 m/s on our 0.5 m circle that's a gentle 7° tilt. But at 2.2 m/s it's
`a = 2.2²/0.5 ≈ 9.7 m/s² ≈ 1 g`, needing a **45° tilt** — and beyond that the drone
simply can't corner that tight without far more thrust than it has. **No controller,
learned or classical, can beat that.** Knowing which failures are the *controller's*
fault and which are the *drone's* is a key engineering skill.

---

## 3. 🛠️ Explore the limits yourself

1. **Move the wall.** In `stress_test.py`, change `RADIUS` to `1.0` m and re-run. A
   bigger circle is gentler to corner — does the failure speed go *up*? Check it
   against the `v² / radius` formula.
2. **Find RL's honest working range.** At what speed does RL's RMSE first cross
   10 cm? That's roughly the fastest circle you'd trust this particular brain on.
3. **Who degrades most gracefully?** Compare the *shape* of the three curves before
   they fail. Which one has no sudden cliff?

---

## 4. The road ahead — how we'd make the RL tracker better

The stress test isn't a dead end; it's a **to-do list**. Here's how a real project
would push the limits, roughly easiest first:

### Make RL robust to more than one circle
- **Train on many speeds and sizes.** Randomize the radius and lap-time every
  episode so the policy practices a whole *range*, not one point. This is
  **domain randomization**, and it directly attacks Lesson 2a.
- **Curriculum learning.** Start slow, and once the drone is good, gradually speed
  the circle up during training — the way you'd coach a student. (Try it by hand:
  train at `--period 6`, then keep training the same model at `--period 3`.)
- **Randomize the whole path**, not just circles. The pro reference
  (`lsy_drone_racing`) trains on *random wiggly trajectories* (smooth splines
  through random points), so the policy learns to follow *anything* smooth — then a
  circle is just one easy case.

### Give the policy more to work with
- **Add the target's velocity to the observation**, or stack the last few
  observations (a short **memory**), so the network can sense how fast the path is
  moving and react sooner.
- **Reward smoothness.** Penalize jerky commands (big changes in thrust/tilt) so the
  flight is gentler and transfers better to real hardware — the same idea as the
  `--tilt-action-coef` knob from the deploy work.
- **A bigger network / longer training** buys some headroom, though it's usually the
  smallest win compared to better *information* and better *practice*.

### Combine the best of both worlds
- **Warm-start MPC with RL** (or vice-versa): let the fast RL policy propose a move
  and a light MPC polish it against the drone's real limits. Hybrid controllers get
  RL's speed *and* MPC's respect for physics.
- **Give MPC a better model** (real attitude dynamics + thrust/tilt limits as
  constraints) so it stops planning impossible moves — this alone would erase its
  high-speed cliff.

### Bigger horizons (pun intended)
- **New shapes:** a figure-eight (the challenge at the end of Lesson 9), then full
  3-D paths that climb and dive.
- **Disturbances:** add wind or payload changes during training so the policy learns
  to reject them — a step toward the real world.
- **Sim-to-real:** carry the tracker through the CrazySim pipeline from Lesson 7 and,
  eventually, onto a real Crazyflie.

---

## ✅ Check your understanding

1. RL was flawless at 0.79 m/s but sloppy at 1.3 m/s. What's the one-word name for
   why, and what would you change in *training* to fix it?
2. MPC was the most accurate controller — so why did it fail *first* as the circle
   sped up?
3. Every controller failed past ~2.2 m/s. Whose "fault" is that, and how would you
   prove it with a formula?
4. Name two different ways to make the RL tracker follow a *faster* circle reliably,
   and say which part of the problem each one attacks.

🏁 **You've completed the journey:** from *what is RL* → hovering → following a
circle → training a real tracker → benchmarking it against PID, MPC and the onboard
controller → and finally pushing it until it breaks and knowing how to make it
better. That last step — finding and explaining the limits — is what turns a cool
demo into real engineering. 🚁
