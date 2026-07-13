# Lesson 11 — The scorecard: run the benchmark

⏱️ ~45 minutes. You've met the five controllers in
[Lesson 10](10_meet_the_controllers.md). Now put them on the **same** circle,
under the **same** physics, and **measure** who follows it best. Open
[`k12_hover/benchmark_traj.py`](../k12_hover/benchmark_traj.py).

> **The one big idea:** "our RL drone follows the circle!" is not a result — it's a
> claim. A scientist turns claims into **numbers**: race it against the controllers
> engineers already trust, on a fair track, and report who wins by how much.

---

## 1. A fair race

Every controller starts **on the circle but standing still**, while the target
immediately takes off around the loop. So each one must first **catch up** to the
moving target (the *catch-up time*), then **keep up** with it (the steady tracking
error). Same start, same path, same physics (`first_principles`), same drone
(`cf21B_500`) — the only thing that differs is the controller's brain.

```bash
conda activate crazyflow
cd k12_RL_quad_traj
python -m k12_hover.benchmark_traj --model models/track_PPO.zip \
    --n-samples 10 --radius 0.5 --period 4 --laps 3
```

It prints a scorecard and saves a figure to `models/benchmark_circle.png`.

---

## 2. What the numbers mean

| Metric | What it measures | Better = |
|--------|------------------|----------|
| **RMSE** | root-mean-square distance from the target over the whole run — the headline accuracy number | lower |
| **mean / max** | average and worst-case error | lower |
| **xy / z** | error split into horizontal vs vertical | lower |
| **% within 5 cm** | how much of the flight was "on target" | higher |
| **catch-up** | time to first get within 5 cm and stay there | lower |
| **ms/step** | how much computer time the controller costs each step | lower |

> **Why RMSE?** It squares every error before averaging, so a few big misses hurt
> more than lots of tiny ones. It's the standard yardstick for "how tightly did you
> follow the path."

---

## 3. Reading the result

A typical scorecard (yours will be very close):

| controller | RMSE | max | catch-up | ms/step |
|------------|------|-----|----------|---------|
| **RL**    | **2.1 cm** | 13 cm | 0.5 s | 1.3 |
| **MPC**   | **2.1 cm** | 12 cm | 0.5 s | 5.2 |
| **PID**   | 4.6 cm | 19 cm | 1.1 s | 0.1 |
| **State** | 11.7 cm | 15 cm | never | 0.01 |

The figure has three panels: the **paths** (top-down), the **error over time**, and
the **RMSE bars**. Three lessons fall right out of the data:

**1. Look-ahead wins.** RL and MPC — the two controllers that can *see the path
coming* (Lesson 10) — tie for best and are **2× tighter than PID**, which only
reacts to the error it feels right now. The biggest win came not from a cleverer
algorithm but from **better information**: knowing where the path goes next. (This
is the exact same lesson as Lesson 9, now proven with a number.)

**2. RL matches MPC for a fraction of the compute.** MPC gets its accuracy by
solving a fresh optimization *every step* (~5 ms). The RL policy learned to make
almost the same moves in one quick network pass (~1 ms). **Training is expensive
once; running the trained brain is cheap forever.** That trade is a big reason RL is
exciting for real robots with small computers.

**3. The built-in controller struggles here — honestly.** The "State" controller
sags below the circle because Crazyflow gives it a deliberately-wrong mass (Lesson
10). It's the weakest here, but it's also the one closest to *real firmware* — a
reminder that a controller that looks fine in theory can drift in the real world.

---

## 4. 🛠️ Experiments

1. **Change the speed.** Re-run at `--period 8` (slow) and `--period 2.5` (fast).
   Does PID fall further behind RL/MPC as the circle speeds up? (Look-ahead should
   matter *more* the faster the target moves.)
2. **Tune the PID.** In `controllers.py`, change `PIDController`'s `kp`, `kd`, `ki`.
   How close to RL/MPC can you get? What does too-large `kp` do to the error plot?
3. **Blind the MPC.** Set `MPCController(horizon=3)` (sees only ~0.06 s ahead). Does
   it start to look like PID? What does that tell you about *why* it was winning?
4. **Accuracy vs cost.** Make a little table of RMSE vs `ms/step`. Which controller
   for a cheap toy drone? Which for a racing drone where every millimeter counts?

## ✅ Check your understanding

1. What does RMSE measure, and why square the errors?
2. RL and MPC tied on accuracy. What's the big *practical* difference at run time?
3. Two controllers couldn't see the path ahead. Which two, and how did it show in
   their scores?
4. If you halved the lap time (doubled the speed), which controller do you predict
   would suffer most, and why?

Next: **[Lesson 12](12_stress_test_and_future.md)** — push the RL tracker until it
breaks, and talk about how we'd make it better.
