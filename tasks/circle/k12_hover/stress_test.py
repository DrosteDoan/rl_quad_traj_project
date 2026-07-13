"""Stress test: how fast a circle can each controller follow before it breaks?

Our RL tracker was trained on ONE circle speed (0.5 m radius, 4 s/lap = 0.79 m/s).
A fair question a scientist always asks is: *where does it stop working?* This
script speeds the circle up and down and measures how each controller's tracking
error grows — and whether it eventually loses the drone entirely (crash / fly
away). That reveals the **limitations** of the learned policy vs the classical
controllers.

It sweeps the lap time from slow to fast and, for each speed, scores RL, PID and
MPC. It saves a figure to ``models/stress_test.png`` and prints a table.

    python -m k12_hover.stress_test
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from k12_hover.benchmark_traj import build_attitude_env, target_at  # noqa: E402
from k12_hover.controllers import CircleReference, MPCController, PIDController  # noqa: E402

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
RADIUS = 0.5
LAPS = 2.0
# Lap times from slow to fast. Speed = 2*pi*R / period.
PERIODS = [8.0, 6.0, 5.0, 4.0, 3.0, 2.5, 2.0, 1.5, 1.2, 1.0]
BOUNDS = 2.0  # if error ever exceeds this, the drone has flown away (failed)


def run_one(act_fn, env, ref, steps, dt):
    """Run one controller; return (rmse_cm, max_cm, failed, fail_time)."""
    obs, _ = env.reset()
    u = env.unwrapped
    errs = []
    failed, fail_t = False, np.nan
    for k in range(steps):
        pos = np.asarray(u.sim.data.states.pos[:, 0, :])[0]
        vel = np.asarray(u.sim.data.states.vel[:, 0, :])[0]
        tgt, _, _ = target_at(ref, k, dt)
        e = float(np.linalg.norm(pos - tgt))
        errs.append(e)
        if e > BOUNDS or pos[2] < 0.0:      # flew away or hit the ground
            failed, fail_t = True, k * dt
            break
        action = act_fn(k, obs, pos, vel)
        obs, _, term, trunc, _ = env.step(jnp.asarray(action.reshape(1, 4), device=u.device))
        if bool(np.asarray(term)[0]):        # env says crashed / out of bounds
            failed, fail_t = True, k * dt
            break
    env.close()
    errs = np.asarray(errs)
    rmse = float(np.sqrt(np.mean(errs**2))) * 100
    return rmse, float(errs.max()) * 100, failed, fail_t


def main():
    from stable_baselines3 import PPO

    model = PPO.load(str(MODELS_DIR / "track_PPO.zip"))
    dt = 1.0 / 50.0
    rows = []  # (period, speed, controller, rmse, max, failed)

    for period in PERIODS:
        speed = 2 * np.pi * RADIUS / period
        ref = CircleReference(radius=RADIUS, period=period, height=1.0)
        steps = int(LAPS * period * 50)
        max_time = LAPS * period + 1.0

        def rl_act(k, obs, pos, vel):
            a, _ = model.predict(np.asarray(obs), deterministic=True)
            return np.asarray(a).reshape(4)

        pid = PIDController(dt=dt)

        def pid_act(k, obs, pos, vel):
            rp, rv, ra = target_at(ref, k, dt)
            return pid.act(pos, vel, rp, rv, ra)

        mpc = MPCController(horizon=20, dt=dt)

        def mpc_act(k, obs, pos, vel):
            t0 = max(k - 1, 0) * dt
            future = np.array([ref.pos(t0 + j * dt) for j in range(mpc.N + 1)])
            return mpc.act(pos, vel, future)

        for name, act, n_s in [("RL", rl_act, 10), ("PID", pid_act, 0), ("MPC", mpc_act, 0)]:
            env = build_attitude_env(RADIUS, period, 1.0, n_s, max_time, "first_principles")
            rmse, mx, failed, ft = run_one(act, env, ref, steps, dt)
            rows.append((period, speed, name, rmse, mx, failed))
            flag = f"FAILED@{ft:.1f}s" if failed else "ok"
            print(f"  period={period:>4}s speed={speed:4.2f} m/s  {name:<4} "
                  f"RMSE={rmse:6.1f} cm  max={mx:6.1f} cm  {flag}")

    _plot(rows)


def _plot(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"RL": "#2b8cbe", "PID": "#e6550d", "MPC": "#31a354"}
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for name in ["RL", "PID", "MPC"]:
        pts = [(s, r, f) for (p, s, n, r, mx, f) in rows if n == name]
        pts.sort()
        speeds = [s for s, r, f in pts]
        rmses = [r for s, r, f in pts]
        ax.plot(speeds, rmses, "-o", color=colors[name], label=name, lw=2, ms=5)
        # Mark failures with a red X.
        for s, r, f in pts:
            if f:
                ax.plot(s, r, "x", color="red", ms=13, mew=3)
    ax.axvline(0.785, color="gray", ls="--", lw=1)
    ax.text(0.80, ax.get_ylim()[1] * 0.9, "RL trained here\n(0.79 m/s)", fontsize=8, color="gray")
    ax.set_xlabel("target speed around the circle (m/s)")
    ax.set_ylabel("position RMSE (cm)")
    ax.set_title(f"Stress test: tracking error vs speed (circle radius {RADIUS} m)\n"
                 "red ✗ = drone lost the target (crashed / flew away)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = MODELS_DIR / "stress_test.png"
    fig.savefig(out, dpi=120)
    print(f"\nSaved stress-test figure to: {out}")


if __name__ == "__main__":
    main()
