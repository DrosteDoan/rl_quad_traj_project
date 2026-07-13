"""Benchmark: race the RL tracker against classical controllers on one circle.

This is Lesson 10 — the *honest scorecard*. We put four controllers on the exact
same circular path, under the exact same physics, and measure who follows it best:

* **RL**    — our trained policy (``models/track_PPO.zip``), the learned tracker.
* **PID**   — the classic reactive autopilot (``controllers.PIDController``).
* **MPC**   — model-predictive control, plans a few steps ahead (``controllers.MPCController``).
* **State** — the simulator's own built-in onboard controller (Crazyflie firmware
              Mellinger, reached through Crazyflow's ``state`` command).

All four start **on the path but at a standstill**, while the target immediately
sets off around the circle. So each controller first has to *catch up* to the
moving target (that's the "tracking time"), then *keep up* with it (that's the
steady tracking error). We report, for each:

* **RMSE (cm)** — root-mean-square position error over the whole run (the headline
  number: lower = follows the path more tightly).
* **catch-up time (s)** — how long until it first gets within 5 cm and stays there.
* **max error, horizontal vs vertical error, % of time within 5 cm.**
* **compute time (ms/step)** — how expensive the controller is to run.

It saves a comparison figure to ``models/benchmark_circle.png``.

Example
-------
    python -m k12_hover.benchmark_traj --model models/track_PPO.zip \
        --n-samples 10 --radius 0.5 --period 4 --laps 3
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from k12_hover.controllers import (  # noqa: E402
    CircleReference,
    MPCController,
    PIDController,
)
from k12_hover.trajectory_env import TrajectoryEnv  # noqa: E402
from k12_hover.wrappers import OBS_ORDER, FlattenObservation, NormalizeActions  # noqa: E402

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
TOL = 0.05  # meters: "within 5 cm" counts as good tracking


# ---------------------------------------------------------------- shared helpers
def build_attitude_env(radius, period, height, n_samples, max_time, physics):
    """A single-drone circle env, started ON the path and at rest (deterministic)."""
    base = TrajectoryEnv(
        num_envs=1, device="cpu", n_samples=n_samples,
        radius=radius, period=period, height=height,
        start_radius=0.0, start_speed=0.0,          # exact, repeatable start
        dynamics=physics, max_episode_time=max_time,
    )
    obs_order = OBS_ORDER + ("local_samples",) if n_samples > 0 else OBS_ORDER
    return FlattenObservation(NormalizeActions(base), obs_order=obs_order)


def target_at(ref: CircleReference, k: int, dt: float):
    """The target the env is graded against at control step k (env lags ref by 1 step)."""
    t = max(k - 1, 0) * dt
    return ref.pos(t), ref.vel(t), ref.acc(t)


def metrics(positions, targets, dt, compute_ms):
    """Turn logged positions + targets into the scorecard numbers."""
    positions = np.asarray(positions)
    targets = np.asarray(targets)
    err = np.linalg.norm(positions - targets, axis=1)          # per-step distance
    rmse = float(np.sqrt(np.mean(err**2)))
    rmse_xy = float(np.sqrt(np.mean(np.sum((positions[:, :2] - targets[:, :2]) ** 2, axis=1))))
    rmse_z = float(np.sqrt(np.mean((positions[:, 2] - targets[:, 2]) ** 2)))
    # Catch-up time: first moment it gets within TOL and never leaves again.
    within = err < TOL
    catch = np.nan
    for i in range(len(within)):
        if within[i] and within[i:].all():
            catch = i * dt
            break
    return {
        "rmse_cm": rmse * 100,
        "mean_cm": float(err.mean()) * 100,
        "max_cm": float(err.max()) * 100,
        "rmse_xy_cm": rmse_xy * 100,
        "rmse_z_cm": rmse_z * 100,
        "pct_within": float(within.mean()) * 100,
        "catch_s": catch,
        "ms_step": compute_ms,
        "err": err,           # kept for the error-vs-time plot
        "pos": positions,     # kept for the top-down plot
    }


# ------------------------------------------------------------- controller runners
def run_attitude(env, act_fn, ref, steps, dt):
    """Drive the attitude env with a function act_fn(k, obs, pos, vel) -> [-1,1]^4."""
    positions, targets = [], []
    obs, _ = env.reset()
    u = env.unwrapped
    t_compute = 0.0
    for k in range(steps):
        pos = np.asarray(u.sim.data.states.pos[:, 0, :])[0]
        vel = np.asarray(u.sim.data.states.vel[:, 0, :])[0]
        tgt, _, _ = target_at(ref, k, dt)
        positions.append(pos.copy())
        targets.append(tgt.copy())
        t0 = time.perf_counter()
        action = act_fn(k, obs, pos, vel)
        t_compute += time.perf_counter() - t0
        obs, _, term, trunc, _ = env.step(jnp.asarray(action.reshape(1, 4), device=u.device))
    env.close()
    return metrics(positions, targets, dt, t_compute / steps * 1000)


def run_state(ref, steps, dt, radius, period, height, physics):
    """Drive Crazyflow's built-in onboard controller in ``state`` command mode."""
    from crazyflow.control.core import Control
    from crazyflow.sim import Sim

    sim = Sim(n_worlds=1, n_drones=1, drone="cf21B_500", device="cpu",
              dynamics=physics, control=Control.state, freq=500,
              state_freq=int(1.0 / dt), attitude_freq=500)
    sim.reset()
    # Start ON the path, at rest, rotors already at hover speed (matches the env).
    start = ref.pos(0.0)
    st = sim.data.states
    sim.data = sim.data.replace(states=st.replace(
        pos=jnp.asarray(start, dtype=jnp.float32).reshape(1, 1, 3),
        vel=jnp.zeros_like(st.vel),
        rotor_vel=10000.0 * jnp.ones_like(st.rotor_vel),
    ))
    n_substeps = sim.freq // int(1.0 / dt)

    positions, targets = [], []
    t_compute = 0.0
    for k in range(steps):
        pos = np.asarray(sim.data.states.pos[:, 0, :])[0]
        tgt, tvel, _ = target_at(ref, k, dt)
        positions.append(pos.copy())
        targets.append(tgt.copy())
        t0 = time.perf_counter()
        cmd = np.zeros((1, 1, 13), dtype=np.float32)
        cmd[0, 0, 0:3] = tgt              # desired position
        cmd[0, 0, 3:6] = tvel             # desired velocity (feed-forward)
        cmd[0, 0, 9] = 0.0                # desired yaw
        t_compute += time.perf_counter() - t0
        sim.state_control(jnp.asarray(cmd))
        sim.step(n_substeps)
    return metrics(positions, targets, dt, t_compute / steps * 1000)


# ------------------------------------------------------------------------- plot
def make_plot(results, ref, steps, dt, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"RL": "#2b8cbe", "PID": "#e6550d", "MPC": "#31a354", "State": "#756bb1"}
    fig = plt.figure(figsize=(15, 5))

    # (1) Top-down XY view: the circle + each controller's path.
    ax1 = fig.add_subplot(1, 3, 1)
    th = np.linspace(0, 2 * np.pi, 200)
    ax1.plot(ref.center[0] + ref.radius * np.cos(th),
             ref.center[1] + ref.radius * np.sin(th),
             "k--", lw=1.5, label="target circle")
    for name, m in results.items():
        ax1.plot(m["pos"][:, 0], m["pos"][:, 1], color=colors[name], lw=1.5, label=name)
    ax1.set_aspect("equal"); ax1.set_xlabel("x (m)"); ax1.set_ylabel("y (m)")
    ax1.set_title("Path followed (top-down)"); ax1.legend(fontsize=8)

    # (2) Tracking error over time.
    ax2 = fig.add_subplot(1, 3, 2)
    t = np.arange(steps) * dt
    for name, m in results.items():
        ax2.plot(t, m["err"] * 100, color=colors[name], lw=1.3, label=name)
    ax2.axhline(TOL * 100, color="gray", ls=":", lw=1, label=f"{TOL*100:.0f} cm tol")
    ax2.set_xlabel("time (s)"); ax2.set_ylabel("position error (cm)")
    ax2.set_title("Tracking error vs time"); ax2.legend(fontsize=8)

    # (3) RMSE bar chart (the headline scorecard).
    ax3 = fig.add_subplot(1, 3, 3)
    names = list(results.keys())
    rmses = [results[n]["rmse_cm"] for n in names]
    bars = ax3.bar(names, rmses, color=[colors[n] for n in names])
    for b, v in zip(bars, rmses):
        ax3.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}", ha="center", va="bottom", fontsize=9)
    ax3.set_ylabel("position RMSE (cm)")
    ax3.set_title("Overall tracking error (lower = better)")

    fig.suptitle(f"Circle tracking benchmark — radius {ref.radius} m, "
                 f"{ref.period} s/lap ({2*np.pi*ref.radius/ref.period:.2f} m/s)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=120)
    print(f"\nSaved comparison figure to: {out_path}")


# ------------------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser(description="Benchmark RL vs PID/MPC/State on a circle.")
    p.add_argument("--model", default=str(MODELS_DIR / "track_PPO.zip"), help="RL tracker .zip.")
    p.add_argument("--n-samples", type=int, default=10, help="Look-ahead the RL model was trained with.")
    p.add_argument("--radius", type=float, default=0.5)
    p.add_argument("--period", type=float, default=4.0, help="Seconds per lap (smaller = harder).")
    p.add_argument("--height", type=float, default=1.0)
    p.add_argument("--laps", type=float, default=3.0, help="How many laps to score over.")
    p.add_argument("--physics", default="first_principles", choices=["so_rpy", "first_principles"])
    p.add_argument("--skip", default="", help="Comma-separated controllers to skip (e.g. MPC).")
    p.add_argument("--out", default=str(MODELS_DIR / "benchmark_circle.png"))
    args = p.parse_args()

    skip = {s.strip() for s in args.skip.split(",") if s.strip()}
    ref = CircleReference(radius=args.radius, period=args.period, height=args.height)
    dt = 1.0 / 50.0
    steps = int(args.laps * args.period * 50)
    max_time = args.laps * args.period + 1.0
    print(f"Benchmark: circle r={args.radius} m, {args.period} s/lap "
          f"({2*np.pi*args.radius/args.period:.2f} m/s), {args.laps} laps = {steps} steps.\n")

    results = {}

    # --- RL ---
    if "RL" not in skip:
        from stable_baselines3 import PPO
        model = PPO.load(args.model)

        def rl_act(k, obs, pos, vel):
            action, _ = model.predict(np.asarray(obs), deterministic=True)
            return np.asarray(action).reshape(4)

        env = build_attitude_env(args.radius, args.period, args.height, args.n_samples, max_time, args.physics)
        results["RL"] = run_attitude(env, rl_act, ref, steps, dt)
        print(f"  RL    done: RMSE {results['RL']['rmse_cm']:.1f} cm")

    # --- PID ---
    if "PID" not in skip:
        pid = PIDController(dt=dt)

        def pid_act(k, obs, pos, vel):
            rp, rv, ra = target_at(ref, k, dt)
            return pid.act(pos, vel, rp, rv, ra)

        env = build_attitude_env(args.radius, args.period, args.height, 0, max_time, args.physics)
        results["PID"] = run_attitude(env, pid_act, ref, steps, dt)
        print(f"  PID   done: RMSE {results['PID']['rmse_cm']:.1f} cm")

    # --- MPC ---
    if "MPC" not in skip:
        mpc = MPCController(horizon=15, dt=dt)

        def mpc_act(k, obs, pos, vel):
            t0 = max(k - 1, 0) * dt
            future = np.array([ref.pos(t0 + j * dt) for j in range(mpc.N + 1)])
            return mpc.act(pos, vel, future)

        env = build_attitude_env(args.radius, args.period, args.height, 0, max_time, args.physics)
        results["MPC"] = run_attitude(env, mpc_act, ref, steps, dt)
        print(f"  MPC   done: RMSE {results['MPC']['rmse_cm']:.1f} cm")

    # --- State (built-in onboard controller) ---
    if "State" not in skip:
        results["State"] = run_state(ref, steps, dt, args.radius, args.period, args.height, args.physics)
        print(f"  State done: RMSE {results['State']['rmse_cm']:.1f} cm")

    # --- Scorecard table ---
    print("\n=== Scorecard (circle tracking) ===")
    hdr = f"{'controller':<8} {'RMSE':>7} {'mean':>7} {'max':>7} {'xy':>6} {'z':>6} {'%<5cm':>6} {'catch':>7} {'ms/step':>8}"
    print(hdr); print("-" * len(hdr))
    for name, m in results.items():
        catch = f"{m['catch_s']:.2f}s" if not np.isnan(m['catch_s']) else "  n/a"
        print(f"{name:<8} {m['rmse_cm']:6.1f}c {m['mean_cm']:6.1f}c {m['max_cm']:6.1f}c "
              f"{m['rmse_xy_cm']:5.1f} {m['rmse_z_cm']:5.1f} {m['pct_within']:5.0f}% {catch:>7} {m['ms_step']:7.2f}")

    make_plot(results, ref, steps, dt, args.out)


if __name__ == "__main__":
    main()
