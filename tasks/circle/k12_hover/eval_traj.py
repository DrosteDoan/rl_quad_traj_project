"""Watch (and score) a drone following a circular path.

This one script covers BOTH lessons:

* **Lesson 8 — reuse the hover brain (no new training).** Point ``--model`` at
  your already-trained ``hover_PPO.zip`` and leave ``--n-samples 0``. The hover
  policy will chase the moving target around the circle. Try turning the circle
  faster (smaller ``--period``) and watch it start to cut the corners — a hover
  brain has no idea where the path goes next, so it always reacts a step late.

* **Lesson 9 — a dedicated tracker.** Point ``--model`` at a policy trained with
  ``train_traj_sb3.py`` and set ``--n-samples`` to the SAME value you trained
  with. That policy can see the path ahead, so it follows the fast circle much
  more tightly.

It reports the **tracking error**: how far (cm) the drone was from the moving
target, on average and at its worst.

Examples
--------
Reuse the hover model on a gentle circle::

    python -m k12_hover.eval_traj --model models/hover_PPO.zip --period 8 --render

Same model, faster circle (watch it lag)::

    python -m k12_hover.eval_traj --model models/hover_PPO.zip --period 3

Score a trained tracker (must match the --n-samples it was trained with)::

    python -m k12_hover.eval_traj --model models/track_PPO.zip --n-samples 10 --period 3
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np  # noqa: E402
from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3  # noqa: E402

from k12_hover import make_traj_sb3_env, target_distance  # noqa: E402

ALGOS = {"PPO": PPO, "A2C": A2C, "SAC": SAC, "TD3": TD3, "DDPG": DDPG}


def guess_algo(model_path: str) -> str:
    """Figure out the algorithm from the file name (e.g. track_PPO.zip -> PPO)."""
    name = Path(model_path).stem.upper()
    for algo in ALGOS:
        if algo in name:
            return algo
    return "PPO"


def main():
    p = argparse.ArgumentParser(description="Evaluate / watch a drone follow a circle.")
    p.add_argument("--model", required=True, help="Path to the saved .zip model.")
    p.add_argument("--algo", default=None, choices=list(ALGOS), help="Override the algorithm.")
    p.add_argument("--episodes", type=int, default=3, help="How many test flights to run.")
    p.add_argument("--render", action="store_true", help="Show the 3D viewer (needs a display).")
    # The path. Smaller period = faster target = harder to follow.
    p.add_argument("--radius", type=float, default=0.5, help="Circle radius (m).")
    p.add_argument("--period", type=float, default=6.0, help="Seconds per lap (smaller = faster).")
    p.add_argument("--height", type=float, default=1.0, help="Circle height (m).")
    # Look-ahead. 0 = hover-compatible observation (reuse a hover model).
    # For a trained tracker, set this to the value it was trained with.
    p.add_argument("--n-samples", type=int, default=0, help="Look-ahead points (match training).")
    p.add_argument("--samples-dt", type=float, default=0.1, help="Seconds between look-ahead points.")
    p.add_argument("--max-time", type=float, default=None, help="Episode length (default: 2 laps).")
    p.add_argument("--jax-device", default="cpu", choices=["cpu", "gpu"])
    p.add_argument("--physics", default="so_rpy", choices=["so_rpy", "first_principles"],
                   help="Must match the physics the model was trained on.")
    args = p.parse_args()

    algo = args.algo or guess_algo(args.model)
    # By default, fly for two full laps so students see it settle into the circle.
    max_time = args.max_time if args.max_time is not None else 2.0 * args.period

    env = make_traj_sb3_env(
        num_envs=1, device=args.jax_device, dynamics=args.physics,
        n_samples=args.n_samples, samples_dt=args.samples_dt,
        radius=args.radius, period=args.period, height=args.height,
        max_episode_time=max_time,
    )
    model = ALGOS[algo].load(args.model)
    print(f"Loaded {algo} model from {args.model}")
    print(f"Circle: radius={args.radius} m, period={args.period} s "
          f"({2 * np.pi * args.radius / args.period:.2f} m/s target speed), "
          f"look-ahead={args.n_samples}")

    all_mean, all_max = [], []
    for ep in range(args.episodes):
        obs = env.reset()
        done = False
        errors = []
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, dones, _ = env.step(action)
            if args.render:
                env.render()
            errors.append(float(target_distance(obs)[0]))  # dist to moving target
            done = bool(dones[0])
        errors = np.array(errors)
        all_mean.append(errors.mean())
        all_max.append(errors.max())
        print(f"  Flight {ep + 1}: mean error = {errors.mean() * 100:5.1f} cm | "
              f"worst = {errors.max() * 100:5.1f} cm")

    mean_cm = float(np.mean(all_mean)) * 100
    print("\n=== Results ===")
    print(f"Average tracking error : {mean_cm:.1f} cm from the moving target")
    print(f"Worst-case error       : {float(np.mean(all_max)) * 100:.1f} cm")
    verdict = ("tight tracking!" if mean_cm < 8 else
               "follows the circle" if mean_cm < 20 else
               "lags / cuts corners — try a slower circle or train a tracker")
    print(f"Verdict                : {verdict}")
    env.close()


if __name__ == "__main__":
    main()
