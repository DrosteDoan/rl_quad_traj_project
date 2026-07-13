"""Train a drone to FOLLOW A CIRCLE using Stable-Baselines3 (Lesson 9).

This is the twin of ``train_sb3.py`` (the hover trainer). It uses the same
algorithm, the same wrappers, and the same reward knobs — the only new ideas are:

* the target **moves** around a circle (``--radius``, ``--period``), and
* the observation can include a **look-ahead** (``--n-samples``): the next few
  points on the path, so the policy can *anticipate* the turns instead of always
  reacting a step late. This is what lets a trained tracker beat the reused hover
  brain on a fast circle.

Examples
--------
Train a tracker that can see the path ahead (the recommended starting point)::

    python -m k12_hover.train_traj_sb3 --jax-device gpu --num-envs 256 \
        --timesteps 3000000 --n-samples 10 --period 4 --save-name track_PPO

Make it easier first (slow circle), then harder (fast circle) — a mini curriculum::

    python -m k12_hover.train_traj_sb3 --period 8 --save-name track_slow
    python -m k12_hover.train_traj_sb3 --period 3 --save-name track_fast

Then watch it::

    python -m k12_hover.eval_traj --model models/track_PPO.zip --n-samples 10 --period 4 --render
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3  # noqa: E402

from k12_hover import make_traj_sb3_env  # noqa: E402

ALGOS = {"PPO": PPO, "A2C": A2C, "SAC": SAC, "TD3": TD3, "DDPG": DDPG}

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def main():
    p = argparse.ArgumentParser(description="Train a circle-following drone with SB3.")
    # What and how long to train.
    p.add_argument("--algo", default="PPO", choices=list(ALGOS), help="RL algorithm.")
    p.add_argument("--timesteps", type=int, default=3_000_000, help="Total training steps.")
    p.add_argument("--num-envs", type=int, default=128, help="Parallel drones (more = faster).")
    p.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility.")
    # The path the target travels.
    p.add_argument("--radius", type=float, default=0.5, help="Circle radius (m).")
    p.add_argument("--period", type=float, default=4.0, help="Seconds per lap (smaller = harder).")
    p.add_argument("--height", type=float, default=1.0, help="Circle height (m).")
    # The look-ahead (how much of the path ahead the drone sees). 0 = hover-style obs.
    p.add_argument("--n-samples", type=int, default=10, help="Look-ahead points (0 = none).")
    p.add_argument("--samples-dt", type=float, default=0.1, help="Seconds between look-ahead points.")
    p.add_argument("--max-time", type=float, default=None, help="Episode length (default: 2 laps).")
    # Where to run the physics and the network.
    p.add_argument("--physics", default="so_rpy", choices=["so_rpy", "first_principles"],
                   help="Physics model. 'first_principles' transfers better to CrazySim.")
    p.add_argument("--jax-device", default="cpu", choices=["cpu", "gpu"], help="Physics device.")
    p.add_argument("--torch-device", default="cpu", help="Network device: cpu/cuda.")
    # The reward knobs (same as hover — the 'criteria' the drone is graded on).
    p.add_argument("--distance-coef", type=float, default=2.0, help="Reward for being close.")
    p.add_argument("--velocity-coef", type=float, default=0.0, help="Penalty for moving fast.")
    p.add_argument("--spin-coef", type=float, default=0.0, help="Penalty for spinning.")
    p.add_argument("--tilt-coef", type=float, default=0.0, help="Penalty for tilting over.")
    p.add_argument("--yaw-coef", type=float, default=0.0, help="Penalty for turning away from yaw 0.")
    p.add_argument("--tilt-action-coef", type=float, default=0.0,
                   help="Penalty on COMMANDED tilt magnitude (keeps tilts small; helps sim-to-real).")
    # Saving / logging.
    p.add_argument("--save-name", default=None, help="Checkpoint name (default: track_<ALGO>).")
    p.add_argument("--tensorboard", action="store_true", help="Log to TensorBoard.")
    args = p.parse_args()

    save_name = args.save_name or f"track_{args.algo}"
    MODELS_DIR.mkdir(exist_ok=True)
    save_path = MODELS_DIR / save_name

    # Default to two full laps per practice run so the drone experiences the whole path.
    max_time = args.max_time if args.max_time is not None else 2.0 * args.period

    reward_coefs = {
        "distance_coef": args.distance_coef,
        "velocity_coef": args.velocity_coef,
        "spin_coef": args.spin_coef,
        "tilt_coef": args.tilt_coef,
        "yaw_coef": args.yaw_coef,
    }

    print(f"Building {args.num_envs} parallel drones on '{args.jax_device}'...")
    print(f"Circle: radius={args.radius} m, period={args.period} s, look-ahead={args.n_samples}")
    env = make_traj_sb3_env(
        num_envs=args.num_envs, device=args.jax_device, reward_coefs=reward_coefs,
        dynamics=args.physics, tilt_action_coef=args.tilt_action_coef,
        radius=args.radius, period=args.period, height=args.height,
        n_samples=args.n_samples, samples_dt=args.samples_dt, max_episode_time=max_time,
    )

    algo_cls = ALGOS[args.algo]
    tb_log = str(MODELS_DIR / "tensorboard") if args.tensorboard else None

    # Same on-policy tuning as the hover trainer (SB3 defaults assume ONE env).
    extra = {}
    if args.algo == "PPO":
        rollout = args.num_envs * 32
        extra = dict(
            n_steps=32,
            batch_size=max(256, rollout // 4),
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            learning_rate=3e-4,
        )
    elif args.algo == "A2C":
        extra = dict(n_steps=8, gamma=0.99)

    model = algo_cls(
        "MlpPolicy",
        env,
        verbose=1,
        seed=args.seed,
        device=args.torch_device,
        tensorboard_log=tb_log,
        **extra,
    )

    print(f"Training {args.algo} for {args.timesteps:,} steps. Watch 'ep_rew_mean' go up!")
    model.learn(total_timesteps=args.timesteps, progress_bar=False)

    model.save(save_path)
    print(f"\nDone. Saved the trained brain to: {save_path}.zip")
    print("Now watch it fly:  python -m k12_hover.eval_traj "
          f"--model {save_path}.zip --n-samples {args.n_samples} "
          f"--period {args.period} --render")
    env.close()


if __name__ == "__main__":
    main()
