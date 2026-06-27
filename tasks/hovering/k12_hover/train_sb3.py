"""Train a drone to hover using Stable-Baselines3 (the easy path).

This is the script most students will run. It mirrors how gym-pybullet-drones
trains with SB3: build an environment, pick an algorithm, call ``.learn()``.

Examples
--------
Train with the default settings (PPO, a few minutes on a GPU)::

    python -m k12_hover.train_sb3

Train faster on the GPU with more parallel drones::

    python -m k12_hover.train_sb3 --jax-device gpu --num-envs 256 --timesteps 2000000

Try a different algorithm (this is the "different RL architectures" lesson)::

    python -m k12_hover.train_sb3 --algo A2C
    python -m k12_hover.train_sb3 --algo SAC --num-envs 8

Change the reward (the "understand the criteria" lesson): punish moving fast so
the drone learns to hold *still*, not just pass through the target::

    python -m k12_hover.train_sb3 --velocity-coef 0.1 --spin-coef 0.02
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

# Stops JAX from grabbing all GPU memory at once (plays nicely with PyTorch).
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3  # noqa: E402

from k12_hover import make_sb3_env  # noqa: E402

# The algorithms a student can pick with --algo. PPO and A2C are "on-policy" and
# love many parallel drones. SAC/TD3/DDPG are "off-policy"; use fewer drones.
ALGOS = {"PPO": PPO, "A2C": A2C, "SAC": SAC, "TD3": TD3, "DDPG": DDPG}

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def main():
    p = argparse.ArgumentParser(description="Train a hovering drone with SB3.")
    # What and how long to train.
    p.add_argument("--algo", default="PPO", choices=list(ALGOS), help="RL algorithm.")
    p.add_argument("--timesteps", type=int, default=1_000_000, help="Total training steps.")
    p.add_argument("--num-envs", type=int, default=64, help="Parallel drones (more = faster).")
    p.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility.")
    # Where to run the physics and the network.
    p.add_argument("--physics", default="so_rpy", choices=["so_rpy", "first_principles"],
                   help="Physics model. 'first_principles' (rigid-body) transfers better to CrazySim.")
    p.add_argument("--jax-device", default="cpu", choices=["cpu", "gpu"], help="Physics device.")
    # The policy is a small MLP, which SB3 runs fastest on the CPU. The GPU is
    # better used for the physics (--jax-device gpu) or for the CleanRL script.
    p.add_argument("--torch-device", default="cpu", help="Network device: cpu/cuda.")
    # The reward knobs (the 'criteria' the drone is graded on).
    p.add_argument("--distance-coef", type=float, default=2.0, help="Reward for being close.")
    p.add_argument("--velocity-coef", type=float, default=0.0, help="Penalty for moving fast.")
    p.add_argument("--spin-coef", type=float, default=0.0, help="Penalty for spinning.")
    p.add_argument("--tilt-coef", type=float, default=0.0, help="Penalty for tilting over.")
    p.add_argument("--yaw-coef", type=float, default=0.0, help="Penalty for turning away from yaw 0.")
    p.add_argument("--tilt-action-coef", type=float, default=0.0,
                   help="Penalty on COMMANDED tilt magnitude (keeps tilts small; helps sim-to-real).")
    # Saving / logging.
    p.add_argument("--save-name", default=None, help="Checkpoint name (default: hover_<ALGO>).")
    p.add_argument("--tensorboard", action="store_true", help="Log to TensorBoard.")
    args = p.parse_args()

    save_name = args.save_name or f"hover_{args.algo}"
    MODELS_DIR.mkdir(exist_ok=True)
    save_path = MODELS_DIR / save_name

    reward_coefs = {
        "distance_coef": args.distance_coef,
        "velocity_coef": args.velocity_coef,
        "spin_coef": args.spin_coef,
        "tilt_coef": args.tilt_coef,
        "yaw_coef": args.yaw_coef,
    }

    print(f"Building {args.num_envs} parallel drones on '{args.jax_device}'...")
    env = make_sb3_env(
        num_envs=args.num_envs, device=args.jax_device, reward_coefs=reward_coefs,
        dynamics=args.physics, tilt_action_coef=args.tilt_action_coef,
    )

    algo_cls = ALGOS[args.algo]
    tb_log = str(MODELS_DIR / "tensorboard") if args.tensorboard else None

    # Hyperparameters. SB3's defaults assume ONE environment; with many parallel
    # drones we must tune the on-policy settings or the update step crawls.
    extra = {}
    if args.algo == "PPO":
        # Collect a modest rollout, then learn from it in big chunks (not the
        # tiny 64-sample default, which would do tens of thousands of updates).
        rollout = args.num_envs * 32                  # total samples per update
        extra = dict(
            n_steps=32,                                # steps per drone per update
            batch_size=max(256, rollout // 4),         # big minibatches = fast on CPU
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            learning_rate=3e-4,
        )
    elif args.algo == "A2C":
        extra = dict(n_steps=8, gamma=0.99)

    model = algo_cls(
        "MlpPolicy",            # a small fully-connected neural network
        env,
        verbose=1,              # print progress every rollout
        seed=args.seed,
        device=args.torch_device,
        tensorboard_log=tb_log,
        **extra,
    )

    print(f"Training {args.algo} for {args.timesteps:,} steps. Watch 'ep_rew_mean' go up!")
    model.learn(total_timesteps=args.timesteps, progress_bar=False)

    model.save(save_path)
    print(f"\nDone. Saved the trained brain to: {save_path}.zip")
    print(f"Now watch it fly:  python -m k12_hover.eval_sb3 --model {save_path}.zip")
    env.close()


if __name__ == "__main__":
    main()
