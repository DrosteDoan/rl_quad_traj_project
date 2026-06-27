"""Watch (and score) a trained hovering drone.

After training, run this to see how well the drone learned. It reports two
numbers per episode:

* **reward**   - the total score the drone collected (higher is better).
* **distance** - how far (in meters) it ended up from the target (lower is better).
                 A well-trained hover is usually a few centimeters (< 0.1 m).

Examples
--------
Score the trained drone over several runs (no graphics, works anywhere)::

    python -m k12_hover.eval_sb3 --model models/hover_PPO.zip

Watch it fly in a 3D window (needs a display)::

    python -m k12_hover.eval_sb3 --model models/hover_PPO.zip --render

On a headless machine you can still open a window with software rendering::

    MUJOCO_GL=egl python -m k12_hover.eval_sb3 --model models/hover_PPO.zip --render
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np  # noqa: E402
from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3  # noqa: E402

from k12_hover import make_sb3_env, target_distance  # noqa: E402

ALGOS = {"PPO": PPO, "A2C": A2C, "SAC": SAC, "TD3": TD3, "DDPG": DDPG}


def guess_algo(model_path: str) -> str:
    """Figure out the algorithm from the file name (e.g. hover_PPO.zip -> PPO)."""
    name = Path(model_path).stem.upper()
    for algo in ALGOS:
        if algo in name:
            return algo
    return "PPO"


def main():
    p = argparse.ArgumentParser(description="Evaluate / watch a trained hover policy.")
    p.add_argument("--model", required=True, help="Path to the saved .zip model.")
    p.add_argument("--algo", default=None, choices=list(ALGOS), help="Override the algorithm.")
    p.add_argument("--episodes", type=int, default=5, help="How many test flights to run.")
    p.add_argument("--render", action="store_true", help="Show the 3D viewer (needs a display).")
    p.add_argument("--jax-device", default="cpu", choices=["cpu", "gpu"])
    p.add_argument("--physics", default="so_rpy", choices=["so_rpy", "first_principles"],
                   help="Must match the physics the model was trained on.")
    args = p.parse_args()

    algo = args.algo or guess_algo(args.model)
    # One drone for a clean view; the same wrapper chain used in training.
    env = make_sb3_env(num_envs=1, device=args.jax_device, dynamics=args.physics)
    model = ALGOS[algo].load(args.model)
    print(f"Loaded {algo} model from {args.model}")

    returns, final_distances = [], []
    for ep in range(args.episodes):
        obs = env.reset()
        done = False
        total_reward = 0.0
        last_dist = float(target_distance(obs)[0])
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, dones, _ = env.step(action)
            if args.render:
                env.render()
            total_reward += float(reward[0])
            last_dist = float(target_distance(obs)[0])
            done = bool(dones[0])
        returns.append(total_reward)
        final_distances.append(last_dist)
        print(f"  Flight {ep + 1}: reward = {total_reward:7.2f} | "
              f"ended {last_dist * 100:5.1f} cm from target")

    print("\n=== Results ===")
    print(f"Average reward   : {np.mean(returns):.2f}")
    print(f"Average distance : {np.mean(final_distances) * 100:.1f} cm from target")
    verdict = "great hover!" if np.mean(final_distances) < 0.1 else (
        "getting there" if np.mean(final_distances) < 0.3 else "needs more training")
    print(f"Verdict          : {verdict}")
    env.close()


if __name__ == "__main__":
    main()
