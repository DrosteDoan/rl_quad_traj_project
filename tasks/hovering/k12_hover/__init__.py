"""k12_hover: a friendly reinforcement-learning playground for hovering a drone.

Big picture
-----------
We teach a Crazyflie 2.1 Brushless (simulated in Crazyflow) to hover using
reinforcement learning. The pieces:

* ``HoverEnv``            - the task (what the drone sees / does / is rewarded for)
* ``FlattenObservation`` - turns the observation into a flat vector for the network
* ``NormalizeActions``   - lets the agent act in the easy range [-1, 1]
* ``make_sb3_env``       - assembles everything for Stable-Baselines3 (the easy path)
* ``make_torch_env``     - assembles everything for the CleanRL script (advanced path)

Start with ``lessons/01_what_is_rl.md``.
"""

from __future__ import annotations

import numpy as np

from k12_hover.hover_env import HoverEnv
from k12_hover.wrappers import FlattenObservation, NormalizeActions

__all__ = [
    "HoverEnv",
    "FlattenObservation",
    "NormalizeActions",
    "make_sb3_env",
    "make_torch_env",
]


def _build_core_env(num_envs: int, device: str, reward_coefs: dict | None, env_kwargs: dict):
    """Build HoverEnv + the action/observation wrappers shared by both paths."""
    reward_coefs = dict(reward_coefs or {})
    env = HoverEnv(num_envs=num_envs, device=device, **reward_coefs, **env_kwargs)
    env = NormalizeActions(env)      # agent acts in [-1, 1]
    env = FlattenObservation(env)    # dict obs -> flat vector
    return env


def make_sb3_env(
    num_envs: int = 64,
    device: str = "cpu",
    reward_coefs: dict | None = None,
    tilt_action_coef: float = 0.0,
    **env_kwargs,
):
    """Build a hover env ready for Stable-Baselines3.

    Returns an SB3 ``VecEnv`` you can hand straight to ``PPO(...)``.

    ``tilt_action_coef`` penalizes large *commanded* tilts (helps sim-to-real).
    """
    from k12_hover.sb3_adapter import CrazyflowSB3VecEnv

    env = _build_core_env(num_envs, device, reward_coefs, env_kwargs)
    # The adapter handles JAX<->NumPy conversion (and CPU/GPU placement) itself.
    return CrazyflowSB3VecEnv(env, tilt_action_coef=tilt_action_coef)


def make_torch_env(
    num_envs: int = 256,
    device: str = "cpu",
    torch_device: str = "cpu",
    reward_coefs: dict | None = None,
    **env_kwargs,
):
    """Build a hover env ready for the CleanRL (PyTorch) training script.

    Returns a Gymnasium vector env whose observations/actions are torch tensors.
    """
    import torch
    from gymnasium.wrappers.vector import JaxToTorch

    env = _build_core_env(num_envs, device, reward_coefs, env_kwargs)
    env = JaxToTorch(env, device=torch.device(torch_device))
    return env


def target_distance(obs: np.ndarray) -> np.ndarray:
    """Helper for evaluation: distance to target, read from the flat observation.

    The first 3 numbers of the observation are ``rel_pos`` (target - position),
    so its length is exactly how far the drone is from the target.
    """
    obs = np.asarray(obs)
    return np.linalg.norm(obs[..., :3], axis=-1)
