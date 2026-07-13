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
from k12_hover.trajectory_env import TrajectoryEnv
from k12_hover.wrappers import OBS_ORDER, FlattenObservation, NormalizeActions

__all__ = [
    "HoverEnv",
    "TrajectoryEnv",
    "FlattenObservation",
    "NormalizeActions",
    "make_sb3_env",
    "make_torch_env",
    "make_traj_sb3_env",
    "make_traj_torch_env",
]


def _build_core_env(num_envs: int, device: str, reward_coefs: dict | None, env_kwargs: dict):
    """Build HoverEnv + the action/observation wrappers shared by both paths."""
    reward_coefs = dict(reward_coefs or {})
    env = HoverEnv(num_envs=num_envs, device=device, **reward_coefs, **env_kwargs)
    env = NormalizeActions(env)      # agent acts in [-1, 1]
    env = FlattenObservation(env)    # dict obs -> flat vector
    return env


def _build_core_traj_env(num_envs: int, device: str, reward_coefs: dict | None, env_kwargs: dict):
    """Build TrajectoryEnv + wrappers. Adds the look-ahead to the flat vector when on.

    When ``n_samples>0`` the observation gains a ``local_samples`` block; we append
    it to the flat vector AFTER the usual 13 hover numbers. Keeping the first 13 in
    the same order means a policy trained on the plain hover observation still lines
    up with slots 0..12 (handy for reusing a hover brain, see ``make_traj_sb3_env``).
    """
    reward_coefs = dict(reward_coefs or {})
    env = TrajectoryEnv(num_envs=num_envs, device=device, **reward_coefs, **env_kwargs)
    env = NormalizeActions(env)      # agent acts in [-1, 1]
    obs_order = OBS_ORDER + ("local_samples",) if env.unwrapped.n_samples > 0 else OBS_ORDER
    env = FlattenObservation(env, obs_order=obs_order)
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


def make_traj_sb3_env(
    num_envs: int = 64,
    device: str = "cpu",
    reward_coefs: dict | None = None,
    tilt_action_coef: float = 0.0,
    **env_kwargs,
):
    """Build a trajectory-following env ready for Stable-Baselines3.

    Returns an SB3 ``VecEnv`` you can hand straight to ``PPO(...)``.

    Set ``n_samples=0`` (the default) to keep the observation identical to the
    hover task — a trained hover model can then be evaluated on it with no
    retraining (Lesson 8). Set ``n_samples>0`` to add the path look-ahead and
    train a dedicated tracker (Lesson 9).

    ``tilt_action_coef`` penalizes large *commanded* tilts (helps sim-to-real).
    """
    from k12_hover.sb3_adapter import CrazyflowSB3VecEnv

    env = _build_core_traj_env(num_envs, device, reward_coefs, env_kwargs)
    return CrazyflowSB3VecEnv(env, tilt_action_coef=tilt_action_coef)


def make_traj_torch_env(
    num_envs: int = 256,
    device: str = "cpu",
    torch_device: str = "cpu",
    reward_coefs: dict | None = None,
    **env_kwargs,
):
    """Build a trajectory-following env ready for the CleanRL (PyTorch) path."""
    import torch
    from gymnasium.wrappers.vector import JaxToTorch

    env = _build_core_traj_env(num_envs, device, reward_coefs, env_kwargs)
    env = JaxToTorch(env, device=torch.device(torch_device))
    return env


def target_distance(obs: np.ndarray) -> np.ndarray:
    """Helper for evaluation: distance to target, read from the flat observation.

    The first 3 numbers of the observation are ``rel_pos`` (target - position),
    so its length is exactly how far the drone is from the target.
    """
    obs = np.asarray(obs)
    return np.linalg.norm(obs[..., :3], axis=-1)
