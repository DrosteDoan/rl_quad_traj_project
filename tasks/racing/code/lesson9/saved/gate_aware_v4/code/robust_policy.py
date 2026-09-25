"""Inference wrapper for a `RobustTrackingEnv`-trained policy (Lesson 9 phase 6).

Deliberately NOT a reuse of `crazy_track.controllers.datt.DATTPolicyController`: that class imports
`WINDOW`/`WINDOW_DT` directly from `crazy_track.envs.datt_env` (0.6 s) and would silently feed a robust
model -- trained with 0.8 s spacing (`robust_env.WINDOW_DT`) -- the WRONG reference window. Same
observation dimensionality (WINDOW's POINT COUNT is unchanged, only the spacing), so there is no shape
error to catch that mistake; it would just quietly degrade tracking. This class is the vendored one's
logic with the window spacing swapped in, nothing else changed.

    ctrl = RobustPolicyController("<path>/datt_ppo_final.zip", control_freq=100)

`driver.py` routes a `robust:<path>` spec to this class instead of `datt:<path>` (which stays on the
vendored controller, for any non-robust policy).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from robust_env import WINDOW_DT  # noqa: E402

from crazy_track.controllers.base import Controller  # noqa: E402
from crazy_track.controllers.utils import RPY_MAX, THRUST_MAX, THRUST_MIN  # noqa: E402
from crazy_track.envs.datt_env import WINDOW  # noqa: E402
from crazy_track.trajectories import Trajectory  # noqa: E402


class RobustPolicyController(Controller):
    """Runs a trained RobustTrackingEnv PPO policy: identical to DATTPolicyController except `_t_offsets`
    uses `robust_env.WINDOW_DT` (0.8 s) instead of the vendored `datt_env.WINDOW_DT` (0.6 s)."""

    def __init__(self, model_path: str, control_freq: int = 100):
        from stable_baselines3 import PPO

        self.model = PPO.load(model_path, device="cpu")
        self._t_offsets = WINDOW_DT * np.arange(1, WINDOW + 1)
        self._traj: Trajectory | None = None
        base_v3 = 3 + 3 + 4 + 3 * WINDOW + 3
        obs_dim = int(np.prod(self.model.observation_space.shape))
        if obs_dim not in (base_v3, base_v3 + 13):
            raise ValueError(f"RobustPolicyController expects a plain v3/v5 obs dim ({base_v3} or "
                             f"{base_v3 + 13}), got {obs_dim} -- this loader does not handle stacked (v6) "
                             f"or recurrent (v7) robust models")
        self.pad = obs_dim - base_v3
        from crazy_track.controllers.l1 import L1Estimator

        self.l1 = L1Estimator(mass=0.04338, n=1, dt=1.0 / control_freq)
        self._last_thrust = 0.04338 * 9.81

    def reset(self, trajectory: Trajectory) -> None:
        self._traj = trajectory
        self.l1.reset()
        self._last_thrust = 0.04338 * 9.81

    def act(self, state: np.ndarray, t: float) -> np.ndarray:
        pos, vel, quat = state[:3], state[3:6], state[6:10]
        win = self._traj.pos(t + self._t_offsets) - pos
        sigma = self.l1.update(vel, quat, np.array([self._last_thrust]))
        frame = np.concatenate([self._traj.pos(t) - pos, vel, quat, win.ravel(), sigma[0]]).astype(np.float32)
        obs = np.concatenate([frame, np.zeros(self.pad, dtype=np.float32)]) if self.pad else frame
        a, _ = self.model.predict(obs, deterministic=True)
        rpy = np.array([a[0] * RPY_MAX * 0.7, a[1] * RPY_MAX * 0.7, 0.0])
        thrust = THRUST_MIN + (np.clip(a[3], -1, 1) + 1) * 0.5 * (THRUST_MAX - THRUST_MIN)
        self._last_thrust = float(thrust)
        return np.array([rpy[0], rpy[1], rpy[2], thrust])
