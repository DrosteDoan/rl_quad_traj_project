from __future__ import annotations

import numpy as np

from crazy_track.controllers.base import Controller
from crazy_track.controllers.l1 import L1Estimator
from crazy_track.controllers.utils import (MASS, RATE_MAX, THRUST_MAX, THRUST_MIN,
                                           ctbr_to_force_torque)
from crazy_track.envs.datt_env import WINDOW, WINDOW_DT
from crazy_track.trajectories import Trajectory


class DATTAcroController(Controller):
    """Runs a CTBR-trained acro policy: obs as datt_env v3, action [thrust, body
    rates] -> onboard-style rate loop -> [fc, tx, ty, tz] (needs a force_torque sim).
    """

    def __init__(self, model_path: str, control_freq: int = 100):
        from stable_baselines3 import PPO

        self.model = PPO.load(model_path, device="cpu")
        self._t_offsets = WINDOW_DT * np.arange(1, WINDOW + 1)
        self.l1 = L1Estimator(MASS, n=1, dt=1.0 / control_freq)
        obs_dim = int(np.prod(self.model.observation_space.shape))
        base_acro2 = 3 + 3 + 4 + 3 * WINDOW + 3 + 3  # +attitude-error rotvec
        self.acro4 = obs_dim == base_acro2 + 6  # +maneuver descriptor block
        self.acro2 = self.acro4 or obs_dim == base_acro2
        self._traj: Trajectory | None = None
        self._last_thrust = MASS * 9.81

    def reset(self, trajectory: Trajectory) -> None:
        self._traj = trajectory
        self.l1.reset()
        self._last_thrust = MASS * 9.81

    def act(self, state: np.ndarray, t: float) -> np.ndarray:
        pos, vel, quat, omega = state[:3], state[3:6], state[6:10], state[10:13]
        win = self._traj.pos(t + self._t_offsets) - pos
        sigma = self.l1.update(vel, quat, np.array([self._last_thrust]))
        parts = [self._traj.pos(t) - pos, vel, quat, win.ravel(), sigma[0]]
        if self.acro2:
            from scipy.spatial.transform import Rotation as R

            ref = R.from_rotvec(self._traj.att_ref_rotvec(t))
            parts.append((ref.inv() * R.from_quat(quat)).as_rotvec())
        if self.acro4:
            # maneuver descriptor, mirroring datt_env._maneuver_descriptor
            desc = np.zeros(6)
            tr = self._traj
            if hasattr(tr, "maneuver_descriptor"):  # FreestyleTrajectory (multi-window)
                desc = np.asarray(tr.maneuver_descriptor(t), dtype=np.float64)
            elif hasattr(tr, "t_rot_start"):  # BallisticFlipTrajectory
                desc[tr.axis] = tr.direction
                desc[3] = np.clip(tr.t_rot_start - t, 0.0, 1.0)
                desc[4] = np.clip((t - tr.t_rot_start) / tr.Tb, 0.0, 1.0)
                desc[5] = 1.0 if tr.t_rot_start <= t <= tr.t_rot_end else 0.0
            parts.append(desc)
        obs = np.concatenate(parts).astype(np.float32)
        a, _ = self.model.predict(obs, deterministic=True)
        thrust = THRUST_MIN + (np.clip(a[0], -1, 1) + 1) * 0.5 * (THRUST_MAX - THRUST_MIN)
        w_des = np.clip(a[1:4], -1, 1) * RATE_MAX
        self._last_thrust = float(thrust)
        return ctbr_to_force_torque(np.array(thrust), w_des, omega)
