from __future__ import annotations

import numpy as np

from crazy_track.controllers.base import Controller
from crazy_track.controllers.utils import acc2attitude
from crazy_track.trajectories import Trajectory


class PIDController(Controller):
    """Cascaded position PID with velocity/acceleration feedforward.

    a_cmd = ref_acc + kp*(ref_pos - pos) + kd*(ref_vel - vel) + ki*int_err
    then converted to [roll, pitch, yaw, thrust] for the firmware attitude loop.
    Gains are in acceleration units (1/s^2, 1/s): kp = omega^2, kd = 2*zeta*omega.
    """

    def __init__(self, kp: float = 16.0, kd: float = 8.0, ki: float = 2.0,
                 int_max: float = 0.5, control_freq: int = 100):
        self.kp, self.kd, self.ki = kp, kd, ki
        self.int_max = int_max
        self.dt = 1.0 / control_freq
        self._traj: Trajectory | None = None
        self._int_err = np.zeros(3)

    def reset(self, trajectory: Trajectory) -> None:
        self._traj = trajectory
        self._int_err = np.zeros(3)

    def act(self, state: np.ndarray, t: float) -> np.ndarray:
        pos, vel, quat = state[:3], state[3:6], state[6:10]
        err = self._traj.pos(t) - pos
        self._int_err = np.clip(self._int_err + err * self.dt, -self.int_max, self.int_max)
        a_cmd = (
            self._traj.acc(t)
            + self.kp * err
            + self.kd * (self._traj.vel(t) - vel)
            + self.ki * self._int_err
        )
        return acc2attitude(a_cmd, quat)
