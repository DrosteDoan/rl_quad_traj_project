"""xadapt_ctrl (muellerlab) as the low-level layer under our PID position loop.

'A Learning-based Quadcopter Controller with Extreme Adaptation' (Zhang et al.,
T-RO 2025). AdapLowLevelControl consumes [omega, proper_acc_z, cmd_bodyrates,
cmd_mass_norm_thrust] + 100-step history, outputs motor speeds (rad/s,
normalized by max). We convert to RPM for crazyflow's rotor_vel mode.

Stack: PID position (this file) -> attitude P -> CTBR -> xadapt -> motors.
Run rollouts at 500 Hz (their training timescale; history stats matter).
Requires: onnxruntime + the cloned repo (XADAPT_PATH, default ~/xadapt_ctrl).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

from crazy_track.controllers.base import Controller
from crazy_track.controllers.utils import GRAVITY, MASS, RPY_MAX, acc2attitude
from crazy_track.trajectories import Trajectory

RAD_S_TO_RPM = 60.0 / (2 * np.pi)
# Calibrated 2026-07-22 hover sweep: steady error 0.139/0.019/0.012/0.007/0.013 m
# at 2400/2800/3200/3600/4000 rad/s — 3600 is the optimum for cf21B_500.
MAX_MOTOR_RAD_S = 3600.0


class _QuadState:
    __slots__ = ("att", "omega", "proper_acc", "cmd_collective_thrust", "cmd_bodyrates")


class XAdaptPIDController(Controller):
    def __init__(self, kp: float = 16.0, kd: float = 8.0, ki: float = 6.0,
                 kp_att: float = 8.0, control_freq: int = 500, motor_order=(0, 1, 2, 3),
                 outer_decimation: int = 5):
        # Outer position/attitude loop runs at control_freq/outer_decimation
        # (100 Hz default) — running it at 500 Hz on white per-sample sensor
        # noise amplifies vel-noise x kd and the 34 Hz Lighthouse position
        # staircase into rate-command jitter (measured: RMSE 0.6-1.2 m).
        self.ki, self.int_max = ki, 1.0
        self._int_err = np.zeros(3)
        self.decim = outer_decimation
        self.outer_dt = outer_decimation / control_freq
        self._calls = 0
        self._rpyt = np.array([0.0, 0.0, 0.0, MASS * GRAVITY])
        self._cmd_rates = np.zeros(3)
        path = os.environ.get("XADAPT_PATH", str(Path.home() / "xadapt_ctrl"))
        if path not in sys.path:
            sys.path.insert(0, path)
        from xadapt_controller.controller import AdapLowLevelControl

        self.ll = AdapLowLevelControl()
        self.ll.set_max_motor_spd(MAX_MOTOR_RAD_S)
        self.kp, self.kd, self.kp_att = kp, kd, kp_att
        self.dt = 1.0 / control_freq
        self.motor_order = list(motor_order)  # crazyflow motor index for each xadapt output
        self._traj: Trajectory | None = None
        self._prev_thrust = MASS * GRAVITY

    def reset(self, trajectory: Trajectory) -> None:
        from xadapt_controller.controller import AdapLowLevelControl

        self._traj = trajectory
        self._prev_thrust = MASS * GRAVITY
        self._int_err = np.zeros(3)
        self._calls = 0
        self._rpyt = np.array([0.0, 0.0, 0.0, MASS * GRAVITY])
        self._cmd_rates = np.zeros(3)
        self.ll = AdapLowLevelControl()  # fresh history buffers per episode
        self.ll.set_max_motor_spd(MAX_MOTOR_RAD_S)

    def _outer_acc(self, pos: np.ndarray, vel: np.ndarray, t: float) -> np.ndarray:
        """Outer position loop -> desired CoM acceleration. Overridable (ADRC variant)."""
        err = self._traj.pos(t) - pos
        self._int_err = np.clip(self._int_err + err * self.outer_dt, -self.int_max, self.int_max)
        return (self._traj.acc(t) + self.kp * err + self.ki * self._int_err
                + self.kd * (self._traj.vel(t) - vel))

    def act(self, state: np.ndarray, t: float) -> np.ndarray:
        pos, vel, quat, omega = state[:3], state[3:6], state[6:10], state[10:13]
        if self._calls % self.decim == 0:  # outer loop at control_freq/decim
            a_cmd = self._outer_acc(pos, vel, t)
            self._rpyt = acc2attitude(a_cmd, quat)
            rot = R.from_quat(quat)
            rot_des = R.from_euler("xyz", [self._rpyt[0], self._rpyt[1], self._rpyt[2]])
            e_rotvec = (rot.inv() * rot_des).as_rotvec()
            self._cmd_rates = np.clip(self.kp_att * e_rotvec, -10.0, 10.0)
        self._calls += 1
        rpyt, cmd_rates = self._rpyt, self._cmd_rates

        # Proper acceleration (specific force) along body z. On hardware this is
        # the IMU accelerometer; differentiating (noisy) estimated velocity at
        # 500 Hz instead amplifies sensor noise ~1000x and poisons the adaptation
        # history (measured: RMSE 0.56-1.47 m under the Lighthouse model). The
        # commanded collective thrust / mass is exact in sim absent actuator
        # faults, and only the z-component is consumed by the model.
        s = _QuadState()
        s.att = quat
        s.omega = omega.astype(np.float32)
        s.proper_acc = np.array([0.0, 0.0, self._prev_thrust / MASS], dtype=np.float32)
        s.cmd_bodyrates = cmd_rates.astype(np.float32)
        s.cmd_collective_thrust = np.float32(rpyt[3] / MASS)  # mass-normalized
        self._prev_thrust = float(rpyt[3])

        spd = self.ll.run(s)  # rad/s in xadapt motor order
        rpm = np.clip(spd * RAD_S_TO_RPM, 0.0, MAX_MOTOR_RAD_S * RAD_S_TO_RPM)
        return rpm[self.motor_order].astype(np.float32)


class XAdaptADRCController(XAdaptPIDController):
    """ADRC outer loop over the xadapt low-level: explicit external-force
    cancellation (velocity ESO) stacked on airframe adaptation. The ESO also
    absorbs the thrust-calibration offset, replacing the PID integrator.
    """

    def __init__(self, omega_obs: float = 7.0, sigma_max: float = 3.0, **kw):
        super().__init__(**kw)
        self.l1g, self.l2g = 2 * omega_obs, omega_obs**2
        self.sigma_max = sigma_max
        self._v_hat = np.zeros(3)
        self._sigma = np.zeros(3)
        self._u_prev = np.zeros(3)

    def reset(self, trajectory: Trajectory) -> None:
        super().reset(trajectory)
        self._v_hat = np.zeros(3)
        self._sigma = np.zeros(3)
        self._u_prev = np.zeros(3)

    def _outer_acc(self, pos: np.ndarray, vel: np.ndarray, t: float) -> np.ndarray:
        e_v = vel - self._v_hat
        self._v_hat += self.outer_dt * (self._u_prev + self._sigma + self.l1g * e_v)
        self._sigma = np.clip(self._sigma + self.outer_dt * self.l2g * e_v,
                              -self.sigma_max, self.sigma_max)
        a_cmd = (self._traj.acc(t) + self.kp * (self._traj.pos(t) - pos)
                 + self.kd * (self._traj.vel(t) - vel) - self._sigma)
        self._u_prev = a_cmd
        return a_cmd
