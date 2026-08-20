from __future__ import annotations

import numpy as np

from crazy_track.controllers.base import Controller
from crazy_track.controllers.utils import MASS, acc2attitude
from crazy_track.trajectories import Trajectory


class ADRCController(Controller):
    """Active Disturbance Rejection Control (translational) with selectable estimator.

    Estimators for the total disturbance acceleration sigma:

    - "eso" (classic ADRC): reduced-order ESO on measured velocity,
          v_hat' = u + sigma + l1 (v - v_hat);  sigma' = l2 (v - v_hat)
      with l = [2w, w^2]. Single knob w couples estimation speed AND noise
      filtering — under noisy sensors (Lighthouse) sigma gets jittery.
    - "l1": piecewise-constant L1 adaptation (controllers/l1.py). Decouples
      fast adaptation from noise rejection via an explicit low-pass filter.

    Additionally, `sigma_lpf_hz` low-pass filters the estimate before
    cancellation (the "L1 filtering layer on top of the ESO" hybrid).
    Control law: a_cmd = ref_acc + kp*e + kd*edot - sigma_filtered.
    """

    def __init__(self, kp: float = 16.0, kd: float = 8.0, omega_obs: float = 7.0,
                 sigma_max: float = 3.0, control_freq: int = 100,
                 estimator: str = "eso", sigma_lpf_hz: float | None = None,
                 a_s: float = -5.0, l1_cutoff_hz: float = 2.0):
        # omega_obs=7 is the balanced default from the 2026-07-22 sweep:
        # low w (3-5) wins under sensor noise + static disturbance, high w (10)
        # wins under 0.7 Hz gusts; w=7 is within ~15% of the best everywhere.
        self.kp, self.kd = kp, kd
        self.l1_gain, self.l2_gain = 2 * omega_obs, omega_obs**2
        self.sigma_max = sigma_max
        self.dt = 1.0 / control_freq
        self.estimator = estimator
        self.alpha = (None if sigma_lpf_hz is None
                      else 1.0 - np.exp(-2 * np.pi * sigma_lpf_hz * self.dt))
        if estimator == "l1":
            from crazy_track.controllers.l1 import L1Estimator

            self._l1 = L1Estimator(MASS, n=1, a_s=a_s, cutoff_hz=l1_cutoff_hz, dt=self.dt)
        if estimator == "adaptive":
            # Innovation-scheduled bandwidth: a persistent disturbance biases the
            # innovation mean; sensor noise is zero-mean. w = w_min..w_max scaled
            # by |mean|/std of the innovation over a short window. Measured fixed-w
            # tradeoff: w=3-5 wins noise/static cells, w=10 wins gusts; no fixed w
            # wins everywhere (2026-07-22 sweep).
            self.w_min, self.w_max = 3.0, 12.0
            self.win = int(0.25 / self.dt)  # 0.25 s innovation window
            self._innov: list[np.ndarray] = []
        self._traj: Trajectory | None = None
        self._reset_states()

    def _reset_states(self) -> None:
        self._v_hat = np.zeros(3)
        self._sigma = np.zeros(3)
        self._sigma_f = np.zeros(3)
        self._u_prev = np.zeros(3)
        self._last_thrust = MASS * 9.81

    def reset(self, trajectory: Trajectory) -> None:
        self._traj = trajectory
        self._reset_states()
        if self.estimator == "l1":
            self._l1.reset()
        if self.estimator == "adaptive":
            self._innov = []

    def act(self, state: np.ndarray, t: float) -> np.ndarray:
        pos, vel, quat = state[:3], state[3:6], state[6:10]

        if self.estimator == "l1":
            sigma = self._l1.update(vel, quat, np.array([self._last_thrust]))[0]
        else:
            e_v = vel - self._v_hat
            if self.estimator == "adaptive":
                self._innov.append(e_v.copy())
                if len(self._innov) > self.win:
                    self._innov.pop(0)
                inn = np.asarray(self._innov)
                ratio = float(np.max(np.abs(inn.mean(0)) / (inn.std(0) + 1e-4)))
                w = self.w_min + (self.w_max - self.w_min) * min(ratio, 1.0)
                self.l1_gain, self.l2_gain = 2 * w, w**2
            self._v_hat += self.dt * (self._u_prev + self._sigma + self.l1_gain * e_v)
            self._sigma = np.clip(self._sigma + self.dt * self.l2_gain * e_v,
                                  -self.sigma_max, self.sigma_max)
            sigma = self._sigma
        if self.alpha is not None:
            self._sigma_f = (1 - self.alpha) * self._sigma_f + self.alpha * sigma
            sigma = self._sigma_f

        a_cmd = (
            self._traj.acc(t)
            + self.kp * (self._traj.pos(t) - pos)
            + self.kd * (self._traj.vel(t) - vel)
            - np.clip(sigma, -self.sigma_max, self.sigma_max)
        )
        self._u_prev = a_cmd
        action = acc2attitude(a_cmd, quat)
        self._last_thrust = float(action[3])
        return action
