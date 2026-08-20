"""L1 adaptive disturbance estimator (translational), scalar and batched.

Piecewise-constant adaptation law (as in DATT, arXiv:2310.09053): a velocity
predictor driven by the commanded thrust produces a prediction error; the
adaptation law inverts the error dynamics over one step, then a low-pass
filter yields the disturbance-acceleration estimate sigma_f (m/s^2).
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R

GRAVITY = np.array([0.0, 0.0, -9.81])


class L1Estimator:
    """Batched over N worlds; use N=1 for scalar use. States shape (N, 3)."""

    def __init__(self, mass: float, n: int = 1, a_s: float = -5.0,
                 cutoff_hz: float = 4.0, dt: float = 0.01):
        self.mass, self.n, self.a_s, self.dt = mass, n, a_s, dt
        self.alpha = 1.0 - np.exp(-2 * np.pi * cutoff_hz * dt)
        self.reset()

    def reset(self) -> None:
        self.v_hat = np.zeros((self.n, 3))
        self.sigma_hat = np.zeros((self.n, 3))
        self.sigma_f = np.zeros((self.n, 3))

    def update(self, vel: np.ndarray, quat_xyzw: np.ndarray,
               thrust_cmd: np.ndarray) -> np.ndarray:
        """vel (N,3), quat (N,4), thrust_cmd (N,) in Newtons. Returns sigma_f (N,3)."""
        vel = np.atleast_2d(vel)
        quat = np.atleast_2d(quat_xyzw)
        thrust = np.atleast_1d(thrust_cmd)
        z_b = R.from_quat(quat).as_matrix()[..., :, 2]
        v_err = self.v_hat - vel
        v_hat_dot = thrust[:, None] * z_b / self.mass + GRAVITY + self.sigma_hat \
            + self.a_s * v_err
        self.v_hat = self.v_hat + self.dt * v_hat_dot
        e = np.exp(self.a_s * self.dt)
        self.sigma_hat = -(self.a_s / (e - 1.0)) * e * (self.v_hat - vel)
        self.sigma_f = (1 - self.alpha) * self.sigma_f + self.alpha * self.sigma_hat
        return self.sigma_f
