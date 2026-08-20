from __future__ import annotations

import numpy as np

from crazy_track.controllers.base import Controller
from crazy_track.controllers.utils import THRUST_MAX, THRUST_MIN
from crazy_track.trajectories import Trajectory


def _load_so_rpy_params(drone: str = "cf21B_500") -> dict:
    from crazyflow.dynamics.core import load_params
    from crazyflow.dynamics.so_rpy.dynamics import dynamics as so_rpy_dyn

    p = load_params(so_rpy_dyn, drone)
    return {k: np.asarray(v, dtype=np.float64).squeeze() for k, v in p.items()}


class MPPIL1Controller(Controller):
    """MPPI over crazyflow's identified so_rpy model + L1 adaptive augmentation.

    Internal model state: [pos(3), rpy(3), vel(3), drpy(3)]; input
    [roll, pitch, yaw, thrust_N] — identical to the sim's attitude interface
    (this mirrors lsy_drone_racing, which uses the same identified model for MPC).

    L1: a piecewise-constant adaptation law estimates the translational
    disturbance acceleration from velocity prediction error; the low-pass
    filtered estimate is injected into the MPPI rollouts so the planner
    compensates for it (as in DATT's L1+MPC baselines, arXiv:2310.09053).
    """

    def __init__(self, horizon: int = 25, n_samples: int = 512, dt_plan: float = 0.02,
                 temperature: float = 0.02, control_freq: int = 100,
                 noise_sigma: tuple = (0.08, 0.08, 0.02, 0.04), noise_beta: float = 0.7,
                 a_s: float = -5.0, l1_cutoff_hz: float = 4.0, seed: int = 0):
        # Defaults = tuned config A (2026-07-22 sweep): 512 samples, lambda 0.02,
        # AR(1) noise beta=0.7, reduced sigma. slow/normal/fast RMSE
        # 0.042/0.045/0.089 -> 0.023/0.035/0.068 vs the original config.
        self.H, self.N, self.dtp = horizon, n_samples, dt_plan
        self.lam = temperature
        self.beta = noise_beta  # AR(1) correlation of noise along the horizon
        self.dt = 1.0 / control_freq
        self.sigma = np.asarray(noise_sigma)
        self.a_s = a_s
        self.alpha_f = 1.0 - np.exp(-2 * np.pi * l1_cutoff_hz * self.dt)  # LPF coefficient
        self.rng = np.random.default_rng(seed)
        self.p = _load_so_rpy_params()
        self.hover_thrust = float(
            (self.p["mass"] * 9.81 - self.p["acc_coef"]) / self.p["cmd_f_coef"]
        )
        self._traj: Trajectory | None = None

    def reset(self, trajectory: Trajectory) -> None:
        self._traj = trajectory
        self.u_mean = np.tile(np.array([0.0, 0.0, 0.0, self.hover_thrust]), (self.H, 1))
        self.v_hat = np.zeros(3)
        self.sigma_hat = np.zeros(3)
        self.sigma_f = np.zeros(3)

    # --- so_rpy model, vectorized over samples ------------------------------
    def _z_axis(self, rpy: np.ndarray) -> np.ndarray:
        r, p, y = rpy[..., 0], rpy[..., 1], rpy[..., 2]
        cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
        return np.stack([cr * sp * cy + sr * sy, cr * sp * sy - sr * cy, cr * cp], axis=-1)

    def _model_step(self, x: np.ndarray, u: np.ndarray, dist: np.ndarray) -> np.ndarray:
        """x: (N, 12), u: (N, 4). Euler integration with dt_plan."""
        p = self.p
        pos, rpy, vel, drpy = x[:, 0:3], x[:, 3:6], x[:, 6:9], x[:, 9:12]
        thrust = p["acc_coef"] + p["cmd_f_coef"] * u[:, 3]
        acc = thrust[:, None] * self._z_axis(rpy) / p["mass"]
        acc = acc + p["gravity_vec"] + dist  # L1 disturbance estimate injected here
        ddrpy = p["rpy_coef"] * rpy + p["rpy_rates_coef"] * drpy + p["cmd_rpy_coef"] * u[:, 0:3]
        return np.concatenate(
            [pos + self.dtp * vel, rpy + self.dtp * drpy, vel + self.dtp * acc,
             drpy + self.dtp * ddrpy], axis=1,
        )

    # --- L1 adaptation ------------------------------------------------------
    def _l1_update(self, vel: np.ndarray, quat: np.ndarray, thrust_cmd: float) -> None:
        from scipy.spatial.transform import Rotation as R

        p = self.p
        z_b = R.from_quat(quat).as_matrix()[:, 2]
        thrust = p["acc_coef"] + p["cmd_f_coef"] * thrust_cmd
        v_err = self.v_hat - vel
        # predictor with the previously committed command
        v_hat_dot = thrust * z_b / p["mass"] + p["gravity_vec"] + self.sigma_hat + self.a_s * v_err
        self.v_hat = self.v_hat + self.dt * v_hat_dot
        # piecewise-constant adaptation (scalar A_s = a_s * I)
        e = np.exp(self.a_s * self.dt)
        self.sigma_hat = -(self.a_s / (e - 1.0)) * e * (self.v_hat - vel)
        self.sigma_f = (1 - self.alpha_f) * self.sigma_f + self.alpha_f * self.sigma_hat

    # --- MPPI ---------------------------------------------------------------
    def act(self, state: np.ndarray, t: float) -> np.ndarray:
        from scipy.spatial.transform import Rotation as R

        pos, vel, quat, omega = state[:3], state[3:6], state[6:10], state[10:13]
        rpy = R.from_quat(quat).as_euler("xyz")
        # body rates -> rpy rates (E^-1(rpy) @ omega)
        cr, sr = np.cos(rpy[0]), np.sin(rpy[0])
        cp, tp = np.cos(rpy[1]), np.tan(rpy[1])
        E_inv = np.array([[1, sr * tp, cr * tp], [0, cr, -sr], [0, sr / cp, cr / cp]])
        drpy = E_inv @ omega

        x0 = np.concatenate([pos, rpy, vel, drpy])
        ref = self._traj.pos(t + self.dtp * np.arange(1, self.H + 1))
        ref_v = self._traj.vel(t + self.dtp * np.arange(1, self.H + 1))

        eps = self.rng.normal(size=(self.N, self.H, 4)) * self.sigma
        if self.beta > 0:  # temporally correlated (smooth) exploration noise
            for k in range(1, self.H):
                eps[:, k] = self.beta * eps[:, k - 1] + np.sqrt(1 - self.beta**2) * eps[:, k]
        u = np.clip(
            self.u_mean[None] + eps,
            [-1.0, -1.0, -0.5, THRUST_MIN], [1.0, 1.0, 0.5, THRUST_MAX],
        )
        x = np.tile(x0, (self.N, 1))
        cost = np.zeros(self.N)
        for k in range(self.H):
            x = self._model_step(x, u[:, k], self.sigma_f)
            cost += 100.0 * np.sum((x[:, 0:3] - ref[k]) ** 2, axis=1)
            cost += 1.0 * np.sum((x[:, 6:9] - ref_v[k]) ** 2, axis=1)
            cost += 0.1 * np.sum(u[:, k, 0:3] ** 2, axis=1)
        cost += self.lam * np.sum(eps * (self.u_mean[None] / self.sigma**2), axis=(1, 2))

        w = np.exp(-(cost - cost.min()) / self.lam)
        w /= w.sum()
        self.u_mean = np.einsum("n,nhk->hk", w, u)
        action = self.u_mean[0].copy()

        # receding horizon: shift mean for warm start
        self.u_mean = np.roll(self.u_mean, -1, axis=0)
        self.u_mean[-1] = np.array([0.0, 0.0, 0.0, self.hover_thrust])

        self._l1_update(vel, quat, action[3])
        return action
