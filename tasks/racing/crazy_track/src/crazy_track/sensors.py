"""Lighthouse (SteamVR 2.0 base stations) measurement model for realistic evals.

Grounded in the Bitcraze dataset paper (Taffanel et al., arXiv:2104.11523),
Table III and Fig. 6, LH2 + onboard EKF:
- position sample frequency 34 +/- 18 Hz (crossing-beam; large std due to
  periodic base-station interference),
- jitter/precision ~0.3 mm (crossing beam) to 0.7 mm (EKF) RMS,
- absolute accuracy vs motion capture 2-4 cm mean (quasi-static spatial bias
  from base-station geometry calibration), flight outliers ~5 cm,
- velocity/attitude come from the onboard EKF fusing IMU at 1 kHz between
  position updates; we model them as lightly-noised true values,
- plus one control step (10 ms) of transport/processing latency.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R


class LighthouseSensor:
    def __init__(self, control_freq: int = 100, update_hz: float = 34.0,
                 update_hz_std: float = 18.0, jitter_std: float = 0.0007,
                 bias_std: float = 0.015, vel_std: float = 0.03,
                 att_std_deg: float = 0.5, gyro_std: float = 0.02,
                 latency_steps: int = 1, seed: int = 0):
        self.dt = 1.0 / control_freq
        self.update_hz, self.update_hz_std = update_hz, update_hz_std
        self.jitter_std, self.bias_std = jitter_std, bias_std
        self.vel_std, self.att_std, self.gyro_std = vel_std, np.radians(att_std_deg), gyro_std
        self.latency = latency_steps
        self._seed = seed
        self.reset()

    def reset(self) -> None:
        self.rng = np.random.default_rng(self._seed)
        # Quasi-static spatial bias -> 3D mean error ~ sqrt(3)*bias_std ~ 2.6 cm
        self.bias = self.rng.normal(0.0, self.bias_std, size=3)
        self.next_update_t = 0.0
        self.last_pos_meas: np.ndarray | None = None
        self.buffer: list[np.ndarray] = []

    def _sample_interval(self) -> float:
        hz = np.clip(self.rng.normal(self.update_hz, self.update_hz_std), 8.0, 100.0)
        return 1.0 / hz

    def measure(self, t: float, true_state: np.ndarray) -> np.ndarray:
        """true_state: [pos(3), vel(3), quat(4), omega(3)] -> noisy/delayed copy."""
        pos, vel, quat, omega = (true_state[:3], true_state[3:6], true_state[6:10],
                                 true_state[10:13])
        # Position: zero-order hold between Lighthouse updates
        if self.last_pos_meas is None or t >= self.next_update_t:
            self.last_pos_meas = pos + self.bias + self.rng.normal(0, self.jitter_std, 3)
            self.next_update_t = t + self._sample_interval()
        vel_m = vel + self.rng.normal(0, self.vel_std, 3)
        rot_err = R.from_rotvec(self.rng.normal(0, self.att_std, 3))
        quat_m = (rot_err * R.from_quat(quat)).as_quat()
        omega_m = omega + self.rng.normal(0, self.gyro_std, 3)
        meas = np.concatenate([self.last_pos_meas, vel_m, quat_m, omega_m])

        self.buffer.append(meas)
        if len(self.buffer) > self.latency + 1:
            self.buffer.pop(0)
        out = self.buffer[0].copy()
        # The gyro is an onboard IMU (~kHz, no transport delay): only the optical
        # position chain carries the latency. Delayed rate feedback destabilizes
        # 500 Hz inner rate loops (measured: xadapt RMSE 0.78 -> 0.05 on fix).
        out[10:13] = omega_m
        return out


class LighthouseSensorBatch:
    """Vectorized Lighthouse model for the training env (N worlds).

    Same parameters as LighthouseSensor; latency is one env step (20 ms at
    50 Hz training freq — slightly harsher than the 10 ms eval model).
    """

    def __init__(self, n: int, control_freq: int = 50, update_hz: float = 34.0,
                 update_hz_std: float = 18.0, jitter_std: float = 0.0007,
                 bias_std: float = 0.015, vel_std: float = 0.03,
                 att_std_deg: float = 0.5, gyro_std: float = 0.02, seed: int = 0,
                 noise_dr: bool = False):
        self.n = n
        self.dt = 1.0 / control_freq
        self.update_hz, self.update_hz_std = update_hz, update_hz_std
        self.jitter_std, self.bias_std = jitter_std, bias_std
        self.vel_std, self.att_std, self.gyro_std = vel_std, np.radians(att_std_deg), gyro_std
        self.noise_dr = noise_dr  # per-episode noise scale in [0, 1.5] (v5 DR)
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self) -> None:
        self.reset_rows(np.ones(self.n, dtype=bool))
        self._delayed = None

    def reset_rows(self, mask: np.ndarray) -> None:
        if not hasattr(self, "bias"):
            self.bias = np.zeros((self.n, 3))
            self.next_update = np.zeros(self.n)
            self.last_pos = np.full((self.n, 3), np.nan)
            self.scale = np.ones(self.n)
        k = int(mask.sum())
        if self.noise_dr:
            self.scale[mask] = self.rng.uniform(0.0, 1.5, size=k)
        self.bias[mask] = (self.rng.normal(0.0, self.bias_std, size=(k, 3))
                           * self.scale[mask, None])
        self.next_update[mask] = 0.0
        self.last_pos[mask] = np.nan

    def measure(self, t: np.ndarray, pos: np.ndarray, vel: np.ndarray, quat: np.ndarray,
                omega: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """t (N,), pos/vel/omega (N,3), quat (N,4). Returns noisy delayed copies."""
        due = (t >= self.next_update) | np.isnan(self.last_pos[:, 0])
        k = int(due.sum())
        if k:
            self.last_pos[due] = (pos[due] + self.bias[due]
                                  + self.rng.normal(0, self.jitter_std, (k, 3)))
            hz = np.clip(self.rng.normal(self.update_hz, self.update_hz_std, k), 8.0, 100.0)
            self.next_update[due] = t[due] + 1.0 / hz
        s = self.scale[:, None]
        vel_m = vel + self.rng.normal(0, self.vel_std, vel.shape) * s
        rot_err = R.from_rotvec(self.rng.normal(0, self.att_std, (self.n, 3)) * s)
        quat_m = (rot_err * R.from_quat(quat)).as_quat()
        omega_m = omega + self.rng.normal(0, self.gyro_std, omega.shape) * s
        meas = (self.last_pos.copy(), vel_m, quat_m, omega_m)
        out = self._delayed if self._delayed is not None else meas  # 1-step latency
        self._delayed = meas
        return out


SENSORS = {"none": None, "lighthouse": LighthouseSensor}
