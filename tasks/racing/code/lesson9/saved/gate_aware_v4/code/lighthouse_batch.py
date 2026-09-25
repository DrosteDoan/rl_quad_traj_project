"""A vectorized Lighthouse sensor for training, parameterized by lambda like `knobs.ScaledLighthouse`
(Lesson 9 phase 6): each of the N worlds draws its own per-episode lambda in [0, lam_max] and the
size errors + update interval scale together with it, exactly as `knobs.py` does at evaluation.

Same external interface as the vendored `crazy_track.sensors.LighthouseSensorBatch` (`__init__(n,
control_freq, seed)`, `.reset()`, `.reset_rows(mask)`, `.measure(t, pos, vel, quat, omega)` -> 4-tuple),
so it is a drop-in replacement for `self.sensor` in a `DATTTrackingEnv` subclass -- but NOT the same
sensor: the vendored batch class draws a noise SCALE in U(0, 1.5) uncoupled from the update rate; this
one draws a single lambda that moves both together, matching what the study actually varies.

One deliberate deviation from the vendored batch class: the GYRO channel is NOT delayed here, matching
`crazy_track.sensors.LighthouseSensor` (the evaluation harness's sensor) and `knobs.ScaledLighthouse` --
the vendored *batch* class (training only) delays it, an existing inconsistency in the vendored code
between training and evaluation that this class does not reproduce, because avoiding exactly this kind
of train/test mismatch is the point of writing it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, str(Path(__file__).resolve().parent))
import knobs as kb  # noqa: E402


class LambdaLighthouseSensorBatch:
    def __init__(self, n: int, control_freq: int = 100, lam_max: float = 0.8, seed: int = 0):
        self.n, self.dt, self.lam_max = n, 1.0 / control_freq, float(lam_max)
        self.p = kb.NOMINAL["lighthouse"]
        self.scale = kb.scales()["lighthouse"]              # the frozen mass_mult... no: lighthouse ceiling
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self) -> None:
        self.reset_rows(np.ones(self.n, dtype=bool))
        self._delayed = None

    def reset_rows(self, mask: np.ndarray) -> None:
        if not hasattr(self, "lam_eff"):
            self.lam_eff = np.zeros(self.n)
            self.bias = np.zeros((self.n, 3))
            self.next_update = np.zeros(self.n)
            self.last_pos = np.full((self.n, 3), np.nan)
        k = int(mask.sum())
        lam = self.rng.uniform(0.0, self.lam_max, size=k)   # the STUDY's own lambda axis, per episode
        self.lam_eff[mask] = lam * self.scale                # -> units of the reference value, as knobs.py does
        self.bias[mask] = self.rng.normal(0.0, self.p["bias_std"], size=(k, 3)) * self.lam_eff[mask, None]
        self.next_update[mask] = 0.0
        self.last_pos[mask] = np.nan

    def measure(self, t: np.ndarray, pos: np.ndarray, vel: np.ndarray, quat: np.ndarray,
                omega: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        p = self.p
        due = (t >= self.next_update) | np.isnan(self.last_pos[:, 0])
        k = int(due.sum())
        if k:
            le = self.lam_eff[due]
            self.last_pos[due] = pos[due] + self.bias[due] + le[:, None] * self.rng.normal(0, p["jitter_std"], (k, 3))
            hz = np.clip(self.rng.normal(p["update_hz"], p["update_hz_std"], k), 8.0, 100.0)
            self.next_update[due] = t[due] + self.dt + le * (1.0 / hz - self.dt)   # dt at lam_eff = 0, 1/hz at 1
        le = self.lam_eff[:, None]
        vel_m = vel + le * self.rng.normal(0, p["vel_std"], vel.shape)
        rot_err = R.from_rotvec(le * self.rng.normal(0, np.radians(p["att_std_deg"]), (self.n, 3)))
        quat_m = (rot_err * R.from_quat(quat)).as_quat()
        omega_m = omega + le * self.rng.normal(0, p["gyro_std"], omega.shape)     # gyro: no latency, see docstring
        meas_delayed = (self.last_pos.copy(), vel_m, quat_m)
        out = self._delayed if self._delayed is not None else meas_delayed
        self._delayed = meas_delayed
        return out[0], out[1], out[2], omega_m
