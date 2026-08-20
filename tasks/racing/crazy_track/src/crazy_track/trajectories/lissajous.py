from __future__ import annotations

import numpy as np

from crazy_track.trajectories.base import Trajectory


class LissajousTrajectory(Trajectory):
    """Figure-eight Lissajous from 'Learning to Fly in Seconds' (arXiv:2311.13081, Fig. 5).

    p(t) = [A*cos(2*pi*t/T), B*sin(4*pi*t/T), z] with A=1, B=0.5 as in the paper.
    Cycle times benchmarked: slow T=15s, normal T=5.5s, fast T=3.5s (up to 3 m/s, 0.9g).
    """

    SPEEDS = {"slow": 15.0, "normal": 5.5, "fast": 3.5, "acro": 2.2}

    def __init__(self, T: float = 5.5, A: float = 1.0, B: float = 0.5, z: float = 1.0,
                 n_cycles: int = 2, vertical: bool = False):
        """vertical=True puts the figure-8 in the x-z plane (thrust-modulation
        heavy — the acro benchmark), otherwise x-y as in the paper."""
        self.T, self.A, self.B, self.z = T, A, B, z
        self.vertical = vertical
        self.duration = n_cycles * T

    @classmethod
    def from_speed(cls, speed: str, **kwargs) -> "LissajousTrajectory":
        return cls(T=cls.SPEEDS[speed], **kwargs)

    def _angles(self, t):
        t = self._clamp(t)
        w = 2 * np.pi / self.T
        return t, w

    def _assemble(self, u: np.ndarray, v: np.ndarray, t: np.ndarray,
                  offset: float = 0.0) -> np.ndarray:
        zc = np.full_like(t, offset)
        if self.vertical:
            return np.stack([u, np.zeros_like(t), zc + v], axis=-1)
        return np.stack([u, v, zc], axis=-1)

    def pos(self, t):
        t, w = self._angles(t)
        return self._assemble(self.A * np.cos(w * t), self.B * np.sin(2 * w * t), t, self.z)

    def vel(self, t):
        t, w = self._angles(t)
        return self._assemble(-self.A * w * np.sin(w * t), 2 * w * self.B * np.cos(2 * w * t), t)

    def acc(self, t):
        t, w = self._angles(t)
        return self._assemble(-self.A * w**2 * np.cos(w * t),
                              -4 * w**2 * self.B * np.sin(2 * w * t), t)
