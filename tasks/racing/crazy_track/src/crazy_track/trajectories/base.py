from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Trajectory(ABC):
    """A 3D reference trajectory r(t) defined on [0, duration].

    All query methods accept a scalar or array of times and return arrays of
    shape (..., 3). Times outside [0, duration] are clamped.
    """

    duration: float

    @abstractmethod
    def pos(self, t: np.ndarray | float) -> np.ndarray: ...

    @abstractmethod
    def vel(self, t: np.ndarray | float) -> np.ndarray: ...

    @abstractmethod
    def acc(self, t: np.ndarray | float) -> np.ndarray: ...

    def ref_window(self, t: float, horizon: int, dt: float) -> np.ndarray:
        """Future position references [r(t), r(t+dt), ..., r(t+(horizon-1)dt)].

        This is the feedforward encoding DATT feeds its policy. Shape (horizon, 3).
        """
        times = t + dt * np.arange(horizon)
        return self.pos(times)

    def att_ref_rotvec(self, t: np.ndarray | float) -> np.ndarray:
        """Reference attitude as rotation vector (world<-body), shape (..., 3).

        Default: identity (level flight). Overridden by acrobatic maneuver
        trajectories (e.g. flips).
        """
        t = np.asarray(t, dtype=np.float64)
        return np.zeros(t.shape + (3,))

    def _clamp(self, t: np.ndarray | float) -> np.ndarray:
        return np.clip(np.asarray(t, dtype=np.float64), 0.0, self.duration)
