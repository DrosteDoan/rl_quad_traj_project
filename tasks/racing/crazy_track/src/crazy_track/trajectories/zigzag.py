from __future__ import annotations

import numpy as np

from crazy_track.trajectories.base import Trajectory


class ZigzagTrajectory(Trajectory):
    """Randomized infeasible reference: piecewise-linear position between waypoints.

    Follows the DATT "infeasible trajectory" family: velocity is discontinuous
    at every waypoint (instantaneous direction changes), so no quadrotor can
    track it exactly — the benchmark measures how gracefully controllers
    degrade.
    """

    def __init__(self, knot_times: np.ndarray, waypoints: np.ndarray):
        self.knot_times = np.asarray(knot_times, dtype=np.float64)
        self.waypoints = np.asarray(waypoints, dtype=np.float64)  # (n_knots, 3)
        self.duration = float(self.knot_times[-1])
        self._seg_vel = np.diff(self.waypoints, axis=0) / np.diff(self.knot_times)[:, None]

    @classmethod
    def random(
        cls,
        rng: np.random.Generator,
        duration: float = 10.0,
        seg_duration_range: tuple[float, float] = (0.5, 1.5),
        pos_range: float = 1.0,
        start_pos: np.ndarray | None = None,
    ) -> "ZigzagTrajectory":
        """Sample random waypoints in [-pos_range, pos_range]^3 with random segment durations."""
        start = np.zeros(3) if start_pos is None else np.asarray(start_pos, dtype=np.float64)
        times = [0.0]
        while times[-1] < duration:
            times.append(times[-1] + rng.uniform(*seg_duration_range))
        knot_times = np.array(times)
        waypoints = start + rng.uniform(-pos_range, pos_range, size=(len(knot_times), 3))
        waypoints[0] = start
        return cls(knot_times, waypoints)

    def _segments(self, t: np.ndarray | float) -> tuple[np.ndarray, np.ndarray, bool]:
        t = self._clamp(t)
        scalar = t.ndim == 0
        t = np.atleast_1d(t)
        seg = np.clip(np.searchsorted(self.knot_times, t, side="right") - 1, 0, len(self._seg_vel) - 1)
        return t, seg, scalar

    def pos(self, t):
        t, seg, scalar = self._segments(t)
        tau = (t - self.knot_times[seg])[:, None]
        out = self.waypoints[seg] + self._seg_vel[seg] * tau
        return out[0] if scalar else out

    def vel(self, t):
        t, seg, scalar = self._segments(t)
        out = self._seg_vel[seg]
        return out[0] if scalar else out.copy()

    def acc(self, t):
        t, seg, scalar = self._segments(t)
        out = np.zeros((len(t), 3))
        return out[0] if scalar else out
