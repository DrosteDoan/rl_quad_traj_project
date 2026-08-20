from __future__ import annotations

import numpy as np

from crazy_track.trajectories.base import Trajectory


class ChainedPolyTrajectory(Trajectory):
    """Randomized smooth reference: chained quintic segments, C^2-continuous.

    Follows the DATT "smooth trajectory" family: random knot states (position,
    velocity, acceleration) are sampled at fixed time intervals and joined by
    quintic polynomials per axis, so position, velocity and acceleration are
    continuous everywhere and the reference is dynamically feasible for a
    quadrotor (bounded acceleration).
    """

    def __init__(self, knot_times: np.ndarray, coeffs: np.ndarray):
        """coeffs: (n_segments, 3, 6) ascending-power quintic coefficients per axis."""
        self.knot_times = np.asarray(knot_times, dtype=np.float64)
        self.coeffs = np.asarray(coeffs, dtype=np.float64)
        self.duration = float(self.knot_times[-1])

    @classmethod
    def random(
        cls,
        rng: np.random.Generator,
        duration: float = 10.0,
        seg_duration: float = 2.0,
        pos_range: float = 1.0,
        vel_range: float = 1.0,
        acc_range: float = 2.0,
        start_pos: np.ndarray | None = None,
    ) -> "ChainedPolyTrajectory":
        """Sample a random smooth trajectory.

        Knot positions are sampled in [-pos_range, pos_range]^3 (relative to
        start_pos), knot velocities/accelerations within their ranges. The
        trajectory starts at rest at start_pos (default origin).
        """
        n_seg = int(np.ceil(duration / seg_duration))
        knot_times = seg_duration * np.arange(n_seg + 1)
        start = np.zeros(3) if start_pos is None else np.asarray(start_pos, dtype=np.float64)

        # Knot states: (n_seg + 1, 3) each. First knot is at rest at start.
        pos = start + rng.uniform(-pos_range, pos_range, size=(n_seg + 1, 3))
        vel = rng.uniform(-vel_range, vel_range, size=(n_seg + 1, 3))
        acc = rng.uniform(-acc_range, acc_range, size=(n_seg + 1, 3))
        pos[0], vel[0], acc[0] = start, 0.0, 0.0

        coeffs = np.empty((n_seg, 3, 6))
        for i in range(n_seg):
            T = knot_times[i + 1] - knot_times[i]
            for ax in range(3):
                coeffs[i, ax] = _quintic(
                    pos[i, ax], vel[i, ax], acc[i, ax],
                    pos[i + 1, ax], vel[i + 1, ax], acc[i + 1, ax], T,
                )
        return cls(knot_times, coeffs)

    def _eval(self, t: np.ndarray | float, deriv: int) -> np.ndarray:
        t = self._clamp(t)
        scalar = t.ndim == 0
        t = np.atleast_1d(t)
        seg = np.clip(np.searchsorted(self.knot_times, t, side="right") - 1, 0, len(self.coeffs) - 1)
        tau = t - self.knot_times[seg]  # local time within segment

        c = self.coeffs[seg]  # (N, 3, 6)
        for _ in range(deriv):
            c = c[..., 1:] * np.arange(1, c.shape[-1])

        # Horner evaluation over ascending powers.
        out = np.zeros(c.shape[:-1])
        for k in range(c.shape[-1] - 1, -1, -1):
            out = out * tau[:, None] + c[..., k]
        return out[0] if scalar else out

    def pos(self, t):
        return self._eval(t, 0)

    def vel(self, t):
        return self._eval(t, 1)

    def acc(self, t):
        return self._eval(t, 2)


def _quintic(p0: float, v0: float, a0: float, p1: float, v1: float, a1: float, T: float) -> np.ndarray:
    """Ascending-power quintic matching (pos, vel, acc) at tau=0 and tau=T."""
    A = np.zeros((6, 6))
    b = np.array([p0, v0, a0, p1, v1, a1], dtype=np.float64)
    powers = np.arange(6)
    A[0, 0], A[1, 1], A[2, 2] = 1.0, 1.0, 2.0
    A[3] = T**powers
    A[4, 1:] = powers[1:] * T ** (powers[1:] - 1)
    A[5, 2:] = powers[2:] * (powers[2:] - 1) * T ** (powers[2:] - 2)
    return np.linalg.solve(A, b)
