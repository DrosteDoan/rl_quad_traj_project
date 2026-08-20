from __future__ import annotations

import numpy as np

from crazy_track.trajectories.base import Trajectory

GRAVITY = 9.81


class FlipTrajectory(Trajectory):
    """Hover -> full 360-degree flip about a body axis -> hover recovery.

    Position reference stays at the hover point (the physically ideal flip
    returns there); the attitude reference sweeps 0..2*pi about `axis` over
    [t0, t0+Tf] with a smooth cosine ramp. Altitude margin matters: thrust
    averages ~zero vertical during the rotation, so the drone drops
    ~0.5*g*(Tf/2)^2-ish — hover point defaults to z=1.5 m.
    """

    def __init__(self, hover=(0.0, 0.0, 1.5), t0: float = 2.0, Tf: float = 0.5,
                 axis: int = 0, direction: int = 1, duration: float = 6.0):
        self.hover = np.asarray(hover, dtype=np.float64)
        self.t0, self.Tf = t0, Tf
        self.axis, self.direction = axis, direction  # 0 = roll, 1 = pitch
        self.duration = duration

    def pos(self, t):
        t = self._clamp(t)
        return np.broadcast_to(self.hover, t.shape + (3,)).copy()

    def vel(self, t):
        t = self._clamp(t)
        return np.zeros(t.shape + (3,))

    def acc(self, t):
        t = self._clamp(t)
        return np.zeros(t.shape + (3,))

    def att_ref_rotvec(self, t):
        t = self._clamp(t)
        u = np.clip((t - self.t0) / self.Tf, 0.0, 1.0)
        theta = self.direction * 2 * np.pi * 0.5 * (1 - np.cos(np.pi * u))  # smooth 0..2pi
        # Represent as rotvec with angle wrapped to (-pi, pi] so the error to the
        # current attitude stays continuous through the flip.
        theta_wrapped = np.mod(theta + np.pi, 2 * np.pi) - np.pi
        out = np.zeros(np.shape(theta) + (3,))
        out[..., self.axis] = theta_wrapped
        return out


class BallisticFlipTrajectory(Trajectory):
    """Dynamically feasible 360-degree flip: boost -> ballistic arc -> brake.

    Plan-then-track (Lupashin flip machines; DDA RSS20; the attitude-segment
    idea of ZJU's Aerobatic-Planner, reduced to its minimal closed-form
    instance): position and attitude references are CONSISTENT at every
    instant, unlike FlipTrajectory's hover-pinned (infeasible) position ref
    that made flip learning a per-seed lottery (see report 2026-07-23).

    Phases, with symmetric net boost/brake acceleration A (level attitude):
      boost     [t0, +Ta]:      acc = +A z,  Ta = g*Tb/(2A)
      ballistic [+Ta, +Ta+Tb]:  acc = -g (zero thrust -> ANY attitude is
                                feasible); attitude sweeps 0..2pi about
                                `axis` with a trapezoidal rate profile
                                (blend fraction 0.2 -> peak rate
                                2*pi/(0.8*Tb), keep <= ~0.75*RATE_MAX)
      brake     [+Ta+Tb, +2Ta+Tb]: acc = +A z, ends at the hover point
                                   with v = 0 (closed-form symmetric).
    Choosing Ta = g*Tb/(2A) makes launch/landing z equal to the hover z and
    the arc stay ABOVE it (apex +g*Tb^2/8 + g^2*Tb^2/(8A)); the drone never
    goes below the start altitude. Feasibility: brake thrust accel A + g
    must stay under TWR*g (cf21B: A <= 8.6; default 7.0). The THRUST_MIN
    floor (~1.97 m/s^2 along body-z) integrates to ~zero net velocity over
    a full rotation (rotating unit vector), displacement error ~2 cm.
    """

    BLEND = 0.2  # rate-trapezoid ramp fraction on each side

    def __init__(self, hover=(0.0, 0.0, 2.0), t0: float = 2.0, Tb: float = 0.7,
                 axis: int = 0, direction: int = 1, a_boost: float = 7.0,
                 duration: float = 6.0):
        self.hover = np.asarray(hover, dtype=np.float64)
        self.t0, self.Tb = t0, Tb
        self.axis, self.direction = axis, direction  # 0 = roll, 1 = pitch
        self.a_boost = a_boost
        self.duration = duration
        self.Ta = GRAVITY * Tb / (2.0 * a_boost)  # boost = brake duration
        self.v1 = GRAVITY * Tb / 2.0              # launch speed (independent of A)
        self.t_rot_start = t0 + self.Ta           # rotation window = ballistic phase
        self.t_rot_end = t0 + self.Ta + Tb
        self.t_end = t0 + 2 * self.Ta + Tb

    def _phase_z(self, t: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(z offset, vz, az) relative to hover, vectorized."""
        tau = np.asarray(t, dtype=np.float64) - self.t0
        A, g, Ta, Tb = self.a_boost, GRAVITY, self.Ta, self.Tb
        z1 = 0.5 * A * Ta**2  # ballistic launch/landing height above hover
        s_b = tau - Ta        # time into ballistic phase
        s_c = tau - Ta - Tb   # time into brake phase
        conds = [tau < 0,
                 tau < Ta,
                 tau < Ta + Tb,
                 tau < 2 * Ta + Tb]
        z = np.select(conds,
                      [0.0,
                       0.5 * A * tau**2,
                       z1 + self.v1 * s_b - 0.5 * g * s_b**2,
                       z1 - self.v1 * s_c + 0.5 * A * s_c**2],
                      default=0.0)
        vz = np.select(conds,
                       [0.0, A * tau, self.v1 - g * s_b, -self.v1 + A * s_c],
                       default=0.0)
        az = np.select(conds, [0.0, A, -g, A], default=0.0)
        return z, vz, az

    def pos(self, t):
        t = self._clamp(t)
        z, _, _ = self._phase_z(t)
        out = np.broadcast_to(self.hover, np.shape(t) + (3,)).copy()
        out[..., 2] += z
        return out

    def vel(self, t):
        t = self._clamp(t)
        _, vz, _ = self._phase_z(t)
        out = np.zeros(np.shape(t) + (3,))
        out[..., 2] = vz
        return out

    def acc(self, t):
        t = self._clamp(t)
        _, _, az = self._phase_z(t)
        out = np.zeros(np.shape(t) + (3,))
        out[..., 2] = az
        return out

    def att_ref_rotvec(self, t):
        t = self._clamp(t)
        u = np.clip((t - self.t_rot_start) / self.Tb, 0.0, 1.0)
        r = self.BLEND
        p = 1.0 / (1.0 - r)  # normalized trapezoid peak rate
        phi = np.select(
            [u < r, u < 1.0 - r],
            [p * u**2 / (2 * r), p * (u - r / 2.0)],
            default=1.0 - p * (1.0 - u)**2 / (2 * r),
        )
        theta = self.direction * 2 * np.pi * phi
        theta_wrapped = np.mod(theta + np.pi, 2 * np.pi) - np.pi
        out = np.zeros(np.shape(theta) + (3,))
        out[..., self.axis] = theta_wrapped
        return out
