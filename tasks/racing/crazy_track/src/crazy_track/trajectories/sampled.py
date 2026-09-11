"""Sampled (tabulated) race trajectory: load an externally planned trajectory —
here TOGT-Planner's CSV — and expose it through the Trajectory interface so the
learned trackers and the race metrics (gate crossings, race_time) run on it
unchanged.

The CSV columns follow TOGT's MincoSnapTrajectory::save():
    t,p_x,p_y,p_z,q_w,q_x,q_y,q_z,v_x,v_y,v_z,w_x,w_y,w_z,a_lin_x,a_lin_y,a_lin_z,...
Only t, p, v, a_lin are used. A stationary lead-in is prepended (rollouts start
at rest, and race_time subtracts it, exactly as FreestyleTrajectory does) and a
short hold is appended so the episode outlives the final gate.
"""

from __future__ import annotations

import numpy as np

from crazy_track.trajectories.base import Trajectory
from crazy_track.trajectories.freestyle import RaceGate


class SampledRaceTrajectory(Trajectory):
    def __init__(self, csv_path: str, gates: list[RaceGate], lead_in: float = 1.5,
                 tail: float = 1.0, obstacles: list | None = None, time_scale: float = 1.0):
        d = np.genfromtxt(csv_path, delimiter=",", names=True)
        # time_scale > 1 stretches the SAME spatial path in time: v /= s, a /= s^2. A feasible
        # plan stays feasible; it is the cleanest "how much slower must it be to be trackable" knob.
        s_ = float(time_scale)
        self.time_scale = s_
        self._t = np.asarray(d["t"], dtype=np.float64) * s_
        self._p = np.stack([d["p_x"], d["p_y"], d["p_z"]], axis=1).astype(np.float64)
        self._v = np.stack([d["v_x"], d["v_y"], d["v_z"]], axis=1).astype(np.float64) / s_
        self._a = np.stack([d["a_lin_x"], d["a_lin_y"], d["a_lin_z"]], axis=1).astype(np.float64) / s_**2
        self.lead_in = float(lead_in)
        self.plan_duration = float(self._t[-1] - self._t[0])   # planner's own lap incl. stop
        self.t_end = self.lead_in + self.plan_duration
        self.duration = self.t_end + float(tail)
        self.gates = list(gates)
        self.flips: list[dict] = []          # no maneuvers: descriptor stays zero
        self.obstacles = list(obstacles or [])
        self.csv_path = csv_path
        self.gate_times = self._crossing_times()

    # ---- interpolation -------------------------------------------------------
    def _interp(self, t, table: np.ndarray, outside_zero: bool) -> np.ndarray:
        t = self._clamp(t)
        scalar = t.ndim == 0
        tt = np.atleast_1d(t)
        s = np.clip(tt - self.lead_in, 0.0, self.plan_duration) + self._t[0]
        out = np.stack([np.interp(s, self._t, table[:, k]) for k in range(3)], axis=1)
        if outside_zero:
            out[(tt < self.lead_in) | (tt > self.t_end)] = 0.0
        return out[0] if scalar else out

    def pos(self, t):
        return self._interp(t, self._p, outside_zero=False)

    def vel(self, t):
        return self._interp(t, self._v, outside_zero=True)

    def acc(self, t):
        return self._interp(t, self._a, outside_zero=True)

    def maneuver_descriptor(self, t) -> np.ndarray:
        t = np.asarray(t, dtype=np.float64)
        return np.zeros((6,) if t.ndim == 0 else t.shape + (6,))

    # ---- gate crossing times of the REFERENCE (for the race metrics) ----------
    def _crossing_times(self) -> list[float]:
        dt = 0.002
        ts = np.arange(self.lead_in, self.t_end, dt)
        P = self.pos(ts)
        times, t_prev = [], self.lead_in
        for g in self.gates:
            loc = g.to_gate_frame(P)
            x = loc[:, 0]
            idx = np.flatnonzero((np.sign(x[1:]) != np.sign(x[:-1])) & (ts[1:] > t_prev))
            hit = None
            for i in idx:
                w = x[i] / (x[i] - x[i + 1])
                yz = loc[i, 1:] + w * (loc[i + 1, 1:] - loc[i, 1:])
                if np.abs(yz).max() < RaceGate.HALF_OPENING:
                    hit = float(ts[i] + w * dt)
                    break
            if hit is None:
                raise ValueError(f"reference never crosses gate {g} inside its opening")
            times.append(hit)
            t_prev = hit
        return times
