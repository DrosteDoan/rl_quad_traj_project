"""Freestyle sequences: chained maneuver primitives through race gates.

Gate geometry follows lsy_drone_racing (TUM LSY, github.com/learnsyslab/
lsy_drone_racing, assets/gate.xml + track configs): square gates, 0.72 m
outer frame, 0.4 m opening, pose = opening center + yaw, pass-through
normal = the gate frame's +x axis. Standard opening-center heights:
0.695 m (short) and 1.195 m (tall). Same drone (cf21B_500) and sim
(crazyflow) as this project, so tracks port 1:1.

Planning approach (paper 2): closed-form chaining of validated primitives —
quintic connect segments (the C^2 family the tracking policy trains on)
between gate crossings, plus TRAVELING ballistic flips: the in-place
boost/arc/brake primitive generalized with a constant horizontal drift
velocity. The generalization keeps exact feasibility for free — thrust is
vertical in boost/brake (level attitude, horizontal velocity untouched) and
zero during the arc (any attitude feasible, horizontal velocity conserved).
A MINCO-style optimizer (ZJU analysis, report 2026-07-23) only becomes
necessary beyond this closed-form regime.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from crazy_track.trajectories.base import Trajectory
from crazy_track.trajectories.chained_poly import _quintic

GRAVITY = 9.81
BLEND = 0.2  # rate-trapezoid ramp fraction (matches BallisticFlipTrajectory)
# lateral-acceleration budget for the segment-duration seed guess; the final
# duration is then TIME-SCALED (classical trick) until the segment's analytic
# peak thrust demand fits under SEG_ACC_LIMIT, so this need not be exact
TURN_ACC_BUDGET = 7.0
SEG_ACC_LIMIT = 17.2  # just under feasibility_report's 0.95*TWR*g = 17.52


@dataclass(frozen=True)
class RaceGate:
    """A racing gate, lsy_drone_racing geometry. pos = opening center."""

    pos: tuple[float, float, float]
    yaw: float = 0.0

    OPENING: ClassVar[float] = 0.4
    HALF_OPENING: ClassVar[float] = 0.2
    EDGE_HALF: ClassVar[float] = 0.36  # collision-geometry extent from center
    OUTER: ClassVar[float] = 0.72

    @property
    def center(self) -> np.ndarray:
        return np.asarray(self.pos, dtype=np.float64)

    @property
    def rotation(self) -> np.ndarray:
        """World-from-gate rotation (yaw about z)."""
        c, s = np.cos(self.yaw), np.sin(self.yaw)
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

    @property
    def normal(self) -> np.ndarray:
        """Pass-through direction: the gate frame's +x axis in world."""
        return self.rotation[:, 0]

    def to_gate_frame(self, points: np.ndarray) -> np.ndarray:
        """(..., 3) world points -> gate frame (x = normal, y/z = in-plane)."""
        return (np.asarray(points, dtype=np.float64) - self.center) @ self.rotation


class _QuinticSegment:
    """Per-axis quintic between full boundary states (pos, vel, acc)."""

    def __init__(self, s0, s1, T: float):
        (p0, v0, a0), (p1, v1, a1) = s0, s1
        self.T = float(T)
        self.coeffs = np.stack([
            _quintic(p0[ax], v0[ax], a0[ax], p1[ax], v1[ax], a1[ax], self.T)
            for ax in range(3)
        ])  # (3, 6) ascending powers

    def _eval(self, s: np.ndarray, deriv: int) -> np.ndarray:
        c = self.coeffs
        for _ in range(deriv):
            c = c[..., 1:] * np.arange(1, c.shape[-1])
        out = np.zeros(np.shape(s) + (3,))
        for k in range(c.shape[-1] - 1, -1, -1):
            out = out * np.asarray(s)[..., None] + c[:, k]
        return out

    def pos(self, s):
        return self._eval(s, 0)

    def vel(self, s):
        return self._eval(s, 1)

    def acc(self, s):
        return self._eval(s, 2)

    def att_ref_rotvec(self, s):
        return np.zeros(np.shape(s) + (3,))


class _TravelingFlipSegment:
    """Ballistic flip with constant horizontal drift: boost -> arc+rotation ->
    brake, all translating at `drift` (m/s, z ignored). drift = 0 recovers the
    in-place BallisticFlipTrajectory phase math exactly. Entry/exit state:
    vel = drift (vz = 0), acc = 0 (the boost accel step is inherent to the
    bang-bang primitive, same as in training)."""

    def __init__(self, p0: np.ndarray, drift: np.ndarray, axis: int, direction: int,
                 Tb: float = 0.7, a_boost: float = 7.0):
        self.p0 = np.asarray(p0, dtype=np.float64)
        self.drift = np.array([drift[0], drift[1], 0.0], dtype=np.float64)
        self.axis, self.direction = axis, direction
        self.Tb, self.a_boost = Tb, a_boost
        self.Ta = GRAVITY * Tb / (2.0 * a_boost)
        self.v1 = GRAVITY * Tb / 2.0
        self.T = 2 * self.Ta + Tb
        self.s_rot_start = self.Ta   # rotation window in segment-local time
        self.s_rot_end = self.Ta + Tb

    def _phase_z(self, s: np.ndarray):
        tau = np.asarray(s, dtype=np.float64)
        A, g, Ta, Tb = self.a_boost, GRAVITY, self.Ta, self.Tb
        z1 = 0.5 * A * Ta**2
        s_b, s_c = tau - Ta, tau - Ta - Tb
        conds = [tau < 0, tau < Ta, tau < Ta + Tb, tau < 2 * Ta + Tb]
        z = np.select(conds, [0.0, 0.5 * A * tau**2,
                              z1 + self.v1 * s_b - 0.5 * g * s_b**2,
                              z1 - self.v1 * s_c + 0.5 * A * s_c**2], default=0.0)
        vz = np.select(conds, [0.0, A * tau, self.v1 - g * s_b, -self.v1 + A * s_c],
                       default=0.0)
        az = np.select(conds, [0.0, A, -g, A], default=0.0)
        return z, vz, az

    def pos(self, s):
        z, _, _ = self._phase_z(s)
        out = self.p0 + np.asarray(s)[..., None] * self.drift
        out = out.astype(np.float64)
        out[..., 2] += z
        return out

    def vel(self, s):
        _, vz, _ = self._phase_z(s)
        out = np.broadcast_to(self.drift, np.shape(s) + (3,)).copy()
        out[..., 2] += vz
        return out

    def acc(self, s):
        _, _, az = self._phase_z(s)
        out = np.zeros(np.shape(s) + (3,))
        out[..., 2] = az
        return out

    def att_ref_rotvec(self, s):
        u = np.clip((np.asarray(s) - self.s_rot_start) / self.Tb, 0.0, 1.0)
        r = BLEND
        p = 1.0 / (1.0 - r)
        phi = np.select([u < r, u < 1.0 - r],
                        [p * u**2 / (2 * r), p * (u - r / 2.0)],
                        default=1.0 - p * (1.0 - u)**2 / (2 * r))
        theta = self.direction * 2 * np.pi * phi
        theta_wrapped = np.mod(theta + np.pi, 2 * np.pi) - np.pi
        out = np.zeros(np.shape(theta) + (3,))
        out[..., self.axis] = theta_wrapped
        return out


class FreestyleTrajectory(Trajectory):
    """A freestyle sequence: hover start -> ops chained in order -> ops end.

    Ops (executed in order, timing derived from `cruise` speed):
      ("gate", RaceGate, speed)          fly through the opening center at
                                         `speed` along the gate normal
      ("flip", point, axis, direction[, Tb[, drift[, a_boost]]])
                                         traveling ballistic flip entered at
                                         `point` with horizontal drift velocity
      ("via", point, vel)                free-space waypoint with a velocity
                                         (routing around obstacles/frames)
      ("hover", point, T)                come to rest at `point` over T seconds

    Connect segments are quintics between full boundary states (C^2 inside,
    C^1 at flip boundaries — the boost accel step is the primitive's own).
    """

    def __init__(self, start, ops, cruise: float = 1.5, min_seg_T: float = 1.0,
                 lead_in: float = 1.5):
        self.lead_in = lead_in
        self._segments: list = []
        self._t0s: list[float] = []
        self.gates: list[RaceGate] = []
        self.gate_times: list[float] = []
        self.flips: list[dict] = []
        self.obstacles: list[tuple] = []  # pole tops (lsy convention), render-only

        p = np.asarray(start, dtype=np.float64)
        v = np.zeros(3)
        a = np.zeros(3)
        t = 0.0

        def add(seg, T):
            nonlocal t
            self._segments.append(seg)
            self._t0s.append(t)
            t += T

        def connect(p1, v1, a1, T=None):
            nonlocal p, v, a
            dist = float(np.linalg.norm(p1 - p))
            # duration law: distance at cruise, floor, AND time to turn the
            # velocity vector within a lateral-acceleration budget (sharp
            # direction changes on short legs otherwise demand infeasible thrust)
            dv = float(np.linalg.norm(np.asarray(v1, float) - v))
            if T is None:
                T = max(dist / cruise, min_seg_T, dv / TURN_ACC_BUDGET)
                # time-scale until the quintic's peak thrust demand is feasible
                for _ in range(12):
                    seg = _QuinticSegment((p, v, a), (p1, v1, a1), T)
                    s = np.linspace(0.0, T, 200)
                    demand = np.linalg.norm(
                        seg.acc(s) + np.array([0.0, 0.0, GRAVITY]), axis=1).max()
                    if demand <= SEG_ACC_LIMIT:
                        break
                    T *= 1.15
                add(seg, T)
            else:
                add(_QuinticSegment((p, v, a), (p1, v1, a1), T), T)
            p, v, a = np.asarray(p1, float), np.asarray(v1, float), np.asarray(a1, float)

        # lead-in hold so rollouts starting at rest see a stationary reference
        add(_QuinticSegment((p, v, a), (p, v, a), lead_in), lead_in)

        for op in ops:
            kind = op[0]
            if kind == "gate":
                gate, speed = op[1], (op[2] if len(op) > 2 else cruise)
                self.gates.append(gate)
                connect(gate.center, gate.normal * speed, np.zeros(3))
                self.gate_times.append(t)  # crossing = segment end
            elif kind == "flip":
                point = np.asarray(op[1], dtype=np.float64)
                axis, direction = int(op[2]), int(op[3])
                Tb = float(op[4]) if len(op) > 4 else 0.7
                drift = np.asarray(op[5], float) if len(op) > 5 else np.zeros(3)
                a_boost = float(op[6]) if len(op) > 6 else 7.0
                seg = _TravelingFlipSegment(point, drift, axis, direction, Tb, a_boost)
                connect(point, seg.drift, np.zeros(3))
                self.flips.append({"axis": axis, "direction": direction, "Tb": Tb,
                                   "t_rot_start": t + seg.s_rot_start,
                                   "t_rot_end": t + seg.s_rot_end})
                add(seg, seg.T)
                p = seg.pos(seg.T)
                v, a = seg.drift.copy(), np.zeros(3)
            elif kind == "via":
                connect(np.asarray(op[1], dtype=np.float64),
                        np.asarray(op[2], dtype=np.float64), np.zeros(3))
            elif kind == "hover":
                point = np.asarray(op[1], dtype=np.float64)
                T = float(op[2]) if len(op) > 2 else None
                connect(point, np.zeros(3), np.zeros(3), T=T)
            else:
                raise ValueError(f"unknown op {kind!r}")

        self._t0s_arr = np.asarray(self._t0s)
        self.duration = float(t)

    def _dispatch(self, t, method: str) -> np.ndarray:
        t = self._clamp(t)
        scalar = t.ndim == 0
        tt = np.atleast_1d(t)
        idx = np.clip(np.searchsorted(self._t0s_arr, tt, side="right") - 1,
                      0, len(self._segments) - 1)
        out = np.zeros(tt.shape + (3,))
        for i in np.unique(idx):
            m = idx == i
            out[m] = getattr(self._segments[i], method)(tt[m] - self._t0s_arr[i])
        return out[0] if scalar else out

    def pos(self, t):
        return self._dispatch(t, "pos")

    def vel(self, t):
        return self._dispatch(t, "vel")

    def acc(self, t):
        return self._dispatch(t, "acc")

    def att_ref_rotvec(self, t):
        return self._dispatch(t, "att_ref_rotvec")

    def maneuver_descriptor(self, t) -> np.ndarray:
        """(..., 6) mirroring datt_env._maneuver_descriptor: signed axis vec,
        countdown (clip 1s), progress, active — per the RELEVANT flip window
        (the active one, else the next upcoming, else the most recent)."""
        t = np.asarray(t, dtype=np.float64)
        scalar = t.ndim == 0
        tt = np.atleast_1d(t)
        out = np.zeros(tt.shape + (6,))
        if self.flips:
            starts = np.asarray([f["t_rot_start"] for f in self.flips])
            ends = np.asarray([f["t_rot_end"] for f in self.flips])
            # nearest window: last one whose start is <= t, else the first ahead
            nxt = np.searchsorted(starts, tt, side="right")
            take = np.where(nxt > 0, nxt - 1, 0)
            # if the previous window already ended and another lies ahead, point ahead
            ended = tt > ends[take]
            has_ahead = nxt < len(self.flips)
            take = np.where(ended & has_ahead, np.minimum(nxt, len(self.flips) - 1), take)
            for i, f in enumerate(self.flips):
                m = take == i
                if not m.any():
                    continue
                out[m, f["axis"]] = f["direction"]
                out[m, 3] = np.clip(f["t_rot_start"] - tt[m], 0.0, 1.0)
                out[m, 4] = np.clip((tt[m] - f["t_rot_start"]) / f["Tb"], 0.0, 1.0)
                out[m, 5] = ((tt[m] >= f["t_rot_start"]) & (tt[m] <= f["t_rot_end"])
                             ).astype(np.float64)
        return out[0] if scalar else out


def lsy_level2_freestyle(cruise: float = 1.5) -> FreestyleTrajectory:
    """Canonical freestyle demo: the lsy_drone_racing level-2 nominal gates in
    track order + one traveling roll between gates 2 and 3, ending in a hover.
    Gate poses verbatim from config/level2.toml."""
    g1 = RaceGate((0.5, 0.25, 0.7), yaw=-0.78)
    g2 = RaceGate((1.05, 0.75, 1.2), yaw=2.35)
    g3 = RaceGate((-1.0, -0.25, 0.7), yaw=3.14)
    g4 = RaceGate((0.0, -0.75, 1.2), yaw=0.0)
    traj = FreestyleTrajectory(
        start=(-1.5, 0.75, 1.0),
        ops=[
            ("gate", g1, cruise),
            ("gate", g2, cruise),
            # flip entry placed so the maneuver EXITS east of gate 3's plane
            # (x = -1.0): drift -0.7 m/s over the 1.68 s primitive ends at
            # x ~ -0.48. An entry further west overshot the plane mid-flip and
            # the reference had to double back across it (measured: the
            # zero-shot rollout then flew over the gate instead of through)
            ("flip", (0.7, 0.55, 1.6), 0, -1, 0.7, (-0.7, -0.35, 0.0)),
            ("gate", g3, cruise),
            # wide U-turn: g3 exits heading -x, g4 wants +x; the via keeps the
            # turn clear of g3's frame (a direct quintic clips it - measured)
            ("via", (-1.7, -1.1, 0.95), (1.2, 0.0, 0.0)),
            ("gate", g4, cruise),
            ("hover", (0.8, -0.75, 1.2), 1.5),
        ],
        cruise=cruise,
    )
    traj.obstacles = LSY_LEVEL2_OBSTACLES
    return traj


def pitch_line_freestyle(cruise: float = 1.5) -> FreestyleTrajectory:
    """Out-and-back line with TWO traveling flips: a pitch+ forward flip over
    the direction of travel on the way out, and a roll+ rotation about the
    travel direction (barrel-roll-like) on the way back. Short/tall lsy gate
    heights. Covers the two variants the level-2 demo doesn't touch."""
    g1 = RaceGate((-1.0, 0.3, 0.695), yaw=0.0)
    g2 = RaceGate((2.6, 0.0, 1.195), yaw=0.0)
    g3 = RaceGate((1.5, -1.0, 0.695), yaw=np.pi)
    return FreestyleTrajectory(
        start=(-2.0, 0.0, 1.0),
        ops=[
            ("gate", g1, cruise),
            # forward flip: pitch+ while drifting +x; exit x ~ 0.2 + 1.34 = 1.5,
            # east of nothing (g2's plane at 2.6 is well ahead)
            ("flip", (0.2, 0.3, 1.5), 1, 1, 0.7, (0.8, 0.0, 0.0)),
            ("gate", g2, cruise),
            ("via", (3.4, -0.8, 1.0), (-1.0, -0.5, 0.0)),
            ("gate", g3, cruise),
            # barrel roll: roll+ about the (-x) travel direction; exit x ~ -1.18
            ("flip", (0.0, -1.0, 1.5), 0, 1, 0.7, (-0.7, 0.0, 0.0)),
            ("hover", (-1.5, -1.0, 1.0), 1.5),
        ],
        cruise=cruise,
    )


def tower_climb_freestyle(cruise: float = 1.5) -> FreestyleTrajectory:
    """Climb through a short then a tall gate, pitch- flip at the top, return
    through both gates in REVERSE traversal (modeled as gates with yaw + pi,
    the lsy signed-gate-order convention)."""
    g1 = RaceGate((0.0, -0.5, 0.695), yaw=np.pi / 2)
    g2 = RaceGate((0.0, 0.9, 1.195), yaw=np.pi / 2)
    g2_rev = RaceGate((0.0, 0.9, 1.195), yaw=-np.pi / 2)
    g1_rev = RaceGate((0.0, -0.5, 0.695), yaw=-np.pi / 2)
    return FreestyleTrajectory(
        start=(0.0, -1.5, 0.8),
        ops=[
            ("gate", g1, cruise),
            ("gate", g2, cruise),
            # pitch- flip drifting +y at the top of the climb; exit y ~ 2.9
            ("flip", (0.0, 1.9, 1.7), 1, -1, 0.7, (0.0, 0.6, 0.0)),
            ("gate", g2_rev, cruise),
            ("gate", g1_rev, cruise),
            ("hover", (0.0, -1.5, 0.8), 1.5),
        ],
        cruise=cruise,
    )


# level2.toml obstacle poles; pos = TOP of the pole (lsy convention)
LSY_LEVEL2_OBSTACLES = [(0.0, 0.75, 1.55), (1.0, 0.25, 1.55),
                        (-1.5, -0.25, 1.55), (-0.5, -0.75, 1.55)]


def lsy_level2_race(cruise: float = 3.0) -> FreestyleTrajectory:
    """Speed-run variant of the level-2 track: gates only, no flip, tight
    segment times (min_seg_T scales inversely with cruise). Race time =
    last-gate crossing minus the stationary lead-in; the lsy leaderboard
    clock instead starts at episode start with the drone on the ground."""
    g1 = RaceGate((0.5, 0.25, 0.7), yaw=-0.78)
    g2 = RaceGate((1.05, 0.75, 1.2), yaw=2.35)
    g3 = RaceGate((-1.0, -0.25, 0.7), yaw=3.14)
    g4 = RaceGate((0.0, -0.75, 1.2), yaw=0.0)
    traj = FreestyleTrajectory(
        start=(-1.5, 0.75, 1.0),
        ops=[
            # g1 -> g2 is a hairpin (exit SE, re-enter NW): slow through both
            # gates and swing wide east through a via, like a racing line
            ("gate", g1, 0.7 * cruise),
            ("via", (1.75, 0.5, 0.95), (0.0, 0.7 * cruise, 0.0)),
            ("gate", g2, 0.7 * cruise),
            ("gate", g3, cruise),
            ("via", (-1.7, -1.1, 0.95), (0.8 * cruise, 0.0, 0.0)),
            ("gate", g4, cruise),
            ("hover", (1.0, -0.75, 1.2), 1.0),
        ],
        cruise=cruise,
        min_seg_T=min(1.0, 1.2 / cruise),
    )
    traj.obstacles = LSY_LEVEL2_OBSTACLES
    return traj


TRACKS = {
    "lsy-level2": lsy_level2_freestyle,
    "pitch-line": pitch_line_freestyle,
    "tower-climb": tower_climb_freestyle,
    "lsy-level2-race": lsy_level2_race,
}


def feasibility_report(traj: FreestyleTrajectory, twr: float = 1.88,
                       clearance_margin: float = 0.07, dt: float = 0.002) -> dict:
    """Analytic feasibility of a freestyle reference. All checks on the REF."""
    t = np.arange(0.0, traj.duration, dt)
    acc = traj.acc(t)
    thrust_acc = acc + np.array([0.0, 0.0, GRAVITY])
    demand = np.linalg.norm(thrust_acc, axis=1)
    in_arc = np.zeros(len(t), dtype=bool)
    for f in traj.flips:
        in_arc |= (t >= f["t_rot_start"]) & (t <= f["t_rot_end"])
    # trapezoid peak rate is analytic: 2*pi / ((1 - BLEND) * Tb)
    peak_rate = max((2 * np.pi / ((1 - BLEND) * f["Tb"]) for f in traj.flips),
                    default=0.0)
    # Gate-plane crossings: every crossing of a gate's plane must go either
    # through the opening (with margin) or fully clear of the frame; the
    # intended crossing (nearest the recorded gate time) must be the former.
    pos = traj.pos(t)
    open_bound = RaceGate.HALF_OPENING - clearance_margin
    frame_clear = RaceGate.EDGE_HALF + clearance_margin
    gate_intended = []   # in-plane offset of the intended crossing, per gate
    crossings_ok = True
    for g, tg in zip(traj.gates, traj.gate_times):
        local = g.to_gate_frame(pos)
        x = local[:, 0]
        cross = np.flatnonzero(np.sign(x[1:]) != np.sign(x[:-1]))
        offsets, times = [], []
        for i in cross:
            w = x[i] / (x[i] - x[i + 1])  # linear interp to the plane
            yz = local[i, 1:] + w * (local[i + 1, 1:] - local[i, 1:])
            offsets.append(float(np.abs(yz).max()))
            times.append(t[i] + w * dt)
        if not offsets:
            gate_intended.append(np.inf)
            crossings_ok = False
            continue
        offsets, times = np.asarray(offsets), np.asarray(times)
        intended = int(np.argmin(np.abs(times - tg)))
        gate_intended.append(float(offsets[intended]))
        crossings_ok &= bool(
            offsets[intended] <= open_bound
            and np.all((offsets <= open_bound) | (offsets >= frame_clear))
        )
    return {
        "max_thrust_acc_outside_arc": float(demand[~in_arc].max()),
        "thrust_acc_limit": 0.95 * twr * GRAVITY,
        "min_thrust_acc_outside_arc": float(demand[~in_arc].min()),
        "min_z": float(pos[:, 2].min()),
        "peak_ref_rate": peak_rate,
        "gate_max_inplane_offset": gate_intended,  # at the intended crossing
        "gate_clearance_bound": open_bound,
        "gate_crossings_ok": crossings_ok,
        "feasible": bool(
            demand[~in_arc].max() <= 0.95 * twr * GRAVITY
            and crossings_ok
            and pos[:, 2].min() > 0.15
        ),
    }
