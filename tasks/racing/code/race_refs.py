"""Shared reference helpers for the racing task's Lesson-6 tools (main AND race venv).

Two things every Lesson-6 tool needs that the vendored crazy_track does not provide:

  GroundStartTrajectory   a race reference on the LEADERBOARD clock. crazy_track's race
      references -- the closed-form `lsy_level2_race()` and TOGT plans loaded through
      `SampledRaceTrajectory` -- start AT REST IN THE AIR at (-1.5, 0.75, 1.0) after a
      1.5 s stationary lead-in, and the parent project's `race_time` runs from motion
      onset to the last gate. The lsy leaderboard starts the drone ON THE GROUND at
      (-1.5, 0.75, 0.01) and its clock runs from t = 0 (Lesson 3 section 3). This
      wrapper prepends a rest-to-rest quintic takeoff from the ground to the plan's
      first point, drops the lead-in and shifts the gate times, so the SAME plan can be
      scored on the clock the leaderboard uses.

  StartBlendTrajectory    a ground-start reference for a race that randomises the start
      pose -- Lesson 8 §5. A ground-start TOGT plan begins at a FIXED point on the floor,
      but Level 1 draws the drone's start within +-0.1 m of it. Holding the plan's first
      point would then demand a rest-to-rest slide across the floor while the rotors are
      still spinning up. This wrapper holds the plan's first point and carries the
      OBSERVED offset alongside it, fading the offset out with a quintic once the hold
      ends.

  closed_form_line(gates, cruise)   `lsy_level2_race()`'s racing line (freestyle.py,
      lines 420-430) rebuilt from the gate poses the race hands a bridge (Lesson 3
      section 4), so a bridge never hard-codes a track.

Plus three diagnostics the planner lesson leans on: crossing angles at the gates, every
crossing of every gate PLANE (feasibility_report's summary bit made visible), and the
clearance from the obstacle poles that neither TOGT nor crazy_track's evaluators model.
"""

from __future__ import annotations

import numpy as np

from crazy_track.trajectories.base import Trajectory
from crazy_track.trajectories.freestyle import (
    LSY_LEVEL2_OBSTACLES,
    FreestyleTrajectory,
    RaceGate,
    _QuinticSegment,
)

# config/level{0,1,2}.toml, [[env.track.drones]] pos: the race starts here, on the ground.
RACE_START = np.array([-1.5, 0.75, 0.01])
# lsy assets/obstacle.xml: a capsule of radius 0.015 m, 1.6 m long, standing on the floor;
# the toml `pos` is the TOP of the pole.
POLE_RADIUS = 0.015


class GroundStartTrajectory(Trajectory):
    """`plan` (hover start, stationary lead-in) rewritten for a ground start at t = 0.

    t in [0, takeoff_t):  rest-to-rest quintic from `start` to the plan's first point
    t >= takeoff_t:       the plan, with its lead-in removed
    gate_times shift by (takeoff_t - plan.lead_in); `lead_in` is 0 so that
    race_time = t_cross - lead_in is measured from t = 0, like the leaderboard.

    start=None is the HOVER start on the race clock: the reference begins at the plan's
    first point (where a hover-start race config places the drone, Lesson 7 section 1);
    takeoff_t is then a settle HOLD at that point (0 = the plan moves from t = 0). With a
    hold of T the race clock reads T more than the benchmark clock's motion-onset time,
    which is what makes the two clocks comparable: subtract the hold.
    """

    def __init__(self, plan: Trajectory, start=RACE_START, takeoff_t: float = 1.5):
        self.plan = plan
        self.takeoff_t = float(takeoff_t)
        self.plan_lead_in = float(getattr(plan, "lead_in", 0.0))
        p1 = np.asarray(plan.pos(self.plan_lead_in), dtype=np.float64)
        # start=None: HOVER start at the plan's first point; takeoff_t then is a settle
        # hold at that point (lsy starts the rotors from rest, so a drone released at
        # hover height sags ~0.5 m in the first 0.4 s -- the benchmark's lead-in hid it)
        self.start = p1 if start is None else np.asarray(start, dtype=np.float64)
        zero = np.zeros(3)
        if self.takeoff_t > 0.0:
            self._takeoff = _QuinticSegment((self.start, zero, zero), (p1, zero, zero), self.takeoff_t)
        else:                                   # the plan from t = 0
            self._takeoff, self.takeoff_t = None, 0.0
        self.takeoff_target = p1
        self.lead_in = 0.0
        self.duration = self.takeoff_t + (float(plan.duration) - self.plan_lead_in)
        self.gates: list[RaceGate] = list(plan.gates)
        self.gate_times = [tg - self.plan_lead_in + self.takeoff_t for tg in plan.gate_times]
        self.flips: list[dict] = []
        self.obstacles = list(getattr(plan, "obstacles", []))

    def _split(self, t, method: str) -> np.ndarray:
        t = self._clamp(t)
        scalar = t.ndim == 0
        tt = np.atleast_1d(t)
        out = np.zeros(tt.shape + (3,))
        m = tt < self.takeoff_t
        if m.any():
            out[m] = getattr(self._takeoff, method)(tt[m])
        if (~m).any():
            out[~m] = getattr(self.plan, method)(tt[~m] - self.takeoff_t + self.plan_lead_in)
        return out[0] if scalar else out

    def pos(self, t):
        return self._split(t, "pos")

    def vel(self, t):
        return self._split(t, "vel")

    def acc(self, t):
        return self._split(t, "acc")

    def maneuver_descriptor(self, t) -> np.ndarray:
        t = np.asarray(t, dtype=np.float64)
        return np.zeros((6,) if t.ndim == 0 else t.shape + (6,))


class StartBlendTrajectory(Trajectory):
    """`inner` plus the start offset `d`, faded out with a quintic after the hold — Lesson 8 §5.

    `inner` is a GroundStartTrajectory built on the PLAN's first point p1 (so it holds p1 for
    `hold` seconds and then flies the plan), and `d = observed_start - p1` is how far the race
    actually put the drone from that point. The reference this class returns is

        p(t) = inner.pos(t) + w(t) * d,      w(t) = 1                            t <= hold
                                             w(t) = 1 - (10 s^3 - 15 s^4 + 6 s^5)
                                                               with s = (t - hold)/T,
                                                                          hold < t < hold + T
                                             w(t) = 0                            t >= hold + T

    so the drone is asked to stay where it is during the hold, and the offset is taken out in
    flight over T seconds. The weight is a quintic smoothstep: w, w' and w'' are continuous and
    w'(hold) = w''(hold) = 0; vel and acc carry the matching d*w' and d*w'' terms. The peak blend
    speed is 1.875 |d| / T and the peak blend acceleration 5.77 |d| / T^2 — for a 0.14 m draw over
    1 s that is 0.26 m/s and 0.81 m/s^2, against 0.88 m/s and 9.05 m/s^2 for the 0.3 s
    rest-to-rest quintic a plain GroundStartTrajectory would build from the observed start.

    Everything else — gates, gate_times, obstacles, duration, flips, lead_in, takeoff_t /
    takeoff_target and the manoeuvre descriptor — is the inner reference's, so the clock and the
    gate times are unchanged. With d = 0 (Level 0 on a plan whose first point IS the start pose)
    this is the inner reference exactly.
    """

    def __init__(self, inner, d, hold: float, T: float):
        self.inner = inner
        self.d = np.asarray(d, dtype=np.float64).reshape(3)
        self.hold, self.T = float(hold), float(T)
        if self.T <= 0.0:
            raise ValueError("StartBlendTrajectory needs T > 0")
        self.duration = float(inner.duration)
        self.gates = list(inner.gates)
        self.gate_times = list(inner.gate_times)
        self.obstacles = list(getattr(inner, "obstacles", []))
        self.flips = list(getattr(inner, "flips", []))
        self.lead_in = float(getattr(inner, "lead_in", 0.0))
        self.takeoff_t = float(getattr(inner, "takeoff_t", hold))
        self.takeoff_target = np.asarray(getattr(inner, "takeoff_target", inner.pos(self.takeoff_t)))
        self.plan = getattr(inner, "plan", inner)

    def _w(self, t):
        """w, w', w'' at t (scalar or array), shape t.shape each."""
        t = np.asarray(t, dtype=np.float64)
        s = np.clip((t - self.hold) / self.T, 0.0, 1.0)
        w = 1.0 - (10.0 * s**3 - 15.0 * s**4 + 6.0 * s**5)
        dw = -(30.0 * s**2 - 60.0 * s**3 + 30.0 * s**4) / self.T
        ddw = -(60.0 * s - 180.0 * s**2 + 120.0 * s**3) / self.T**2
        return w, dw, ddw

    def _blend(self, t, method: str, which: int) -> np.ndarray:
        t = self._clamp(t)
        base = getattr(self.inner, method)(t)
        wk = self._w(t)[which]
        return base + wk[..., None] * self.d

    def pos(self, t):
        return self._blend(t, "pos", 0)

    def vel(self, t):
        return self._blend(t, "vel", 1)

    def acc(self, t):
        return self._blend(t, "acc", 2)

    def maneuver_descriptor(self, t) -> np.ndarray:
        if hasattr(self.inner, "maneuver_descriptor"):
            return self.inner.maneuver_descriptor(t)
        t = np.asarray(t, dtype=np.float64)
        return np.zeros((6,) if t.ndim == 0 else t.shape + (6,))


def closed_form_line(gates: list[RaceGate], cruise: float = 3.0,
                     start=(-1.5, 0.75, 1.0), line: str = "lsy") -> FreestyleTrajectory:
    """The closed-form racing line, built from the gates handed in.

    line="lsy"   lsy_level2_race() verbatim (identical to it for the nominal poses). Its
                 exit from gate 3 and its approach to gate 4 pass ~0.1 m from the poles
                 the track designers stood 0.5 m out on those gate axes -- a faithful
                 tracker touches them above ~2 m/s (measured, Lesson 6 section 5).
    line="safe"  the same gates and hairpin, but gate 3 taken at 0.5 x cruise and three
                 vias after it that keep the drone south of poles 3 and 4 (>= 0.14 m from
                 their surface on the reference; the lsy line has 0.03-0.08 m). Found by a
                 small search over via positions with this file's obstacle_clearance and
                 feasibility_report (2026-09-10); about 0.8 s slower than "lsy" at cruise
                 2.5, and infeasible above cruise ~2.7 (no via placement clears both poles).
    """
    g1, g2, g3, g4 = gates[:4]
    c = cruise
    min_seg_T = min(1.0, 1.2 / c)
    if line == "lsy":
        ops = [
            ("gate", g1, 0.7 * c),
            ("via", (1.75, 0.5, 0.95), (0.0, 0.7 * c, 0.0)),
            ("gate", g2, 0.7 * c),
            ("gate", g3, c),
            ("via", (-1.7, -1.1, 0.95), (0.8 * c, 0.0, 0.0)),
            ("gate", g4, c),
            ("hover", (1.0, -0.75, 1.2), 1.0),
        ]
    elif line == "safe":
        ops = [
            ("gate", g1, 0.7 * c),
            ("via", (1.75, 0.5, 0.95), (0.0, 0.7 * c, 0.0)),
            ("gate", g2, 0.7 * c),
            ("gate", g3, 0.5 * c),
            # turn south-west right after gate 3: pole 3 stands 0.5 m behind it on its axis
            ("via", (-1.4, -0.7, 0.9), (-0.4 * c, -0.8 * c, 0.0)),
            ("via", (-1.7, -1.35, 0.95), (0.8 * c, 0.0, 0.0)),
            # stay south of pole 4 (0.5 m in front of gate 4 on its axis) until the last 0.55 m,
            # then a short, still-turning approach into the gate
            ("via", (-0.55, -0.95, 1.1), (0.85 * c, 0.4 * c, 0.0)),
            ("gate", g4, c),
            ("hover", (1.0, -0.75, 1.2), 1.0),
        ]
        min_seg_T = 0.2   # the approach legs are short by design; the time-scaler guards feasibility
    else:
        raise ValueError(f"line must be 'lsy' or 'safe', got {line!r}")
    traj = FreestyleTrajectory(start=start, ops=ops, cruise=c, min_seg_T=min_seg_T)
    traj.obstacles = LSY_LEVEL2_OBSTACLES
    return traj


# ----------------------------------------------------------------- diagnostics
def crossing_angles(traj) -> list[dict]:
    """Per gate: the angle (deg) between the reference velocity at the crossing and the
    gate normal, the in-plane speed and the total speed. A tracking lag of `dt` puts
    `dt * in-plane speed` of error INTO THE GATE PLANE: 0 deg is harmless, 90 deg is a
    miss regardless of how precise the tracker is."""
    out = []
    for g, tg in zip(traj.gates, traj.gate_times):
        v = np.asarray(traj.vel(tg), dtype=np.float64)
        v_n = float(v @ g.normal)
        v_p = float(np.linalg.norm(v - v_n * g.normal))
        out.append({"angle_deg": float(np.degrees(np.arctan2(v_p, v_n))),
                    "inplane_speed": v_p, "speed": float(np.linalg.norm(v))})
    return out


def plane_crossings(traj, dt: float = 0.002, clearance_margin: float = 0.07) -> list[list[dict]]:
    """Every crossing of every gate's PLANE, with its in-plane offset and a verdict:
    'opening' (inside the 0.4 m opening with margin), 'FRAME' (would hit the 0.72 m
    frame) or 'clear' (beside the frame). feasibility_report's `gate_crossings_ok`
    is the AND over all of these; this shows which crossing tripped it."""
    t = np.arange(0.0, traj.duration, dt)
    pos = traj.pos(t)
    open_bound = RaceGate.HALF_OPENING - clearance_margin
    frame_clear = RaceGate.EDGE_HALF + clearance_margin
    out = []
    for g, tg in zip(traj.gates, traj.gate_times):
        local = g.to_gate_frame(pos)
        x = local[:, 0]
        rows = []
        for i in np.flatnonzero(np.sign(x[1:]) != np.sign(x[:-1])):
            w = x[i] / (x[i] - x[i + 1])
            yz = local[i, 1:] + w * (local[i + 1, 1:] - local[i, 1:])
            off = float(np.abs(yz).max())
            tc = float(t[i] + w * dt)
            verdict = "opening" if off <= open_bound else ("clear" if off >= frame_clear else "FRAME")
            rows.append({"t": tc, "offset": off, "verdict": verdict,
                         "intended": abs(tc - tg) < 0.05})
        out.append(rows)
    return out


def obstacle_clearance(pos: np.ndarray, obstacles=LSY_LEVEL2_OBSTACLES,
                       pole_radius: float = POLE_RADIUS) -> list[float]:
    """Smallest horizontal distance from each pole's SURFACE over the samples that are
    below the pole top. The parent project's planner and evaluators ignore the poles;
    the race does not -- a contact ends the episode as a failure."""
    pos = np.asarray(pos, dtype=np.float64)
    out = []
    for ob in obstacles:
        below = pos[:, 2] < ob[2]
        if not below.any():
            out.append(float("inf"))
            continue
        d = np.linalg.norm(pos[below, :2] - np.asarray(ob[:2], dtype=np.float64), axis=1)
        out.append(float(d.min() - pole_radius))
    return out
