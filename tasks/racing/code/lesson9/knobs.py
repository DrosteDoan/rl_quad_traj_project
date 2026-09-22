"""The lambda knobs: disturbance and sensor objects whose severity is a fraction of a range (Lesson 9 phase 3).

    dist, sensor = make_conditions("wind_gust", lam=0.4, seed=7)        # (None, None) at lam = 0

`lam` in [0, 1] is the fraction of each disturbance's own range: lam = 0 is exactly nominal (no object at
all, so a run is bit-identical to one without the knob), lam = 1 is the ceiling. The ceiling of every
disturbance is `SCALES[name]` times Lesson 7's value (`NOMINAL`), so lesson 7's condition is lam = 1/scale:
with the starting scale of 2 it is lam = 0.5. Phase 5 calibrates the scales on the dev tracks and freezes them
in `maxima.json` (METHODOLOGY.md section 4); this module reads that file if it exists.

What lam scales (METHODOLOGY.md section 4), with lam_eff = lam * scale, so lam_eff = 1 is Lesson 7's value:
    wind_const   the steady force along +x
    payload      the downward force, extra_mass * g
    wind_gust    mean push, sinusoid amplitude and turbulence sigma together (0.7 Hz and tau = 0.5 s stay fixed)
    lighthouse   the size errors (jitter, per-run bias, velocity, attitude, gyro) AND the position update
                 interval together; the 1-step (10 ms) latency is fixed whenever it is on (a jump at 0+)
    combined     gust + payload + Lighthouse, each at the same lam (never wind_const: it stacks with the gust mean)

Common random numbers: every random quantity is a stream of unit normals fixed by (seed, channel), only scaled
by lam. One seed is the same draw at every lam and for every member, so curves are paired. At lam_eff = 1 the
gust reproduces `crazy_track.disturbances.GustWind(seed)` exactly (same generator, same order); the Lighthouse
sensor has the same distributions as `crazy_track.sensors.LighthouseSensor` but per-channel streams (it cannot
also keep that class's interleaved draw order and stay paired across lam).

The objects are indexed by control step, t = i * dt, one call per step from i = 0, as `envs.rollout.rollout`
does; `horizon` steps of noise are drawn up front.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

from crazy_track.disturbances import Disturbance

HERE = Path(__file__).resolve().parent
G = 9.81

# Lesson 7's conditions, as `crazy_track.disturbances` / `race_eval.build_conditions` apply them
NOMINAL = {
    "wind_const": {"force": (0.11, 0.0, 0.0)},
    "payload": {"extra_mass": 0.010},
    "wind_gust": {"mean": (0.08, 0.0, 0.0), "gust_amp": 0.08, "gust_dir": (1.0, 0.3, 0.0), "gust_hz": 0.7,
                  "ou_sigma": 0.04, "ou_tau": 0.5},
    "lighthouse": {"update_hz": 34.0, "update_hz_std": 18.0, "jitter_std": 0.0007, "bias_std": 0.015,
                   "vel_std": 0.03, "att_std_deg": 0.5, "gyro_std": 0.02, "latency_steps": 1},
}
CONDITIONS = ("wind_const", "payload", "wind_gust", "lighthouse", "combined")
EXTRA_CONDITIONS = ("latency_only",)      # calibration only: the fixed 1-step Lighthouse delay and nothing else
START_SCALES = {"wind_const": 2.0, "payload": 2.0, "wind_gust": 2.0, "lighthouse": 2.0}
MAXIMA_FILE = HERE / "maxima.json"
_scales_override: dict | None = None


def scales() -> dict:
    """The ceiling of each disturbance in units of Lesson 7's value: the frozen file, else the starting 2x."""
    if _scales_override is not None:
        return dict(_scales_override)
    if MAXIMA_FILE.exists():
        return {**START_SCALES, **json.loads(MAXIMA_FILE.read_text())["scales"]}
    return dict(START_SCALES)


def set_scales(new: dict | None) -> None:
    """In-process override (phase 5 calibration); None goes back to the file / the starting values."""
    global _scales_override
    _scales_override = None if new is None else {**START_SCALES, **new}


# ---- disturbances (world-frame force on the centre of mass, N) ---------------------------------------------------
class ScaledWind(Disturbance):
    name = "wind_const"

    def __init__(self, lam_eff: float):
        self.f = float(lam_eff) * np.asarray(NOMINAL["wind_const"]["force"], dtype=np.float64)

    def force(self, t, state):
        return self.f


class ScaledPayload(Disturbance):
    name = "payload"

    def __init__(self, lam_eff: float):
        self.f = np.array([0.0, 0.0, -float(lam_eff) * NOMINAL["payload"]["extra_mass"] * G])

    def force(self, t, state):
        return self.f


class ScaledGust(Disturbance):
    """`crazy_track.disturbances.GustWind` with its three amplitudes multiplied by lam_eff, driven by the same unit
    normals: the Ornstein-Uhlenbeck part is linear in sigma, so it is stored once for sigma = 1 and scaled."""

    name = "wind_gust"

    def __init__(self, lam_eff: float, seed: int = 0, dt: float = 0.01, horizon: int = 6000):
        p = NOMINAL["wind_gust"]
        self.lam, self.dt, self.horizon = float(lam_eff), float(dt), int(horizon)
        self.mean = np.asarray(p["mean"], dtype=np.float64)
        self.amp, self.hz = p["gust_amp"], p["gust_hz"]
        self.dir = np.asarray(p["gust_dir"], dtype=np.float64)
        self.sigma = p["ou_sigma"]
        a = self.dt / p["ou_tau"]
        z = np.random.default_rng(int(seed)).normal(size=(self.horizon, 3))     # GustWind draws size=3 per call
        ou = np.zeros((self.horizon, 3))
        state = np.zeros(3)
        for i in range(self.horizon):                                            # the update comes BEFORE the read
            state = state - a * state + np.sqrt(2.0 * a) * z[i]
            ou[i] = state
        self._ou_unit = ou

    def force(self, t, state):
        i = int(round(t / self.dt))
        if not 0 <= i < self.horizon:
            raise IndexError(f"step {i} outside the {self.horizon}-step noise horizon")
        gust = self.amp * np.sin(2.0 * np.pi * self.hz * t) * self.dir
        return self.lam * (self.mean + gust + self.sigma * self._ou_unit[i])


class SumDisturbance(Disturbance):
    name = "sum"

    def __init__(self, parts: list[Disturbance]):
        self.parts = parts

    def reset(self, rng=None):
        for p in self.parts:
            p.reset()

    def force(self, t, state):
        return sum(p.force(t, state) for p in self.parts)


# ---- the Lighthouse measurement model --------------------------------------------------------------------------
class ScaledLighthouse:
    """`crazy_track.sensors.LighthouseSensor` with size errors and update interval scaled by lam_eff.

    Position is a zero-order hold refreshed at a random interval; velocity, attitude and gyro carry noise; a
    per-run position bias is drawn once. lam_eff = 1 is the literature model (34 +- 18 Hz, 0.7 mm jitter, 1.5 cm
    bias, 0.03 m/s, 0.5 deg, 0.02 rad/s). The refresh interval is  dt + lam_eff * (T - dt)  with T the drawn
    nominal interval, so it is one control step at lam = 0 and T at lam_eff = 1. The 1-step latency (position,
    velocity, attitude; the gyro is an onboard IMU and is not delayed) is fixed. Every noise channel is its own
    stream of unit normals, so the draws do not depend on lam.
    """

    CHANNELS = ("bias", "jitter", "interval", "vel", "att", "gyro")

    def __init__(self, lam_eff: float, seed: int = 0, control_freq: int = 100, horizon: int = 6000):
        p = NOMINAL["lighthouse"]
        self.lam, self.dt, self.horizon = float(lam_eff), 1.0 / control_freq, int(horizon)
        self.p = p
        children = np.random.SeedSequence(int(seed)).spawn(len(self.CHANNELS))
        rng = {c: np.random.default_rng(s) for c, s in zip(self.CHANNELS, children)}
        self._z_bias = rng["bias"].normal(size=3)
        self._z = {c: rng[c].normal(size=(self.horizon, 3)) for c in ("jitter", "vel", "att", "gyro")}
        self._z_int = rng["interval"].normal(size=self.horizon)
        # sizes at this lam
        self.jitter_std, self.vel_std = self.lam * p["jitter_std"], self.lam * p["vel_std"]
        self.att_std, self.gyro_std = np.radians(self.lam * p["att_std_deg"]), self.lam * p["gyro_std"]
        self.bias = self.lam * p["bias_std"] * self._z_bias
        self.latency = int(p["latency_steps"])
        self.reset()

    def reset(self) -> None:
        self.next_update_t = 0.0
        self.n_updates = 0
        self.last_pos_meas: np.ndarray | None = None
        self.buffer: list[np.ndarray] = []

    def _interval(self) -> float:
        p = self.p
        hz = float(np.clip(p["update_hz"] + p["update_hz_std"] * self._z_int[self.n_updates], 8.0, 100.0))
        self.n_updates += 1
        return self.dt + self.lam * (1.0 / hz - self.dt)

    def measure(self, t: float, true_state: np.ndarray) -> np.ndarray:
        """true_state: [pos(3), vel(3), quat_xyzw(4), omega(3)] -> the (delayed, noisy) copy the controller sees."""
        i = int(round(t / self.dt))
        if not 0 <= i < self.horizon:
            raise IndexError(f"step {i} outside the {self.horizon}-step noise horizon")
        pos, vel, quat, omega = (true_state[:3], true_state[3:6], true_state[6:10], true_state[10:13])
        if self.last_pos_meas is None or t >= self.next_update_t - 1e-9:     # tolerance: i * dt is inexact
            self.last_pos_meas = pos + self.bias + self.jitter_std * self._z["jitter"][i]
            self.next_update_t = t + self._interval()
        vel_m = vel + self.vel_std * self._z["vel"][i]
        quat_m = (R.from_rotvec(self.att_std * self._z["att"][i]) * R.from_quat(quat)).as_quat()
        omega_m = omega + self.gyro_std * self._z["gyro"][i]
        meas = np.concatenate([self.last_pos_meas, vel_m, quat_m, omega_m])
        self.buffer.append(meas)
        if len(self.buffer) > self.latency + 1:
            self.buffer.pop(0)
        out = self.buffer[0].copy()
        out[10:13] = omega_m                                       # the gyro is not delayed
        return out


# ---- the entry point -------------------------------------------------------------------------------------------
def make_conditions(cond: str, lam: float, seed: int = 0, control_freq: int = 100, horizon: int = 6000,
                    scale: dict | None = None):
    """(disturbance, sensor) for `cond` at severity `lam` in [0, 1]. (None, None) at lam = 0."""
    if cond not in CONDITIONS + EXTRA_CONDITIONS:
        raise ValueError(f"unknown condition {cond!r}; one of {CONDITIONS + EXTRA_CONDITIONS}")
    if not 0.0 <= lam <= 1.0:
        raise ValueError(f"lam must be in [0, 1], got {lam}")
    if lam == 0.0:
        return None, None
    if cond == "latency_only":                       # any lam > 0: the delay, with every error size at zero
        return None, ScaledLighthouse(0.0, seed, control_freq, horizon)
    sc = {**scales(), **(scale or {})}
    dt = 1.0 / control_freq
    if cond == "wind_const":
        return ScaledWind(lam * sc["wind_const"]), None
    if cond == "payload":
        return ScaledPayload(lam * sc["payload"]), None
    if cond == "wind_gust":
        return ScaledGust(lam * sc["wind_gust"], seed, dt, horizon), None
    sensor = ScaledLighthouse(lam * sc["lighthouse"], seed, control_freq, horizon)
    if cond == "lighthouse":
        return None, sensor
    return SumDisturbance([ScaledGust(lam * sc["wind_gust"], seed, dt, horizon),
                           ScaledPayload(lam * sc["payload"])]), sensor


def describe(cond: str, lam: float, scale: dict | None = None) -> dict:
    """The physical values behind lam (for axes and tables): forces in N and m/s^2, sensor sizes."""
    sc = {**scales(), **(scale or {})}
    out: dict = {}
    m = 0.04338
    if cond in ("wind_const",):
        f = lam * sc["wind_const"] * NOMINAL["wind_const"]["force"][0]
        out["wind_force_N"], out["wind_acc_m_s2"] = f, f / m
    if cond in ("payload", "combined"):
        f = lam * sc["payload"] * NOMINAL["payload"]["extra_mass"] * G
        out["payload_force_N"], out["payload_acc_m_s2"] = f, f / m
    if cond in ("wind_gust", "combined"):
        le = lam * sc["wind_gust"]
        g = NOMINAL["wind_gust"]
        out["gust_mean_N"], out["gust_amp_N"], out["gust_ou_sigma_N"] = (le * g["mean"][0], le * g["gust_amp"],
                                                                        le * g["ou_sigma"])
    if cond in ("lighthouse", "combined"):
        le = lam * sc["lighthouse"]
        lh = NOMINAL["lighthouse"]
        out["lh_bias_cm"], out["lh_vel_noise"], out["lh_att_deg"] = (le * lh["bias_std"] * 100, le * lh["vel_std"],
                                                                    le * lh["att_std_deg"])
        out["lh_mean_interval_ms_vs_nominal"] = f"dt + {le:.2f} * (nominal - dt)"
    return out
