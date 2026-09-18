"""Identify the race simulator: scripted attitude probes — Lesson 8 §2.

The MPC predicts with crazyflow's *identified* attitude model `so_rpy` (Lesson 6 §3). Nobody
had checked that model against the simulator the race actually runs (crazyflow's
first_principles physics at 500 Hz, its Mellinger attitude loop and its rotor dynamics). This
file is one lsy `Controller` subclass that flies a SCRIPTED command sequence through the legal
attitude interface ([roll, pitch, yaw, thrust_N] at 50 Hz) and logs every 50 Hz sample
(t, pos, vel, quat, ang_vel, Euler angles, the command, the phase) to a CSV.
`probe_fit.py` turns those CSVs into the table of §2.

There is no solver in here: a probe episode costs 1-10 s on top of the ~30 s the race
environment needs to build.

Install (lsy's loader wants the file in that folder, ONE Controller subclass per file — copy
the FILE, not the folder; docs/4-troubleshooting.md):

    cp tasks/racing/code/race_probe.py repos/lsy_drone_racing/lsy_drone_racing/control/
    cd repos/lsy_drone_racing
    PROBE=climb PROBE_THRUST=0.8 /opt/venvs/race/bin/python scripts/sim.py \
        --config level0.toml --controller race_probe.py --n_runs 3 --render False

Probes (env PROBE), and what each one measures:

    climb       thrust = PROBE_THRUST (default 0.8 N) with a level attitude, from t = 0 on the
                ground. Ends when z > PROBE_ZMAX (1.6 m) or t >= PROBE_T (2.0 s). Gives the
                rotor spin-up from rest, the lift-off time, the thrust gain and, from the
                airborne part, the linear drag in z.
    hold        thrust = PROBE_THRUST (default 0.4256 N = m*g) on the ground for PROBE_T
                (2.0 s): does the drone lift at m*g? Does it lift at the MPC's hover (0.4395)?
    hoverstep   a PD climb to PROBE_HOVER (-2.0, 0.75, 1.0) along a 1.5 s quintic, settle, then
                at t = PROBE_T0 (3.0 s) OPEN-LOOP roll steps +A, -A, 0 (A = PROBE_STEP, 0.3 rad,
                PROBE_STEP_T 0.5 s each) at the settled thrust; PD recovery to the hover point;
                then the same PITCH steps; PD recovery; end at PROBE_T (9.5 s). This is the
                probe behind the attitude block (a, b, c), its dc gain, rise time and damping.
    thruststep  PD hover as above; at t = PROBE_T0 thrust -> PROBE_FSTEP (0.6 N) for 0.3 s with
                a level attitude, then the mirrored down-step (2*f_hold - 0.6) for 0.3 s, then
                PD recovery; end at PROBE_T (5.0 s). Gives the thrust gain and the thrust lag.
    lateral     PD hover; at t = PROBE_T0 pitch = +PROBE_PITCH (0.5 rad) with thrust =
                f_hold / cos(pitch) for 0.5 s, then pitch = -PROBE_PITCH for 0.5 s, then level
                for 0.2 s, then PD recovery; end at PROBE_T (5.5 s). Gives the tilt-to-
                acceleration map at a large angle, where the 0.73 vs 0.94 gain shows up as
                30 % of missing acceleration.

f_hold = the PD's settled thrust command (mean of the last 0.2 s before PROBE_T0): the
simulator's true hover thrust. It is written into the CSV's header comment.

Other knobs: PROBE_T (episode length), PROBE_TILT (climb only: a pitch held from t = 0, to see
whether the attitude loop works while the drone still sits on the floor), PROBE_T0, PROBE_ZMAX,
PROBE_CLIMB_T, PROBE_LAT_T, PROBE_HOVER ("x,y,z"), PROBE_TAG (a file-name label) and
PROBE_LOG_DIR (default tasks/racing/figures/probes). CSV name:
<PROBE>_<TAG>_ep<NN>.csv (NN counts the existing files with that prefix).

Timing convention of the log: row i holds the observation at t_i = i / 50 (before the command
is applied) and the command returned at that tick (held over [t_i, t_i + 0.02)). The earliest
sample that can show a response to the command of row i is row i + 1 — which is how
`probe_fit.py` can put a bound on the pure delay.

Level 0's action noise and dynamics disturbance stay ON: these are probes of the environment
the race is scored in, not of a clean model.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from lsy_drone_racing.control.controller import Controller
from scipy.spatial.transform import Rotation as R

G = 9.81
MASS = 0.04338                        # cf21B_500 (crazyflow drones/params.toml)
CMD_F_COEF = 0.96836458               # so_rpy identified thrust gain
THRUST_MIN = 4 * 0.02136263065537499  # 0.0855 N
THRUST_MAX = 4 * 0.2                  # 0.8 N
HOVER_MG = MASS * G                   # 0.4256 N -- what the drone actually needs
HOVER_MODEL = HOVER_MG / CMD_F_COEF   # 0.4395 N -- what MPCController.hover believes
RPY_MAX = 1.0

PHASE = {"pd": 0, "open": 1, "catch": 2, "climb": 3, "hold": 4}


def _env(name: str, default):
    v = os.environ.get(name)
    if v is None:
        return default
    return type(default)(v) if not isinstance(default, str) else v


def acc2attitude(a_des: np.ndarray, quat_xyzw: np.ndarray, mass: float = MASS,
                 yaw_des: float = 0.0) -> np.ndarray:
    """Desired CoM acceleration -> [roll, pitch, yaw, thrust_N] (a copy of
    crazy_track.controllers.utils.acc2attitude, so the probe has no crazy_track dependency and
    runs from the race venv alone)."""
    f_des = mass * (a_des + np.array([0.0, 0.0, G]))
    z_body = R.from_quat(quat_xyzw).as_matrix()[:, 2]
    thrust = float(np.clip(np.dot(f_des, z_body), THRUST_MIN, THRUST_MAX))
    z_des = f_des / max(np.linalg.norm(f_des), 1e-6)
    x_c = np.array([np.cos(yaw_des), np.sin(yaw_des), 0.0])
    y_des = np.cross(z_des, x_c)
    y_des /= max(np.linalg.norm(y_des), 1e-6)
    x_des = np.cross(y_des, z_des)
    rpy = R.from_matrix(np.stack([x_des, y_des, z_des], axis=-1)).as_euler("xyz")
    rpy[:2] = np.clip(rpy[:2], -RPY_MAX, RPY_MAX)
    return np.array([rpy[0], rpy[1], rpy[2], thrust])


def quintic(p0: np.ndarray, p1: np.ndarray, T: float, t: float):
    """Rest-to-rest quintic p0 -> p1 over T seconds: (pos, vel, acc) at t (clamped)."""
    tau = min(max(t / T, 0.0), 1.0)
    s = 6 * tau**5 - 15 * tau**4 + 10 * tau**3
    ds = (30 * tau**4 - 60 * tau**3 + 30 * tau**2) / T
    dds = (120 * tau**3 - 180 * tau**2 + 60 * tau) / T**2
    d = p1 - p0
    return p0 + s * d, ds * d, dds * d


def _repo_root() -> Path:
    """The repo root, found from inside the lsy clone as well as from the repo.

    lsy's loader wants the controller file INSIDE its clone, so the probes are run as
    `cd repos/lsy_drone_racing && ... scripts/sim.py --controller race_probe.py` -- and this
    file then sits in the clone, where nothing above it looks like the repo. Use the same
    anchor the bridges use (`RACE_CODE_DIR`, the folder holding race_refs.py), fall back to
    walking up from this file for a run out of the repo, and finally to the container path.
    """
    code = os.environ.get("RACE_CODE_DIR")
    if code:
        root = Path(code).resolve().parents[2]          # tasks/racing/code -> the repo root
        if (root / "tasks" / "racing" / "code").is_dir():
            return root
    for parent in Path(__file__).resolve().parents:
        if (parent / "tasks" / "racing" / "code" / "race_refs.py").exists():
            return parent
    return Path("/workspace")


def _resolve_log_dir(value: str) -> Path:
    """PROBE_LOG_DIR: absolute as given, relative to the REPO ROOT (not the cwd)."""
    p = Path(value)
    return p if p.is_absolute() else _repo_root() / p


class RaceProbeController(Controller):
    """Scripted attitude-interface probes with a 50 Hz state log."""

    def __init__(self, obs: dict, info: dict, config: dict):
        super().__init__(obs, info, config)
        self.freq = int(config.env.freq)
        self.dt = 1.0 / self.freq
        self.probe = os.environ.get("PROBE", "climb")
        if str(config.env.control_mode) != "attitude":
            raise RuntimeError(
                'race_probe needs control_mode = "attitude" in the race config. '
                "Edit [env] control_mode in repos/lsy_drone_racing/config/<level>.toml (Lesson 3 §4)."
            )
        defaults_T = {"climb": 2.0, "hold": 2.0, "hoverstep": 9.5, "thruststep": 5.0,
                      "lateral": 5.5}
        if self.probe not in defaults_T:
            raise ValueError(f"unknown PROBE {self.probe!r}: {sorted(defaults_T)}")
        self.T = _env("PROBE_T", defaults_T[self.probe])
        self.thrust = _env("PROBE_THRUST", 0.8 if self.probe == "climb" else HOVER_MG)
        self.zmax = _env("PROBE_ZMAX", 1.6)
        self.t0 = _env("PROBE_T0", 3.0)
        self.step_a = _env("PROBE_STEP", 0.3)
        self.step_t = _env("PROBE_STEP_T", 0.5)
        self.f_step = _env("PROBE_FSTEP", 0.6)
        self.pitch_lat = _env("PROBE_PITCH", 0.5)
        self.lat_t = _env("PROBE_LAT_T", 0.5)
        self.climb_t = _env("PROBE_CLIMB_T", 1.5)
        self.tilt = _env("PROBE_TILT", 0.0)     # climb only: pitch command held from t = 0
        hov = os.environ.get("PROBE_HOVER", "-2.0,0.75,1.0")
        self.home = np.array([float(v) for v in hov.split(",")])
        self.log_dir = _resolve_log_dir(
            os.environ.get("PROBE_LOG_DIR", "tasks/racing/figures/probes"))
        self.tag = os.environ.get("PROBE_TAG", "run")

        self.start = np.asarray(obs["pos"], dtype=np.float64).reshape(3)
        # PD (crazy_track PIDController gains: kp = omega^2, kd = 2 zeta omega, ki)
        self.kp, self.kd, self.ki, self.int_max = 16.0, 8.0, 2.0, 0.5
        self._int = np.zeros(3)
        self._tick = 0
        self._rows: list[list[float]] = []
        self._pd_thrust: list[float] = []   # PD thrust history (for f_hold)
        self._f_hold: float | None = None
        self._catch_from: np.ndarray | None = None
        self._catch_t: float | None = None
        self._end_reason = ""
        # segments of the scripted part: list of (t_start, t_end, kind, params)
        self._script = self._build_script()

    # ------------------------------------------------------------------ script
    def _build_script(self):
        A, Ts, t0 = self.step_a, self.step_t, self.t0
        seg = []
        if self.probe == "hoverstep":
            # roll steps
            seg += [(t0, t0 + Ts, "rpy", (A, 0.0)), (t0 + Ts, t0 + 2 * Ts, "rpy", (-A, 0.0)),
                    (t0 + 2 * Ts, t0 + 3 * Ts, "rpy", (0.0, 0.0))]
            t1 = t0 + 3 * Ts + 2.0            # 2.0 s PD recovery
            self._pitch_t0 = t1
            seg += [(t1, t1 + Ts, "rpy", (0.0, A)), (t1 + Ts, t1 + 2 * Ts, "rpy", (0.0, -A)),
                    (t1 + 2 * Ts, t1 + 3 * Ts, "rpy", (0.0, 0.0))]
        elif self.probe == "thruststep":
            seg += [(t0, t0 + 0.3, "thrust", "up"), (t0 + 0.3, t0 + 0.6, "thrust", "down")]
        elif self.probe == "lateral":
            P, Tl = self.pitch_lat, self.lat_t
            seg += [(t0, t0 + Tl, "lat", P), (t0 + Tl, t0 + 2 * Tl, "lat", -P),
                    (t0 + 2 * Tl, t0 + 2 * Tl + 0.2, "lat", 0.0)]
        return seg

    def _segment(self, t: float):
        for s in self._script:
            if s[0] <= t < s[1]:
                return s
        return None

    # ------------------------------------------------------------------ PD hover
    def _setpoint(self, t: float):
        """(pos, vel, acc) of the PD reference: a quintic climb from the start to home, then
        hold; after an open-loop segment, a 1.0 s quintic from the catch point home."""
        if self._catch_t is not None and t >= self._catch_t:
            return quintic(self._catch_from, self.home, 1.0, t - self._catch_t)
        return quintic(self.start, self.home, self.climb_t, t)

    def _pd(self, pos, vel, quat, t):
        p_ref, v_ref, a_ref = self._setpoint(t)
        err = p_ref - pos
        self._int = np.clip(self._int + err * self.dt, -self.int_max, self.int_max)
        a_cmd = a_ref + self.kp * err + self.kd * (v_ref - vel) + self.ki * self._int
        u = acc2attitude(a_cmd, quat)
        return u, p_ref

    # ------------------------------------------------------------------ control
    def compute_control(self, obs: dict, info: dict | None = None) -> np.ndarray:
        t = self._tick * self.dt
        pos = np.asarray(obs["pos"], dtype=np.float64).reshape(3)
        vel = np.asarray(obs["vel"], dtype=np.float64).reshape(3)
        quat = np.asarray(obs["quat"], dtype=np.float64).reshape(4)
        omega = np.asarray(obs["ang_vel"], dtype=np.float64).reshape(3)
        rpy = R.from_quat(quat).as_euler("xyz")
        sp = np.full(3, np.nan)
        phase = PHASE["pd"]

        if self.probe == "climb":
            u = np.array([0.0, self.tilt, 0.0, self.thrust])
            phase = PHASE["climb"]
        elif self.probe == "hold":
            u = np.array([0.0, 0.0, 0.0, self.thrust])
            phase = PHASE["hold"]
        else:
            seg = self._segment(t)
            if seg is None:
                # PD phase (before t0, or the recovery after an open-loop segment)
                if self._catch_from is not None and self._catch_t is None:
                    self._catch_from, self._catch_t = pos.copy(), t
                    self._int[:] = 0.0
                u, sp = self._pd(pos, vel, quat, t)
                if t < self.t0:
                    self._pd_thrust.append(u[3])
                phase = PHASE["pd"] if self._catch_t is None else PHASE["catch"]
            else:
                if self._f_hold is None:
                    n = max(1, int(0.2 * self.freq))
                    self._f_hold = float(np.mean(self._pd_thrust[-n:])) if self._pd_thrust else HOVER_MODEL
                self._catch_from, self._catch_t = pos.copy(), None   # arm the catch for later
                phase = PHASE["open"]
                kind, par = seg[2], seg[3]
                if kind == "rpy":
                    u = np.array([par[0], par[1], 0.0, self._f_hold])
                elif kind == "thrust":
                    f = self.f_step if par == "up" else max(THRUST_MIN, 2 * self._f_hold - self.f_step)
                    u = np.array([0.0, 0.0, 0.0, f])
                else:  # lateral
                    f = self._f_hold / np.cos(self.pitch_lat) if par != 0.0 else self._f_hold
                    u = np.array([0.0, par, 0.0, min(THRUST_MAX, f)])
        u = np.asarray(u, dtype=np.float64)
        self._rows.append([t, phase, *pos, *vel, *quat, *omega, *rpy, *u, *sp])
        return u.astype(np.float32)

    def step_callback(self, action, obs, reward, terminated, truncated, info) -> bool:
        self._tick += 1
        t = self._tick * self.dt
        z = float(np.asarray(obs["pos"]).reshape(3)[2])
        if self.probe == "climb" and z > self.zmax:
            self._end_reason = f"z>{self.zmax}"
            return True
        if t >= self.T:
            self._end_reason = "PROBE_T"
            return True
        if terminated or truncated:
            self._end_reason = "env_terminated" if terminated else "env_truncated"
        return False

    def episode_callback(self):
        self._write()

    def episode_reset(self):
        if self._rows:          # written already in episode_callback, unless it was skipped
            self._write()

    def _write(self):
        if not self._rows:
            return
        self.log_dir.mkdir(parents=True, exist_ok=True)
        prefix = f"{self.probe}_{self.tag}_ep"
        idx = len(list(self.log_dir.glob(prefix + "*.csv")))
        path = self.log_dir / f"{prefix}{idx:02d}.csv"
        hdr = ("t,phase,x,y,z,vx,vy,vz,qx,qy,qz,qw,wx,wy,wz,roll,pitch,yaw,"
               "cmd_roll,cmd_pitch,cmd_yaw,cmd_thrust,sx,sy,sz")
        meta = (f"# probe={self.probe} T={self.T} thrust={self.thrust} t0={self.t0} "
                f"step={self.step_a} step_t={self.step_t} fstep={self.f_step} "
                f"pitch={self.pitch_lat} tilt={self.tilt} f_hold={self._f_hold} home={self.home.tolist()} "
                f"start={self.start.tolist()} end={self._end_reason or 'loop_exit'} "
                f"n={len(self._rows)}")
        with open(path, "w") as fh:
            fh.write(meta + "\n")
            np.savetxt(fh, np.asarray(self._rows, dtype=np.float64), delimiter=",",
                       header=hdr, comments="", fmt="%.6f")
        print(f"[race_probe] wrote {path} ({len(self._rows)} rows, end={self._end_reason}, "
              f"f_hold={self._f_hold})")
        self._rows = []
