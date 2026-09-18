"""The vendored MPC, re-implemented with switches — Lesson 8 §4.

`MPCDev` is a self-contained copy of the tracker the racing task has been flying since
Lesson 6,

    tasks/racing/crazy_track/src/crazy_track/controllers/mpc.py   (MPCController)

with the SAME `act(state, t)` / `reset(trajectory)` interface, the SAME 12-state so_rpy
Euler-angle model built from crazyflow's identified parameters, the same multiple-shooting
NLP in CasADi Opti + ipopt with warm start, and the same ESO / L1 disturbance estimators.
The difference is that every part of it you might want to change — the model, the
integrator, the horizon, the weights, the lookahead, the estimator, a mass adaptation — is
an OPTION, and every option defaults to what the vendored file does. `mpcdev` and `mpc`
are the same controller: max |Δu| = 0 over 40 steps of the same state sequence, at 50 Hz
and at 100 Hz, for the plain, the ESO and the L1 variants (receipt:
`mpcgap/logs/check_dev.log` in the instructor's archive; `test_race_refs.py` re-checks
10 steps of it in the main venv).

The vendored `mpc.py` stays exactly as it was identified upstream — it is a snapshot of a
research repository and is never edited. The corrected model lives here.

Where the specs are accepted: `race_eval.py --controller`, the bridge's `RACE_CONTROLLER`
and `race_runner.py` (through the bridge).

    mpcdev                        the vendored controller, exactly
    mpcdev:key=val,key=val,...    unknown keys, duplicate keys and bad values raise

    key          default   meaning
    int          euler     discretisation of xdot over dtp: euler (vendored, x + dtp*xdot) | rk4
    H            20        horizon length (prediction steps)
    dtp          0.04      prediction step [s] (horizon = H*dtp = 0.8 s)
    wp           1.0       stage weight on |p_k - p_ref,k|^2
    wv           0.05      stage weight on |v_k - v_ref,k|^2
    wu           0.02      weight on |roll_cmd, pitch_cmd|^2
    wf           0.02      weight on (thrust - hover_thrust)^2
    wdu          0.1       weight on |u_k - u_{k-1}|^2 (k >= 1)
    wt           0.0       EXTRA terminal position weight on X[:, H] vs the last reference
                           sample (0 = off)
    tau          0.0       lookahead [s]: the reference is sampled at t + tau + dtp*k, k = 1..H
    wa           0.0       reference-acceleration feed-forward: weight on
                           |acc_model(x_k, u_k) - a_ref,k|^2
    accff        auto      0/1 switch for that term; default = (wa > 0). accff=1 with wa=0
                           adds nothing.
    dist         none      disturbance model in the prediction: none | eso | l1 (the vendored
                           estimators, unchanged: mpc_offsetfree = dist=eso, mpc_l1 = dist=l1)
    eso_w        7.0       velocity-ESO bandwidth [rad/s]          (dist=eso)
    l1_as        -5.0      L1 A_s                                  (dist=l1)
    l1_c         4.0       L1 low-pass cutoff [Hz]                 (dist=l1)
    ramp         time      estimator soft-start: time = min(1, t/ramp_t) from t = 0 (vendored)
                           | liftoff = 0 until the drone has left the ground
                             (z > z_first + liftoff_dz), the estimator states held
                             (sigma = 0, v_hat = measured v) until then, then
                             min(1, (t - t_liftoff)/ramp_t) | none = full authority from t = 0
    ramp_t       1.5       soft-start duration [s]
    liftoff_dz   0.03      lift-off threshold above the z of the first act() call [m]; a start
                           above z = 0.5 (a hover-start toml) counts as airborne at once
    mass         0         thrust-scale adaptation: 1 = estimate a scale k from the vertical
                           acceleration residual once airborne; the model uses cmd_f_coef * k
    mass_gain    2.0       adaptation gain [1/s] of the first-order estimate
    mass_min     0.8       lower clip of k
    mass_max     1.25      upper clip of k
    mass_delay   0.3       seconds after lift-off before k starts adapting (spin-up transient)
    rp_max       1.0       |roll|, |pitch| command bound [rad]
    yaw_max      0.3       |yaw| command bound [rad]
    iter         60        ipopt max_iter
    tol          1e-4      ipopt tol
  -- the MEASURED model of Lesson 8 §2, all default = vendored --
    att          sorpy     roll AND pitch parameters (rpy_coef, rpy_rates_coef, cmd_rpy_coef) in
                           the convention ddtheta = a*theta + b*dtheta + c*u: sorpy =
                           crazyflow's (-189.0, -12.8, 138.1) (vendored) | sim = the race
                           simulator's 6-episode fit (-338.1, -40.8, 318.3) | sim03 = the
                           0.3-rad-step fit (-282.9, -35.4, 266.3). Yaw is never changed.
    ka, kb, kc    --       numeric overrides of a, b, c for roll and pitch, applied AFTER the
                           preset (any subset; e.g. att=sim,kb=-45)
    drag         0.0       linear drag kd [1/s] in the translational model:
                           acc -= diag(kd, kd, kdz) * vel (world frame). The simulator has
                           0.495 (xy) / 0.544 (z). 0 = off (vendored).
    dragz        = drag    separate z coefficient kdz (default: the same as drag)
    fgain        0.968..   thrust gain cmd_f_coef; fgain=1.0 sets it to 1.0 AND recomputes the
                           hover thrust (mass*g - acc_coef)/fgain — 0.4256 N instead of
                           0.4395 N. Default: the crazyflow parameter (0.968...).
    flag         0.0       a first-order thrust lag tau_f [s] as a 13th model state f_eff, with
                           f_eff' = (u_f - f_eff)/tau_f and the thrust map applied to f_eff.
                           The measured value is 0.057. The controller keeps its own f_eff
                           estimate (exact first-order response to its commands; rotors at rest
                           -> f_eff = 0 at reset). 0 = off.

The same model (att, drag, fgain, flag) is used in `_model_acc`, i.e. by the ESO / L1
estimators and by the mass adaptation, so an estimator never has to absorb what the model
already knows. Combining `mass=1` with `dist=eso|l1` double-counts the vertical residual
(both absorb it): use one or the other unless that is the point.

The two named specs of Lesson 8:

    M1        mpcdev:att=sim,drag=0.495,fgain=1.0           the measured model (§2's three
                                                            first-order corrections, with the
                                                            vendored weights, horizon and
                                                            Euler step)
    M1+mass   mpcdev:att=sim,drag=0.495,fgain=1.0,mass=1    the same plus the thrust-scale
                                                            adaptation for Level 1 (§5)

Quote a spec in your shell if it complains about the commas (docs/4-troubleshooting.md).

`control_freq` is NOT a spec key: the caller passes it (50 in the lsy race, 100 in the
crazyflow harness). It sets the estimators' integration step and the mass-adaptation step,
as in the vendored class; with dist=none and mass=0 it does not touch the commands.

Example:
    parse_spec("mpcdev:int=rk4,H=30,wp=3,tau=0.02,dist=eso,ramp=liftoff", control_freq=50)

Deviations from the vendored file (all inactive at the defaults):
  * the options and their plumbing (above);
  * `solve_ms` (per-call ipopt wall time) and `n_fail` (ipopt exceptions; the vendored
    fall-back to the last iterate is kept) are recorded, so the bridge, `race_eval.py` and
    `race_runner.py` can report what a solve costs;
  * the resolved options are printed once at construction (verbose=False silences it).
"""

from __future__ import annotations

import time

import numpy as np

from crazy_track.controllers.base import Controller
from crazy_track.controllers.mppi_l1 import _load_so_rpy_params
from crazy_track.controllers.utils import THRUST_MAX, THRUST_MIN
from crazy_track.trajectories import Trajectory

INTEGRATORS = ("euler", "rk4")
DISTURBANCES = ("none", "eso", "l1")
RAMPS = ("time", "liftoff", "none")
# roll/pitch (a, b, c) in ddtheta = a*theta + b*dtheta + c*u  (Lesson 8 §2)
ATT_PRESETS: dict[str, tuple[float, float, float] | None] = {
    "sorpy": None,                              # crazyflow's so_rpy (-189.0, -12.8, 138.1), untouched
    "sim": (-338.1, -40.8, 318.3),              # race-sim fit, 6-episode mean (RMSE 0.019 rad)
    "sim03": (-282.9, -35.4, 266.3),            # race-sim fit, 0.3 rad steps
}
# the simulator's linear drag (body frame ~ world frame at small tilt), for the callers
SIM_DRAG_XY, SIM_DRAG_Z = 0.495, 0.544


def _scaled(w: float, expr):
    """w * expr, but leave the CasADi graph untouched when w == 1 (the vendored cost adds the
    position term without a coefficient; keeping the graph identical keeps the default
    bit-exact)."""
    return expr if w == 1.0 else w * expr


class MPCDev(Controller):
    """Nonlinear receding-horizon MPC on crazyflow's so_rpy model, with the options listed in the
    module docstring. Defaults == crazy_track.controllers.mpc.MPCController."""

    def __init__(self, horizon: int = 20, dt_plan: float = 0.04, control_freq: int = 50,
                 integrator: str = "euler",
                 wp: float = 1.0, wv: float = 0.05, wu: float = 0.02, wf: float = 0.02,
                 wdu: float = 0.1, wt: float = 0.0, tau: float = 0.0,
                 acc_ff: bool | None = None, wa: float = 0.0,
                 disturbance: str = "none", eso_w: float = 7.0, l1_a_s: float = -5.0,
                 l1_cutoff_hz: float = 4.0, ramp: str = "time", ramp_t: float = 1.5,
                 liftoff_dz: float = 0.03,
                 mass_adapt: bool = False, mass_gain: float = 2.0, mass_min: float = 0.8,
                 mass_max: float = 1.25, mass_delay: float = 0.3,
                 rp_max: float = 1.0, yaw_max: float = 0.3, max_iter: int = 60, tol: float = 1e-4,
                 att: str = "sorpy", ka: float | None = None, kb: float | None = None,
                 kc: float | None = None, drag: float = 0.0, dragz: float | None = None,
                 fgain: float | None = None, flag: float = 0.0,
                 verbose: bool = True):
        import casadi as cs

        if integrator not in INTEGRATORS:
            raise ValueError(f"integrator must be one of {INTEGRATORS}, got {integrator!r}")
        if disturbance not in DISTURBANCES:
            raise ValueError(f"disturbance must be one of {DISTURBANCES}, got {disturbance!r}")
        if ramp not in RAMPS:
            raise ValueError(f"ramp must be one of {RAMPS}, got {ramp!r}")
        if int(horizon) < 1 or float(dt_plan) <= 0.0:
            raise ValueError("horizon must be >= 1 and dt_plan > 0")
        self.H, self.dtp = int(horizon), float(dt_plan)
        self.integrator = integrator
        self.disturbance = disturbance
        self.offset_free = disturbance != "none"   # kept for callers that read it (vendored name)
        self.dt = 1.0 / control_freq
        self.control_freq = int(control_freq)
        self.wp, self.wv, self.wu, self.wf, self.wdu, self.wt = (
            float(wp), float(wv), float(wu), float(wf), float(wdu), float(wt))
        self.tau = float(tau)
        self.wa = float(wa)
        self.acc_ff = (self.wa > 0.0) if acc_ff is None else bool(acc_ff)
        self.ramp, self.ramp_t, self.liftoff_dz = ramp, float(ramp_t), float(liftoff_dz)
        self.mass_adapt = bool(mass_adapt)
        self.mass_gain, self.mass_min, self.mass_max = float(mass_gain), float(mass_min), float(mass_max)
        self.mass_delay = float(mass_delay)
        self.rp_max, self.yaw_max = float(rp_max), float(yaw_max)
        self.max_iter, self.tol = int(max_iter), float(tol)
        # estimators, exactly as vendored
        self._w = float(eso_w)                    # ESO bandwidth for the disturbance estimate
        self._a_s = float(l1_a_s)
        self._alpha_f = 1.0 - np.exp(-2 * np.pi * float(l1_cutoff_hz) * self.dt)  # L1 LPF
        # ---- the model parameters: crazyflow's so_rpy, optionally replaced by the measured ones
        # (Lesson 8 §2). The dict entries keep the loader's types (0-d / 1-d float64 arrays) so the
        # CasADi graph is identical to the vendored one whenever nothing is overridden.
        if att not in ATT_PRESETS:
            raise ValueError(f"att must be one of {tuple(ATT_PRESETS)}, got {att!r}")
        p = dict(_load_so_rpy_params())
        self.att = att
        preset = ATT_PRESETS[att]
        if preset is not None or any(v is not None for v in (ka, kb, kc)):
            a, b, c = preset if preset is not None else (None, None, None)
            a = float(ka) if ka is not None else a
            b = float(kb) if kb is not None else b
            c = float(kc) if kc is not None else c
            for key, val in (("rpy_coef", a), ("rpy_rates_coef", b), ("cmd_rpy_coef", c)):
                if val is not None:
                    arr = np.array(p[key], dtype=np.float64).reshape(3).copy()
                    arr[0:2] = val                          # roll and pitch; yaw untouched
                    p[key] = arr
        self.att_abc = (float(np.asarray(p["rpy_coef"]).reshape(3)[0]),
                        float(np.asarray(p["rpy_rates_coef"]).reshape(3)[0]),
                        float(np.asarray(p["cmd_rpy_coef"]).reshape(3)[0]))
        if fgain is not None:
            if float(fgain) <= 0.0:
                raise ValueError("fgain must be > 0")
            p["cmd_f_coef"] = np.asarray(float(fgain), dtype=np.float64)   # same type as the loader's
        self.fgain = float(p["cmd_f_coef"])
        self.drag = float(drag)
        self.dragz = float(drag) if dragz is None else float(dragz)
        if self.drag < 0.0 or self.dragz < 0.0:
            raise ValueError("drag / dragz must be >= 0")
        self.kd = np.array([self.drag, self.drag, self.dragz], dtype=np.float64)
        self.use_drag = bool(np.any(self.kd != 0.0))
        self.flag = float(flag)
        if self.flag < 0.0:
            raise ValueError("flag (thrust lag tau_f) must be >= 0")
        self.use_flag = self.flag > 0.0
        self.p = p
        self.hover = float((p["mass"] * 9.81 - p["acc_coef"]) / p["cmd_f_coef"])
        self.nx = 13 if self.use_flag else 12

        # so_rpy Euler dynamics: x = [pos(3), rpy(3), vel(3), drpy(3) (, f_eff)], u = [rpy_cmd(3), f]
        x, u = cs.MX.sym("x", self.nx), cs.MX.sym("u", 4)
        rpy, vel, drpy = x[3:6], x[6:9], x[9:12]
        cr, sr = cs.cos(rpy[0]), cs.sin(rpy[0])
        cp, sp = cs.cos(rpy[1]), cs.sin(rpy[1])
        cy, sy = cs.cos(rpy[2]), cs.sin(rpy[2])
        z_axis = cs.vertcat(cr * sp * cy + sr * sy, cr * sp * sy - sr * cy, cr * cp)
        DIST = cs.MX.sym("dist", 3)
        KS = cs.MX.sym("ks")                      # thrust scale (mass adaptation), 1 = nominal
        f_cmd = x[12] if self.use_flag else u[3]  # effective thrust: lagged state or the command
        if self.mass_adapt:
            thrust = p["acc_coef"] + p["cmd_f_coef"] * KS * f_cmd
        else:
            thrust = p["acc_coef"] + p["cmd_f_coef"] * f_cmd
        acc = thrust * z_axis / p["mass"] + cs.DM(p["gravity_vec"]) + DIST
        if self.use_drag:
            acc = acc - cs.DM(self.kd) * vel
        ddrpy = (cs.DM(p["rpy_coef"]) * rpy + cs.DM(p["rpy_rates_coef"]) * drpy
                 + cs.DM(p["cmd_rpy_coef"]) * u[0:3])
        if self.use_flag:
            xdot = cs.vertcat(vel, drpy, acc, ddrpy, (u[3] - x[12]) / self.flag)
        else:
            xdot = cs.vertcat(vel, drpy, acc, ddrpy)
        ins = [x, u, DIST] + ([KS] if self.mass_adapt else [])
        if self.integrator == "euler":
            f = cs.Function("f", ins, [x + self.dtp * xdot])          # Euler (vendored)
        else:                                                          # classical RK4, u held over dtp
            xdot_fn = cs.Function("xdot", ins, [xdot])
            extra = [DIST] + ([KS] if self.mass_adapt else [])
            k1 = xdot
            k2 = xdot_fn(x + 0.5 * self.dtp * k1, u, *extra)
            k3 = xdot_fn(x + 0.5 * self.dtp * k2, u, *extra)
            k4 = xdot_fn(x + self.dtp * k3, u, *extra)
            f = cs.Function("f", ins, [x + (self.dtp / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)])
        acc_fn = cs.Function("acc", ins, [acc]) if self.acc_ff else None
        # the model's translational acceleration and its state derivative as stand-alone functions
        # (for checks / notes only: NOT part of the NLP graph, so the default stays bit-exact)
        self._acc_model = cs.Function("acc_model", ins, [acc])
        self._xdot_model = cs.Function("xdot_model", ins, [xdot])

        # Multiple shooting NLP
        opti = cs.Opti()
        X = opti.variable(self.nx, self.H + 1)
        U = opti.variable(4, self.H)
        X0 = opti.parameter(self.nx)
        REF = opti.parameter(3, self.H)
        REFV = opti.parameter(3, self.H)
        DISTP = opti.parameter(3)
        KSP = opti.parameter() if self.mass_adapt else None
        AREF = opti.parameter(3, self.H) if self.acc_ff else None
        extra_p = [DISTP] + ([KSP] if self.mass_adapt else [])
        cost = 0
        opti.subject_to(X[:, 0] == X0)
        for k in range(self.H):
            opti.subject_to(X[:, k + 1] == f(X[:, k], U[:, k], *extra_p))
            cost += _scaled(self.wp, cs.sumsqr(X[0:3, k + 1] - REF[:, k]))
            cost += self.wv * cs.sumsqr(X[6:9, k + 1] - REFV[:, k])
            cost += self.wu * cs.sumsqr(U[0:2, k]) + self.wf * cs.sumsqr(U[3, k] - self.hover)
            if k > 0:
                cost += self.wdu * cs.sumsqr(U[:, k] - U[:, k - 1])
            if self.acc_ff:
                cost += self.wa * cs.sumsqr(acc_fn(X[:, k], U[:, k], *extra_p) - AREF[:, k])
        if self.wt != 0.0:
            cost += self.wt * cs.sumsqr(X[0:3, self.H] - REF[:, self.H - 1])
        opti.subject_to(opti.bounded(-self.rp_max, U[0, :], self.rp_max))
        opti.subject_to(opti.bounded(-self.rp_max, U[1, :], self.rp_max))
        opti.subject_to(opti.bounded(-self.yaw_max, U[2, :], self.yaw_max))
        opti.subject_to(opti.bounded(THRUST_MIN, U[3, :], THRUST_MAX))
        opti.minimize(cost)
        opti.solver("ipopt", {"print_time": False, "ipopt.print_level": 0,
                              "ipopt.max_iter": self.max_iter, "ipopt.tol": self.tol,
                              "ipopt.warm_start_init_point": "yes"})
        self.opti, self.X, self.U, self.X0, self.REF, self.REFV = opti, X, U, X0, REF, REFV
        self.DISTP, self.KSP, self.AREF = DISTP, KSP, AREF
        self._prev = None
        self._traj: Trajectory | None = None
        self.solve_ms: list[float] = []
        self.n_fail = 0
        self._eso_reset()
        self._flight_reset()
        if verbose:
            print(f"[mpc_dev] MPCDev options: {self.describe()}")

    # ---- bookkeeping ---------------------------------------------------------
    def options(self) -> dict:
        return {"int": self.integrator, "H": self.H, "dtp": self.dtp, "freq": self.control_freq,
                "wp": self.wp, "wv": self.wv, "wu": self.wu, "wf": self.wf, "wdu": self.wdu,
                "wt": self.wt, "tau": self.tau, "accff": int(self.acc_ff), "wa": self.wa,
                "dist": self.disturbance, "eso_w": self._w, "l1_as": self._a_s,
                "l1_c": round(-np.log(1.0 - self._alpha_f) / (2 * np.pi * self.dt), 6),
                "ramp": self.ramp, "ramp_t": self.ramp_t, "liftoff_dz": self.liftoff_dz,
                "mass": int(self.mass_adapt), "mass_gain": self.mass_gain,
                "mass_min": self.mass_min, "mass_max": self.mass_max, "mass_delay": self.mass_delay,
                "rp_max": self.rp_max, "yaw_max": self.yaw_max, "iter": self.max_iter,
                "tol": self.tol,
                "att": self.att, "ka": self.att_abc[0], "kb": self.att_abc[1], "kc": self.att_abc[2],
                "drag": self.drag, "dragz": self.dragz, "fgain": round(self.fgain, 6),
                "hover": round(self.hover, 5), "flag": self.flag}

    def describe(self) -> str:
        return ",".join(f"{k}={v:g}" if isinstance(v, float) else f"{k}={v}"
                        for k, v in self.options().items())

    def _eso_reset(self) -> None:
        self._v_hat = np.zeros(3)
        self._sigma = np.zeros(3)      # ESO disturbance state / L1 raw estimate
        self._sigma_f = np.zeros(3)    # L1 low-passed estimate (what the MPC sees)
        self._last_thrust = self.hover

    def _flight_reset(self) -> None:
        self._z0 = None                # z at the first act() after reset
        self._airborne = False
        self._t_lift = None
        self._k = 1.0                  # thrust scale (mass adaptation)
        self._vel_prev = None
        self.k_trace: list[tuple[float, float]] = []
        self._f_eff = 0.0              # thrust-lag state (flag > 0): rotors at rest at reset

    def reset(self, trajectory: Trajectory) -> None:
        self._traj = trajectory
        self._prev = None
        self._eso_reset()
        self._flight_reset()
        self.solve_ms = []
        self.n_fail = 0

    # ---- model pieces shared by the estimators --------------------------------
    def _thrust_eff(self) -> float:
        """The thrust the model applies now: the lagged state (flag > 0) or the last command."""
        return self._f_eff if self.use_flag else self._last_thrust

    def _model_acc(self, quat: np.ndarray, vel: np.ndarray | None = None) -> np.ndarray:
        """Nominal translational acceleration of the model for the thrust it applies now -- the
        SAME map the MPC predicts with (thrust gain, thrust scale k, thrust lag, drag)."""
        from scipy.spatial.transform import Rotation as R

        p = self.p
        z_b = R.from_quat(quat).as_matrix()[:, 2]
        a = ((p["acc_coef"] + p["cmd_f_coef"] * self._k * self._thrust_eff()) * z_b
             / p["mass"] + p["gravity_vec"])
        if self.use_drag:
            if vel is None:
                raise ValueError("_model_acc needs the velocity when drag is modelled")
            a = a - self.kd * np.asarray(vel, dtype=np.float64)
        return a

    def _estimate(self, vel: np.ndarray, quat: np.ndarray) -> np.ndarray:
        """Disturbance-acceleration estimate for the prediction model (vendored, unchanged)."""
        if self.disturbance == "eso":
            e_v = vel - self._v_hat
            self._v_hat += self.dt * (self._model_acc(quat, vel) + self._sigma + 2 * self._w * e_v)
            self._sigma = np.clip(self._sigma + self.dt * self._w**2 * e_v, -3.0, 3.0)
            return self._sigma
        # "l1": piecewise-constant adaptation (DATT, arXiv:2310.09053)
        v_err = self._v_hat - vel
        self._v_hat += self.dt * (self._model_acc(quat, vel) + self._sigma + self._a_s * v_err)
        e = np.exp(self._a_s * self.dt)
        self._sigma = np.clip(-(self._a_s / (e - 1.0)) * e * (self._v_hat - vel), -3.0, 3.0)
        self._sigma_f = (1 - self._alpha_f) * self._sigma_f + self._alpha_f * self._sigma
        return self._sigma_f

    def _update_flight_phase(self, pos: np.ndarray, t: float) -> None:
        if self._z0 is None:
            self._z0 = float(pos[2])
            if self._z0 > 0.5:                       # started in the air (hover-start toml)
                self._airborne, self._t_lift = True, float(t)
        if not self._airborne and pos[2] > self._z0 + self.liftoff_dz:
            self._airborne, self._t_lift = True, float(t)

    def _ramp_factor(self, t: float) -> float:
        if self.ramp == "time":
            return min(1.0, t / self.ramp_t)          # vendored: min(1, t / 1.5)
        if self.ramp == "none":
            return 1.0
        if not self._airborne:                        # liftoff
            return 0.0
        if self.ramp_t <= 0.0:
            return 1.0
        return min(1.0, (t - self._t_lift) / self.ramp_t)

    def _adapt_mass(self, vel: np.ndarray, quat: np.ndarray, t: float) -> None:
        """First-order estimate of a thrust scale k on the vertical residual:
        k += dt * gain * (a_z,measured - a_z,model) / a_z,thrust, clipped. a_z,measured is the
        finite difference of the measured velocity over the last control step, a_z,model the
        so_rpy prediction for the thrust commanded over that step (with the current k)."""
        from scipy.spatial.transform import Rotation as R

        if (self._vel_prev is not None and self._airborne
                and t >= self._t_lift + self.mass_delay):
            a_meas = (vel - self._vel_prev) / self.dt
            a_model = self._model_acc(quat, self._vel_prev)   # over the step just measured
            z_b = R.from_quat(quat).as_matrix()[:, 2]
            p = self.p
            a_thrust_z = float((p["acc_coef"] + p["cmd_f_coef"] * self._k * self._thrust_eff())
                               * z_b[2] / p["mass"])
            if a_thrust_z > 3.0:                     # not inverted / not free-falling
                r = float(a_meas[2] - a_model[2])
                self._k = float(np.clip(self._k + self.dt * self.mass_gain * r / a_thrust_z,
                                        self.mass_min, self.mass_max))
                self.k_trace.append((float(t), self._k))
        self._vel_prev = np.array(vel, dtype=np.float64, copy=True)

    # ---- the control step ------------------------------------------------------
    def act(self, state: np.ndarray, t: float) -> np.ndarray:
        from scipy.spatial.transform import Rotation as R

        pos, vel, quat, omega = state[:3], state[3:6], state[6:10], state[10:13]
        rpy = R.from_quat(quat).as_euler("xyz")
        cr, sr = np.cos(rpy[0]), np.sin(rpy[0])
        cp, tp = np.cos(rpy[1]), np.tan(rpy[1])
        E_inv = np.array([[1, sr * tp, cr * tp], [0, cr, -sr], [0, sr / cp, cr / cp]])
        x0 = np.concatenate([pos, rpy, vel, E_inv @ omega])
        if self.use_flag:
            x0 = np.concatenate([x0, [self._f_eff]])

        if self.ramp == "liftoff" or self.mass_adapt:
            self._update_flight_phase(pos, t)
        if self.mass_adapt:
            self._adapt_mass(vel, quat, t)
            self.opti.set_value(self.KSP, self._k)

        if self.disturbance != "none":
            if self.ramp == "liftoff" and not self._airborne:
                # on the ground the floor reaction is not a disturbance to learn: hold the
                # estimate at zero and keep the velocity state synchronised (no phantom
                # innovation at lift-off)
                self._v_hat = np.array(vel, dtype=np.float64, copy=True)
                self._sigma[:] = 0.0
                self._sigma_f[:] = 0.0
                sigma = np.zeros(3)
            else:
                sigma = self._estimate(vel, quat)
            self.opti.set_value(self.DISTP, sigma * self._ramp_factor(t))
        else:
            self.opti.set_value(self.DISTP, np.zeros(3))

        times = t + self.tau + self.dtp * np.arange(1, self.H + 1)
        self.opti.set_value(self.X0, x0)
        self.opti.set_value(self.REF, self._traj.pos(times).T)
        self.opti.set_value(self.REFV, self._traj.vel(times).T)
        if self.acc_ff:
            self.opti.set_value(self.AREF, self._traj.acc(times).T)
        if self._prev is not None:
            Xp, Up = self._prev
            self.opti.set_initial(self.X, Xp)
            self.opti.set_initial(self.U, Up)
        t0 = time.perf_counter()
        try:
            sol = self.opti.solve()
            Xs, Us = sol.value(self.X), sol.value(self.U)
        except RuntimeError:  # ipopt failed: fall back to last iterate (vendored behaviour)
            self.n_fail += 1
            Xs = self.opti.debug.value(self.X)
            Us = self.opti.debug.value(self.U)
        self.solve_ms.append(1e3 * (time.perf_counter() - t0))
        # warm start next solve with shifted solution
        Xp = np.hstack([Xs[:, 1:], Xs[:, -1:]])
        Up = np.hstack([Us[:, 1:], Us[:, -1:]])
        self._prev = (Xp, Up)
        u0 = Us[:, 0]
        self._last_thrust = float(np.clip(u0[3], THRUST_MIN, THRUST_MAX))
        if self.use_flag:
            # exact first-order response of the lag state to the command held over one control step
            self._f_eff += (1.0 - np.exp(-self.dt / self.flag)) * (self._last_thrust - self._f_eff)
        return np.array([u0[0], u0[1], u0[2], self._last_thrust])


# ---- spec strings -------------------------------------------------------------------
def _bool(v: str) -> bool:
    s = v.strip().lower()
    if s in ("1", "true", "yes", "on"):
        return True
    if s in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"expected a boolean (0/1), got {v!r}")


SPEC_KEYS: dict[str, tuple[str, type]] = {
    "int": ("integrator", str), "H": ("horizon", int), "dtp": ("dt_plan", float),
    "wp": ("wp", float), "wv": ("wv", float), "wu": ("wu", float), "wf": ("wf", float),
    "wdu": ("wdu", float), "wt": ("wt", float), "tau": ("tau", float),
    "wa": ("wa", float), "accff": ("acc_ff", _bool),
    "dist": ("disturbance", str), "eso_w": ("eso_w", float), "l1_as": ("l1_a_s", float),
    "l1_c": ("l1_cutoff_hz", float), "ramp": ("ramp", str), "ramp_t": ("ramp_t", float),
    "liftoff_dz": ("liftoff_dz", float),
    "mass": ("mass_adapt", _bool), "mass_gain": ("mass_gain", float),
    "mass_min": ("mass_min", float), "mass_max": ("mass_max", float),
    "mass_delay": ("mass_delay", float),
    "rp_max": ("rp_max", float), "yaw_max": ("yaw_max", float),
    "iter": ("max_iter", int), "tol": ("tol", float),
    # the measured model (Lesson 8 §2)
    "att": ("att", str), "ka": ("ka", float), "kb": ("kb", float), "kc": ("kc", float),
    "drag": ("drag", float), "dragz": ("dragz", float), "fgain": ("fgain", float),
    "flag": ("flag", float),
}


def parse_spec_kwargs(spec: str) -> dict:
    """'mpcdev' -> {}, 'mpcdev:H=30,int=rk4' -> {'horizon': 30, 'integrator': 'rk4'}."""
    if spec == "mpcdev":
        return {}
    if not spec.startswith("mpcdev:"):
        raise ValueError(f"not an mpcdev spec: {spec!r} (expected 'mpcdev' or 'mpcdev:key=val,...')")
    kw: dict = {}
    for item in spec[len("mpcdev:"):].split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"mpcdev spec item {item!r} is not key=val")
        key, val = (s.strip() for s in item.split("=", 1))
        if key not in SPEC_KEYS:
            raise ValueError(f"unknown mpcdev key {key!r}; known keys: {', '.join(SPEC_KEYS)}")
        name, conv = SPEC_KEYS[key]
        if name in kw:
            raise ValueError(f"mpcdev key {key!r} given twice")
        try:
            kw[name] = conv(val)
        except ValueError as e:
            raise ValueError(f"mpcdev key {key!r}: bad value {val!r} ({e})") from None
    return kw


def parse_spec(spec: str, control_freq: int = 50, verbose: bool = True) -> MPCDev:
    """Build an MPCDev from a spec string. control_freq comes from the caller (50 in the race,
    100 in the crazyflow harness), never from the spec."""
    return MPCDev(control_freq=control_freq, verbose=verbose, **parse_spec_kwargs(spec))


def is_mpcdev_spec(spec: str) -> bool:
    return spec == "mpcdev" or spec.startswith("mpcdev:")
