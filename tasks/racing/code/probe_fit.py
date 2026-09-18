"""Fit the identification probes: what the MPC's model got wrong — Lesson 8 §2.

    python tasks/racing/code/probe_fit.py tasks/racing/figures/probes

Runs in the MAIN venv (numpy + scipy) over the CSVs `race_probe.py` wrote — one per probe
episode, named `<probe>_<tag>_ep<NN>.csv`. It

  * fits an ARX(2,1) model to every open-loop attitude step and converts it to the MPC's own
    convention ddtheta = a*theta + b*dtheta + c*u, so the fit can be read against so_rpy's
    (-189.0, -12.8, 138.1) parameter for parameter (c/(-a) is the dc gain, -b/(2*sqrt(-a)) the
    damping);
  * replays the so_rpy model — continuous, and forward-Euler at the MPC's 40 ms prediction
    step — on the MEASURED command sequence, and reports each one's RMSE against what the
    simulator actually did. This is the number that says whether the error is the parameters
    or the discretisation;
  * regresses the airborne part of the climbs for the thrust gain and the linear drag
    (az = a0(F) - kd*vz), and reads the lift-off times and the rotor spin-up from the floor;
  * integrates crazyflow's rotor ODE from rest, for the spin-up times a hold has to respect.

Outputs, in the probe folder (or --out): `probe_fit.csv` (one row per episode with every
metric), `attitude_models.csv` (the step response of each model) and `summary.txt` (everything
printed below). The last block printed is the comparison table of Lesson 8 §2 — the race
simulator, the model the MPC predicts with, and the MPC's forward Euler of it at 40 ms.

Caveats to write next to these numbers: the fit describes THIS simulator (crazyflow's Mellinger
loop and rotor model), not a Crazyflie; the ARX model covers the first 0.5 s of a step (the slow
tail from 0.94 to 1.00 is outside it); yaw is not probed (the race reference has no yaw); and
the drag quoted below as `crazyflow` is the simulator's own `drag_matrix` parameter, quoted, not
fitted — the fitted value is the one from the climbs.
"""

from __future__ import annotations

import argparse
import glob
import os
import re

import numpy as np
from scipy.linalg import expm

G = 9.81
MASS = 0.04338
CMD_F_COEF = 0.96836458
# so_rpy identified attitude model (roll = pitch): dd = a*th + b*dth + c*u
A_RP, B_RP, C_RP = -188.991, -12.7803, 138.0834
# crazyflow first_principles drag (body frame, N per m/s) -> 1/s
KD_XY = 0.02149163 / MASS
KD_Z = 0.02359736 / MASS
RPM2THRUST = np.array([0.0, -3.133427287299859e-7, 4.407354891648379e-10])
ROTOR_DYN = np.array([13.996001897562685, 0.00011093207920685363, 5.933168530682111,
                      0.00031951312393561264])
DT = 0.02


# ----------------------------------------------------------------------------- io
def load(path):
    with open(path) as fh:
        first = fh.readline().strip()
    meta = dict(re.findall(r"(\w+)=(\[[^\]]*\]|\S+)", first))
    d = np.genfromtxt(path, delimiter=",", names=True, skip_header=1)
    return meta, d


def fnum(meta, key, default=np.nan):
    try:
        return float(meta[key])
    except (KeyError, ValueError):
        return default


def cdiff(y, dt=DT):
    """Central difference (forward/backward at the ends)."""
    dy = np.empty_like(y)
    dy[1:-1] = (y[2:] - y[:-2]) / (2 * dt)
    dy[0] = (y[1] - y[0]) / dt
    dy[-1] = (y[-1] - y[-2]) / dt
    return dy


def interp_time(t, y, level):
    """First t at which y crosses `level` upwards (linear interpolation), or nan."""
    idx = np.where((y[:-1] < level) & (y[1:] >= level))[0]
    if len(idx) == 0:
        return np.nan
    i = idx[0]
    return t[i] + (level - y[i]) / (y[i + 1] - y[i]) * (t[i + 1] - t[i])


# ----------------------------------------------------------------------------- rotor model
def rotor_spinup(F_cmd, dt=1e-4, T=0.6):
    """crazyflow first_principles rotor ODE from rest, per-motor force F_cmd/4:
    returns t, total thrust(t)."""
    c1, c2 = RPM2THRUST[1], RPM2THRUST[2]
    f_m = F_cmd / 4
    w_cmd = (-c1 + np.sqrt(c1**2 + 4 * c2 * f_m)) / (2 * c2)
    n = int(T / dt)
    w = 0.0
    ts, Fs = np.zeros(n), np.zeros(n)
    for i in range(n):
        ts[i], Fs[i] = i * dt, 4 * (c1 * w + c2 * w**2)
        if w_cmd > w:
            dw = ROTOR_DYN[0] * (w_cmd - w) + ROTOR_DYN[1] * (w_cmd**2 - w**2)
        else:
            dw = ROTOR_DYN[2] * (w_cmd - w) + ROTOR_DYN[3] * (w_cmd**2 - w**2)
        w += dt * dw
    return ts, Fs


# ----------------------------------------------------------------------------- attitude models
def exact_zoh(a, b, c, dt):
    A = np.array([[0.0, 1.0], [a, b]])
    B = np.array([[0.0], [c]])
    Ad = expm(A * dt)
    Bd = np.linalg.solve(A, (Ad - np.eye(2)) @ B)
    return Ad, Bd


def euler_zoh(a, b, c, dt):
    A = np.array([[0.0, 1.0], [a, b]])
    B = np.array([[0.0], [c]])
    return np.eye(2) + dt * A, dt * B


def rk4_zoh(a, b, c, dt):
    A = np.array([[0.0, 1.0], [a, b]])
    B = np.array([[0.0], [c]])
    I = np.eye(2)
    # x' = A x + B u, u const: x_{k+1} = Phi x_k + Gamma u
    Phi = I + dt * A + dt**2 / 2 * A @ A + dt**3 / 6 * A @ A @ A + dt**4 / 24 * A @ A @ A @ A
    Gam = (dt * I + dt**2 / 2 * A + dt**3 / 6 * A @ A + dt**4 / 24 * A @ A @ A) @ B
    return Phi, Gam


def simulate(Ad, Bd, u, x0):
    x = np.array(x0, dtype=float).reshape(2)
    out = np.zeros((len(u), 2))
    for k in range(len(u)):
        out[k] = x
        x = Ad @ x + (Bd[:, 0] * u[k])
    return out


def step_metrics(t, th, A_step, t_on):
    """Rise time 10-90 %, overshoot, steady state of a step of size A_step commanded at t_on."""
    m = t >= t_on
    tt, y = t[m] - t_on, th[m]
    base = th[(t < t_on) & (t > t_on - 0.2)].mean() if np.any((t < t_on) & (t > t_on - 0.2)) else th[0]
    y = y - base
    ss = y[tt >= tt[-1] - 0.1].mean()
    if abs(ss) < 1e-9:
        return dict(gain=np.nan, rise=np.nan, overshoot=np.nan, ss=ss, peak=np.nan, t_peak=np.nan)
    s = np.sign(ss)
    t10 = interp_time(tt, s * y, 0.1 * abs(ss))
    t90 = interp_time(tt, s * y, 0.9 * abs(ss))
    pk = s * (s * y).max()
    return dict(gain=ss / A_step, rise=t90 - t10, overshoot=(abs(pk) - abs(ss)) / abs(ss),
                ss=ss, peak=pk, t_peak=tt[np.argmax(s * y)])


def model_step_table(models, A_step=0.3, T=1.5):
    """Clean step-response metrics of each (Ad, Bd, dt) model + equivalent continuous poles."""
    rows = []
    for name, (Ad, Bd, dt) in models.items():
        n = int(T / dt)
        u = np.full(n, A_step)
        x = simulate(Ad, Bd, u, [0, 0])
        t = np.arange(n) * dt
        # fine interpolation for the rise time on the coarse models
        tf = np.arange(0, T, 1e-3)
        thf = np.interp(tf, t, x[:, 0])
        mt = step_metrics(np.concatenate([[-0.1], tf]), np.concatenate([[0.0], thf]), A_step, 0.0)
        ev = np.linalg.eigvals(Ad)
        s = np.log(ev.astype(complex)) / dt
        wn = float(np.sqrt(abs(s[0] * s[1])))
        zeta = float(-np.real(s[0] + s[1]) / (2 * wn)) if wn > 0 else np.nan
        dc = float((np.linalg.solve(np.eye(2) - Ad, Bd))[0, 0])
        rows.append(dict(model=name, dt=dt, dc_gain=dc, wn=wn, zeta=zeta, pole_mag=abs(ev[0]),
                         s1=float(np.real(s[0])), s2=float(np.real(s[1])),
                         s_imag=float(abs(np.imag(s[0]))),
                         rise=mt["rise"], overshoot=mt["overshoot"], t_peak=mt["t_peak"],
                         settle_2pct=_settle(tf, thf, 0.02)))
    return rows


def _settle(t, y, frac):
    ss = y[-50:].mean()
    if abs(ss) < 1e-9:
        return np.nan
    out = np.abs(y - ss) > frac * abs(ss)
    idx = np.where(out)[0]
    return t[idx[-1]] if len(idx) else 0.0


# ----------------------------------------------------------------------------- per-probe
def analyse_climb(meta, d, rows, name):
    t, z, vz = d["t"], d["z"], d["vz"]
    F = fnum(meta, "thrust")
    az = cdiff(vz)
    zf = z.min()
    # leaving the floor: first sample after the initial drop with z > z_floor + 2 mm and vz > 0.02
    i_floor = np.argmin(z)
    leave = np.where((z > zf + 0.002) & (vz > 0.02) & (np.arange(len(z)) > i_floor))[0]
    t_leave = t[leave[0]] if len(leave) else np.nan
    t_lift = interp_time(t, z, 0.03)
    t07, t10 = interp_time(t, z, 0.7), interp_time(t, z, 1.0)
    res = dict(file=name, probe="climb", thrust=F, z0=z[0], z_floor=zf, t_floor=t[i_floor],
               t_leave=t_leave, t_lift_z003=t_lift, t_z07=t07, t_z10=t10, t_end=t[-1], z_end=z[-1],
               vz_end=vz[-1], end=meta.get("end", ""), tilt_cmd=fnum(meta, "tilt", 0.0),
               pitch_max=d["pitch"].max(), roll_absmax=np.abs(d["roll"]).max(),
               yaw_absmax=np.abs(d["yaw"]).max(), x_end=d["x"][-1], vx_end=d["vx"][-1])
    if np.isfinite(t_leave):
        k = int(np.argmin(np.abs(t - t_leave)))
        res.update(pitch_at_leave=d["pitch"][k], pitch_at_030=np.interp(0.30, t, d["pitch"]),
                   vx_at_030=np.interp(0.30, t, d["vx"]), z_at_030=np.interp(0.30, t, z))
        ax = cdiff(d["vx"])
        m_air = (t > t_leave + 0.15) & (t < t[-1] - 0.02)
        if m_air.sum() >= 3:
            res.update(ax_air_mean=ax[m_air].mean(), pitch_air_mean=d["pitch"][m_air].mean())
    if np.isfinite(t_leave):
        # airborne, rotors spun up: fit az = a0 - kd*vz on t > t_leave + 0.15
        m = (t > t_leave + 0.15) & (t < t[-1] - 0.02)
        if m.sum() >= 5:
            X = np.stack([np.ones(m.sum()), -vz[m]], 1)
            coef, *_ = np.linalg.lstsq(X, az[m], rcond=None)
            res.update(a0_fit=coef[0], kd_fit=coef[1], az_mean=az[m].mean(),
                       gain_thrust_fit=(coef[0] + G) * MASS / F,
                       az_model_gain=CMD_F_COEF * F / MASS - G, az_model_nogain=F / MASS - G)
            # drag-corrected effective thrust and its 90 % time
            f_eff = MASS * (az + G + KD_Z * vz)
            m2 = t >= t_leave
            i90 = np.where((f_eff >= 0.9 * F) & m2)[0]
            res.update(t_f90=t[i90[0]] if len(i90) else np.nan,
                       f_eff_leave=f_eff[leave[0]], f_eff_end=f_eff[-5:].mean())
    rows.append(res)
    return res


def analyse_hold(meta, d, rows, name):
    t, z, vz = d["t"], d["z"], d["vz"]
    F = fnum(meta, "thrust")
    rows.append(dict(file=name, probe="hold", thrust=F, z0=z[0], z_min=z.min(), z_max=z.max(),
                     z_end=z[-1], vz_max=vz.max(), t_end=t[-1], end=meta.get("end", ""),
                     lifted=bool(z.max() > 0.03)))


def _as_sets(th, u, mask):
    """Accept one (th, u, mask) or a list of them (a pooled fit over separate windows)."""
    if isinstance(th, (list, tuple)):
        return list(zip(th, u, mask))
    return [(th, u, mask)]


def fit_second_order(t, th, u, mask):
    """LS fit dd = a*th + b*dth + c*u on the masked samples (central differences).
    th/u/mask may be lists (one entry per window; rows are pooled, signals never mixed)."""
    Xs, ys = [], []
    for th_i, u_i, m_i in _as_sets(th, u, mask):
        dth, ddth = cdiff(th_i), cdiff(cdiff(th_i))
        Xs.append(np.stack([th_i[m_i], dth[m_i], u_i[m_i]], 1))
        ys.append(ddth[m_i])
    X, y = np.vstack(Xs), np.concatenate(ys)
    coef, res, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ coef
    r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return coef, r2


def fit_arx(th, u, mask, d_max=3):
    """th[k+1] = p1*th[k] + p2*th[k-1] + q*u[k-d]; returns the best delay d and its fit.
    th/u/mask may be lists (pooled over windows)."""
    sets = _as_sets(th, u, mask)
    best = None
    for dly in range(d_max + 1):
        Xs, ys = [], []
        for th_i, u_i, m_i in sets:
            ks = np.where(m_i)[0]
            ks = ks[(ks >= 1 + dly) & (ks + 1 < len(th_i))]
            Xs.append(np.stack([th_i[ks], th_i[ks - 1], u_i[ks - dly]], 1))
            ys.append(th_i[ks + 1])
        X, y = np.vstack(Xs), np.concatenate(ys)
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        rmse = np.sqrt(((X @ coef - y) ** 2).mean())
        if best is None or rmse < best[1] * 0.98:      # need a clear 2 % improvement to add delay
            best = (dly, rmse, coef)
    dly, rmse, (p1, p2, q) = best
    # equivalent continuous poles (two real ones, or a complex pair)
    ev = np.roots([1, -p1, -p2]).astype(complex)
    s = np.log(ev) / DT
    s_sorted = sorted(s, key=lambda v: -np.real(v))      # slow pole first
    wn = float(np.sqrt(abs(s[0] * s[1])))
    zeta = float(-np.real(s[0] + s[1]) / (2 * wn)) if wn > 0 else np.nan
    dc = q / (1 - p1 - p2)
    real = bool(np.all(np.abs(np.imag(ev)) < 1e-9) and np.all(np.real(ev) > 0))
    # continuous a, b, c of dd = a*th + b*dth + c*u with the same poles and dc gain:
    # characteristic polynomial s^2 - b s - a = (s - s1)(s - s2) -> a = -s1 s2, b = s1 + s2
    a_c = -float(np.real(s[0] * s[1]))
    b_c = float(np.real(s[0] + s[1]))
    return dict(arx_delay=dly, arx_rmse=rmse, arx_wn=wn, arx_zeta=zeta, arx_dc=dc,
                arx_real_poles=real, arx_s_slow=float(np.real(s_sorted[0])),
                arx_s_fast=float(np.real(s_sorted[1])), arx_a=a_c, arx_b=b_c, arx_c=-a_c * dc)


def response_delay(t, th, u, t_on, thresh=0.01):
    """Samples after the command row at t_on until |th - base| > thresh (1 = the next sample)."""
    k_on = int(np.argmin(np.abs(t - t_on)))
    base = th[max(0, k_on - 10):k_on].mean()
    for j in range(1, 8):
        if k_on + j < len(th) and abs(th[k_on + j] - base) > thresh:
            return j
    return np.nan


def overlay_models(t, th, u, mask, fitted):
    """RMSE of each model vs the sim on the masked window, same input, same initial state."""
    k0 = np.where(mask)[0][0] - 1
    dth = cdiff(th)
    x0 = [th[k0], dth[k0]]
    tt, uu, yy = t[k0:], u[k0:], th[k0:]
    mm = mask[k0:]
    out = {}
    models = {
        "so_rpy_exact_20ms": exact_zoh(A_RP, B_RP, C_RP, DT) + (DT,),
        "so_rpy_euler_20ms": euler_zoh(A_RP, B_RP, C_RP, DT) + (DT,),
        "so_rpy_euler_40ms": euler_zoh(A_RP, B_RP, C_RP, 2 * DT) + (2 * DT,),
        "so_rpy_rk4_40ms": rk4_zoh(A_RP, B_RP, C_RP, 2 * DT) + (2 * DT,),
        "fitted_exact_20ms": exact_zoh(*fitted, DT) + (DT,),
        "fitted_euler_40ms": euler_zoh(*fitted, 2 * DT) + (2 * DT,),
        "fitted_rk4_40ms": rk4_zoh(*fitted, 2 * DT) + (2 * DT,),
    }
    for name, (Ad, Bd, dt) in models.items():
        step = int(round(dt / DT))
        us = uu[::step]
        x = simulate(Ad, Bd, us, x0)
        ts = tt[::step]
        ms = mm[::step]
        out[name] = float(np.sqrt(((x[ms, 0] - yy[::step][ms]) ** 2).mean()))
        out[name + "_maxerr"] = float(np.abs(x[ms, 0] - yy[::step][ms]).max())
    return out


def analyse_hoverstep(meta, d, rows, name, tables):
    t = d["t"]
    t0, A, Ts = fnum(meta, "t0"), fnum(meta, "step"), fnum(meta, "step_t")
    t1 = t0 + 3 * Ts + 2.0
    res = dict(file=name, probe="hoverstep", A=A, Ts=Ts, f_hold=fnum(meta, "f_hold"))
    # state at t0: was the drone settled?
    m_pre = (t > t0 - 0.2) & (t < t0)
    res.update(z_pre=d["z"][m_pre].mean(), vz_pre=d["vz"][m_pre].mean(),
               pos_err_pre=np.hypot(d["x"][m_pre].mean() + 2.0, d["y"][m_pre].mean() - 0.75))
    fits = {}
    for axis, th_key, u_key, ton, w_key in (("roll", "roll", "cmd_roll", t0, "wx"),
                                            ("pitch", "pitch", "cmd_pitch", t1, "wy")):
        th, u = d[th_key], d[u_key]
        mask = (t >= ton) & (t < ton + 3 * Ts)
        # first step (0 -> +A)
        seg = (t >= ton - 0.2) & (t < ton + Ts)
        mt = step_metrics(t[seg], th[seg], A, ton)
        res.update({f"{axis}_gain": mt["gain"], f"{axis}_rise": mt["rise"],
                    f"{axis}_overshoot": mt["overshoot"], f"{axis}_ss": mt["ss"],
                    f"{axis}_t_peak": mt["t_peak"],
                    f"{axis}_delay_samples_001rad": response_delay(t, th, u, ton, 0.01),
                    f"{axis}_delay_samples_0003rad": response_delay(t, th, u, ton, 0.003)})
        # second step (+A -> -A): gain from the settled values
        seg2 = (t >= ton + Ts - 0.1) & (t < ton + Ts)
        seg3 = (t >= ton + 2 * Ts - 0.1) & (t < ton + 2 * Ts)
        res[f"{axis}_gain_step2"] = (th[seg3].mean() - th[seg2].mean()) / (-2 * A)
        coef, r2 = fit_second_order(t, th, u, mask)
        res.update({f"{axis}_fit_a": coef[0], f"{axis}_fit_b": coef[1], f"{axis}_fit_c": coef[2],
                    f"{axis}_fit_r2": r2, f"{axis}_fit_dc": -coef[2] / coef[0],
                    f"{axis}_fit_wn": np.sqrt(max(-coef[0], 0)),
                    f"{axis}_fit_zeta": -coef[1] / (2 * np.sqrt(max(-coef[0], 1e-9)))})
        res.update({f"{axis}_{k}": v for k, v in fit_arx(th, u, mask).items()})
        # ang_vel frame check: body rate vs Euler-angle derivative on the window
        dth = cdiff(th)
        res[f"{axis}_angvel_vs_deuler_rmse"] = float(np.sqrt(((d[w_key][mask] - dth[mask]) ** 2).mean()))
        res[f"{axis}_deuler_rms"] = float(np.sqrt((dth[mask] ** 2).mean()))
        fits[axis] = (coef, mask, th, u)
    # overlays with the roll+pitch pooled fits: (i) a continuous LS fit on central differences,
    # (ii) an ARX(2,1) pooled over both axes, converted to continuous a, b, c (the one we quote)
    th_all = [d["roll"], d["pitch"]]
    u_all = [d["cmd_roll"], d["cmd_pitch"]]
    mask_all = [fits["roll"][1], fits["pitch"][1]]
    coef_all, r2_all = fit_second_order(t, th_all, u_all, mask_all)
    res.update(pool_fit_a=coef_all[0], pool_fit_b=coef_all[1], pool_fit_c=coef_all[2],
               pool_fit_r2=r2_all, pool_fit_dc=-coef_all[2] / coef_all[0],
               pool_fit_wn=np.sqrt(max(-coef_all[0], 0)),
               pool_fit_zeta=-coef_all[1] / (2 * np.sqrt(max(-coef_all[0], 1e-9))))
    arx_all = fit_arx(th_all, u_all, mask_all)
    res.update({f"pool_{k}": v for k, v in arx_all.items()})
    coef_arx = (arx_all["arx_a"], arx_all["arx_b"], arx_all["arx_c"])
    for axis in ("roll", "pitch"):
        coef, mask, th, u = fits[axis]
        ov = overlay_models(t, th, u, mask, coef_arx)
        res.update({f"{axis}_rmse_{k}": v for k, v in ov.items()})
    rows.append(res)
    tables.setdefault("fits", []).append(tuple(coef_all))
    tables.setdefault("arx", []).append(coef_arx)
    return res


def analyse_thruststep(meta, d, rows, name):
    t, vz, z = d["t"], d["vz"], d["z"]
    t0, fs, fh = fnum(meta, "t0"), fnum(meta, "fstep"), fnum(meta, "f_hold")
    az = cdiff(vz)
    m_pre = (t > t0 - 0.3) & (t < t0)
    res = dict(file=name, probe="thruststep", f_hold=fh, f_step=fs, z_pre=z[m_pre].mean(),
               vz_pre=vz[m_pre].mean(), az_pre=az[m_pre].mean(),
               gain_hover=MASS * G / fh, f_hover_mg=MASS * G, f_hover_model=MASS * G / CMD_F_COEF)
    for lab, ta, tb, df in (("up", t0, t0 + 0.3, fs - fh), ("down", t0 + 0.3, t0 + 0.6,
                                                            max(0.0855, 2 * fh - fs) - fs)):
        m = (t >= ta) & (t < tb)
        m_late = (t >= tb - 0.15) & (t < tb)
        # reference level: before t0 for the up-step, the up-step's plateau for the down-step
        m_ref = m_pre if lab == "up" else ((t >= ta - 0.1) & (t < ta))
        raw = az[m_late].mean() - az[m_ref].mean()
        corr = (az[m_late] + KD_Z * vz[m_late]).mean() - (az[m_ref] + KD_Z * vz[m_ref]).mean()
        model = CMD_F_COEF * df / MASS
        res.update({f"{lab}_daz_raw": raw, f"{lab}_daz_dragcorr": corr, f"{lab}_daz_model": model,
                    f"{lab}_daz_model_nogain": df / MASS, f"{lab}_gain_raw": raw * MASS / df,
                    f"{lab}_gain_dragcorr": corr * MASS / df})
        # first-order lag fit on the drag-corrected acceleration:
        # a(t) = a0 + da * (1 - e^{-(t - ta - td)/tau})
        tt = t[m] - ta
        y = az[m] + KD_Z * vz[m]
        m_prev = (t >= ta - 0.1) & (t < ta)
        a_prev = ((az[m_pre] + KD_Z * vz[m_pre]).mean() if lab == "up"
                  else (az[m_prev] + KD_Z * vz[m_prev]).mean())
        best = None
        for td in np.arange(0.0, 0.081, 0.01):
            for tau in np.arange(0.005, 0.2, 0.005):
                basis = np.clip(1 - np.exp(-(tt - td) / tau), 0, None) * (tt >= td)
                X = np.stack([np.ones_like(tt), basis], 1)
                c, *_ = np.linalg.lstsq(X, y, rcond=None)
                e = np.sqrt(((X @ c - y) ** 2).mean())
                if best is None or e < best[0]:
                    best = (e, td, tau, c[1], c[0])
        res.update({f"{lab}_lag_td": best[1], f"{lab}_lag_tau": best[2], f"{lab}_lag_da": best[3],
                    f"{lab}_lag_a0": best[4], f"{lab}_lag_rmse": best[0],
                    f"{lab}_a_prev": a_prev})
    rows.append(res)
    return res


def analyse_lateral(meta, d, rows, name):
    t, vx, vz = d["t"], d["vx"], d["vz"]
    t0, P, fh = fnum(meta, "t0"), fnum(meta, "pitch"), fnum(meta, "f_hold")
    Tl = 0.5
    ax, az = cdiff(vx), cdiff(vz)
    res = dict(file=name, probe="lateral", pitch_cmd=P, f_hold=fh, thrust_cmd=fh / np.cos(P))
    for lab, ta, tb, pc in (("plus", t0, t0 + Tl, P), ("minus", t0 + Tl, t0 + 2 * Tl, -P)):
        m_late = (t >= tb - 0.2) & (t < tb)
        m_all = (t >= ta) & (t < tb)
        pitch = d["pitch"][m_late].mean()
        axm = ax[m_late].mean()
        vxm = vx[m_late].mean()
        res.update({
            f"{lab}_pitch_meas": pitch, f"{lab}_pitch_gain": pitch / pc,
            f"{lab}_ax_meas": axm, f"{lab}_ax_dragcorr": axm + KD_XY * vxm, f"{lab}_vx_mean": vxm,
            f"{lab}_ax_gtan_073": G * np.tan(pc) * 0.7306, f"{lab}_ax_gtan": G * np.tan(pc),
            f"{lab}_ax_gtan_meas_pitch": G * np.tan(pitch),
            # with the commanded thrust f_hold/cos(P) and the achieved pitch, the model says:
            f"{lab}_ax_model_fullthrust": CMD_F_COEF * (fh / np.cos(P)) * np.sin(pitch) / MASS,
            f"{lab}_az_meas": az[m_late].mean(), f"{lab}_vz_end": vz[m_all][-1],
            f"{lab}_ax_peak": ax[m_all].max() if pc > 0 else ax[m_all].min(),
            f"{lab}_pitch_rise": step_metrics(t[(t >= ta - 0.2) & (t < tb)],
                                              d["pitch"][(t >= ta - 0.2) & (t < tb)],
                                              pc if lab == "plus" else -2 * P, ta)["rise"],
        })
    rows.append(res)
    return res


# ----------------------------------------------------------------------------- the Lesson-8 table
def _avg(rows, probe, *keys):
    """Mean of the first of `keys` that has finite values, over the episodes of one probe."""
    for key in keys:
        vals = [r[key] for r in rows if r.get("probe") == probe
                and isinstance(r.get(key), (int, float, np.floating)) and np.isfinite(r.get(key))]
        if vals:
            return float(np.mean(vals))
    return float("nan")


def _f(v, p=3):
    return "n/a" if v is None or not np.isfinite(v) else f"{v:.{p}f}"


def comparison_table(rows, models, arx_mean, climb) -> list[str]:
    """The table of Lesson 8 §2: the race simulator, so_rpy, and the MPC's Euler-40 ms of it."""
    by = {m["model"]: m for m in models}
    sim = by.get("sim_ARXfit_mean(exact)")
    sorpy = by.get("so_rpy_continuous(exact)")
    eul = by.get("so_rpy_euler_40ms(MPC)")
    L = ["", "## Lesson 8 §2 -- the race simulator vs the model the MPC predicts with", "",
         f"{'quantity':42s} {'race sim (measured)':>24s} {'so_rpy':>16s} {'Euler 40 ms':>14s}"]

    def row(label, a, b, c):
        L.append(f"{label:42s} {a:>24s} {b:>16s} {c:>14s}")

    g_meas = _avg(rows, "hoverstep", "roll_gain")
    row("attitude step 0.3 rad: dc gain",
        _f(g_meas, 2) + (f"  (model {_f(sim['dc_gain'], 2)})" if sim else ""),
        _f(sorpy["dc_gain"], 2) if sorpy else "n/a",
        _f(eul["dc_gain"], 2) if eul else "n/a")
    if sim and sorpy and eul:
        row("poles (1/s)", f"{sim['s1']:.1f}, {sim['s2']:.1f}",
            f"{sorpy['s1']:.1f}+-{sorpy['s_imag']:.1f}j", f"{eul['s1']:.1f}+-{eul['s_imag']:.1f}j")
        row("omega_n (rad/s) / zeta", f"{sim['wn']:.1f} / {sim['zeta']:.2f}",
            f"{sorpy['wn']:.1f} / {sorpy['zeta']:.2f}", f"{eul['wn']:.1f} / {eul['zeta']:.2f}")
    row("rise 10-90 % (s)", _f(_avg(rows, "hoverstep", "roll_rise")),
        _f(sorpy["rise"]) if sorpy else "n/a", _f(eul["rise"]) if eul else "n/a")
    row("overshoot", _f(_avg(rows, "hoverstep", "roll_overshoot"), 2),
        _f(sorpy["overshoot"], 2) if sorpy else "n/a", _f(eul["overshoot"], 2) if eul else "n/a")
    row("open-loop RMSE on the commands (rad)",
        _f(_avg(rows, "hoverstep", "roll_rmse_fitted_exact_20ms")),
        _f(_avg(rows, "hoverstep", "roll_rmse_so_rpy_exact_20ms")),
        _f(_avg(rows, "hoverstep", "roll_rmse_so_rpy_euler_40ms")))
    row("  ... its max error (rad)",
        _f(_avg(rows, "hoverstep", "roll_rmse_fitted_exact_20ms_maxerr")),
        _f(_avg(rows, "hoverstep", "roll_rmse_so_rpy_exact_20ms_maxerr")),
        _f(_avg(rows, "hoverstep", "roll_rmse_so_rpy_euler_40ms_maxerr")))
    f_hold = _avg(rows, "thruststep", "f_hold")
    row("thrust gain (hover N; m*g = %.4f)" % (MASS * G),
        (f"{_f(_avg(rows, 'thruststep', 'gain_hover'), 2)}  (hover {_f(f_hold, 4)} N)"
         if np.isfinite(f_hold) else _f(climb.get("gain_pooled", np.nan), 2)),
        f"{CMD_F_COEF:.3f}  (hover {MASS * G / CMD_F_COEF:.4f} N)", "")
    row("linear drag (1/s)",
        f"{_f(climb.get('kd_pooled', np.nan), 3)} fitted z (crazyflow {KD_XY:.3f} xy / {KD_Z:.3f} z)",
        "none", "")
    lag = _avg(rows, "thruststep", "up_lag_tau")
    dly = _avg(rows, "hoverstep", "roll_delay_samples_001rad")
    row("thrust lag / pure delay",
        f"tau {_f(lag, 3)} s / <= {_f(dly, 0)} sample(s)", "none", "")
    row("spin-up from rest: leaves the floor",
        f"{_f(_avg(rows, 'climb', 't_leave'))} s  (z 0.7 m at {_f(_avg(rows, 'climb', 't_z07'))}, "
        f"1.0 m at {_f(_avg(rows, 'climb', 't_z10'))})", "", "")
    if arx_mean is not None:
        # Which preset does THIS fit match? `att=sim` is the six-episode mean over the
        # 0.3 AND 0.15 rad steps; `att=sim03` is the 0.3 rad set alone. A student who ran only
        # the 0.3 rad step fits sim03, and telling them to fly `sim` would be telling them to
        # fly parameters they never measured (Lesson 8 §2).
        presets = {"sim": (-338.1, -40.8, 318.3), "sim03": (-282.9, -35.4, 266.3)}
        dist = {k: sum(abs((arx_mean[i] - v[i]) / v[i]) for i in range(3))
                for k, v in presets.items()}
        best = min(dist, key=dist.get)
        L += ["", f"In the MPC's own convention  dd(theta) = a*theta + b*d(theta) + c*u  the fitted "
                  f"roll/pitch parameters are",
              f"  ({arx_mean[0]:.1f}, {arx_mean[1]:.1f}, {arx_mean[2]:.1f})   against so_rpy's "
              f"({A_RP:.1f}, {B_RP:.1f}, {C_RP:.1f}).",
              "  c/(-a) is the dc gain and -b/(2*sqrt(-a)) the damping.",
              f"  Nearest mpc_dev preset: att={best}  (mean relative distance "
              f"{dist[best] / 3:.1%}; the other, att={'sim03' if best == 'sim' else 'sim'}, "
              f"is {dist['sim03' if best == 'sim' else 'sim'] / 3:.1%} away).",
              f"    att=sim   = (-338.1, -40.8, 318.3), the six-episode mean over the 0.3 AND "
              f"0.15 rad steps",
              f"    att=sim03 = (-282.9, -35.4, 266.3), the 0.3 rad steps alone",
              f"  To fly the preset:   mpcdev:att={best},drag=0.495,fgain=1.0   (Lesson 8 §4)",
              f"  To fly YOUR numbers: mpcdev:att={best},ka={arx_mean[0]:.1f},kb={arx_mean[1]:.1f},"
              f"kc={arx_mean[2]:.1f},drag=0.495,fgain=1.0"]
    return L


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("logs", nargs="?", default="tasks/racing/figures/probes",
                    help="the folder race_probe.py wrote its CSVs into")
    ap.add_argument("--tag", default=None, help="only files whose tag matches (e.g. rep|f08)")
    ap.add_argument("--out", default=None, help="where to write the outputs (default: the log folder)")
    args = ap.parse_args()
    out = args.out or args.logs
    os.makedirs(out, exist_ok=True)
    files = sorted(glob.glob(os.path.join(args.logs, "*_ep*.csv")))
    if not files:
        raise SystemExit(f"no probe CSVs (<probe>_<tag>_ep<NN>.csv) in {args.logs}\n"
                         "   run race_probe.py first -- Lesson 8 §2")
    rows, tables = [], {}
    for f in files:
        name = os.path.basename(f)
        probe, tag = name.split("_")[0], name.split("_")[1]
        if args.tag and not re.fullmatch(args.tag, tag):
            continue
        meta, d = load(f)
        try:
            if probe == "climb":
                analyse_climb(meta, d, rows, name)
            elif probe == "hold":
                analyse_hold(meta, d, rows, name)
            elif probe == "hoverstep":
                analyse_hoverstep(meta, d, rows, name, tables)
            elif probe == "thruststep":
                analyse_thruststep(meta, d, rows, name)
            elif probe == "lateral":
                analyse_lateral(meta, d, rows, name)
        except Exception as e:  # keep going, report
            rows.append(dict(file=name, probe=probe, error=repr(e)))
    # the long table: one row per episode
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(os.path.join(out, "probe_fit.csv"), "w") as fh:
        fh.write(",".join(keys) + "\n")
        for r in rows:
            fh.write(",".join(_fmt(r.get(k, "")) for k in keys) + "\n")
    # summary: mean / std per probe+tag group of every numeric metric
    lines = []
    groups = {}
    for r in rows:
        g = r["file"].rsplit("_ep", 1)[0]
        groups.setdefault(g, []).append(r)
    for g, rs in groups.items():
        lines.append(f"\n## {g}  (n = {len(rs)})")
        for k in keys:
            vals = [r[k] for r in rs if isinstance(r.get(k), (int, float, np.floating, np.integer, bool))
                    and not isinstance(r.get(k), str)]
            if not vals or k in ("file",):
                continue
            v = np.asarray(vals, dtype=float)
            if np.all(np.isnan(v)):
                continue
            lines.append(f"{k:36s} mean {np.nanmean(v):10.4f}  std {np.nanstd(v):8.4f}  "
                         f"min {np.nanmin(v):9.4f}  max {np.nanmax(v):9.4f}")
        for r in rs:
            if "error" in r:
                lines.append(f"ERROR {r['file']}: {r['error']}")
    # attitude model table (so_rpy continuous / Euler / RK4 / the pooled fits)
    fits = tables.get("fits", [])
    models = {
        "so_rpy_continuous(exact)": exact_zoh(A_RP, B_RP, C_RP, 1e-3) + (1e-3,),
        "so_rpy_euler_20ms": euler_zoh(A_RP, B_RP, C_RP, DT) + (DT,),
        "so_rpy_euler_40ms(MPC)": euler_zoh(A_RP, B_RP, C_RP, 2 * DT) + (2 * DT,),
        "so_rpy_rk4_40ms": rk4_zoh(A_RP, B_RP, C_RP, 2 * DT) + (2 * DT,),
    }
    if fits:
        fm = np.mean(np.asarray(fits), 0)
        models["sim_LSfit_mean(exact)"] = exact_zoh(*fm, 1e-3) + (1e-3,)
        lines.append(f"\npooled second-order LS fit (central differences), mean over {len(fits)} "
                     f"episodes: a {fm[0]:.2f} b {fm[1]:.2f} c {fm[2]:.2f} "
                     f"(so_rpy: {A_RP} {B_RP} {C_RP})")
    arx = tables.get("arx", [])
    arx_mean = None
    if arx:
        am = np.mean(np.asarray(arx), 0)
        arx_mean = tuple(float(v) for v in am)
        models["sim_ARXfit_mean(exact)"] = exact_zoh(*am, 1e-3) + (1e-3,)
        models["sim_ARXfit_euler_40ms"] = euler_zoh(*am, 2 * DT) + (2 * DT,)
        models["sim_ARXfit_rk4_40ms"] = rk4_zoh(*am, 2 * DT) + (2 * DT,)
        s = np.roots([1, -am[1], -am[0]])
        lines.append(f"pooled ARX(2,1) fit -> continuous, mean over {len(arx)} episodes: a {am[0]:.2f} "
                     f"b {am[1]:.2f} c {am[2]:.2f}; poles {np.round(s, 2)}; dc {-am[2] / am[0]:.3f}")
    # pooled climb regression: az = a0(F) - kd*vz over all climb episodes (a common kd)
    climb_summary: dict[str, float] = {}
    climb = [r for r in rows if r.get("probe") == "climb" and "a0_fit" in r
             and r.get("tilt_cmd", 0.0) == 0.0]          # level climbs only
    if climb:
        Xs, ys, Fs = [], [], sorted({r["thrust"] for r in climb})
        for r in climb:
            meta, d = load(os.path.join(args.logs, r["file"]))
            t, vz = d["t"], d["vz"]
            az = cdiff(vz)
            m = (t > r["t_leave"] + 0.15) & (t < t[-1] - 0.02)
            X = np.zeros((m.sum(), len(Fs) + 1))
            X[:, Fs.index(r["thrust"])] = 1.0
            X[:, -1] = -vz[m]
            Xs.append(X)
            ys.append(az[m])
        X, y = np.vstack(Xs), np.concatenate(ys)
        c, *_ = np.linalg.lstsq(X, y, rcond=None)
        climb_summary["kd_pooled"] = float(c[-1])
        lines.append(f"\n## pooled climb regression az = a0(F) - kd*vz over {len(climb)} episodes: "
                     f"kd {c[-1]:.3f} 1/s (the sim's drag_matrix z / m = {KD_Z:.3f})")
        gains = []
        for F, a0 in zip(Fs, c[:-1]):
            gain = (a0 + G) * MASS / F
            gains.append(gain)
            lines.append(f"  F {F:.3f} N: a0 {a0:.3f} m/s^2 -> thrust gain {gain:.4f} "
                         f"(so_rpy cmd_f_coef {CMD_F_COEF:.4f}: a0 {CMD_F_COEF * F / MASS - G:.3f}; "
                         f"gain 1: a0 {F / MASS - G:.3f})")
        climb_summary["gain_pooled"] = float(np.mean(gains))
        # the same with kd fixed at the simulator's own value
        for F in Fs:
            vals = []
            for r in climb:
                if r["thrust"] != F:
                    continue
                meta, d = load(os.path.join(args.logs, r["file"]))
                t, vz = d["t"], d["vz"]
                az = cdiff(vz)
                m = (t > r["t_leave"] + 0.15) & (t < t[-1] - 0.02)
                vals.append((az[m] + KD_Z * vz[m]).mean())
            lines.append(f"  F {F:.3f} N, kd fixed {KD_Z:.3f}: a0 {np.mean(vals):.3f} "
                         f"+- {np.std(vals):.3f} -> gain {(np.mean(vals) + G) * MASS / F:.4f}")
    mt = model_step_table(models)
    with open(os.path.join(out, "attitude_models.csv"), "w") as fh:
        ks = list(mt[0].keys())
        fh.write(",".join(ks) + "\n")
        for r in mt:
            fh.write(",".join(_fmt(r[k]) for k in ks) + "\n")
    lines.append("\n## attitude models: a clean 0.3 rad step (dc gain, wn, zeta, rise 10-90, "
                 "overshoot, peak time, 2 % settling)")
    for r in mt:
        lines.append(f"{r['model']:28s} dt {r['dt']:.3f}  dc {r['dc_gain']:.3f}  wn {r['wn']:6.2f}  "
                     f"zeta {r['zeta']:.3f}  poles {r['s1']:.1f}/{r['s2']:.1f}+-{r['s_imag']:.1f}j  "
                     f"|z| {r['pole_mag']:.3f}  rise {r['rise']:.3f}  "
                     f"os {r['overshoot']:.3f}  tpk {r['t_peak']:.3f}  ts2 {r['settle_2pct']:.3f}")
    # rotor spin-up model
    lines.append("\n## crazyflow's rotor ODE from rest (the model, no floor): time for the total "
                 "thrust to reach")
    for F in (0.8, 0.6, 0.5):
        ts, Fs_ = rotor_spinup(F)
        tmg = interp_time(ts, Fs_, MASS * G)
        t90 = interp_time(ts, Fs_, 0.9 * F)
        t99 = interp_time(ts, Fs_, 0.99 * F)
        lines.append(f"cmd {F:.2f} N: m*g ({MASS * G:.3f} N) at {tmg:.3f} s, 90 % at {t90:.3f} s, "
                     f"99 % at {t99:.3f} s")
    lines += comparison_table(rows, mt, arx_mean, climb_summary)
    lines.append(f"\nwrote {os.path.join(out, 'probe_fit.csv')}, "
                 f"{os.path.join(out, 'attitude_models.csv')}, {os.path.join(out, 'summary.txt')}")
    with open(os.path.join(out, "summary.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def _fmt(v):
    if isinstance(v, (float, np.floating)):
        return f"{v:.6g}"
    return str(v)


if __name__ == "__main__":
    main()
