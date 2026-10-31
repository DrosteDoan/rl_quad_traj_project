"""Shared definitions for re-analysing the study sweep, following the paper's rules."""
import glob, os, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
_L9 = os.environ.get("LESSON9_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
ROOT = os.path.join(_L9, "results", "eval", "study_sweep")
TRACKDIR = os.path.join(_L9, "tracks", "study")
MPC = ["M1", "M1+ESO", "M1+L1", "M1+mass", "mppi_l1"]
FA = ["fa0", "fa1", "fa2"]
S7 = ["s0", "s1", "s2"]
CONDS = ["wind_const", "payload", "mass_mult", "wind_gust", "lighthouse", "combined"]
STOCH = {"wind_gust", "lighthouse", "combined"}
COARSE = [round(0.1 * i, 1) for i in range(11)]
TRACKS = sorted(int(os.path.basename(p).split("_")[1]) for p in glob.glob(f"{ROOT}/track_*"))

_cache = None


def load():
    global _cache
    if _cache is None:
        fs = glob.glob(f"{ROOT}/track_*/*/laps.csv")
        d = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
        d["lam"] = d["lam"].round(2)
        _cache = d
    return _cache


def comp_table(cond):
    """completion rate per (track, member, lam)."""
    d = load()
    d = d[d.cond == cond]
    return d.groupby(["track", "member", "lam"])["completed"].mean()


def viable_members(cond, track, members):
    ct = comp_table(cond)
    out = []
    for m in members:
        try:
            v = ct.loc[(track, m, 0.0)]
        except KeyError:
            continue
        if v >= 0.5:
            out.append(m)
    return out


def curve(cond, track, members, how="max", lams=COARSE):
    """family line over lam: how in max/median/min, over the given members."""
    ct = comp_table(cond)
    vals = []
    for lam in lams:
        xs = []
        for m in members:
            try:
                xs.append(ct.loc[(track, m, lam)])
            except KeyError:
                pass
        if not xs:
            vals.append(np.nan)
        else:
            vals.append({"max": np.max, "median": np.median, "min": np.min}[how](xs))
    return pd.Series(vals, index=lams)


def lam_star(delta_series, u=0.8, delta=0.2):
    lams = [l for l in delta_series.index if l <= u + 1e-9]
    for i, l in enumerate(lams):
        if delta_series[l] > delta and all(delta_series[l2] >= -delta for l2 in lams[i + 1:]):
            return l
    return None


def entering_tracks(cond, rl_members):
    out = []
    for t in TRACKS:
        if viable_members(cond, t, MPC) and viable_members(cond, t, rl_members):
            out.append(t)
    return out


def family_lines(cond, track, rl_members, rl_how="max", mpc_how="max", filt=True, lams=COARSE):
    mpc_m = viable_members(cond, track, MPC) if filt else MPC
    rl_m = viable_members(cond, track, rl_members) if filt else rl_members
    if mpc_how == "median":
        mpc_m = MPC if not filt else mpc_m
    if rl_how == "median":
        rl_m = rl_members  # median over all three seeds, per the paper
    return curve(cond, track, mpc_m, mpc_how, lams), curve(cond, track, rl_m, rl_how, lams)
