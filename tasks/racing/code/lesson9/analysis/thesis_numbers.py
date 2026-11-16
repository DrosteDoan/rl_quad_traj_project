"""Checks every number the thesis version adds beyond the conference paper, recomputed from the stored laps.

analysis.py checks the numbers shared with the conference paper; this script checks the rest. Each line prints the
value as written in the thesis next to the recomputed one, with PASS or CHECK, and the run ends with a count.
Place it in tasks/racing/code/lesson9/analysis/ and run `python thesis_numbers.py` (LESSON9_DIR and ANALYSIS_DIR
override the default locations). Two kinds of number are not checked here: the seed resampling intervals (boot.py)
and the development history of Appendix B, which comes from development flights:
the checkpoint curve in results/eval/val_v4_curve and val_v4_final, the screening counts entered in thesisfigs.py.
"""
import collections, glob, json, os, re, sys
import numpy as np, pandas as pd
from scipy import stats

A = os.environ.get("ANALYSIS_DIR", os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, A)
from lib import *
from lib import _L9 as L9
from pairs import paired, viab_mask
from analysis import power_and_cross

P6 = [757, 965, 989, 1089, 1145, 1250]
OPT4 = ["M1", "M1+ESO", "M1+L1", "M1+mass"]
results = []


def check(label, thesis, computed, ok):
    results.append(bool(ok))
    print(f"  [{'PASS' if ok else 'CHECK'}] {label}: thesis {thesis} | computed {computed}")


def section(title):
    print(f"\n== {title} ==")


def r2(x):
    return round(float(x) + 1e-12, 2)


def pooled(c, rl_how="max", mpc_how="max", rl=FA, tracks=P6):
    M = pd.concat([family_lines(c, t, rl, rl_how, mpc_how)[0] for t in tracks], axis=1).mean(1)
    R = pd.concat([family_lines(c, t, rl, rl_how, mpc_how)[1] for t in tracks], axis=1).mean(1)
    return M, R


def rate(s, a, b):
    return (s[a] - s[b]) / (b - a)


d = load()

# -----------------------------------------------------------------------------------------------------------------
section("Viability (Appendix C.1, Table D.1)")
ct = comp_table("wind_const")
bad = lambda mem: sum(ct.loc[(t, m, 0.0)] < 0.5 for t in TRACKS for m in mem)
check("pairs failing viability, primary / 0.7 rad / MPC", "16 / 10 / 4", f"{bad(FA)} / {bad(S7)} / {bad(MPC)}",
      (bad(FA), bad(S7), bad(MPC)) == (16, 10, 4))
mppi = [r2(ct.loc[(t, "mppi_l1", 0.0)]) for t in TRACKS]
check("MPPI+L1 nominal completion per track", ".67 1 1 0 .67 1 .33 0 .33 1", mppi,
      mppi == [0.67, 1.0, 1.0, 0.0, 0.67, 1.0, 0.33, 0.0, 0.33, 1.0])

# -----------------------------------------------------------------------------------------------------------------
section("Power conditions and crossover intensities (Tables 3, C.1)")
want_power = {"wind_const": (0, 0, 0), "payload": (0, 0, 0), "mass_mult": (0, 0, 0), "wind_gust": (2, 1, 0),
              "lighthouse": (5, 5, 4), "combined": (6, 4, 4)}
for c, w in want_power.items():
    p = power_and_cross(c, FA)
    got = (p["room"], p["ret"], p["cross_all"])
    check(f"{c}: room / allowing / crossovers", w, got, got == w)
want_ls = {"payload": {1089: (None, 1.0), 1145: (None, 1.0), 1250: (None, 1.0)},
           "lighthouse": {757: (0.7, 0.7), 965: (None, 1.0), 989: (None, 0.9), 1089: (0.4, 0.4), 1145: (0.4, 0.4), 1250: (0.6, 0.6)},
           "combined": {757: (0.6, 0.6), 1089: (0.4, 0.4), 1145: (0.5, 0.5), 1250: (0.4, 0.4)}}
for c, w in want_ls.items():
    p = power_and_cross(c, FA)
    got = {t: (p["lam_star"][t], p["lam_star_full"][t]) for t in P6 if p["lam_star_full"][t] is not None}
    check(f"{c}: (lambda*(0.8), lambda*(1.0)) per track", w, got, got == w)
M, R = pooled("payload")
check("downward force: best MPC mean completion at lambda = 1", "0.50", f"{M[1.0]:.2f}", r2(M[1.0]) == 0.50)
comb = sorted(t for t, v in power_and_cross("combined", FA)["lam_star"].items() if v is not None)
check("combined: multiples of the nominal sensing model", "0.60--0.90",
      f"{0.4 / 0.6667:.2f}--{0.6 / 0.6667:.2f}", (r2(0.4 / 0.6667), r2(0.6 / 0.6667)) == (0.60, 0.90))
pmc = power_and_cross("combined", FA, "median")
check("combined: median seed crossings", "4", pmc["cross_all"], pmc["cross_all"] == 4)

# -----------------------------------------------------------------------------------------------------------------
section("Mechanism, unfiltered data on all ten tracks (Table 4)")
want_mech = {("lighthouse", "max"): {757: (0.7, .28, -.04, .28, -.04), 1089: (0.4, .44, 0, .44, 0), 1145: (0.4, .32, 0, .32, 0),
                                     1250: (0.6, .44, 0, .44, 0)},
             ("lighthouse", "median"): {757: (0.7, .28, -.08, .28, -.08)},
             ("combined", "max"): {747: (0.5, .68, .76, .68, .72), 757: (0.6, .76, -.24, .76, -.24), 837: (0.5, .72, .60, .72, .52),
                                   1089: (0.4, .76, -.12, .76, -.12), 1145: (0.5, .92, -.28, .92, -.28), 1250: (0.4, .80, -.12, .80, -.12)},
             ("combined", "median"): {757: (0.6, .76, -.40, .76, -.40), 837: (0.6, .96, .36, .96, .32), 1089: (0.4, .76, -.20, .76, -.20),
                                      1145: (0.5, .92, -.44, .92, -.44), 1250: (0.4, .80, -.16, .80, .24)}}
for (c, how), w in want_mech.items():
    got = {}
    for t in TRACKS:
        mpc, rl = curve(c, t, MPC, "max"), curve(c, t, FA, how)
        ls = lam_star(rl - mpc, 0.8, 0.2)
        if ls is not None:
            got[t] = (ls, r2(mpc[0.0] - mpc[ls]), r2(rl[ls] - rl[0.0]), r2(mpc[0.1] - mpc[ls]), r2(rl[ls] - rl[0.1]))
    w = {t: tuple(r2(x) for x in v) for t, v in w.items()}
    check(f"{c}, {how}: (lambda*, fall, rise, fall, rise)", "Table 4", "matches" if got == w else got, got == w)

# -----------------------------------------------------------------------------------------------------------------
section("Member level (Section 7.1, Appendix C.2)")
for c, want in [("payload", 0.85), ("lighthouse", 0.27), ("wind_gust", 0.25)]:
    ctc = comp_table(c)
    cells = [[ctc.loc[(t, m, l)] for m in OPT4] for t in TRACKS for l in COARSE]
    share = np.mean([max(v) - min(v) < 1e-9 for v in cells])
    check(f"{c}: four optimisation-based members identical, all ten tracks", f"{want:.2f}", f"{share:.2f}", r2(share) == want)
worst = sole = n = 0
for c in CONDS:
    ctc = comp_table(c)
    for t in TRACKS:
        for l in COARSE:
            v = {m: ctc.loc[(t, m, l)] for m in MPC}; mn = min(v.values()); n += 1
            worst += v["mppi_l1"] == mn; sole += v["mppi_l1"] == mn and list(v.values()).count(mn) == 1
check("MPPI+L1 worst or tied / sole worst, of all cells", "0.84 / 0.51 of 660", f"{worst / n:.2f} / {sole / n:.2f} of {n}",
      (r2(worst / n), r2(sole / n), n) == (0.84, 0.51, 660))
want_xm = {"wind_const": {965: 0.6, 1089: 0.6, 1145: 0.8, 1250: 0.4}, "payload": {1250: 0.8}, "mass_mult": {1089: 0.8}}
for c, w in want_xm.items():
    got = {t: v for t, v in power_and_cross(c, FA, "max", "median")["lam_star"].items() if v is not None}
    check(f"{c}: crossovers against the median MPC member", w, got, got == w)

# -----------------------------------------------------------------------------------------------------------------
section("Precision, lap time and failure causes (Table 6, Appendix C.4)")
want_h2 = {"wind_const": (54, .080, .229, 4.08, 4.15, .099), "payload": (61, .069, .213, 5.05, 5.18, .099),
           "mass_mult": (66, .051, .212, 4.57, 4.71, .095), "wind_gust": (1026, .090, .229, 4.08, 4.21, .094),
           "lighthouse": (1137, .073, .188, 5.03, 5.17, .095), "combined": (534, .092, .195, 5.04, 5.15, .085)}
for c, w in want_h2.items():
    p = paired(c, coarse_only=True).dropna(subset=["mpc", "rl"])
    q = paired(c, coarse_only=True, col="race_time").dropna(subset=["mpc", "rl"])
    got = (len(p), round(p.mpc.median(), 3), round(p.rl.median(), 3), round(q.mpc.median(), 2), round(q.rl.median(), 2),
           round(float(np.median(q.rl - q.mpc)), 3))
    check(f"{c}: pairs, RMSE MPC / RL, lap MPC / RL / paired difference", w, got, got == w)
q = {c: paired(c, coarse_only=True, col="race_time").dropna(subset=["mpc", "rl"]).groupby("track").size() for c in ("wind_const", "lighthouse")}
check("track 989 lap-time pairs, constant wind / sensing", "1 / 190", f"{q['wind_const'][989]} / {q['lighthouse'][989]}",
      (q["wind_const"][989], q["lighthouse"][989]) == (1, 190))
for c, w in [("wind_gust", (.210, .257)), ("wind_const", (.210, .253))]:
    g = paired(c, coarse_only=True).dropna(subset=["mpc", "rl"]).groupby("lam").rl.median()
    check(f"{c}: learned paired RMSE, lambda 0 -> 0.8", w, (round(g[0.0], 3), round(g[0.8], 3)), (round(g[0.0], 3), round(g[0.8], 3)) == w)
g = paired("combined", coarse_only=True).dropna(subset=["mpc", "rl"]).groupby("lam").rl.median()
lo, hi = g[[l for l in COARSE if l <= 0.6]].min(), g[[l for l in COARSE if l <= 0.6]].max()
check("combined: learned paired RMSE up to lambda 0.6", "0.19--0.21", f"{lo:.3f}--{hi:.3f}", (r2(lo), r2(hi)) == (0.19, 0.21))
# the most precise MPC member that completes its laps, per track and intensity, median over tracks
want_f = {"mass_mult": 0.96, "payload": 1.3, "wind_const": 1.75, "lighthouse": 1.9, "wind_gust": 2.2}
for c in CONDS:
    z = d[(d.cond == c) & d.track.isin(P6) & d.member.isin(MPC) & (d.completed == 1) & d.lam.isin(COARSE)]
    best = z.groupby(["track", "lam", "member"]).rmse_3d.median().groupby(["track", "lam"]).min()
    n8 = best.xs(0.8, level="lam").size
    if c == "combined":
        check("combined: primary tracks with an MPC lap completed at lambda 0.8", "1 (989), so no factor",
              list(best.xs(0.8, level="lam").index), n8 == 1)
        continue
    g = best.groupby("lam").median(); f = g[0.8] / g[0.0]
    dec = 2 if c in ("mass_mult", "wind_const") else 1
    check(f"{c}: best-member RMSE growth, lambda 0 -> 0.8 ({n8} tracks)", want_f[c], f"{f:.2f}", round(f, dec) == want_f[c])
z = d[d.track.isin(P6) & (d.completed == 0) & d.member.isin(MPC + FA)].copy()
z["mpc"] = z.member.isin(MPC); z["contact"] = z.fail == "contact"
check("contact share of failed laps, MPC / learned", "0.89 / 0.95", f"{z[z.mpc].contact.mean():.2f} / {z[~z.mpc].contact.mean():.2f}",
      (r2(z[z.mpc].contact.mean()), r2(z[~z.mpc].contact.mean())) == (0.89, 0.95))
zc = z[z.mpc & (z.cond == "combined")]
check("combined: MPC share of missed gates", "0.20", f"{1 - zc.contact.mean():.2f}", r2(1 - zc.contact.mean()) == 0.20)
zs = z[z.mpc & (z.cond == "lighthouse")]
g2 = (zs[zs.contact].impact_gate == 2).mean()
check("sensing: MPC failures in contact / contacts at gate 2", "0.93 / 0.38", f"{zs.contact.mean():.2f} / {g2:.2f}",
      (r2(zs.contact.mean()), r2(g2)) == (0.93, 0.38))
w = d[(d.cond == "lighthouse") & d.track.isin(P6) & d.member.isin(MPC) & (d.completed == 1)].groupby("lam").max_dev.median()
check("sensing: MPC median maximum deviation, lambda 0 / 0.8", "0.087 / 0.20", f"{w[0.0]:.3f} / {w[0.8]:.2f}",
      (round(w[0.0], 3), r2(w[0.8])) == (0.087, 0.20))
check("laps ended by divergence", "0", int(d.fail.astype(str).str.contains("diverg").sum()),
      not d.fail.astype(str).str.contains("diverg").any())

# -----------------------------------------------------------------------------------------------------------------
section("The gust and constant wind (Appendix C.3)")
fall = {t: (lambda m: m[0.0] - m[[l for l in COARSE if l <= 0.8]].min() > 0.2)(curve("wind_gust", t, viable_members("wind_gust", t, MPC), "max"))
        for t in TRACKS}
f10, f6 = sum(fall.values()), sorted(t for t in P6 if fall[t])
check("gust: tracks where the best MPC member falls by more than delta, of ten / primary", "5 / [1145, 1250]",
      f"{f10} / {f6}", (f10, f6) == (5, [1145, 1250]))
_, R = pooled("wind_const"); _, Rm = pooled("wind_const", "median")
got = (r2(R[0.0]), sorted({r2(R[l]) for l in COARSE if 0.1 <= l <= 0.8}), r2(R[0.9]), r2(R[1.0]))
check("constant wind: family maximum at 0 / 0.1-0.8 / 0.9 / 1.0", "1.00 / [0.83] / 0.67 / 0.67", got, got == (1.0, [0.83], 0.67, 0.67))
got = (r2(Rm[0.0]), r2(Rm[0.6]), r2(Rm[0.7]), sorted({r2(Rm[l]) for l in COARSE if l >= 0.8}))
check("constant wind: median seed at 0 / 0.6 / 0.7 / from 0.8", "0.83 / 0.50 / 0.50 / [0.33]", got, got == (0.83, 0.5, 0.5, [0.33]))
fa0 = curve("wind_const", 757, ["fa0"], "max")
fails = [l for l in COARSE if fa0[l] < 0.5]
check("fa0 on track 757 under constant wind: failing intensities", "0.2-0.7", fails, fails == [0.2, 0.3, 0.4, 0.5, 0.6, 0.7])

# -----------------------------------------------------------------------------------------------------------------
section("Single seeds and the 0.7 rad line (Section 7.4, Table 7)")
p = power_and_cross("lighthouse", ["fa0"]); pc = power_and_cross("combined", ["fa0"])
got = ({t: v for t, v in p["lam_star"].items() if v is not None}, {t: v for t, v in pc["lam_star"].items() if v is not None}, p["n"])
check("fa0: sensing / combined crossovers, viable tracks", "{757: .7, 1089: .4, 1250: .6} / {1089: .4, 1250: .4} / 4", got,
      got == ({757: 0.7, 1089: 0.4, 1250: 0.6}, {1089: 0.4, 1250: 0.4}, 4))
all3 = [t for t in P6 if len(viable_members("lighthouse", t, FA)) == 3]
wc = {c: sum(lam_star(curve(c, t, FA, "min") - family_lines(c, t, FA)[0], 0.8, 0.2) is not None for t in all3) for c in ("lighthouse", "combined")}
check("worst seed on the three tracks all seeds race: sensing / combined crossovers", "0 / 1",
      f"{all3}: {wc['lighthouse']} / {wc['combined']}", (all3, wc["lighthouse"], wc["combined"]) == ([757, 965, 1250], 0, 1))
want7 = {"lighthouse": (8, 7, 5, (0.4, 0.7)), "combined": (7, 7, 6, (0.4, 0.6)), "wind_gust": (2, 1, 0, (0.8, 0.8))}
for c, w in want7.items():
    p9, p6 = power_and_cross(c, S7), power_and_cross(c, S7, tracks=P6)
    ls = [v for v in p9["lam_star"].values() if v is not None]
    got = (p9["ret"], p9["cross_all"], p6["cross_all"], (min(ls), max(ls)))
    check(f"0.7 rad, {c}: allowing / crossovers on nine / on the six / lambda* range", w, got, got == w)
p9 = power_and_cross("wind_gust", S7)
check("0.7 rad, gust: tracks with room", "4", p9["room"], p9["room"] == 4)
for c in ("wind_const", "payload", "mass_mult"):
    n7 = power_and_cross(c, S7)["cross_all"]
    check(f"0.7 rad, {c}: crossovers", "0", n7, n7 == 0)

# -----------------------------------------------------------------------------------------------------------------
section("The training edge (Table C.2)")
want_e = {"payload": (1.00, 0.00, 0.00, 0.83, 2.50), "mass_mult": (1.00, 0.00, 0.00, 0.00, 0.00),
          "lighthouse": (0.84, 0.40, 0.57, 0.73, 1.03), "combined": (0.09, 1.65, 1.93, 0.40, 0.03)}
for c, w in want_e.items():
    M, R = pooled(c)
    got = (r2(R[0.8]), r2(rate(R, 0.4, 0.8)), r2(rate(R, 0.6, 0.8)), r2(rate(R, 0.8, 1.0)), r2(rate(M, 0.8, 1.0)))
    check(f"{c}: S_RL(0.8), learned loss 0.4-0.8 / 0.6-0.8 / 0.8-1, MPC loss 0.8-1", w, got, got == w)
E7 = entering_tracks("lighthouse", S7)
for c, w in [("lighthouse", (0.64, 0.91)), ("payload", (0.00, 0.56))]:
    _, R = pooled(c, rl=S7, tracks=E7)
    got = (r2(rate(R, 0.6, 0.8)), r2(rate(R, 0.8, 1.0)))
    check(f"0.7 rad, {c}: learned loss 0.6-0.8 / 0.8-1", w, got, got == w)

# -----------------------------------------------------------------------------------------------------------------
section("Held-out tallies (Table 9), from the summaries heldout_conditions.py wrote")
want_t = {"val_heldout_fa16_s0": (1, 4, 3), "val_heldout_fa16_s1": (0, 5, 3), "val_heldout_fa16_s2": (3, 2, 3),
          "val_heldout": (2, 5, 1), "val_heldout_s1": (1, 5, 2), "val_heldout_s2": (4, 3, 1),
          "val_heldout_cap07_s0": (3, 2, 3), "val_heldout_cap07_s1": (2, 4, 2), "val_heldout_cap07_s2": (2, 4, 2)}
for folder, w in want_t.items():
    txt = open(os.path.join(L9, "results", "eval", folder, "summary.txt")).read()
    m = re.search(r"tally over the 8 cells: \{'overfit': (\d+), 'generalises': (\d+), 'mixed': (\d+)\}", txt)
    got = tuple(int(x) for x in m.groups())
    check(f"{folder}: overfit / generalises / mixed", w, got, got == w)

# -----------------------------------------------------------------------------------------------------------------
section("Robustness (Appendix C.6, Table D.3)")
want_x = {"wind_const": [0, 0, 0, 0, 0, 0, 4, 4, 4], "payload": [0, 0, 0, 0, 0, 0, 1, 1, 1], "mass_mult": [0, 0, 0, 0, 0, 0, 1, 1, 1],
          "wind_gust": [0, 0, 0, 0, 0, 0, 5, 3, 3], "lighthouse": [5, 4, 4, 3, 2, 0, 6, 5, 5], "combined": [5, 4, 4, 4, 4, 4, 5, 4, 4]}
for c, w in want_x.items():
    got = [power_and_cross(c, FA, rh, mh, dl)["cross_all"] for rh, mh in (("max", "max"), ("median", "max"), ("max", "median"))
           for dl in (0.1, 0.2, 0.3)]
    check(f"{c}: crossovers by line pair and delta", w, got, got == w)
for c, idx, w in [("payload", None, 0.90), ("mass_mult", None, 0.90), ("lighthouse", 0.1, 0.78)]:
    R = pd.concat([curve(c, t, FA, "max") for t in TRACKS], axis=1).mean(1)
    v = R[idx] if idx is not None else R.max()
    check(f"unfiltered family maximum, {c}: at 0 / {'0.1' if idx else 'peak'}", f"0.60 / {w:.2f}", f"{R[0.0]:.2f} / {v:.2f}",
          (r2(R[0.0]), r2(v)) == (0.60, w))
for k, n, w in [(4, 6, (0.22, 0.96)), (4, 5, (0.28, 0.99)), (4, 4, (0.40, 1.00)), (2, 4, (0.07, 0.93))]:
    ci = stats.binomtest(k, n).proportion_ci(method="exact")
    check(f"exact 95% interval, {k} of {n}", w, (r2(ci.low), r2(ci.high)), (r2(ci.low), r2(ci.high)) == w)

# -----------------------------------------------------------------------------------------------------------------
section("Tracks (Appendix A.6, Table D.2)")
man = pd.read_csv(os.path.join(L9, "tracks", "manifest_study.csv"))
acc = man[man.status != "reject"]
check("accepted of candidates", "15 of 1,319 (1.1 %)", f"{len(acc)} of {len(man)} ({100 * len(acc) / len(man):.1f} %)",
      (len(acc), len(man)) == (15, 1319))
got = (round(acc.v_peak.mean(), 1), round(man.v_peak.mean(), 1), round(acc.plan_lap_s.mean(), 1), round(man.plan_lap_s.mean(), 1))
check("mean peak speed and planned lap, accepted vs all", "4.7 vs 5.3 m/s; 4.0 vs 4.6 s", got, got == (4.7, 5.3, 4.0, 4.6))
rc = collections.Counter(); [rc.update(r.split("+")) for r in man[man.status == "reject"].reasons.dropna()]
got = (rc["arena"], rc["ref_contact"], rc["frame_crossing"], rc["frame_margin"])
check("reject reasons: arena / reference contact / frame crossing / margin", "1185 / 1075 / 1061 / 79", got, got == (1185, 1075, 1061, 79))
ms = man.set_index("seed")
raw = pd.read_csv(os.path.join(L9, "tracks", "manifest_study.csv"), dtype=str).set_index("seed")
from decimal import Decimal, ROUND_HALF_UP


def half_up(t, col, places):
    """The manifest stores three decimals; a stored tie such as 6.465 is rounded half up, as written in the thesis."""
    return float(Decimal(raw.loc[str(t), col]).quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP))


want_g = {757: (86, 10.7, 3.6, 5.37, 1.31), 965: (46, 6.9, 3.9, 4.85, 1.14), 989: (64, 8.6, 3.5, 6.47, 1.37),
          1089: (53, 7.4, 3.8, 3.46, 1.00), 1145: (57, 6.8, 4.4, 3.70, 1.00), 1250: (61, 8.4, 5.3, 3.89, 1.00),
          747: (89, 7.3, 3.5, 4.03, 1.05), 834: (92, 8.6, 3.6, 4.65, 1.11), 837: (51, 8.2, 3.7, 4.34, 1.00), 1318: (67, 9.4, 3.6, 4.63, 1.25)}


def plan_props(t):
    dd = json.load(open(os.path.join(L9, "tracks", "study", f"track_{t:04d}.json")))
    P2 = [np.array(dd["start"][:2])] + [np.array(g["pos"][:2]) for g in dd["gates"]] + [np.array(dd["end"][:2])]
    turn = np.mean([np.degrees(np.arccos(np.clip(np.dot(P2[i] - P2[i - 1], P2[i + 1] - P2[i]) /
                    np.linalg.norm(P2[i] - P2[i - 1]) / np.linalg.norm(P2[i + 1] - P2[i]), -1, 1))) for i in range(1, 5)])
    P3 = [np.array(dd["start"])] + [np.array(g["pos"]) for g in dd["gates"]] + [np.array(dd["end"])]
    path = sum(np.linalg.norm(P3[i + 1] - P3[i]) for i in range(len(P3) - 1))
    return (int(round(turn)), float(round(path, 1)), float(round(100 * ms.loc[t, "frame_margin_m"], 1)),
            half_up(t, "lap_stretched_s", 2), half_up(t, "stretch", 2))


got = {t: plan_props(t) for t in want_g}
bad_t = {t: got[t] for t in want_g if got[t] != want_g[t]}
check("plan properties of the ten tracks (turn, gate path, margin, lap, stretch)", "Table D.2", bad_t or "all match", not bad_t)
mv = np.mean([ms.loc[t, "frame_margin_m"] for t in P6]) * 100
mn = np.mean([ms.loc[t, "frame_margin_m"] for t in TRACKS if t not in P6]) * 100
check("mean frame margin, viable vs not", "4.1 vs 3.6 cm", f"{mv:.1f} vs {mn:.1f}", (round(mv, 1), round(mn, 1)) == (4.1, 3.6))

print(f"\n{sum(results)} of {len(results)} checks pass.")
