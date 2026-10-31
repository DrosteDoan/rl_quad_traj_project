"""Reproduce every number reported in the ECC manuscript from the stored laps.

Run from anywhere:  python analysis.py [path/to/lesson9]
The lesson9 directory defaults to the parent of this file's directory (i.e. the script lives in
lesson9/analysis/), or to $LESSON9_DIR. Every check prints the manuscript value next to the computed
one and PASS or CHECK; nothing is written to disk.
"""
import glob, json, os, re, sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
L9 = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(os.environ.get("LESSON9_DIR", HERE.parent))
os.environ["LESSON9_DIR"] = str(L9)

import lib  # noqa: E402  (reads LESSON9_DIR)
from lib import (COARSE, CONDS, FA, MPC, S7, STOCH, comp_table, curve, entering_tracks,  # noqa: E402
                 family_lines, lam_star, load, viable_members)
from pairs import paired  # noqa: E402

PRIMARY = [757, 965, 989, 1089, 1145, 1250]
OPT4 = ["M1", "M1+ESO", "M1+L1", "M1+mass"]
NOMINAL_SENSING = 1 / 1.5          # lambda that reproduces the nominal sensing model (ceiling = 1.5x)
RESULTS = []


def check(label, paper, computed, ok):
    RESULTS.append(ok)
    print(f"  [{'PASS' if ok else 'CHECK'}] {label}: paper {paper} | computed {computed}")


def near(a, b, tol):
    return abs(float(a) - float(b)) <= tol


# ---------------------------------------------------------------------------------------------------------------
# Crossover rule (Problem Formulation): lambda*, room, retention
# ---------------------------------------------------------------------------------------------------------------
def power_and_cross(cond, rl_mem, rl_how="max", mpc_how="max", delta=0.2, tracks=None, u=0.8):
    """Room: best MPC below 1 - delta at some lambda <= u. Retention: learned line >= 0.5 at one of those
    lambdas. A crossover is lambda*(u) existing (first lambda with Delta > delta, never below -delta after)."""
    tracks = tracks if tracks is not None else entering_tracks(cond, rl_mem)
    lams_u = [l for l in COARSE if l <= u + 1e-9]
    out = dict(n=len(tracks), room=0, ret=0, cross_pow=0, cross_all=0, lam_star={}, lam_star_full={})
    for t in tracks:
        mpc, rl = family_lines(cond, t, rl_mem, rl_how, mpc_how)
        room_l = [l for l in lams_u if mpc[l] < 1 - delta]
        has_room = bool(room_l)
        has_ret = has_room and any(rl[l] >= 0.5 for l in room_l)
        ls, ls1 = lam_star(rl - mpc, u, delta), lam_star(rl - mpc, 1.0, delta)
        out["room"] += has_room; out["ret"] += has_ret
        out["cross_all"] += ls is not None; out["cross_pow"] += (ls is not None) and has_ret
        out["lam_star"][t], out["lam_star_full"][t] = ls, ls1
    return out


def section(title):
    print(f"\n== {title} ==")


def main():
    d = load()
    print(f"laps loaded: {len(d):,} (paper: 148,503)")
    check("total laps", "148,503", f"{len(d):,}", len(d) == 148503)

    # ---------------------------------------------------------------------------------------------------------
    section("Viability and sample")
    ent_fa, ent_s7 = entering_tracks("lighthouse", FA), entering_tracks("lighthouse", S7)
    check("primary tracks (full authority)", PRIMARY, ent_fa, ent_fa == PRIMARY)
    check("0.7 rad line viable tracks", 9, len(ent_s7), len(ent_s7) == 9)
    nonviable = [t for t in lib.TRACKS if t not in ent_fa]
    check("0.7 rad flies of the 4 non-viable", "3 (747, 837, 1318)", [t for t in nonviable if t in ent_s7],
          [t for t in nonviable if t in ent_s7] == [747, 837, 1318])

    # ---------------------------------------------------------------------------------------------------------
    section("Results 7.1: where a crossover can occur (primary line, delta = 0.2)")
    falls = []
    for c in ["wind_const", "payload", "mass_mult"]:
        for t in lib.TRACKS:
            m = curve(c, t, viable_members(c, t, MPC), "max")
            falls.append(m[0.0] - m[[l for l in COARSE if l <= 0.8]].min())
    check("covered channels, largest best-MPC fall for lambda <= 0.8, all ten tracks", "0.00", f"{max(falls):.2f}",
          max(falls) < 1e-9)
    p = {c: power_and_cross(c, FA) for c in CONDS}
    check("gust room / retention / crossovers", "2 / 1 / 0",
          f"{p['wind_gust']['room']} / {p['wind_gust']['ret']} / {p['wind_gust']['cross_all']}",
          (p['wind_gust']['room'], p['wind_gust']['ret'], p['wind_gust']['cross_all']) == (2, 1, 0))
    s = p["lighthouse"]
    check("sensing: tracks allowing a crossover, crossovers", "4 of 5", f"{s['cross_pow']} of {s['ret']}",
          (s['cross_pow'], s['ret']) == (4, 5))
    want = {1089: 0.4, 1145: 0.4, 1250: 0.6, 757: 0.7}
    got = {t: v for t, v in s["lam_star"].items() if v is not None}
    check("sensing lambda* per track", want, got, got == want)
    check("sensing above the edge (989, 965)", "0.9, 1.0", f"{s['lam_star_full'][989]}, {s['lam_star_full'][965]}",
          (s['lam_star_full'][989], s['lam_star_full'][965]) == (0.9, 1.0))
    rng = (min(got.values()) / NOMINAL_SENSING, max(got.values()) / NOMINAL_SENSING)
    check("sensing crossovers as multiples of the nominal model", "0.60--1.05", f"{rng[0]:.2f}--{rng[1]:.2f}",
          near(rng[0], 0.60, 0.005) and near(rng[1], 1.05, 0.005))
    check("MPC starts losing laps (lambda 0.3) as multiple", "0.45", f"{0.3 / NOMINAL_SENSING:.2f}",
          near(0.3 / NOMINAL_SENSING, 0.45, 0.005))
    cb = p["combined"]
    cb_t = sorted(t for t, v in cb["lam_star"].items() if v is not None)
    check("combined channel crossovers", "4 of 6 (757, 1089, 1145, 1250)", f"{cb['cross_all']} of {cb['n']} {cb_t}",
          cb["cross_all"] == 4 and cb_t == [757, 1089, 1145, 1250])

    # pooled family lines under sensing (Fig. 2 left)
    M = pd.concat([family_lines("lighthouse", t, FA)[0] for t in PRIMARY], axis=1).mean(1)
    R = pd.concat([family_lines("lighthouse", t, FA)[1] for t in PRIMARY], axis=1).mean(1)
    check("pooled best MPC, lambda 0.4 -> 0.8", "0.86 -> 0.47", f"{M[0.4]:.2f} -> {M[0.8]:.2f}",
          near(M[0.4], 0.86, .005) and near(M[0.8], 0.47, .005))
    check("pooled learned family at 0.4 / 0.6 / 0.8", "1.00 / 0.95 / 0.84", f"{R[0.4]:.2f} / {R[0.6]:.2f} / {R[0.8]:.2f}",
          near(R[0.4], 1, .005) and near(R[0.6], .95, .005) and near(R[0.8], .84, .005))

    # mechanism: unfiltered best-MPC fall and learned rise from lambda0 in {0, 0.1}
    fall_vals, rises = [], []
    for t, ls in got.items():
        mpc = curve("lighthouse", t, MPC, "max"); rl = curve("lighthouse", t, FA, "max")
        for l0 in (0.0, 0.1):
            fall_vals.append(mpc[l0] - mpc[ls]); rises.append(rl[ls] - rl[l0])
    check("sensing crossovers: MPC fall (unfiltered)", "0.28--0.44", f"{min(fall_vals):.2f}--{max(fall_vals):.2f}",
          near(min(fall_vals), .28, .005) and near(max(fall_vals), .44, .005))
    check("sensing crossovers: learned rise <= 0 from both baselines", "no rise", f"max rise {max(rises):+.2f}",
          max(rises) <= 1e-9)

    # ---------------------------------------------------------------------------------------------------------
    section("Table 2: member level (best or tied at lambda = 0.8; mean completion over lambda in [0.1, 0.8])")
    ct = {c: comp_table(c) for c in CONDS}
    table2_paper = {
        "wind_const": ("0 6 5 0 2", ".62 1.00 .98 .62 .62", 4), "payload": ("3 6 6 5 2", ".92 1.00 .96 .98 .50", 1),
        "mass_mult": ("6 3 5 6 1", "1.00 .85 .90 1.00 .46", 1), "wind_gust": ("0 0 6 0 0", ".65 .85 .96 .65 .36", 3),
        "lighthouse": ("2 1 0 3 2", ".78 .65 .65 .78 .38", 5), "combined": ("0 1 1 1 0", ".44 .38 .47 .51 .16", 4)}
    lam_win = [l for l in COARSE if 0.1 - 1e-9 <= l <= 0.8 + 1e-9]
    for c in CONDS:
        best, means = [], []
        for m in MPC:
            # best or tied for best; a track on which every member completes nothing counts for no member
            best.append(sum((max(ct[c].loc[(t, mm, 0.8)] for mm in MPC) > 0)
                            and ct[c].loc[(t, m, 0.8)] >= max(ct[c].loc[(t, mm, 0.8)] for mm in MPC) - 1e-12
                            for t in PRIMARY))
            means.append(np.mean([ct[c].loc[(t, m, l)] for t in PRIMARY for l in lam_win]))
        xmed = power_and_cross(c, FA, "max", "median")["cross_all"]
        b_s = " ".join(str(b) for b in best)
        # three decimals: several means are exact ties at the third decimal (0.625, 0.845, 0.955), which the
        # paper's two-decimal values round either way, so the means are compared within half a unit (0.005)
        m_s = " ".join(f"{v:.3f}" for v in means)
        pb, pm, px = table2_paper[c]
        means_ok = all(abs(v - float(pv)) <= 0.005 + 1e-9 for v, pv in zip(means, pm.split()))
        check(f"{c:10s} best-or-tied / means / x-median", f"{pb} | {pm} | {px}", f"{b_s} | {m_s} | {xmed}",
              b_s == pb and means_ok and xmed == px)
    xm = power_and_cross("wind_const", FA, "max", "median")["lam_star"]
    check("constant wind vs median member", "965, 1089 at 0.6; 1145 at 0.8; 1250 at 0.4",
          {t: v for t, v in xm.items() if v is not None},
          {t: v for t, v in xm.items() if v is not None} == {965: 0.6, 1089: 0.6, 1145: 0.8, 1250: 0.4})

    # ---------------------------------------------------------------------------------------------------------
    section("Results 7.2: precision (paired laps, coarse grid)")
    P = {c: paired(c, coarse_only=True).dropna(subset=["mpc", "rl"]).assign(cond=c) for c in CONDS}
    counts = {c: len(P[c]) for c in CONDS}
    allp = pd.concat(P.values(), ignore_index=True)
    check("paired laps per channel", "54 61 66 1026 1137 534 (2,878)",
          f"{' '.join(str(counts[c]) for c in CONDS)} ({len(allp):,})", len(allp) == 2878)
    ratio = allp.rl / allp.mpc
    check("median learned/MPC RMSE ratio over paired laps", "2.5", f"{ratio.median():.2f}", near(ratio.median(), 2.5, 0.05))
    rng_b = np.random.default_rng(0)
    tr = np.array(PRIMARY); boots = []
    by_t = {t: ratio[allp.track == t].to_numpy() for t in PRIMARY}
    for _ in range(4000):
        pick = rng_b.choice(tr, len(tr), replace=True)
        boots.append(np.median(np.concatenate([by_t[t] for t in pick])))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    check("track-resampled 95% interval (4,000)", "2.4--2.8", f"{lo:.2f}--{hi:.2f}",
          near(lo, 2.4, 0.05) and near(hi, 2.8, 0.05))
    f = (allp.rl / allp.mpc).groupby([allp.cond, allp.track]).median()   # median per-pair ratio per combination
    check("track x channel combinations with learned RMSE higher; factor range", "36 of 36; 1.9--6.1",
          f"{(f > 1).sum()} of {len(f)}; {f.min():.1f}--{f.max():.1f}",
          (f > 1).all() and len(f) == 36 and near(f.min(), 1.9, .05) and near(f.max(), 6.1, .05))
    check("share of paired laps where learned is more precise", "1.8%", f"{100 * (allp.rl < allp.mpc).mean():.1f}%",
          near(100 * (allp.rl < allp.mpc).mean(), 1.8, 0.05))

    # Wilcoxon signed-rank per stochastic channel-intensity cell with >= 10 pairs, Holm across cells
    cells = []
    for c in sorted(STOCH):
        for l, g in P[c].groupby("lam"):
            if len(g) >= 10:
                pval = stats.wilcoxon(g.rl, g.mpc).pvalue
                cells.append((c, l, len(g), pval, float(np.median(g.rl - g.mpc))))
    m = len(cells)
    order = np.argsort([x[3] for x in cells]); adj = np.empty(m); run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (m - rank) * cells[i][3])); adj[i] = run
    sig_mpc = sum(1 for i, x in enumerate(cells) if adj[i] < 0.05 and x[4] > 0)
    sig_rl = sum(1 for i, x in enumerate(cells) if adj[i] < 0.05 and x[4] < 0)
    per = {c: sum(1 for x in cells if x[0] == c) for c in sorted(STOCH)}
    check("Wilcoxon/Holm: tested cells (gust, sensing, combined)", "26 (10, 10, 6)",
          f"{m} ({per['wind_gust']}, {per['lighthouse']}, {per['combined']})", m == 26)
    check("cells favouring the MPC after Holm / favouring learned", "26 / 0", f"{sig_mpc} / {sig_rl}",
          sig_mpc == 26 and sig_rl == 0)

    # lap time on paired laps
    PT = {c: paired(c, coarse_only=True, col="race_time").dropna(subset=["mpc", "rl"]) for c in CONDS}
    diffs = {c: float(np.median(PT[c].rl - PT[c].mpc)) for c in CONDS}
    check("median paired lap-time difference per channel", "0.09--0.10 s (table: +0.085 to +0.099)",
          " ".join(f"{diffs[c]:+.3f}" for c in CONDS), all(0.08 <= v <= 0.10 for v in diffs.values()))

    # sensing RMSE trend (Fig. 2 right)
    sm = P["lighthouse"].groupby("lam")[["mpc", "rl"]].median()
    check("sensing paired RMSE, MPC lambda 0 -> 0.8", "0.052 -> 0.103", f"{sm.mpc[0.0]:.3f} -> {sm.mpc[0.8]:.3f}",
          near(sm.mpc[0.0], .052, .0005) and near(sm.mpc[0.8], .103, .0005))
    check("sensing paired RMSE, learned lambda 0 -> 0.8", "0.210 -> 0.178", f"{sm.rl[0.0]:.3f} -> {sm.rl[0.8]:.3f}",
          near(sm.rl[0.0], .210, .0005) and near(sm.rl[0.8], .178, .0005))
    npl = P["lighthouse"].groupby("lam").size()
    check("Fig. 2 pairs per intensity", "6 at 0, 52--150 elsewhere", f"{npl[0.0]} at 0, {npl[npl.index > 0].min()}--{npl.max()}",
          npl[0.0] == 6 and npl[npl.index > 0].min() == 52 and npl.max() == 150)
    G = [family_lines("wind_gust", t, FA) for t in PRIMARY]
    gm, gr = np.mean([g[0][0.8] for g in G]), np.mean([g[1][0.8] for g in G])
    check("gust at lambda 0.8: learned vs MPC", "0.49 vs 0.84", f"{gr:.2f} vs {gm:.2f}", near(gr, .49, .005) and near(gm, .84, .005))

    # ---------------------------------------------------------------------------------------------------------
    section("Results 7.3: geometry")
    man = pd.read_csv(L9 / "tracks/manifest_study.csv").set_index("seed")

    def mean_turn(t):
        dd = json.load(open(L9 / f"tracks/study/track_{t:04d}.json"))
        P_ = [np.array(dd["start"][:2])] + [np.array(g["pos"][:2]) for g in dd["gates"]] + [np.array(dd["end"][:2])]
        a = [np.degrees(np.arccos(np.clip(np.dot(P_[i] - P_[i - 1], P_[i + 1] - P_[i]) /
                                          np.linalg.norm(P_[i] - P_[i - 1]) / np.linalg.norm(P_[i + 1] - P_[i]), -1, 1)))
             for i in range(1, 5)]
        return float(np.mean(a))

    turn = {t: mean_turn(t) for t in lib.TRACKS}
    lam_full = s["lam_star_full"]
    x = [lam_full[t] for t in PRIMARY]
    for name, vals, paper_rho, paper_p in [("planned lap (as flown)", [man.loc[t, "lap_stretched_s"] for t in PRIMARY], .81, .05),
                                           ("speed stretch", [man.loc[t, "stretch"] for t in PRIMARY], .77, .07),
                                           ("mean turn", [turn[t] for t in PRIMARY], .06, None),
                                           ("frame margin", [man.loc[t, "frame_margin_m"] for t in PRIMARY], -.38, None)]:
        r = stats.spearmanr(x, vals)
        ptxt = f", p = {paper_p}" if paper_p else ""
        check(f"Spearman lambda*(1.0) vs {name}", f"rho = {paper_rho}{ptxt}", f"rho = {r.statistic:.2f}, p = {r.pvalue:.2f}",
              near(r.statistic, paper_rho, .005) and (paper_p is None or near(r.pvalue, paper_p, .005)))
    tv = np.mean([turn[t] for t in PRIMARY]); tn = np.mean([turn[t] for t in nonviable])
    check("mean turn, non-viable vs viable", "75 vs 61 deg", f"{tn:.0f} vs {tv:.0f}", round(tn) == 75 and round(tv) == 61)
    check("757 / 837 mean turn", "86 / 51", f"{turn[757]:.0f} / {turn[837]:.0f}", round(turn[757]) == 86 and round(turn[837]) == 51)
    nom = d[(d.cond == "wind_const") & (d.lam == 0.0) & d.member.isin(OPT4)]
    rv = nom[nom.track.isin(PRIMARY)].rmse_3d.median(); rn = nom[nom.track.isin(nonviable)].rmse_3d.median()
    check("nominal MPC RMSE, non-viable vs viable", "0.053 vs 0.054", f"{rn:.3f} vs {rv:.3f}", near(rn, .053, .0005) and near(rv, .054, .0005))
    gates = {}
    for t in nonviable:
        z = d[(d.cond == "wind_const") & (d.lam == 0.0) & (d.track == t) & d.member.isin(FA)]
        gates[t] = sorted(set(int(g) if g == g and g != "" else int(re.sub(r"\D", "", str(f)) or 0)
                              for g, f in zip(z.impact_gate, z.fail)))
    same = [t for t in nonviable if len(gates[t]) == 1]
    check("non-viable tracks where all three seeds stop at the same gate", "3 of 4 (747, 837, 1318)",
          f"{len(same)} of 4 {same} {gates}", same == [747, 837, 1318])

    # ---------------------------------------------------------------------------------------------------------
    section("Results 7.4 and Table 3: the authority corners")
    for name, mem, paper in [("full authority (primary)", FA, ("4 of 5", "2 of 4")), ("0.7 rad, MPC 1.0", S7, ("7 of 8", "5 of 7"))]:
        b = power_and_cross("lighthouse", mem); md = power_and_cross("lighthouse", mem, "median")
        cov = sum(power_and_cross(c, mem)["cross_all"] for c in ["wind_const", "payload", "mass_mult"])
        got_b, got_m = f"{b['cross_pow']} of {b['ret']}", f"{md['cross_pow']} of {md['ret']}"
        check(f"{name}: sensing family max / median seed / covered crossovers", f"{paper[0]} / {paper[1]} / 0",
              f"{got_b} / {got_m} / {cov}", (got_b, got_m, cov) == (paper[0], paper[1], 0))
    med = [power_and_cross("lighthouse", FA, "median", "max", dl)["cross_all"] for dl in (0.1, 0.2, 0.3)]
    check("primary median seed at delta = 0.1 / 0.2 / 0.3", "3 / 2 / 0", " / ".join(map(str, med)), med == [3, 2, 0])
    fa0 = power_and_cross("lighthouse", ["fa0"])
    check("validation-selected seed (fa0): sensing crossovers", "3 of its 4 tracks", f"{fa0['cross_all']} of {fa0['n']}",
          (fa0["cross_all"], fa0["n"]) == (3, 4))
    ev = L9 / "results/eval"

    def val_count(folder, member):
        z = pd.concat([pd.read_csv(f_) for f_ in glob.glob(str(ev / folder / "*.csv"))])
        z = z[(z.member == member) & (z.lam == 0.0)]
        return int(z.groupby("track").completed.max().sum())

    v_fa = [val_count("fa_16M_check", f"fa{i}_16M") for i in range(3)]
    v_s7 = [val_count("val_v4_final", "v4_16M"), val_count("val_seeds", "s1"), val_count("val_seeds", "s2")]
    v_m1 = val_count("fa_16M_check", "M1")
    check("validation tracks at nominal, full authority / 0.7 rad / M1", "12,5,10 / 13,13,6 / 22",
          f"{v_fa} / {v_s7} / {v_m1}", v_fa == [12, 5, 10] and v_s7 == [13, 13, 6] and v_m1 == 22)
    check("validation means", "9.0 vs 10.7", f"{np.mean(v_fa):.1f} vs {np.mean(v_s7):.1f}",
          near(np.mean(v_fa), 9.0, .05) and near(np.mean(v_s7), 10.7, .05))

    def verdict(folder):
        txt = open(ev / folder / "summary.txt").read()
        mm = re.search(r"tally over the 8 cells: .*->\s+(\w+)", txt)
        return {"MIXED": "M", "GENERALISES": "G", "OVERFIT": "O"}.get(mm.group(1).upper().rstrip(":"), mm.group(1))

    for label, folders, paper in [("full authority vs uncapped", ["val_heldout_fa16_s0", "val_heldout_fa16_s1", "val_heldout_fa16_s2"], "M, G, M"),
                                  ("0.7 rad vs uncapped", ["val_heldout", "val_heldout_s1", "val_heldout_s2"], "G, G, M"),
                                  ("0.7 rad vs capped 0.7", ["val_heldout_cap07_s0", "val_heldout_cap07_s1", "val_heldout_cap07_s2"], "M, M, M")]:
        got_v = ", ".join(verdict(f_) for f_ in folders)
        check(f"held-out verdicts, {label}", paper, got_v, got_v == paper)

    # ---------------------------------------------------------------------------------------------------------
    section("Results 7.5: held-out cells and the training edge")
    light, edge = [], []
    for f_ in ["val_heldout_fa16_s0", "val_heldout_fa16_s1", "val_heldout_fa16_s2", "val_heldout", "val_heldout_s1", "val_heldout_s2"]:
        txt = open(ev / f_ / "summary.txt").read()
        mm = re.search(r"^light\s+1\.0\s+([+-]?\d+)\s+([+-]?\d+)\s+(.+)$", txt, re.M)
        light.append((int(mm.group(1)), int(mm.group(2)), "OVERFIT" in mm.group(3)))
        e = re.search(r"\(here ([+-]?\d+)\)", txt)
        edge.append(int(e.group(1)) if e else None)
    over = [l_ for l_ in light if l_[2]]
    vals = [v for l_ in over for v in l_[:2]]
    check("lighter drone at lambda 1: seeds with the overfit signature, DiD range", "5 of 6, -43 to -127",
          f"{len(over)} of 6, {max(vals)} to {min(vals)}", len(over) == 5 and max(vals) == -43 and min(vals) == -127)
    sig = [e for e in edge if e is not None and e >= 25]
    check("lift-edge signature (>= 25): seeds, range", "4 of 6, 40--83", f"{len(sig)} of 6, {min(sig)}--{max(sig)}",
          len(sig) == 4 and min(sig) == 40 and max(sig) == 83)
    per_seed = {}
    for mem in FA + S7:
        tr_ = [t for t in lib.TRACKS if viable_members("lighthouse", t, [mem])]
        c8 = np.mean([comp_table("lighthouse").loc[(t, mem, 0.8)] for t in tr_])
        c10 = np.mean([comp_table("lighthouse").loc[(t, mem, 1.0)] for t in tr_])
        per_seed[mem] = (round(c8, 2), round(c10, 2))
    check("sensing 0.8 -> 1.0 per seed, all declining", "gradual decline for every seed", per_seed,
          all(a > b for a, b in per_seed.values()))

    # ---------------------------------------------------------------------------------------------------------
    section("Methodology: timing")
    dd_ = d[(d.steps > 0)].copy(); dd_["ms"] = 1000 * dd_.wall_s.astype(float) / dd_.steps
    pol = dd_[dd_.member.isin(FA + S7)].ms.median(); mppi = dd_[dd_.member == "mppi_l1"].ms.median()
    solve = pd.to_numeric(d[d.member.isin(OPT4)].solve_ms_mean, errors="coerce").median()
    check("per-step wall time: policy / MPPI+L1; MPC solve", "about 4 / 16 ms; 30 ms",
          f"{pol:.1f} / {mppi:.1f} ms; {solve:.1f} ms", near(pol, 4, .5) and near(mppi, 16, .5) and near(solve, 30, .7))

    print(f"\n{sum(RESULTS)} of {len(RESULTS)} checks pass.")


if __name__ == "__main__":
    main()
