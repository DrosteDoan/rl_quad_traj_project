"""Extra figures for the thesis version, from the stored laps (same rules as analysis/). Sized for a 16 cm text width."""
import os, sys, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
A = os.environ.get("ANALYSIS_DIR", os.path.dirname(os.path.abspath(__file__)))   # default: run from analysis/
sys.path.insert(0, A)
from lib import *
from pairs import paired
from lib import _L9 as L9
OUT = os.path.join(os.environ.get("FIG_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")), "")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({"font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "font.family": "serif", "font.serif": ["Liberation Serif", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix", "axes.edgecolor": "#52514e", "axes.linewidth": 0.6, "xtick.color": "#52514e",
    "ytick.color": "#52514e", "text.color": "#0b0b0b", "axes.labelcolor": "#0b0b0b", "pdf.fonttype": 42})
INK = "#0b0b0b"; INK2 = "#52514e"; GRID = "#e4e3df"; BLUE = "#2a78d6"; ORANGE = "#eb6834"; GREY = "#9a9893"
P6 = [757, 965, 989, 1089, 1145, 1250]
NAMES = {"wind_const": "Constant wind", "payload": "Steady downward force", "mass_mult": "Mass error",
         "wind_gust": "Gust", "lighthouse": "Sensing degradation", "combined": "Combined"}
X = np.array(COARSE)

def style(ax, xl=True):
    ax.grid(axis="y", color=GRID, lw=0.5); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.tick_params(length=2, width=0.5)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    if xl: ax.set_xlabel(r"intensity $\lambda$")

def marks(ax, c):
    ax.axvline(0.8, color=INK2, lw=0.8, ls=(0, (3, 2)))
    if c in ("lighthouse", "combined"): ax.axvline(0.67, color=INK2, lw=0.8, ls=(0, (4, 1.5, 1, 1.5)))

def pooled(c, rl_how="max", mpc_how="max", rl=FA):
    M = pd.concat([family_lines(c, t, rl, rl_how, mpc_how)[0] for t in P6], axis=1)
    R = pd.concat([family_lines(c, t, rl, rl_how, mpc_how)[1] for t in P6], axis=1)
    return M, R

# ------------------------------------------------------------------ family lines, six channels
fig, axs = plt.subplots(2, 3, figsize=(6.3, 4.1), sharey=True)
for ax, c in zip(axs.flat, CONDS):
    Mb, Rb = pooled(c); Mm, _ = pooled(c, "max", "median"); _, Rm = pooled(c, "median", "max")
    ax.fill_between(X, Rb.min(1), Rb.max(1), color=BLUE, alpha=0.10, lw=0)
    ax.fill_between(X, Mb.min(1), Mb.max(1), color=ORANGE, alpha=0.10, lw=0)
    ax.plot(X, Mb.mean(1), color=ORANGE, lw=1.4, marker="s", ms=3, mec="white", mew=0.4)
    ax.plot(X, Mm.mean(1), color=ORANGE, lw=1.1, ls=(0, (1, 1.2)))
    ax.plot(X, Rb.mean(1), color=BLUE, lw=1.4, marker="o", ms=3, mec="white", mew=0.4)
    ax.plot(X, Rm.mean(1), color=BLUE, lw=1.1, ls=(0, (1, 1.2)))
    marks(ax, c); style(ax, ax in axs[1]); ax.set_ylim(-0.03, 1.05); ax.set_title(NAMES[c], loc="left", pad=3)
for ax in axs[:, 0]: ax.set_ylabel("completion")
h = [Line2D([], [], color=BLUE, lw=1.4, marker="o", ms=3, label="learned, family maximum"),
     Line2D([], [], color=BLUE, lw=1.1, ls=(0, (1, 1.2)), label="learned, median seed"),
     Line2D([], [], color=ORANGE, lw=1.4, marker="s", ms=3, label="MPC, family maximum"),
     Line2D([], [], color=ORANGE, lw=1.1, ls=(0, (1, 1.2)), label="MPC, median member"),
     Line2D([], [], color=INK2, lw=0.8, ls=(0, (3, 2)), label="training edge"),
     Line2D([], [], color=INK2, lw=0.8, ls=(0, (4, 1.5, 1, 1.5)), label="nominal sensing model")]
fig.legend(handles=h, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.52, 1.0))
fig.subplots_adjust(left=0.09, right=0.99, bottom=0.11, top=0.83, wspace=0.12, hspace=0.42)
fig.savefig(OUT + "fig_family_lines.pdf"); plt.close(fig)

# ------------------------------------------------------------------ every member on the unmodelled channels
fig, axs = plt.subplots(1, 2, figsize=(6.3, 2.7), sharey=True)
oc = {"M1": "#7a2e10", "M1+ESO": "#c4501f", "M1+L1": "#eb6834", "M1+mass": "#f2a07a", "mppi_l1": "#b5a79f"}
bc = {"fa0": "#104281", "fa1": "#2a78d6", "fa2": "#8fbbf0"}
lab = {"M1": "M1", "M1+ESO": "M1+ESO", "M1+L1": "M1+L1", "M1+mass": "M1+mass", "mppi_l1": "MPPI+L1", "fa0": "fa0", "fa1": "fa1", "fa2": "fa2"}
for ax, c in zip(axs, ["lighthouse", "combined"]):
    for m in MPC + FA:
        y = pd.concat([curve(c, t, [m], "max") for t in P6], axis=1).mean(1)
        col = oc.get(m, bc.get(m)); mk = "s" if m in MPC else "o"
        ax.plot(X, y, color=col, lw=1.1, marker=mk, ms=2.6, label=lab[m])
    marks(ax, c); style(ax); ax.set_title(NAMES[c], loc="left", pad=3); ax.set_ylim(-0.03, 1.05)
axs[0].set_ylabel("completion")
fig.legend(*axs[0].get_legend_handles_labels(), loc="center right", frameon=False, bbox_to_anchor=(1.0, 0.5))
fig.subplots_adjust(left=0.09, right=0.84, bottom=0.17, top=0.9, wspace=0.1)
fig.savefig(OUT + "fig_members_uncovered.pdf"); plt.close(fig)

# ------------------------------------------------------------------ paired RMSE on the stochastic channels
fig, axs = plt.subplots(1, 3, figsize=(6.3, 2.3), sharey=True)
for ax, c in zip(axs, ["wind_gust", "lighthouse", "combined"]):
    p = paired(c, coarse_only=True).dropna(subset=["mpc", "rl"])
    g = p.groupby("lam")[["mpc", "rl"]].median(); n = p.groupby("lam").size()
    g = g[n >= 10]
    ax.plot(g.index, g.rl, color=BLUE, lw=1.4, marker="o", ms=3, mec="white", mew=0.4, label="learned")
    ax.plot(g.index, g.mpc, color=ORANGE, lw=1.4, marker="s", ms=3, mec="white", mew=0.4, label="MPC")
    marks(ax, c); style(ax); ax.set_title(NAMES[c], loc="left", pad=3); ax.set_ylim(0, 0.3); ax.set_xlim(-0.03, 1.03)
axs[0].set_ylabel("median paired RMSE [m]")
axs[0].legend(frameon=False, loc="center left")
fig.subplots_adjust(left=0.1, right=0.99, bottom=0.2, top=0.88, wspace=0.1)
fig.savefig(OUT + "fig_rmse.pdf"); plt.close(fig)

# ------------------------------------------------------------------ failure causes
d = load()
z = d[d.track.isin(P6) & (d.completed == 0) & d.member.isin(MPC + FA)].copy()
z["fam"] = np.where(z.member.isin(MPC), "MPC", "learned")
fig, ax = plt.subplots(figsize=(6.3, 2.9))
rows = []
for c in CONDS:
    for fam in ["MPC", "learned"]:
        y = z[(z.cond == c) & (z.fam == fam)]
        rows.append((f"{NAMES[c]}, {fam}", (y.fail == "contact").mean(), 1 - (y.fail == "contact").mean(), len(y)))
yy = np.arange(len(rows))[::-1]
for yi, (lbl, ct, ms, n) in zip(yy, rows):
    col = ORANGE if lbl.endswith("MPC") else BLUE
    ax.barh(yi, ct, color=col, height=0.7)
    ax.barh(yi, ms, left=ct, color=col, alpha=0.35, height=0.7, hatch="////", edgecolor="white", lw=0)
    ax.text(1.01, yi, f"{n:,} failed laps", va="center", fontsize=7.5, color=INK2)
ax.set_yticks(yy); ax.set_yticklabels([r[0] for r in rows], fontsize=7.5)
ax.set_xlim(0, 1); ax.set_xlabel("share of failed laps")
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.tick_params(length=2, width=0.5)
ax.legend(handles=[Patch(color=GREY, label="frame contact"), Patch(facecolor=GREY, alpha=0.35, hatch="////", edgecolor="white", label="missed gate")],
          frameon=False, loc="upper center", bbox_to_anchor=(0.4, 1.12), ncol=2)
fig.subplots_adjust(left=0.3, right=0.85, bottom=0.15, top=0.9)
fig.savefig(OUT + "fig_failures.pdf"); plt.close(fig)

# ------------------------------------------------------------------ plan properties, viable vs not
man = pd.read_csv(os.path.join(L9, "tracks", "manifest_study.csv")).set_index("seed")
def mean_turn(t):
    dd = json.load(open(os.path.join(L9, "tracks", "study", f"track_{t:04d}.json")))
    P_ = [np.array(dd["start"][:2])] + [np.array(g["pos"][:2]) for g in dd["gates"]] + [np.array(dd["end"][:2])]
    return float(np.mean([np.degrees(np.arccos(np.clip(np.dot(P_[i] - P_[i-1], P_[i+1] - P_[i]) /
            np.linalg.norm(P_[i] - P_[i-1]) / np.linalg.norm(P_[i+1] - P_[i]), -1, 1))) for i in range(1, 5)]))
def gate_path(t):
    dd = json.load(open(os.path.join(L9, "tracks", "study", f"track_{t:04d}.json")))
    P_ = [np.array(dd["start"])] + [np.array(g["pos"]) for g in dd["gates"]] + [np.array(dd["end"])]
    return float(sum(np.linalg.norm(P_[i+1] - P_[i]) for i in range(len(P_) - 1)))
props = {"mean turn [°]": {t: mean_turn(t) for t in TRACKS},
         "gate path [m]": {t: gate_path(t) for t in TRACKS},
         "frame margin [cm]": {t: 100 * man.loc[t, "frame_margin_m"] for t in TRACKS},
         "planned lap, as flown [s]": {t: man.loc[t, "lap_stretched_s"] for t in TRACKS}}
fig, axs = plt.subplots(1, 4, figsize=(6.3, 2.4))
for ax, (k, v) in zip(axs, props.items()):
    for i, grp in enumerate([P6, [t for t in TRACKS if t not in P6]]):
        vals = [v[t] for t in grp]
        xs = np.full(len(vals), i) + np.linspace(-0.12, 0.12, len(vals))
        ax.scatter(xs, vals, s=16, color=BLUE if i == 0 else GREY, zorder=3)
        ax.plot([i - 0.25, i + 0.25], [np.mean(vals)] * 2, color=INK, lw=1)
        for xx, vv, t in zip(xs, vals, grp):
            if t in (757, 837): ax.annotate(str(t), (xx, vv), xytext=(4, 0), textcoords="offset points", fontsize=7, va="center")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["viable", "not\nviable"]); ax.set_xlim(-0.6, 1.6)
    ax.set_title(k, fontsize=8.5, pad=3)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.tick_params(length=2, width=0.5); ax.grid(axis="y", color=GRID, lw=0.5); ax.set_axisbelow(True)
fig.subplots_adjust(left=0.06, right=0.99, bottom=0.18, top=0.88, wspace=0.45)
fig.savefig(OUT + "fig_geometry.pdf"); plt.close(fig)
pd.DataFrame(props).round(2).to_csv(OUT + "geometry_table.csv")

# ------------------------------------------------------------------ development: validation curve of the gate-aware recipe
# development numbers, not the study sweep: the checkpoint curve is stored in ../results/eval/val_v4_curve and
# val_v4_final; the screening-track counts are entered here (their laps: ../results/eval/screen)
ck = [2, 4, 6, 8, 10, 12, 14, 16]
comp = [0, 3, 10, 11, 10, 14, 16, 13]; gates = [0, 43, 64, 65, 67, 74, 77, 68]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.3, 2.6), gridspec_kw={"width_ratios": [1.15, 1]})
a1.plot(ck, comp, color=BLUE, lw=1.4, marker="o", ms=3.5, label="gate-aware recipe, seed 0")
a1.axhline(22, color=ORANGE, lw=1.6, zorder=3, label="M1 (no re-tuning)")
a1.axhline(11, color=INK2, lw=0.8, ls=(0, (3, 2)), label="viability bar (fixed in advance)")
a1.scatter([16], [13], s=60, facecolor="none", edgecolor=INK, zorder=4)
a1.set_ylim(0, 24.5); a1.set_ylabel("validation tracks completed (of 22)"); a1.set_xlabel("training steps [M]")
a1.legend(frameon=False, loc="lower right", fontsize=7.5)
names = ["open-\nspace", "authority\nscreen", "gate-\naware v1", "gate-\naware v2", "gate-\naware v3"]
g6 = [10, 8, 0, 2, 11]; c6 = [1, 0, 0, 0, 1]
a2.bar(range(5), g6, color=BLUE, alpha=0.8); a2.set_xticks(range(5)); a2.set_xticklabels(names, fontsize=7)
for i, (g, c) in enumerate(zip(g6, c6)): a2.text(i, g + 0.4, f"{c}/6 laps", ha="center", fontsize=7, color=INK2)
a2.set_ylabel("gates passed (of 24)"); a2.set_ylim(0, 14)
for ax in (a1, a2):
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.tick_params(length=2, width=0.5); ax.grid(axis="y", color=GRID, lw=0.5); ax.set_axisbelow(True)
a1.set_xticks(ck)
fig.subplots_adjust(left=0.09, right=0.99, bottom=0.25, top=0.95, wspace=0.3)
fig.savefig(OUT + "fig_development.pdf"); plt.close(fig)
print("done")
