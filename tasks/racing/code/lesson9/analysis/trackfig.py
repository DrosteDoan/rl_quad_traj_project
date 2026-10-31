"""The ten swept study tracks, top-down: gates, pass directions, planned path coloured by height, and the
direction of the constant wind and of the gust's oscillation (both fixed in the world frame for every track)."""
import json
import os

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D

from lib import TRACKS, _L9

plt.rcParams.update({"font.size":7,"axes.titlesize":7,"axes.labelsize":7,"xtick.labelsize":6,"ytick.labelsize":6,
    "font.family":"serif","axes.edgecolor":"#52514e","axes.linewidth":0.6,"xtick.color":"#52514e","ytick.color":"#52514e",
    "text.color":"#0b0b0b","axes.labelcolor":"#0b0b0b","pdf.fonttype":42,"mathtext.fontset":"stix"})
INK="#0b0b0b"; INK2="#52514e"; GRID="#e4e3df"
OUT = os.path.join(os.environ.get("FIG_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")), "")
os.makedirs(OUT, exist_ok=True)
REPO = os.path.abspath(os.path.join(_L9, "..", "..", "..", ".."))
PRIMARY = {757, 965, 989, 1089, 1145, 1250}         # tracks on which both families have a viable member
ARENA = (-2.5, -1.5, 2.5, 1.5)                       # track_lib.ARENA_LOW / ARENA_HIGH
WIND = np.array([1.0, 0.0]); GUST = np.array([1.0, 0.3]) / np.hypot(1.0, 0.3)   # knobs.NOMINAL
height = LinearSegmentedColormap.from_list("height", ["#cfe2fa", "#9ec5f4", "#2a78d6", "#104281"])
norm = Normalize(0.0, 1.3)

man = pd.read_csv(os.path.join(_L9, "tracks", "manifest_study.csv")).set_index("seed")
order = sorted(TRACKS, key=lambda t: man.loc[t, "lap_stretched_s"])

fig, axs = plt.subplots(2, 5, figsize=(7.0, 2.75))
for ax, t in zip(axs.flat, order):
    trk = json.load(open(os.path.join(_L9, "tracks", "study", f"track_{t:04d}.json")))
    plan = pd.read_csv(os.path.join(REPO, trk["plan_csv"]))
    p = plan[["p_x", "p_y", "p_z"]].to_numpy()
    seg = np.stack([p[:-1, :2], p[1:, :2]], axis=1)
    lc = LineCollection(seg, cmap=height, norm=norm, lw=1.3, capstyle="round")
    lc.set_array((p[:-1, 2] + p[1:, 2]) / 2)
    ax.add_collection(lc)
    ax.add_patch(plt.Rectangle(ARENA[:2], ARENA[2] - ARENA[0], ARENA[3] - ARENA[1], fill=False, ec=GRID, lw=0.8))
    for i, g in enumerate(trk["gates"], start=1):
        c, n = np.asarray(g["pos"][:2]), np.array([np.cos(g["yaw"]), np.sin(g["yaw"])])
        tang = np.array([-n[1], n[0]])
        a, b = c - 0.2 * tang, c + 0.2 * tang                       # the 0.4 m opening
        ax.plot([a[0], b[0]], [a[1], b[1]], color=INK, lw=2.0, solid_capstyle="butt", zorder=3)
        ax.annotate("", xy=c + 0.55 * n, xytext=c + 0.08 * n, arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.8,
                    mutation_scale=6, shrinkA=0, shrinkB=0), zorder=3)
        off = -0.32 * n + 0.08 * tang
        ax.text(*(c + off), str(i), fontsize=6, ha="center", va="center", color=INK, zorder=4)
    ax.plot(*trk["start"][:2], marker="s", ms=3.5, color=INK, ls="none", zorder=4)
    star = "" if t in PRIMARY else r"$^\dagger$"
    ax.set_title(f"{t}{star}  ({man.loc[t, 'lap_stretched_s']:.1f} s)", loc="left", pad=2)
    ax.set_xlim(-2.6, 2.6); ax.set_ylim(-1.6, 1.6); ax.set_aspect("equal")
    ax.set_xticks([-2, 0, 2]); ax.set_yticks([-1, 0, 1]); ax.tick_params(length=2, width=0.5, pad=1)
    for s in ax.spines.values(): s.set_visible(False)
    if ax not in axs[:, 0]: ax.set_yticklabels([])
    if ax not in axs[1]: ax.set_xticklabels([])
for ax in axs[1]: ax.set_xlabel("x [m]", labelpad=1)
for ax in axs[:, 0]: ax.set_ylabel("y [m]", labelpad=1)

fig.subplots_adjust(left=0.06, right=0.89, bottom=0.12, top=0.77, wspace=0.06, hspace=0.12)

# key: wind directions (same in every panel, world frame)
key = fig.add_axes([0.06, 0.855, 0.30, 0.13]); key.set_xlim(0, 10); key.set_ylim(0, 2); key.axis("off")
for y, d, lbl, style in [(1.45, WIND, r"constant wind and gust mean: $+x$", "-|>"),
                         (0.45, GUST, r"gust oscillation axis: $(1, 0.3, 0)$", "<|-|>")]:
    key.annotate("", xy=(0.9 + 0.75 * d[0], y + 0.75 * d[1] * 0.6), xytext=(0.9 - (0.75 * d[0] if style == "<|-|>" else 0),
                 y - (0.75 * d[1] * 0.6 if style == "<|-|>" else 0)),
                 arrowprops=dict(arrowstyle=style, color=INK2, lw=1.0, mutation_scale=7))
    key.text(2.1, y, lbl, va="center", fontsize=6.5, color=INK)
h = [Line2D([], [], color=INK, lw=2.0, label="gate (0.4 m opening), arrow: pass direction"),
     Line2D([], [], marker="s", ms=3.5, color=INK, ls="none", label="ground start")]
fig.legend(handles=h, loc="upper left", bbox_to_anchor=(0.40, 0.995), frameon=False, fontsize=6.5, handlelength=1.6)
fig.text(0.40, 0.845, r"gates 1, 3 at 0.7 m and 2, 4 at 1.2 m height;  $^\dagger$ not raced by the primary policy",
         fontsize=6.5, color=INK2, va="center")

cax = fig.add_axes([0.905, 0.17, 0.012, 0.55])
cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=height), cax=cax)
cb.set_label("planned height z [m]", fontsize=6.5); cb.set_ticks([0, 0.7, 1.2])
cb.ax.tick_params(length=2, width=0.5, labelsize=6); cb.outline.set_visible(False)

fig.savefig(OUT + "fig_tracks.pdf"); fig.savefig(OUT + "fig_tracks.png", dpi=250)
print("done")
