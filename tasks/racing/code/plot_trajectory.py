"""Draw a race reference trajectory WITH its speed profile — Lesson 5 §2.

A trajectory is not a line, it is a *schedule*: where to be AND how fast to be
going. A path that looks smooth on paper can hide an acceleration spike that no
controller can deliver (thrust-to-weight is 1.88 — physics does not negotiate).
This script makes both visible:

  figure 1  top-down track map, the path colored by speed, gates + obstacles
  figure 2  the speed profile |v|(t), altitude z(t), and the thrust demand
            |a + g| against the feasibility limit — the "can this even be
            flown?" panel

Run in the MAIN venv (it needs only numpy/matplotlib + the vendored
crazy_track):

    python tasks/racing/code/plot_trajectory.py --track lsy-level2-race --cruise 3.0

Overlay a flown lap (recorded by race_bridge.py when RACE_LOG_DIR is set,
see Lesson 5 §3) to see where reality peeled away from the plan:

    python tasks/racing/code/plot_trajectory.py --track lsy-level2-race --cruise 3.0 \
        --flown tasks/racing/figures/v5_s0/flown_ep01.csv

Figures land in tasks/racing/figures/ (git-ignored).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

from crazy_track.trajectories.freestyle import (
    GRAVITY,
    RaceGate,
    TRACKS,
    feasibility_report,
)

TWR = 1.88                      # cf21B_500 thrust-to-weight (measured, Lesson 1)
ACC_LIMIT = 0.95 * TWR * GRAVITY  # the same bound feasibility_report enforces

# One sequential colormap for one job: speed = magnitude. Perceptually uniform
# and colorblind-safe; never a rainbow (jet distorts magnitudes).
SPEED_CMAP = "viridis"


# --------------------------------------------------------------------- helpers
def sample(traj, dt: float = 0.01):
    """Sample the reference on a fixed grid: t, pos (N,3), speed |v| (N,)."""
    t = np.arange(0.0, traj.duration, dt)
    pos = traj.pos(t)
    vel = traj.vel(t)
    speed = np.linalg.norm(vel, axis=1)
    return t, pos, speed


def speed_colored_path(ax, xy: np.ndarray, speed: np.ndarray, vmax: float):
    """Draw a 2-D path as segments colored by speed. Returns the mappable."""
    pts = xy.reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc = LineCollection(segs, cmap=SPEED_CMAP, linewidth=2.5, zorder=3)
    lc.set_array(speed[:-1])
    lc.set_clim(0.0, vmax)
    ax.add_collection(lc)
    return lc


def draw_gate(ax, gate: RaceGate, name: str):
    """Top-down gate: the frame (thin), the opening (thick), the fly-through
    arrow along the gate normal, and a small label."""
    c = gate.center[:2]
    # In-plane axis (perpendicular to the normal, in the XY plane).
    inplane = gate.rotation[:2, 1]
    for half, lw in ((RaceGate.OUTER / 2.0, 1.0), (RaceGate.HALF_OPENING, 3.0)):
        p0, p1 = c - half * inplane, c + half * inplane
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color="0.25", lw=lw,
                solid_capstyle="butt", zorder=4)
    n = gate.normal[:2]
    ax.annotate("", xy=c + 0.22 * n, xytext=c,
                arrowprops=dict(arrowstyle="->", color="0.25", lw=1.0), zorder=4)
    ax.annotate(name, c, textcoords="offset points", xytext=(6, 6),
                fontsize=9, color="0.25", zorder=5)


# --------------------------------------------------------------------- figures
def figure_track_map(traj, t, pos, speed, flown, out: Path):
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    vmax = float(speed.max())

    lc = speed_colored_path(ax, pos[:, :2], speed, vmax)
    fig.colorbar(lc, ax=ax, label="reference speed  (m/s)", shrink=0.85)

    for i, g in enumerate(traj.gates, start=1):
        draw_gate(ax, g, f"G{i}")
    for ob in getattr(traj, "obstacles", []):
        ax.plot(ob[0], ob[1], marker="x", ms=8, color="0.55", mew=1.5, zorder=2)

    ax.plot(*pos[0, :2], marker="o", ms=8, color="0.25", zorder=5)
    ax.annotate("start", pos[0, :2], textcoords="offset points", xytext=(6, -12),
                fontsize=9, color="0.25")

    if flown is not None:
        ax.plot(flown[:, 1], flown[:, 2], color="0.45", lw=1.2, ls="--",
                zorder=2, label="flown")
        ax.legend(loc="best", frameon=False)

    ax.set_xlabel("x  (m)")
    ax.set_ylabel("y  (m)")
    ax.set_title("Track map — reference colored by speed")
    ax.set_aspect("equal")
    ax.margins(0.08)
    ax.grid(True, color="0.9", lw=0.5)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    print(f"wrote {out}")


def figure_speed_profile(traj, t, pos, speed, flown, out: Path):
    acc = traj.acc(t)
    demand = np.linalg.norm(acc + np.array([0.0, 0.0, GRAVITY]), axis=1)

    fig, axes = plt.subplots(3, 1, figsize=(9, 7.5), sharex=True)
    ax_v, ax_z, ax_a = axes

    ax_v.plot(t, speed, lw=2.0, color=plt.get_cmap(SPEED_CMAP)(0.55))
    ax_v.set_ylabel("speed  (m/s)")
    ax_v.set_title("Speed profile — where the lap time actually lives")

    if flown is not None:
        ft = flown[:, 0]
        fspeed = np.linalg.norm(np.gradient(flown[:, 1:4], ft, axis=0), axis=1)
        ax_v.plot(ft, fspeed, lw=1.2, ls="--", color="0.45", label="flown")
        ax_v.legend(loc="best", frameon=False)

    ax_z.plot(t, pos[:, 2], lw=2.0, color="0.35")
    ax_z.set_ylabel("altitude z  (m)")
    ax_z.axhline(0.0, color="0.8", lw=0.8)

    ax_a.plot(t, demand, lw=2.0, color="0.35")
    ax_a.axhline(ACC_LIMIT, color="0.15", lw=1.2, ls=":")
    ax_a.annotate(f"feasibility limit 0.95·TWR·g = {ACC_LIMIT:.1f}",
                  (t[-1], ACC_LIMIT), textcoords="offset points",
                  xytext=(-4, 6), ha="right", fontsize=9, color="0.15")
    ax_a.set_ylabel("thrust demand |a+g|  (m/s²)")
    ax_a.set_xlabel("time  (s)")

    # Gate crossings: the lap's milestones, on every panel.
    for ax in axes:
        for i, tg in enumerate(traj.gate_times, start=1):
            ax.axvline(tg, color="0.85", lw=0.8, zorder=0)
    y_top = ax_v.get_ylim()[1]
    for i, tg in enumerate(traj.gate_times, start=1):
        ax_v.annotate(f"G{i}", (tg, y_top), textcoords="offset points",
                      xytext=(0, -4), ha="center", va="top",
                      fontsize=9, color="0.4")

    for ax in axes:
        ax.grid(True, color="0.92", lw=0.5)
        ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(out, dpi=160)
    print(f"wrote {out}")


# ------------------------------------------------------------------------ main
def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--track", default="lsy-level2-race", choices=sorted(TRACKS),
                   help="Which reference to draw (from crazy_track's TRACKS).")
    p.add_argument("--cruise", type=float, default=3.0,
                   help="Cruise speed handed to the track builder (m/s).")
    p.add_argument("--flown", type=Path, default=None,
                   help="CSV of a flown lap (t,x,y,z — from race_bridge logging).")
    p.add_argument("--out-dir", type=Path,
                   default=Path("tasks/racing/figures"),
                   help="Where the PNGs go.")
    args = p.parse_args()

    traj = TRACKS[args.track](cruise=args.cruise)
    t, pos, speed = sample(traj)

    flown = None
    if args.flown is not None:
        flown = np.loadtxt(args.flown, delimiter=",", skiprows=1)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.track}_c{args.cruise:g}"
    figure_track_map(traj, t, pos, speed, flown,
                     args.out_dir / f"{tag}_map.png")
    figure_speed_profile(traj, t, pos, speed, flown,
                         args.out_dir / f"{tag}_profile.png")

    # The habit this course keeps teaching: never fly an unchecked reference.
    rep = feasibility_report(traj)
    print(f"\nfeasibility_report({args.track}, cruise={args.cruise:g}):")
    for k, v in rep.items():
        if isinstance(v, float):
            print(f"  {k:32s} {v:.3f}")
        else:
            print(f"  {k:32s} {v}")
    if not rep["feasible"]:
        print("\n⚠️  NOT feasible — no controller can track this. Lower --cruise.")
    print(f"\nduration {traj.duration:.2f} s, last gate at "
          f"{traj.gate_times[-1]:.2f} s, peak speed {speed.max():.2f} m/s")


if __name__ == "__main__":
    main()
