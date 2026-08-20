"""Render a freestyle rollout npz as an animated 3D GIF (paper/demo videos).

Usage: python -m crazy_track.eval.freestyle_gif --npz <rollout.npz>
           --track lsy-level2 --out flight.gif [--fps 25] [--slowmo 4]

Shows the quad as an X-frame with a red front arm and a body-z stub so
rotations are visible; flip windows play at `slowmo`x slow motion.
"""

from __future__ import annotations

import argparse

import numpy as np

from crazy_track.trajectories.freestyle import TRACKS


def _frame_times(t_end: float, windows: list[tuple[float, float]],
                 fps: int, slowmo: int) -> np.ndarray:
    """Playback timeline: real-time steps of 1/fps, 1/(fps*slowmo) in windows."""
    times, t = [], 0.0
    while t < t_end:
        times.append(t)
        in_win = any(a - 0.15 <= t <= b + 0.3 for a, b in windows)
        t += 1.0 / (fps * slowmo) if in_win else 1.0 / fps
    return np.asarray(times)


def render_gif(npz_path: str, track: str, out: str, fps: int = 25,
               slowmo: int = 4) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from scipy.spatial.transform import Rotation as R

    data = np.load(npz_path)
    traj = TRACKS[track]()
    t, pos, quat = data["t"], data["pos"], data["quat"]
    ref = traj.pos(np.arange(0.0, traj.duration, 0.02))
    windows = [(f["t_rot_start"], f["t_rot_end"]) for f in traj.flips]
    ftimes = _frame_times(float(t[-1]), windows, fps, slowmo)

    fig = plt.figure(figsize=(8, 6.5))
    ax = fig.add_subplot(projection="3d")
    ax.plot(ref[:, 0], ref[:, 1], ref[:, 2], "k--", lw=0.8, alpha=0.6)
    for k, g in enumerate(traj.gates):
        e_y = g.rotation[:, 1] * g.HALF_OPENING
        e_z = np.array([0.0, 0.0, g.HALF_OPENING])
        corners = np.array([g.center + s * e_y + u * e_z
                            for s, u in [(1, 1), (-1, 1), (-1, -1), (1, -1), (1, 1)]])
        ax.plot(corners[:, 0], corners[:, 1], corners[:, 2], "r-", lw=2)
        ax.plot([g.center[0]] * 2, [g.center[1]] * 2,
                [g.center[2] - g.HALF_OPENING, 0.0], "-", color="0.6", lw=1)
        ax.text(*(g.center + np.array([0, 0, 0.3])), f"G{k + 1}", color="r", fontsize=9)

    trail, = ax.plot([], [], [], "tab:blue", lw=1.2)
    arm1, = ax.plot([], [], [], "-", color="crimson", lw=2.5)
    arm2, = ax.plot([], [], [], "-", color="0.2", lw=2.5)
    zstub, = ax.plot([], [], [], "-", color="tab:green", lw=2)
    shadow, = ax.plot([], [], [], ".", color="0.7", ms=6)
    title = ax.set_title("")

    lims = np.concatenate([ref, pos])
    lo, hi = lims.min(axis=0) - 0.3, lims.max(axis=0) + 0.3
    lo[2] = 0.0
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
    ax.set_box_aspect(hi - lo)
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_zlabel("z [m]")
    ax.view_init(elev=22, azim=-60)

    L = 0.18  # exaggerated arm half-length for visibility

    def update(ti):
        i = min(np.searchsorted(t, ti), len(t) - 1)
        p, rot = pos[i], R.from_quat(quat[i])
        x, y, z = rot.apply([L, 0, 0]), rot.apply([0, L, 0]), rot.apply([0, 0, L])
        trail.set_data(pos[:i + 1, 0], pos[:i + 1, 1])
        trail.set_3d_properties(pos[:i + 1, 2])
        arm1.set_data([p[0] - x[0], p[0] + x[0]], [p[1] - x[1], p[1] + x[1]])
        arm1.set_3d_properties([p[2] - x[2], p[2] + x[2]])
        arm2.set_data([p[0] - y[0], p[0] + y[0]], [p[1] - y[1], p[1] + y[1]])
        arm2.set_3d_properties([p[2] - y[2], p[2] + y[2]])
        zstub.set_data([p[0], p[0] + z[0]], [p[1], p[1] + z[1]])
        zstub.set_3d_properties([p[2], p[2] + z[2]])
        shadow.set_data([p[0]], [p[1]])
        shadow.set_3d_properties([0.0])
        slow = any(a - 0.15 <= ti <= b + 0.3 for a, b in windows)
        title.set_text(f"freestyle {track}  t={ti:5.2f}s"
                       + (f"   FLIP  ({slowmo}x slow-mo)" if slow else ""))
        return trail, arm1, arm2, zstub, shadow, title

    anim = FuncAnimation(fig, update, frames=ftimes, blit=False)
    anim.save(out, writer=PillowWriter(fps=fps), dpi=80)
    plt.close(fig)
    print(f"wrote {out} ({len(ftimes)} frames @ {fps} fps)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", required=True)
    parser.add_argument("--track", required=True, choices=sorted(TRACKS))
    parser.add_argument("--out", required=True)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--slowmo", type=int, default=4)
    args = parser.parse_args()
    render_gif(args.npz, args.track, args.out, args.fps, args.slowmo)


if __name__ == "__main__":
    main()
