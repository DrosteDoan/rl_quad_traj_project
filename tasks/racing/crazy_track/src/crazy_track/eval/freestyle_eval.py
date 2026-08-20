"""Freestyle-sequence evaluation: gates + flips chained (lsy_drone_racing track).

Usage: python -m crazy_track.eval.freestyle_eval --reason "..."          # plan only
       python -m crazy_track.eval.freestyle_eval --model <path> --reason "..."

Plan-only prints/logs the analytic feasibility report and renders the
reference. With --model, an acro4 policy is rolled out ZERO-SHOT (it never
trained on gates, multi-flip descriptors, traveling flips, or this track).
"""

from __future__ import annotations

import argparse

import numpy as np

from crazy_track.eval.runlog import RunLogger, tracking_metrics
from crazy_track.trajectories import RaceGate, feasibility_report
from crazy_track.trajectories.freestyle import TRACKS, FreestyleTrajectory

CLEAN_BOUND = 0.13  # opening half-width minus drone half-extent margin


def gate_crossing_metrics(pos: np.ndarray, t: np.ndarray,
                          traj: FreestyleTrajectory) -> list[dict]:
    """For each gate: in-plane offset of the actual crossing nearest the
    intended time (inf if the plane is never crossed near it)."""
    out = []
    for g, tg in zip(traj.gates, traj.gate_times):
        local = g.to_gate_frame(pos)
        x = local[:, 0]
        cross = np.flatnonzero(np.sign(x[1:]) != np.sign(x[:-1]))
        best = None
        for i in cross:
            w = float(x[i] / (x[i] - x[i + 1]))
            tc = float(t[i] + w * (t[i + 1] - t[i]))
            if abs(tc - tg) > 1.0:  # only count crossings near the intended pass
                continue
            yz = local[i, 1:] + w * (local[i + 1, 1:] - local[i, 1:])
            off = float(np.abs(yz).max())
            if best is None or abs(tc - tg) < abs(best[0] - tg):
                best = (tc, off)
        off = best[1] if best else np.inf
        out.append({"t_gate": round(tg, 2),
                    "t_cross": round(best[0], 3) if best else np.inf,
                    "offset": round(off, 3),
                    "passed": bool(off < RaceGate.HALF_OPENING),
                    "clean": bool(off < CLEAN_BOUND)})
    return out


def flip_rotation_metrics(quat: np.ndarray, t: np.ndarray,
                          traj: FreestyleTrajectory) -> list[dict]:
    from scipy.spatial.transform import Rotation as R

    out = []
    for f in traj.flips:
        m = (t >= f["t_rot_start"] - 0.2) & (t <= f["t_rot_end"] + 0.3)
        rots = R.from_quat(quat[m])
        rel = (rots[:-1].inv() * rots[1:]).as_rotvec()
        total = float(np.sum(rel[:, f["axis"]]))
        out.append({
            "axis": f["axis"], "direction": f["direction"],
            "total_rotation_deg": round(np.degrees(total), 1),
            "rotation_complete": bool(abs(abs(total) - 2 * np.pi) < np.radians(45)),
        })
    return out


def plot_track(traj: FreestyleTrajectory, path: np.ndarray | None, outfile,
               title: str = "Freestyle sequence") -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = np.arange(0.0, traj.duration, 0.01)
    ref = traj.pos(t)
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax, (i, j, xl, yl) in zip(axes, [(0, 1, "x [m]", "y [m]"),
                                         (0, 2, "x [m]", "z [m]")]):
        ax.plot(ref[:, i], ref[:, j], "k--", lw=1.2, label="reference")
        if path is not None:
            ax.plot(path[:, i], path[:, j], "tab:blue", lw=1.0, label="drone")
        for k, g in enumerate(traj.gates):
            half = g.HALF_OPENING
            if (i, j) == (0, 1):  # top view: opening extent along gate y-axis
                e = g.rotation[:, 1] * half
                ax.plot([g.center[0] - e[0], g.center[0] + e[0]],
                        [g.center[1] - e[1], g.center[1] + e[1]], "r-", lw=3)
                ax.annotate(f"G{k + 1}", g.center[:2], fontsize=9, color="r")
            else:  # side view: vertical opening extent
                ax.plot([g.center[0], g.center[0]],
                        [g.center[2] - half, g.center[2] + half], "r-", lw=3)
        for f in traj.flips:
            tw = np.linspace(f["t_rot_start"], f["t_rot_end"], 50)
            w = traj.pos(tw)
            ax.plot(w[:, i], w[:, j], "tab:orange", lw=3, alpha=0.6,
                    label="flip window" if ax is axes[0] else None)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.axis("equal")
        ax.grid(alpha=0.3)
    axes[0].legend(loc="best", fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(outfile, dpi=130)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None,
                        help="acro4 policy zip; omit for plan-only")
    parser.add_argument("--reason", required=True)
    parser.add_argument("--track", default="lsy-level2", choices=sorted(TRACKS))
    parser.add_argument("--cruise", type=float, default=1.5)
    args = parser.parse_args()

    traj = TRACKS[args.track](cruise=args.cruise)
    kind = "freestyle-eval" if args.model else "freestyle-plan"
    log = RunLogger(tag=f"{kind}-{args.track}",
                    reason=args.reason,
                    config={"model": args.model, "track": args.track,
                            "cruise": args.cruise, "duration": traj.duration,
                            "feasibility": dict(feasibility_report(traj))})
    rep = feasibility_report(traj)
    print(f"Logging to {log.dir}")
    print(f"plan feasibility: {rep}")
    title = f"Freestyle: {args.track} (cruise {args.cruise} m/s)"
    if not args.model:
        plot_track(traj, None, log.dir / "freestyle_plan.png", title=title)
        return

    from crazy_track.controllers.datt_acro import DATTAcroController
    from crazy_track.envs.rollout import make_sim, rollout

    sim = make_sim(control="force_torque")
    ctrl = DATTAcroController(args.model, control_freq=100)
    data = rollout(ctrl, traj, control_freq=100, sim=sim)
    pos, t = data["pos"], data["t"]
    gates = gate_crossing_metrics(pos, t, traj)
    flips = flip_rotation_metrics(data["quat"], t, traj)
    ref_dev = np.linalg.norm(pos - traj.pos(t), axis=1)
    metrics = {
        **tracking_metrics(pos, data["ref_pos"], t),
        "max_ref_dev": round(float(ref_dev.max()), 3),
        "min_z": round(float(pos[:, 2].min()), 3),
        "gates_passed": sum(g["passed"] for g in gates),
        "gates_clean": sum(g["clean"] for g in gates),
        "flips_complete": sum(f["rotation_complete"] for f in flips),
        # motion onset (post-lead-in) -> actual last-gate crossing; the lsy
        # leaderboard clock instead starts at episode start on the ground
        "race_time": round(gates[-1]["t_cross"] - traj.lead_in, 3)
                     if gates and all(g["passed"] for g in gates) else np.inf,
    }
    log.log_rollout("datt_acro", f"freestyle-{args.track}", data, metrics)
    plot_track(traj, pos, log.dir / "freestyle_rollout.png", title=title)
    for k, g in enumerate(gates):
        print(f"gate {k + 1}: {g}")
    for f in flips:
        print(f"flip: {f}")
    print(f"summary: {metrics}")


if __name__ == "__main__":
    main()
