"""Flip-maneuver evaluation: rotation completeness, position deviation, recovery.

Usage: python -m crazy_track.eval.flip_eval --model <path> --reason "..."
       [--ballistic]  # evaluate on the feasible ballistic reference (paper 2)
"""

from __future__ import annotations

import argparse

import numpy as np
from scipy.spatial.transform import Rotation as R

from crazy_track.controllers.datt_acro import DATTAcroController
from crazy_track.envs.rollout import make_sim, rollout
from crazy_track.eval.runlog import RunLogger
from crazy_track.trajectories import BallisticFlipTrajectory, FlipTrajectory

HOVER = np.array([0.0, 0.0, 2.5])  # acro2.2 training altitude range midpoint
HOVER_BALLISTIC = np.array([0.0, 0.0, 2.0])  # arc stays ABOVE the hover point


def flip_metrics(data: dict, traj, hover: np.ndarray, t_settle: float) -> dict:
    quat, pos, t = data["quat"], data["pos"], data["t"]
    # Total rotation about the flip axis: sum of per-step relative rotations.
    rots = R.from_quat(quat)
    rel = (rots[:-1].inv() * rots[1:]).as_rotvec()
    total_rot = float(np.sum(rel[:, traj.axis]))
    ref_dev = np.linalg.norm(pos - traj.pos(t), axis=1)  # vs the actual reference
    after = t >= t_settle
    return {
        "total_rotation_deg": round(np.degrees(total_rot), 1),
        "rotation_complete": bool(abs(abs(total_rot) - 2 * np.pi) < np.radians(45)),
        "max_ref_dev": round(float(ref_dev.max()), 3),
        "max_hover_dev": round(float(np.linalg.norm(pos - hover, axis=1).max()), 3),
        "min_z": round(float(pos[:, 2].min()), 3),
        "recovery_err": round(float(np.linalg.norm(pos[after] - hover, axis=1).mean())
                              if after.any() else np.nan, 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--reason", required=True)
    parser.add_argument("--ballistic", action="store_true",
                        help="feasible boost/ballistic/brake reference (paper 2)")
    args = parser.parse_args()

    hover = HOVER_BALLISTIC if args.ballistic else HOVER
    log = RunLogger(tag="flip-eval-ballistic" if args.ballistic else "flip-eval",
                    reason=args.reason,
                    config={"model": args.model, "hover_z": float(hover[2]),
                            "ballistic": args.ballistic})
    print(f"Logging to {log.dir}")
    sim = make_sim(control="force_torque")
    for axis, name in ((0, "roll"), (1, "pitch")):
        for direction in (1, -1):
            if args.ballistic:
                traj = BallisticFlipTrajectory(hover=hover, t0=2.0, Tb=0.7, axis=axis,
                                               direction=direction, a_boost=7.0,
                                               duration=6.0)
                t_settle = traj.t_end + 1.5
            else:
                traj = FlipTrajectory(hover=hover, t0=2.0, Tf=0.5, axis=axis,
                                      direction=direction, duration=6.0)
                t_settle = traj.t0 + traj.Tf + 1.5
            ctrl = DATTAcroController(args.model, control_freq=100)
            data = rollout(ctrl, traj, control_freq=100, sim=sim)
            m = flip_metrics(data, traj, hover, t_settle)
            log.log_rollout("datt_acro", f"{name}{'+' if direction > 0 else '-'}", data, m)
            print(f"flip {name}{'+' if direction > 0 else '-'}: {m}")


if __name__ == "__main__":
    main()
