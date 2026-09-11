"""Race a learned tracker on a TOGT-planned trajectory (lsy level-2 nominal gates).

    python -m crazy_track.eval.togt_race_eval --csv <togt.csv> --label togt \
        [--model results/<run>/datt_ppo_final.zip] --reason "..."

Plan-only (no --model): feasibility report + reference render + planner lap time.
With --model: zero-shot rollout, per-gate crossing offsets, race_time (motion
onset -> actual last-gate crossing), the same metric freestyle_eval reports for
the closed-form planner, so the two planners compare directly.
"""

from __future__ import annotations

import argparse

import numpy as np

from crazy_track.eval.freestyle_eval import gate_crossing_metrics, plot_track
from crazy_track.eval.runlog import RunLogger, tracking_metrics
from crazy_track.trajectories.freestyle import GRAVITY, feasibility_report, lsy_level2_race
from crazy_track.trajectories.sampled import SampledRaceTrajectory


def make_race_controller(spec: str):
    """Controller from a lissajous_benchmark spec (datt_acro:<zip>, mpc, mpc_l1, mppi_l1,
    pid, adrc, xadapt_adrc, ...) plus the sim control mode and rate it needs."""
    from crazy_track.eval.lissajous_benchmark import make_controller

    if spec.startswith("datt_acro"):
        mode, freq = "force_torque", 100
    elif spec.startswith("xadapt"):
        mode, freq = "rotor_vel", 500
    else:
        mode, freq = "attitude", 100
    return make_controller(spec), mode, freq


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="TOGT-Planner trajectory CSV")
    parser.add_argument("--label", default="togt", help="planner variant tag (togt|aos|...)")
    parser.add_argument("--model", default=None,
                        help="acro4 policy zip (shorthand for --controller datt_acro:<zip>)")
    parser.add_argument("--controller", default=None,
                        help="any lissajous_benchmark spec: mpc, mpc_offsetfree, mpc_l1, mppi_l1, pid, ...")
    parser.add_argument("--reason", required=True)
    parser.add_argument("--time-scale", type=float, default=1.0,
                        help="stretch the plan in time (same path, v/s, a/s^2)")
    args = parser.parse_args()
    spec = args.controller or (f"datt_acro:{args.model}" if args.model else None)

    ref_track = lsy_level2_race()          # same gates / obstacles / start / end
    traj = SampledRaceTrajectory(args.csv, gates=ref_track.gates,
                                 obstacles=ref_track.obstacles, time_scale=args.time_scale)
    rep = feasibility_report(traj)
    t_plan = np.arange(traj.lead_in, traj.t_end, 0.002)
    thrust = np.linalg.norm(traj.acc(t_plan) + np.array([0.0, 0.0, GRAVITY]), axis=1)
    plan = {
        "plan_lap_s": round(traj.gate_times[-1] - traj.lead_in, 3),   # onset -> last gate
        "plan_total_s": round(traj.plan_duration, 3),                 # incl. stop at the end
        "plan_vmax": round(float(np.linalg.norm(traj.vel(t_plan), axis=1).max()), 2),
        "plan_thrust_max": round(float(thrust.max()), 2),
        "plan_thrust_min": round(float(thrust.min()), 2),
        "time_scale": args.time_scale,
    }
    kind = "togt-race-eval" if spec else "togt-race-plan"
    log = RunLogger(tag=f"{kind}-{args.label}", reason=args.reason,
                    config={"csv": args.csv, "model": args.model, "controller": spec, "label": args.label,
                            "plan": plan, "feasibility": dict(rep)})
    print(f"Logging to {log.dir}")
    print(f"plan: {plan}")
    print(f"feasibility: {rep}")
    title = f"TOGT ({args.label}) on lsy level-2: planned lap {plan['plan_lap_s']} s"
    if spec is None:
        plot_track(traj, None, log.dir / "togt_plan.png", title=title)
        return

    from crazy_track.envs.rollout import make_sim, rollout

    ctrl, mode, freq = make_race_controller(spec)
    sim = make_sim(control=mode)
    data = rollout(ctrl, traj, control_freq=freq, sim=sim)
    pos, t = data["pos"], data["t"]
    gates = gate_crossing_metrics(pos, t, traj)
    ref_dev = np.linalg.norm(pos - traj.pos(t), axis=1)
    metrics = {
        **tracking_metrics(pos, data["ref_pos"], t),
        "max_ref_dev": round(float(ref_dev.max()), 3),
        "min_z": round(float(pos[:, 2].min()), 3),
        "gates_passed": sum(g["passed"] for g in gates),
        "gates_clean": sum(g["clean"] for g in gates),
        "plan_lap_s": plan["plan_lap_s"],
        "race_time": round(gates[-1]["t_cross"] - traj.lead_in, 3)
                     if all(g["passed"] for g in gates) else np.inf,
    }
    log.log_rollout(spec.split(":")[0], f"togt-{args.label}", data, metrics)
    plot_track(traj, pos, log.dir / "togt_rollout.png", title=title)
    for k, g in enumerate(gates):
        print(f"gate {k + 1}: {g}")
    print(f"summary: {metrics}")


if __name__ == "__main__":
    main()
