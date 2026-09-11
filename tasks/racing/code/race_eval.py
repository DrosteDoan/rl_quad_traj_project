"""Race ONE (plan, controller) pair in the crazyflow harness, on either clock — Lesson 6 §3–§5.

    python tasks/racing/code/race_eval.py --plan closed-form --cruise 3.0 --controller mpc --reason "..."
    python tasks/racing/code/race_eval.py --plan tasks/racing/plans/tube_f0.95.csv --time-scale 1.05 \
        --controller datt:tasks/racing/crazy_track/results/<run>/datt_ppo_final.zip --reason "..."

Runs in the MAIN venv. `--controller` takes any spec the vendored benchmark knows
(`crazy_track/eval/lissajous_benchmark.py`, make_controller): mpc, mpc_offsetfree,
mpc_l1, mppi_l1, pid, adrc, datt:<zip>, datt_acro:<zip>. The sim interface is chosen
per family, exactly as the parent project's race evaluators do.

THE TWO CLOCKS (the one thing to get right before comparing anything):

  --start hover   (default)  the crazy_track BENCHMARK clock. The drone starts at rest at
                  the plan's first point IN THE AIR, the reference holds still for a 1.5 s
                  lead-in, and race_time = motion onset -> last-gate crossing. Every number
                  in the parent project's reports (4.464 s, 4.085 s, 3.592 s) is on this clock.
  --start ground  the LEADERBOARD clock. The drone starts ON THE GROUND at the race start
                  pose (-1.5, 0.75, 0.01); a rest-to-rest takeoff of --takeoff-t seconds
                  climbs to the plan's first point; race_time = t = 0 -> last-gate crossing.
                  Same plan, same controller, one more term in the clock (Lesson 3 §3:
                  takeoff is inside your lap time). It still is NOT the leaderboard: one
                  deterministic episode, no randomisation, no obstacle contacts — Lesson 6
                  §5 races the same stack in the real environment.

Every run writes results/<stamp>_race-eval-<plan>-<start>-<controller>/ inside the
vendored crazy_track (metadata.yaml with your --reason, summary.csv, the rollout .npz,
rollout.png). `race_table.py` turns those directories into the comparison table.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from race_refs import (  # noqa: E402
    RACE_START,
    GroundStartTrajectory,
    crossing_angles,
    obstacle_clearance,
)

from crazy_track.eval.freestyle_eval import gate_crossing_metrics, plot_track  # noqa: E402
from crazy_track.eval.lissajous_benchmark import make_controller  # noqa: E402
from crazy_track.eval.runlog import RunLogger, tracking_metrics  # noqa: E402
from crazy_track.trajectories.freestyle import GRAVITY, feasibility_report, lsy_level2_race  # noqa: E402
from crazy_track.trajectories.sampled import SampledRaceTrajectory  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
DISTURBANCES = ("none", "wind_const", "wind_gust", "payload", "ground")


def make_race_controller(spec: str, seed: int = 0):
    """The vendored benchmark's controller specs + the sim interface each family needs
    (as crazy_track.eval.togt_race_eval does), with the seed passed through (MPPI sampling)."""
    if spec.startswith("datt_acro"):
        mode, freq = "force_torque", 100
    elif spec.startswith("xadapt"):
        mode, freq = "rotor_vel", 500
    else:
        mode, freq = "attitude", 100
    return make_controller(spec, seed=seed), mode, freq


def build_conditions(args, ctrl_freq: int):
    """crazy_track's disturbance and sensor models, exactly as lissajous_benchmark builds them
    (Lesson 7): a world-frame force on the CoM, and a Lighthouse measurement model between the
    true state and what the controller sees. Metrics always use the true state."""
    disturbance = None
    if args.disturbance != "none":
        from crazy_track.disturbances import SCENARIOS

        cls = SCENARIOS[args.disturbance]
        disturbance = cls(seed=args.seed) if args.disturbance == "wind_gust" else cls()
    sensor = None
    if args.sensor == "lighthouse":
        from crazy_track.sensors import LighthouseSensor

        s = args.sensor_scale
        sensor = LighthouseSensor(control_freq=ctrl_freq, latency_steps=max(1, ctrl_freq // 100),
                                  jitter_std=0.0007 * s, bias_std=0.015 * s, vel_std=0.03 * s,
                                  att_std_deg=0.5 * s, seed=args.seed)
    parts = [] if args.disturbance == "none" else [args.disturbance]
    if args.sensor != "none":
        parts.append(args.sensor if args.sensor_scale == 1.0 else f"{args.sensor}x{args.sensor_scale:g}")
    cond = "+".join(parts) or "nominal"
    return disturbance, sensor, cond


def build_plan(args):
    """The hover-start plan and a short name for it."""
    if args.plan == "closed-form":
        return lsy_level2_race(cruise=args.cruise), f"closed-form-c{args.cruise:g}"
    ref = lsy_level2_race()
    csv = Path(args.plan)
    plan = SampledRaceTrajectory(str(csv), gates=ref.gates, obstacles=ref.obstacles,
                                 time_scale=args.time_scale)
    return plan, f"{csv.stem}-x{args.time_scale:.2f}"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--plan", default="closed-form",
                   help="'closed-form' (lsy_level2_race at --cruise) or a TOGT plan CSV")
    p.add_argument("--cruise", type=float, default=3.0, help="closed-form plan only")
    p.add_argument("--time-scale", type=float, default=1.0,
                   help="CSV plans only: stretch the plan in time (same path, v/s, a/s^2)")
    p.add_argument("--controller", default="mpc",
                   help="mpc | mpc_offsetfree | mpc_l1 | mppi_l1 | pid | adrc | datt:<zip> | datt_acro:<zip>")
    p.add_argument("--label", default=None,
                   help="name for this controller in the table (default: the spec before ':')")
    p.add_argument("--start", choices=["hover", "ground"], default="hover",
                   help="hover = benchmark clock (parent project); ground = leaderboard clock")
    p.add_argument("--takeoff-t", type=float, default=1.5,
                   help="--start ground: seconds for the rest-to-rest climb to the plan's first point")
    p.add_argument("--disturbance", choices=DISTURBANCES, default="none",
                   help="crazy_track disturbance model applied during the lap (Lesson 7)")
    p.add_argument("--sensor", choices=["none", "lighthouse"], default="none",
                   help="what the controller sees: the true state, or a Lighthouse measurement model")
    p.add_argument("--sensor-scale", type=float, default=1.0,
                   help="scale every Lighthouse error term together (1 = the literature model)")
    p.add_argument("--seed", type=int, default=0,
                   help="seeds the gust realisation, the sensor bias and MPPI's sampling")
    p.add_argument("--reason", required=True, help="why this run is being made (logged)")
    p.add_argument("--no-plot", action="store_true")
    args = p.parse_args()

    plan, plan_name = build_plan(args)
    if args.start == "ground":
        traj = GroundStartTrajectory(plan, RACE_START, takeoff_t=args.takeoff_t)
    else:
        traj = plan
    label = args.label or args.controller.split(":")[0]
    ctrl, mode, freq = make_race_controller(args.controller, seed=args.seed)
    disturbance, sensor, cond = build_conditions(args, freq)

    # ---- the plan, before anything flies ------------------------------------
    rep = feasibility_report(traj)
    t_ref = np.arange(traj.lead_in, traj.gate_times[-1], 0.002)
    thrust = np.linalg.norm(traj.acc(t_ref) + np.array([0.0, 0.0, GRAVITY]), axis=1)
    angles = crossing_angles(traj)
    plan_info = {
        "plan_lap_s": round(float(traj.gate_times[-1] - traj.lead_in), 3),
        "plan_vmax": round(float(np.linalg.norm(traj.vel(t_ref), axis=1).max()), 2),
        "plan_thrust_max": round(float(thrust.max()), 2),
        "plan_gate1_s": round(float(traj.gate_times[0] - traj.lead_in), 3),
        "crossing_angles_deg": [round(a["angle_deg"], 1) for a in angles],
    }
    log = RunLogger(
        tag=f"race-eval-{plan_name}-{args.start}-{cond}-{label}", reason=args.reason,
        config={"plan": args.plan, "plan_name": plan_name, "cruise": args.cruise,
                "time_scale": args.time_scale, "start": args.start,
                "takeoff_t": args.takeoff_t if args.start == "ground" else 0.0,
                "controller": args.controller, "label": label, "plan_info": plan_info,
                "disturbance": args.disturbance, "sensor": args.sensor,
                "sensor_scale": args.sensor_scale, "seed": args.seed, "cond": cond,
                "feasibility": {k: (v if not isinstance(v, np.generic) else float(v))
                                for k, v in rep.items()}})
    print(f"Logging to {log.dir}")
    print(f"condition: {cond}" + (f" (seed {args.seed})" if cond != "nominal" else ""))
    print(f"plan {plan_name} on the {args.start.upper()} clock: lap {plan_info['plan_lap_s']} s, "
          f"gate 1 at {plan_info['plan_gate1_s']} s, vmax {plan_info['plan_vmax']} m/s, "
          f"thrust max {plan_info['plan_thrust_max']} / {rep['thrust_acc_limit']:.1f} m/s^2, "
          f"gate_crossings_ok {rep['gate_crossings_ok']}, min_z {rep['min_z']:.2f}")

    # ---- fly ----------------------------------------------------------------
    from crazy_track.envs.rollout import make_sim, rollout

    sim = make_sim(control=mode)
    t0 = time.time()
    data = rollout(ctrl, traj, control_freq=freq, sim=sim, disturbance=disturbance, sensor=sensor)
    wall = time.time() - t0
    pos, t = data["pos"], data["t"]
    gates = gate_crossing_metrics(pos, t, traj)
    ref_dev = np.linalg.norm(pos - traj.pos(t), axis=1)
    clear = obstacle_clearance(pos, traj.obstacles)
    passed_all = all(g["passed"] for g in gates)
    metrics = {
        **tracking_metrics(pos, data["ref_pos"], t),
        "max_ref_dev": round(float(ref_dev.max()), 3),
        "min_z": round(float(pos[:, 2].min()), 3),
        "gates_passed": sum(g["passed"] for g in gates),
        "gates_clean": sum(g["clean"] for g in gates),
        "obstacle_min_clearance": round(float(min(clear)), 3),
        "plan_lap_s": plan_info["plan_lap_s"],
        # motion onset (hover clock) or t = 0 (ground clock) -> actual last-gate crossing
        "race_time": round(gates[-1]["t_cross"] - traj.lead_in, 3) if passed_all else np.inf,
        "t_gate1": round(gates[0]["t_cross"] - traj.lead_in, 3) if gates[0]["passed"] else np.inf,
        "wall_time_s": round(wall, 1),
    }
    log.log_rollout(label, plan_name, data, metrics)
    if not args.no_plot:
        plot_track(traj, pos, log.dir / "rollout.png",
                   title=f"{label} on {plan_name} ({args.start} start): "
                         f"{metrics['gates_passed']}/4 gates, race_time {metrics['race_time']}")
    for k, g in enumerate(gates):
        print(f"gate {k + 1}: offset {g['offset']:.3f} m {'PASS' if g['passed'] else 'MISS'}"
              f"  at t={g['t_cross']}  (plan {g['t_gate']}; crossing angle {angles[k]['angle_deg']:.0f} deg)")
    print(f"obstacle clearance (pole surface): " + " ".join(f"{c:.2f}" for c in clear) + " m"
          + ("   [!] < 0.10 m -- the race would likely score a contact" if min(clear) < 0.10 else ""))
    print(f"RESULT plan={plan_name} start={args.start} cond={cond} seed={args.seed} controller={label} "
          f"gates={metrics['gates_passed']}/4 race_time={metrics['race_time']} "
          f"t_gate1={metrics['t_gate1']} max_dev={metrics['max_ref_dev']} "
          f"rmse={metrics['rmse_3d']:.3f} wall={wall:.0f}s")


if __name__ == "__main__":
    main()
