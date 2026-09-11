"""Figure-5 Lissajous benchmark runner (arXiv:2311.13081, Table III style).

Usage (inside the WSL venv):
    python -m crazy_track.eval.lissajous_benchmark --controllers pid --reason "why this run"
"""

from __future__ import annotations

import argparse
import time

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from crazy_track.envs.rollout import make_sim, rollout
from crazy_track.eval.runlog import RunLogger, tracking_metrics
from crazy_track.trajectories import LissajousTrajectory

N_CYCLES = 2
CONTROL_FREQ = 100


def make_controller(name: str, seed: int = 0):
    if name == "pid":
        from crazy_track.controllers.pid import PIDController
        return PIDController(control_freq=CONTROL_FREQ)
    if name == "adrc":
        from crazy_track.controllers.adrc import ADRCController
        return ADRCController(control_freq=CONTROL_FREQ)
    if name == "mppi_l1":
        from crazy_track.controllers.mppi_l1 import MPPIL1Controller
        return MPPIL1Controller(control_freq=CONTROL_FREQ, seed=seed)
    if name == "adrc_adaptive":
        from crazy_track.controllers.adrc import ADRCController
        return ADRCController(control_freq=CONTROL_FREQ, estimator="adaptive")
    if name.startswith("mpc_offsetfree"):  # optional _w<N> suffix: ESO bandwidth
        import re
        from crazy_track.controllers.mpc import MPCController
        m = re.fullmatch(r"mpc_offsetfree(?:_w(\d+))?", name)
        if m is None:
            raise ValueError(f"Unknown controller: {name}")
        return MPCController(control_freq=CONTROL_FREQ, offset_free=True,
                             eso_w=float(m.group(1)) if m.group(1) else 7.0)
    if name == "mpc":
        from crazy_track.controllers.mpc import MPCController
        return MPCController(control_freq=CONTROL_FREQ)
    if name.startswith("mpc_l1"):  # optional _c<Hz> suffix: L1 low-pass cutoff (default 4 Hz)
        import re
        from crazy_track.controllers.mpc import MPCController
        m = re.fullmatch(r"mpc_l1(?:_c(\d+))?", name)
        if m is None:
            raise ValueError(f"Unknown controller: {name}")
        return MPCController(control_freq=CONTROL_FREQ, disturbance="l1",
                             l1_cutoff_hz=float(m.group(1)) if m.group(1) else 4.0)
    if name == "xadapt":
        from crazy_track.controllers.xadapt import XAdaptPIDController
        return XAdaptPIDController(control_freq=500)
    if name == "xadapt_adrc":
        from crazy_track.controllers.xadapt import XAdaptADRCController
        return XAdaptADRCController(control_freq=500)
    if name.startswith("datt_acro:"):
        from crazy_track.controllers.datt_acro import DATTAcroController
        return DATTAcroController(name.split(":", 1)[1], control_freq=CONTROL_FREQ)
    if name.startswith("datt_rnn:"):  # v7: recurrent (GRU) actor
        from crazy_track.controllers.datt import DATTPolicyController
        return DATTPolicyController(name.split(":", 1)[1], control_freq=CONTROL_FREQ,
                                    recurrent=True)
    if name.startswith("datt:"):
        from crazy_track.controllers.datt import DATTPolicyController
        return DATTPolicyController(name.split(":", 1)[1], control_freq=CONTROL_FREQ)
    raise ValueError(f"Unknown controller: {name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--controllers", nargs="+", default=["pid"])
    parser.add_argument("--speeds", nargs="+", default=["slow", "normal", "fast"])
    parser.add_argument("--reason", required=True, help="why this run is being made (logged)")
    parser.add_argument("--tag", default="lissajous")
    parser.add_argument("--disturbance", default="none",
                        choices=["none", "wind_const", "wind_gust", "ground", "payload"])
    parser.add_argument("--sensor", default="none", choices=["none", "lighthouse"])
    parser.add_argument("--sensor-scale", type=float, default=1.0,
                        help="scale every Lighthouse error term together (bias, jitter, "
                             "velocity and attitude noise). 1.0 = the literature-grounded "
                             "model; <1 = better sensing, >1 = worse. The gyro is an "
                             "onboard IMU and is deliberately left unscaled.")
    parser.add_argument("--vertical", action="store_true",
                        help="figure-8 in the x-z plane (acro benchmark)")
    parser.add_argument("--seed", type=int, default=0,
                        help="seeds sensor bias, gust realization, MPPI sampling")
    args = parser.parse_args()

    from crazy_track.disturbances import SCENARIOS
    disturbance = None
    if args.disturbance != "none":
        cls = SCENARIOS[args.disturbance]
        disturbance = cls(seed=args.seed) if args.disturbance == "wind_gust" else cls()
    traj_z = 0.08 if args.disturbance == "ground" else 1.0  # IGE needs low altitude
    if args.tag == "lissajous":
        if args.disturbance != "none":
            args.tag = f"lissajous-{args.disturbance}"
        if args.sensor != "none":
            args.tag += f"-{args.sensor}"

    log = RunLogger(
        tag=args.tag, reason=args.reason,
        config={"controllers": args.controllers, "speeds": args.speeds,
                "n_cycles": N_CYCLES, "control_freq": CONTROL_FREQ,
                "drone": "cf21B_500", "dynamics": "first_principles",
                "disturbance": args.disturbance, "sensor": args.sensor,
                "sensor_scale": args.sensor_scale, "seed": args.seed, "traj_z": traj_z},
    )
    print(f"Logging to {log.dir}")
    acro = [c.startswith("datt_acro") for c in args.controllers]
    xadapt = [c.startswith("xadapt") for c in args.controllers]
    if (any(acro) and not all(acro)) or (any(xadapt) and not all(xadapt)):
        raise SystemExit("datt_acro (force_torque) / xadapt (rotor_vel) cannot be mixed "
                         "with attitude-mode controllers in one run — invoke separately.")
    mode = "force_torque" if any(acro) else "rotor_vel" if any(xadapt) else "attitude"
    ctrl_freq = 500 if any(xadapt) else CONTROL_FREQ  # xadapt runs at its training rate
    sensor = None
    if args.sensor == "lighthouse":
        from crazy_track.sensors import LighthouseSensor
        # keep latency time-consistent (~10 ms) across control rates
        s = args.sensor_scale  # one knob on overall measurement quality
        sensor = LighthouseSensor(control_freq=ctrl_freq,
                                  latency_steps=max(1, ctrl_freq // 100),
                                  jitter_std=0.0007 * s, bias_std=0.015 * s,
                                  vel_std=0.03 * s, att_std_deg=0.5 * s,
                                  seed=args.seed)
    sim = make_sim(control=mode)

    fig, axes = plt.subplots(len(args.controllers), len(args.speeds),
                             figsize=(5 * len(args.speeds), 4.4 * len(args.controllers)),
                             squeeze=False)
    seen: dict[str, int] = {}
    for ci, cspec in enumerate(args.controllers):
        cname = cspec.split(":")[0]  # file/label-safe name (model paths follow the colon)
        seen[cname] = seen.get(cname, 0) + 1
        if seen[cname] > 1:  # avoid npz/row collisions when comparing same-type models
            cname = f"{cname}{seen[cname]}"
        for si, speed in enumerate(args.speeds):
            traj = LissajousTrajectory.from_speed(speed, n_cycles=N_CYCLES, z=traj_z,
                                                  vertical=args.vertical)
            ctrl = make_controller(cspec, seed=args.seed)
            t0 = time.time()
            data = rollout(ctrl, traj, control_freq=ctrl_freq, sim=sim,
                           disturbance=disturbance, sensor=sensor)
            wall = time.time() - t0
            metrics = tracking_metrics(data["pos"], data["ref_pos"], data["t"])
            metrics["wall_time_s"] = round(wall, 1)
            log.log_rollout(cname, speed, data, metrics)
            print(f"{cname:10s} {speed:7s} rmse_3d={metrics['rmse_3d']:.3f} m "
                  f"rmse_xy={metrics['rmse_xy']:.3f} m max={metrics['max_err']:.3f} m "
                  f"({wall:.0f}s wall)")

            # vertical fig-8s live in the x-z plane; horizontal ones in x-y
            k, axis_name = (2, "z") if args.vertical else (1, "y")
            ax = axes[ci][si]
            ax.plot(data["ref_pos"][:, 0], data["ref_pos"][:, k], "k-", lw=1, label="reference")
            ax.plot(data["pos"][:, 0], data["pos"][:, k], lw=1, label=cname)
            ax.set(title=f"{cname} | {speed} (T={LissajousTrajectory.SPEEDS[speed]}s) "
                         f"RMSE={metrics['rmse_3d']:.3f}m",
                   xlabel="x [m]", ylabel=f"{axis_name} [m]", aspect="equal")
            ax.legend(fontsize=8)
    fig.tight_layout()
    plot_name = "tracking_xz.png" if args.vertical else "tracking_xy.png"
    fig.savefig(log.dir / plot_name, dpi=150)
    print(f"Saved plot to {log.dir / plot_name}")


if __name__ == "__main__":
    main()
