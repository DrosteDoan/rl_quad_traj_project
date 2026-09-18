"""Race N episodes and log what `sim.py` does not — Lesson 8 §5.

    cp tasks/racing/code/race_bridge_mpc.py repos/lsy_drone_racing/lsy_drone_racing/control/
    RACE_PLAN=tasks/racing/plans/poles_f0.85.csv RACE_TAKEOFF_T=1.5 RACE_CONTROLLER=mpc \
      /opt/venvs/race/bin/python tasks/racing/code/race_runner.py --config level1.toml \
      --n 20 --label adopted_mpc_L1 --out-dir tasks/racing/figures/lesson8

Runs in the RACE venv, from the repo root. It reproduces lsy's `scripts/sim.py` `simulate()`
loop exactly — the same environment construction, the same step loop, the same termination —
for any bridge in `repos/lsy_drone_racing/lsy_drone_racing/control/`, and records per episode
what `sim.py` never prints:

  * the SAMPLED drone mass and inertia. Level 1 randomises them at reset, and the sampled mass
    lives in `env.unwrapped.data.sim_data.params.mass` — the `Sim` object's own copy stays
    nominal, and the controller never sees either. Lesson 8 §1 could only read the mass through
    proxies; this is the mass itself, and §5's whole table rests on it;
  * the start pose the race drew, the outcome, and a failure verdict (FRAME_Gk / POLEk / OOB /
    OK) from the last flown sample against the contact model of §1;
  * the gate crossings with their lag and in-plane offset against the BRIDGE'S OWN reference,
    the lift-off time, the mean height offset in cruise, the maximum deviation and the pole-4
    clearance;
  * the solve time, the ipopt failures and the thrust scale k of the mass adaptation.

It takes the same `RACE_*` environment as the bridge (`RACE_CONTROLLER`, `RACE_PLAN`,
`RACE_TAKEOFF_T`, `RACE_START_BLEND`, `RACE_TIME_SCALE`, `RACE_START`, ... — see
`race_bridge_mpc.py`), with two conveniences: a relative `RACE_PLAN` is resolved against the
repo root, and `RACE_LOG_DIR` defaults to `<out-dir>/<label>_flown` so the flown paths land
next to the run. `RACE_LSY_DIR` (default `/workspace/repos/lsy_drone_racing`) says where the
race lives.

Per episode it prints the three lines `sim.py` prints (`Flight time (s)` / `Finished` /
`Gates passed`, so the Lesson-6 and -7 greps keep working) plus one `EP` line; at the end,

    <label>: k/n laps, mean lap X s (± std), light half a/b, heavy half c/d

on the leaderboard clock, with the halves split at the nominal mass (0.04338 kg). At Level 0
every mass ratio is 1.000 and the halves degenerate — that is the point: run the same cell at
both levels and the difference is the randomisation.

Outputs in `--out-dir` (default `tasks/racing/figures/lesson8`): `<label>.csv` (one row per
episode, every field above), `<label>.log` (everything printed) and
`<label>_flown/flown_epNN.csv` (the flown path at 50 Hz, for `plot_trajectory.py --flown`).

A 20-episode MPC cell takes 75-105 s on an idle machine, plus ~30 s to build the environment.
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]           # tasks/racing/code -> rl_quad_control/
CODE = Path(__file__).resolve().parent
M_NOMINAL = 0.04338          # crazyflow cf21B_500 params.mass (the nominal drone)
POLE_R = 0.015               # lsy assets/obstacle.xml: a capsule of radius 0.015 m
LIFTOFF_DZ = 0.03


class _Tee:
    """Write everything the run prints to the terminal AND to <label>.log."""

    def __init__(self, stream, path: Path):
        self.stream = stream
        self.fh = open(path, "w", encoding="utf-8")

    def write(self, s):
        self.stream.write(s)
        self.fh.write(s)
        return len(s)

    def flush(self):
        self.stream.flush()
        self.fh.flush()

    def close(self):
        self.fh.close()


def sampled_mass(env):
    """The mass the physics steps with: the env's own sim_data, which the reset pipeline
    randomises. `env.unwrapped.sim.data` stays NOMINAL, so reading it would silently report
    0.04338 for every Level-1 episode (Lesson 8 §5)."""
    u = env.unwrapped
    for getter in (lambda: u.data.sim_data.params, lambda: u.sim.data.params):
        try:
            p = getter()
            m = float(np.asarray(p.mass).ravel()[0])
            J = np.asarray(p.J).reshape(-1, 3, 3)[0].diagonal().tolist()
            return m, J
        except Exception:  # noqa: BLE001  -- the attribute path moves between crazyflow versions
            continue
    return float("nan"), [float("nan")] * 3


def _quat_to_rpy(q):
    from scipy.spatial.transform import Rotation as R

    return R.from_quat(np.asarray(q, dtype=np.float64).reshape(4)).as_euler("xyz")


def analyse_flown(flown, traj, gates, obstacles, n_gates_passed, takeoff_t):
    """Features of one flown path (a list of (t, x, y, z)) against the reference `traj`."""
    A = np.asarray(flown, dtype=np.float64)
    if len(A) < 3:
        return {"verdict": "?short"}
    t, P = A[:, 0], A[:, 1:4]
    R = traj.pos(t)
    dev = np.linalg.norm(P - R, axis=1)
    dz = P[:, 2] - R[:, 2]
    # lift-off: the first sample 3 cm above the start height
    z0 = P[0, 2]
    idx = np.flatnonzero(P[:, 2] > z0 + LIFTOFF_DZ)
    t_lift = float(t[idx[0]]) if len(idx) else float("nan")
    # cruise window: after the takeoff / hold + 0.3 s, to the end
    cr = t >= (takeoff_t + 0.3)
    dz_cruise = float(dz[cr].mean()) if cr.any() else float("nan")
    dz_min = float(dz[cr].min()) if cr.any() else float("nan")
    # gate crossings in order: the first crossing of each plane inside the 0.45 m box, after the
    # previous gate's
    xs = {}
    t_prev = -1.0
    for j, g in enumerate(gates):
        loc = g.to_gate_frame(P)
        x = loc[:, 0]
        ii = np.flatnonzero((np.sign(x[1:]) != np.sign(x[:-1])) & (t[1:] > t_prev))
        for i in ii:
            w = x[i] / (x[i] - x[i + 1])
            yz = loc[i, 1:] + w * (loc[i + 1, 1:] - loc[i, 1:])
            if np.abs(yz).max() < 0.45:
                tc = float(t[i] + w * (t[i + 1] - t[i]))
                ryz = g.to_gate_frame(traj.pos(tc)[None])[0, 1:]
                lag = tc - float(traj.gate_times[j]) if j < len(traj.gate_times) else float("nan")
                xs[j + 1] = dict(t=tc, y=float(yz[0]), z=float(yz[1]), ry=float(ryz[0]),
                                 rz=float(ryz[1]), lag=lag)
                t_prev = tc
                break
    # the terminal feature: WHAT the episode ended on. lsy does not log the contact partner, so
    # this is a heuristic on the last flown sample -- a '?' suffix means "probably" (§7).
    pe = P[-1]
    v_end = float(np.linalg.norm(np.diff(P[-3:], axis=0), axis=1).mean() / np.diff(t[-3:]).mean())
    best = None
    for j, ob in enumerate(obstacles):
        dd = np.linalg.norm(P[:, :2] - np.asarray(ob[:2]), axis=1) - POLE_R
        dd = np.where(P[:, 2] < ob[2], dd, np.inf)
        i = int(dd.argmin())
        cand = (float(dd[-1]), j + 1, float(dd[i]), float(t[i]))
        if best is None or cand[0] < best[0]:
            best = cand
    d_end_pole, jpole, d_min_pole, t_min_pole = best
    gbest = None
    for j, g in enumerate(gates):
        loc = g.to_gate_frame(P[-3:])
        plane = float(abs(loc[-1, 0]))
        off = float(np.abs(loc[-1, 1:]).max())
        cand = (plane, j + 1, off, float(loc[-1, 1]), float(loc[-1, 2]))
        if gbest is None or cand[0] < gbest[0]:
            gbest = cand
    plane_d, jg, off, off_y, off_z = gbest
    oob = (abs(pe[0]) > 2.5) or (abs(pe[1]) > 1.5) or (pe[2] > 2.0) or (pe[2] < -0.001)
    if n_gates_passed >= len(gates):
        verdict = "OK"
    elif d_end_pole < 0.12:
        verdict = f"POLE{jpole}"
    elif plane_d < 0.20 and 0.10 <= off < 0.45:
        verdict = f"FRAME_G{jg}"
    elif oob:
        verdict = "OOB"
    elif plane_d < 0.30 and off > 0.08:
        verdict = f"FRAME_G{jg}?"
    elif d_end_pole < 0.20:
        verdict = f"POLE{jpole}?"
    elif pe[2] < 0.03 and t[-1] < 1.0:
        verdict = "FLOOR?"
    else:
        verdict = "?"
    # pole-4 minimum clearance (the tight one on this track)
    p4_min = float("nan")
    if len(obstacles) > 3:
        p4 = obstacles[3]
        d4 = np.linalg.norm(P[:, :2] - np.asarray(p4[:2]), axis=1) - POLE_R
        p4_min = float(d4.min())
    return dict(verdict=verdict, t_lift=t_lift, dz_cruise=dz_cruise, dz_min=dz_min,
                dev_max=float(dev.max()), dev_end=float(dev[-1]), v_end=v_end,
                end_pole=jpole, d_end_pole=d_end_pole, d_min_pole=d_min_pole,
                end_gate=jg, plane_d=plane_d, off=off, off_y=off_y, off_z=off_z,
                p4_min=p4_min, crossings=xs)


FIELDS = ["ep", "mass", "mass_ratio", "Jx", "Jy", "Jz", "x0", "y0", "z0", "roll0", "pitch0", "yaw0",
          "gates", "flight_time", "finished", "terminated", "truncated", "ctrl_finished", "steps",
          "x_end", "y_end", "z_end", "verdict", "solve_mean_ms", "solve_max_ms", "ipopt_fail",
          "k_end", "k_at_2s", "t_lift", "dz_cruise", "dz_min", "dev_max", "p4_min",
          "g1_t", "g1_lag", "g1_y", "g1_z", "g1_ry", "g1_rz",
          "g2_t", "g2_lag", "g2_y", "g2_z", "g2_ry", "g2_rz",
          "g3_t", "g3_lag", "g3_y", "g3_z", "g3_ry", "g3_rz",
          "g4_t", "g4_lag", "g4_y", "g4_z", "g4_ry", "g4_rz",
          "end_gate", "plane_d", "off_y", "off_z", "end_pole", "d_end_pole", "d_min_pole", "v_end"]


def run(args) -> None:
    import gymnasium
    from gymnasium.wrappers.jax_to_numpy import JaxToNumpy
    from lsy_drone_racing.utils import load_config, load_controller

    lsy = Path(args.lsy)
    cfg_path = lsy / "config" / args.config
    if not cfg_path.exists():
        sys.exit(f"config not found: {cfg_path}\n"
                 "   set RACE_LSY_DIR (or --lsy) to your lsy_drone_racing clone (Lesson 3 §4)")
    bridge = lsy / "lsy_drone_racing" / "control" / args.bridge
    if not bridge.exists():
        sys.exit(f"bridge not installed: {bridge}\n"
                 f"   cp tasks/racing/code/{args.bridge} {bridge.parent}/")
    config = load_config(cfg_path)
    config.sim.render = False
    controller_cls = load_controller(bridge)
    env = gymnasium.make(
        config.env.id, freq=config.env.freq, sim_config=config.sim,
        sensor_range=config.env.sensor_range, control_mode=config.env.control_mode,
        track=config.env.track, disturbances=config.env.get("disturbances"),
        randomizations=config.env.get("randomizations"), seed=config.env.seed,
    )
    env = JaxToNumpy(env)

    spec = os.environ.get("RACE_CONTROLLER", "mpc")
    takeoff_t = float(os.environ.get("RACE_TAKEOFF_T", "1.5"))
    blend = float(os.environ.get("RACE_START_BLEND", "0"))
    out_dir = Path(args.out_dir)
    csv_path = out_dir / f"{args.label}.csv"
    rows: list[dict] = []
    print(f"[race_runner] label={args.label} config={args.config} n={args.n} spec={spec} "
          f"plan={os.environ.get('RACE_PLAN', 'closed-form')} takeoff_t={takeoff_t} blend={blend} "
          f"seed={config.env.seed} bridge={args.bridge}", flush=True)
    t_all = time.perf_counter()
    for ep in range(args.n):
        obs, info = env.reset()
        mass, J = sampled_mass(env)
        pos0 = np.asarray(obs["pos"], dtype=np.float64).reshape(3)
        rpy0 = _quat_to_rpy(obs["quat"])
        controller = controller_cls(obs, info, config)
        i = 0
        terminated = truncated = ctrl_finished = False
        t_ep = time.perf_counter()
        while True:
            curr_time = i / config.env.freq
            action = controller.compute_control(obs, info)
            obs, reward, terminated, truncated, info = env.step(action)
            ctrl_finished = controller.step_callback(action, obs, reward, terminated, truncated, info)
            if terminated or truncated or ctrl_finished:
                break
            i += 1
        wall = time.perf_counter() - t_ep
        controller.episode_callback()
        n_gates = int(obs["n_gates_passed"])
        finished = bool(n_gates == int(np.atleast_1d(obs["gate_sequence"]).shape[0]))
        # lsy's terminal observation carries a (-1, -1, -1) sentinel position: take the last
        # position the controller saw (the flown path, one 20 ms step before the end)
        # the bridges record the flown path when RACE_LOG_DIR is set (main() always sets it)
        flown = list(getattr(controller, "_flown", []))
        pos_end = (np.asarray(flown[-1][1:4], dtype=np.float64) if flown
                   else np.asarray(obs["pos"], dtype=np.float64).reshape(3))
        # the same three lines sim.py logs (Lesson 6 / 7 grep them)
        print(f"Flight time (s): {curr_time}\nFinished: {finished}\nGates passed: {n_gates}", flush=True)
        # the controller's internals, BEFORE episode_reset clears them (a policy bridge has
        # neither a solver nor an inner tracker: those columns are then nan / 0)
        solve = list(getattr(controller, "_solve_ms", []))
        ms = np.asarray(solve[1:] or solve or [np.nan])
        ctrl = getattr(controller, "ctrl", None)
        n_fail = int(getattr(ctrl, "n_fail", 0))
        k_end = float(getattr(ctrl, "_k", np.nan)) if getattr(ctrl, "mass_adapt", False) else np.nan
        k_2 = np.nan
        tr = getattr(ctrl, "k_trace", None)
        if tr:
            tt = np.asarray([a for a, _ in tr])
            kk = np.asarray([b for _, b in tr])
            j = int(np.argmin(np.abs(tt - 2.0)))
            k_2 = float(kk[j]) if abs(tt[j] - 2.0) < 0.1 else np.nan
            kdir = Path(os.environ["RACE_LOG_DIR"])
            kdir.mkdir(parents=True, exist_ok=True)
            np.savetxt(kdir / f"ktrace_ep{ep:02d}.csv", np.column_stack([tt, kk]), delimiter=",",
                       header="t,k", comments="", fmt="%.4f")
        obstacles = list(getattr(controller.traj, "obstacles", []))
        if not obstacles:
            from crazy_track.trajectories.freestyle import LSY_LEVEL2_OBSTACLES
            obstacles = list(LSY_LEVEL2_OBSTACLES)
        feat = analyse_flown(flown, controller.traj, controller.gates, obstacles,
                             n_gates, takeoff_t)
        controller.episode_reset()     # prints the solve-time line, writes flown_epNN.csv
        row = dict(ep=ep, mass=mass, mass_ratio=mass / M_NOMINAL, Jx=J[0], Jy=J[1], Jz=J[2],
                   x0=pos0[0], y0=pos0[1], z0=pos0[2], roll0=rpy0[0], pitch0=rpy0[1], yaw0=rpy0[2],
                   gates=n_gates, flight_time=curr_time, finished=int(finished),
                   terminated=int(terminated), truncated=int(truncated),
                   ctrl_finished=int(ctrl_finished), steps=i + 1,
                   x_end=pos_end[0], y_end=pos_end[1], z_end=pos_end[2],
                   verdict=feat.get("verdict", "?"), solve_mean_ms=float(np.nanmean(ms)),
                   solve_max_ms=float(np.nanmax(ms)), ipopt_fail=n_fail, k_end=k_end, k_at_2s=k_2)
        for key in ("t_lift", "dz_cruise", "dz_min", "dev_max", "p4_min", "end_gate", "plane_d",
                    "off_y", "off_z", "end_pole", "d_end_pole", "d_min_pole", "v_end"):
            row[key] = feat.get(key, np.nan)
        xs = feat.get("crossings", {})
        for g in range(1, 5):
            c = xs.get(g)
            for key in ("t", "lag", "y", "z", "ry", "rz"):
                row[f"g{g}_{key}"] = c[key] if c else np.nan
        rows.append(row)
        cross = "; ".join(f"G{g}@{c['t']:.2f} lag{c['lag']:+.3f} y{c['y']:+.3f} z{c['z']:+.3f} "
                          f"(ref y{c['ry']:+.2f} z{c['rz']:+.2f})" for g, c in sorted(xs.items()))
        k_str = "" if not np.isfinite(k_end) else f"k_end={k_end:.3f} k@2s={k_2:.3f} "
        print(f"EP {ep:2d} mass={mass:.5f} ({mass / M_NOMINAL - 1.0:+.1%}) "
              f"start=({pos0[0]:+.3f},{pos0[1]:+.3f},{pos0[2]:.3f}) "
              f"rpy0=({rpy0[0]:+.3f},{rpy0[1]:+.3f},{rpy0[2]:+.3f}) -> gates={n_gates} "
              f"t={curr_time:.2f} {'OK' if finished else feat.get('verdict', '?')} "
              f"term={int(terminated)} trunc={int(truncated)} "
              f"end=({pos_end[0]:+.2f},{pos_end[1]:+.2f},{pos_end[2]:.2f}) "
              f"solve={row['solve_mean_ms']:.1f}/{row['solve_max_ms']:.0f}ms fail={n_fail} "
              f"{k_str}t_lift={row['t_lift']:.2f} dz_cruise={row['dz_cruise']:+.3f} "
              f"dz_min={row['dz_min']:+.3f} dev_max={row['dev_max']:.3f} "
              f"p4_min={row['p4_min']:.3f} wall={wall:.1f}s | {cross}", flush=True)
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            for r in rows:
                w.writerow({k: (f"{v:.5g}" if isinstance(v, float) else v) for k, v in r.items()})
    env.close()

    # ---- the summary, on the leaderboard clock -------------------------------
    if not rows:
        print(f"{args.label}: 0/0 laps (nothing to run: --n {args.n})", flush=True)
        return
    ok = [r for r in rows if r["finished"]]
    laps = np.asarray([r["flight_time"] for r in ok], dtype=float)
    mass_a = np.asarray([r["mass"] for r in rows], dtype=float)
    fin = np.asarray([bool(r["finished"]) for r in rows])
    heavy = mass_a > M_NOMINAL          # at Level 0 every mass is nominal: the heavy half is empty
    light = ~heavy
    mean_lap = float(laps.mean()) if len(laps) else float("nan")
    std_lap = float(laps.std()) if len(laps) else float("nan")
    print(f"{args.label}: {len(ok)}/{len(rows)} laps, mean lap {mean_lap:.3f} s "
          f"(+- {std_lap:.3f}), light half {int(fin[light].sum())}/{int(light.sum())}, "
          f"heavy half {int(fin[heavy].sum())}/{int(heavy.sum())}", flush=True)
    fails = Counter(r["verdict"] for r in rows if not r["finished"])
    solve = float(np.nanmean([r["solve_mean_ms"] for r in rows])) if rows else float("nan")
    print(f"  failures {dict(fails)}; laps {'/'.join(f'{x:.2f}' for x in laps)}; "
          f"solve {solve:.1f} ms mean / {np.nanmax([r['solve_max_ms'] for r in rows]):.0f} ms max "
          f"per 20 ms step; ipopt failures {int(np.nansum([r['ipopt_fail'] for r in rows]))}; "
          f"wall {time.perf_counter() - t_all:.0f} s", flush=True)
    print(f"  wrote {csv_path}, {out_dir / (args.label + '.log')}, "
          f"{os.environ['RACE_LOG_DIR']}/flown_epNN.csv", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", default="level1.toml",
                   help="a toml in repos/lsy_drone_racing/config (level0.toml | level1.toml)")
    p.add_argument("--n", type=int, default=20, help="episodes (the leaderboard protocol is 20)")
    p.add_argument("--label", required=True, help="names the CSV, the log and the summary line")
    p.add_argument("--out-dir", default="tasks/racing/figures/lesson8",
                   help="where <label>.csv, <label>.log and <label>_flown/ go")
    p.add_argument("--bridge", default="race_bridge_mpc.py",
                   help="the controller file, already copied into lsy_drone_racing/control/")
    p.add_argument("--lsy", default=os.environ.get("RACE_LSY_DIR", "/workspace/repos/lsy_drone_racing"),
                   help="the lsy_drone_racing clone (env RACE_LSY_DIR)")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    args.out_dir = str(out_dir)
    # the bridge reads these from the environment: give it absolute paths, so the runner can be
    # started from anywhere and the flown paths land next to this run
    os.environ.setdefault("RACE_CODE_DIR", str(CODE))
    os.environ.setdefault("RACE_LOG_DIR", str(out_dir / f"{args.label}_flown"))
    plan = os.environ.get("RACE_PLAN")
    if plan and not Path(plan).is_absolute():
        os.environ["RACE_PLAN"] = str((ROOT / plan).resolve())

    tee = _Tee(sys.stdout, out_dir / f"{args.label}.log")
    sys.stdout = tee
    try:
        run(args)
    finally:
        sys.stdout = tee.stream
        tee.close()


if __name__ == "__main__":
    logging.basicConfig()
    logging.getLogger("lsy_drone_racing").setLevel(logging.WARNING)
    main()
