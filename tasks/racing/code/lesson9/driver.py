"""Lesson 9 phase 4: fly the cells of the sweep and write one CSV row per lap.

    python tasks/racing/code/lesson9/driver.py --role study --track 4 --cond wind_gust \
        --lams 0,0.1,0.2 --seeds 25 --member policy=datt:/workspace/.../datt_ppo_final.zip

Runs in the MAIN venv, in the container. One process flies every member over one (track, condition) task and
keeps ONE crazyflow Sim and one controller per member alive across laps (`sim.reset()` between laps), because
most of a lap's wall time in a fresh process is JIT compilation.

What one lap is (METHODOLOGY.md sections 3, 4, 7):
  * the plan of the track's JSON, stretched by its `stretch`, on the GROUND clock: `GroundStartTrajectory`
    with the 0.2 s hold (Lesson 8 section 3); race_time = t = 0 -> last-gate crossing;
  * the disturbance and the sensor of `knobs.make_conditions(cond, lam, seed)` (None at lam = 0);
  * the loop of `crazy_track.envs.rollout.rollout` with three additions, and nothing else changed:
      - the swept contact test of `contact.py` on every step; the lap ENDS at the first frame contact;
      - the lap ends when the last gate is crossed (lsy ends the race there) or a gate is missed (the target
        gate's clock + 1.0 s passes without a crossing inside its opening), or the drone diverges;
      - the Sim is reused, and the external force is zeroed at the start of every lap.
  * completed = all four gates passed in order (`gate_crossing_metrics`, the same test as race_eval) AND no
    contact before the last crossing.
  * RMSE and the maximum deviation are over the racing segment only: from the end of the hold to the last-gate
    crossing (or to where a failed lap ended; those are censored, `completed` says so).

Rows are appended as they finish and a (lam, seed, member) already in the file is skipped, so a task resumes.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CODE = HERE.parent
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(HERE))
import contact as ct  # noqa: E402
import knobs as kb  # noqa: E402
import track_lib as tl  # noqa: E402
from race_eval import make_race_controller as _vendored_make_controller  # noqa: E402
from race_refs import RACE_START, GroundStartTrajectory  # noqa: E402

from crazy_track.envs.rollout import apply_force, get_state, make_sim, set_drone_state  # noqa: E402
from crazy_track.eval.freestyle_eval import gate_crossing_metrics  # noqa: E402
from crazy_track.trajectories.freestyle import RaceGate  # noqa: E402
from crazy_track.trajectories.sampled import SampledRaceTrajectory  # noqa: E402

FREQ = 100                                   # the vendored harness's attitude-mode control rate
HOLD = 0.2                                   # s: the rotor spin-up hold of a ground-start plan (Lesson 8 section 2)
M1 = "mpcdev:att=sim,drag=0.495,fgain=1.0"
# Preview-window equalisation (user, 2026-09-22): every member should look the SAME distance ahead. M1 family
# already sits at H=20, dtp=0.04 -> 0.8 s. mppi_l1's own default is H=25, dtp=0.02 -> 0.5 s; there is no spec-
# string way to override its horizon (unlike mpcdev's comma syntax), so it is constructed directly below with
# horizon EXTENDED to 40 (40*0.02 = 0.8 s), keeping dt_plan=0.02 -- its own tuned rollout integration step (the
# 2026-07-22 sweep the class's docstring cites) -- unchanged, so only how far it looks changes, not how finely
# it integrates. "mppi_l1" therefore means this 0.8 s configuration from 2026-09-22 on, not the vendored default.
MPPI_HORIZON, MPPI_DTP = 40, 0.02
DEFAULT_MEMBERS = {"M1": M1, "M1+ESO": M1 + ",dist=eso", "M1+L1": M1 + ",dist=l1", "M1+mass": M1 + ",mass=1",
                   "mppi_l1": "mppi_l1"}


def make_race_controller(spec: str, seed: int = 0, control_freq: int = FREQ):
    """The vendored spec table, with `mppi_l1` reconfigured to an 0.8 s preview horizon (see MPPI_HORIZON above)
    and a new `robust:<path>` spec for a RobustTrackingEnv-trained policy (0.8 s window, freq=100 -- NOT the
    same as `datt:<path>`, which loads through the vendored DATTPolicyController and its 0.6 s window; using
    `datt:` on a robust model would silently feed it the wrong reference window). Everything else is
    unchanged -- delegated straight to race_eval.make_race_controller."""
    if spec == "mppi_l1":
        from crazy_track.controllers.mppi_l1 import MPPIL1Controller

        return (MPPIL1Controller(horizon=MPPI_HORIZON, dt_plan=MPPI_DTP, control_freq=control_freq, seed=seed),
                "attitude", control_freq)
    if spec.startswith("robust:"):
        from robust_policy import RobustPolicyController

        return RobustPolicyController(spec.split(":", 1)[1], control_freq=control_freq), "attitude", control_freq
    return _vendored_make_controller(spec, seed=seed)
RESULTS = HERE / "results"
COLUMNS = ["role", "track", "cond", "lam", "seed", "member", "completed", "gates_passed", "gates_clean", "fail",
           "t_impact", "impact_gate", "impact_box", "race_time", "t_gate1", "rmse_3d", "rmse_xy", "max_dev",
           "steps", "wall_s", "solve_ms_mean", "solve_ms_max", "ipopt_fail", "mass_mult"]


# ---- tracks ----------------------------------------------------------------------------------------------------
def load_track(role: str, seed: int) -> dict:
    return json.loads((HERE / "tracks" / role / f"track_{seed:04d}.json").read_text())


def level2_track() -> dict:
    """Lesson 8's level-2 ground plan on the level-2 gates: the regression target (test_driver.py)."""
    return {"role": "level2", "seed": 0, "gates": tl.level2_gates(),
            "plan_csv": "tasks/racing/plans/ground_f0.85.csv", "stretch": 1.0}


def build_traj(track: dict, hold: float = HOLD) -> GroundStartTrajectory:
    gates = tl.race_gates(track["gates"])
    plan = SampledRaceTrajectory(str(tl.ROOT / track["plan_csv"]), gates=gates, obstacles=[],
                                 time_scale=float(track["stretch"]))
    return GroundStartTrajectory(plan, RACE_START, takeoff_t=hold)


# ---- one lap ---------------------------------------------------------------------------------------------------
class Flyer:
    """One Sim and one controller per member, kept across laps."""

    def __init__(self, members: dict[str, str]):
        self.members = dict(members)
        self.sim = make_sim(control="attitude")
        self.ctrls: dict = {}

    def controller(self, label: str):
        if label not in self.ctrls:
            ctrl, mode, freq = make_race_controller(self.members[label], seed=0, control_freq=FREQ)
            assert (mode, freq) == ("attitude", FREQ), f"{label}: {self.members[label]} needs {mode}@{freq}"
            self.ctrls[label] = ctrl
        return self.ctrls[label]

    def fly(self, label: str, traj: GroundStartTrajectory, cond: str, lam: float, seed: int,
            keep_path: bool = False) -> dict:
        ctrl, sim = self.controller(label), self.sim
        if hasattr(ctrl, "rng"):                                    # MPPI samples: one seeded stream per lap
            ctrl.rng = np.random.default_rng(int(seed))
        dt, gates = 1.0 / FREQ, traj.gates
        n_steps, n_sub = int(traj.duration * FREQ), sim.freq // FREQ
        dist, sensor = kb.make_conditions(cond, lam, seed, control_freq=FREQ, horizon=n_steps + 5)

        sim.reset()
        set_drone_state(sim, traj.pos(0.0))
        apply_force(sim, np.zeros(3))                               # nothing may leak in from the previous lap
        mult = kb.mass_scale(cond, lam, seed)
        if mult != 1.0:
            # reset() restores the nominal mass, exactly like the force above; override it the same way, on
            # the CURRENT (just-reset, nominal) params, so nothing accumulates across laps
            sim.data = sim.data.replace(params=sim.data.params.replace(mass=sim.data.params.mass * mult))
        ctrl.reset(traj)
        if hasattr(ctrl, "opti"):
            # MPCDev sets no initial guess on the first step of a lap (`_prev is None`), so CasADi's Opti would
            # start ipopt from whatever the LAST lap left behind; a fresh controller starts from zeros. Zero it,
            # so a lap does not depend on the laps flown before it (mpc_dev.py stays untouched).
            ctrl.opti.set_initial(ctrl.X, np.zeros(ctrl.X.shape))
            ctrl.opti.set_initial(ctrl.U, np.zeros(ctrl.U.shape))
        if dist is not None:
            dist.reset()
        if sensor is not None:
            sensor.reset()

        gt = [float(x) for x in traj.gate_times]
        log_t, log_pos, log_q = [], [], []
        hit, fail, gate_idx, t_cross_last = None, "", 0, None
        prev = None
        t0 = time.perf_counter()
        for i in range(n_steps):
            t = i * dt
            state = get_state(sim)
            pos, quat = state[:3], state[6:10]
            if not np.all(np.isfinite(state)) or pos[2] < -0.3 or np.abs(pos).max() > 8.0:
                fail = "diverged"
                break
            log_t.append(t)
            log_pos.append(pos.copy())
            log_q.append(quat.copy())
            if prev is not None:
                # contact on the swept segment of the last step (skipped when nowhere near a gate)
                step = float(np.linalg.norm(pos - prev[0]))
                if min(np.linalg.norm(pos - g.center) for g in gates) < ct.NEAR + step:
                    hit = ct.first_contact([t - dt, t], np.stack([prev[0], pos]), np.stack([prev[1], quat]), gates)
                    if hit is not None:
                        fail = "contact"
                        break
                # the target gate: crossed inside the opening near its clock?
                g = gates[gate_idx]
                x0, x1 = g.to_gate_frame(prev[0]), g.to_gate_frame(pos)
                if np.sign(x0[0]) != np.sign(x1[0]):
                    w = x0[0] / (x0[0] - x1[0])
                    tc = (t - dt) + w * dt
                    yz = x0[1:] + w * (x1[1:] - x0[1:])
                    if abs(tc - gt[gate_idx]) <= 1.0 and float(np.abs(yz).max()) < RaceGate.HALF_OPENING:
                        gate_idx += 1
                        t_cross_last = tc
                        if gate_idx == len(gates):
                            break                                   # the last gate: the race ends here
                if t > gt[gate_idx] + 1.0:
                    fail = f"missed_gate_{gate_idx + 1}"
                    break
            prev = (pos.copy(), quat.copy())
            meas = sensor.measure(t, state) if sensor is not None else state
            action = np.asarray(ctrl.act(meas, t), dtype=np.float32)
            if dist is not None:
                apply_force(sim, dist.force(t, state))
            sim.attitude_control(action.reshape(1, 1, 4))
            sim.step(n_sub)
        wall = time.perf_counter() - t0
        return self._row(traj, cond, lam, seed, label, np.array(log_t), np.array(log_pos), hit, fail, wall, ctrl,
                         keep_path, np.array(log_q))

    def _row(self, traj, cond, lam, seed, label, t, pos, hit, fail, wall, ctrl, keep_path, quat) -> dict:
        gm = gate_crossing_metrics(pos, t, traj) if len(t) > 2 else []
        passed = [g["passed"] for g in gm]
        n_pass = int(sum(passed))
        done = len(passed) == len(traj.gates) and all(passed) and hit is None
        if not done and not fail:
            fail = "timeout"
        t_last = float(gm[-1]["t_cross"]) if done else (float(t[-1]) if len(t) else 0.0)
        w = (t >= traj.takeoff_t) & (t <= t_last)
        err = pos[w] - traj.pos(t[w]) if w.any() else np.zeros((1, 3))
        ms = np.asarray(getattr(ctrl, "solve_ms", None) or [])           # policies have no solver
        row = {"track": None, "cond": cond, "lam": lam, "seed": seed, "member": label, "completed": int(done),
               "gates_passed": n_pass, "gates_clean": int(sum(g["clean"] for g in gm)), "fail": fail,
               "t_impact": "" if hit is None else round(hit["t"], 3), "impact_gate": "" if hit is None else hit["gate"],
               "impact_box": "" if hit is None else hit["box"],
               "race_time": round(t_last - traj.lead_in, 3) if done else "",
               "t_gate1": round(float(gm[0]["t_cross"]) - traj.lead_in, 3) if gm and gm[0]["passed"] else "",
               "rmse_3d": round(float(np.sqrt(np.mean(np.sum(err ** 2, axis=1)))), 4),
               "rmse_xy": round(float(np.sqrt(np.mean(np.sum(err[:, :2] ** 2, axis=1)))), 4),
               "max_dev": round(float(np.linalg.norm(err, axis=1).max()), 3), "steps": len(t),
               "wall_s": round(wall, 2), "solve_ms_mean": round(float(ms.mean()), 1) if len(ms) else "",
               "solve_ms_max": round(float(ms.max()), 1) if len(ms) else "", "ipopt_fail": int(getattr(ctrl, "n_fail", 0)),
               "mass_mult": round(kb.mass_scale(cond, lam), 4)}
        if keep_path:
            row["_path"] = {"t": t, "pos": pos, "quat": quat, "gates": gm}
        return row


# ---- a task: one track, one condition -----------------------------------------------------------------------------
def _existing(path: Path) -> set:
    if not path.exists():
        return set()
    with open(path) as fh:
        return {(round(float(r["lam"]), 4), int(r["seed"]), r["member"]) for r in csv.DictReader(fh)}


def run_task(role: str, track_seed: int, cond: str, lams: list[float], seeds: list[int], members: dict[str, str],
             out: Path | None = None, hold: float = HOLD, flyer: Flyer | None = None, verbose: bool = True,
             stretch_mult: float = 1.0) -> Path:
    track = level2_track() if role == "level2" else load_track(role, track_seed)
    track = {**track, "stretch": float(track["stretch"]) * float(stretch_mult)}    # 1.0 = the plan as generated
    traj = build_traj(track, hold)
    out = Path(out) if out else RESULTS / f"laps_{role}_{track_seed:06d}_{cond}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = _existing(out)
    flyer = flyer or Flyer(members)
    new_file = not out.exists()
    with open(out, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        if new_file:
            w.writeheader()
        for lam in lams:
            for seed in ([0] if lam == 0 else seeds):                # lam = 0 is deterministic: one run
                for label in members:
                    if (round(lam, 4), seed, label) in done:
                        continue
                    row = flyer.fly(label, traj, cond, lam, seed)
                    row.update({"role": role, "track": track_seed})
                    w.writerow({k: row[k] for k in COLUMNS})
                    fh.flush()
                    if verbose:
                        print(f"track {track_seed} {cond} lam {lam:.2f} seed {seed:2d} {label:8s} "
                              f"{'OK ' + str(row['race_time']) if row['completed'] else 'FAIL ' + row['fail']:22s} "
                              f"rmse {row['rmse_3d']:.3f} wall {row['wall_s']:.0f}s", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--role", choices=["study", "dev", "level2"], default="study")
    ap.add_argument("--track", type=int, default=0, help="track seed (see tracks/<role>/)")
    ap.add_argument("--cond", choices=kb.CONDITIONS + kb.EXTRA_CONDITIONS, required=True)
    ap.add_argument("--lams", default="0,0.5,1.0", help="comma-separated lambda values")
    ap.add_argument("--seeds", type=int, default=1, help="number of noise seeds (0..N-1) for lam > 0")
    ap.add_argument("--member", action="append", default=[], metavar="LABEL=SPEC",
                    help="add a member (repeatable), e.g. rl0=datt:/workspace/.../datt_ppo_final.zip")
    ap.add_argument("--only", default=None, help="comma-separated member labels to fly (default: all)")
    ap.add_argument("--no-default-members", action="store_true")
    ap.add_argument("--scales", default=None, metavar="NAME=X,...",
                    help="override the disturbance ceilings (units of Lesson 7's value), e.g. wind_const=8; calibration")
    ap.add_argument("--stretch-mult", type=float, default=1.0,
                    help="multiply the track's time-stretch (same path, slower: less thrust demand); experiments only")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    if args.scales:
        kb.set_scales({k: float(v) for k, v in (p.split("=") for p in args.scales.split(","))})
    members = {} if args.no_default_members else dict(DEFAULT_MEMBERS)
    for m in args.member:
        label, spec = m.split("=", 1)
        members[label] = spec
    if args.only:
        members = {k: v for k, v in members.items() if k in args.only.split(",")}
    out = run_task(args.role, args.track, args.cond, [float(x) for x in args.lams.split(",")],
                   list(range(args.seeds)), members, args.out, stretch_mult=args.stretch_mult)
    print(f"rows in {out}")


if __name__ == "__main__":
    main()
