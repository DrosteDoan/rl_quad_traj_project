"""Checks for the driver (main venv, in the container; about 4-6 minutes, mostly M1 laps):

    python tasks/racing/code/lesson9/test_driver.py        # no pytest needed

The point of these tests: the driver's loop is the vendored `rollout` loop plus contact / finish / miss stops and
a reused Sim, so at Lesson 7's disturbance values the flown path must be IDENTICAL to the vendored harness's.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import driver as dr  # noqa: E402
import knobs as kb  # noqa: E402
import track_lib as tl  # noqa: E402
from race_eval import make_race_controller  # noqa: E402

from crazy_track.disturbances import ConstantWind, GustWind, Payload  # noqa: E402
from crazy_track.envs.rollout import make_sim, rollout  # noqa: E402
from crazy_track.eval.freestyle_eval import gate_crossing_metrics  # noqa: E402

TRACK = dr.level2_track()


def _zip() -> str | None:
    zs = sorted((tl.ROOT / "tasks" / "racing" / "crazy_track" / "results").glob("*/datt_ppo_final.zip"))
    return str(zs[-1]) if zs else None


def _reference(spec: str, traj, dist=None):
    """The vendored harness, exactly as race_eval.py flies it: a fresh Sim, `rollout`, no early stop."""
    ctrl, _, freq = make_race_controller(spec)
    data = rollout(ctrl, traj, control_freq=freq, sim=make_sim(control="attitude"), disturbance=dist, sensor=None)
    gm = gate_crossing_metrics(data["pos"], data["t"], traj)
    return data, gm


def _same_path(row, data) -> float:
    p, n = row["_path"]["pos"], len(row["_path"]["pos"])
    return float(np.abs(p - data["pos"][:n]).max())


def test_driver_flies_the_vendored_path_at_lesson7_values():
    z = _zip()
    if z is None:
        print("  (skipped: no trained policy zip in results/)")
        return
    spec = f"datt:{z}"
    traj = dr.build_traj(TRACK)
    fl = dr.Flyer({"pol": spec})
    cases = [("wind_const", 0.5, ConstantWind()), ("payload", 0.5, Payload()), ("wind_gust", 0.5, GustWind(seed=0))]
    for cond, lam, ref_dist in [("wind_const", 0.0, None)] + cases:
        row = fl.fly("pol", traj, cond, lam, 0, keep_path=True)
        data, gm = _reference(spec, traj, ref_dist)
        dev = _same_path(row, data)
        print(f"  {cond:10s} lam {lam}: {row['steps']} steps, max |path - vendored| = {dev:.2e} m, "
              f"fail={row['fail'] or '-'} ")
        assert dev < 1e-5, (cond, dev)
        if row["completed"] and all(g["passed"] for g in gm):
            assert abs(row["race_time"] - round(gm[-1]["t_cross"], 3)) < 2e-3
            assert abs(row["t_gate1"] - round(gm[0]["t_cross"], 3)) < 2e-3


def test_reused_sim_is_bit_identical_to_a_fresh_one():
    z = _zip()
    if z is None:
        print("  (skipped: no trained policy zip in results/)")
        return
    traj = dr.build_traj(TRACK)
    a = dr.Flyer({"pol": f"datt:{z}"}).fly("pol", traj, "wind_const", 0.0, 0, keep_path=True)
    fl = dr.Flyer({"pol": f"datt:{z}"})
    fl.fly("pol", traj, "wind_const", 1.0, 0)                          # a disturbed lap first: nothing may leak
    fl.fly("pol", traj, "lighthouse", 0.6, 3)
    b = fl.fly("pol", traj, "wind_const", 0.0, 0, keep_path=True)
    assert a["steps"] == b["steps"] and np.array_equal(a["_path"]["pos"], b["_path"]["pos"])


def test_m1_matches_the_vendored_harness_on_the_level2_ground_plan():
    """M1 flies a whole lap without contact, so this compares full paths: nominal, and Lesson 7's constant wind and
    gust (lam = 0.5 with the starting scale of 2) against the vendored disturbance classes."""
    spec = dr.M1
    traj = dr.build_traj(TRACK)
    fl = dr.Flyer({"M1": spec})
    for cond, lam, ref_dist in (("wind_const", 0.0, None), ("wind_const", 0.5, ConstantWind()),
                                ("wind_gust", 0.5, GustWind(seed=0))):
        row = fl.fly("M1", traj, cond, lam, 0, keep_path=True)
        data, gm = _reference(spec, traj, ref_dist)
        dev = _same_path(row, data)
        print(f"  M1 {cond} lam {lam}: completed={row['completed']} fail={row['fail'] or '-'} race_time={row['race_time']} "
              f"vendored {round(gm[-1]['t_cross'], 3) if gm[-1]['passed'] else 'miss'}, {row['steps']} steps, "
              f"max path deviation {dev:.2e} m, solve {row['solve_ms_mean']} ms")
        assert dev < 1e-5, (cond, lam, dev)
        if lam == 0.0:
            assert row["completed"] == 1 and abs(row["race_time"] - round(gm[-1]["t_cross"], 3)) < 2e-3


def test_reused_m1_is_bit_identical_to_a_fresh_one():
    """The regression for ipopt's leftover initial guess: an MPC lap must not depend on the laps flown before it."""
    traj = dr.build_traj(TRACK)
    a = dr.Flyer({"M1": dr.M1}).fly("M1", traj, "wind_const", 0.0, 0, keep_path=True)
    fl = dr.Flyer({"M1": dr.M1})
    fl.fly("M1", traj, "wind_const", 1.0, 0)
    b = fl.fly("M1", traj, "wind_const", 0.0, 0, keep_path=True)
    dev = float(np.abs(a["_path"]["pos"] - b["_path"]["pos"]).max()) if a["steps"] == b["steps"] else float("inf")
    print(f"  M1 fresh vs after a disturbed lap: steps {a['steps']}/{b['steps']}, max path deviation {dev:.2e} m")
    assert a["steps"] == b["steps"] and dev == 0.0


def test_a_failing_lap_stops_early_and_says_why():
    z = _zip()
    if z is None:
        print("  (skipped: no trained policy zip in results/)")
        return
    traj = dr.build_traj(TRACK)
    row = dr.Flyer({"pol": f"datt:{z}"}).fly("pol", traj, "wind_const", 0.0, 0)
    print(f"  old policy, nominal: completed={row['completed']} fail={row['fail']} "
          f"gates={row['gates_passed']}/4 steps={row['steps']} of {int(traj.duration * dr.FREQ)}")
    assert row["completed"] == 0 and row["fail"] and row["steps"] < int(traj.duration * dr.FREQ)
    assert row["race_time"] == ""


def test_lighthouse_is_deterministic_per_seed_and_task_resumes():
    z = _zip()
    if z is None:
        print("  (skipped: no trained policy zip in results/)")
        return
    members = {"pol": f"datt:{z}"}
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "laps.csv"
        dr.RESULTS = Path(tmp)
        import csv

        fl = dr.Flyer(members)
        dr.run_task("level2", 0, "lighthouse", [0.0, 0.4], [0, 1], members, out, flyer=fl, verbose=False)
        rows = list(csv.DictReader(open(out)))
        assert len(rows) == 1 + 2, len(rows)                            # lam = 0 once, lam > 0 for two seeds
        dr.run_task("level2", 0, "lighthouse", [0.0, 0.4], [0, 1], members, out, flyer=fl, verbose=False)
        assert len(list(csv.DictReader(open(out)))) == 3               # nothing re-flown
        a = fl.fly("pol", dr.build_traj(TRACK), "lighthouse", 0.4, 1)
        assert a["rmse_3d"] == float(rows[2]["rmse_3d"]) and a["race_time"] == (
            float(rows[2]["race_time"]) if rows[2]["race_time"] else "")


if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as e:  # noqa: BLE001
                fails += 1
                print(f"FAIL {name}: {type(e).__name__}: {e}")
    sys.exit(1 if fails else 0)
