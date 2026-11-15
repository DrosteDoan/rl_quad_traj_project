"""Checks for the driver (main venv, in the container; about 4-6 minutes, mostly M1 laps):

    python tasks/racing/code/lesson9/tests/test_driver.py        # no pytest needed

The point of these tests: the driver's loop is the vendored `rollout` loop plus contact / finish / miss stops and
a reused Sim, so at Lesson 7's disturbance values the flown path must be IDENTICAL to the vendored harness's.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # the lesson9 modules
import driver as dr  # noqa: E402
import knobs as kb  # noqa: E402
import track_lib as tl  # noqa: E402
from race_eval import make_race_controller  # noqa: E402

from crazy_track.controllers.mppi_l1 import MPPIL1Controller  # noqa: E402
from crazy_track.controllers.utils import THRUST_MAX, THRUST_MIN  # noqa: E402
from crazy_track.disturbances import ConstantWind, GustWind, Payload  # noqa: E402
from crazy_track.envs.rollout import make_sim, rollout  # noqa: E402
from crazy_track.eval.freestyle_eval import gate_crossing_metrics  # noqa: E402

TRACK = dr.level2_track()


def _zip() -> str | None:
    roots = (tl.ROOT / "tasks" / "racing" / "crazy_track" / "results", tl.ROOT / "tasks" / "racing" / "code" / "lesson9" / "policies")
    zs = sorted((z for r in roots for z in r.glob("*/datt_ppo_final.zip")), key=lambda z: z.parent.name)
    return str(zs[-1]) if zs else None


def _lesson7_scale():
    """Force the ORIGINAL Lesson-7-exact scale (2x) regardless of what calibrate.py has since frozen, so
    lam = 0.5 reproduces Lesson 7's value exactly -- the property these regression tests check."""
    kb.set_scales({"wind_const": 2.0, "payload": 2.0, "wind_gust": 2.0, "lighthouse": 2.0})


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
    _lesson7_scale()
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
    kb.set_scales(None)


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
    _lesson7_scale()
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
    kb.set_scales(None)


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


def test_mass_mult_changes_the_flight_and_does_not_leak():
    """A heavier drone sags/lags relative to the nominal reference (untouched, same model); the next lap at
    lam = 0 must be bit-identical to a fresh one -- the mass override must not survive a reset."""
    traj = dr.build_traj(TRACK)
    fl = dr.Flyer({"M1": dr.M1})
    nominal = fl.fly("M1", traj, "wind_const", 0.0, 0, keep_path=True)
    heavy = fl.fly("M1", traj, "mass_mult", 1.0, 0, keep_path=True)
    print(f"  M1 nominal vs +11.5% mass: mass_mult={heavy['mass_mult']} completed={heavy['completed']} "
          f"fail={heavy['fail'] or '-'} rmse {nominal['rmse_3d']:.3f} -> {heavy['rmse_3d']:.3f}")
    assert heavy["mass_mult"] > 1.0
    n = min(len(nominal["_path"]["pos"]), len(heavy["_path"]["pos"]))
    assert not np.allclose(nominal["_path"]["pos"][:n], heavy["_path"]["pos"][:n])
    back = fl.fly("M1", traj, "wind_const", 0.0, 0, keep_path=True)      # a DIFFERENT condition: mass_mult only
    assert back["steps"] == nominal["steps"] and np.array_equal(back["_path"]["pos"], nominal["_path"]["pos"])


def test_m1_mass_adaptation_engages_under_mass_mult():
    """M1+mass's thrust-scale estimate k should move away from 1.0 under a heavier drone and stay near 1.0
    without it (default mass=0 on plain M1 has no k at all)."""
    traj = dr.build_traj(TRACK)
    fl = dr.Flyer({"M1+mass": dr.DEFAULT_MEMBERS["M1+mass"]})
    fl.fly("M1+mass", traj, "wind_const", 0.0, 0)
    ctrl = fl.ctrls["M1+mass"]
    assert hasattr(ctrl, "_k") and ctrl.mass_adapt
    k_nominal = ctrl._k
    fl.fly("M1+mass", traj, "mass_mult", 1.0, 0)
    k_heavy = fl.ctrls["M1+mass"]._k
    print(f"  M1+mass thrust-scale k: nominal {k_nominal:.3f}, +11.5% mass {k_heavy:.3f}")
    # k is a thrust-EFFECTIVENESS scale (predicted acc uses cmd_f_coef * k): a heavier real drone produces LESS
    # acceleration per unit commanded thrust than the model's nominal mass assumes, so the estimator must drive
    # k DOWN to match; the controller then commands proportionally more raw thrust for the same target acc.
    assert k_heavy < k_nominal - 0.02


def test_mppi_l1_uses_the_0point8s_preview_horizon():
    ctrl, mode, freq = dr.make_race_controller("mppi_l1", seed=0, control_freq=dr.FREQ)
    assert (ctrl.H, ctrl.dtp) == (dr.MPPI_HORIZON, dr.MPPI_DTP)
    np.testing.assert_allclose(ctrl.H * ctrl.dtp, 0.8)
    assert (mode, freq) == ("attitude", dr.FREQ)


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


def test_full_seeds_at_zero_flies_every_seed_at_lam_zero_when_set():
    """Found 2026-09-30: study_sweep.py escalates a sampling-based member (mppi_l1) to 3 seeds specifically
    BECAUSE its own internal randomness varies even when the external disturbance is off at lam=0 (its own
    module docstring says so) -- but `run_task`'s lam=0 shortcut silently collapsed that back to 1 seed
    anyway, since it had no notion of per-member escalation. `full_seeds_at_zero` is the opt-in fix: default
    False (this test's own sibling above, unchanged, proves the OLD default behavior still holds), True
    flies the full seed list even at lam=0."""
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
        dr.run_task("level2", 0, "lighthouse", [0.0, 0.4], [0, 1], members, out, flyer=fl, verbose=False,
                   full_seeds_at_zero=True)
        rows = list(csv.DictReader(open(out)))
        assert len(rows) == 2 + 2, len(rows)          # BOTH seeds flown at lam=0 too, not just seed 0
        assert {r["seed"] for r in rows if float(r["lam"]) == 0.0} == {"0", "1"}


def test_gate_shift_scores_against_the_displaced_gates_and_leaves_the_plan_alone():
    """At lam = 1 the controller still gets the nominal traj (same first actions), but the laps are scored on shifted gates."""
    traj = dr.build_traj(TRACK)
    nominal = [(g.pos, g.yaw) for g in traj.gates]
    fl = dr.Flyer({"M1": dr.DEFAULT_MEMBERS["M1"]})
    a = fl.fly("M1", traj, "wind_const", 0.0, 0, keep_path=True)
    b = fl.fly("M1", traj, "gate_shift", 1.0, 0, keep_path=True)
    assert [(g.pos, g.yaw) for g in traj.gates] == nominal, "the plan's gates are never displaced"
    np.testing.assert_array_equal(a["_path"]["pos"][:100], b["_path"]["pos"][:100])       # same flight until a gate matters
    true = kb.displace_gates(traj.gates, "gate_shift", 1.0, 0)
    off = [g["offset"] for g in b["_path"]["gates"]]
    exp = [g["offset"] for g in gate_crossing_metrics(b["_path"]["pos"], b["_path"]["t"], dr._TrueGates(traj, true))]
    assert off == exp, "the row's gate metrics are the ones against the displaced gates"


# ---- the roll/pitch-capped mppi twin (dr.CappedMPPIL1Controller, spec `mppi_l1,rp_max=<rad>`) ----------------------
# Unit level: `act()` is fed hand-made level states along a real study-track reference, no simulator.
_CAP_TRACK, _CAP_STEPS = 747, 60          # any accepted study track (only its reference is used); 0.6 s at 100 Hz


def _cap_traj():
    return dr.build_traj(dr.load_track("study", _CAP_TRACK))


def _cap_kwargs(seed=0):
    return dict(horizon=dr.MPPI_HORIZON, dt_plan=dr.MPPI_DTP, control_freq=dr.FREQ, seed=seed)


def _cap_states(traj, offset=np.zeros(3), n=_CAP_STEPS):
    out = []
    for i in range(n):
        t = 0.3 + i / dr.FREQ
        pos = np.asarray(traj.pos(np.array([t])))[0] + offset
        vel = np.asarray(traj.vel(np.array([t])))[0]
        out.append((np.concatenate([pos, vel, [0.0, 0.0, 0.0, 1.0], np.zeros(3)]), t))
    return out


def _cap_run(ctrl, traj, states):
    ctrl.reset(traj)
    return np.array([ctrl.act(s, t) for s, t in states])


def test_capped_mppi_at_1_is_bit_identical_to_vendored():
    traj = _cap_traj()
    for offset in (np.zeros(3), np.array([0.6, -0.6, 0.2])):    # tracking, and a large error that saturates
        states = _cap_states(traj, offset)
        a_ref = _cap_run(MPPIL1Controller(**_cap_kwargs()), traj, states)
        a_cap = _cap_run(dr.CappedMPPIL1Controller(**_cap_kwargs(), rp_max=1.0), traj, states)
        assert np.array_equal(a_ref, a_cap), f"offset {offset}: max |diff| {np.abs(a_ref - a_cap).max()}"


def test_capped_mppi_at_1_matches_vendored_internal_state():
    traj = _cap_traj()
    states = _cap_states(traj, np.array([0.3, 0.2, 0.0]))
    ref, cap = MPPIL1Controller(**_cap_kwargs(3)), dr.CappedMPPIL1Controller(**_cap_kwargs(3), rp_max=1.0)
    _cap_run(ref, traj, states), _cap_run(cap, traj, states)
    for attr in ("u_mean", "v_hat", "sigma_hat", "sigma_f"):
        assert np.array_equal(getattr(ref, attr), getattr(cap, attr)), attr
    assert np.array_equal(ref.rng.normal(size=8), cap.rng.normal(size=8)), "rng streams diverged"


def test_capped_mppi_cap_binds_and_is_respected():
    """At a large tracking error the vendored controller commands more than 0.7 rad (so the test is not vacuous),
    and the capped one never does -- neither in its actions nor anywhere in its planned mean sequence."""
    traj = _cap_traj()
    states = _cap_states(traj, np.array([1.0, -1.0, 0.0]))
    a_ref = _cap_run(MPPIL1Controller(**_cap_kwargs()), traj, states)
    assert np.abs(a_ref[:, :2]).max() > 0.7, "scenario too gentle: vendored never exceeds 0.7, cap untested"
    cap = dr.CappedMPPIL1Controller(**_cap_kwargs(), rp_max=0.7)
    cap.reset(traj)
    for s, t in states:
        a = cap.act(s, t)
        assert np.abs(a[:2]).max() <= 0.7 + 1e-12, a
        assert np.abs(cap.u_mean[:, :2]).max() <= 0.7 + 1e-12
    assert cap._u_lo[2:].tolist() == [-0.5, THRUST_MIN] and cap._u_hi[2:].tolist() == [0.5, THRUST_MAX]


def test_capped_mppi_reset_and_reseed_reproduce_a_lap():
    """What Flyer does between laps (reseed ctrl.rng, ctrl.reset(traj)) makes a lap independent of the one before."""
    traj = _cap_traj()
    cap = dr.CappedMPPIL1Controller(**_cap_kwargs(), rp_max=0.7)
    states = _cap_states(traj, np.array([0.4, 0.0, 0.1]))
    cap.rng = np.random.default_rng(5)
    first = _cap_run(cap, traj, states)
    _cap_run(cap, traj, _cap_states(traj, np.array([-0.5, 0.5, 0.0])))      # an unrelated lap in between
    cap.rng = np.random.default_rng(5)
    assert np.array_equal(first, _cap_run(cap, traj, states))


def test_capped_mppi_rejects_bad_rp_max():
    for bad in (0.0, -0.1, 1.2):
        try:
            dr.CappedMPPIL1Controller(**_cap_kwargs(), rp_max=bad)
        except ValueError:
            continue
        raise AssertionError(f"rp_max={bad} accepted")


def test_capped_mppi_spec_routing():
    ctrl, mode, freq = dr.make_race_controller("mppi_l1,rp_max=0.7", seed=0)
    assert isinstance(ctrl, dr.CappedMPPIL1Controller) and (mode, freq) == ("attitude", dr.FREQ)
    assert (ctrl.H, ctrl.dtp, ctrl.rp_max) == (dr.MPPI_HORIZON, dr.MPPI_DTP, 0.7)
    plain, _, _ = dr.make_race_controller("mppi_l1", seed=0)
    assert type(plain) is MPPIL1Controller                      # the original member is untouched
    for bad in ("mppi_l1,rp_max=0.7,foo=1", "mppi_l1,horizon=20"):
        try:
            dr.make_race_controller(bad)
        except ValueError:
            continue
        raise AssertionError(f"{bad!r} accepted")


def test_capped_member_tables():
    assert dr.RP_MAX_MEMBERS["mppi_l1@0.7"] == "mppi_l1,rp_max=0.7"
    assert set(dr.RP_MAX_MEMBERS) == {f"{k}@0.7" for k in dr.DEFAULT_MEMBERS}   # all five twinned
    assert list(dr.DEFAULT_MEMBERS) == ["M1", "M1+ESO", "M1+L1", "M1+mass", "mppi_l1"]   # defaults unchanged
    assert not any("@" in k for k in dr.DEFAULT_MEMBERS)


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
