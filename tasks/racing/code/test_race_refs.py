"""Checks for the Lesson-6 reference helpers and the Lesson-8 tracker (main venv):

    python -m pytest tasks/racing/code/test_race_refs.py -q
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from race_refs import (  # noqa: E402
    RACE_START,
    GroundStartTrajectory,
    StartBlendTrajectory,
    closed_form_line,
    crossing_angles,
    obstacle_clearance,
    plane_crossings,
)

from crazy_track.trajectories.freestyle import lsy_level2_race  # noqa: E402


def test_closed_form_line_matches_the_vendored_track():
    ref = lsy_level2_race(cruise=3.0)
    line = closed_form_line(ref.gates, cruise=3.0)
    t = np.linspace(0.0, ref.duration, 500)
    np.testing.assert_allclose(line.pos(t), ref.pos(t), atol=1e-9)
    np.testing.assert_allclose(line.gate_times, ref.gate_times)


def test_ground_start_is_on_the_ground_at_t0_and_continuous_at_the_takeoff():
    plan = lsy_level2_race(cruise=2.5)
    tr = GroundStartTrajectory(plan, RACE_START, takeoff_t=1.5)
    np.testing.assert_allclose(tr.pos(0.0), RACE_START)
    np.testing.assert_allclose(tr.vel(0.0), 0.0, atol=1e-9)
    # the takeoff ends at rest exactly where the plan begins to move
    np.testing.assert_allclose(tr.pos(1.5), plan.pos(plan.lead_in), atol=1e-9)
    np.testing.assert_allclose(tr.vel(1.5 - 1e-6), 0.0, atol=1e-3)
    # the clock runs from t = 0: lead-in is zero, gate times shift by takeoff - lead_in
    assert tr.lead_in == 0.0
    shift = 1.5 - plan.lead_in
    np.testing.assert_allclose(tr.gate_times, np.asarray(plan.gate_times) + shift)
    assert tr.duration == pytest.approx(plan.duration + shift)
    # after the takeoff the reference IS the plan
    for tt in (2.0, 3.3, 5.0):
        np.testing.assert_allclose(tr.pos(tt), plan.pos(tt - shift), atol=1e-9)


def test_diagnostics_on_the_closed_form_line():
    ref = lsy_level2_race(cruise=3.0)
    angles = crossing_angles(ref)
    assert len(angles) == 4 and all(a["angle_deg"] < 1.0 for a in angles)   # gates crossed along the normal
    crossings = plane_crossings(ref)
    assert all(any(r["intended"] and r["verdict"] == "opening" for r in rows) for rows in crossings)
    t = np.arange(0.0, ref.duration, 0.01)
    clear = obstacle_clearance(ref.pos(t), ref.obstacles)
    assert len(clear) == 4 and min(clear) > 0.0


def test_safe_line_clears_the_poles_the_lsy_line_shaves():
    gates = lsy_level2_race().gates
    for line, bound in (("lsy", 0.10), ("safe", 0.13)):
        tr = GroundStartTrajectory(closed_form_line(gates, cruise=2.5, line=line), RACE_START, 1.5)
        t = np.arange(0.0, tr.duration, 0.005)
        clear = obstacle_clearance(tr.pos(t), tr.obstacles)
        if line == "lsy":
            assert min(clear) < bound          # the documented weakness
        else:
            assert min(clear) >= bound


def _held_plan(hold=0.3):
    """A ground-start reference that HOLDS the plan's first point: what the bridge builds
    before it adds the start blend (Lesson 8 §5)."""
    plan = lsy_level2_race(cruise=2.5)
    p1 = np.asarray(plan.pos(plan.lead_in), dtype=np.float64)
    return plan, p1, GroundStartTrajectory(plan, start=p1, takeoff_t=hold)


def test_start_blend_with_no_offset_is_the_inner_reference():
    _, _, inner = _held_plan()
    blend = StartBlendTrajectory(inner, np.zeros(3), hold=0.3, T=1.0)
    t = np.linspace(0.0, inner.duration, 400)
    np.testing.assert_allclose(blend.pos(t), inner.pos(t), atol=1e-12)
    np.testing.assert_allclose(blend.vel(t), inner.vel(t), atol=1e-12)
    np.testing.assert_allclose(blend.acc(t), inner.acc(t), atol=1e-12)
    # the clock and the gates are the inner reference's, untouched
    assert blend.lead_in == inner.lead_in
    assert blend.duration == pytest.approx(inner.duration)
    np.testing.assert_allclose(blend.gate_times, inner.gate_times)


def test_start_blend_holds_the_offset_and_fades_it_out():
    _, p1, inner = _held_plan(hold=0.3)
    d = np.array([0.08, -0.06, 0.01])          # the race's ±0.1 m start draw
    blend = StartBlendTrajectory(inner, d, hold=0.3, T=1.0)
    # at t = 0 the reference is where the drone actually is, not where the plan starts
    np.testing.assert_allclose(blend.pos(0.0), inner.pos(0.0) + d, atol=1e-12)
    np.testing.assert_allclose(blend.pos(0.0), p1 + d, atol=1e-12)
    # the whole hold carries the offset (w = 1), so the drone is asked to stand still
    np.testing.assert_allclose(blend.pos(0.3), p1 + d, atol=1e-12)
    np.testing.assert_allclose(blend.vel(0.15), 0.0, atol=1e-12)
    # and by hold + T the offset is gone: the reference IS the plan again
    np.testing.assert_allclose(blend.pos(1.3), inner.pos(1.3), atol=1e-12)
    np.testing.assert_allclose(blend.vel(1.3), inner.vel(1.3), atol=1e-12)
    for tt in (1.5, 2.0, 3.0):
        np.testing.assert_allclose(blend.pos(tt), inner.pos(tt), atol=1e-12)


def test_start_blend_is_smooth_at_both_ends():
    _, _, inner = _held_plan(hold=0.3)
    d = np.array([0.08, -0.06, 0.01])
    T = 1.0
    blend = StartBlendTrajectory(inner, d, hold=0.3, T=T)
    # the quintic weight has w' = w'' = 0 at both ends: no velocity or acceleration step
    for edge in (0.3, 0.3 + T):
        np.testing.assert_allclose(blend.vel(edge), inner.vel(edge), atol=1e-12)
        np.testing.assert_allclose(blend.acc(edge), inner.acc(edge), atol=1e-12)
    # continuous velocity across the whole blend window: the blend's own contribution d*w'(t)
    # never jumps, and the finite difference of the position matches the reported velocity
    t = np.arange(0.0, 2.0, 1e-3)
    dv = blend.vel(t) - inner.vel(t)
    assert np.abs(np.diff(dv, axis=0)).max() < 2e-3         # 1 ms steps
    fd = np.diff(blend.pos(t), axis=0) / 1e-3
    np.testing.assert_allclose(fd, blend.vel(t)[:-1], atol=2e-2)
    # the peak blend speed is 1.875 |d| / T -- far below the 0.88 m/s a rest-to-rest quintic
    # over a 0.3 s hold would demand for the same draw, on a drone still spinning its rotors up
    assert np.linalg.norm(dv, axis=1).max() < 1.875 * np.linalg.norm(d) / T + 1e-3


def test_hover_start_is_the_plan_from_t0_with_an_optional_hold():
    plan = lsy_level2_race(cruise=2.5)
    p1 = plan.pos(plan.lead_in)
    hover = GroundStartTrajectory(plan, start=None, takeoff_t=0.0)      # the plan moves at t = 0
    np.testing.assert_allclose(hover.pos(0.0), p1)
    assert hover.takeoff_t == 0.0 and hover.lead_in == 0.0
    np.testing.assert_allclose(hover.gate_times, np.asarray(plan.gate_times) - plan.lead_in)
    held = GroundStartTrajectory(plan, start=None, takeoff_t=1.0)       # a 1 s settle hold first
    np.testing.assert_allclose(held.pos(0.5), p1, atol=1e-9)             # holding the first point
    np.testing.assert_allclose(held.vel(0.5), 0.0, atol=1e-9)
    np.testing.assert_allclose(held.gate_times, np.asarray(hover.gate_times) + 1.0)


# --------------------------------------------------------------- Lesson 8: mpc_dev
def test_mpcdev_specs_parse_and_unknown_keys_raise():
    from mpc_dev import is_mpcdev_spec, parse_spec, parse_spec_kwargs

    assert parse_spec_kwargs("mpcdev") == {}                 # the bare spec IS the vendored one
    # the two named specs of Lesson 8
    assert parse_spec_kwargs("mpcdev:att=sim,drag=0.495,fgain=1.0") == {
        "att": "sim", "drag": 0.495, "fgain": 1.0}
    assert parse_spec_kwargs("mpcdev:att=sim,drag=0.495,fgain=1.0,mass=1") == {
        "att": "sim", "drag": 0.495, "fgain": 1.0, "mass_adapt": True}
    assert is_mpcdev_spec("mpcdev") and is_mpcdev_spec("mpcdev:H=30")
    assert not is_mpcdev_spec("mpc") and not is_mpcdev_spec("mpc_offsetfree")
    for bad in ("mpcdev:foo=1", "mpcdev:H=abc", "mpcdev:int=rk5", "mpcdev:ramp=t0",
                "mpcdev:H=20,H=30", "mpcdev:att=measured", "mpcdev:H", "mpcx"):
        with pytest.raises(ValueError):
            parse_spec(bad, control_freq=50, verbose=False)


def test_bare_mpcdev_is_the_vendored_controller_action_for_action():
    """The whole point of mpc_dev.py: every option defaults to the vendored MPCController, so
    the bare spec must produce the SAME actions -- not close, identical (Lesson 8 §4)."""
    from crazy_track.controllers.mpc import MPCController

    from mpc_dev import parse_spec

    traj = GroundStartTrajectory(lsy_level2_race(cruise=2.5), RACE_START, takeoff_t=1.5)
    vendored = MPCController(control_freq=50)
    dev = parse_spec("mpcdev", control_freq=50, verbose=False)
    vendored.reset(traj)
    dev.reset(traj)
    offset = np.array([0.05, -0.04, 0.03])
    quat = np.array([0.01, -0.02, 0.0, 1.0])
    quat = quat / np.linalg.norm(quat)
    worst = 0.0
    for i in range(10):                                   # the same synthetic state sequence
        t = 1.6 + i / 50.0
        off = offset * np.cos(0.3 * i)
        state = np.concatenate([traj.pos(t) + off, traj.vel(t) + 0.1 * off, quat,
                                [0.05, -0.03, 0.01]])
        u_v = vendored.act(state.copy(), t)
        u_d = dev.act(state.copy(), t)
        worst = max(worst, float(np.max(np.abs(np.asarray(u_v) - np.asarray(u_d)))))
    assert worst == 0.0, f"mpcdev drifted from the vendored MPC by {worst:.3e}"
    assert len(dev.solve_ms) == 10 and dev.n_fail == 0     # what the SOLVE line reports
