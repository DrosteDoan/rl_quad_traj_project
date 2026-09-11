"""Checks for the Lesson-6 reference helpers (main venv):

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
