"""SampledRaceTrajectory: an externally planned CSV must round-trip through the
Trajectory interface with the same gate-crossing times the source had."""
import numpy as np
import pytest

from crazy_track.trajectories.freestyle import lsy_level2_race
from crazy_track.trajectories.sampled import SampledRaceTrajectory

COLS = ["t", "p_x", "p_y", "p_z", "q_w", "q_x", "q_y", "q_z", "v_x", "v_y", "v_z",
        "w_x", "w_y", "w_z", "a_lin_x", "a_lin_y", "a_lin_z", "a_rot_x", "a_rot_y", "a_rot_z",
        "u_1", "u_2", "u_3", "u_4", "jerk_x", "jerk_y", "jerk_z", "snap_x", "snap_y", "snap_z"]


@pytest.fixture(scope="module")
def csv_and_ref(tmp_path_factory):
    ref = lsy_level2_race(cruise=3.0)
    t = np.arange(ref.lead_in, ref.duration, 0.005)          # motion onset .. hover tail
    P, V, A = ref.pos(t), ref.vel(t), ref.acc(t)
    n = len(t); Z = np.zeros((n, 1)); O = np.ones((n, 1))
    M = np.hstack([(t - t[0])[:, None], P, O, Z, Z, Z, V, Z, Z, Z, A] + [Z] * 13)
    path = tmp_path_factory.mktemp("togt") / "plan.csv"
    np.savetxt(path, M, delimiter=",", header=",".join(COLS), comments="")
    return str(path), ref


def test_gate_times_round_trip(csv_and_ref):
    path, ref = csv_and_ref
    tr = SampledRaceTrajectory(path, gates=ref.gates, lead_in=ref.lead_in)
    assert len(tr.gate_times) == 4
    np.testing.assert_allclose(tr.gate_times, ref.gate_times, atol=6e-3)


def test_lead_in_hold_and_tail(csv_and_ref):
    path, ref = csv_and_ref
    tr = SampledRaceTrajectory(path, gates=ref.gates, lead_in=1.5, tail=1.0)
    np.testing.assert_allclose(tr.pos(0.0), tr.pos(1.4))          # stationary lead-in
    np.testing.assert_allclose(tr.vel(0.5), 0.0)
    assert tr.duration == pytest.approx(tr.t_end + 1.0)
    np.testing.assert_allclose(tr.pos(tr.duration), tr.pos(tr.t_end))


def test_time_scale_stretches_time_not_path(csv_and_ref):
    path, ref = csv_and_ref
    a = SampledRaceTrajectory(path, gates=ref.gates)
    b = SampledRaceTrajectory(path, gates=ref.gates, time_scale=1.5)
    np.testing.assert_allclose(b.plan_duration, 1.5 * a.plan_duration)
    for ta, tb in zip(a.gate_times, b.gate_times):                 # same crossing points
        np.testing.assert_allclose(b.pos(tb), a.pos(ta), atol=2e-2)
        np.testing.assert_allclose(tb - b.lead_in, 1.5 * (ta - a.lead_in), atol=6e-3)
    s = np.linalg.norm(a.vel(a.gate_times[2]))
    np.testing.assert_allclose(np.linalg.norm(b.vel(b.gate_times[2])), s / 1.5, rtol=0.05)


def test_descriptor_is_zero(csv_and_ref):
    path, ref = csv_and_ref
    tr = SampledRaceTrajectory(path, gates=ref.gates)
    assert tr.maneuver_descriptor(2.0).shape == (6,)
    np.testing.assert_allclose(tr.maneuver_descriptor(np.linspace(0, tr.duration, 9)), 0.0)


def test_missing_gate_is_an_error(csv_and_ref, tmp_path):
    path, ref = csv_and_ref
    M = np.genfromtxt(path, delimiter=",", names=True)
    keep = M["t"] < 0.6 * M["t"][-1]                                # truncate before gate 4
    np.savetxt(tmp_path / "short.csv", np.array(M[keep].tolist()), delimiter=",",
               header=",".join(COLS), comments="")
    with pytest.raises(ValueError, match="never crosses"):
        SampledRaceTrajectory(str(tmp_path / "short.csv"), gates=ref.gates)
