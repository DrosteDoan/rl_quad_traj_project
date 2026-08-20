import numpy as np
import pytest

from crazy_track.trajectories import RaceGate, feasibility_report
from crazy_track.trajectories.freestyle import TRACKS

EPS = 1e-5
TWR = 1.88
GRAVITY = 9.81
RATE_MAX_RP = 15.0  # rad/s roll/pitch command limit (controllers.utils.RATE_MAX)


@pytest.fixture(scope="module", params=list(TRACKS))
def traj(request):
    return TRACKS[request.param]()


class TestRaceGate:
    def test_lsy_geometry(self):
        g = RaceGate((0.0, 0.0, 1.195), yaw=0.0)
        assert g.OPENING == 0.4 and g.OUTER == 0.72
        np.testing.assert_allclose(g.normal, [1.0, 0.0, 0.0])

    def test_gate_frame_roundtrip(self):
        g = RaceGate((1.0, -2.0, 0.7), yaw=0.9)
        pts = np.array([[1.0, -2.0, 0.7], [1.0 + g.normal[0], -2.0 + g.normal[1], 0.7]])
        local = g.to_gate_frame(pts)
        np.testing.assert_allclose(local[0], np.zeros(3), atol=1e-12)
        np.testing.assert_allclose(local[1], [1.0, 0.0, 0.0], atol=1e-12)


class TestFreestyle:
    def test_starts_at_rest(self, traj):
        np.testing.assert_allclose(traj.vel(0.0), np.zeros(3), atol=1e-9)
        np.testing.assert_allclose(traj.acc(0.0), np.zeros(3), atol=1e-9)

    def test_ends_at_rest(self, traj):
        np.testing.assert_allclose(traj.vel(traj.duration), np.zeros(3), atol=1e-7)

    def test_c1_continuity_everywhere(self, traj):
        for tb in traj._t0s_arr[1:]:
            np.testing.assert_allclose(traj.pos(tb - EPS), traj.pos(tb + EPS), atol=1e-3)
            np.testing.assert_allclose(traj.vel(tb - EPS), traj.vel(tb + EPS), atol=1e-2)

    def test_gate_crossings_exact(self, traj):
        for g, tg in zip(traj.gates, traj.gate_times):
            np.testing.assert_allclose(traj.pos(tg), g.center, atol=1e-6)
            v = traj.vel(tg)
            cos = v @ g.normal / np.linalg.norm(v)
            assert cos > 0.999  # crossing velocity along the gate normal

    def test_gate_corridor_clearance(self, traj):
        rep = feasibility_report(traj)
        for c in rep["gate_max_inplane_offset"]:
            assert c <= rep["gate_clearance_bound"], rep

    def test_thrust_feasible(self, traj):
        rep = feasibility_report(traj, twr=TWR)
        assert rep["max_thrust_acc_outside_arc"] <= 0.95 * TWR * GRAVITY, rep
        assert rep["min_thrust_acc_outside_arc"] >= 0.5, rep  # no negative thrust
        assert rep["min_z"] > 0.15, rep
        assert rep["feasible"], rep

    def test_rotation_rate_within_limits(self, traj):
        rep = feasibility_report(traj)
        assert rep["peak_ref_rate"] <= 0.76 * RATE_MAX_RP

    def test_flip_completes_full_rotation(self, traj):
        for f in traj.flips:
            t = np.linspace(f["t_rot_start"], f["t_rot_end"], 2001)
            ang = np.unwrap(traj.att_ref_rotvec(t)[:, f["axis"]])
            total = ang[-1] - ang[0]
            np.testing.assert_allclose(abs(total), 2 * np.pi, atol=1e-6)
            assert np.sign(total) == f["direction"]

    def test_attitude_level_outside_windows(self, traj):
        t = np.linspace(0, traj.duration, 3001)
        outside = np.ones(len(t), dtype=bool)
        for f in traj.flips:
            outside &= ~((t >= f["t_rot_start"] - 1e-3) & (t <= f["t_rot_end"] + 1e-3))
        assert np.abs(traj.att_ref_rotvec(t)[outside]).max() < 1e-9

    def test_derivative_consistency(self, traj):
        # sample away from segment boundaries (acc steps at flip edges are by design)
        t = np.linspace(0.1, traj.duration - 0.1, 400)
        keep = np.all(np.abs(t[:, None] - traj._t0s_arr[None, :]) > 0.01, axis=1)
        t = t[keep]
        num_vel = (traj.pos(t + EPS) - traj.pos(t - EPS)) / (2 * EPS)
        np.testing.assert_allclose(num_vel, traj.vel(t), atol=1e-3)

    def test_maneuver_descriptor_windows(self, traj):
        if not traj.flips:
            d = traj.maneuver_descriptor(np.linspace(0, traj.duration, 7))
            np.testing.assert_allclose(d, 0.0)  # flip-free track: zeros throughout
            return
        f = traj.flips[0]
        mid = 0.5 * (f["t_rot_start"] + f["t_rot_end"])
        d = traj.maneuver_descriptor(mid)
        assert d[f["axis"]] == f["direction"] and d[5] == 1.0
        d0 = traj.maneuver_descriptor(0.0)
        assert d0[5] == 0.0  # not active at start
        # countdown = clip(t0 - t, 0, 1) saturates at 1 when the flip is >1s ahead,
        # matching how flip episodes condition from t=0 in training
        np.testing.assert_allclose(d0[3], 1.0)
        before = traj.maneuver_descriptor(f["t_rot_start"] - 0.5)
        np.testing.assert_allclose(before[3], 0.5, atol=1e-9)
        assert before[5] == 0.0

    def test_traveling_flip_drift_preserved(self, traj):
        for f in traj.flips:
            v0 = traj.vel(f["t_rot_start"] - EPS)
            v1 = traj.vel(f["t_rot_end"] + EPS)
            np.testing.assert_allclose(v0[:2], v1[:2], atol=1e-6)  # horizontal conserved

    def test_shapes_and_clamping(self, traj):
        assert traj.pos(1.0).shape == (3,)
        assert traj.pos(np.linspace(0, traj.duration, 7)).shape == (7, 3)
        assert traj.maneuver_descriptor(np.linspace(0, 1, 5)).shape == (5, 6)
        np.testing.assert_allclose(traj.pos(traj.duration + 5.0), traj.pos(traj.duration))
