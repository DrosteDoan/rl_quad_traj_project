import numpy as np
import pytest

from crazy_track.trajectories import ChainedPolyTrajectory, ZigzagTrajectory

EPS = 1e-4


@pytest.fixture
def rng():
    return np.random.default_rng(0)


class TestChainedPoly:
    def test_starts_at_rest_at_origin(self, rng):
        traj = ChainedPolyTrajectory.random(rng)
        np.testing.assert_allclose(traj.pos(0.0), np.zeros(3), atol=1e-9)
        np.testing.assert_allclose(traj.vel(0.0), np.zeros(3), atol=1e-9)
        np.testing.assert_allclose(traj.acc(0.0), np.zeros(3), atol=1e-9)

    def test_c2_continuity_at_knots(self, rng):
        traj = ChainedPolyTrajectory.random(rng, duration=10.0, seg_duration=2.0)
        for tk in traj.knot_times[1:-1]:
            for fn in (traj.pos, traj.vel, traj.acc):
                left, right = fn(tk - EPS), fn(tk + EPS)
                np.testing.assert_allclose(left, right, atol=1e-2)

    def test_derivative_consistency(self, rng):
        """vel/acc match finite differences of pos."""
        traj = ChainedPolyTrajectory.random(rng)
        t = np.linspace(0.1, traj.duration - 0.1, 50)
        num_vel = (traj.pos(t + EPS) - traj.pos(t - EPS)) / (2 * EPS)
        np.testing.assert_allclose(num_vel, traj.vel(t), atol=1e-4)
        num_acc = (traj.vel(t + EPS) - traj.vel(t - EPS)) / (2 * EPS)
        np.testing.assert_allclose(num_acc, traj.acc(t), atol=1e-4)

    def test_shapes_and_clamping(self, rng):
        traj = ChainedPolyTrajectory.random(rng)
        assert traj.pos(1.0).shape == (3,)
        assert traj.pos(np.linspace(0, 20, 7)).shape == (7, 3)
        np.testing.assert_allclose(traj.pos(traj.duration + 5.0), traj.pos(traj.duration))

    def test_ref_window(self, rng):
        traj = ChainedPolyTrajectory.random(rng)
        win = traj.ref_window(1.0, horizon=10, dt=0.02)
        assert win.shape == (10, 3)
        np.testing.assert_allclose(win[0], traj.pos(1.0))


class TestZigzag:
    def test_starts_at_origin(self, rng):
        traj = ZigzagTrajectory.random(rng)
        np.testing.assert_allclose(traj.pos(0.0), np.zeros(3), atol=1e-9)

    def test_hits_waypoints(self, rng):
        traj = ZigzagTrajectory.random(rng)
        for tk, wp in zip(traj.knot_times, traj.waypoints):
            if tk <= traj.duration:
                np.testing.assert_allclose(traj.pos(tk), wp, atol=1e-9)

    def test_position_continuous_velocity_discontinuous(self, rng):
        traj = ZigzagTrajectory.random(rng)
        interior = traj.knot_times[1:-1]
        vel_jumps = [
            np.linalg.norm(traj.vel(tk + EPS) - traj.vel(tk - EPS)) for tk in interior
        ]
        for tk in interior:
            np.testing.assert_allclose(traj.pos(tk - EPS), traj.pos(tk + EPS), atol=1e-2)
        assert max(vel_jumps) > 0.1  # infeasible by construction

    def test_shapes(self, rng):
        traj = ZigzagTrajectory.random(rng)
        assert traj.vel(1.0).shape == (3,)
        assert traj.acc(np.linspace(0, 5, 4)).shape == (4, 3)


class TestBallisticFlip:
    """Dynamic-feasibility guarantees of the paper-2 flip reference."""

    def _traj(self, **kw):
        from crazy_track.trajectories import BallisticFlipTrajectory
        defaults = dict(hover=(0.0, 0.0, 2.0), t0=2.0, Tb=0.7, axis=0,
                        direction=1, a_boost=7.0, duration=6.0)
        defaults.update(kw)
        return BallisticFlipTrajectory(**defaults)

    def test_returns_to_hover_at_rest(self):
        traj = self._traj()
        np.testing.assert_allclose(traj.pos(traj.t_end), [0.0, 0.0, 2.0], atol=1e-9)
        np.testing.assert_allclose(traj.vel(traj.t_end), np.zeros(3), atol=1e-9)
        np.testing.assert_allclose(traj.pos(traj.duration), [0.0, 0.0, 2.0], atol=1e-9)

    def test_derivative_consistency(self):
        traj = self._traj()
        t = np.linspace(0.1, traj.duration - 0.1, 1200)
        dt = t[1] - t[0]
        v_fd = (traj.pos(t + dt / 2) - traj.pos(t - dt / 2)) / dt
        np.testing.assert_allclose(v_fd, traj.vel(t), atol=0.05)

    def test_velocity_continuous_at_phase_boundaries(self):
        traj = self._traj()
        for tb in (traj.t0, traj.t_rot_start, traj.t_rot_end, traj.t_end):
            np.testing.assert_allclose(traj.vel(tb - 1e-6), traj.vel(tb + 1e-6),
                                       atol=1e-3)

    def test_thrust_feasible_everywhere(self):
        """Required thrust accel ||acc + g z|| within [0, TWR*g] at all times."""
        traj = self._traj()
        t = np.linspace(0.0, traj.duration, 2000)
        a_req = traj.acc(t)
        a_req[..., 2] += 9.81
        mag = np.linalg.norm(a_req, axis=-1)
        assert mag.max() <= 1.88 * 9.81 * 0.95  # margin under cf21B TWR
        assert mag.min() >= -1e-9  # ballistic phase: exactly zero thrust

    def test_never_below_start_altitude(self):
        traj = self._traj()
        t = np.linspace(0.0, traj.duration, 2000)
        assert traj.pos(t)[..., 2].min() >= 2.0 - 1e-9

    def test_attitude_sweeps_full_rotation_within_rate_limit(self):
        traj = self._traj()
        t = np.linspace(traj.t_rot_start, traj.t_rot_end, 2001)
        rv = traj.att_ref_rotvec(t)[..., 0]
        theta = np.unwrap(rv)
        np.testing.assert_allclose(theta[-1] - theta[0], 2 * np.pi, atol=1e-6)
        rate = np.abs(np.diff(theta)) / (t[1] - t[0])
        assert rate.max() <= 15.0 * 0.8  # <= 80% of RATE_MAX roll/pitch
        # level outside the rotation window
        assert np.allclose(traj.att_ref_rotvec(traj.t0 - 0.1), 0.0)
        assert np.allclose(traj.att_ref_rotvec(traj.t_end + 0.1), 0.0)

    def test_zero_thrust_during_rotation(self):
        """The rotation happens only where the reference demands free fall."""
        traj = self._traj()
        t = np.linspace(traj.t_rot_start + 1e-3, traj.t_rot_end - 1e-3, 500)
        np.testing.assert_allclose(traj.acc(t)[..., 2], -9.81, atol=1e-9)
