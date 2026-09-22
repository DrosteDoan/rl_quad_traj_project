"""Checks for the contrast-group training environment (main venv, in the container):

    python tasks/racing/code/lesson9/test_contrast_env.py        # no pytest needed

MECHANICS ONLY, like test_robust_env.py. Never `.learn()` -- no training happens here or anywhere in
Lesson 9 yet (train_contrast.py has not been run).

The point of every test below is the SAME point the class exists to make: this env differs from
RacingTrackingEnv in exactly the window and the frequency, and from RobustTrackingEnv in exactly the
disturbance-training ranges (reverted to vendored defaults) -- nothing else, on either side.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from contrast_env import FREQ, WINDOW_DT, ContrastTrackingEnv  # noqa: E402
from robust_env import RobustTrackingEnv  # noqa: E402

from crazy_track.envs.datt_env import PERTURB_ACC_MAX, WINDOW  # noqa: E402
from crazy_track.sensors import LighthouseSensorBatch  # noqa: E402


def _make_env(num_envs=4, seed=0):
    return ContrastTrackingEnv(num_envs=num_envs, seed=seed, v3=True, v5=True)


def test_matches_robust_on_window_and_frequency():
    c, r = _make_env(), RobustTrackingEnv(num_envs=4, seed=0, v3=True, v5=True)
    assert c.freq == r.freq == FREQ == 100
    np.testing.assert_allclose(c._t_offsets, r._t_offsets)
    np.testing.assert_allclose(c._t_offsets, WINDOW_DT * np.arange(1, WINDOW + 1))
    assert c.single_observation_space.shape == r.single_observation_space.shape == (56,)


def test_uses_the_vendored_unmatched_force_box_not_the_study_matched_one():
    """No FORCE_XY_MAX/FORCE_Z_LO/HI override here: the vendored +-3.5 m/s^2 box, unlike RobustTrackingEnv's
    study-matched +-4.4/-3.6..+1.8."""
    env = _make_env(num_envs=300)
    env.reset()
    acc = env.perturb_force / 0.04338
    assert acc.max() <= PERTURB_ACC_MAX + 1e-6 and acc.min() >= -PERTURB_ACC_MAX - 1e-6
    assert acc[:, 0].max() > PERTURB_ACC_MAX * 0.8       # the vendored box is actually exercised
    # the vendored z-halving (acc[:,2] *= 0.5) is still in effect -- z should be visibly narrower than x/y
    assert acc[:, 2].max() < PERTURB_ACC_MAX * 0.6


def test_uses_the_vendored_noise_scale_sensor_not_the_lambda_coupled_one():
    env = _make_env(num_envs=4)
    assert isinstance(env.sensor, LighthouseSensorBatch)
    assert not hasattr(env, "_mass_rng")        # RobustTrackingEnv has this; ContrastTrackingEnv must not
    assert not hasattr(env, "dr_lam_max")       # RobustTrackingEnv's knob; meaningless here


def test_no_mass_channel_exists():
    env = _make_env(num_envs=50)
    env.reset()
    default_mass = np.asarray(env.sim.default_data.params.mass)
    actual_mass = np.asarray(env.sim.data.params.mass)
    np.testing.assert_allclose(actual_mass, default_mass)   # never touched, at any point


def test_reset_and_a_few_steps_do_not_crash():
    env = _make_env(num_envs=6)
    obs, _ = env.reset()
    assert obs.shape == (6, 56) and np.all(np.isfinite(obs))
    rng = np.random.default_rng(0)
    for _ in range(20):
        actions = rng.uniform(-1, 1, size=(6, 4)).astype(np.float32)
        obs, reward, crashed, truncated, info = env.step(actions)
        assert obs.shape == (6, 56) and np.all(np.isfinite(obs))
        assert reward.shape == (6,) and np.all(np.isfinite(reward))


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
