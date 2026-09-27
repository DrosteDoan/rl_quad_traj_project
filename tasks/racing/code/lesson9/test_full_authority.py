"""Checks for `full_authority_env.py` (main venv, in the container):

    python tasks/racing/code/lesson9/test_full_authority.py        # no pytest needed

MECHANICS ONLY, like test_robust_env.py/test_contrast_env.py. Never `.learn()`. Also checks driver.py's
`robust_full:` spec routing, without constructing a PPO model. `train_full_authority.py` (the pre-gate-aware
standalone trainer this file used to also check) was deleted 2026-09-27 as unrepresentative of the study --
every reported full-authority checkpoint came from `train_gate_aware.py --group robust_full` instead, whose
CLI is tested in `test_gate_aware.py` and `test_train_scripts_cli.py`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from full_authority_env import RPY_SCALE, FullAuthorityTrackingEnv  # noqa: E402
from robust_env import RobustTrackingEnv  # noqa: E402

from crazy_track.controllers.utils import RPY_MAX  # noqa: E402
from crazy_track.envs.datt_env import WINDOW  # noqa: E402


def _make_env(num_envs=4, seed=0):
    return FullAuthorityTrackingEnv(num_envs=num_envs, seed=seed, v3=True, v5=True)


def test_rpy_scale_is_1_not_the_vendored_0_7():
    assert RPY_SCALE == 1.0


def test_matches_robust_on_everything_except_the_action_scale():
    f, r = _make_env(), RobustTrackingEnv(num_envs=4, seed=0, v3=True, v5=True)
    assert f.freq == r.freq == 100
    np.testing.assert_allclose(f._t_offsets, r._t_offsets)
    assert f.single_observation_space.shape == r.single_observation_space.shape == (56,)
    # same disturbance training: both draw force/lighthouse/mass the same way (RobustTrackingEnv unchanged)
    assert hasattr(f, "_mass_rng") and hasattr(f, "dr_lam_max")


def test_denorm_action_uses_the_full_rpy_range():
    """The actual point of the class: a raw action of +-1 maps to +-RPY_MAX, not +-0.7*RPY_MAX."""
    env = _make_env(num_envs=3)
    env.reset()
    a = np.ones((3, 4), dtype=np.float32)          # saturated raw action on every channel
    cmd = env._denorm_action(a)
    np.testing.assert_allclose(cmd[:, 0, 0], RPY_MAX, rtol=1e-5)
    np.testing.assert_allclose(cmd[:, 0, 1], RPY_MAX, rtol=1e-5)
    a_neg = -np.ones((3, 4), dtype=np.float32)
    cmd_neg = env._denorm_action(a_neg)
    np.testing.assert_allclose(cmd_neg[:, 0, 0], -RPY_MAX, rtol=1e-5)


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


def test_driver_routes_robust_full_spec_to_the_right_controller_class():
    import driver as drv

    assert "robust_full:" not in drv.M1  # sanity: not accidentally matching an unrelated spec
    # make_race_controller must import FullAuthorityPolicyController for this prefix, not RobustPolicyController;
    # checked by prefix dispatch, not by constructing one (no trained checkpoint exists yet)
    import inspect

    src = inspect.getsource(drv.make_race_controller)
    assert 'spec.startswith("robust_full:")' in src
    assert "FullAuthorityPolicyController" in src


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
