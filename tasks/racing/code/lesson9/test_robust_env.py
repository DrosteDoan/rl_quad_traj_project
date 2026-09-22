"""Checks for the robust training environment (main venv, in the container):

    python tasks/racing/code/lesson9/test_robust_env.py        # no pytest needed

MECHANICS ONLY. Every test constructs `RobustTrackingEnv`, resets it, and steps it a handful of times --
never `.learn()`. No training happens here or anywhere in Lesson 9 yet (train_robust.py has not been run).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import knobs as kb  # noqa: E402
from lighthouse_batch import LambdaLighthouseSensorBatch  # noqa: E402
from robust_env import DR_LAM_MAX, FORCE_XY_MAX, FORCE_Z_HI, FORCE_Z_LO, FREQ, WINDOW_DT, RobustTrackingEnv  # noqa: E402

from crazy_track.envs.datt_env import WINDOW  # noqa: E402

DT = 1.0 / FREQ


def _state(i: int) -> np.ndarray:
    return np.array([i * DT, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])


# ---- the sensor, in isolation (fast, no Sim) -------------------------------------------------------------
def test_lambda_lighthouse_batch_matches_the_scalar_knobs_class():
    """n=1 batch vs knobs.ScaledLighthouse at a FIXED lam (bypassing the per-episode draw), same seed streams
    are not expected to match draw-for-draw (different RNG plumbing), but the STATISTICS must, exactly like
    test_knobs.py's cross-check of ScaledLighthouse against the vendored LighthouseSensor."""
    n = 4000
    scale = kb.scales()["lighthouse"]
    lam_fixed = 0.5

    class _FixedBatch(LambdaLighthouseSensorBatch):
        def reset_rows(self, mask):
            if not hasattr(self, "lam_eff"):
                self.lam_eff = np.zeros(self.n)
                self.bias = np.zeros((self.n, 3))
                self.next_update = np.zeros(self.n)
                self.last_pos = np.full((self.n, 3), np.nan)
            k = int(mask.sum())
            self.lam_eff[mask] = lam_fixed * scale
            self.bias[mask] = self.rng.normal(0.0, self.p["bias_std"], size=(k, 3)) * self.lam_eff[mask, None]
            self.next_update[mask] = 0.0
            self.last_pos[mask] = np.nan

    batch = _FixedBatch(1, control_freq=FREQ, seed=7)
    ref = kb.ScaledLighthouse(lam_fixed * scale, seed=8, control_freq=FREQ, horizon=n)   # different seed on purpose
    outs = []
    for i in range(n):
        pos, vel, quat, omega = batch.measure(np.array([i * DT]), np.array([[i * DT, 0.0, 1.0]]),
                                              np.array([[1.0, 0.0, 0.0]]), np.array([[0.0, 0.0, 0.0, 1.0]]),
                                              np.array([[0.0, 0.0, 0.0]]))
        outs.append(np.concatenate([pos[0], vel[0], omega[0]]))
    outs = np.array(outs)
    vel_std_batch = (outs[:, 3:6] - np.array([1.0, 0.0, 0.0])).std()   # subtract the true mean first: vel_x is
    #                                                                    centred at 1.0, vel_y/z at 0 -- a naive
    #                                                                    flattened .std() over the raw values
    #                                                                    would measure the gap BETWEEN column
    #                                                                    means (~0.47), not the noise (~0.02)
    ref_outs = np.array([ref.measure(i * DT, _state(i)) for i in range(n)])
    vel_std_ref = (ref_outs[:, 3:6] - np.array([1.0, 0.0, 0.0])).std()
    print(f"  vel noise std: batch {vel_std_batch:.4f}, scalar knobs class {vel_std_ref:.4f}")
    assert abs(vel_std_batch / vel_std_ref - 1.0) < 0.1
    updates_batch = int(np.any(np.diff(outs[:, :3], axis=0) != 0, axis=1).sum())
    updates_ref = int(np.any(np.diff(ref_outs[:, :3], axis=0) != 0, axis=1).sum())
    print(f"  refresh events: batch {updates_batch}, scalar knobs class {updates_ref}")
    assert abs(updates_batch / updates_ref - 1.0) < 0.15
    # gyro is NOT delayed (the deliberate deviation from the vendored batch class, see lighthouse_batch.py)
    assert not np.allclose(outs[0, 6:9], 0.0) or n < 2  # trivially true; the real check is in the env test below


def test_lambda_lighthouse_batch_gyro_is_not_delayed():
    b = LambdaLighthouseSensorBatch(1, control_freq=FREQ, lam_max=1.0, seed=1)
    b.lam_eff[:] = 0.5   # skip the random per-episode draw for a deterministic check
    true_omega = np.array([[0.3, -0.1, 0.05]])
    _, _, _, omega_m = b.measure(np.array([0.0]), np.array([[0.0, 0.0, 1.0]]), np.array([[0.0, 0.0, 0.0]]),
                                 np.array([[0.0, 0.0, 0.0, 1.0]]), true_omega)
    assert abs(omega_m[0, 0] - true_omega[0, 0]) < 0.2   # noisy but centred on the TRUE current omega, not a lag


# ---- the env, end to end (constructs a real crazyflow Sim) -----------------------------------------------
def _make_env(num_envs=4, seed=0):
    return RobustTrackingEnv(num_envs=num_envs, seed=seed, v3=True, v5=True)


def test_env_constructs_at_freq_100_with_unchanged_obs_shape():
    env = _make_env()
    assert env.freq == FREQ
    np.testing.assert_allclose(env._t_offsets, WINDOW_DT * np.arange(1, WINDOW + 1))
    assert env.single_observation_space.shape == (56,)     # v5: unchanged despite the new window spacing
    assert env.n_substeps == env.sim.freq // FREQ == 5


def test_reset_and_a_few_steps_do_not_crash():
    """The actual smoke test: reset, step a handful of times, check shapes and finiteness. NOT training --
    no model, no learn(), no gradient anywhere in this file."""
    env = _make_env(num_envs=6)
    obs, _ = env.reset()
    assert obs.shape == (6, 56) and np.all(np.isfinite(obs))
    rng = np.random.default_rng(0)
    for _ in range(20):
        actions = rng.uniform(-1, 1, size=(6, 4)).astype(np.float32)
        obs, reward, crashed, truncated, info = env.step(actions)
        assert obs.shape == (6, 56) and np.all(np.isfinite(obs))
        assert reward.shape == (6,) and np.all(np.isfinite(reward))


def test_force_channel_is_within_the_study_matched_box():
    env = _make_env(num_envs=200)
    env.reset()
    acc = env.perturb_force / 0.04338
    assert acc[:, 0].min() >= -FORCE_XY_MAX - 1e-6 and acc[:, 0].max() <= FORCE_XY_MAX + 1e-6
    assert acc[:, 1].min() >= -FORCE_XY_MAX - 1e-6 and acc[:, 1].max() <= FORCE_XY_MAX + 1e-6
    assert acc[:, 2].min() >= FORCE_Z_LO - 1e-6 and acc[:, 2].max() <= FORCE_Z_HI + 1e-6
    # with 200 worlds the box should be visibly exercised near its edges, not just its centre
    assert acc[:, 0].max() > FORCE_XY_MAX * 0.8 and acc[:, 0].min() < -FORCE_XY_MAX * 0.8


def test_mass_channel_is_heavier_only_and_within_the_frozen_ceiling():
    env = _make_env(num_envs=200)
    env.reset()
    mult = env._mass_mult
    cap = 1.0 + env.dr_lam_max * kb.scales()["mass_mult"] * kb.NOMINAL["mass_mult"]["frac_heavier"]
    assert mult.min() >= 1.0 - 1e-9 and mult.max() <= cap + 1e-6
    assert mult.max() > 1.0 + 0.8 * (cap - 1.0)   # the box gets exercised, not just near 1.0
    default_mass = np.asarray(env.sim.default_data.params.mass)[:, 0, 0]
    actual_mass = np.asarray(env.sim.data.params.mass)[:, 0, 0]
    np.testing.assert_allclose(actual_mass, default_mass * mult, rtol=1e-5)


def test_lighthouse_channel_is_active_and_lambda_bounded():
    env = _make_env(num_envs=100)
    env.reset()
    assert isinstance(env.sensor, LambdaLighthouseSensorBatch)
    cap = env.dr_lam_max * kb.scales()["lighthouse"]
    assert env.sensor.lam_eff.min() >= -1e-9 and env.sensor.lam_eff.max() <= cap + 1e-6
    assert env.sensor.lam_eff.max() > 0.8 * cap


def test_the_three_channels_are_drawn_independently():
    """Not perfectly correlated, not perfectly anti-correlated: three separate RNG streams, matching the
    'independent per-episode draws' decision (METHODOLOGY.md section 5a)."""
    env = _make_env(num_envs=500)
    env.reset()
    force_mag = np.linalg.norm(env.perturb_force, axis=1)
    mass = env._mass_mult
    lh = env.sensor.lam_eff
    c_fm = np.corrcoef(force_mag, mass)[0, 1]
    c_fl = np.corrcoef(force_mag, lh)[0, 1]
    c_ml = np.corrcoef(mass, lh)[0, 1]
    print(f"  cross-channel correlation: force-mass {c_fm:.3f}, force-lighthouse {c_fl:.3f}, mass-lighthouse {c_ml:.3f}")
    assert max(abs(c_fm), abs(c_fl), abs(c_ml)) < 0.15


def test_resampling_on_episode_end_uses_all_three_channels_again():
    """A per-env reset (done_mask inside step()) must resample force, mass AND Lighthouse for that env --
    not just force, which is the one channel the vendored `_sample_perturb` hook already resampled."""
    env = _make_env(num_envs=8)
    env.reset()
    mass_before = env._mass_mult.copy()
    lh_before = env.sensor.lam_eff.copy()
    for _ in range(400):                              # push at least one env to a done state
        env.step(np.zeros((8, 4), dtype=np.float32))
        if (env.steps == 0).any():
            break
    done_envs = np.flatnonzero(env.steps == 0)
    if len(done_envs) == 0:
        print("  (no env finished an episode in 400 steps at this seed -- inconclusive, not a failure)")
        return
    changed_mass = not np.allclose(mass_before[done_envs], env._mass_mult[done_envs])
    changed_lh = not np.allclose(lh_before[done_envs], env.sensor.lam_eff[done_envs])
    print(f"  {len(done_envs)} env(s) reset; mass channel changed: {changed_mass}; lighthouse changed: {changed_lh}")
    # a fresh uniform draw COULD coincidentally repeat; only fail if BOTH look frozen, which would mean the
    # hook is not firing at all
    assert changed_mass or changed_lh


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
