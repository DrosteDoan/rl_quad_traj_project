"""Checks for the lambda knobs (main venv, in the container):

    python tasks/racing/code/lesson9/test_knobs.py        # no pytest needed

The regression anchor of METHODOLOGY.md section 4: at lam_eff = 1 (Lesson 7's value; lam = 0.5 with the starting
scale of 2) the constant forces and the gust are IDENTICAL to `crazy_track.disturbances`, and the Lighthouse sensor
has the same distributions as `crazy_track.sensors.LighthouseSensor`. Plus the design properties: lam = 0 is no
object, one seed is the same draw at every lam, the latency is one step and the gyro is not delayed.
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import knobs as kb  # noqa: E402

from crazy_track.disturbances import ConstantWind, GustWind, Payload  # noqa: E402
from crazy_track.sensors import LighthouseSensor  # noqa: E402

DT = 0.01


def _state(i: int) -> np.ndarray:
    """A drone flying straight at 1 m/s, level, no rotation: [pos, vel, quat_xyzw, omega]."""
    return np.array([i * DT, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])


def test_nominal_table_matches_the_vendored_defaults():
    np.testing.assert_allclose(ConstantWind().f, kb.NOMINAL["wind_const"]["force"])
    np.testing.assert_allclose(Payload().f, [0.0, 0.0, -kb.NOMINAL["payload"]["extra_mass"] * kb.G])
    g, p = GustWind(), kb.NOMINAL["wind_gust"]
    np.testing.assert_allclose(g.mean, p["mean"])
    assert (g.gust_amp, g.gust_hz, g.ou_sigma, g.ou_tau, g.dt) == (p["gust_amp"], p["gust_hz"], p["ou_sigma"], p["ou_tau"], DT)
    d = {k: v.default for k, v in inspect.signature(LighthouseSensor.__init__).parameters.items()}
    lh = kb.NOMINAL["lighthouse"]
    for k in ("update_hz", "update_hz_std", "jitter_std", "bias_std", "vel_std", "att_std_deg", "gyro_std"):
        assert d[k] == lh[k], k


def test_lam_zero_is_no_object():
    for cond in kb.CONDITIONS:
        assert kb.make_conditions(cond, 0.0, seed=3) == (None, None), cond


def test_constant_forces_at_lesson7_value_are_identical():
    d, s = kb.make_conditions("wind_const", 0.5, scale={"wind_const": 2.0})
    assert s is None
    np.testing.assert_array_equal(d.force(0.0, None), ConstantWind().f)
    d, s = kb.make_conditions("payload", 0.5, scale={"payload": 2.0})
    np.testing.assert_allclose(d.force(0.0, None), Payload().f, atol=1e-15)


def test_gust_at_lesson7_value_is_identical_to_gustwind_and_scales_linearly():
    ref = GustWind(seed=3)
    new = kb.ScaledGust(1.0, seed=3)
    for i in range(1000):
        t = i * DT
        np.testing.assert_allclose(new.force(t, None), ref.force(t, None), atol=1e-12, err_msg=f"step {i}")
    half = kb.ScaledGust(0.5, seed=3)
    for i in (0, 10, 500, 999):
        np.testing.assert_allclose(half.force(i * DT, None), 0.5 * new.force(i * DT, None), atol=1e-14)
    other = kb.ScaledGust(1.0, seed=4)
    assert not np.allclose(other.force(3.0, None), new.force(3.0, None))               # a different seed differs


def test_lighthouse_streams_are_paired_across_lam():
    a, b = kb.ScaledLighthouse(0.4, seed=5), kb.ScaledLighthouse(0.8, seed=5)
    np.testing.assert_allclose(b.bias, 2.0 * a.bias, atol=1e-15)
    for i in range(300):
        oa, ob = a.measure(i * DT, _state(i)), b.measure(i * DT, _state(i))
        np.testing.assert_allclose(ob[3:6] - _state(i - 1)[3:6] if i else ob[3:6] - _state(i)[3:6],
                                   2.0 * (oa[3:6] - _state(i - 1)[3:6] if i else oa[3:6] - _state(i)[3:6]), atol=1e-12)
        np.testing.assert_allclose(ob[10:13], 2.0 * oa[10:13], atol=1e-12)               # gyro noise, undelayed


def test_lighthouse_latency_is_one_step_and_the_gyro_is_not_delayed():
    s = kb.ScaledLighthouse(1.0, seed=2)
    prev_vel_noise = None
    for i in range(50):
        out = s.measure(i * DT, _state(i))
        vel_noise_now = s.vel_std * s._z["vel"][i]
        expected_vel = _state(i)[3:6] + (prev_vel_noise if prev_vel_noise is not None else vel_noise_now)
        np.testing.assert_allclose(out[3:6], expected_vel, atol=1e-12, err_msg=f"step {i}")
        np.testing.assert_allclose(out[10:13], s.gyro_std * s._z["gyro"][i], atol=1e-12)
        prev_vel_noise = vel_noise_now


def test_lighthouse_at_zero_is_a_perfect_sensor_apart_from_the_latency():
    s = kb.ScaledLighthouse(0.0, seed=1)
    for i in range(40):
        out = s.measure(i * DT, _state(i))
        expected_pos = _state(max(i - 1, 0))[:3]                                        # updates every step, 1 step old
        np.testing.assert_allclose(out[:3], expected_pos, atol=1e-12)
        np.testing.assert_allclose(out[3:6], _state(max(i - 1, 0))[3:6], atol=1e-12)


def _stats(make_sensor, n: int = 20000):
    """Noise statistics of a sensor on the straight-flight state: velocity, attitude, gyro, jitter, refresh rate."""
    s = make_sensor(0)
    outs = np.array([s.measure(i * DT, _state(i)) for i in range(n)])
    vel = outs[:, 3:6] - np.array([1.0, 0.0, 0.0])
    ang = np.linalg.norm(np.array([  # rotation angle of the attitude error, rad
        2 * np.arctan2(np.linalg.norm(q[:3]), abs(q[3])) for q in outs[:, 6:10]]))
    att_rms = np.sqrt(np.mean(np.array([2 * np.arctan2(np.linalg.norm(q[:3]), abs(q[3])) for q in outs[:, 6:10]]) ** 2))
    changes = np.flatnonzero(np.any(np.diff(outs[:, :3], axis=0) != 0.0, axis=1)) + 1     # ZOH refresh events
    jit = np.array([outs[i, :3] - _state(i - 1)[:3] - s.bias for i in changes if i >= 1])
    return {"vel_std": vel.std(), "att_rms_deg": np.degrees(att_rms), "gyro_std": outs[:, 10:13].std(),
            "updates": len(changes), "jitter_std": jit.std(), "_": ang}


def test_lighthouse_at_lesson7_value_has_the_vendored_distributions():
    ref = _stats(lambda k: LighthouseSensor(control_freq=100, latency_steps=1, seed=11 + k))
    new = _stats(lambda k: kb.ScaledLighthouse(1.0, seed=11 + k, control_freq=100, horizon=20000))
    for key in ("vel_std", "att_rms_deg", "gyro_std", "updates"):
        assert abs(new[key] / ref[key] - 1.0) < 0.05, (key, new[key], ref[key])
    assert abs(new["jitter_std"] / ref["jitter_std"] - 1.0) < 0.08, (new["jitter_std"], ref["jitter_std"])
    # the per-run bias: std over many seeds, 0.015 m per axis
    b_ref = np.array([LighthouseSensor(control_freq=100, latency_steps=1, seed=s).bias for s in range(600)])
    b_new = np.array([kb.ScaledLighthouse(1.0, seed=s).bias for s in range(600)])
    assert abs(b_new.std() / b_ref.std() - 1.0) < 0.08 and abs(b_new.std() - 0.015) < 0.0015


def test_refresh_interval_grows_with_lam():
    counts = []
    for lam in (0.0, 0.25, 0.5, 1.0):
        s = kb.ScaledLighthouse(lam, seed=6)                                    # lam here is lam_eff
        outs = np.array([s.measure(i * DT, _state(i))[:3] for i in range(5000)])
        counts.append(int((np.any(np.diff(outs, axis=0) != 0.0, axis=1)).sum()))
    assert counts[0] > 4900 and counts == sorted(counts, reverse=True) and counts[3] < counts[2] * 0.75, counts
    # lam_eff = 1 is the vendored model: 1/clip(N(34, 18) Hz, 8, 100) has a mean interval of ~45 ms (Jensen), i.e. ~22 Hz
    assert 900 < counts[3] < 1300, counts


def test_latency_only_is_the_delay_and_nothing_else():
    d, s = kb.make_conditions("latency_only", 0.0)
    assert (d, s) == (None, None)
    d, s = kb.make_conditions("latency_only", 0.7, seed=4)
    assert d is None and s.lam == 0.0
    for i in range(20):
        out = s.measure(i * DT, _state(i))
        np.testing.assert_allclose(out[:6], _state(max(i - 1, 0))[:6], atol=1e-12)
        np.testing.assert_allclose(out[10:13], 0.0, atol=1e-12)


def test_mass_scale_is_one_except_mass_mult():
    for cond in kb.CONDITIONS:
        if cond == "mass_mult":
            continue
        assert kb.mass_scale(cond, 0.7) == 1.0, cond
    assert kb.mass_scale("mass_mult", 0.0) == 1.0
    sc = kb.scales()
    a, b = kb.mass_scale("mass_mult", 0.4), kb.mass_scale("mass_mult", 0.8)
    assert a > 1.0 and b > a                                              # heavier only, monotone in lam
    np.testing.assert_allclose(a - 1.0, 0.4 * sc["mass_mult"] * kb.NOMINAL["mass_mult"]["frac_heavier"])
    np.testing.assert_allclose(b - 1.0, 2.0 * (a - 1.0), atol=1e-12)      # linear in lam
    for bad in (-0.1, 1.1):
        try:
            kb.mass_scale("mass_mult", bad)
        except ValueError:
            continue
        raise AssertionError("a lam outside [0, 1] must be refused")


def test_mass_mult_at_lesson8_value_is_the_level1_extreme():
    mult = kb.mass_scale("mass_mult", 1.0, scale={"mass_mult": 1.0})
    m = 0.04338
    np.testing.assert_allclose((mult - 1.0) * m, 0.005, atol=1e-5)        # Level 1's +-0.005 kg heavy end


def test_make_conditions_shapes_and_validation():
    d, s = kb.make_conditions("combined", 0.3, seed=1)
    sc = kb.scales()                                             # the frozen ceilings
    gust, pay = kb.ScaledGust(0.3 * sc['wind_gust'], 1), kb.ScaledPayload(0.3 * sc['payload'])
    np.testing.assert_allclose(d.force(1.0, None), gust.force(1.0, None) + pay.force(1.0, None), atol=1e-14)
    assert isinstance(s, kb.ScaledLighthouse)
    d, s = kb.make_conditions("lighthouse", 0.3)
    assert d is None and isinstance(s, kb.ScaledLighthouse)
    for bad in (-0.1, 1.1):
        try:
            kb.make_conditions("wind_const", bad)
        except ValueError:
            continue
        raise AssertionError("a lam outside [0, 1] must be refused")
    assert abs(kb.describe("payload", 0.5)["payload_force_N"] - 0.0981) < 1e-6           # Lesson 7's 10 g


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
