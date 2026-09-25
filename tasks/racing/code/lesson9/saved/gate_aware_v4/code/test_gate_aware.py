"""Checks for the gate-aware viability screen, version 4 (main venv, in the container):

    python tasks/racing/code/lesson9/test_gate_aware.py        # no pytest needed

MECHANICS ONLY, like every other lesson9 env test. Never `.learn()` -- no training happens here or anywhere
in Lesson 9 yet (train_gate_aware.py v4 has not been run). Version 2 flies the dev tracks from a REAL ground
start (v1 re-origined them airborne and scored 0/6, every failure a gate-1 contact), so the tests that pinned
"starts safely elevated" are gone; in their place are tests that a ground start does NOT trip the floor
crash, and that the floor threshold is the one the eval harness uses.
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import driver as drv  # noqa: E402
from gate_aware_env import (DEV_SEEDS, EPISODE_TIME, FLOOR_PENALTY, FLOOR_Z, GATE_PENALTY, MISS_WINDOW,  # noqa: E402
                            POOL_SPEC, POOL_TRAJS, GateAwareTrackingEnv, gate_contacts, gate_progress, summarise_gate_stats,
                            train_seeds)
from robust_env import RobustTrackingEnv  # noqa: E402

from race_refs import RACE_START  # noqa: E402


def _make_env(num_envs=4, seed=0):
    return GateAwareTrackingEnv(num_envs=num_envs, seed=seed, v3=True, v5=True)


def test_references_are_the_same_ground_start_trajectories_eval_builds():
    for (role, seed), traj in list(zip(POOL_SPEC, POOL_TRAJS))[::7]:          # every 7th: the pool is large
        ref = drv.build_traj(drv.load_track(role, seed))
        assert type(traj) is type(ref)
        np.testing.assert_allclose(traj.pos(0.0), RACE_START)
        np.testing.assert_allclose(traj.pos(np.array([0.5, 1.5, 3.0])), ref.pos(np.array([0.5, 1.5, 3.0])))
        assert traj.gate_times == ref.gate_times


def test_the_reference_really_does_start_on_the_ground():
    for traj in POOL_TRAJS:
        assert traj.pos(0.0)[2] < 0.05, "this screen exists to train a real ground launch"


def test_floor_threshold_is_the_eval_harnesss_not_the_vendored_one():
    assert FLOOR_Z == -0.3
    src = inspect.getsource(drv.Flyer.fly)
    assert "pos[2] < -0.3" in src, "driver.py's divergence check moved; FLOOR_Z must follow it"
    for traj in POOL_TRAJS:
        assert traj.pos(0.0)[2] > FLOOR_Z + 0.2


def test_ground_start_does_not_trip_the_floor_crash():
    """The whole reason FLOOR_Z moved: with the vendored 0.05, a z=0.01 start crashes on step one. The vendored
    `_set_states` adds +-0.05 m of noise to the start, so z0 is in [-0.04, 0.06]: some envs start above 0.05 (that is
    not a bug), and with 16 envs at least one starts below it -- exactly the ones the old threshold would kill."""
    env = _make_env(num_envs=16)
    env.reset()
    z0 = np.array(env.sim.data.states.pos)[:, 0, 2]
    assert (z0 < 0.07).all(), "envs should start at ground level (0.01 + noise)"
    assert (z0 < 0.05).any(), "no env started below the vendored floor threshold: this test proves nothing"
    _, reward, crashed, _, _ = env.step(np.zeros((16, 4), dtype=np.float32))
    assert not crashed.any(), f"a legitimate ground start was flagged as a crash: {crashed}"
    assert (reward > -3.5).all()


def test_episode_time_covers_every_dev_traj_with_margin():
    longest = max(t.duration for t in POOL_TRAJS)
    assert EPISODE_TIME > longest + 0.3, f"EPISODE_TIME={EPISODE_TIME} too tight against longest={longest:.2f}"


def test_gate_1_keeps_its_normal_lead_time():
    """v1 truncated gate 1 to 0.28-0.38 s after the episode start; v2 must give it the plan's real lead-in."""
    for traj in POOL_TRAJS:
        assert traj.gate_times[0] > 1.0, f"gate 1 at {traj.gate_times[0]:.2f} s: approach is truncated"
        assert all(b > a for a, b in zip(traj.gate_times, traj.gate_times[1:]))


def test_matches_robust_on_everything_except_reference_contact_and_floor():
    g, r = _make_env(num_envs=4), RobustTrackingEnv(num_envs=4, seed=0, v3=True, v5=True)
    assert g.freq == r.freq == 100
    np.testing.assert_allclose(g._t_offsets, r._t_offsets)
    assert g.single_observation_space.shape == r.single_observation_space.shape == (56,)
    assert g.max_steps == int(EPISODE_TIME * 100)


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


def test_sample_traj_reaches_every_pool_track():
    """Drawn directly (no sim needed): 2000 draws over the pool leave a given track unseen with probability
    (1 - 1/n)^2000, ~1e-13 at n=69."""
    env = _make_env(num_envs=4)
    seen = set()
    for k in range(2000):
        env._sample_traj(k % 4)
        seen.add(int(env._dev_idx[k % 4]))
    assert seen == set(range(len(POOL_TRAJS))), f"pool tracks never drawn: {sorted(set(range(len(POOL_TRAJS))) - seen)}"


def _seeds(role):
    return {int(p.stem.split("_")[1]) for p in (Path(__file__).resolve().parent / "tracks" / role).glob("track_*.json")}


def test_pool_is_larger_than_the_three_dev_tracks():
    assert len(POOL_SPEC) > len(DEV_SEEDS), "run gen_pool.py: v4 exists to train on a larger pool"
    assert len(train_seeds()) >= 20, f"only {len(train_seeds())} train tracks -- too few to test transfer"


def test_pool_is_disjoint_from_study_and_val_and_inside_its_fixed_windows():
    train, val, study, dev = _seeds("train"), _seeds("val"), _seeds("study"), _seeds("dev")
    assert train and val and study and dev
    assert not (train & study) and not (train & val) and not (train & dev), "the training pool leaks into another role"
    assert not (val & study) and not (val & dev)
    assert all(200000 <= x < 206000 for x in train), "a train track outside gen_pool.py's fixed seed windows"
    assert all(300000 <= x < 302000 for x in val), "a val track outside gen_pool.py's fixed seed windows"
    assert {r for r, _ in POOL_SPEC} == {"dev", "train"}, "val (or study) tracks must never enter the training pool"


def test_selection_and_untouched_study_tracks_are_not_in_the_pool():
    pool_seeds = {x for _, x in POOL_SPEC}
    assert not (pool_seeds & {4, 25, 93, 387, 504}), "a selection-set track is in the training pool"
    assert not (pool_seeds & {747, 757, 834, 837, 965, 989, 1089, 1145, 1250, 1318}), "an untouched study track leaked"


def test_gate_contact_triggers_a_crash():
    """Force env 0 onto a guaranteed-overlap pose (the centre of one gate's own contact box) and confirm
    step() reports it as crashed -- this checks the WIRING (grouping by _dev_idx, feeding into `crashed`),
    not the geometry test itself (test_contact.py already covers that in isolation)."""
    import jax.numpy as jnp

    env = _make_env(num_envs=3)
    env.reset()
    for i in range(3):
        env._traj[i] = POOL_TRAJS[i]
        env._dev_idx[i] = i
    gate = POOL_TRAJS[0].gates[0]
    # "top" box local centre (0, 0, 0.28), half-extents (0.01, 0.36, 0.08) -- placing the drone exactly
    # there guarantees SAT overlap regardless of orientation choice, so identity quat is enough
    contact_pos = np.asarray(gate.center) + np.asarray(gate.rotation) @ np.array([0.0, 0.0, 0.28])
    pos = np.array(env.sim.data.states.pos)
    quat = np.array(env.sim.data.states.quat)
    pos[0, 0] = contact_pos
    quat[0, 0] = np.array([0.0, 0.0, 0.0, 1.0])
    env.sim.data = env.sim.data.replace(states=env.sim.data.states.replace(
        pos=jnp.asarray(pos), quat=jnp.asarray(quat)))
    _, reward, crashed, _, _ = env.step(np.zeros((3, 4), dtype=np.float32))
    assert crashed[0], "a pose at a gate box's own centre must register as a contact crash"
    assert reward[0] == -GATE_PENALTY == -3.5
    assert env.stats["contact"] == 1 and env.stats["miss"] == 0


def test_gate_contacts_prefilter_is_exactly_the_unfiltered_check():
    """The distance pre-filter is an optimisation: on random poses (many within a metre of a gate, random
    attitudes, random tracks) it must agree with calling `pose_contacts` on every pose of every track group."""
    import contact as ct
    from scipy.spatial.transform import Rotation as R

    rng = np.random.default_rng(3)
    n = 4000
    idx = rng.integers(0, len(POOL_TRAJS), size=n)
    pos = np.zeros((n, 3))
    for k in range(n):
        g = POOL_TRAJS[int(idx[k])].gates[int(rng.integers(0, 4))]
        pos[k] = np.asarray(g.center) + rng.normal(0.0, 0.35, size=3)
    quat = R.random(n, random_state=3).as_quat()
    naive = np.zeros(n, dtype=bool)
    for i in np.unique(idx):
        grp = np.flatnonzero(idx == i)
        naive[grp] = ct.pose_contacts(pos[grp], quat[grp], POOL_TRAJS[int(i)].gates)[0]
    fast = gate_contacts(pos, quat, idx)
    assert naive.sum() > 100, "the test poses should include plenty of real contacts"
    assert (fast == naive).all(), f"{int((fast != naive).sum())} of {n} disagree"


# ---- v3: missed-gate termination (driver.py's rule), penalties, logging ------------------------------------
def _cross(gate, t_frac_offset=0.0, lateral=0.0, dz=0.0):
    """prev/cur positions straddling `gate`'s plane, `lateral` metres off-centre in the gate's y."""
    n = np.asarray(gate.normal); y = np.asarray(gate.rotation)[:, 1]
    c = np.asarray(gate.center) + lateral * y + np.array([0.0, 0.0, dz])
    return c - 0.02 * n, c + 0.02 * n


def test_gate_progress_advances_on_a_clean_crossing():
    traj = POOL_TRAJS[0]; g = traj.gates
    prev, cur = _cross(g[0])
    idx, missed = gate_progress(prev, cur, traj.gate_times[0], 0.01, g, traj.gate_times, 0)
    assert (idx, missed) == (1, False)


def test_gate_progress_ignores_a_crossing_outside_the_opening_then_misses():
    traj = POOL_TRAJS[0]; g = traj.gates
    prev, cur = _cross(g[0], lateral=0.5)                      # 0.5 m off-centre: outside HALF_OPENING = 0.2
    idx, missed = gate_progress(prev, cur, traj.gate_times[0], 0.01, g, traj.gate_times, 0)
    assert (idx, missed) == (0, False), "an outside-the-opening crossing must not advance, nor miss yet"
    still = np.asarray(g[0].center) - 0.5 * np.asarray(g[0].normal)
    _, missed = gate_progress(still, still, traj.gate_times[0] + MISS_WINDOW + 0.01, 0.01, g, traj.gate_times, 0)
    assert missed


def test_gate_progress_window_edges_match_drivers_rule():
    traj = POOL_TRAJS[0]; g = traj.gates; gt = traj.gate_times
    far = np.asarray(g[0].center) - 0.5 * np.asarray(g[0].normal)
    assert gate_progress(far, far, gt[0] + MISS_WINDOW - 0.01, 0.01, g, gt, 0) == (0, False)
    assert gate_progress(far, far, gt[0] + MISS_WINDOW + 0.01, 0.01, g, gt, 0) == (0, True)
    prev, cur = _cross(g[0])                                    # a LATE crossing inside the window still counts
    assert gate_progress(prev, cur, gt[0] + 0.9, 0.01, g, gt, 0) == (1, False)
    idx, missed = gate_progress(prev, cur, gt[0] + 1.2, 0.01, g, gt, 0)   # ... but not outside it
    assert idx == 0 and missed


def test_a_finished_lap_never_misses():
    traj = POOL_TRAJS[0]
    far = np.array([9.0, 9.0, 1.0])
    assert gate_progress(far, far, 99.0, 0.01, traj.gates, traj.gate_times, len(traj.gates)) == (4, False)


def test_gate_progress_matches_drivers_source_constants():
    src = inspect.getsource(drv.Flyer.fly)
    assert "<= 1.0 and float(np.abs(yz).max()) < RaceGate.HALF_OPENING" in src
    assert "t > gt[gate_idx] + 1.0" in src
    assert MISS_WINDOW == 1.0


def _place(env, i, pos):
    import jax.numpy as jnp

    p = np.array(env.sim.data.states.pos); v = np.array(env.sim.data.states.vel)
    p[i, 0] = pos; v[i, 0] = 0.0
    env.sim.data = env.sim.data.replace(states=env.sim.data.states.replace(pos=jnp.asarray(p), vel=jnp.asarray(v)))


def test_env_terminates_on_a_missed_gate_with_the_gate_penalty():
    """Env 0 sits on its reference (so no divergence, no contact) at the moment gate 1's window closes with no
    crossing: it must end as a miss, at -GATE_PENALTY, and be counted as a miss, not contact or floor."""
    env = _make_env(num_envs=3)
    env.reset()
    for i in range(3):
        env._traj[i] = POOL_TRAJS[1]; env._dev_idx[i] = 1; env._gate_idx[i] = 0
    env.steps[:] = 0
    env.steps[0] = int(round((POOL_TRAJS[1].gate_times[0] + MISS_WINDOW) * 100))   # next step: t > gt + 1.0
    t_next = (env.steps[0] + 1) / 100
    _place(env, 0, POOL_TRAJS[1].pos(t_next))
    _, reward, crashed, _, _ = env.step(np.zeros((3, 4), dtype=np.float32))
    assert crashed[0] and reward[0] == -GATE_PENALTY, (crashed, reward)
    assert not crashed[1] and not crashed[2]
    assert env.stats["miss"] == 1 and env.stats["contact"] == 0 and env.stats["floor_div"] == 0
    assert env._gate_idx[0] == 0, "the episode reset must restart the target gate"


def test_no_miss_termination_before_the_window_closes():
    env = _make_env(num_envs=2)
    env.reset()
    for i in range(2):
        env._traj[i] = POOL_TRAJS[1]; env._dev_idx[i] = 1; env._gate_idx[i] = 0
    env.steps[:] = int(round((POOL_TRAJS[1].gate_times[0] + MISS_WINDOW) * 100)) - 5
    _place(env, 0, POOL_TRAJS[1].pos((env.steps[0] + 1) / 100))
    _, _, crashed, _, _ = env.step(np.zeros((2, 4), dtype=np.float32))
    assert not crashed[0]


def test_divergence_keeps_the_vendored_penalty_not_the_gate_one():
    env = _make_env(num_envs=2)
    env.reset()
    for i in range(2):
        env._traj[i] = POOL_TRAJS[1]; env._dev_idx[i] = 1; env._gate_idx[i] = 0
    env.steps[:] = 60
    _place(env, 0, POOL_TRAJS[1].pos(0.61) + np.array([3.0, 0.0, 0.0]))     # 3 m off the reference: err > 2.0
    _, reward, crashed, _, _ = env.step(np.zeros((2, 4), dtype=np.float32))
    assert crashed[0] and reward[0] == -FLOOR_PENALTY == -5.0
    assert env.stats["floor_div"] == 1 and env.stats["miss"] == 0


def test_gate_penalty_is_the_modest_one_the_user_asked_for():
    assert GATE_PENALTY == 3.5 and FLOOR_PENALTY == 5.0


def test_summarise_gate_stats():
    a = {"episodes": 10, "contact": 2, "miss": 3, "floor_div": 1, "trunc": 4, "gates_sum": 12, "full_laps": 2}
    b = {"episodes": 30, "contact": 6, "miss": 9, "floor_div": 3, "trunc": 12, "gates_sum": 42, "full_laps": 8}
    out = summarise_gate_stats(a, b)
    assert out["gate/episodes"] == 20 and out["gate/passed_per_ep"] == 1.5
    assert out["gate/frac_contact"] == 0.2 and out["gate/frac_miss"] == 0.3
    assert out["gate/frac_floor_div"] == 0.1 and out["gate/frac_trunc"] == 0.4 and out["gate/full_lap_frac"] == 0.3
    assert summarise_gate_stats(b, b) == {}


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
