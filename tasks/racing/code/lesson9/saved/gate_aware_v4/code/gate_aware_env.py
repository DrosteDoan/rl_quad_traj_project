"""The gate-aware viability screen (Lesson 9, 2026-09-23/24): trains on REAL TOGT-planned dev-track references
with a genuine contact consequence, instead of open-space `ChainedPolyTrajectory` curves with none. This is
deliberately the smaller question, not the full research question: not "does this close the whole gap
across the study," just "can the recipe pass the viability gate at all, given the budget left" -- one seed,
evaluated on the existing 6-track screen set, at one number: completions at lambda=0.

VERSION 2 (kept inside v3). Version 1 re-origined each dev track at the first moment its own climb cleared
z > 0.5 m (a `TimeShiftedTrajectory`), so every training episode started already airborne, 0.28-0.38 s before
gate 1. It trained cleanly (reward 8 -> 282, episode length 48 -> 516 of 700, still climbing at 8M steps) and
then scored 0/6 completions, 0/24 gates, every failure a gate-1 contact at t = 1.2-1.7 s on every track,
including `level2`. The eval harness always flies a real ground start (`GroundStartTrajectory`: ground hold,
climb-out, then gate 1); v1 never trained on one. M1 has no such gap -- it re-solves from an explicit model at
every step, so a ground launch is just another initial condition, not a situation it needs prior exposure to
(it flies this exact sequence on tracks 4/93/387 with < 9 cm deviation). Whether the v1 failure was the missing
launch or gate 1's truncated lead-time (v1 gave gate 1 a fraction of the lead-time gates 2-4 got) was not
separated; v2 removes both at once.

VERSION 4 (this file, 2026-09-25) -- a larger training pool. v3 learned its 3 dev tracks (pooled train full-lap
rate 0.535 over 142 episodes; 2/3 dev tracks and 10/12 gates in the harness) but scored 1/6 completions, 11/24 gates
on the held-out selection set: level with the open-space baseline, i.e. a transfer gap with a 3-track pool (n=6
makes that suggestive, not conclusive). v4 trains on the 3 dev tracks PLUS every track `gen_pool.py` accepted in
its fixed `train` seed windows (200000..205999), drawn by the same generator and filters as the study tracks so the
training distribution matches the study distribution by construction while every evaluated instance stays held
out. A separate `val` pool (300000..301999) is never trained on; RL design decisions are judged on it from here on,
not on study tracks (METHODOLOGY.md limitation 15). Mechanically nothing else changes: the pool is discovered from
`tracks/train/`, `EPISODE_TIME` is computed from the longest pool track instead of hardcoded, and the contact
check groups envs by track (at most `num_envs` groups, so at most 16 `pose_contacts` calls per step).

VERSION 3 (kept inside v4, 2026-09-24). v2 also failed (0/6, 2/24 gates) and, flown on its OWN training tracks,
completed only 1 of 3: 100082 and 100092 missed gate 1 with no contact, swerving 0.4-0.8 m sideways from ~0.2 s
before the gate and recovering after it. Training ended an episode on contact, gross divergence or the floor,
but not on a missed gate. Contact therefore forfeits the rest of the episode (~200 reward, a rough estimate) on
top of its penalty, while a swerve costs ~0.3 in total -- so a policy that cannot reliably hold a 3.5-5 cm
margin is pushed to swerve around the gate instead of through it. v3 makes a miss end the episode too, using
`driver.py`'s own rule (`gate_progress` below), so training and eval share one definition of failure.
Not verified: that "won't thread" rather than "can't hold the line" is the cause -- the same trace fits both,
and a miss-termination cannot tell them apart; it only removes the cheap way out. Penalties: gate contact and
a missed gate are -GATE_PENALTY (3.5); floor/gross divergence keep the vendored -5.0. The terminal constant is
the smaller lever -- the forfeited remaining return dominates it -- so -3.5 vs -5.0 is chosen for scale
(gentler on the value function) and is not expected to change behaviour by itself. Finishing all four gates does
NOT end the episode (ending it would forfeit the remaining reward for succeeding); it runs to truncation.

What v2 changed (kept in v3), and why each change reuses something already validated instead of inventing:

  * References are `driver.build_traj(track)` for each of the 3 dev tracks -- the SAME `GroundStartTrajectory`
    (0.2 s hold, quintic climb from `RACE_START` z=0.01, the plan, gate 1 at its normal lead-time) the
    evaluation harness constructs. Nothing re-origined.
  * The training floor-crash threshold moves from the vendored `pos[:, 2] < 0.05` to `pos[:, 2] < FLOOR_Z`
    with FLOOR_Z = -0.3, exactly `driver.py`'s own divergence check (`Flyer.fly`: `pos[2] < -0.3`). 0.05 was
    only ever safe because every other training env starts well clear of the ground; a ground-start reference
    at z=0.01 would trip it on step one. -0.3 is below the physical floor, so it only catches a fall through
    it, as in eval, and needs no time-windowed exemption. `err > 2.0` still catches "gave up and sat on the
    floor" once the reference has moved away.
  * `episode_time` = 8.0 s: the full ground-start durations are 5.8 / 7.2 / 6.4 s (longest 7.196 s), above the
    vendored 6.0 s default, which would have truncated an episode before gate 4 on the longest track.
    `SampledRaceTrajectory.pos(t)` holds at its final point past its own duration (confirmed in source).

The simulator was never the obstacle -- every controller already flies this launch in the harness. The obstacle
was `datt_env.py`'s crash rule, written when nothing trained near the ground.

`contact.py`'s `pose_contacts(pos, quat, gates)` is batch-shaped (pos (M,3), quat (M,4)), so the contact check
drops into `step()` across all parallel envs, grouped by which of the 3 dev tracks each env is flying.

v4 has NOT been trained -- see `train_gate_aware.py`'s module docstring.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CODE = HERE.parent
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(HERE))
import contact as ct  # noqa: E402
import driver as drv  # noqa: E402
from robust_env import RobustTrackingEnv  # noqa: E402

from crazy_track.trajectories.freestyle import RaceGate  # noqa: E402

DEV_SEEDS = (100023, 100082, 100092)
FLOOR_Z = -0.3       # m; driver.py's divergence check (`pos[2] < -0.3`), NOT the vendored 0.05
GATE_PENALTY = 3.5   # contact with a frame, or a missed gate (was the vendored 5.0 for every crash in v1/v2)
FLOOR_PENALTY = 5.0  # floor / gross divergence: the vendored value, unchanged
MISS_WINDOW = 1.0    # s; driver.py: a crossing counts within +-1.0 s of the gate's clock, a miss is clock + 1.0 s


def train_seeds() -> list[int]:
    """Every track `gen_pool.py` accepted in the `train` seed windows, in seed order (none until it has run)."""
    return sorted(int(p.stem.split("_")[1]) for p in (HERE / "tracks" / "train").glob("track_*.json"))


POOL_SPEC = [("dev", s) for s in DEV_SEEDS] + [("train", s) for s in train_seeds()]
POOL_TRAJS = [drv.build_traj(drv.load_track(role, s)) for role, s in POOL_SPEC]
# s; above the longest full ground-start duration in the pool with margin, in steps of 0.5 s. The dev tracks alone
# gave 8.0 (longest 7.196 s); every episode runs this long, the reference holding at its final point once a lap ends.
EPISODE_TIME = math.ceil((max(t.duration for t in POOL_TRAJS) + 0.3) * 2) / 2


def gate_progress(prev: np.ndarray, cur: np.ndarray, t: float, dt: float, gates, gate_times, idx: int):
    """`driver.py`'s target-gate rule (`Flyer.fly`) for ONE env and ONE step: returns (new_idx, missed).
    The target gate is crossed when the step changes the sign of the gate-frame x, inside the opening
    (`RaceGate.HALF_OPENING`) and within MISS_WINDOW of the gate's clock; it is missed when the clock plus
    MISS_WINDOW passes without that. A lap that has passed every gate (idx == len(gates)) never misses."""
    if idx >= len(gates):
        return idx, False
    g = gates[idx]
    x0, x1 = g.to_gate_frame(prev), g.to_gate_frame(cur)
    if np.sign(x0[0]) != np.sign(x1[0]):
        w = x0[0] / (x0[0] - x1[0])
        tc = (t - dt) + w * dt
        yz = x0[1:] + w * (x1[1:] - x0[1:])
        if abs(tc - gate_times[idx]) <= MISS_WINDOW and float(np.abs(yz).max()) < RaceGate.HALF_OPENING:
            idx += 1
            if idx >= len(gates):
                return idx, False
    if t > gate_times[idx] + MISS_WINDOW:
        return idx, True
    return idx, False


def gate_contacts(pos: np.ndarray, quat: np.ndarray, track_idx: np.ndarray) -> np.ndarray:
    """Per env: is the drone box touching a frame of ITS pool track's gates (`contact.pose_contacts`, batched
    per track). `pose_contacts` already ignores poses farther than `ct.NEAR` from a gate, so envs with no gate
    within NEAR are dropped BEFORE the call: it builds a scipy rotation for every pose it is given, which cost
    ~2 ms per step across up to 16 track groups for a result that is always 'no contact' far from the gates."""
    hit = np.zeros(len(pos), dtype=bool)
    for idx in np.unique(track_idx):
        group = np.flatnonzero(track_idx == idx)
        gates = POOL_TRAJS[int(idx)].gates
        d = np.min([np.linalg.norm(pos[group] - np.asarray(g.center), axis=1) for g in gates], axis=0)
        near = group[d < ct.NEAR]
        if len(near):
            hit[near] = ct.pose_contacts(pos[near], quat[near], gates)[0]
    return hit


def summarise_gate_stats(prev: dict, cur: dict) -> dict:
    """Per-rollout view of `GateAwareTrackingEnv.stats` (cumulative counters): how episodes ended, and how many
    gates they had passed. `full_lap_frac` is the training-time completion rate (all gates passed, then ran to
    truncation)."""
    d = {k: cur[k] - prev[k] for k in cur}
    n = d["episodes"]
    if n <= 0:
        return {}
    return {"gate/episodes": n, "gate/passed_per_ep": d["gates_sum"] / n,
            "gate/frac_contact": d["contact"] / n, "gate/frac_miss": d["miss"] / n,
            "gate/frac_floor_div": d["floor_div"] / n, "gate/frac_trunc": d["trunc"] / n,
            "gate/full_lap_frac": d["full_laps"] / n}


class GateAwareTrackingEnv(RobustTrackingEnv):
    """`RobustTrackingEnv` (0.7 rad authority, study-matched disturbance training, 0.8 s window, freq=100 --
    ALL unchanged) with three things changed: references are the real pool tracks (3 dev + `train`) flown from a real ground
    start, `step()` adds a genuine gate-frame contact check to the crash condition (`contact.py`'s exact box
    test, the one the eval harness uses), and the floor-crash threshold matches eval's (-0.3, not 0.05)."""

    def __init__(self, *args, episode_time: float = EPISODE_TIME, **kwargs):
        super().__init__(*args, episode_time=episode_time, **kwargs)
        self._dev_idx = np.zeros(self.num_envs, dtype=int)
        self._gate_idx = np.zeros(self.num_envs, dtype=int)     # next target gate per env (driver.py's gate_idx)
        self.stats = {"episodes": 0, "contact": 0, "miss": 0, "floor_div": 0, "trunc": 0,
                      "gates_sum": 0, "full_laps": 0}

    def _sample_traj(self, i: int) -> None:
        super()._sample_traj(i)                    # flip/maneuver bookkeeping (all off) -- overwrite the traj
        idx = int(self.rng.integers(0, len(POOL_TRAJS)))
        self._traj[i] = POOL_TRAJS[idx]
        self._dev_idx[i] = idx
        self._gate_idx[i] = 0

    def step(self, action: np.ndarray):
        assert not self.ballistic_flips and not self.acro2 and not getattr(self, "acro4", False), \
            "GateAwareTrackingEnv only handles the plain v3/v5 racing path (no maneuver reward branches)"
        pos_prev = np.array(self._state_arrays()[0])   # copy: the crossing test needs the pre-step position
        cmd = self._denorm_action(np.asarray(action))
        if self.v3:
            import jax.numpy as jnp

            force = jnp.asarray(self.perturb_force[:, None, :], dtype=jnp.float32)
            self.sim.data = self.sim.data.replace(states=self.sim.data.states.replace(force=force))
        self._apply_cmd(cmd)
        self.sim.step(self.n_substeps)
        self.steps += 1
        self._meas = self._measured_arrays()
        if self.v3:
            _, vel_m, quat_m = self._meas
            thrust_cmd = cmd[:, 0, 0] if self.ctbr else cmd[:, 0, 3]
            self.l1.update(vel_m, quat_m, thrust_cmd)

        pos, vel, quat = self._state_arrays()
        t = self.steps / self.freq
        ref, _ = self._refs(t)
        err = np.linalg.norm(ref - pos, axis=1)

        gate_hit = gate_contacts(pos, quat, self._dev_idx)

        missed = np.zeros(self.num_envs, dtype=bool)
        dt = 1.0 / self.freq
        for i in range(self.num_envs):
            traj = self._traj[i]
            self._gate_idx[i], missed[i] = gate_progress(pos_prev[i], pos[i], float(t[i]), dt, traj.gates,
                                                         traj.gate_times, int(self._gate_idx[i]))

        floor_div = (pos[:, 2] < FLOOR_Z) | (err > 2.0)
        crashed = floor_div | gate_hit | missed
        truncated = self.steps >= self.max_steps
        reward = np.exp(-2.0 * err) - 0.02 * np.linalg.norm(action[:, 0:2], axis=1)
        penalty = np.where(floor_div, FLOOR_PENALTY, GATE_PENALTY)
        reward = np.where(crashed, -penalty, reward).astype(np.float32)

        done_mask = crashed | truncated
        info = {}
        if done_mask.any():
            n_gates = np.array([len(self._traj[i].gates) for i in range(self.num_envs)])
            st = self.stats
            st["episodes"] += int(done_mask.sum())
            st["floor_div"] += int((floor_div & done_mask).sum())
            st["contact"] += int((gate_hit & ~floor_div & done_mask).sum())
            st["miss"] += int((missed & ~gate_hit & ~floor_div & done_mask).sum())
            st["trunc"] += int((truncated & ~crashed).sum())
            st["gates_sum"] += int(self._gate_idx[done_mask].sum())
            st["full_laps"] += int((truncated & ~crashed & (self._gate_idx >= n_gates)).sum())
            info["terminal_obs"] = self._obs()
            import jax.numpy as jnp

            self.sim.reset(mask=jnp.asarray(done_mask))
            for i in np.flatnonzero(done_mask):
                self._sample_traj(i)
            self.steps[done_mask] = 0
            self._set_states(done_mask)
            if self.v3:
                self._sample_perturb(done_mask)
                self.l1.v_hat[done_mask] = 0.0
                self.l1.sigma_hat[done_mask] = 0.0
                self.l1.sigma_f[done_mask] = 0.0
            if self.noisy_sensor:
                self.sensor.reset_rows(done_mask)
            self._meas = self._measured_arrays()

        if self.v6:
            self._push_stack(self._base_obs(), refill_mask=done_mask if done_mask.any() else None)
        return self._obs(), reward, crashed, truncated, info
