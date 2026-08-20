# Lesson 1 — The tracking problem

⏱️ ~45 minutes, no training. **Open the files as you go** — every code block
below is real, and cited by path and line so you can read it in context.

> **The one big idea:** in your last tutorial the drone chased *where the target
> is now*. That works until the target moves fast, then you lag and cut corners.
> The fix is to also show the policy **where the target is going**. That single
> change is what separates a hover brain following a slow circle from a real
> trajectory tracker.

All paths below are relative to the project root. `tasks/racing/crazy_track/`
is the pinned, vendored crazy_track source (see its `VENDORED.md`) — the lessons
cite file and line numbers in it, and they match exactly.

---

## 1. Tracking is not planning

| | **Planning** | **Tracking** ← this course |
|---|---|---|
| Question | *Which path should I fly?* | *How do I fly the path I was given?* |
| Output | a trajectory | attitude + thrust commands |
| Fails by | choosing a slow or infeasible route | drifting off the route |

We build the **tracker only**. This has a consequence you should internalise
now: **your controller can only be as fast as the path it is handed.** In the
parent project we measured a case where the tracker was so good it *beat* its
own reference by cutting corners — the plan was the slow part. Knowing which
half of the problem you are optimising is the point of this lesson.

## 2. What the policy sees

Two constants set the whole design —
`tasks/racing/crazy_track/src/crazy_track/envs/datt_env.py:20`:

```python
WINDOW = 10  # future reference samples
WINDOW_DT = 0.06  # s between samples (0.6 s lookahead)
```

Ten future reference points, 0.06 s apart: **0.6 seconds of look-ahead.**

Now the observation itself —
`tasks/racing/crazy_track/src/crazy_track/envs/datt_env.py:229`:

```python
    def _base_obs(self) -> np.ndarray:
        """The deployable (noisy-view) observation frame, (N, OBS_DIM_V3|OBS_DIM)."""
        pos, vel, quat = self._meas  # measured (v4+: noisy/delayed) state
        t = self.steps / self.freq
        ref, win = self._refs(t)
        rel_win = (win - pos[:, None, :]).reshape(self.num_envs, -1)
        parts = [ref - pos, vel, quat, rel_win]
        if self.v3:
            parts.append(self.l1.sigma_f.astype(np.float32))
```

Read `parts` — that is the observation, in order:

| block | size | meaning |
|---|---|---|
| `ref - pos` | 3 | position error — the arrow you already know from hovering |
| `vel` | 3 | how fast the drone is moving |
| `quat` | 4 | orientation |
| `rel_win` | **30** | the 10 future reference points, **relative to the drone** |
| `l1.sigma_f` | 3 | on-line estimate of the force disturbing the drone |

**43 numbers.** Note `rel_win` is `win - pos`: the future path is expressed
*relative to the drone*, not in world coordinates. The policy is asked "where is
the path, from where I am now?", which is the same question at every point on
the track — so one learned answer generalises.

For the `--v5` variant you will train, the vector is **56**: a 13-number
privileged block is appended for the **critic only**
(`datt_env.py:24`, `PRIV_DIM = 3 + 3 + 4 + 3`). During training the critic sees
the *true* state while the actor sees the noisy one; only the actor is deployed.
Better-informed critic → lower-variance learning signal, and the deployed policy
is still honest. This is called an **asymmetric actor-critic**.

> ✅ Check it yourself: the smoke test printed `observation (2, 56)`.

## 3. What the policy does

`tasks/racing/crazy_track/src/crazy_track/envs/datt_env.py:292`:

```python
    def _denorm_action(self, a: np.ndarray) -> np.ndarray:
        if self.ctbr:
            ...
        cmd = np.zeros((self.num_envs, 1, 4), dtype=np.float32)
        cmd[:, 0, 0:2] = np.clip(a[:, 0:2], -1, 1) * RPY_MAX * 0.7
        cmd[:, 0, 2] = 0.0  # yaw fixed
        thrust = THRUST_MIN + (np.clip(a[:, 3], -1, 1) + 1) * 0.5 * (THRUST_MAX - THRUST_MIN)
        cmd[:, 0, 3] = thrust
        return cmd
```

The network emits four numbers in `[-1, 1]`; this maps them to physical units:

| output | becomes | limit |
|---|---|---|
| `a[0]`, `a[1]` | roll, pitch setpoint | `RPY_MAX * 0.7` = **±0.7 rad** |
| `a[2]` | yaw | **fixed to 0** — you never need to spin to follow a path |
| `a[3]` | collective thrust | `THRUST_MIN`…`THRUST_MAX` = **0.0855…0.80 N** |

Compare 0.80 N to the drone's weight, **0.4256 N** (your smoke test): maximum
thrust is **1.88 × weight**. That thrust-to-weight ratio is the hard ceiling on
everything in Lesson 4 — no reward function buys acceleration the motors cannot
produce.

> 🔑 **This action format is exactly what the race accepts** in `attitude` mode:
> `[roll, pitch, yaw, thrust]`. That is why your policy can race in Lesson 3
> without any translation layer.

## 4. What the policy is paid for

`tasks/racing/crazy_track/src/crazy_track/envs/datt_env.py:336`:

```python
        crashed = (pos[:, 2] < 0.05) | (err > 2.0)
        truncated = self.steps >= self.max_steps
        reward = np.exp(-2.0 * err) - 0.02 * np.linalg.norm(action[:, 0:2], axis=1)
```

Three statements:

1. **`exp(-2·err)`** — 1.0 when perfect, decaying smoothly. *Smooth matters*: a
   reward with a cliff gives the optimiser no direction to walk in.
2. **`-0.02·‖roll,pitch‖`** — a small tax on violent attitude commands, for
   smoothness. Small on purpose: make it large and the policy discovers that the
   cheapest way to avoid the tax is to barely fly.
3. **crash = `z < 0.05` or `err > 2.0`**, worth `-5.0` (line 386). Note the
   second one: drifting 2 m from the reference counts as a crash even if the
   drone is flying beautifully. It is a *tracking* task.

> 🔍 **A real wart, left in deliberately.** In the body-rate (`--ctbr`) variants,
> `action[:, 0:2]` is *thrust and roll-rate*, not roll and pitch — the penalty
> was written for the attitude layout and never re-indexed. It is small, and
> every published number in the parent project includes it, so it stays for
> comparability. **Finding things like this in inherited code is normal.**
> Deciding whether to fix it (and re-measure everything) is a judgement call.

## 5. ★ How the training trajectories are generated

This is the part most people skip, and it is the part that decides whether your
policy works. **The policy never trains on the race track.** It trains on a
*distribution* of random smooth paths.

### 5.1 Picking the difficulty

`tasks/racing/crazy_track/src/crazy_track/envs/datt_env.py:109`:

```python
    def _sample_traj(self, i: int) -> None:
        # Randomized difficulty per trajectory: covers the full Lissajous benchmark
        # envelope (fast reaches ~3 m/s and ~9 m/s^2; policies trained only on
        # gentle refs fail on it — RMSE 0.95 m observed with vel<=1, acc<=2).
```

Read that comment twice. It is a measured result: **a policy trained only on
gentle references failed on fast ones with 0.95 m of error.** Nearly a metre.
The training distribution, not the algorithm, was the bug.

The ranges that fixed it — `datt_env.py:161`:

```python
        if self.ctbr:
            vel_range = float(self.rng.uniform(1.0, 5.0))
            acc_range = float(self.rng.uniform(3.0, 15.0))
        else:
            vel_range = float(self.rng.uniform(0.5, 3.5))
            acc_range = float(self.rng.uniform(1.0, 10.0))
        self._traj[i] = ChainedPolyTrajectory.random(
            self.rng, duration=self.max_steps / self.freq + WINDOW * WINDOW_DT + 1.0,
            seg_duration=self.rng.uniform(1.0, 2.5), pos_range=1.0,
            vel_range=vel_range, acc_range=acc_range, start_pos=START,
        )
```

Every episode draws **a different difficulty**: speed up to 3.5 m/s and
acceleration up to 10 m/s² in attitude mode. Not one hard setting — a *range*,
so the policy meets easy and hard paths and cannot specialise to either.

> 📌 **Remember this block.** In Lesson 4 you will measure the speed and
> acceleration your *race* trajectory demands and compare them to these numbers.
> If the race is outside this box, the policy has never seen the regime you are
> asking it to fly, and that is your bug — a one-line fix.

### 5.2 Building one path

`tasks/racing/crazy_track/src/crazy_track/trajectories/chained_poly.py:41`:

```python
        n_seg = int(np.ceil(duration / seg_duration))
        knot_times = seg_duration * np.arange(n_seg + 1)
        start = np.zeros(3) if start_pos is None else np.asarray(start_pos, dtype=np.float64)

        # Knot states: (n_seg + 1, 3) each. First knot is at rest at start.
        pos = start + rng.uniform(-pos_range, pos_range, size=(n_seg + 1, 3))
        vel = rng.uniform(-vel_range, vel_range, size=(n_seg + 1, 3))
        acc = rng.uniform(-acc_range, acc_range, size=(n_seg + 1, 3))
        pos[0], vel[0], acc[0] = start, 0.0, 0.0

        coeffs = np.empty((n_seg, 3, 6))
        for i in range(n_seg):
            T = knot_times[i + 1] - knot_times[i]
            for ax in range(3):
                coeffs[i, ax] = _quintic(
                    pos[i, ax], vel[i, ax], acc[i, ax],
                    pos[i + 1, ax], vel[i + 1, ax], acc[i + 1, ax], T,
                )
```

The recipe, in words:

1. Chop the episode into segments of 1–2.5 s.
2. At each **knot** (segment boundary) roll a random position, velocity **and**
   acceleration.
3. Join consecutive knots with a **quintic polynomial per axis**.

Why a quintic (degree 5)? Because it has 6 coefficients, and you are imposing
6 constraints: position, velocity and acceleration at *both* ends. Exactly
determined — and that is the whole reason for the choice.

`chained_poly.py:88`:

```python
def _quintic(p0: float, v0: float, a0: float, p1: float, v1: float, a1: float, T: float) -> np.ndarray:
    """Ascending-power quintic matching (pos, vel, acc) at tau=0 and tau=T."""
    A = np.zeros((6, 6))
    b = np.array([p0, v0, a0, p1, v1, a1], dtype=np.float64)
```

Six equations, six unknowns, one `np.linalg.solve`. No optimisation, no
learning — just linear algebra.

**Why continuous acceleration matters physically.** For a quadrotor,
acceleration determines the thrust vector, which determines the attitude. If
acceleration jumped at a knot, the reference would demand an *instantaneous
change of orientation* — physically impossible, so the tracking error could
never go to zero and the policy would be chasing a target that does not exist.
Matching acceleration at both ends (that is the C² in "C²-continuous") is what
makes the reference *flyable*.

> 🔑 **The transferable idea:** a policy trained on one path memorises that
> path. Train on a *distribution* of paths and you learn the skill. In Lesson 3
> you fly a track the policy has never seen — and it should work. That property
> is **zero-shot transfer**, and it is why this approach is worth the trouble.

### 5.3 See it for yourself

```bash
python -c "
import numpy as np
from crazy_track.trajectories import ChainedPolyTrajectory
rng = np.random.default_rng(0)
traj = ChainedPolyTrajectory.random(rng, duration=6.0, seg_duration=1.5, vel_range=3.5, acc_range=10.0)
t = np.linspace(0, 6, 601)
p, v, a = traj.pos(t), traj.vel(t), traj.acc(t)
print('max speed        %.2f m/s' % np.linalg.norm(v, axis=1).max())
print('max acceleration %.2f m/s^2' % np.linalg.norm(a, axis=1).max())
print('thrust needed    %.2f m/s^2  (limit 0.95*TWR*g = 17.5)' %
      np.linalg.norm(a + np.array([0,0,9.81]), axis=1).max())"
```

Run it a few times with different seeds. Ask yourself: **does the hardest draw
stay inside the thrust limit?** If not, the environment is sometimes asking for
the impossible — which is worth knowing before you blame your policy.

---

## 🛠️ Before you move on

1. **Predict, then check.** What happens to tracking error if `WINDOW` were 1
   instead of 10? Write the prediction down; you can test it in Lesson 4.
2. **Do the arithmetic.** TWR is 1.88. Flying level, how much *horizontal*
   acceleration can the drone produce at most? (The thrust vector must hold up
   the weight first.) This bounds how sharply any controller can corner.
3. **Read `_refs`** (`datt_env.py:223`) and explain how `WINDOW_DT` turns into
   the 30 numbers of `rel_win`.
4. **Change one number.** Set `vel_range` to `uniform(0.1, 0.5)` in
   `datt_env.py:165`, and predict what Lesson 2's fast-tracking result will be.
   Put it back afterwards — or better, keep the edit on a branch and actually
   measure it in Lesson 4.

**Next:** [Lesson 2 — Train a DATT policy](02-train-datt.md)
