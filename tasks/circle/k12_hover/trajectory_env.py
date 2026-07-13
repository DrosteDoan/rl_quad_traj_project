"""The Trajectory environment: teach the drone to FOLLOW A MOVING TARGET.

This is the next step after hovering. The big idea in one sentence:

    A hover is a target that stands still.
    A trajectory is the SAME target, but it slides along a path over time.

So we reuse almost everything from ``HoverEnv`` (the same observation pieces, the
same reward knobs, the same "where does it start" code) and change exactly one
thing: instead of a single fixed ``target_pos``, the target walks around a
**circle** as the seconds tick by. The drone has to chase it.

Read this file next to ``lessons/08_follow_a_circle.md`` and
``lessons/09_train_a_tracker.md``.

Two ways to use this env
------------------------
1. **Reuse your hover brain (no new training).** Set ``n_samples=0``. Then the
   observation is the *exact same 13 numbers* the hover policy already knows
   (``rel_pos`` now points at the *moving* target). Your trained ``hover_PPO.zip``
   will chase the circle with no retraining at all. (Lesson 8.)

2. **Train a dedicated tracker.** Set ``n_samples>0``. Now the observation also
   contains a **look-ahead**: where the path will be a few moments from now. This
   lets the policy anticipate the turns instead of always reacting late, so it can
   follow a *fast* circle that the hover brain cuts the corners on. (Lesson 9.)
"""

from __future__ import annotations

from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
from gymnasium import spaces
from gymnasium.vector.utils import batch_space
from jax import Array

from crazyflow.dynamics import Dynamics

from k12_hover.hover_env import HoverEnv


class TrajectoryEnv(HoverEnv):
    """Make the drone follow a moving target that travels around a circle.

    Everything the hover task taught the drone still applies: fly toward the
    target, hold steady, don't crash. The only new thing is that the target keeps
    moving, so "fly toward the target" now means "keep up with it".
    """

    def __init__(
        self,
        # --- the shape of the path the target travels ---
        shape: Literal["circle"] = "circle",
        radius: float = 0.5,          # how big the circle is (meters)
        period: float = 6.0,          # how many seconds one full loop takes
        height: float = 1.0,          # how high off the ground the circle sits
        center: tuple[float, float] = (0.0, 0.0),  # where the circle is centered (x, y)
        # --- the "look-ahead" (0 = off, i.e. hover-compatible observation) ---
        n_samples: int = 0,           # how many future path points the drone sees
        samples_dt: float = 0.1,      # how far apart (in seconds) those points are
        # --- reward knobs (identical to HoverEnv) ---
        distance_coef: float = 2.0,
        velocity_coef: float = 0.0,
        spin_coef: float = 0.0,
        tilt_coef: float = 0.0,
        yaw_coef: float = 0.0,
        crash_penalty: float = -1.0,
        # --- how hard the task is / where it starts ---
        start_radius: float = 0.3,
        start_speed: float = 0.3,
        bounds: float = 2.0,
        # --- simulation settings (mostly leave as-is) ---
        num_envs: int = 1,
        max_episode_time: float = 8.0,
        dynamics: Dynamics = Dynamics.so_rpy,
        drone: str = "cf21B_500",
        freq: int = 50,
        device: str = "cpu",
    ):
        """Create a trajectory-following environment.

        Args:
            shape: The kind of path. For now only ``"circle"`` (easy to extend).
            radius: Radius of the circle in meters.
            period: Seconds for the target to travel once around the circle.
                Smaller = faster target = harder to follow.
            height: The (constant) height of the circle above the ground.
            center: The (x, y) center of the circle.
            n_samples: How many upcoming path points to add to the observation
                (the "look-ahead"). ``0`` keeps the observation identical to the
                hover task, so a trained hover policy can be reused as-is.
            samples_dt: Seconds between look-ahead points.
            distance_coef..crash_penalty: The reward knobs, same as HoverEnv.
            start_radius: How far from the *start* of the path the drone may begin.
            start_speed: Initial random speed of the drone.
            bounds: End the episode if the drone gets this far from the *current*
                target (it has fallen too far behind / flown away).
            num_envs, max_episode_time, dynamics, drone, freq, device:
                Simulation settings, same meaning as HoverEnv.
        """
        # Remember the path settings BEFORE calling super().__init__, because the
        # base class kicks off the simulation and we want them available in obs().
        self.shape = shape
        self.radius = float(radius)
        self.period = float(period)
        self.height = float(height)
        self.center = (float(center[0]), float(center[1]))
        self.n_samples = int(n_samples)
        self.samples_dt = float(samples_dt)

        # The circle starts at angle 0, i.e. at (center_x + radius, center_y, height).
        # We hand this to HoverEnv as its "target_pos" so that the drone is dropped
        # near the START of the path each episode (HoverEnv randomizes around it).
        start_point = (self.center[0] + self.radius, self.center[1], self.height)

        # Let HoverEnv build the whole environment (physics, action space, reward
        # knobs, reset randomization, the 13-number hover observation, ...).
        super().__init__(
            target_pos=start_point,
            distance_coef=distance_coef,
            velocity_coef=velocity_coef,
            spin_coef=spin_coef,
            tilt_coef=tilt_coef,
            yaw_coef=yaw_coef,
            crash_penalty=crash_penalty,
            start_radius=start_radius,
            start_speed=start_speed,
            bounds=bounds,
            num_envs=num_envs,
            max_episode_time=max_episode_time,
            dynamics=dynamics,
            drone=drone,
            freq=freq,
            device=device,
        )

        # ----- Build the path (one point per control step, for a full loop) -----
        # We precompute the whole circle once as a big table of (x, y, z) points and
        # then just look up "where should the target be right now?" every step. The
        # target loops forever by wrapping around the table (modulo its length).
        self.loop_len = max(1, int(round(self.period * self.freq)))
        self.trajectory = jnp.asarray(
            self._build_trajectory(self.loop_len), dtype=jnp.float32, device=self.device
        )

        # Look-ahead: the table indices "samples_dt, 2*samples_dt, ... seconds from
        # now". e.g. at 50 Hz with samples_dt=0.1 these are 0, 5, 10, 15, ... steps.
        self.sample_offsets = jnp.asarray(
            np.arange(self.n_samples) * self.freq * self.samples_dt,
            dtype=jnp.int32,
            device=self.device,
        )

        # If look-ahead is on, the observation gains a "local_samples" block. When
        # n_samples==0 we leave the observation exactly as HoverEnv defined it, so a
        # hover policy plugs straight in.
        if self.n_samples > 0:
            spec = {k: v for k, v in self.single_observation_space.items()}
            spec["local_samples"] = spaces.Box(
                -np.inf, np.inf, shape=(3 * self.n_samples,)
            )
            self.single_observation_space = spaces.Dict(spec)
            self.observation_space = batch_space(
                self.single_observation_space, self.sim.n_worlds
            )

    # ------------------------------------------------------ build the path table
    def _build_trajectory(self, n: int) -> np.ndarray:
        """Return an ``(n, 3)`` array of points the target visits, one per step."""
        # angle goes 0 -> 2*pi across the loop. endpoint=False so the last point
        # joins smoothly back to the first (no double-visited start point).
        theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
        if self.shape == "circle":
            x = self.center[0] + self.radius * np.cos(theta)
            y = self.center[1] + self.radius * np.sin(theta)
            z = np.full_like(theta, self.height)
        else:
            raise ValueError(f"Unknown trajectory shape: {self.shape!r}")
        return np.stack([x, y, z], axis=-1)  # (n, 3)

    # ----------------------------------------- where along the path are we now?
    @property
    def steps(self) -> Array:
        """Current control-step index, per world (shape ``(n_worlds,)``).

        ``sim.data.core.steps`` counts the fast physics sub-steps; dividing by
        ``n_substeps`` converts that to *our* control steps. It resets to 0 each
        episode, so the target restarts at the beginning of the path every run.
        """
        step = self.sim.data.core.steps // self.n_substeps - 1
        return jnp.clip(step, 0).reshape(-1)  # (n_worlds,)

    def current_target(self) -> Array:
        """Where the moving target is *right now*, per world (shape ``(n_worlds, 3)``)."""
        idx = self.steps % self.loop_len
        return self.trajectory[idx]

    # ----------------------------------------------------------------- observation
    def obs(self) -> dict[str, Array]:
        """What the drone sees. Same as hover, but relative to the MOVING target."""
        pos = self.sim.data.states.pos[:, 0, :]
        target = self.current_target()
        observation = {
            "rel_pos": target - pos,  # arrow toward where the target is NOW
            "vel": self.sim.data.states.vel[:, 0, :],
            "quat": self.sim.data.states.quat[:, 0, :],
            "ang_vel": self.sim.data.states.ang_vel[:, 0, :],
        }
        # Optional look-ahead: relative position of the next n_samples path points.
        if self.n_samples > 0:
            # For each world, indices of the upcoming points (wrapping around).
            idx = (self.steps[:, None] + self.sample_offsets[None, :]) % self.loop_len
            future = self.trajectory[idx]                 # (n_worlds, n_samples, 3)
            dpos = future - pos[:, None, :]               # relative to the drone
            observation["local_samples"] = dpos.reshape(-1, 3 * self.n_samples)
        return observation

    # ---------------------------------------------------------------------- reward
    def reward(self) -> Array:
        """Reuse HoverEnv's reward, but graded against the MOVING target."""
        return self._reward(
            self.terminated(),
            self.sim.data.states.pos[:, 0, :],
            self.sim.data.states.vel[:, 0, :],
            self.sim.data.states.quat[:, 0, :],
            self.sim.data.states.ang_vel[:, 0, :],
            self.current_target(),           # <-- the only change vs HoverEnv
            self.distance_coef,
            self.velocity_coef,
            self.spin_coef,
            self.tilt_coef,
            self.yaw_coef,
            self.crash_penalty,
        )

    # ------------------------------------------------------------------- terminated
    def terminated(self) -> Array:
        """End early if the drone crashes or falls too far behind the target."""
        return self._terminated(
            self.sim.data.states.pos[:, 0, :], self.current_target(), self.bounds
        )

    # --------------------------------------------------------------------- render
    def render(self):
        """Draw the scene: the whole circle (faint dots) + the moving target (red)."""
        if self.sim.viewer is not None:
            import mujoco

            # Draw the full path as small grey dots so students see the circle.
            path = np.asarray(self.trajectory)
            for p in path[::2]:  # every other point is plenty
                self.sim.viewer.viewer.add_marker(
                    type=mujoco.mjtGeom.mjGEOM_SPHERE,
                    size=np.array([0.01, 0.01, 0.01]),
                    pos=p,
                    rgba=np.array([0.6, 0.6, 0.6, 0.5]),
                )
            # Draw the current target as a bigger red ball (the thing to chase).
            self.sim.viewer.viewer.add_marker(
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=np.array([0.03, 0.03, 0.03]),
                pos=np.asarray(self.current_target()[0]),
                rgba=np.array([1.0, 0.0, 0.0, 0.8]),
            )
        self.sim.render()
