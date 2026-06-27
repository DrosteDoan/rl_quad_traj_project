"""The Hover environment: teach a Crazyflie 2.1 Brushless to stay in one place.

This is the "build your own environment" lesson. An RL environment answers three
questions for the learning agent (the "brain" we are training):

1. **What does the drone see?**  -> the *observation*
2. **What can the drone do?**    -> the *action*
3. **How good was that?**        -> the *reward*

We build on top of Crazyflow's ``DroneEnv``, which already runs the drone physics
on the GPU for hundreds of drones at the same time (that is why training is fast).
We only have to add the three things above for the *hover* task.

Read this file together with ``lessons/02_build_the_env.md``.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from gymnasium import spaces
from gymnasium.vector.utils import batch_space
from jax import Array

from crazyflow.dynamics import Dynamics
from crazyflow.envs.drone_env import DroneEnv
from crazyflow.sim.data import SimData
from crazyflow.utils import leaf_replace


class HoverEnv(DroneEnv):
    """Make the drone fly to a target point and stay there (hover).

    The target is a single fixed point in the air (default: 1 meter above the
    origin). At the start of every episode the drone is dropped somewhere near
    the target with a random position and a small random velocity, and it has to
    fly back to the target and hold still.

    The reward is built from a few easy-to-understand pieces. Each piece has a
    *coefficient* (a knob) so students can turn behaviours up or down and watch
    what happens. This is the "understand the criteria" lesson.
    """

    def __init__(
        self,
        target_pos: tuple[float, float, float] = (0.0, 0.0, 1.0),
        # --- reward knobs (students change these!) ---
        distance_coef: float = 2.0,
        velocity_coef: float = 0.0,
        spin_coef: float = 0.0,
        tilt_coef: float = 0.0,
        yaw_coef: float = 0.0,
        crash_penalty: float = -1.0,
        # --- how hard the task is ---
        start_radius: float = 0.5,
        start_speed: float = 0.5,
        bounds: float = 2.0,
        # --- simulation settings (mostly leave as-is) ---
        num_envs: int = 1,
        max_episode_time: float = 5.0,
        dynamics: Dynamics = Dynamics.so_rpy,
        drone: str = "cf21B_500",  # Crazyflie 2.1 Brushless
        freq: int = 50,
        device: str = "cpu",
    ):
        """Create a hover environment.

        Args:
            target_pos: The point in space (x, y, z) the drone should hover at.
            distance_coef: How strongly to reward being close to the target.
            velocity_coef: How strongly to punish moving fast (0 = ignore).
            spin_coef: How strongly to punish spinning/rotating (0 = ignore).
            tilt_coef: How strongly to punish tilting away from upright (0 = ignore).
            crash_penalty: Reward given on the step the drone hits the ground.
            start_radius: How far from the target the drone can start (meters).
            start_speed: How fast the drone can be moving at the start (m/s).
            bounds: If the drone flies further than this from the target, end the
                episode (it has flown away and is not coming back).
            num_envs: How many drones to simulate in parallel (more = faster training).
            max_episode_time: How many seconds one practice run (episode) lasts.
            dynamics: Which physics (dynamics) model to use. ``so_rpy`` is fit to real
                flight data; ``first_principles`` is full rigid-body physics.
            drone: Which drone. ``cf21B_500`` is the Crazyflie 2.1 Brushless.
            freq: How many decisions per second the agent makes (control rate, Hz).
            device: ``"cpu"`` or ``"gpu"``.
        """
        # Save the target point as a JAX array on the right device. We build it
        # later in __init__ once we know the device, so just remember the tuple now.
        self._target_pos_np = np.asarray(target_pos, dtype=np.float32)

        # Reward knobs.
        self.distance_coef = float(distance_coef)
        self.velocity_coef = float(velocity_coef)
        self.spin_coef = float(spin_coef)
        self.tilt_coef = float(tilt_coef)
        self.yaw_coef = float(yaw_coef)
        self.crash_penalty = float(crash_penalty)

        # Task difficulty.
        self.start_radius = float(start_radius)
        self.start_speed = float(start_speed)
        self.bounds = float(bounds)

        # Build the reset-randomization function (where the drone starts each episode).
        target = jnp.asarray(target_pos, dtype=jnp.float32)
        reset_randomization = partial(
            self._reset_randomization,
            target=target,
            radius=self.start_radius,
            speed=self.start_speed,
        )

        super().__init__(
            num_envs=num_envs,
            max_episode_time=max_episode_time,
            dynamics=dynamics,
            drone=drone,
            freq=freq,
            device=device,
            reset_randomization=reset_randomization,
        )

        # The target as an on-device array, broadcast to every parallel world.
        self.target_pos = jnp.asarray(target_pos, dtype=jnp.float32, device=self.device)

        # ----- Tell the agent what it will see (the observation) -----
        # We give it: the vector from the drone to the target (rel_pos), its
        # velocity, its orientation (quat) and its rotation speed (ang_vel).
        spec = {
            "rel_pos": spaces.Box(-np.inf, np.inf, shape=(3,)),
            "vel": spaces.Box(-np.inf, np.inf, shape=(3,)),
            "quat": spaces.Box(-np.inf, np.inf, shape=(4,)),
            "ang_vel": spaces.Box(-np.inf, np.inf, shape=(3,)),
        }
        self.single_observation_space = spaces.Dict(spec)
        self.observation_space = batch_space(self.single_observation_space, self.sim.n_worlds)

    # --------------------------------------------------------------------- render
    def render(self):
        """Draw the scene, plus a red ball showing where the drone should hover."""
        if self.sim.viewer is not None:
            import mujoco

            self.sim.viewer.viewer.add_marker(
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=np.array([0.03, 0.03, 0.03]),
                pos=np.asarray(self.target_pos),
                rgba=np.array([1.0, 0.0, 0.0, 0.6]),
            )
        self.sim.render()

    # ----------------------------------------------------------------- observation
    def obs(self) -> dict[str, Array]:
        """What the drone sees on every step."""
        pos = self.sim.data.states.pos[:, 0, :]
        return {
            "rel_pos": self.target_pos - pos,  # points from drone toward target
            "vel": self.sim.data.states.vel[:, 0, :],
            "quat": self.sim.data.states.quat[:, 0, :],
            "ang_vel": self.sim.data.states.ang_vel[:, 0, :],
        }

    # ---------------------------------------------------------------------- reward
    def reward(self) -> Array:
        """How good the current situation is (bigger = better)."""
        return self._reward(
            self.terminated(),
            self.sim.data.states.pos[:, 0, :],
            self.sim.data.states.vel[:, 0, :],
            self.sim.data.states.quat[:, 0, :],
            self.sim.data.states.ang_vel[:, 0, :],
            self.target_pos,
            self.distance_coef,
            self.velocity_coef,
            self.spin_coef,
            self.tilt_coef,
            self.yaw_coef,
            self.crash_penalty,
        )

    @staticmethod
    @partial(jax.jit, static_argnums=(6, 7, 8, 9, 10, 11))
    def _reward(
        terminated: Array,
        pos: Array,
        vel: Array,
        quat: Array,
        ang_vel: Array,
        target: Array,
        distance_coef: float,
        velocity_coef: float,
        spin_coef: float,
        tilt_coef: float,
        yaw_coef: float,
        crash_penalty: float,
    ) -> Array:
        # 1) Main reward: close to the target is good. exp(...) is 1.0 when the
        #    drone is exactly on the target and shrinks toward 0 as it gets far.
        distance = jnp.linalg.norm(pos - target, axis=-1)
        reward = jnp.exp(-distance_coef * distance)

        # 2) Optional penalties (only matter if their coefficient is > 0).
        #    Move slowly: punish speed.
        reward = reward - velocity_coef * jnp.linalg.norm(vel, axis=-1)
        #    Don't spin: punish angular velocity.
        reward = reward - spin_coef * jnp.linalg.norm(ang_vel, axis=-1)
        #    Stay upright: quat is (x, y, z, w); for a level drone x,y are ~0.
        #    tilt grows from 0 (upright) toward 1 (on its side).
        tilt = 1.0 - (1.0 - 2.0 * (quat[:, 0] ** 2 + quat[:, 1] ** 2))
        reward = reward - tilt_coef * tilt
        #    Keep facing forward (yaw ~ 0): quat z = sin(yaw/2). Penalizing it keeps
        #    the drone from spinning to a random heading, which makes the learned
        #    policy deployable on hardware/CrazySim where we hold yaw at 0.
        reward = reward - yaw_coef * (quat[:, 2] ** 2)

        # 3) Big penalty if the drone crashed this step.
        reward = jnp.where(terminated, crash_penalty, reward)
        return reward

    # ------------------------------------------------------------------- terminated
    def terminated(self) -> Array:
        """The episode ends early (failure) if the drone crashes or flies away."""
        return self._terminated(
            self.sim.data.states.pos[:, 0, :], self.target_pos, self.bounds
        )

    @staticmethod
    @partial(jax.jit, static_argnums=(2,))
    def _terminated(pos: Array, target: Array, bounds: float) -> Array:
        crashed = pos[:, 2] < 0.0  # hit the ground
        flew_away = jnp.linalg.norm(pos - target, axis=-1) > bounds
        return crashed | flew_away

    # ------------------------------------------------------- where the drone starts
    @staticmethod
    def _reset_randomization(
        data: SimData, _: SimData, mask: Array, target: Array, radius: float, speed: float
    ) -> SimData:
        """Drop the drone at a random spot near the target with a small push.

        This runs inside the simulation reset (it is JAX-compiled). Crazyflow passes
        ``(data, default_data, mask)``; we use ``data`` and ``mask`` and return an
        updated ``data``.
        """
        shape = (data.core.n_worlds, data.core.n_drones, 3)
        key, pos_key, vel_key = jax.random.split(data.core.rng_key, 3)
        data = data.replace(core=data.core.replace(rng_key=key))
        # Start position: target +/- radius in each direction.
        pos = target + jax.random.uniform(
            pos_key, shape=shape, minval=-radius, maxval=radius
        )
        # Start velocity: a small random push.
        vel = jax.random.uniform(vel_key, shape=shape, minval=-speed, maxval=speed)
        # Spin the rotors up to ~hover speed at reset. This matters for the
        # first_principles physics (which models rotor dynamics); so_rpy ignores it.
        rotor_vel = 10000.0 * jnp.ones_like(data.states.rotor_vel)
        data = data.replace(
            states=leaf_replace(data.states, mask, pos=pos, vel=vel, rotor_vel=rotor_vel)
        )
        return data
