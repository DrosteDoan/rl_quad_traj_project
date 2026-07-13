"""Small wrappers that turn the raw Crazyflow env into something a learner can use.

A neural network does not understand a *dictionary* of observations like
``{"rel_pos": ..., "vel": ...}``. It wants one flat list of numbers. The
``FlattenObservation`` wrapper glues all the pieces of the observation into a
single vector, always in the same order, so the network sees a consistent input.

We also re-export Crazyflow's ``NormalizeActions`` wrapper for convenience. It
lets the agent output actions in the easy range ``[-1, 1]`` and rescales them to
the real thrust/angle commands the drone expects.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from gymnasium import spaces
from gymnasium.vector import VectorEnv, VectorObservationWrapper
from jax import Array

# Re-export so students import everything hover-related from one place.
from crazyflow.envs.norm_actions_wrapper import NormalizeActions  # noqa: F401

# The order we concatenate the observation pieces. Keep this fixed: the network
# learns "slot 0-2 is where I am relative to the target", etc.
OBS_ORDER = ("rel_pos", "vel", "quat", "ang_vel")


class FlattenObservation(VectorObservationWrapper):
    """Turn the dict observation into a single flat vector (per drone).

    Example: rel_pos(3) + vel(3) + quat(4) + ang_vel(3) -> a length-13 vector.
    """

    def __init__(self, env: VectorEnv, obs_order: tuple[str, ...] = OBS_ORDER):
        super().__init__(env)
        self.obs_order = obs_order
        # Work out the total length by adding up the size of each piece.
        size = sum(
            int(np.prod(env.single_observation_space[k].shape)) for k in obs_order
        )
        self.single_observation_space = spaces.Box(-np.inf, np.inf, shape=(size,))
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(env.num_envs, size)
        )

    def observations(self, observations: dict[str, Array]) -> Array:
        # Glue the requested pieces together along the last axis.
        return jnp.concatenate(
            [jnp.reshape(observations[k], (observations[k].shape[0], -1)) for k in self.obs_order],
            axis=-1,
        )
