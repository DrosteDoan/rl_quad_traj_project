"""Bridge: let Stable-Baselines3 (SB3) train inside the Crazyflow simulator.

SB3 is the friendly, batteries-included RL library used in many tutorials
(including gym-pybullet-drones). It expects a *vectorized environment* that
follows SB3's own ``VecEnv`` rules. Crazyflow is also vectorized, but it follows
the (slightly different) Gymnasium vector rules. This file is a small translator
between the two so a K12 student can just write::

    from stable_baselines3 import PPO
    model = PPO("MlpPolicy", env)
    model.learn(100_000)

You do not need to understand every line here. The important idea: it forwards
``reset``/``step`` to Crazyflow and reshapes the outputs into what SB3 wants,
including the per-episode reward summary SB3 prints as ``ep_rew_mean``.

A note for the curious / for teachers
-------------------------------------
Crazyflow auto-resets a finished drone on the *next* step (Gymnasium
"NEXT_STEP" mode), while SB3 assumes the reset happens on the *same* step. This
causes a harmless one-step seam at episode boundaries. For a stable task like
hovering it has no practical effect on learning. The fully "correct" alternative
is the CleanRL script, which talks to Crazyflow directly.
"""

from __future__ import annotations

from typing import Any, Optional

import jax.numpy as jnp
import numpy as np
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvStepReturn


class CrazyflowSB3VecEnv(VecEnv):
    """Wrap a (JAX) Crazyflow vector env so SB3 can train on it.

    Expects ``env`` to produce a flat observation vector and accept actions in
    ``[-1, 1]``, i.e. typically::

        FlattenObservation(NormalizeActions(HoverEnv(...)))

    SB3 speaks NumPy on the CPU; Crazyflow speaks JAX (maybe on the GPU). This
    class does the conversion in both directions, placing actions on the
    simulator's device so CPU and GPU runs both work.
    """

    def __init__(self, env, tilt_action_coef: float = 0.0):
        self.env = env
        # The JAX device the physics runs on (cpu or gpu), so we can put actions there.
        self._device = env.unwrapped.device
        # Penalty on the *commanded* roll/pitch magnitude. Crazyflow's attitude
        # controller is soft, so without this the policy learns to command huge
        # (saturated) tilts that don't transfer to a real flight controller. Penalizing
        # the command keeps tilts small and makes the policy deployable (sim-to-real).
        self.tilt_action_coef = float(tilt_action_coef)
        super().__init__(
            num_envs=env.num_envs,
            observation_space=env.single_observation_space,
            action_space=env.single_action_space,
        )
        self._actions: Optional[np.ndarray] = None
        # Bookkeeping so we can report finished-episode reward and length to SB3.
        self._episode_return = np.zeros(self.num_envs, dtype=np.float64)
        self._episode_length = np.zeros(self.num_envs, dtype=np.int64)

    # SB3 splits a step into "send the actions" and "collect the results".
    def step_async(self, actions: np.ndarray) -> None:
        self._actions = actions

    def step_wait(self) -> VecEnvStepReturn:
        # NumPy action from SB3 -> JAX array on the physics device.
        jax_actions = jnp.asarray(self._actions, device=self._device)
        obs, rewards, terminated, truncated, _ = self.env.step(jax_actions)
        # Use np.array (a writable copy): SB3 writes into obs and rewards
        # (e.g. when bootstrapping the value of a timed-out episode).
        obs = np.array(obs)
        rewards = np.array(rewards, dtype=np.float32)
        terminated = np.asarray(terminated)
        truncated = np.asarray(truncated)
        dones = np.logical_or(terminated, truncated)

        # Penalize large commanded tilts (action[:, 0:2] = roll, pitch in [-1, 1]).
        if self.tilt_action_coef > 0.0:
            a = np.asarray(self._actions, dtype=np.float32)
            rewards = rewards - self.tilt_action_coef * (a[:, 0] ** 2 + a[:, 1] ** 2)

        # Track running episode totals.
        self._episode_return += rewards
        self._episode_length += 1

        infos: list[dict[str, Any]] = [{} for _ in range(self.num_envs)]
        for i in np.nonzero(dones)[0]:
            # SB3 reads these to bootstrap value estimates correctly.
            infos[i]["terminal_observation"] = obs[i]
            infos[i]["TimeLimit.truncated"] = bool(truncated[i] and not terminated[i])
            # SB3's logger reads info["episode"] to compute ep_rew_mean / ep_len_mean.
            infos[i]["episode"] = {
                "r": float(self._episode_return[i]),
                "l": int(self._episode_length[i]),
            }
            self._episode_return[i] = 0.0
            self._episode_length[i] = 0

        return obs, rewards, dones, infos

    def reset(self) -> np.ndarray:
        obs, _ = self.env.reset()
        self._episode_return[:] = 0.0
        self._episode_length[:] = 0
        return np.array(obs)  # writable copy

    def close(self) -> None:
        self.env.close()

    def render(self, mode: Optional[str] = None):
        return self.env.render()

    # --- The methods below are required by the VecEnv interface. For our single
    # --- wrapped batch they have simple implementations.
    def seed(self, seed: Optional[int] = None):
        return [seed for _ in range(self.num_envs)]

    def get_attr(self, attr_name: str, indices=None) -> list:
        value = getattr(self.env, attr_name, getattr(self, attr_name, None))
        return [value for _ in self._indices(indices)]

    def set_attr(self, attr_name: str, value, indices=None) -> None:
        setattr(self.env, attr_name, value)

    def env_method(self, method_name: str, *args, indices=None, **kwargs) -> list:
        method = getattr(self.env, method_name)
        return [method(*args, **kwargs) for _ in self._indices(indices)]

    def env_is_wrapped(self, wrapper_class, indices=None) -> list:
        return [False for _ in self._indices(indices)]

    def _indices(self, indices):
        if indices is None:
            return range(self.num_envs)
        if isinstance(indices, int):
            return [indices]
        return indices
