from __future__ import annotations

import numpy as np

from crazy_track.controllers.base import Controller
from crazy_track.controllers.utils import RPY_MAX, THRUST_MAX, THRUST_MIN
from crazy_track.envs.datt_env import WINDOW, WINDOW_DT
from crazy_track.trajectories import Trajectory


class DATTPolicyController(Controller):
    """Runs a trained DATT PPO policy (obs/action conventions from datt_env)."""

    def __init__(self, model_path: str, control_freq: int = 100,
                 recurrent: bool = False):
        self.recurrent = recurrent  # v7: GRU actor (sb3-contrib RecurrentPPO)
        if recurrent:
            from sb3_contrib import RecurrentPPO  # local: optional dependency

            self.model = RecurrentPPO.load(model_path, device="cpu")
        else:
            from stable_baselines3 import PPO

            self.model = PPO.load(model_path, device="cpu")
        self._state = None  # GRU hidden state (v7 only)
        self._first = True
        self._t_offsets = WINDOW_DT * np.arange(1, WINDOW + 1)
        self._traj: Trajectory | None = None
        # Model detection by obs dim: v3/v4 = 43; v5 = 43 + 13 privileged (zero-
        # padded at eval, actor ignores it); v6a = 43*K stacked frames + 13.
        obs_dim = int(np.prod(self.model.observation_space.shape))
        base_v3 = 3 + 3 + 4 + 3 * WINDOW + 3
        self.v3 = True
        if obs_dim == base_v3 - 3:
            self.stack, self.pad, self.v3 = 1, 0, False
        elif obs_dim in (base_v3, base_v3 + 13):
            self.stack, self.pad = 1, obs_dim - base_v3
        elif (obs_dim - 13) % base_v3 == 0:
            self.stack, self.pad = (obs_dim - 13) // base_v3, 13
        else:
            raise ValueError(f"Unrecognized DATT model obs dim: {obs_dim}")
        self._frames: list[np.ndarray] = []
        # v6 models were trained with 50 Hz frames; at higher eval rates only
        # push every `skip`-th frame so the stack spans the same time window.
        self._skip = max(1, control_freq // 50)
        self._calls = 0
        self._rnn_calls = 0  # separate counter: v6's is deliberately not reset
        if self.v3:
            from crazy_track.controllers.l1 import L1Estimator

            self.l1 = L1Estimator(mass=0.04338, n=1, dt=1.0 / control_freq)
        self._last_thrust = 0.04338 * 9.81

    def reset(self, trajectory: Trajectory) -> None:
        self._traj = trajectory
        self._frames = []
        self._state = None
        self._first = True
        self._rnn_calls = 0
        if self.v3:
            self.l1.reset()
            self._last_thrust = 0.04338 * 9.81

    def act(self, state: np.ndarray, t: float) -> np.ndarray:
        pos, vel, quat = state[:3], state[3:6], state[6:10]
        win = self._traj.pos(t + self._t_offsets) - pos
        parts = [self._traj.pos(t) - pos, vel, quat, win.ravel()]
        if self.v3:
            sigma = self.l1.update(vel, quat, np.array([self._last_thrust]))
            parts.append(sigma[0])
        frame = np.concatenate(parts).astype(np.float32)
        if self.stack > 1:  # v6a: oldest-first frame stack, first call fills all slots
            if not self._frames:
                self._frames = [frame] * self.stack
            elif self._calls % self._skip == 0:
                self._frames = self._frames[1:] + [frame]
            else:  # hold the stack but keep the newest slot current
                self._frames[-1] = frame
            self._calls += 1
            frame = np.concatenate(self._frames)
        obs = np.concatenate([frame, np.zeros(self.pad, dtype=np.float32)]) \
            if self.pad else frame
        if self.recurrent:
            # v7 was trained on 50 Hz sequences. Query the GRU every control step
            # with a fresh observation, but only COMMIT the returned hidden state
            # every `skip`-th call, so the hidden-state sequence advances at the
            # training cadence (same trick as the v6 frame stack above).
            a, state = self.model.predict(obs, state=self._state,
                                          episode_start=np.array([self._first]),
                                          deterministic=True)
            if self._rnn_calls % self._skip == 0:
                self._state, self._first = state, False
            self._rnn_calls += 1
        else:
            a, _ = self.model.predict(obs, deterministic=True)
        rpy = np.array([a[0] * RPY_MAX * 0.7, a[1] * RPY_MAX * 0.7, 0.0])
        thrust = THRUST_MIN + (np.clip(a[3], -1, 1) + 1) * 0.5 * (THRUST_MAX - THRUST_MIN)
        self._last_thrust = float(thrust)
        return np.array([rpy[0], rpy[1], rpy[2], thrust])
