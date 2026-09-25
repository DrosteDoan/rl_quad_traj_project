"""A drop-in replacement for `crazy_track.training.ppo_train.SB3Adapter` that populates the `"episode"` key
SB3 needs to fill `ep_info_buffer` and log `rollout/ep_rew_mean` / `rollout/ep_len_mean` (Lesson 9 phase 6).

The vendored `SB3Adapter.step_wait()` never sets `infos[i]["episode"]`, so SB3's `_update_info_buffer`
(`stable_baselines3/common/base_class.py`) never has anything to add to `ep_info_buffer`, and
`OnPolicyAlgorithm.train()` skips the `rollout/ep_rew_mean` / `ep_len_mean` log lines entirely -- not
intermittently, structurally: `train_racing.py`'s own runs have the same gap, unnoticed until now because
those policies were validated by flying them afterward, not by watching training curves live.

Purely additive and purely for logging: `_update_info_buffer` only appends to `ep_info_buffer` /
`ep_success_buffer`, which nothing in the actual PPO update (GAE, the policy/value loss, clipping) reads --
confirmed by inspecting `on_policy_algorithm.py` and `ppo.py`. Wrapping the adapter changes what gets
LOGGED, not how the policy trains.
"""

from __future__ import annotations

import time

import numpy as np

from crazy_track.envs.datt_env import DATTTrackingEnv
from crazy_track.training.ppo_train import SB3Adapter


class MonitoredSB3Adapter(SB3Adapter):
    def __init__(self, env: DATTTrackingEnv):
        super().__init__(env)
        self._ep_rew = np.zeros(env.num_envs)
        self._ep_len = np.zeros(env.num_envs, dtype=np.int64)
        self._t0 = time.time()

    def step_wait(self):
        obs, reward, done, infos = super().step_wait()
        self._ep_rew += reward
        self._ep_len += 1
        for i in np.flatnonzero(done):
            infos[i]["episode"] = {"r": float(self._ep_rew[i]), "l": int(self._ep_len[i]),
                                   "t": round(time.time() - self._t0, 3)}
            self._ep_rew[i] = 0.0
            self._ep_len[i] = 0
        return obs, reward, done, infos
