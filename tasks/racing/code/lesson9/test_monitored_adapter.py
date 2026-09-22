"""Checks for MonitoredSB3Adapter (main venv, in the container):

    python tasks/racing/code/lesson9/test_monitored_adapter.py        # no pytest needed

MECHANICS ONLY -- constructs a tiny env, steps it, checks the `episode` info key appears on done and that
reward/length accumulate correctly. No `.learn()` anywhere.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from contrast_env import ContrastTrackingEnv  # noqa: E402
from monitored_adapter import MonitoredSB3Adapter  # noqa: E402


def test_episode_key_appears_only_on_done_and_resets_after():
    env = ContrastTrackingEnv(num_envs=4, seed=0, v3=True, v5=True)
    adapter = MonitoredSB3Adapter(env)
    adapter.reset()
    zero_action = np.zeros((4, 4), dtype=np.float32)
    saw_episode = False
    for _ in range(700):                              # episode_time=6s * freq=100 = 600 steps: guarantees a done
        adapter.step_async(zero_action)
        obs, reward, done, infos = adapter.step_wait()
        for i in np.flatnonzero(done):
            saw_episode = True
            ep = infos[i]["episode"]
            assert set(ep) == {"r", "l", "t"}
            assert ep["l"] > 0 and np.isfinite(ep["r"])
            assert adapter._ep_rew[i] == 0.0 and adapter._ep_len[i] == 0    # reset after reporting
        for i in range(4):
            if i not in np.flatnonzero(done):
                assert "episode" not in infos[i]        # never present on a non-done env
    assert saw_episode, "no env finished an episode in 700 steps -- inconclusive, re-check episode_time/freq"


def test_reward_and_length_accumulate_correctly():
    env = ContrastTrackingEnv(num_envs=2, seed=1, v3=True, v5=True)
    adapter = MonitoredSB3Adapter(env)
    adapter.reset()
    action = np.zeros((2, 4), dtype=np.float32)
    manual_rew, manual_len = np.zeros(2), np.zeros(2, dtype=int)
    for step in range(50):
        adapter.step_async(action)
        obs, reward, done, infos = adapter.step_wait()
        manual_rew += reward
        manual_len += 1
        for i in np.flatnonzero(done):
            manual_rew[i] = 0.0
            manual_len[i] = 0
        np.testing.assert_allclose(adapter._ep_rew, manual_rew, atol=1e-5)
        np.testing.assert_array_equal(adapter._ep_len, manual_len)


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
