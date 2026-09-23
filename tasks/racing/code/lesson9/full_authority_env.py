"""The control-authority screen (Lesson 9, 2026-09-23): isolates ONE variable against `RobustTrackingEnv` --
whether the 0.7x roll/pitch authority cap every RL policy in this course has trained under (since the very
first commit of the vendored racing task, `datt_env.py`'s `_denorm_action`) is costing the robust group
precision relative to the MPC family, which gets the full `RPY_MAX` (`mpc_dev.py`'s `rp_max`, 1.0 rad by
default, a hard OCP bound -- confirmed cheaply testable without touching that file, since `rp_max` is already
spec-string configurable: `mpcdev:...,rp_max=0.7` flew all three of the M1 same-path tracks it was checked on
even capped to RL's own ceiling, ruling OUT authority alone as sufficient to explain the RL crashes, but
still leaving open whether it costs RL something on top of its existing tracking-precision gap).

Same racing envelope, same disturbance-training ranges, same preview window (0.8 s) and frequency (100 Hz),
same reward -- everything `RobustTrackingEnv` already has. The ONE thing this class changes is the roll/pitch
scale baked into `_denorm_action` (vendored, `datt_env.py`, cannot be edited in place): 1.0x `RPY_MAX` instead
of 0.7x, matching the MPC family's own `rp_max=1.0` default exactly. Nothing else about the action interface
changes -- thrust bounds, yaw-fixed-to-0, and the underlying PPO action distribution's own [-1, 1] range are
all untouched, so this widens what a given raw action CAN command physically, not what the network is asked
to produce.

Deliberately reuses `RobustTrackingEnv`'s force/Lighthouse/mass domain randomization unchanged (not the
contrast group's vendored ranges) -- this screen is about ONE robust-recipe seed with authority as the only
new variable, evaluated against the existing `robust_s0_screen` checkpoint (corrected hyperparameters, same
seed 0 RNG stream), not a new group in the study roster.

NOT yet used to train anything -- see `train_full_authority.py`'s module docstring.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from robust_env import RobustTrackingEnv  # noqa: E402

from crazy_track.controllers.utils import RPY_MAX, THRUST_MAX, THRUST_MIN  # noqa: E402

RPY_SCALE = 1.0   # vs the vendored/robust default of 0.7 -- matches mpc_dev.py's rp_max=1.0 default exactly


class FullAuthorityTrackingEnv(RobustTrackingEnv):
    """`RobustTrackingEnv` with `_denorm_action`'s roll/pitch scale widened from 0.7x `RPY_MAX` to 1.0x --
    everything else (reward, disturbance training, window, frequency) is `RobustTrackingEnv`, untouched."""

    def _denorm_action(self, a: np.ndarray) -> np.ndarray:
        assert not self.ctbr, "FullAuthorityTrackingEnv only handles the attitude-mode branch (ctbr=False)"
        cmd = np.zeros((self.num_envs, 1, 4), dtype=np.float32)
        cmd[:, 0, 0:2] = np.clip(a[:, 0:2], -1, 1) * RPY_MAX * RPY_SCALE
        cmd[:, 0, 2] = 0.0  # yaw fixed, unchanged
        thrust = THRUST_MIN + (np.clip(a[:, 3], -1, 1) + 1) * 0.5 * (THRUST_MAX - THRUST_MIN)
        cmd[:, 0, 3] = thrust
        return cmd
