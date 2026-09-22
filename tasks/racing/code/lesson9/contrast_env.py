"""The contrast-group training environment (Lesson 9 phase 6b): isolates ONE variable against
`RobustTrackingEnv` -- whether matching the domain-randomization ranges to the study's frozen ceilings
buys anything beyond generic, unmatched robustness training.

Same racing envelope, same PPO settings, same preview-window (0.8 s) and frequency (100 Hz) fixes as the
robust recipe -- those correct an asymmetry between the MPC and RL families (limitation 14) and are not
part of what this contrast is testing, so they are held equal across both groups. The ONE thing that
differs from `RobustTrackingEnv` is left untouched at its vendored default: the force perturbation box
(+-3.5 m/s^2, `DATTTrackingEnv._sample_perturb`, PERTURB_ACC_MAX in datt_env.py) and the Lighthouse sensor
(a per-episode noise SCALE in U(0, 1.5), the vendored `LighthouseSensorBatch`, not `knobs`-coupled). There
is no mass channel at all -- the vendored recipe never had one, and this group does not gain one either.

`RacingTrackingEnv._sample_perturb` already delegates to the vendored default box whenever `perturb=True`
(train_racing.py), and `DATTTrackingEnv.__init__` already builds the vendored `LighthouseSensorBatch` when
`noisy_sensor` is set -- so this subclass overrides NOTHING about disturbance training, only the window and
the frequency, which is the point: the diff against `RacingTrackingEnv` should be exactly two lines.

A model trained with this class is evaluated the SAME way a robust model is, through `robust_policy.py`'s
`RobustPolicyController` (`driver.py`'s `robust:<path>` spec) -- that wrapper's only job is to use the 0.8 s
window at the harness's own frequency, which is equally correct for a contrast-trained model; nothing about
it assumes matched disturbance training. `datt:<path>` (the vendored controller, 0.6 s window) is wrong for
either of this study's two RL groups.

NOT yet used to train anything -- see `train_contrast.py`'s module docstring.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CODE = HERE.parent
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(HERE))
from robust_env import FREQ, WINDOW_DT  # noqa: E402  (the SAME window/freq fixes, held equal across both groups)
from train_racing import RacingTrackingEnv  # noqa: E402

from crazy_track.envs.datt_env import WINDOW  # noqa: E402


class ContrastTrackingEnv(RacingTrackingEnv):
    """`RacingTrackingEnv` with the window/frequency fixes and NOTHING about disturbance training changed."""

    def __init__(self, *args, window_dt: float = WINDOW_DT, **kwargs):
        kwargs["freq"] = FREQ                              # matches RobustTrackingEnv; not a caller kwarg
        super().__init__(*args, **kwargs)
        self._t_offsets = float(window_dt) * np.arange(1, WINDOW + 1)   # matches RobustTrackingEnv's window
