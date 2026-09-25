"""The robust-RL training environment (Lesson 9 phase 6): `train_racing.py`'s racing envelope, plus a
per-episode domain-randomization box matched to the frozen ceilings (METHODOLOGY.md section 5a), plus the
preview-window and training-frequency changes agreed with the user on 2026-09-22.

Three independent per-episode channels, each its own RNG stream (mirroring how the vendored recipe already
keeps force and Lighthouse noise independent -- METHODOLOGY.md section 5a), all set to 0.8 x the frozen
ceiling:

    force        x, y: +-4.4 m/s^2 (0.8 x wind_const's 1.5x-of-Lesson-7 ceiling, unioned with wind_gust's)
                 z:    -3.6 to +1.8 m/s^2 (0.8 x payload's 2.0x-of-Lesson-7 ceiling, downward-biased)
    lighthouse   lambda in U(0, 0.8) of the study's own axis, sizes + update interval coupled
                 (`lighthouse_batch.LambdaLighthouseSensorBatch`, not the vendored noise-scale sensor)
    mass         lambda in U(0, 0.8) of the study's own axis, heavier only, via `knobs.mass_scale`

WINDOW stays 10 (unchanged observation dimensionality: OBS_DIM depends only on the WINDOW count, not the
spacing -- `datt_env.py`'s OBS_DIM = 3+3+4+3*WINDOW), but WINDOW_DT becomes 0.08 s (10 x 0.08 = 0.8 s, matching
the MPC family's horizon; the vendored 0.06 s gives only 0.6 s). freq is fixed at 100 (was 50), so the deployed
policy's own L1 estimator (in `robust_policy.py`) is integrated at the SAME rate in training and evaluation --
no more info at https://knobs.py than the frozen scales, and no plumbing beyond what `freq` already threads
through `DATTTrackingEnv.__init__` (the L1 estimator's dt, the Lighthouse control_freq, `n_substeps`).

NOT yet used to train anything -- see `train_robust.py`'s module docstring.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CODE = HERE.parent
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(HERE))
import knobs as kb  # noqa: E402
from lighthouse_batch import LambdaLighthouseSensorBatch  # noqa: E402
from train_racing import RacingTrackingEnv  # noqa: E402

from crazy_track.envs.datt_env import MASS, WINDOW  # noqa: E402

FREQ = 100                     # fixed: matches the harness, removes the L1-estimator dt mismatch (limitation 14)
WINDOW_DT = 0.08               # 10 * 0.08 = 0.8 s, matching the M1 family's horizon (was 0.06 = 0.6 s)
FORCE_XY_MAX = 4.4             # m/s^2; 0.8 * 1.5 * wind_const's Lesson-7 value (2.54 ~= 4.4/1.5/0.8)
FORCE_Z_LO, FORCE_Z_HI = -3.6, 1.8   # m/s^2; 0.8 * 2.0 * payload's Lesson-7 value, small upward margin
DR_LAM_MAX = 0.8                # the training box: 0.8 of the study's own lambda axis, for lighthouse and mass


class RobustTrackingEnv(RacingTrackingEnv):
    """`RacingTrackingEnv` (the racing envelope) with force/Lighthouse/mass domain randomization matched to
    the study's frozen ceilings, a 0.8 s preview window, and freq=100. Everything else -- policy class, PPO
    settings, reward, the 56-number v5 observation shape -- is untouched."""

    def __init__(self, *args, dr_lam_max: float = DR_LAM_MAX, window_dt: float = WINDOW_DT, **kwargs):
        kwargs["freq"] = FREQ                              # not a caller-adjustable kwarg: this IS the design
        super().__init__(*args, **kwargs)
        self.dr_lam_max = float(dr_lam_max)
        self._t_offsets = float(window_dt) * np.arange(1, WINDOW + 1)   # datt_env.py line 107, same count, new spacing
        if self.noisy_sensor:
            # replace the vendored noise-scale sensor with the lambda-coupled one (same external interface)
            self.sensor = LambdaLighthouseSensorBatch(self.num_envs, control_freq=FREQ, lam_max=self.dr_lam_max,
                                                       seed=self.rng.integers(1 << 30))
            self.sensor.reset()
        self._mass_rng = np.random.default_rng(int(kwargs.get("seed", 0)) + 2)   # its own stream, like Lighthouse's seed+1
        self._mass_mult = np.ones(self.num_envs)

    def _sample_perturb(self, mask: np.ndarray) -> None:
        """Overrides the vendored PERTURB_ACC_MAX=3.5 box with the study-matched ranges, AND samples mass --
        piggybacking on this hook because the base class already calls it at both resample points (full
        `reset()` and, per env, inside `step()`'s done-mask branch), independently of `_sample_traj`."""
        n = int(mask.sum())
        acc = np.empty((n, 3))
        acc[:, 0] = self.rng.uniform(-FORCE_XY_MAX, FORCE_XY_MAX, size=n)
        acc[:, 1] = self.rng.uniform(-FORCE_XY_MAX, FORCE_XY_MAX, size=n)
        acc[:, 2] = self.rng.uniform(FORCE_Z_LO, FORCE_Z_HI, size=n)
        self.perturb_force[mask] = MASS * acc
        self._sample_mass(mask)

    def _sample_mass(self, mask: np.ndarray) -> None:
        n = int(mask.sum())
        lam = self._mass_rng.uniform(0.0, self.dr_lam_max, size=n)
        mult = np.array([kb.mass_scale("mass_mult", float(x)) for x in lam])
        self._mass_mult[mask] = mult
        default_mass = np.asarray(self.sim.default_data.params.mass)      # (num_envs, n_drones, 1), nominal
        new_mass = np.asarray(self.sim.data.params.mass).copy()
        new_mass[mask] = default_mass[mask] * mult[:, None, None]
        import jax.numpy as jnp

        self.sim.data = self.sim.data.replace(params=self.sim.data.params.replace(mass=jnp.asarray(new_mass)))
