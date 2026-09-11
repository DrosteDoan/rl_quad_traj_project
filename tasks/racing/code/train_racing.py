"""Train a racing-envelope tracking policy in ONE run — Lesson 7 §2.

    python tasks/racing/code/train_racing.py --timesteps 4000000 --seed 0 --reason "..."

Same policy (asymmetric actor-critic, Lesson 2's `--v5`), same environment, same PPO
settings and the same 56-number observation as the course's baseline, so the result loads
in `race_bridge.py` and `race_eval.py` unchanged. The ONE difference is the distribution
of reference trajectories the policy trains on:

    baseline (`ppo_train --v5`):  |v| up to U(0.5, 3.5) m/s,  |a| up to U(1, 10) m/s^2
    this file (`racing` envelope): |v| up to U(1.0, 5.0) m/s,  |a| up to U(3, 15) m/s^2

That envelope already exists in the vendored environment -- it is what the parent
project's body-rate (CTBR) acrobatic policies train on (`datt_env.py`, `_sample_traj`) --
but the attitude-mode policy the race accepts never saw it. The parent project's best racer
is a body-rate policy reached through four acrobatic training recipes (acro2 -> 4.2), an
interface the race does not allow. The question this script answers: does ONE run on the
right envelope, in the LEGAL interface, close the gap to the model-based trackers?

Knobs (one variable per experiment, Lesson 4):
    --vel-max / --acc-max     the top of the sampled envelope (default 5.0 / 15.0)
    --seg-min / --seg-max     segment duration range of the chained quintics (default 1.0 / 2.5;
                              shorter segments = more direction changes per second)
    --perturb                 keep the DATT per-episode force perturbation (default on; --no-perturb
                              trains a nominal-only policy for the Lesson 7 disturbance experiment)

Output: tasks/racing/crazy_track/results/<stamp>_racing-train/datt_ppo_final.zip (+ tensorboard).
"""

from __future__ import annotations

import argparse

import numpy as np

from crazy_track.envs.datt_env import START, WINDOW, WINDOW_DT, DATTTrackingEnv
from crazy_track.eval.runlog import RunLogger
from crazy_track.trajectories import ChainedPolyTrajectory


class RacingTrackingEnv(DATTTrackingEnv):
    """DATTTrackingEnv (v5) with the racing reference envelope, nothing else changed."""

    def __init__(self, *args, vel_max: float = 5.0, acc_max: float = 15.0,
                 seg_min: float = 1.0, seg_max: float = 2.5, perturb: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.vel_max, self.acc_max = float(vel_max), float(acc_max)
        self.seg_min, self.seg_max = float(seg_min), float(seg_max)
        self.perturb = perturb

    def _sample_traj(self, i: int) -> None:
        # the per-env maneuver bookkeeping the base class expects (no flips here)
        self._is_flip[i] = False
        self._man_dir[i] = 0.0
        self._rot_acc[i] = 0.0
        self._rot_done[i] = True
        vel_range = float(self.rng.uniform(1.0, self.vel_max))
        acc_range = float(self.rng.uniform(3.0, self.acc_max))
        self._traj[i] = ChainedPolyTrajectory.random(
            self.rng, duration=self.max_steps / self.freq + WINDOW * WINDOW_DT + 1.0,
            seg_duration=float(self.rng.uniform(self.seg_min, self.seg_max)), pos_range=1.0,
            vel_range=vel_range, acc_range=acc_range, start_pos=START,
        )

    def _sample_perturb(self, mask: np.ndarray) -> None:
        if self.perturb:
            super()._sample_perturb(mask)
        else:
            self.perturb_force[mask] = 0.0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--timesteps", type=int, default=4_000_000)
    p.add_argument("--n-envs", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--reason", required=True)
    p.add_argument("--vel-max", type=float, default=5.0)
    p.add_argument("--acc-max", type=float, default=15.0)
    p.add_argument("--seg-min", type=float, default=1.0)
    p.add_argument("--seg-max", type=float, default=2.5)
    p.add_argument("--no-perturb", action="store_true")
    p.add_argument("--tag", default="racing-train")
    args = p.parse_args()

    from stable_baselines3 import PPO

    from crazy_track.training.asymmetric import AsymmetricPolicy
    from crazy_track.training.ppo_train import SB3Adapter

    log = RunLogger(tag=args.tag, reason=args.reason, config={**vars(args), "recipe": "racing-v5"})
    print(f"Logging to {log.dir}", flush=True)
    env = SB3Adapter(RacingTrackingEnv(num_envs=args.n_envs, seed=args.seed, v3=True, v5=True,
                                       vel_max=args.vel_max, acc_max=args.acc_max,
                                       seg_min=args.seg_min, seg_max=args.seg_max,
                                       perturb=not args.no_perturb))
    print(f"obs dim: {env.env.single_observation_space.shape} (must be 56: the v5 layout)", flush=True)
    model = PPO(
        AsymmetricPolicy, env, verbose=1, seed=args.seed,
        n_steps=256, batch_size=1024, learning_rate=3e-4, gamma=0.98,
        tensorboard_log=str(log.dir / "tb"),
    )
    model.learn(total_timesteps=args.timesteps, progress_bar=False)
    model.save(log.dir / "datt_ppo_final")
    print(f"Saved model to {log.dir / 'datt_ppo_final.zip'}", flush=True)


if __name__ == "__main__":
    main()
