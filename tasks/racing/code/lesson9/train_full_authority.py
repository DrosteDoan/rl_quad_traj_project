"""Train the control-authority screen for Lesson 9 -- design in `full_authority_env.py`'s module docstring.

    python tasks/racing/code/lesson9/train_full_authority.py --reason "..."

**THIS SCRIPT HAS NOT BEEN RUN.** Written 2026-09-23 at the user's request ("go ahead") after a cheap,
already-run diagnostic (flying M1 capped to RL's own 0.7x roll/pitch ceiling, `mpcdev:...,rp_max=0.7`) showed
the cap alone does not explain the RL crashes -- M1 still completed all three tracks checked, just with
2-3x worse precision on the tightest-margin one -- but left open whether the SAME cap costs something on top
of the RL family's existing (much larger) tracking-precision gap. Checked by constructing
`FullAuthorityTrackingEnv`, resetting it, and stepping it a few times (`test_full_authority_env.py`), never
by calling `.learn()`. Launching a real run is a separate, explicit decision.

ONE seed only (0 -- forced, not a CLI choice), reusing the SAME RNG stream as `robust_s0`/`robust_s0_screen`,
so this isolates control authority as the only variable relative to `robust_s0_screen` specifically (the
existing seed-0 checkpoint already at the corrected gamma/n_steps/batch_size defaults -- this screen uses
those same corrected defaults, not the original round 1-3 settings, for a clean single-variable comparison
against the MOST CURRENT seed-0 baseline). This is a screen, not a new group in the study roster: if it
meaningfully closes the gap, a design decision follows (retrain all 6 seeds at full authority, or fold it in
alongside whichever reference-distribution option gets chosen); if it barely moves the numbers, authority was
never the dominant explanation and the reference-distribution question stands alone.

What actually differs from `train_robust.py` (`FullAuthorityTrackingEnv`, `full_authority_env.py`):
  * `_denorm_action`'s roll/pitch scale is `RPY_MAX * 1.0` instead of the vendored `* 0.7`, matching the MPC
    family's own `rp_max=1.0` default exactly (`mpc_dev.py`). Nothing else -- same reward, same disturbance
    training, same window, same frequency, same envelope as `RobustTrackingEnv`.

Output: tasks/racing/crazy_track/results/<stamp>_racing-full-authority-train/datt_ppo_final.zip (+ tensorboard).
Evaluate ONLY through `full_authority_policy.py` (`driver.py`'s `robust_full:<path>` spec) -- `robust:<path>`
would silently apply the wrong (0.7x) scale to a model trained expecting the full range.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from full_authority_env import RPY_SCALE, FullAuthorityTrackingEnv  # noqa: E402
from robust_env import DR_LAM_MAX, FORCE_XY_MAX, FORCE_Z_HI, FORCE_Z_LO, FREQ, WINDOW_DT  # noqa: E402

from crazy_track.eval.runlog import RunLogger  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--timesteps", type=int, default=8_000_000)
    p.add_argument("--n-envs", type=int, default=16)
    p.add_argument("--seed", type=int, default=0, choices=[0],
                   help="one seed only, reusing robust_s0/robust_s0_screen's RNG stream (see the module "
                        "docstring) -- this is a single-variable screen, not a new group in the roster")
    p.add_argument("--reason", required=True)
    p.add_argument("--vel-max", type=float, default=5.0)
    p.add_argument("--acc-max", type=float, default=15.0)
    p.add_argument("--seg-min", type=float, default=1.0)
    p.add_argument("--seg-max", type=float, default=2.5)
    p.add_argument("--dr-lam-max", type=float, default=DR_LAM_MAX)
    p.add_argument("--no-perturb", action="store_true")
    p.add_argument("--gamma", type=float, default=0.99,
                   help="the corrected default (matches robust_s0_screen, not round 1-3's 0.98) -- this "
                        "screen isolates control authority alone against the most current seed-0 baseline")
    p.add_argument("--n-steps", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--tag", default="racing-full-authority-train")
    return p


def main() -> None:
    args = build_parser().parse_args()

    from stable_baselines3 import PPO

    from crazy_track.training.asymmetric import AsymmetricPolicy

    from monitored_adapter import MonitoredSB3Adapter

    log = RunLogger(tag=args.tag, reason=args.reason,
                    config={**vars(args), "recipe": "racing-full-authority-v5", "rpy_scale": RPY_SCALE,
                            "freq": FREQ, "window_dt": WINDOW_DT,
                            "force_xy_max": FORCE_XY_MAX, "force_z": [FORCE_Z_LO, FORCE_Z_HI]})
    print(f"Logging to {log.dir}", flush=True)
    env = MonitoredSB3Adapter(FullAuthorityTrackingEnv(num_envs=args.n_envs, seed=args.seed, v3=True, v5=True,
                                       vel_max=args.vel_max, acc_max=args.acc_max,
                                       seg_min=args.seg_min, seg_max=args.seg_max,
                                       perturb=not args.no_perturb, dr_lam_max=args.dr_lam_max))
    print(f"obs dim: {env.env.single_observation_space.shape} (must be 56: the v5 layout, unchanged)", flush=True)
    model = PPO(
        AsymmetricPolicy, env, verbose=1, seed=args.seed,
        n_steps=args.n_steps, batch_size=args.batch_size, learning_rate=3e-4, gamma=args.gamma,
        tensorboard_log=str(log.dir / "tb"),
    )
    model.learn(total_timesteps=args.timesteps, progress_bar=False)
    model.save(log.dir / "datt_ppo_final")
    print(f"Saved model to {log.dir / 'datt_ppo_final.zip'}", flush=True)


if __name__ == "__main__":
    main()
