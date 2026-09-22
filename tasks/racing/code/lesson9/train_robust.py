"""Train a robust racing policy for Lesson 9 -- design in METHODOLOGY.md section 5a.

    python tasks/racing/code/lesson9/train_robust.py --timesteps 8000000 --seed 0 --reason "..."

**THIS SCRIPT HAS NOT BEEN RUN.** Written 2026-09-22 at the user's request ("go ahead and do that, but do
not start training"); it has been checked by constructing `RobustTrackingEnv`, resetting it, and stepping
it a few times (`test_robust_env.py`), never by calling `.learn()`. Launching a real run is a separate,
explicit decision.

Same policy (asymmetric actor-critic), same PPO settings, same 56-number v3/v5 observation layout as
`train_racing.py` (Lesson 7 section 2) and `RacingTrackingEnv`'s racing envelope for `_sample_traj` --
unchanged, so a trained model is loadable by anything that loads a v5 model BY SHAPE. It is NOT
loadable by `race_bridge.py`/`DATTPolicyController` for CORRECT inference, though: those import the
vendored `WINDOW_DT` (0.6 s) and would silently feed the wrong reference window to a model trained with
`RobustTrackingEnv`'s 0.8 s spacing. Evaluate a model from this script only through `robust_policy.py`
(`driver.py`'s `robust:<path>` spec), never through `datt:<path>`.

What actually differs from `train_racing.py` (`RobustTrackingEnv`, `robust_env.py`):
  * freq = 100 (was 50) -- matches the harness; removes the deployed L1 estimator's train/test dt mismatch
    (limitation 14). Halves the simulated experience per env-step at a fixed --timesteps, hence the default
    below is DOUBLE train_racing.py's (8,000,000 vs 4,000,000) to keep it comparable;
  * the reference preview window is 0.8 s (WINDOW_DT=0.08, WINDOW=10 unchanged: same 56-number shape),
    matching the MPC family's horizon instead of the vendored 0.6 s;
  * the per-episode force perturbation range is matched to the study's frozen ceilings at 0.8x instead of
    the vendored +-3.5 m/s^2 box (x,y: +-4.4 m/s^2; z: -3.6 to +1.8 m/s^2);
  * the Lighthouse sensor draws a per-episode lambda in U(0, 0.8) (the study's own axis) instead of a noise
    SCALE in U(0, 1.5), coupling the update interval to the same lambda as the size errors;
  * a THIRD channel, mass, drawn as a per-episode lambda in U(0, 0.8) and applied as `knobs.mass_scale`
    (heavier only), absent entirely from the vendored recipe.
  * all three channels are independent per-episode draws, each its own RNG stream (METHODOLOGY.md section
    5a's "what independence costs" note applies here, specifically to the force/Lighthouse pair).

Output: tasks/racing/crazy_track/results/<stamp>_racing-robust-train/datt_ppo_final.zip (+ tensorboard).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from robust_env import DR_LAM_MAX, FORCE_XY_MAX, FORCE_Z_HI, FORCE_Z_LO, FREQ, WINDOW_DT, RobustTrackingEnv  # noqa: E402

from crazy_track.eval.runlog import RunLogger  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--timesteps", type=int, default=8_000_000,
                   help="env-step budget; DOUBLE train_racing.py's 4,000,000 because freq=100 halves the "
                        "simulated seconds per step (see the module docstring)")
    p.add_argument("--n-envs", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--reason", required=True)
    p.add_argument("--vel-max", type=float, default=5.0)
    p.add_argument("--acc-max", type=float, default=15.0)
    p.add_argument("--seg-min", type=float, default=1.0)
    p.add_argument("--seg-max", type=float, default=2.5)
    p.add_argument("--dr-lam-max", type=float, default=DR_LAM_MAX,
                   help="training box, as a fraction of the study's own lambda axis (Lighthouse and mass)")
    p.add_argument("--no-perturb", action="store_true", help="also disables the mass channel (both ride the "
                                                              "same _sample_perturb hook -- see robust_env.py)")
    p.add_argument("--tag", default="racing-robust-train")
    args = p.parse_args()

    from stable_baselines3 import PPO

    from crazy_track.training.asymmetric import AsymmetricPolicy
    from crazy_track.training.ppo_train import SB3Adapter

    log = RunLogger(tag=args.tag, reason=args.reason,
                    config={**vars(args), "recipe": "racing-robust-v5",
                            "freq": FREQ, "window_dt": WINDOW_DT,
                            "force_xy_max": FORCE_XY_MAX, "force_z": [FORCE_Z_LO, FORCE_Z_HI]})
    print(f"Logging to {log.dir}", flush=True)
    env = SB3Adapter(RobustTrackingEnv(num_envs=args.n_envs, seed=args.seed, v3=True, v5=True,
                                       vel_max=args.vel_max, acc_max=args.acc_max,
                                       seg_min=args.seg_min, seg_max=args.seg_max,
                                       perturb=not args.no_perturb, dr_lam_max=args.dr_lam_max))
    print(f"obs dim: {env.env.single_observation_space.shape} (must be 56: the v5 layout, unchanged)", flush=True)
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
