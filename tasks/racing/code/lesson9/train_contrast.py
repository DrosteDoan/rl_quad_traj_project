"""Train a contrast-group policy for Lesson 9 -- design in METHODOLOGY.md section 5b.

    python tasks/racing/code/lesson9/train_contrast.py --timesteps 8000000 --seed 0 --reason "..."

**THIS SCRIPT HAS NOT BEEN RUN.** Written 2026-09-22 at the user's request ("build that ... seeds 0-2");
checked by constructing `ContrastTrackingEnv`, resetting it, and stepping it a few times
(`test_contrast_env.py`), never by calling `.learn()`. Launching a real run is a separate, explicit decision.

Why this group exists: the robust-RL recipe (`train_robust.py`) matches its domain-randomization ranges to
the study's frozen ceilings precisely -- three independent channels, each at 0.8x the same numbers the
study measures against. That precision risks a reasonable objection: does a crossover near the training
edge reflect something about learned control, or just that the policy was trained on the exam? This group
answers it by holding every OTHER difference from the robust recipe equal (racing envelope, PPO settings,
the 0.8 s preview window, freq=100 -- see `contrast_env.py`'s docstring for why those are held equal, not
varied) and reverting ONLY the disturbance-training ranges to the vendored, unmatched defaults: a +-3.5 m/s^2
force box, a Lighthouse noise SCALE in U(0, 1.5) (uncoupled from the update rate), and no mass channel.

Three seeds (0, 1, 2), matching the robust group's count -- the user's reasoning: 6 seeds total split into
3 concurrent-training-pairs uses every slot under the 2-concurrent-session hardware limit; 5 would waste
one. It also means neither group is trusted on fewer seeds than the other (Lesson 7's `racing_s1`: a single
bad seed is not a bad recipe, and reading that plainly needs a comparable seed count on both sides).

Chart treatment (agreed, not yet built -- phase 8): thin/muted lines, own colour family, distinct from both
the model-based and robust-RL lines; NOT folded into either group's story. Used only to check whether the
robust family's crossover location moves relative to this baseline, or stays where it is.

Output: tasks/racing/crazy_track/results/<stamp>_racing-contrast-train/datt_ppo_final.zip (+ tensorboard).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from contrast_env import FREQ, WINDOW_DT, ContrastTrackingEnv  # noqa: E402

from crazy_track.eval.runlog import RunLogger  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--timesteps", type=int, default=8_000_000,
                   help="env-step budget; DOUBLE train_racing.py's 4,000,000 because freq=100 halves the "
                        "simulated seconds per step (same reasoning as train_robust.py)")
    p.add_argument("--n-envs", type=int, default=16)
    p.add_argument("--seed", type=int, default=0, choices=[0, 1, 2],
                   help="3 seeds only, matching the robust group's count (see the module docstring)")
    p.add_argument("--reason", required=True)
    p.add_argument("--vel-max", type=float, default=5.0)
    p.add_argument("--acc-max", type=float, default=15.0)
    p.add_argument("--seg-min", type=float, default=1.0)
    p.add_argument("--seg-max", type=float, default=2.5)
    p.add_argument("--no-perturb", action="store_true",
                   help="disables the vendored force channel entirely (a nominal-only contrast; not the default)")
    p.add_argument("--gamma", type=float, default=0.99,
                   help="discount factor. 0.99 restores the vendored recipe's ~1.0 s effective value-"
                        "function horizon at freq=100 (0.98 at freq=50 gave 1/(1-0.98)/50 = 1.0 s; the same "
                        "0.98 at freq=100 halves that to 0.5 s -- shorter than the 0.8 s observation window "
                        "the policy is trained to look ahead over). Override to 0.98 to reproduce round 1-3's "
                        "exact setting.")
    p.add_argument("--n-steps", type=int, default=512,
                   help="rollout length per env before each PPO update. 512 restores the vendored recipe's "
                        "~5.12 s / ~85%% episode-coverage rollout window at freq=100 (256 at freq=50 covered "
                        "256/50=5.12 s of a 300-step episode; the same 256 at freq=100 covers only 2.56 s of "
                        "a 600-step episode, 43%% instead of 85%%). Override to 256 to reproduce round 1-3.")
    p.add_argument("--batch-size", type=int, default=2048,
                   help="scaled with --n-steps to keep the same 4-minibatch-per-epoch structure (n_steps * "
                        "n_envs / batch_size = 4, as in the vendored 256*16/1024). Override to 1024 with "
                        "--n-steps 256 to reproduce round 1-3.")
    p.add_argument("--tag", default="racing-contrast-train")
    return p


def main() -> None:
    args = build_parser().parse_args()

    from stable_baselines3 import PPO

    from crazy_track.training.asymmetric import AsymmetricPolicy

    from monitored_adapter import MonitoredSB3Adapter

    log = RunLogger(tag=args.tag, reason=args.reason,
                    config={**vars(args), "recipe": "racing-contrast-v5", "freq": FREQ, "window_dt": WINDOW_DT})
    print(f"Logging to {log.dir}", flush=True)
    env = MonitoredSB3Adapter(ContrastTrackingEnv(num_envs=args.n_envs, seed=args.seed, v3=True, v5=True,
                                         vel_max=args.vel_max, acc_max=args.acc_max,
                                         seg_min=args.seg_min, seg_max=args.seg_max,
                                         perturb=not args.no_perturb))
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
