"""Train the gate-aware viability screen for Lesson 9 -- design in `gate_aware_env.py`'s module docstring.

    python tasks/racing/code/lesson9/gen_pool.py            # once: builds the train / val track pools
    python tasks/racing/code/lesson9/train_gate_aware.py --reason "..."

**v4 (robust group, seed 0) WAS RUN AND SAVED (`saved/gate_aware_v4/`); the contrast group and other seeds have not.** History: v1 (references re-origined airborne) 0/6 completions, 0/24 gates,
every failure a gate-1 contact. v2 (real ground start, floor crash at eval's -0.3) 0/6, 2/24; on its own training
tracks it completed 1 of 3, the other two swerving wide of gate 1. v3 (any missed gate ends the episode, -3.5 for
contact or a miss, `gate/*` logging) learned its training tracks -- pooled train full-lap rate 0.535, 2/3 dev
tracks -- but scored 1/6 completions, 11/24 gates on the held-out selection set, level with the open-space
baseline (10/24, 1/6): a transfer gap with a 3-track pool, still-rising curves at 8M steps. v4 attacks both at
once: (1) the training pool is the 3 dev tracks plus every track `gen_pool.py` accepted in its fixed `train` seed
windows (~66 expected; same generator and filters as the study tracks), with a separate `val` pool never trained
on; (2) 16M steps instead of 8M, with a checkpoint every --ckpt-every steps so ONE run also answers whether
longer training helps transfer or just fits the pool better (fly each checkpoint on `val` with `eval_pool.py`).
Checked by constructing `GateAwareTrackingEnv`, resetting it and stepping it (`test_gate_aware.py`), never by
calling `.learn()`. Launching a real run is a separate, explicit decision.

Judge it on `val`, not on study tracks (METHODOLOGY.md limitation 15): every RL design decision so far was made on
the 6-track selection set, which the MPC family was never tuned on. The 10 untouched study tracks stay unflown until
the recipe is frozen.

TWO GROUPS, three seeds each (the study design, METHODOLOGY.md sections 5a/5b), sharing every gate-aware element:

    --group robust    GateAwareTrackingEnv   disturbance training matched to 0.8x the study's frozen ceilings
                      (this is the saved v4 recipe at seed 0; seeds 1 and 2 are the same recipe)
    --group contrast  GateAwareContrastEnv   the vendored, unmatched ranges: +-3.5 m/s^2 force box, Lighthouse noise
                      scale U(0, 1.5) uncoupled from the update rate, NO mass channel
    --group contrast_force  GateAwareForceContrastEnv   an ABLATION, not a control group: the vendored force box and no mass channel, but v4's
                      lambda-coupled Lighthouse sensor: differs from `robust` in the force and mass channels ONLY.
                      Added 2026-09-25 after the fully vendored contrast run failed to learn in 16M steps
                      (METHODOLOGY.md section 5c).

The two differ in exactly the disturbance training (`GateAwareMixin` holds the gate logic, so it is shared by
construction, and the refactor is tested bit-identical to the saved v4 code). Why the contrast group exists: v4 was
trained on ranges matched to the same ceilings the study measures against, so its lambda > 0 robustness cannot be
told apart from "trained on the exam"; the contrast group is what can (METHODOLOGY.md section 5b). Judge both on the
`val` pool, not on study tracks (limitation 15).

What differs from `train_robust.py` (`GateAwareTrackingEnv`, `gate_aware_env.py`):
  * references are the pool tracks flown from a real ground start, 100% of episodes, drawn uniformly per episode;
  * `step()` ends the episode on gate-frame contact (`contact.py`'s exact box test), on a missed gate
    (`driver.py`'s rule), or on floor/gross divergence (`pos[:, 2] < -0.3`, eval's check, not the vendored 0.05);
  * `episode_time` is computed from the longest pool track (8.0 s for the dev tracks alone);
  * a callback writes `gate/*` scalars to tensorboard each rollout: how episodes ended (contact / miss /
    floor-or-divergence / truncated), gates passed per episode, and `gate/full_lap_frac`, the training-time
    completion rate.
  Everything else -- disturbance training, 0.8 s window, freq=100, 0.7 rad authority, PPO settings -- is
  `RobustTrackingEnv`, untouched.

Output: tasks/racing/crazy_track/results/<stamp>_racing-gate-aware-train/datt_ppo_final.zip (+ tensorboard, and
ckpt/ppo_<steps>_steps.zip every --ckpt-every steps). Evaluate through `robust:<path>` (`driver.py`) -- authority
is unchanged from `RobustTrackingEnv` (0.7 rad), so no new controller wrapper is needed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gate_aware_env import (DEV_SEEDS, EPISODE_TIME, FLOOR_PENALTY, FLOOR_Z, GATE_PENALTY, POOL_SPEC,  # noqa: E402
                            GateAwareContrastEnv, GateAwareForceContrastEnv, GateAwareTrackingEnv, summarise_gate_stats)
from robust_env import DR_LAM_MAX, FORCE_XY_MAX, FORCE_Z_HI, FORCE_Z_LO, FREQ, WINDOW_DT  # noqa: E402

from crazy_track.eval.runlog import RunLogger  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--timesteps", type=int, default=16_000_000,
                   help="16M (v3 used 8M and its curves were still rising; a larger pool also needs more steps)")
    p.add_argument("--ckpt-every", type=int, default=2_000_000,
                   help="save ckpt/ppo_<steps>_steps.zip every this many env steps (0 = off)")
    p.add_argument("--n-envs", type=int, default=16)
    p.add_argument("--group", choices=["robust", "contrast", "contrast_force"], default="robust",
                   help="which disturbance training the gate-aware recipe sits on (see the module docstring)")
    p.add_argument("--seed", type=int, default=0, choices=[0, 1, 2],
                   help="the study's three seeds per group; seed 0 of `robust` is the saved v4 run")
    p.add_argument("--reason", required=True)
    p.add_argument("--dr-lam-max", type=float, default=DR_LAM_MAX)
    p.add_argument("--no-perturb", action="store_true")
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--n-steps", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--tag", default=None, help="run-directory tag (default racing-gate-aware-train, or "
                                               "racing-gate-aware-contrast-train for --group contrast)")
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.group == "contrast" and args.dr_lam_max != DR_LAM_MAX:
        raise SystemExit("--dr-lam-max is the matched-disturbance box; the contrast group has none (it would be ignored)")
    if len(POOL_SPEC) <= len(DEV_SEEDS):
        raise SystemExit("the `train` track pool is empty -- run gen_pool.py first (v4 exists to train on a larger pool)")

    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback

    from crazy_track.training.asymmetric import AsymmetricPolicy

    from monitored_adapter import MonitoredSB3Adapter

    tag = args.tag or {"robust": "racing-gate-aware-train", "contrast": "racing-gate-aware-contrast-train",
                       "contrast_force": "racing-gate-aware-contrast-force-train"}[args.group]
    log = RunLogger(tag=tag, reason=args.reason,
                    config={**vars(args), "floor_z": FLOOR_Z, "gate_penalty": GATE_PENALTY, "floor_penalty": FLOOR_PENALTY,
                            "recipe": "racing-gate-aware-v4-pool" + {"robust": "", "contrast": "-contrast",
                                                          "contrast_force": "-contrast-force"}[args.group],
                            "group": args.group, "episode_time": EPISODE_TIME,
                            "pool_size": len(POOL_SPEC), "pool_train_seeds": [x for r, x in POOL_SPEC if r == "train"],
                            "freq": FREQ, "window_dt": WINDOW_DT,
                            "force_xy_max": FORCE_XY_MAX, "force_z": [FORCE_Z_LO, FORCE_Z_HI]})
    print(f"Logging to {log.dir}", flush=True)
    print(f"track pool: {len(POOL_SPEC)} tracks ({len(DEV_SEEDS)} dev + {len(POOL_SPEC) - len(DEV_SEEDS)} train), "
          f"episode_time {EPISODE_TIME} s", flush=True)
    if args.group == "robust":
        raw_env = GateAwareTrackingEnv(num_envs=args.n_envs, seed=args.seed, v3=True, v5=True,
                                       perturb=not args.no_perturb, dr_lam_max=args.dr_lam_max)
    elif args.group == "contrast_force":
        raw_env = GateAwareForceContrastEnv(num_envs=args.n_envs, seed=args.seed, v3=True, v5=True,
                                            perturb=not args.no_perturb, dr_lam_max=args.dr_lam_max)
    else:
        raw_env = GateAwareContrastEnv(num_envs=args.n_envs, seed=args.seed, v3=True, v5=True,
                                       perturb=not args.no_perturb)
    env = MonitoredSB3Adapter(raw_env)
    print(f"obs dim: {env.env.single_observation_space.shape} (must be 56: the v5 layout, unchanged)", flush=True)
    model = PPO(
        AsymmetricPolicy, env, verbose=1, seed=args.seed,
        n_steps=args.n_steps, batch_size=args.batch_size, learning_rate=3e-4, gamma=args.gamma,
        tensorboard_log=str(log.dir / "tb"),
    )

    class GateStatsCallback(BaseCallback):
        """Writes the env's per-rollout `gate/*` summary to tensorboard (see `summarise_gate_stats`)."""

        def __init__(self, gate_env):
            super().__init__()
            self.gate_env, self.last = gate_env, dict(gate_env.stats)

        def _on_step(self) -> bool:
            return True

        def _on_rollout_end(self) -> None:
            cur = dict(self.gate_env.stats)
            for k, v in summarise_gate_stats(self.last, cur).items():
                self.logger.record(k, v)
            self.last = cur

    callbacks = [GateStatsCallback(env.env)]
    if args.ckpt_every > 0:
        callbacks.append(CheckpointCallback(save_freq=max(args.ckpt_every // args.n_envs, 1),
                                            save_path=str(log.dir / "ckpt"), name_prefix="ppo"))
    model.learn(total_timesteps=args.timesteps, progress_bar=False, callback=callbacks)
    model.save(log.dir / "datt_ppo_final")
    print(f"Saved model to {log.dir / 'datt_ppo_final.zip'}", flush=True)


if __name__ == "__main__":
    main()
