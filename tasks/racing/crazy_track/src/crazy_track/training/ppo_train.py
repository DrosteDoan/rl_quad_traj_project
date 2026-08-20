"""PPO training for the DATT-style tracking policy.

Run (WSL venv):
    python -m crazy_track.training.ppo_train --timesteps 2000000 --reason "..."
"""

from __future__ import annotations

import argparse

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnv

from crazy_track.envs.datt_env import DATTTrackingEnv
from crazy_track.eval.runlog import RunLogger


class SB3Adapter(VecEnv):
    """Adapt DATTTrackingEnv (gymnasium-vector-style) to the SB3 VecEnv API."""

    def __init__(self, env: DATTTrackingEnv):
        self.env = env
        super().__init__(env.num_envs, env.single_observation_space, env.single_action_space)
        self._actions = None

    def reset(self):
        obs, _ = self.env.reset()
        return obs

    def step_async(self, actions):
        self._actions = actions

    def step_wait(self):
        obs, reward, term, trunc, info = self.env.step(self._actions)
        done = term | trunc
        infos = [{} for _ in range(self.num_envs)]
        if done.any():
            terminal_obs = info["terminal_obs"]
            for i in np.flatnonzero(done):
                infos[i]["terminal_observation"] = terminal_obs[i]
                infos[i]["TimeLimit.truncated"] = bool(trunc[i] and not term[i])
        return obs, reward, done, infos

    def close(self):
        pass

    def get_attr(self, attr_name, indices=None):
        return [getattr(self.env, attr_name)] * self.num_envs

    def set_attr(self, attr_name, value, indices=None):
        setattr(self.env, attr_name, value)

    def env_method(self, method_name, *args, indices=None, **kwargs):
        return [getattr(self.env, method_name)(*args, **kwargs)] * self.num_envs

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * self.num_envs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=2_000_000)
    parser.add_argument("--n-envs", type=int, default=16)
    parser.add_argument("--reason", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--v2", action="store_true",
                        help="v2: no L1 channel, no force perturbation (env v3=False); "
                             "randomized-difficulty refs only")
    parser.add_argument("--noisy-sensor", action="store_true",
                        help="v4: train on Lighthouse-noisy observations")
    parser.add_argument("--v5", action="store_true",
                        help="v5: asymmetric actor-critic + sensor-noise domain randomization")
    parser.add_argument("--v6", action="store_true",
                        help="v6a: v5 + frame-stacked actor obs (noise-level observability)")
    parser.add_argument("--v7", action="store_true",
                        help="v7: v5 env + recurrent (GRU-64) actor, feedforward "
                             "privileged critic (RecurrentPPO)")
    parser.add_argument("--ctbr", action="store_true",
                        help="acro: CTBR body-rate action space + aggressive ref distribution")
    parser.add_argument("--acro2", action="store_true",
                        help="acro phase 2: flip primitives (attitude-ref obs + reward)")
    parser.add_argument("--acro3", action="store_true",
                        help="acro phase 3: flips on the feasible ballistic reference "
                             "(consistent pos+att refs, balanced reward)")
    parser.add_argument("--acro4", action="store_true",
                        help="acro phase 4: acro3 + maneuver-conditioned obs "
                             "(descriptor block) + rotation-completion bonus")
    parser.add_argument("--resume-from", default=None,
                        help="path to a datt_ppo_final.zip to continue training; "
                             "--timesteps is then ADDITIONAL steps on top of the "
                             "loaded count (SB3 reset_num_timesteps=False adds the "
                             "current count to the target)")
    args = parser.parse_args()
    if args.v2 and (args.v5 or args.v6 or args.v7 or args.noisy_sensor or args.ctbr
                     or args.acro2 or args.acro3 or args.acro4):
        raise SystemExit("--v2 cannot combine with other variant flags")
    if args.v7 and (args.v2 or args.v5 or args.v6 or args.noisy_sensor or args.ctbr
                    or args.acro2 or args.acro3 or args.acro4):
        raise SystemExit("--v7 cannot combine with other variant flags")
    if args.v7 and args.resume_from:
        raise SystemExit("--resume-from is not supported for --v7 (RecurrentPPO)")

    log = RunLogger(tag="datt-train", reason=args.reason, config=vars(args))
    print(f"Logging to {log.dir}", flush=True)

    # v7 shares v5's env exactly; only the policy architecture differs.
    env = SB3Adapter(DATTTrackingEnv(num_envs=args.n_envs, seed=args.seed, v3=not args.v2,
                                     noisy_sensor=args.noisy_sensor,
                                     v5=args.v5 or args.v7,
                                     v6=args.v6, ctbr=args.ctbr, acro2=args.acro2,
                                     acro3=args.acro3, acro4=args.acro4))
    print(f"obs dim: {env.env.single_observation_space.shape}", flush=True)
    if args.v7:
        from sb3_contrib import RecurrentPPO

        from crazy_track.training.recurrent import AsymmetricGRUPolicy

        model = RecurrentPPO(
            AsymmetricGRUPolicy, env, verbose=1, seed=args.seed,
            n_steps=256, batch_size=1024, learning_rate=3e-4, gamma=0.98,
            tensorboard_log=str(log.dir / "tb"),
        )
        model.learn(total_timesteps=args.timesteps, progress_bar=False)
        model.save(log.dir / "datt_ppo_final")
        print(f"Saved model to {log.dir / 'datt_ppo_final.zip'}", flush=True)
        return
    if args.v5 or args.v6:
        from crazy_track.training.asymmetric import AsymmetricPolicy

        policy, policy_kwargs = AsymmetricPolicy, {}
    else:
        policy, policy_kwargs = "MlpPolicy", dict(net_arch=[64, 64])
    if args.resume_from:
        model = PPO.load(args.resume_from, env=env, device="cpu")
        model.tensorboard_log = str(log.dir / "tb")
        model.learn(total_timesteps=args.timesteps, progress_bar=False,
                    reset_num_timesteps=False)
    else:
        model = PPO(
            policy, env, verbose=1, seed=args.seed,
            n_steps=256, batch_size=1024, learning_rate=3e-4, gamma=0.98,
            policy_kwargs=policy_kwargs,
            tensorboard_log=str(log.dir / "tb"),
        )
        model.learn(total_timesteps=args.timesteps, progress_bar=False)
    model.save(log.dir / "datt_ppo_final")
    print(f"Saved model to {log.dir / 'datt_ppo_final.zip'}", flush=True)


if __name__ == "__main__":
    main()
