"""Advanced track: PPO written out by hand (CleanRL style).

The SB3 script hides PPO inside ``model.learn()``. This script shows what is
actually happening, so an advanced student can *see* the algorithm. It is a
trimmed, single-file PPO based on CleanRL (https://docs.cleanrl.dev/) and the
same approach used in the lsy_drone_racing project.

Because Crazyflow lives on the GPU, this script keeps the drones (JAX) and the
neural network (PyTorch) both on the GPU and never copies data to the CPU. That
is why it can train on hundreds or thousands of drones at once.

Run it::

    python -m k12_hover.train_cleanrl --jax-device gpu --num-envs 512

Read it top to bottom alongside ``lessons/06_advanced_cleanrl.md``. The five
big ideas to look for:

  1. Agent     - the actor (chooses actions) + critic (judges states) networks.
  2. Rollout   - run the current policy and record what happened.
  3. GAE       - estimate how much better each action was than expected.
  4. PPO loss  - nudge the policy toward good actions, but not too far at once.
  5. Update    - repeat for several epochs over the collected data.
"""

from __future__ import annotations

import argparse
import os
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from torch.distributions.normal import Normal  # noqa: E402

from k12_hover import make_torch_env  # noqa: E402


# ----------------------------------------------------------------------- network
def layer_init(layer, std=np.sqrt(2), bias=0.0):
    """A standard, stable way to initialise a linear layer."""
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias)
    return layer


class Agent(nn.Module):
    """The drone's brain: a critic that judges, and an actor that acts."""

    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        # Critic: looks at the observation, guesses "how good is this situation?"
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        # Actor: looks at the observation, decides the *average* action to take.
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, act_dim), std=0.01), nn.Tanh(),
        )
        # How much randomness to add while exploring (learned, shrinks over time).
        self.actor_logstd = nn.Parameter(torch.zeros(1, act_dim))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None, deterministic=False):
        mean = self.actor_mean(x)
        std = torch.exp(self.actor_logstd.expand_as(mean))
        dist = Normal(mean, std)
        if action is None:
            action = mean if deterministic else dist.sample()
        return action, dist.log_prob(action).sum(1), dist.entropy().sum(1), self.critic(x)


def main():
    p = argparse.ArgumentParser(description="Hand-written PPO for hovering (CleanRL style).")
    p.add_argument("--timesteps", type=int, default=1_000_000)
    p.add_argument("--num-envs", type=int, default=512, help="Parallel drones.")
    p.add_argument("--num-steps", type=int, default=16, help="Steps per drone per rollout.")
    p.add_argument("--jax-device", default="cpu", choices=["cpu", "gpu"])
    p.add_argument("--torch-device", default="auto")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99, help="How much the future matters.")
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.2, help="PPO trust-region size.")
    p.add_argument("--update-epochs", type=int, default=4)
    p.add_argument("--num-minibatches", type=int, default=4)
    p.add_argument("--ent-coef", type=float, default=0.0)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save-name", default="hover_cleanrl")
    args = p.parse_args()

    # Reproducibility.
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.torch_device == "auto":
        args.torch_device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(args.torch_device)

    batch_size = args.num_envs * args.num_steps
    minibatch_size = batch_size // args.num_minibatches
    num_iterations = args.timesteps // batch_size

    # The drones (JAX physics) feeding torch tensors straight onto the GPU.
    envs = make_torch_env(
        num_envs=args.num_envs, device=args.jax_device, torch_device=args.torch_device
    )
    obs_dim = int(np.prod(envs.single_observation_space.shape))
    act_dim = int(np.prod(envs.single_action_space.shape))

    agent = Agent(obs_dim, act_dim).to(device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=args.lr, eps=1e-5)

    # Storage for one rollout.
    obs = torch.zeros((args.num_steps, args.num_envs, obs_dim), device=device)
    actions = torch.zeros((args.num_steps, args.num_envs, act_dim), device=device)
    logprobs = torch.zeros((args.num_steps, args.num_envs), device=device)
    rewards = torch.zeros((args.num_steps, args.num_envs), device=device)
    dones = torch.zeros((args.num_steps, args.num_envs), device=device)
    values = torch.zeros((args.num_steps, args.num_envs), device=device)

    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = next_obs.to(device).float()
    next_done = torch.zeros(args.num_envs, device=device)

    # Track average episode reward (so we can print progress like SB3 does).
    ep_return = torch.zeros(args.num_envs, device=device)
    recent_returns: list[float] = []

    global_step = 0
    start = time.time()
    for iteration in range(1, num_iterations + 1):
        # --- 2) ROLLOUT: run the current policy and remember everything ---
        for step in range(args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
            actions[step], logprobs[step], values[step] = action, logprob, value.flatten()

            next_obs, reward, terminated, truncated, _ = envs.step(action)
            next_obs = next_obs.float()
            reward = reward.float()
            rewards[step] = reward
            ep_return += reward
            done = (terminated | truncated).float()
            # Record finished-episode returns, then zero those counters.
            for r in ep_return[done.bool()]:
                recent_returns.append(float(r))
            ep_return[done.bool()] = 0.0
            next_done = done

        # --- 3) GAE: how much better was each action than the critic expected? ---
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = (
                    delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                )
            returns = advantages + values

        # Flatten the rollout into one big batch.
        b_obs = obs.reshape(-1, obs_dim)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape(-1, act_dim)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # --- 4 & 5) PPO LOSS + UPDATE: improve the policy, several passes ---
        inds = np.arange(batch_size)
        for _ in range(args.update_epochs):
            np.random.shuffle(inds)
            for start_i in range(0, batch_size, minibatch_size):
                mb = inds[start_i:start_i + minibatch_size]
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb], b_actions[mb]
                )
                ratio = (newlogprob - b_logprobs[mb]).exp()  # new vs old prob
                adv = b_advantages[mb]
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)  # normalise

                # Clipped policy loss: take the worse (more pessimistic) of the two,
                # which stops the policy from changing too much in one update.
                pg1 = -adv * ratio
                pg2 = -adv * torch.clamp(ratio, 1 - args.clip, 1 + args.clip)
                pg_loss = torch.max(pg1, pg2).mean()

                # Value loss: critic should predict the returns well.
                v_loss = 0.5 * ((newvalue.view(-1) - b_returns[mb]) ** 2).mean()
                ent_loss = entropy.mean()

                loss = pg_loss - args.ent_coef * ent_loss + args.vf_coef * v_loss
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
                optimizer.step()

        # Progress report.
        if recent_returns:
            mean_ret = np.mean(recent_returns[-200:])
        else:
            mean_ret = float("nan")
        sps = int(global_step / (time.time() - start))
        print(f"iter {iteration:4d}/{num_iterations} | steps {global_step:>9,} | "
              f"ep_rew_mean {mean_ret:7.2f} | {sps:>6,} steps/s")

    # Save just the network weights.
    from pathlib import Path

    models_dir = Path(__file__).resolve().parent.parent / "models"
    models_dir.mkdir(exist_ok=True)
    save_path = models_dir / f"{args.save_name}.pt"
    torch.save(agent.state_dict(), save_path)
    print(f"\nSaved trained network to {save_path}")
    envs.close()


if __name__ == "__main__":
    main()
