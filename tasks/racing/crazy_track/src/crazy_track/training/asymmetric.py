"""Asymmetric actor-critic policy for SB3 PPO (learning-to-fly style).

The env observation is concat(actor_obs, privileged_obs). The actor MLP only
reads the first ACTOR_DIM entries (the noisy onboard view); the value MLP
reads everything, including privileged true-state/disturbance information
that exists only in simulation. At deployment the privileged tail is zero-
padded and ignored by the actor.
"""

from __future__ import annotations

import torch
from stable_baselines3.common.policies import ActorCriticPolicy
from torch import nn

from crazy_track.envs.datt_env import PRIV_DIM


def _mlp(in_dim: int, hidden: list[int]) -> nn.Sequential:
    layers, d = [], in_dim
    for h in hidden:
        layers += [nn.Linear(d, h), nn.Tanh()]
        d = h
    return nn.Sequential(*layers)


class AsymmetricExtractor(nn.Module):
    def __init__(self, full_dim: int, hidden: list[int]):
        super().__init__()
        # Everything except the privileged tail is the deployable actor view
        # (43 for v5; 43*STACK for v6a frame stacking).
        self.actor_dim = full_dim - PRIV_DIM
        self.policy_net = _mlp(self.actor_dim, hidden)
        self.value_net = _mlp(full_dim, hidden)
        self.latent_dim_pi = hidden[-1]
        self.latent_dim_vf = hidden[-1]

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        return self.policy_net(features[..., :self.actor_dim])

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        return self.value_net(features)


class AsymmetricPolicy(ActorCriticPolicy):
    def _build_mlp_extractor(self) -> None:
        hidden = [64, 64]
        self.mlp_extractor = AsymmetricExtractor(self.features_dim, hidden)
