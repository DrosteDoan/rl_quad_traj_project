"""v7: recurrent (GRU) actor with an asymmetric feedforward privileged critic.

Same env recipe as v5 (56-dim obs = 43-dim deployable actor frame + 13-dim
privileged tail, sensor-noise domain randomization, force perturbation), but
the actor MLP is replaced by a single GRU layer over the 43-dim actor frame.
The critic stays FEEDFORWARD on the full 56 dims, exactly as in v5.

This is the pre-registered test of "domain randomization averages, recurrence
adapts": v5's memoryless actor can only learn the noise/disturbance-averaged
policy, while a recurrent actor can in principle infer the current noise level
and constant force perturbation from the observation history.

Implementation notes (sb3-contrib 2.9.0):

* ``RecurrentActorCriticPolicy`` is instantiated with ``shared_lstm=False`` and
  ``enable_critic_lstm=False``. In that configuration the base class builds
  ``self.critic = nn.Linear(features_dim, lstm_hidden_size)`` and routes the
  critic as ``features -> critic -> mlp_extractor.forward_critic -> value_net``,
  i.e. a purely feedforward critic that sees the *full* 56-dim observation.
  That is precisely the privileged critic we want, so the critic path is left
  untouched.
* ``self.lstm_actor`` is swapped for an ``nn.GRU(43, 64)`` after construction
  and the optimizer is rebuilt so it owns the new parameters.
* ``_process_sequence`` is overridden. Two deviations from the base version:
  (a) features are sliced to the first ``lstm.input_size`` (= 43) columns
  *before* the batch->sequence reshape, so the privileged tail never reaches
  the actor; (b) the GRU carries a single hidden tensor instead of the LSTM's
  ``(h, c)`` pair. The ``RNNStates`` plumbing (rollout buffer, ``predict``)
  assumes 2-tuples throughout, so a 2-tuple is kept: slot 0 is the real GRU
  hidden state, slot 1 is passed through unchanged (it stays all-zeros and is
  never read). The episode-start masking loop reproduces the base semantics
  exactly: the incoming hidden state is zeroed for every sequence whose
  ``episode_start`` is 1 at that timestep.
"""

from __future__ import annotations

from typing import Any

import torch as th
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy
from torch import nn

from crazy_track.envs.datt_env import PRIV_DIM

GRU_HIDDEN = 64


class AsymmetricGRUPolicy(RecurrentActorCriticPolicy):
    """GRU actor over the deployable obs view + feedforward privileged critic."""

    def __init__(self, observation_space, action_space, lr_schedule, **kwargs: Any):
        kwargs["lstm_hidden_size"] = GRU_HIDDEN
        kwargs["n_lstm_layers"] = 1
        kwargs["shared_lstm"] = False
        kwargs["enable_critic_lstm"] = False  # critic = feedforward on full obs
        kwargs.setdefault("net_arch", [64, 64])
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)

        # The actor must only ever see the deployable part of the observation.
        self.actor_dim = self.features_dim - PRIV_DIM
        self.lstm_actor = nn.GRU(self.actor_dim, GRU_HIDDEN, num_layers=1)
        # self.critic (nn.Linear(features_dim, GRU_HIDDEN)) is kept as built by
        # the base class: it reads all features_dim dims, privileged tail included.
        self.optimizer = self.optimizer_class(  # rebuild: parameter set changed
            self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
        )

    @staticmethod
    def _process_sequence(
        features: th.Tensor,
        lstm_states: tuple[th.Tensor, th.Tensor],
        episode_starts: th.Tensor,
        lstm: nn.GRU,
    ) -> tuple[th.Tensor, tuple[th.Tensor, th.Tensor]]:
        """Forward pass through the actor GRU (base version, adapted to GRU).

        ``lstm_states`` is the usual ``(h, c)`` pair; only ``h`` is used and
        updated, ``c`` is passed through so the RNNStates plumbing is unchanged.
        """
        # (sequence length, batch size, features dim); batch size = n_envs during
        # data collection, n_seq during the gradient update.
        n_seq = lstm_states[0].shape[1]
        # Drop the privileged tail BEFORE the reshape (the reshape uses input_size).
        features = features[..., : lstm.input_size]
        features_sequence = features.reshape((n_seq, -1, lstm.input_size)).swapaxes(0, 1)
        episode_starts = episode_starts.reshape((n_seq, -1)).swapaxes(0, 1)
        hidden_state = lstm_states[0]

        # No reset inside the sequence -> one cuDNN call instead of a python loop.
        if th.all(episode_starts == 0.0):
            gru_output, hidden_state = lstm(features_sequence, hidden_state)
            gru_output = th.flatten(gru_output.transpose(0, 1), start_dim=0, end_dim=1)
            return gru_output, (hidden_state, lstm_states[1])

        gru_output = []
        for feat, episode_start in zip(features_sequence, episode_starts, strict=True):
            hidden, hidden_state = lstm(
                feat.unsqueeze(dim=0),
                # Reset the state at the beginning of a new episode
                (1.0 - episode_start).view(1, n_seq, 1) * hidden_state,
            )
            gru_output += [hidden]
        # (sequence length, n_seq, gru_out_dim) -> (batch_size, gru_out_dim)
        gru_output = th.flatten(th.cat(gru_output).transpose(0, 1), start_dim=0, end_dim=1)
        return gru_output, (hidden_state, lstm_states[1])
