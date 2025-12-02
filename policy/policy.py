import numpy as np
from typing import Callable, Dict, List, Optional, Tuple, Type, Union
import gymnasium as gym
from gymnasium import spaces
import torch
import torch as th
import torch.nn as nn
import os
from stable_baselines3.common.policies import ActorCriticPolicy

from model.model import TemporalHGAT


class HGATNetwork(nn.Module):
    """
    Policy/value network wrapper around TemporalHGAT.
    Expects flattened observations: [ind, pos, neg, ts_flat] with ts_flat = num_stocks * lookback * input_dim.
    """
    def __init__(
            self,
            feature_dim: int,
            last_layer_dim_pi: int = 64,  # Acts as num_stocks
            last_layer_dim_vf: int = 64,
            n_head=4,
            hidden_dim=128,
            no_ind=False,
            no_neg=False,
            lookback=30,
    ):
        super(HGATNetwork, self).__init__()
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf
        self.num_stocks = last_layer_dim_pi
        self.lookback = lookback

        adj_size = 3 * self.num_stocks * self.num_stocks
        remaining = feature_dim - adj_size
        if remaining <= 0:
            raise ValueError(
                f"Observation too small for TemporalHGAT: feature_dim={feature_dim}, adj_size={adj_size}"
            )
        # Reserve prev_weights length
        remaining_minus_prev = remaining - self.num_stocks
        if remaining_minus_prev <= 0:
            raise ValueError(
                f"Observation too small after accounting for prev_weights: remaining={remaining}, num_stocks={self.num_stocks}"
            )
        if remaining_minus_prev % (self.num_stocks * self.lookback) != 0:
            raise ValueError(
                f"Cannot derive per-stock feature dim: remaining_minus_prev={remaining_minus_prev} not divisible by "
                f"num_stocks*lookback={self.num_stocks * self.lookback}"
            )
        self.n_features = remaining_minus_prev // (self.num_stocks * self.lookback)
        if os.environ.get("DEBUG_MODEL_SHAPES"):
            expected_len = 3 * self.num_stocks * self.num_stocks + self.num_stocks * self.lookback * self.n_features + self.num_stocks
            print(
                f"[ModelDebug] feature_dim={feature_dim} num_stocks={self.num_stocks} "
                f"lookback={self.lookback} n_features={self.n_features} expected_obs_len={expected_len}"
            )
        if self.n_features <= 0:
            raise ValueError(f"Invalid derived n_features={self.n_features}")

        self.policy_net = TemporalHGAT(
            num_stocks=self.num_stocks,
            input_dim=self.n_features,
            lookback=self.lookback,
            num_heads=n_head,
            hidden_dim=hidden_dim,
            no_ind=no_ind,
            no_neg=no_neg,
        )

        self.value_net = TemporalHGAT(
            num_stocks=self.num_stocks,
            input_dim=self.n_features,
            lookback=self.lookback,
            num_heads=n_head,
            hidden_dim=hidden_dim,
            no_ind=no_ind,
            no_neg=no_neg,
        )

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.policy_net(features), self.value_net(features)

    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        return self.policy_net(features)

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        return self.value_net(features)


class HGATActorCriticPolicy(ActorCriticPolicy):
    def __init__(self,
                 observation_space: spaces.Space,
                 action_space: spaces.Space,
                 lr_schedule: Callable[[float], float],
                 net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
                 activation_fn: Type[nn.Module] = nn.Tanh,
                 *args,
                 **kwargs,
                 ):
        self.last_layer_dim_pi = kwargs.pop('last_layer_dim_pi', 64)
        self.last_layer_dim_vf = kwargs.pop('last_layer_dim_vf', 64)
        self.n_head = kwargs.pop('n_head', 4)
        self.hidden_dim = kwargs.pop('hidden_dim', 128)
        self.no_ind = kwargs.pop('no_ind', False)
        self.no_neg = kwargs.pop('no_neg', False)
        self.lookback = kwargs.pop('lookback', 30)

        super(HGATActorCriticPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            *args,
            **kwargs,
        )
        self.ortho_init = False

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = HGATNetwork(
            last_layer_dim_pi=self.last_layer_dim_pi,
            last_layer_dim_vf=self.last_layer_dim_vf,
            feature_dim=self.observation_space.shape[0],
            n_head=self.n_head,
            hidden_dim=self.hidden_dim,
            no_ind=self.no_ind,
            no_neg=self.no_neg,
            lookback=self.lookback,
        )

    def forward(self, obs: th.Tensor, deterministic: bool = False) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        actions, values, log_prob = super().forward(obs, deterministic)
        return actions, values, log_prob

    def _predict(self, observation, deterministic: bool = False) -> th.Tensor:
        actions, values, log_prob = self.forward(observation, deterministic)
        return actions
