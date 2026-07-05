"""Flat-vector PPO baseline model (roadmap phase 4).

Deliberately simple: every observation field is normalized and concatenated
into one vector rather than encoding the board as a graph (§2's GNN encoder
is phase 6 work, once curriculum stages start randomizing board layouts).
On a *fixed* board -- which is what curriculum stage 1 uses -- a flat
encoding is a reasonable v0: the design doc's own comparison table flags
weak generalization across random layouts as the flat encoding's specific
weakness, not "doesn't work at all."
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from env.pettingzoo_env import MAX_ACTIONS, NUM_PLAYERS, PHASE_LIST

NEG_INF = -1e9


def _onehot(index: int, n: int) -> np.ndarray:
    v = np.zeros(n, dtype=np.float32)
    if 0 <= index < n:
        v[index] = 1.0
    return v


def flatten_observation(obs: dict) -> np.ndarray:
    parts = [
        obs["hex_terrain"].astype(np.float32) / 5.0,
        obs["hex_number"].astype(np.float32) / 12.0,
        obs["robber"].astype(np.float32),
        (obs["vertex_owner"].astype(np.float32) + 1) / NUM_PLAYERS,
        obs["vertex_type"].astype(np.float32) / 2.0,
        obs["vertex_port_generic"].astype(np.float32),
        (obs["vertex_port_resource"].astype(np.float32) + 1) / 5.0,
        (obs["edge_owner"].astype(np.float32) + 1) / NUM_PLAYERS,
        obs["own_resources"].astype(np.float32) / 19.0,
        obs["own_dev_cards"].astype(np.float32) / 25.0,
        obs["own_dev_cards_playable"].astype(np.float32) / 25.0,
        obs["public_hand_size"].astype(np.float32) / 40.0,
        obs["public_visible_vp"].astype(np.float32) / 12.0,
        obs["public_settlements"].astype(np.float32) / 5.0,
        obs["public_cities"].astype(np.float32) / 4.0,
        obs["public_roads"].astype(np.float32) / 15.0,
        obs["public_knights_played"].astype(np.float32) / 14.0,
        obs["public_dev_card_count"].astype(np.float32) / 25.0,
        _onehot(int(obs["longest_road_holder"][0]) + 1, NUM_PLAYERS + 1),
        _onehot(int(obs["largest_army_holder"][0]) + 1, NUM_PLAYERS + 1),
        _onehot(int(obs["current_player"][0]), NUM_PLAYERS),
        _onehot(int(obs["acting_player"][0]), NUM_PLAYERS),
        _onehot(int(obs["phase"][0]), len(PHASE_LIST)),
        obs["dice_roll"].astype(np.float32) / 6.0,
        obs["pending_trade_give"].astype(np.float32) / 19.0,
        obs["pending_trade_want"].astype(np.float32) / 19.0,
        _onehot(int(obs["pending_trade_proposer"][0]) + 1, NUM_PLAYERS + 1),
    ]
    return np.concatenate(parts)


def observation_dim() -> int:
    from env.pettingzoo_env import CatanAECEnv
    env = CatanAECEnv(randomize_board=False, seed=0)
    env.reset(seed=0)
    obs = env.observe(env.agent_selection)
    return flatten_observation(obs).shape[0]


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int = MAX_ACTIONS, hidden: int = 256):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.policy_head = nn.Linear(hidden, action_dim)
        self.value_head = nn.Linear(hidden, 1)

    def forward(self, obs_batch: torch.Tensor, mask_batch: torch.Tensor):
        features = self.trunk(obs_batch)
        logits = self.policy_head(features)
        logits = logits.masked_fill(mask_batch == 0, NEG_INF)
        value = self.value_head(features).squeeze(-1)
        return logits, value

    def value(self, obs_batch: torch.Tensor) -> torch.Tensor:
        """Value-only forward pass, used to bootstrap GAE at a truncated
        (not truly terminal) episode boundary without sampling an action."""
        features = self.trunk(obs_batch)
        return self.value_head(features).squeeze(-1)

    def act(self, obs_batch: torch.Tensor, mask_batch: torch.Tensor, deterministic: bool = False):
        logits, value = self.forward(obs_batch, mask_batch)
        dist = Categorical(logits=logits)
        action = torch.argmax(logits, dim=-1) if deterministic else dist.sample()
        logprob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, logprob, entropy, value

    def evaluate_actions(self, obs_batch: torch.Tensor, mask_batch: torch.Tensor,
                          actions: torch.Tensor):
        logits, value = self.forward(obs_batch, mask_batch)
        dist = Categorical(logits=logits)
        logprob = dist.log_prob(actions)
        entropy = dist.entropy()
        return logprob, entropy, value
