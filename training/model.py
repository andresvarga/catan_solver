"""Flat-vector observation encoding shared by the hierarchical model
(training/hier_model.py): every observation field is normalized and
concatenated into one vector. The original flat-action-space PPO model that
used to live here (`ActorCritic`, roadmap phase 4) was superseded by the
pointer-based hierarchical/GNN models and removed; `flatten_observation`/
`observation_dim` remain because the hierarchical model's trunk still
consumes this same flat encoding.
"""
from __future__ import annotations

import numpy as np

from env.pettingzoo_env import NUM_PLAYERS, PHASE_LIST


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
        obs["counter_trade_give"].astype(np.float32) / 19.0,
        obs["counter_trade_want"].astype(np.float32) / 19.0,
        _onehot(int(obs["counter_trade_proposer"][0]) + 1, NUM_PLAYERS + 1),
    ]
    # Optional card-counting features (env's `public_hand_features` flag):
    # keyed on presence so the same encoder serves both observation layouts.
    if "public_est_resources" in obs:
        parts.append(obs["public_est_resources"].astype(np.float32).reshape(-1) / 19.0)
        parts.append(obs["public_est_unknown"].astype(np.float32) / 40.0)
    return np.concatenate(parts)


def observation_dim(public_hand_features: bool = False) -> int:
    from env.pettingzoo_env import CatanAECEnv
    env = CatanAECEnv(randomize_board=False, seed=0, public_hand_features=public_hand_features)
    env.reset(seed=0)
    obs = env.observe(env.agent_selection)
    return flatten_observation(obs).shape[0]
