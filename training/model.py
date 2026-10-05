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
from env.state import MAX_TRADE_PROPOSALS_PER_TURN


def _onehot(index: int, n: int) -> np.ndarray:
    v = np.zeros(n, dtype=np.float32)
    if 0 <= index < n:
        v[index] = 1.0
    return v


def _rel(seat: int, me: int) -> int:
    """Absolute seat -> observer-relative seat (0 = me, 1..3 = turn order
    after me); -1 (nobody) stays -1."""
    return -1 if seat < 0 else (seat - me) % NUM_PLAYERS


def flatten_observation(obs: dict) -> np.ndarray:
    """Every field normalized and concatenated, with all seat references made
    relative to the observer (audit F-26): ownership scalars, per-player rows
    (rotated so row 0 is the observer), holders, current/acting player and
    trade parties. The same policy then plays every seat with the same
    weights instead of learning seat-specific decodings."""
    me = int(obs["observer"][0])
    order = [(me + k) % NUM_PLAYERS for k in range(NUM_PLAYERS)]  # rows: me, +1, +2, +3

    def owners(arr):  # -1 = nobody -> 0; otherwise (relative seat + 1) / 4
        a = arr.astype(np.int64)
        return np.where(a < 0, 0.0, ((a - me) % NUM_PLAYERS + 1) / NUM_PLAYERS).astype(np.float32)

    def per_player(key, scale):
        return (obs[key][order].astype(np.float32) / scale).reshape(-1)

    parts = [
        obs["hex_terrain"].astype(np.float32) / 5.0,
        obs["hex_number"].astype(np.float32) / 12.0,
        obs["robber"].astype(np.float32),
        owners(obs["vertex_owner"]),
        obs["vertex_type"].astype(np.float32) / 2.0,
        obs["vertex_port_generic"].astype(np.float32),
        (obs["vertex_port_resource"].astype(np.float32) + 1) / 5.0,
        owners(obs["edge_owner"]),
        obs["own_resources"].astype(np.float32) / 19.0,
        obs["own_dev_cards"].astype(np.float32) / 25.0,
        obs["own_dev_cards_playable"].astype(np.float32) / 25.0,
        per_player("public_hand_size", 40.0),
        per_player("public_visible_vp", 12.0),
        per_player("public_settlements", 5.0),
        per_player("public_cities", 4.0),
        per_player("public_roads", 15.0),
        per_player("public_knights_played", 14.0),
        per_player("public_dev_card_count", 25.0),
        per_player("public_expected_vp", 5.0),
        per_player("public_expected_knights", 14.0),
        per_player("public_dev_purchase_age", 1.0),
        per_player("public_last_offer_give", 3.0),
        per_player("public_last_offer_want", 3.0),
        per_player("public_last_offer_age", 1.0),
        _onehot(_rel(int(obs["longest_road_holder"][0]), me) + 1, NUM_PLAYERS + 1),
        _onehot(_rel(int(obs["largest_army_holder"][0]), me) + 1, NUM_PLAYERS + 1),
        _onehot(_rel(int(obs["current_player"][0]), me), NUM_PLAYERS),
        _onehot(_rel(int(obs["acting_player"][0]), me), NUM_PLAYERS),
        _onehot(int(obs["phase"][0]), len(PHASE_LIST)),
        obs["dice_roll"].astype(np.float32) / 6.0,
        obs["pending_trade_give"].astype(np.float32) / 19.0,
        obs["pending_trade_want"].astype(np.float32) / 19.0,
        _onehot(_rel(int(obs["pending_trade_proposer"][0]), me) + 1, NUM_PLAYERS + 1),
        obs["counter_trade_give"].astype(np.float32) / 19.0,
        obs["counter_trade_want"].astype(np.float32) / 19.0,
        _onehot(_rel(int(obs["counter_trade_proposer"][0]), me) + 1, NUM_PLAYERS + 1),
        obs["pending_trade_targets"][order].astype(np.float32),
        obs["trades_proposed_this_turn"].astype(np.float32) / MAX_TRADE_PROPOSALS_PER_TURN,
        obs["bank"].astype(np.float32) / 19.0,
        obs["dev_deck_size"].astype(np.float32) / 25.0,
    ]
    # Optional card-counting features (env's `public_hand_features` flag):
    # keyed on presence so the same encoder serves both observation layouts.
    if "public_est_resources" in obs:
        parts.append(obs["public_est_resources"][order].astype(np.float32).reshape(-1) / 19.0)
        parts.append(obs["public_est_unknown"][order].astype(np.float32) / 40.0)
    return np.concatenate(parts)


def observation_dim(public_hand_features: bool = False) -> int:
    from env.pettingzoo_env import CatanAECEnv
    env = CatanAECEnv(randomize_board=False, seed=0, public_hand_features=public_hand_features)
    env.reset(seed=0)
    obs = env.observe(env.agent_selection)
    return flatten_observation(obs).shape[0]
