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


# --------------------------------------------------------------------------
# Fused encoder (simulation performance audit): `encode_flat(state, pid, phf)` is
# bit-identical to `flatten_observation(build_observation(state, pid, ...))` but
# skips the intermediate observation dict and ~40 small array allocations. Every
# feature is written as its raw float32 value into one buffer that is divided once,
# elementwise, by a constant float32 scale vector -- the same IEEE float32 ops the
# per-part `x.astype(float32) / scale` performs, so the result is identical.
# --------------------------------------------------------------------------
_LAYOUT_CACHE: dict[bool, tuple] = {}


def _layout(phf: bool):
    if phf in _LAYOUT_CACHE:
        return _LAYOUT_CACHE[phf]
    from env.pettingzoo_env import NUM_EDGES, NUM_HEXES, NUM_VERTICES
    parts = [  # (name, length, scale) in flatten_observation order
        ("hex_terrain", NUM_HEXES, 5.0), ("hex_number", NUM_HEXES, 12.0), ("robber", NUM_HEXES, 1.0),
        ("vertex_owner", NUM_VERTICES, 4.0), ("vertex_type", NUM_VERTICES, 2.0),
        ("port_generic", NUM_VERTICES, 1.0), ("port_resource", NUM_VERTICES, 5.0),
        ("edge_owner", NUM_EDGES, 4.0),
        ("own_resources", 5, 19.0), ("own_dev", 5, 25.0), ("own_playable", 5, 25.0),
        ("hand", 4, 40.0), ("visible_vp", 4, 12.0), ("settlements", 4, 5.0), ("cities", 4, 4.0),
        ("roads", 4, 15.0), ("knights", 4, 14.0), ("dev_count", 4, 25.0),
        ("exp_vp", 4, 5.0), ("exp_kn", 4, 14.0), ("dev_age", 4, 1.0),
        ("offer_give", 20, 3.0), ("offer_want", 20, 3.0), ("offer_age", 4, 1.0),
        ("lr_holder", 5, 1.0), ("la_holder", 5, 1.0), ("current", 4, 1.0), ("acting", 4, 1.0),
        ("phase", len(PHASE_LIST), 1.0), ("dice", 2, 6.0),
        ("pending_give", 5, 19.0), ("pending_want", 5, 19.0), ("pending_proposer", 5, 1.0),
        ("counter_give", 5, 19.0), ("counter_want", 5, 19.0), ("counter_proposer", 5, 1.0),
        ("pending_targets", 4, 1.0), ("trades_proposed", 1, float(MAX_TRADE_PROPOSALS_PER_TURN)),
        ("bank", 5, 19.0), ("deck", 1, 25.0),
    ]
    if phf:
        parts += [("est", 20, 19.0), ("est_unknown", 4, 40.0)]
    off, pos = {}, 0
    for name, n, _ in parts:
        off[name] = pos
        pos += n
    scale = np.concatenate([np.full(n, s, dtype=np.float32) for _, n, s in parts])
    _LAYOUT_CACHE[phf] = (off, pos, scale)
    return _LAYOUT_CACHE[phf]


def _static_raw(board, phf: bool) -> np.ndarray:
    """Board-static raw values (terrain, numbers, ports), cached on the Board."""
    key = "_flat_raw_static_phf" if phf else "_flat_raw_static"
    cached = getattr(board, key, None)
    if cached is None:
        from env.pettingzoo_env import HEXTYPE_INDEX, NUM_HEXES, NUM_VERTICES, RESOURCE_INDEX
        off, size, _ = _layout(phf)
        cached = np.zeros(size, dtype=np.float32)
        for hx in board.hexes.values():
            cached[off["hex_terrain"] + hx.id] = HEXTYPE_INDEX[hx.terrain]
            cached[off["hex_number"] + hx.id] = hx.number or 0
        cached[off["port_resource"]: off["port_resource"] + NUM_VERTICES] = 0.0  # (-1 + 1)
        for vid, v in board.vertices.items():
            if v.port_generic:
                cached[off["port_generic"] + vid] = 1.0
            if v.port is not None:
                cached[off["port_resource"] + vid] = RESOURCE_INDEX[v.port] + 1
        setattr(board, key, cached)
    return cached


def encode_flat(state, pid: int, public_hand_features: bool = False) -> np.ndarray:
    from env.engine import acting_player
    from env.pettingzoo_env import PHASE_INDEX, RESOURCE_INDEX, RESOURCE_LIST, DEV_CARD_LIST, \
        public_hand_estimate_arrays
    from env.public_beliefs import expected_dev_cards_all, last_offer, turns_since_dev_purchase
    from env.state import DevCard
    off, size, scale = _layout(public_hand_features)
    raw = _static_raw(state.board, public_hand_features).copy()
    me_ = pid
    rel = lambda seat: (seat - me_) % NUM_PLAYERS  # noqa: E731

    raw[off["robber"] + state.board.robber_hex] = 1.0
    vo, vt, eo = off["vertex_owner"], off["vertex_type"], off["edge_owner"]
    for other, p in state.players.items():
        r1 = rel(other) + 1
        for vid in p.settlements:
            raw[vo + vid] = r1
            raw[vt + vid] = 1
        for vid in p.cities:
            raw[vo + vid] = r1
            raw[vt + vid] = 2
        for eid in p.roads:
            raw[eo + eid] = r1

    me = state.players[pid]
    for i, r in enumerate(RESOURCE_LIST):
        raw[off["own_resources"] + i] = me.resources[r]
    for i, c in enumerate(DEV_CARD_LIST):
        raw[off["own_dev"] + i] = me.dev_cards[c]
        raw[off["own_playable"] + i] = me.dev_cards[c] - me.dev_cards_bought_this_turn[c]

    beliefs = expected_dev_cards_all(state, pid)
    f32 = np.float32
    for row, seat in enumerate((pid + k) % NUM_PLAYERS for k in range(NUM_PLAYERS)):
        p = state.players[seat]
        raw[off["hand"] + row] = p.hand_size()
        raw[off["visible_vp"] + row] = p.visible_vp()
        raw[off["settlements"] + row] = len(p.settlements)
        raw[off["cities"] + row] = len(p.cities)
        raw[off["roads"] + row] = len(p.roads)
        raw[off["knights"] + row] = p.knights_played
        raw[off["dev_count"] + row] = p.total_dev_cards()
        exp = beliefs[seat]
        raw[off["exp_vp"] + row] = f32(exp[DevCard.VICTORY_POINT])
        raw[off["exp_kn"] + row] = f32(exp[DevCard.KNIGHT])
        raw[off["dev_age"] + row] = f32(turns_since_dev_purchase(state, seat))
        g, w, age = last_offer(state, seat)
        for r, k in g.items():
            raw[off["offer_give"] + row * 5 + RESOURCE_INDEX[r]] = k
        for r, k in w.items():
            raw[off["offer_want"] + row * 5 + RESOURCE_INDEX[r]] = k
        raw[off["offer_age"] + row] = f32(age)

    lr = state.longest_road_holder
    raw[off["lr_holder"] + (0 if lr is None else rel(lr) + 1)] = 1.0
    la = state.largest_army_holder
    raw[off["la_holder"] + (0 if la is None else rel(la) + 1)] = 1.0
    raw[off["current"] + rel(state.current_player)] = 1.0
    raw[off["acting"] + rel(acting_player(state))] = 1.0
    raw[off["phase"] + PHASE_INDEX[state.phase]] = 1.0
    if state.dice_roll is not None:
        raw[off["dice"]], raw[off["dice"] + 1] = state.dice_roll
    pt = state.pending_trade
    if pt is not None:
        for r, k in pt.give.items():
            raw[off["pending_give"] + RESOURCE_INDEX[r]] = k
        for r, k in pt.want.items():
            raw[off["pending_want"] + RESOURCE_INDEX[r]] = k
        raw[off["pending_proposer"] + rel(pt.proposer) + 1] = 1.0
        for t in pt.targets:
            raw[off["pending_targets"] + rel(t)] = 1.0
    else:
        raw[off["pending_proposer"]] = 1.0
    ct = state.trade_counter_context
    if ct is not None:
        for r, k in ct.give.items():
            raw[off["counter_give"] + RESOURCE_INDEX[r]] = k
        for r, k in ct.want.items():
            raw[off["counter_want"] + RESOURCE_INDEX[r]] = k
        raw[off["counter_proposer"] + rel(ct.proposer) + 1] = 1.0
    else:
        raw[off["counter_proposer"]] = 1.0
    raw[off["trades_proposed"]] = state.trades_proposed_this_turn
    for i, r in enumerate(RESOURCE_LIST):
        raw[off["bank"] + i] = state.bank[r]
    raw[off["deck"]] = len(state.dev_card_deck)
    if public_hand_features:
        est, unknown = public_hand_estimate_arrays(state)
        order = [(pid + k) % NUM_PLAYERS for k in range(NUM_PLAYERS)]
        raw[off["est"]: off["est"] + 20] = est[order].reshape(-1)
        raw[off["est_unknown"]: off["est_unknown"] + 4] = unknown[order]
    return raw / scale
