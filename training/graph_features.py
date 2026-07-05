"""Per-node graph observation for the GNN model (roadmap phase 6).

Unlike `training/model.py`'s `flatten_observation` (which concatenates
everything, including the board, into one vector), this keeps hex/vertex/edge
features in their natural per-node shape so `training/gnn_model.py` can run
real message passing over them. Ownership fields are encoded *relative to the
observing player* (0=none, 1=me, 2..4=other seats in turn order starting
after me) rather than by absolute seat number, so the network never has to
learn seat-specific weights -- the same policy plays correctly regardless of
which of the 4 seats it happens to occupy that game.
"""
from __future__ import annotations

import numpy as np

from env.board import HEX_TO_RESOURCE, PIP_COUNT, HexType, Resource
from env.state import DevCard, GameState, Phase

RESOURCE_LIST = list(Resource)
DEV_CARD_LIST = list(DevCard)
PHASE_LIST = list(Phase)

HEX_FEAT_DIM = 9       # 6 terrain one-hot + number/12 + pips/5 + robber flag
VERTEX_FEAT_DIM = 15   # 5 owner-relative one-hot + 3 type one-hot + port_generic + 6 port-resource one-hot
EDGE_FEAT_DIM = 5      # 5 owner-relative one-hot
PLAYER_FEAT_DIM = 23
OPPONENT_FEAT_DIM = 9
NUM_OPPONENTS = 3
CONTEXT_FEAT_DIM = 25
SELF_ID_DIM = 4  # one-hot of the observing player's absolute seat (0-3)


def _relative_seat_onehot(owner: int | None, me: int, size: int = 5) -> np.ndarray:
    """0=none, 1=me, 2..size-1 = other seats in turn order after me."""
    v = np.zeros(size, dtype=np.float32)
    if owner is None:
        v[0] = 1.0
    elif owner == me:
        v[1] = 1.0
    else:
        v[2 + ((owner - me - 1) % 3)] = 1.0
    return v


def build_graph_observation(state: GameState, pid: int) -> dict[str, np.ndarray]:
    board = state.board
    me = state.players[pid]

    terrain_order = list(HexType)

    hex_features = np.zeros((19, HEX_FEAT_DIM), dtype=np.float32)
    for hx in board.hexes.values():
        row = hex_features[hx.id]
        row[terrain_order.index(hx.terrain)] = 1.0
        row[6] = (hx.number or 0) / 12.0
        row[7] = PIP_COUNT.get(hx.number, 0) / 5.0
        row[8] = 1.0 if board.robber_hex == hx.id else 0.0

    owner_of: dict[int, tuple[int, str]] = {}
    for other_pid, p in state.players.items():
        for vid in p.settlements:
            owner_of[vid] = (other_pid, "settlement")
        for vid in p.cities:
            owner_of[vid] = (other_pid, "city")

    vertex_features = np.zeros((54, VERTEX_FEAT_DIM), dtype=np.float32)
    for vid, v in board.vertices.items():
        row = vertex_features[vid]
        owner, kind = owner_of.get(vid, (None, None))
        row[0:5] = _relative_seat_onehot(owner, pid)
        if kind is None:
            row[5] = 1.0
        elif kind == "settlement":
            row[6] = 1.0
        else:
            row[7] = 1.0
        row[8] = 1.0 if v.port_generic else 0.0
        if v.port is None:
            row[9] = 1.0
        else:
            row[10 + RESOURCE_LIST.index(v.port)] = 1.0

    road_owner: dict[int, int] = {}
    for other_pid, p in state.players.items():
        for eid in p.roads:
            road_owner[eid] = other_pid

    edge_features = np.zeros((72, EDGE_FEAT_DIM), dtype=np.float32)
    for eid in board.edges:
        edge_features[eid] = _relative_seat_onehot(road_owner.get(eid), pid)

    player_features = np.zeros(PLAYER_FEAT_DIM, dtype=np.float32)
    player_features[0:5] = [me.resources[r] / 19.0 for r in RESOURCE_LIST]
    player_features[5:10] = [me.dev_cards[c] / 14.0 for c in DEV_CARD_LIST]
    player_features[10:15] = [(me.dev_cards[c] - me.dev_cards_bought_this_turn[c]) / 14.0 for c in DEV_CARD_LIST]
    player_features[15] = me.visible_vp() / 12.0
    player_features[16] = me.hidden_vp() / 5.0
    player_features[17] = len(me.settlements) / 5.0
    player_features[18] = len(me.cities) / 4.0
    player_features[19] = len(me.roads) / 15.0
    player_features[20] = me.knights_played / 14.0
    player_features[21] = 1.0 if state.longest_road_holder == pid else 0.0
    player_features[22] = 1.0 if state.largest_army_holder == pid else 0.0

    opponent_features = np.zeros((NUM_OPPONENTS, OPPONENT_FEAT_DIM), dtype=np.float32)
    for offset in range(NUM_OPPONENTS):
        opp_pid = (pid + 1 + offset) % 4
        p = state.players[opp_pid]
        row = opponent_features[offset]
        row[0] = p.hand_size() / 40.0
        row[1] = p.visible_vp() / 12.0
        row[2] = len(p.settlements) / 5.0
        row[3] = len(p.cities) / 4.0
        row[4] = len(p.roads) / 15.0
        row[5] = p.knights_played / 14.0
        row[6] = p.total_dev_cards() / 25.0
        row[7] = 1.0 if state.longest_road_holder == opp_pid else 0.0
        row[8] = 1.0 if state.largest_army_holder == opp_pid else 0.0

    context_features = np.zeros(CONTEXT_FEAT_DIM, dtype=np.float32)
    context_features[PHASE_LIST.index(state.phase)] = 1.0
    dice = state.dice_roll or (0, 0)
    context_features[8] = dice[0] / 6.0
    context_features[9] = dice[1] / 6.0
    if state.pending_trade is not None:
        for r, amt in state.pending_trade.give.items():
            context_features[10 + RESOURCE_LIST.index(r)] = amt / 19.0
        for r, amt in state.pending_trade.want.items():
            context_features[15 + RESOURCE_LIST.index(r)] = amt / 19.0
        context_features[20:25] = _relative_seat_onehot(state.pending_trade.proposer, pid)
    else:
        context_features[20] = 1.0

    # Absolute seat of the observing player -- used only as a deterministic
    # index (never a learned input) to map the GNN's seat-relative opponent
    # embeddings back onto the engine's absolute-player-id action space (see
    # GraphActorCritic._head_logits's "player" case). Every other field here
    # stays relative-to-observer by design (module docstring); this is the
    # one narrow exception, added specifically so the robber-victim/trade-
    # partner head can point at a specific opponent's own embedding instead
    # of an identity-blind pooled summary.
    self_id = np.zeros(SELF_ID_DIM, dtype=np.float32)
    self_id[pid] = 1.0

    return {
        "hex": hex_features,
        "vertex": vertex_features,
        "edge": edge_features,
        "player": player_features,
        "opponent": opponent_features,
        "context": context_features,
        "self_id": self_id,
    }
