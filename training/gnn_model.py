"""GNN-encoded hierarchical policy (roadmap phase 6).

Same action representation as `training/hier_model.py` (type-then-pointer
heads -- that representational fix is orthogonal to the encoder and stays
exactly as-is, including its head-dispatch/masking helpers, which this module
imports and reuses rather than reimplementing). What changes is the encoder:
instead of a flat MLP over one concatenated observation vector, the board is
now real message passing over its actual hex/vertex/edge adjacency
(`training/board_topology.py`), and the vertex/edge/hex pointer heads score
each candidate by a dot-product against *that node's own learned embedding*
rather than an arbitrary output-neuron weight vector that has no inherent
relationship to "vertex 12" beyond what gradient descent happened to assign
it. Two concrete benefits this is meant to buy over the flat model:

- Weight sharing across topologically-equivalent board positions, so the
  network isn't relearning "a 6/8 hex next to two others" from scratch at
  every distinct vertex index -- this should generalize better across the
  astronomically many random board layouts than a flat vector can.
- A pointer head whose target *is* the node's own representation, so
  "point at vertex 12" has a stable, board-topology-grounded meaning instead
  of depending on an opaque per-index weight row.

Simplification versus the design doc's "relational GAT" suggestion: this
uses mean-aggregated relational message passing (closer to R-GCN) rather
than attention over neighbors. Real attention would need padding every node
to a fixed max degree to batch cleanly; mean aggregation needs no padding
since in-degree is already static per node (precomputed once, see
`board_topology.py`) and is a reasonable, well-precedented GNN simplification
-- upgrading specific relations to attention later is a incremental change,
not a redesign.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from env.actions import Action
from training.board_topology import (
    EDGE_TO_VERTEX, HEX_TO_VERTEX, NUM_EDGES, NUM_HEXES, NUM_VERTICES,
    VERTEX_TO_EDGE, VERTEX_TO_HEX, VERTEX_TO_VERTEX,
)
from training.graph_features import (
    CONTEXT_FEAT_DIM, EDGE_FEAT_DIM, HEX_FEAT_DIM, NUM_OPPONENTS,
    PLAYER_FEAT_DIM, VERTEX_FEAT_DIM, opponent_feat_dim,
)
from env.state import NUM_PLAYERS
from training.hier_model import (
    ACTION_TYPES, ACTION_TYPE_INDEX, HEAD_NAMES, HEAD_SIZES, NEG_INF, NUM_ACTION_TYPES,
    SUBMASK_PAD, TYPE_TO_HEADS, _pad, group_by_type, masked_sample, match_action,
    prepare_transition_batch, stage1_mask, stage2_mask,
)


class RelationalGNNLayer(nn.Module):
    """One round of mean-aggregated relational message passing across all
    five hex/vertex/edge relations, with a residual (add, not replace) update
    per node type."""

    def __init__(self, hidden: int):
        super().__init__()
        self.w_hex_to_vertex = nn.Linear(hidden, hidden)
        self.w_vertex_to_hex = nn.Linear(hidden, hidden)
        self.w_vertex_to_edge = nn.Linear(hidden, hidden)
        self.w_edge_to_vertex = nn.Linear(hidden, hidden)
        self.w_vertex_to_vertex = nn.Linear(hidden, hidden)
        self.update_hex = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.ReLU())
        self.update_vertex = nn.Sequential(nn.Linear(hidden * 3, hidden), nn.ReLU())
        self.update_edge = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.ReLU())

    @staticmethod
    def _aggregate(src_emb: torch.Tensor, src_idx: torch.Tensor, dst_idx: torch.Tensor,
                    dst_size: int, count: torch.Tensor) -> torch.Tensor:
        batch, _, hidden = src_emb.shape
        messages = src_emb[:, src_idx, :]
        out = torch.zeros(batch, dst_size, hidden, device=src_emb.device, dtype=src_emb.dtype)
        out.index_add_(1, dst_idx, messages)
        return out / count.view(1, dst_size, 1)

    def forward(self, hex_emb, vertex_emb, edge_emb, topo: "GraphTopology"):
        msg_v_from_hex = self._aggregate(self.w_hex_to_vertex(hex_emb), topo.hex_to_vertex_src,
                                          topo.hex_to_vertex_dst, NUM_VERTICES, topo.vertex_from_hex_count)
        msg_h_from_vertex = self._aggregate(self.w_vertex_to_hex(vertex_emb), topo.vertex_to_hex_src,
                                             topo.vertex_to_hex_dst, NUM_HEXES, topo.hex_from_vertex_count)
        msg_e_from_vertex = self._aggregate(self.w_vertex_to_edge(vertex_emb), topo.vertex_to_edge_src,
                                             topo.vertex_to_edge_dst, NUM_EDGES, topo.edge_from_vertex_count)
        msg_v_from_edge = self._aggregate(self.w_edge_to_vertex(edge_emb), topo.edge_to_vertex_src,
                                           topo.edge_to_vertex_dst, NUM_VERTICES, topo.vertex_from_edge_count)
        msg_v_from_vertex = self._aggregate(self.w_vertex_to_vertex(vertex_emb), topo.vertex_to_vertex_src,
                                             topo.vertex_to_vertex_dst, NUM_VERTICES, topo.vertex_from_vertex_count)

        new_hex = hex_emb + self.update_hex(torch.cat([hex_emb, msg_h_from_vertex], dim=-1))
        new_vertex = vertex_emb + self.update_vertex(
            torch.cat([vertex_emb, msg_v_from_hex + msg_v_from_edge, msg_v_from_vertex], dim=-1))
        new_edge = edge_emb + self.update_edge(torch.cat([edge_emb, msg_e_from_vertex], dim=-1))
        return new_hex, new_vertex, new_edge


class GraphTopology(nn.Module):
    """Holds the static adjacency as buffers so `.to(device)` moves them
    along with the rest of the model automatically."""

    def __init__(self):
        super().__init__()
        self.register_buffer("hex_to_vertex_src", torch.as_tensor(HEX_TO_VERTEX[0], dtype=torch.long))
        self.register_buffer("hex_to_vertex_dst", torch.as_tensor(HEX_TO_VERTEX[1], dtype=torch.long))
        self.register_buffer("vertex_to_hex_src", torch.as_tensor(VERTEX_TO_HEX[0], dtype=torch.long))
        self.register_buffer("vertex_to_hex_dst", torch.as_tensor(VERTEX_TO_HEX[1], dtype=torch.long))
        self.register_buffer("vertex_to_edge_src", torch.as_tensor(VERTEX_TO_EDGE[0], dtype=torch.long))
        self.register_buffer("vertex_to_edge_dst", torch.as_tensor(VERTEX_TO_EDGE[1], dtype=torch.long))
        self.register_buffer("edge_to_vertex_src", torch.as_tensor(EDGE_TO_VERTEX[0], dtype=torch.long))
        self.register_buffer("edge_to_vertex_dst", torch.as_tensor(EDGE_TO_VERTEX[1], dtype=torch.long))
        self.register_buffer("vertex_to_vertex_src", torch.as_tensor(VERTEX_TO_VERTEX[0], dtype=torch.long))
        self.register_buffer("vertex_to_vertex_dst", torch.as_tensor(VERTEX_TO_VERTEX[1], dtype=torch.long))

        def counts(dst, size):
            c = torch.zeros(size)
            c.index_add_(0, torch.as_tensor(dst, dtype=torch.long), torch.ones(len(dst)))
            return c.clamp(min=1)

        self.register_buffer("vertex_from_hex_count", counts(HEX_TO_VERTEX[1], NUM_VERTICES))
        self.register_buffer("hex_from_vertex_count", counts(VERTEX_TO_HEX[1], NUM_HEXES))
        self.register_buffer("edge_from_vertex_count", counts(VERTEX_TO_EDGE[1], NUM_EDGES))
        self.register_buffer("vertex_from_edge_count", counts(EDGE_TO_VERTEX[1], NUM_VERTICES))
        self.register_buffer("vertex_from_vertex_count", counts(VERTEX_TO_VERTEX[1], NUM_VERTICES))


class GraphActorCritic(nn.Module):
    def __init__(self, hidden: int = 128, gnn_layers: int = 3,
                 public_hand_features: bool = False):
        super().__init__()
        self.hidden = hidden
        self.public_hand_features = public_hand_features
        self.topo = GraphTopology()

        self.hex_embed = nn.Linear(HEX_FEAT_DIM, hidden)
        self.vertex_embed = nn.Linear(VERTEX_FEAT_DIM, hidden)
        self.edge_embed = nn.Linear(EDGE_FEAT_DIM, hidden)
        self.gnn_layers = nn.ModuleList([RelationalGNNLayer(hidden) for _ in range(gnn_layers)])

        self.player_mlp = nn.Sequential(nn.Linear(PLAYER_FEAT_DIM, hidden), nn.Tanh())
        self.opponent_mlp = nn.Sequential(
            nn.Linear(opponent_feat_dim(public_hand_features), hidden), nn.Tanh())
        self.context_mlp = nn.Sequential(nn.Linear(CONTEXT_FEAT_DIM, hidden), nn.Tanh())
        self.fusion = nn.Sequential(
            nn.Linear(hidden * 6, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )

        self.vertex_query = nn.Linear(hidden, hidden)
        self.edge_query = nn.Linear(hidden, hidden)
        self.hex_query = nn.Linear(hidden, hidden)
        self.player_query = nn.Linear(hidden, hidden)
        self.type_head = nn.Linear(hidden, NUM_ACTION_TYPES)
        # "player" head's 2 non-opponent slots: [own seat (always illegal --
        # masked out -- but needs a value in the tensor), "no one"/"none"].
        # The other 3 (real opponent) slots come from player_query · opp_emb,
        # not this linear -- see _head_logits.
        self.player_none_head = nn.Linear(hidden, 2)
        self.resource_head = nn.Linear(hidden, 5)
        self.resource2_head = nn.Linear(hidden, 5)
        self.discard_index_head = nn.Linear(hidden, 100)
        self.value_head = nn.Linear(hidden, 1)

        # Auxiliary head (not used for acting or evaluate_actions -- see
        # vertex_aux_loss). Encourages vertex embeddings to encode a pure
        # board-geometry fact (production value if settled here) that the
        # edge/vertex pointer heads currently have to reconstruct from reward
        # alone, several message-passing hops away from the hexes that
        # actually determine it.
        self.vertex_aux_head = nn.Linear(hidden, 1)

    def encode(self, batch: dict[str, torch.Tensor]):
        hex_emb = self.hex_embed(batch["hex"])
        vertex_emb = self.vertex_embed(batch["vertex"])
        edge_emb = self.edge_embed(batch["edge"])
        for layer in self.gnn_layers:
            hex_emb, vertex_emb, edge_emb = layer(hex_emb, vertex_emb, edge_emb, self.topo)

        opp_emb = self.opponent_mlp(batch["opponent"])  # (B, NUM_OPPONENTS, hidden) -- kept unpooled
        opp_pooled = opp_emb.mean(dim=1)                # for the whole-board context vector only
        player_emb = self.player_mlp(batch["player"])
        context_emb = self.context_mlp(batch["context"])

        fused = torch.cat([hex_emb.mean(dim=1), vertex_emb.mean(dim=1), edge_emb.mean(dim=1),
                            player_emb, opp_pooled, context_emb], dim=-1)
        features = self.fusion(fused)
        return features, hex_emb, vertex_emb, edge_emb, opp_emb

    def _player_head_logits(self, features: torch.Tensor, opp_emb: torch.Tensor,
                             self_id: torch.Tensor) -> torch.Tensor:
        """Pointer over the 3 *unpooled* opponent embeddings for the 3 slots
        that can ever legally be chosen (robber victim / trade partner --
        never yourself), instead of a flat Linear over features that already
        lost per-opponent identity to mean-pooling. `self_id` (B, 4 one-hot)
        is only ever used as a fixed index -- it maps each opponent's
        seat-relative embedding (graph_features.py encodes opponents relative
        to the observer, offsets 0..2 = seats pid+1, pid+2, pid+3 mod 4) back
        onto the engine's absolute player-id action space (5 slots: players
        0-3, then "none"), which is what the shared stage1_mask/stage2_mask/
        match_action in hier_model.py expect and what training/model.py's
        flat model also uses -- this keeps that contract identical for both
        models, changing only how the 3 real-opponent logits are computed."""
        B = features.shape[0]
        query = self.player_query(features)                          # (B, hidden)
        opp_logits = torch.einsum("bh,bnh->bn", query, opp_emb)        # (B, 3) -- offsets 0,1,2
        extra_logits = self.player_none_head(features)                 # (B, 2) -- [own seat, "none"]

        self_pid = torch.argmax(self_id, dim=-1)                       # (B,)
        offsets = torch.arange(NUM_OPPONENTS, device=features.device)  # (3,)
        abs_idx_opp = (self_pid.unsqueeze(1) + 1 + offsets.unsqueeze(0)) % NUM_PLAYERS  # (B, 3)
        none_col = torch.full((B, 1), NUM_PLAYERS, dtype=torch.long, device=features.device)
        full_idx = torch.cat([abs_idx_opp, self_pid.unsqueeze(1), none_col], dim=1)  # (B, 5)
        full_val = torch.cat([opp_logits, extra_logits], dim=1)                     # (B, 5)

        logits = torch.zeros(B, NUM_PLAYERS + 1, device=features.device, dtype=full_val.dtype)
        return logits.scatter(1, full_idx, full_val)

    def _head_logits(self, head_name: str, features: torch.Tensor,
                      hex_emb: torch.Tensor, vertex_emb: torch.Tensor, edge_emb: torch.Tensor,
                      opp_emb: torch.Tensor | None = None, self_id: torch.Tensor | None = None) -> torch.Tensor:
        if head_name == "vertex":
            return torch.einsum("bh,bnh->bn", self.vertex_query(features), vertex_emb)
        if head_name == "edge":
            return torch.einsum("bh,bnh->bn", self.edge_query(features), edge_emb)
        if head_name == "hex":
            return torch.einsum("bh,bnh->bn", self.hex_query(features), hex_emb)
        if head_name == "player":
            return self._player_head_logits(features, opp_emb, self_id)
        if head_name == "resource":
            return self.resource_head(features)
        if head_name == "resource2":
            return self.resource2_head(features)
        if head_name == "discard_index":
            return self.discard_index_head(features)
        raise ValueError(f"unhandled head {head_name}")

    def vertex_target_production(self, hex_batch: torch.Tensor) -> torch.Tensor:
        """Ground-truth auxiliary regression target for `vertex_aux_loss`:
        total pip count of the hexes touching each vertex, i.e. the dominant
        term of `agents.heuristic.vertex_production_value` (which also adds
        small resource-diversity/port bonuses this proxy skips). A pure
        board-geometry fact -- derivable from the `hex` features already in
        every observation, so no demo-collection changes were needed to add
        this signal."""
        pips = hex_batch[..., 7] * 5.0  # undo the /5 scaling in graph_features.py
        batch = pips.shape[0]
        agg = torch.zeros(batch, NUM_VERTICES, device=pips.device, dtype=pips.dtype)
        messages = pips[:, self.topo.hex_to_vertex_src]
        agg.index_add_(1, self.topo.hex_to_vertex_dst, messages)
        return agg / 15.0  # normalize (max = 3 hexes * pip 5)

    def vertex_aux_loss(self, obs_batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """MSE between the vertex_aux_head's per-vertex prediction and the
        board's actual production value at that vertex. Purely a
        representation-shaping term -- it never touches the policy or value
        heads directly, so it can only help by making "is this vertex worth
        building toward" a legible feature of the vertex embedding, not by
        telling the policy what to prefer."""
        _, _, vertex_emb, _, _ = self.encode(obs_batch)
        pred = self.vertex_aux_head(vertex_emb).squeeze(-1)
        target = self.vertex_target_production(obs_batch["hex"])
        return torch.nn.functional.mse_loss(pred, target)

    def value(self, obs_batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Value-only forward pass (no action sampling). Used by
        `hier_ppo.collect_episode` to bootstrap V(s_T) when an episode is
        truncated by the step cap (see `training/ppo.compute_gae`)."""
        features, _, _, _, _ = self.encode(obs_batch)
        return self.value_head(features).squeeze(-1)

    def act(self, obs_batch: dict[str, torch.Tensor], legal_actions: list[Action],
            deterministic: bool = False) -> tuple[Action, float, float, dict]:
        device = next(iter(obs_batch.values())).device
        features, hex_emb, vertex_emb, edge_emb, opp_emb = self.encode(obs_batch)
        self_id = obs_batch["self_id"]
        by_type = group_by_type(legal_actions)

        type_mask = np.zeros(NUM_ACTION_TYPES, dtype=np.float32)
        for t in by_type:
            type_mask[ACTION_TYPE_INDEX[t]] = 1.0
        type_logits = self.type_head(features).squeeze(0)
        type_idx, logprob = masked_sample(type_logits, torch.as_tensor(type_mask, device=device),
                                           deterministic)

        chosen_type = ACTION_TYPES[type_idx]
        actions_of_type = by_type[chosen_type]
        stage1_head, stage2_head = TYPE_TO_HEADS[chosen_type]

        idx1, idx2 = -1, -1
        mask1_padded = np.zeros(SUBMASK_PAD, dtype=np.float32)
        mask2_padded = np.zeros(SUBMASK_PAD, dtype=np.float32)

        if stage1_head is not None:
            mask1 = stage1_mask(chosen_type, actions_of_type)
            logits1 = self._head_logits(stage1_head, features, hex_emb, vertex_emb, edge_emb,
                                         opp_emb, self_id).squeeze(0)
            idx1, lp1 = masked_sample(logits1, torch.as_tensor(mask1, device=device), deterministic)
            logprob += lp1
            mask1_padded = _pad(mask1)

            if stage2_head is not None:
                mask2 = stage2_mask(chosen_type, actions_of_type, idx1)
                logits2 = self._head_logits(stage2_head, features, hex_emb, vertex_emb, edge_emb,
                                             opp_emb, self_id).squeeze(0)
                idx2, lp2 = masked_sample(logits2, torch.as_tensor(mask2, device=device), deterministic)
                logprob += lp2
                mask2_padded = _pad(mask2)

        result_action = match_action(chosen_type, actions_of_type,
                                      idx1 if idx1 != -1 else None, idx2 if idx2 != -1 else None)
        value = self.value_head(features).squeeze(0).squeeze(-1)

        head_data = {
            "type_mask": type_mask, "type_idx": type_idx,
            "stage1_head": stage1_head, "stage2_head": stage2_head,
            "sub_mask_1": mask1_padded, "sub_idx_1": idx1,
            "sub_mask_2": mask2_padded, "sub_idx_2": idx2,
        }
        return result_action, logprob, float(value.item()), head_data

    def evaluate_actions(self, obs_batch: dict[str, torch.Tensor], transitions):
        """See HierarchicalActorCritic.evaluate_actions -- `transitions` is
        either a prepared tensor batch (`prepare_transition_batch`) or a raw
        list of transition dicts (converted here for tests/ad-hoc callers)."""
        features, hex_emb, vertex_emb, edge_emb, opp_emb = self.encode(obs_batch)
        self_id = obs_batch["self_id"]
        n = features.shape[0]
        device = features.device
        tb = transitions if isinstance(transitions, dict) else prepare_transition_batch(transitions, device)

        type_logits = self.type_head(features).masked_fill(tb["type_mask"] == 0, NEG_INF)
        type_dist = Categorical(logits=type_logits, validate_args=False)
        logprob = type_dist.log_prob(tb["type_idx"])
        entropy = type_dist.entropy()
        value = self.value_head(features).squeeze(-1)

        extra_logprob = torch.zeros(n, device=device)
        extra_entropy = torch.zeros(n, device=device)

        for stage in (1, 2):
            head_ids = tb[f"head{stage}_id"]
            for head_id in torch.unique(head_ids).tolist():
                if head_id < 0:
                    continue
                head_name = HEAD_NAMES[head_id]
                rows = (head_ids == head_id).nonzero(as_tuple=True)[0]
                size = HEAD_SIZES[head_name]
                sub_masks = tb[f"sub_mask_{stage}"][rows, :size]
                sub_idxs = tb[f"sub_idx_{stage}"][rows]
                logits = self._head_logits(head_name, features[rows], hex_emb[rows],
                                            vertex_emb[rows], edge_emb[rows],
                                            opp_emb[rows], self_id[rows])
                logits = logits.masked_fill(sub_masks == 0, NEG_INF)
                dist = Categorical(logits=logits, validate_args=False)
                extra_logprob[rows] += dist.log_prob(sub_idxs)
                extra_entropy[rows] += dist.entropy()

        return logprob + extra_logprob, entropy + extra_entropy, value
