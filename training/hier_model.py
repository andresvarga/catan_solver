"""Hierarchical, pointer-based policy head (replaces the flat `Discrete(400)
index-into-legal_actions` scheme from `training/model.py`).

Why this exists: training the flat model (see README "Status") showed it
collapses the moment the legal-action *list composition* changes (trading
turned on) even though nothing about the board did -- because index i's
meaning was "the i-th thing legal_actions() happened to return this step,"
which shifts whenever the mix of action types shifts. §3 of the design doc
called for exactly the fix implemented here: pick the action *type* first
(a fixed, permanent 19-way categorical -- type index 3 is always
BUILD_SETTLEMENT, whether or not trading is legal that turn), then point at
a *stable* game ID (vertex/edge/hex/resource/player) for the parameter,
conditioned on the chosen type. Vertex 12 is always vertex 12.

Simplifications kept deliberately narrow in scope:
- DISCARD keeps a flat "index into this step's enumerated discard combos"
  sub-head. That's fine specifically because a DISCARD decision is always a
  homogeneous list of DISCARD-only combos -- there's no cross-type mixing,
  which is the actual failure mode being fixed here.
- The second resource pick in a pair (YEAR_OF_PLENTY's 2nd resource,
  MARITIME_TRADE's receive, PROPOSE/COUNTER_TRADE's want) is conditioned on
  the first pick only through masking (which second-picks remain legal),
  not by feeding the sampled first choice back into the trunk. A deeper
  autoregressive version would condition the features themselves; this one
  doesn't need to, since the observation/action spaces here are small enough
  that mask-only conditioning already recovers the correct legal joint
  distribution -- it just can't express a *preference* over r2 that depends
  on which r1 was picked beyond legality. Good enough for a first version.
"""
from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from env.actions import Action, ActionType
from env.board import Resource

NEG_INF = -1e9
RESOURCE_LIST = list(Resource)
ACTION_TYPES = list(ActionType)
ACTION_TYPE_INDEX = {t: i for i, t in enumerate(ACTION_TYPES)}

NUM_ACTION_TYPES = len(ACTION_TYPES)
VERTEX_SIZE = 54
EDGE_SIZE = 72
HEX_SIZE = 19
PLAYER_SIZE = 5  # 4 players + "none" (used by robber-victim / knight-victim)
RESOURCE_SIZE = 5
DISCARD_INDEX_SIZE = 100  # matches the engine's DISCARD combo enumeration cap
SUBMASK_PAD = 100  # uniform storage width for whichever sub-head mask is active

NO_PARAM_TYPES = {ActionType.ROLL_DICE, ActionType.END_TURN, ActionType.BUY_DEV_CARD,
                   ActionType.PLAY_ROAD_BUILDING, ActionType.ACCEPT_TRADE,
                   ActionType.REJECT_TRADE, ActionType.CANCEL_TRADE}
VERTEX_TYPES = {ActionType.BUILD_SETTLEMENT, ActionType.BUILD_CITY}
EDGE_TYPES = {ActionType.BUILD_ROAD}
HEX_PLAYER_TYPES = {ActionType.MOVE_ROBBER, ActionType.PLAY_KNIGHT}
RESOURCE_SINGLE_TYPES = {ActionType.PLAY_MONOPOLY}
RESOURCE_PAIR_TYPES = {ActionType.PLAY_YEAR_OF_PLENTY, ActionType.MARITIME_TRADE,
                        ActionType.PROPOSE_TRADE, ActionType.COUNTER_TRADE}
PLAYER_ONLY_TYPES = {ActionType.CONFIRM_TRADE}
DISCARD_TYPES = {ActionType.DISCARD}

# type -> (stage1 head name, stage2 head name or None)
TYPE_TO_HEADS: dict[ActionType, tuple[str | None, str | None]] = {}
for t in NO_PARAM_TYPES:
    TYPE_TO_HEADS[t] = (None, None)
for t in VERTEX_TYPES:
    TYPE_TO_HEADS[t] = ("vertex", None)
for t in EDGE_TYPES:
    TYPE_TO_HEADS[t] = ("edge", None)
for t in HEX_PLAYER_TYPES:
    TYPE_TO_HEADS[t] = ("hex", "player")
for t in RESOURCE_SINGLE_TYPES:
    TYPE_TO_HEADS[t] = ("resource", None)
for t in RESOURCE_PAIR_TYPES:
    TYPE_TO_HEADS[t] = ("resource", "resource2")
for t in PLAYER_ONLY_TYPES:
    TYPE_TO_HEADS[t] = ("player", None)
for t in DISCARD_TYPES:
    TYPE_TO_HEADS[t] = ("discard_index", None)

HEAD_SIZES = {
    "vertex": VERTEX_SIZE, "edge": EDGE_SIZE, "hex": HEX_SIZE, "player": PLAYER_SIZE,
    "resource": RESOURCE_SIZE, "resource2": RESOURCE_SIZE, "discard_index": DISCARD_INDEX_SIZE,
}
# Stable integer ids for head names so a prepared transition batch can encode
# "which sub-head was active" as a plain tensor (-1 = no sub-head).
HEAD_NAMES = ["vertex", "edge", "hex", "player", "resource", "resource2", "discard_index"]
HEAD_NAME_INDEX = {name: i for i, name in enumerate(HEAD_NAMES)}


def group_by_type(legal_actions: list[Action]) -> dict[ActionType, list[Action]]:
    by_type: dict[ActionType, list[Action]] = defaultdict(list)
    for a in legal_actions:
        by_type[a.type].append(a)
    return by_type


def _mask_from_ids(actions: list[Action], field: str, size: int) -> np.ndarray:
    mask = np.zeros(size, dtype=np.float32)
    for a in actions:
        mask[a.params[field]] = 1.0
    return mask


def _mask_from_none_or_id(actions: list[Action], field: str, size: int) -> np.ndarray:
    mask = np.zeros(size, dtype=np.float32)
    for a in actions:
        v = a.params[field]
        mask[size - 1 if v is None else v] = 1.0
    return mask


def stage1_mask(action_type: ActionType, actions_of_type: list[Action]) -> np.ndarray | None:
    head, _ = TYPE_TO_HEADS[action_type]
    if head is None:
        return None
    if head == "vertex":
        return _mask_from_ids(actions_of_type, "vertex_id", VERTEX_SIZE)
    if head == "edge":
        return _mask_from_ids(actions_of_type, "edge_id", EDGE_SIZE)
    if head == "hex":
        return _mask_from_ids(actions_of_type, "hex_id", HEX_SIZE)
    if head == "resource":
        if action_type == ActionType.PLAY_MONOPOLY:
            mask = np.zeros(RESOURCE_SIZE, dtype=np.float32)
            for a in actions_of_type:
                mask[RESOURCE_LIST.index(a.params["resource"])] = 1.0
            return mask
        if action_type == ActionType.PLAY_YEAR_OF_PLENTY:
            mask = np.zeros(RESOURCE_SIZE, dtype=np.float32)
            for a in actions_of_type:
                for r in a.params["resources"]:
                    mask[RESOURCE_LIST.index(r)] = 1.0
            return mask
        if action_type == ActionType.MARITIME_TRADE:
            mask = np.zeros(RESOURCE_SIZE, dtype=np.float32)
            for a in actions_of_type:
                mask[RESOURCE_LIST.index(a.params["give"])] = 1.0
            return mask
        if action_type in (ActionType.PROPOSE_TRADE, ActionType.COUNTER_TRADE):
            mask = np.zeros(RESOURCE_SIZE, dtype=np.float32)
            for a in actions_of_type:
                r = next(iter(a.params["give"]))
                mask[RESOURCE_LIST.index(r)] = 1.0
            return mask
    if head == "player":
        return _mask_from_ids(actions_of_type, "target", PLAYER_SIZE)
    if head == "discard_index":
        mask = np.zeros(DISCARD_INDEX_SIZE, dtype=np.float32)
        mask[: len(actions_of_type)] = 1.0
        return mask
    raise ValueError(f"unhandled stage1 head {head} for {action_type}")


def stage2_mask(action_type: ActionType, actions_of_type: list[Action], idx1: int) -> np.ndarray | None:
    _, head = TYPE_TO_HEADS[action_type]
    if head is None:
        return None
    if head == "player":  # hex -> victim, conditioned on the chosen hex
        mask = np.zeros(PLAYER_SIZE, dtype=np.float32)
        for a in actions_of_type:
            if a.params["hex_id"] == idx1:
                v = a.params["victim"]
                mask[PLAYER_SIZE - 1 if v is None else v] = 1.0
        return mask
    if head == "resource2":
        r1 = RESOURCE_LIST[idx1]
        mask = np.zeros(RESOURCE_SIZE, dtype=np.float32)
        if action_type == ActionType.PLAY_YEAR_OF_PLENTY:
            for a in actions_of_type:
                combo = list(a.params["resources"])
                if r1 in combo:
                    combo.remove(r1)
                    mask[RESOURCE_LIST.index(combo[0])] = 1.0
        elif action_type == ActionType.MARITIME_TRADE:
            for a in actions_of_type:
                if a.params["give"] == r1:
                    mask[RESOURCE_LIST.index(a.params["receive"])] = 1.0
        elif action_type in (ActionType.PROPOSE_TRADE, ActionType.COUNTER_TRADE):
            for a in actions_of_type:
                if next(iter(a.params["give"])) == r1:
                    mask[RESOURCE_LIST.index(next(iter(a.params["want"])))] = 1.0
        return mask
    raise ValueError(f"unhandled stage2 head {head} for {action_type}")


def match_action(action_type: ActionType, actions_of_type: list[Action],
                  idx1: int | None, idx2: int | None) -> Action:
    if action_type in NO_PARAM_TYPES:
        return actions_of_type[0]
    if action_type in VERTEX_TYPES:
        return next(a for a in actions_of_type if a.params["vertex_id"] == idx1)
    if action_type in EDGE_TYPES:
        return next(a for a in actions_of_type if a.params["edge_id"] == idx1)
    if action_type in HEX_PLAYER_TYPES:
        victim = None if idx2 == PLAYER_SIZE - 1 else idx2
        return next(a for a in actions_of_type if a.params["hex_id"] == idx1 and a.params["victim"] == victim)
    if action_type == ActionType.PLAY_MONOPOLY:
        r = RESOURCE_LIST[idx1]
        return next(a for a in actions_of_type if a.params["resource"] == r)
    if action_type == ActionType.PLAY_YEAR_OF_PLENTY:
        target = Counter([RESOURCE_LIST[idx1], RESOURCE_LIST[idx2]])
        return next(a for a in actions_of_type if Counter(a.params["resources"]) == target)
    if action_type == ActionType.MARITIME_TRADE:
        give_r, want_r = RESOURCE_LIST[idx1], RESOURCE_LIST[idx2]
        return next(a for a in actions_of_type if a.params["give"] == give_r and a.params["receive"] == want_r)
    if action_type in (ActionType.PROPOSE_TRADE, ActionType.COUNTER_TRADE):
        give_r, want_r = RESOURCE_LIST[idx1], RESOURCE_LIST[idx2]
        return next(a for a in actions_of_type
                    if next(iter(a.params["give"])) == give_r and next(iter(a.params["want"])) == want_r)
    if action_type == ActionType.CONFIRM_TRADE:
        return next(a for a in actions_of_type if a.params["target"] == idx1)
    if action_type == ActionType.DISCARD:
        return actions_of_type[idx1]
    raise ValueError(f"unhandled action type {action_type}")


def action_to_indices(action_type: ActionType, actions_of_type: list[Action],
                       action: Action) -> tuple[int | None, int | None]:
    """Inverse of `match_action`: given a concrete Action some other agent
    (heuristic, random, a human) chose, recover the (idx1, idx2) pointer
    indices this model's heads would need to reproduce it. Used to turn
    demonstrations into supervised training targets for behavior cloning --
    the resulting (idx1, idx2) plug directly into the same head_data shape
    `model.act()` returns, so a demonstration and a self-generated rollout
    transition are interchangeable to `evaluate_actions`."""
    if action_type in NO_PARAM_TYPES:
        return None, None
    if action_type in VERTEX_TYPES:
        return action.params["vertex_id"], None
    if action_type in EDGE_TYPES:
        return action.params["edge_id"], None
    if action_type in HEX_PLAYER_TYPES:
        victim = action.params["victim"]
        return action.params["hex_id"], (PLAYER_SIZE - 1 if victim is None else victim)
    if action_type == ActionType.PLAY_MONOPOLY:
        return RESOURCE_LIST.index(action.params["resource"]), None
    if action_type == ActionType.PLAY_YEAR_OF_PLENTY:
        r1, r2 = action.params["resources"]
        return RESOURCE_LIST.index(r1), RESOURCE_LIST.index(r2)
    if action_type == ActionType.MARITIME_TRADE:
        return RESOURCE_LIST.index(action.params["give"]), RESOURCE_LIST.index(action.params["receive"])
    if action_type in (ActionType.PROPOSE_TRADE, ActionType.COUNTER_TRADE):
        give_r = next(iter(action.params["give"]))
        want_r = next(iter(action.params["want"]))
        return RESOURCE_LIST.index(give_r), RESOURCE_LIST.index(want_r)
    if action_type == ActionType.CONFIRM_TRADE:
        return action.params["target"], None
    if action_type == ActionType.DISCARD:
        return actions_of_type.index(action), None
    raise ValueError(f"unhandled action type {action_type}")


def _pad(mask: np.ndarray) -> np.ndarray:
    out = np.zeros(SUBMASK_PAD, dtype=np.float32)
    out[: len(mask)] = mask
    return out


def masked_sample(logits: torch.Tensor, mask: torch.Tensor, deterministic: bool) -> tuple[int, float]:
    """Sample (or argmax) from a masked categorical without constructing a
    torch.distributions.Categorical -- profiling showed the Distribution
    machinery (arg validation, constraint checks, dispatch) dominates
    single-sample rollout inference, while the math itself is three ops.
    `logits`/`mask` are 1-D. Returns (index, log-prob of that index).
    Gradient-free by design (`act` is rollout-side only; the gradient path
    recomputes log-probs in `evaluate_actions`), hence the detach."""
    logits = logits.detach().masked_fill(mask == 0, NEG_INF)
    logp = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
    if deterministic:
        idx = int(torch.argmax(logits))
    else:
        idx = int(torch.multinomial(torch.exp(logp), 1))
    return idx, float(logp[idx])


def prepare_transition_batch(transitions: list[dict], device: str) -> dict[str, torch.Tensor]:
    """Convert stored per-transition head data (numpy masks + python ints)
    into one dict of device tensors, built once per rollout instead of once
    per minibatch per epoch inside `evaluate_actions`. Consumed by both
    `HierarchicalActorCritic.evaluate_actions` and the GNN model's."""
    head_ids = {1: [], 2: []}
    for t in transitions:
        for stage, key in ((1, "stage1_head"), (2, "stage2_head")):
            name = t[key]
            head_ids[stage].append(-1 if name is None else HEAD_NAME_INDEX[name])
    return {
        "type_mask": torch.as_tensor(np.array([t["type_mask"] for t in transitions]), device=device),
        "type_idx": torch.tensor([t["type_idx"] for t in transitions], dtype=torch.long, device=device),
        "head1_id": torch.tensor(head_ids[1], dtype=torch.long, device=device),
        "sub_mask_1": torch.as_tensor(np.array([t["sub_mask_1"] for t in transitions]), device=device),
        "sub_idx_1": torch.tensor([t["sub_idx_1"] for t in transitions], dtype=torch.long, device=device),
        "head2_id": torch.tensor(head_ids[2], dtype=torch.long, device=device),
        "sub_mask_2": torch.as_tensor(np.array([t["sub_mask_2"] for t in transitions]), device=device),
        "sub_idx_2": torch.tensor([t["sub_idx_2"] for t in transitions], dtype=torch.long, device=device),
    }


class HierarchicalActorCritic(nn.Module):
    def __init__(self, obs_dim: int, hidden: int = 256):
        super().__init__()
        self.obs_dim = obs_dim
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.type_head = nn.Linear(hidden, NUM_ACTION_TYPES)
        self.vertex_head = nn.Linear(hidden, VERTEX_SIZE)
        self.edge_head = nn.Linear(hidden, EDGE_SIZE)
        self.hex_head = nn.Linear(hidden, HEX_SIZE)
        self.player_head = nn.Linear(hidden, PLAYER_SIZE)
        self.resource_head = nn.Linear(hidden, RESOURCE_SIZE)
        self.resource2_head = nn.Linear(hidden, RESOURCE_SIZE)
        self.discard_index_head = nn.Linear(hidden, DISCARD_INDEX_SIZE)
        self.value_head = nn.Linear(hidden, 1)
        self._head_modules = {
            "vertex": self.vertex_head, "edge": self.edge_head, "hex": self.hex_head,
            "player": self.player_head, "resource": self.resource_head,
            "resource2": self.resource2_head, "discard_index": self.discard_index_head,
        }

    def features(self, obs_batch: torch.Tensor) -> torch.Tensor:
        return self.trunk(obs_batch)

    def value(self, obs_batch: torch.Tensor) -> torch.Tensor:
        """Value-only forward pass (no action sampling). Not used by the
        training loop anymore -- truncation is treated as terminal in
        compute_gae, so nothing bootstraps from it -- but kept as a cheap
        utility for analysis and future centralized-critic work."""
        feats = self.features(obs_batch)
        return self.value_head(feats).squeeze(-1)

    def act(self, obs_tensor: torch.Tensor, legal_actions: list[Action],
            deterministic: bool = False) -> tuple[Action, float, float, dict]:
        """obs_tensor: shape (1, obs_dim). Returns (concrete Action, joint logprob,
        value, head_data-for-storage)."""
        device = obs_tensor.device
        feats = self.features(obs_tensor)  # (1, hidden)
        by_type = group_by_type(legal_actions)

        type_mask = np.zeros(NUM_ACTION_TYPES, dtype=np.float32)
        for t in by_type:
            type_mask[ACTION_TYPE_INDEX[t]] = 1.0
        type_mask_t = torch.as_tensor(type_mask, device=device)

        type_logits = self.type_head(feats).squeeze(0)
        type_idx, logprob = masked_sample(type_logits, type_mask_t, deterministic)

        chosen_type = ACTION_TYPES[type_idx]
        actions_of_type = by_type[chosen_type]
        stage1_head, stage2_head = TYPE_TO_HEADS[chosen_type]

        idx1, idx2 = -1, -1
        mask1_padded = np.zeros(SUBMASK_PAD, dtype=np.float32)
        mask2_padded = np.zeros(SUBMASK_PAD, dtype=np.float32)

        if stage1_head is not None:
            mask1 = stage1_mask(chosen_type, actions_of_type)
            logits1 = self._head_modules[stage1_head](feats).squeeze(0)
            idx1, lp1 = masked_sample(logits1, torch.as_tensor(mask1, device=device), deterministic)
            logprob += lp1
            mask1_padded = _pad(mask1)

            if stage2_head is not None:
                mask2 = stage2_mask(chosen_type, actions_of_type, idx1)
                logits2 = self._head_modules[stage2_head](feats).squeeze(0)
                idx2, lp2 = masked_sample(logits2, torch.as_tensor(mask2, device=device), deterministic)
                logprob += lp2
                mask2_padded = _pad(mask2)

        result_action = match_action(chosen_type, actions_of_type,
                                      idx1 if idx1 != -1 else None, idx2 if idx2 != -1 else None)
        value = self.value_head(feats).squeeze(0).squeeze(-1)

        head_data = {
            "type_mask": type_mask, "type_idx": type_idx,
            "stage1_head": stage1_head, "stage2_head": stage2_head,
            "sub_mask_1": mask1_padded, "sub_idx_1": idx1,
            "sub_mask_2": mask2_padded, "sub_idx_2": idx2,
        }
        return result_action, logprob, float(value.item()), head_data

    def evaluate_actions(self, obs_batch: torch.Tensor, transitions) -> tuple[
            torch.Tensor, torch.Tensor, torch.Tensor]:
        """Batched recomputation of (logprob, entropy, value) for stored transitions
        under the *current* parameters -- the PPO importance-ratio side. Groups
        rows by which head was active at each stage so the sub-head forward
        passes stay vectorized instead of looping per-transition.

        `transitions` is either a prepared tensor batch (see
        `prepare_transition_batch` -- what `ppo_update` passes, built once per
        rollout) or a raw list of transition dicts (converted here, kept for
        tests/ad-hoc callers)."""
        feats = self.features(obs_batch)  # (N, hidden)
        n = feats.shape[0]
        device = feats.device
        tb = transitions if isinstance(transitions, dict) else prepare_transition_batch(transitions, device)

        type_logits = self.type_head(feats).masked_fill(tb["type_mask"] == 0, NEG_INF)
        type_dist = Categorical(logits=type_logits, validate_args=False)
        logprob = type_dist.log_prob(tb["type_idx"])
        entropy = type_dist.entropy()

        value = self.value_head(feats).squeeze(-1)

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
                logits = self._head_modules[head_name](feats[rows]).masked_fill(sub_masks == 0, NEG_INF)
                dist = Categorical(logits=logits, validate_args=False)
                extra_logprob[rows] += dist.log_prob(sub_idxs)
                extra_entropy[rows] += dist.entropy()

        return logprob + extra_logprob, entropy + extra_entropy, value
