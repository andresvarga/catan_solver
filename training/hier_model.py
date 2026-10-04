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
from env.engine import ALL_OPPONENTS, is_template, make_trade
from env.state import MAX_TRADE_CARDS_PER_SIDE

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
RESOURCE_PAIR_TYPES = {ActionType.PLAY_YEAR_OF_PLENTY, ActionType.MARITIME_TRADE}
PLAYER_ONLY_TYPES = {ActionType.CONFIRM_TRADE, ActionType.PROPOSE_TRADE}  # PROPOSE: target (slot 4 = all)
# Structured domestic trades: after the type (and PROPOSE's target) the bundle
# is decoded by TradeCountHead -- see "Structured trade bundles" below.
TRADE_TYPES = {ActionType.PROPOSE_TRADE, ActionType.COUNTER_TRADE}
NO_PARAM_TYPES = NO_PARAM_TYPES | {ActionType.COUNTER_TRADE}
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
    if head == "player":
        if action_type == ActionType.PROPOSE_TRADE:  # template: opponents + "all" (slot 4)
            mask = np.zeros(PLAYER_SIZE, dtype=np.float32)
            for t in actions_of_type[0].params["target_options"]:
                mask[PLAYER_SIZE - 1 if t == ALL_OPPONENTS else t] = 1.0
            return mask
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
    if action_type == ActionType.PROPOSE_TRADE:
        targets = action.params["targets"]
        return (targets[0] if len(targets) == 1 else PLAYER_SIZE - 1), None
    if action_type == ActionType.CONFIRM_TRADE:
        return action.params["target"], None
    if action_type == ActionType.DISCARD:
        return actions_of_type.index(action), None
    raise ValueError(f"unhandled action type {action_type}")


# --------------------------------------------------------------------------
# Structured trade bundles
# --------------------------------------------------------------------------
# A domestic trade (PROPOSE_TRADE / COUNTER_TRADE) is decoded as 10 small
# categorical decisions after the type (and PROPOSE's target): how many of
# each resource to give (W, B, S, H, O), then how many of each to want, each
# 0..MAX_TRADE_CARDS_PER_SIDE. Decisions are autoregressive -- TradeCountHead
# sees the counts chosen so far -- and masked so every completed bundle is
# legal: 1..3 cards per side, give <= own hand, no resource on both sides.

TRADE_STEPS = 2 * RESOURCE_SIZE
TRADE_COUNT_SIZE = MAX_TRADE_CARDS_PER_SIDE + 1
NO_TRADE_COUNTS = np.full(TRADE_STEPS, -1, dtype=np.int64)
NO_TRADE_MASKS = np.zeros((TRADE_STEPS, TRADE_COUNT_SIZE), dtype=np.float32)


def trade_step_mask(k: int, counts: list[int], hand: dict) -> np.ndarray:
    """Legal counts for decision `k` given the earlier choices `counts`
    (len k). Zero is masked out exactly when choosing it would leave no way
    to put at least one card on this side."""
    mask = np.zeros(TRADE_COUNT_SIZE, dtype=np.float32)
    j = k % RESOURCE_SIZE
    if k < RESOURCE_SIZE:  # give side
        used = sum(counts[:k])
        cap = min(MAX_TRADE_CARDS_PER_SIDE - used, hand.get(RESOURCE_LIST[j], 0))
        later_possible = any(hand.get(RESOURCE_LIST[i], 0) > 0 for i in range(j + 1, RESOURCE_SIZE))
    else:  # want side: never a resource being given
        give = counts[:RESOURCE_SIZE]
        used = sum(counts[RESOURCE_SIZE:k])
        cap = 0 if give[j] > 0 else MAX_TRADE_CARDS_PER_SIDE - used
        later_possible = any(give[i] == 0 for i in range(j + 1, RESOURCE_SIZE))
    mask[:max(cap, 0) + 1] = 1.0
    if used == 0 and not later_possible:
        mask[0] = 0.0
    return mask


def trade_counts_of(action: Action) -> list[int]:
    give, want = action.params["give"], action.params["want"]
    return [give.get(r, 0) for r in RESOURCE_LIST] + [want.get(r, 0) for r in RESOURCE_LIST]


def trade_head_data(template: Action, action: Action) -> tuple[np.ndarray, np.ndarray]:
    """(counts, masks) supervision for a demonstrated concrete trade -- the
    same arrays `act()` stores for a sampled one."""
    counts = trade_counts_of(action)
    masks = np.stack([trade_step_mask(k, counts[:k], template.params["hand"])
                      for k in range(TRADE_STEPS)])
    for k, c in enumerate(counts):
        if masks[k, c] == 0:
            raise ValueError(f"demonstrated trade {action!r} is outside the bundle space at step {k}")
    return np.array(counts, dtype=np.int64), masks


class TradeCountHead(nn.Module):
    """Shared autoregressive count head: decision k sees the trunk features,
    the (normalized) counts of decisions < k, and a one-hot of k."""

    def __init__(self, hidden: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(hidden + 2 * TRADE_STEPS, hidden), nn.ReLU(),
                                 nn.Linear(hidden, TRADE_COUNT_SIZE))
        self.register_buffer("prefix_mask", torch.tril(torch.ones(TRADE_STEPS, TRADE_STEPS), diagonal=-1),
                             persistent=False)
        self.register_buffer("step_onehot", torch.eye(TRADE_STEPS), persistent=False)

    def forward(self, feats: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
        """feats (B, hidden), counts (B, TRADE_STEPS) long (entries >= the
        decoded step are ignored) -> logits (B, TRADE_STEPS, TRADE_COUNT_SIZE)."""
        b = feats.shape[0]
        c = counts.clamp(min=0).to(feats.dtype) / MAX_TRADE_CARDS_PER_SIDE
        prefix = c.unsqueeze(1) * self.prefix_mask.unsqueeze(0)
        steps = self.step_onehot.unsqueeze(0).expand(b, -1, -1)
        x = torch.cat([feats.unsqueeze(1).expand(-1, TRADE_STEPS, -1), prefix, steps], dim=-1)
        return self.net(x)


def trade_logprob_entropy(head: TradeCountHead, feats: torch.Tensor, tb: dict):
    """Per-row trade-bundle log-prob (joint) and entropy (mean over the
    bundle's decisions); zero for non-trade rows -- the evaluate_actions side
    of decode_trade."""
    n = feats.shape[0]
    lp = torch.zeros(n, device=feats.device)
    ent = torch.zeros(n, device=feats.device)
    tc = tb.get("trade_counts")
    if tc is None:
        return lp, ent
    rows = (tc[:, 0] >= 0).nonzero(as_tuple=True)[0]
    if rows.numel() == 0:
        return lp, ent
    counts = tc[rows]
    logits = head(feats[rows], counts).masked_fill(tb["trade_masks"][rows] == 0, NEG_INF)
    dist = Categorical(logits=logits, validate_args=False)
    lp = lp.index_add(0, rows, dist.log_prob(counts).sum(-1))
    # Entropy is averaged (not summed) over the 10 bundle decisions so a trade
    # contributes to the entropy bonus about as much as one pointer head --
    # summed, it added up to ~10*ln4 nats and rewarded trade-happy policies.
    # The log-prob above stays the exact joint (sum) for the PPO ratio.
    ent = ent.index_add(0, rows, dist.entropy().mean(-1))
    return lp, ent


def _pad(mask: np.ndarray) -> np.ndarray:
    out = np.zeros(SUBMASK_PAD, dtype=np.float32)
    out[: len(mask)] = mask
    return out


def masked_sample_np(logits: np.ndarray, mask: np.ndarray, deterministic: bool,
                     rng: np.random.Generator) -> tuple[int, float]:
    """Sample (or argmax) from a masked categorical in NumPy. Rollout-side
    only: tiny per-decision torch ops were ~90% of `act` time (Phase 4
    profiling), and an explicit per-episode Generator keeps rollouts a pure
    function of the episode seed however decisions are batched. Returns
    (index, log-prob of that index); the gradient path recomputes log-probs
    in torch (`evaluate_actions`)."""
    l = np.where(mask > 0, logits.astype(np.float64), -np.inf)
    mx = l.max()
    z = np.exp(l - mx)
    total = z.sum()
    if deterministic:
        idx = int(np.argmax(l))
    else:
        idx = int(np.searchsorted(np.cumsum(z), rng.random() * total, side="right"))
        idx = min(idx, len(z) - 1)
        while z[idx] == 0.0:  # float edge at the very top of the cumsum
            idx -= 1
    return idx, float(l[idx] - mx - np.log(total))


def _trade_mlp_np(head: "TradeCountHead"):
    """Trade head weights as NumPy arrays (W1, b1, W2, b2), cached per call
    of act_batch."""
    lin1, lin2 = head.net[0], head.net[2]
    return (lin1.weight.detach().float().cpu().numpy(), lin1.bias.detach().float().cpu().numpy(),
            lin2.weight.detach().float().cpu().numpy(), lin2.bias.detach().float().cpu().numpy())


def decode_trade(trade_mlp, feats_row: np.ndarray, template: Action, idx1: int,
                 deterministic: bool, rng: np.random.Generator):
    """Sample a concrete bundle for `template` (NumPy forward of
    TradeCountHead; same function as its torch forward). Returns (Action,
    logprob, counts array, masks array)."""
    w1, b1, w2, b2 = trade_mlp
    hidden = feats_row.shape[0]
    base = w1[:, :hidden] @ feats_row + b1           # feature part, shared by all steps
    w_prefix = w1[:, hidden:hidden + TRADE_STEPS]
    w_step = w1[:, hidden + TRADE_STEPS:]
    hand = template.params["hand"]
    counts: list[int] = []
    masks = []
    logprob = 0.0
    prefix = np.zeros(TRADE_STEPS, dtype=np.float32)
    for k in range(TRADE_STEPS):
        m = trade_step_mask(k, counts, hand)
        h = np.maximum(base + w_prefix @ prefix + w_step[:, k], 0.0)
        logits = w2 @ h + b2
        c, lp = masked_sample_np(logits, m, deterministic, rng)
        counts.append(c)
        masks.append(m)
        logprob += lp
        prefix[k] = c / MAX_TRADE_CARDS_PER_SIDE
    give = {RESOURCE_LIST[i]: c for i, c in enumerate(counts[:RESOURCE_SIZE]) if c}
    want = {RESOURCE_LIST[i]: c for i, c in enumerate(counts[RESOURCE_SIZE:]) if c}
    if template.type == ActionType.PROPOSE_TRADE:
        target = ALL_OPPONENTS if idx1 == PLAYER_SIZE - 1 else idx1
        action = make_trade(template.type, give, want, actor=template.params["actor"], target=target)
    else:
        action = make_trade(template.type, give, want)
    return action, logprob, np.array(counts, dtype=np.int64), np.stack(masks)


def sample_decision(row_logits: dict[str, np.ndarray], legal_actions: list[Action], deterministic: bool,
                    rng: np.random.Generator, feats_row: np.ndarray, trade_mlp) -> tuple[Action, float, dict]:
    """One decision from precomputed head logits (one batch row): type ->
    stage-1/stage-2 pointers -> (trade bundle). Returns (Action, joint
    logprob, head_data for storage)."""
    by_type = group_by_type(legal_actions)
    type_mask = np.zeros(NUM_ACTION_TYPES, dtype=np.float32)
    for t in by_type:
        type_mask[ACTION_TYPE_INDEX[t]] = 1.0
    type_idx, logprob = masked_sample_np(row_logits["type"], type_mask, deterministic, rng)
    chosen_type = ACTION_TYPES[type_idx]
    actions_of_type = by_type[chosen_type]
    stage1_head, stage2_head = TYPE_TO_HEADS[chosen_type]

    idx1, idx2 = -1, -1
    mask1_padded = np.zeros(SUBMASK_PAD, dtype=np.float32)
    mask2_padded = np.zeros(SUBMASK_PAD, dtype=np.float32)
    if stage1_head is not None:
        mask1 = stage1_mask(chosen_type, actions_of_type)
        idx1, lp1 = masked_sample_np(row_logits[stage1_head], mask1, deterministic, rng)
        logprob += lp1
        mask1_padded = _pad(mask1)
        if stage2_head is not None:
            mask2 = stage2_mask(chosen_type, actions_of_type, idx1)
            idx2, lp2 = masked_sample_np(row_logits[stage2_head], mask2, deterministic, rng)
            logprob += lp2
            mask2_padded = _pad(mask2)

    trade_counts, trade_masks = NO_TRADE_COUNTS, NO_TRADE_MASKS
    if chosen_type in TRADE_TYPES:
        action, lp_t, trade_counts, trade_masks = decode_trade(
            trade_mlp, feats_row, actions_of_type[0], idx1, deterministic, rng)
        logprob += lp_t
    else:
        action = match_action(chosen_type, actions_of_type,
                              idx1 if idx1 != -1 else None, idx2 if idx2 != -1 else None)
    head_data = {
        "type_mask": type_mask, "type_idx": type_idx,
        "stage1_head": stage1_head, "stage2_head": stage2_head,
        "sub_mask_1": mask1_padded, "sub_idx_1": idx1,
        "sub_mask_2": mask2_padded, "sub_idx_2": idx2,
        "trade_counts": trade_counts, "trade_masks": trade_masks,
    }
    return action, logprob, head_data


def _default_rng() -> np.random.Generator:
    # No Generator given: derive one from torch's global RNG so callers that
    # seed torch (torch.manual_seed) still get reproducible sampling.
    return np.random.default_rng(int(torch.randint(0, 2 ** 62, (1,))))


class ActorSampling:
    """Shared rollout-side acting for both model families. Subclasses
    implement `head_logits_batch(obs_batch) -> (feats np (B, hidden),
    {head_name: np (B, size)} incl. "type", values np (B,))`."""

    def act_batch(self, obs_batch, legal_lists: list[list[Action]], deterministic: bool = False,
                  rngs: list[np.random.Generator] | None = None) -> list[tuple[Action, float, float, dict]]:
        """One forward pass for the whole batch, then per-row NumPy sampling.
        Returns [(Action, joint logprob, value, head_data)] in row order."""
        with torch.inference_mode():
            feats, logits, values = self.head_logits_batch(obs_batch)
        trade_mlp = _trade_mlp_np(self.trade_head)
        out = []
        for i, legal in enumerate(legal_lists):
            rng = rngs[i] if rngs is not None else _default_rng()
            row = {k: v[i] for k, v in logits.items()}
            action, logprob, head_data = sample_decision(row, legal, deterministic, rng, feats[i], trade_mlp)
            out.append((action, logprob, float(values[i]), head_data))
        return out

    def act(self, obs_batch, legal_actions: list[Action], deterministic: bool = False,
            rng: np.random.Generator | None = None) -> tuple[Action, float, float, dict]:
        """Single decision: obs_batch holds one observation (batch dim 1)."""
        return self.act_batch(obs_batch, [legal_actions], deterministic,
                              [rng] if rng is not None else None)[0]


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
    n = len(transitions)
    trade_counts = np.array([t.get("trade_counts", NO_TRADE_COUNTS) for t in transitions], dtype=np.int64)
    trade_masks = np.array([t.get("trade_masks", NO_TRADE_MASKS) for t in transitions], dtype=np.float32)
    return {
        "trade_counts": torch.as_tensor(trade_counts.reshape(n, TRADE_STEPS), device=device),
        "trade_masks": torch.as_tensor(trade_masks.reshape(n, TRADE_STEPS, TRADE_COUNT_SIZE), device=device),
        "type_mask": torch.as_tensor(np.array([t["type_mask"] for t in transitions]), device=device),
        "type_idx": torch.tensor([t["type_idx"] for t in transitions], dtype=torch.long, device=device),
        "head1_id": torch.tensor(head_ids[1], dtype=torch.long, device=device),
        "sub_mask_1": torch.as_tensor(np.array([t["sub_mask_1"] for t in transitions]), device=device),
        "sub_idx_1": torch.tensor([t["sub_idx_1"] for t in transitions], dtype=torch.long, device=device),
        "head2_id": torch.tensor(head_ids[2], dtype=torch.long, device=device),
        "sub_mask_2": torch.as_tensor(np.array([t["sub_mask_2"] for t in transitions]), device=device),
        "sub_idx_2": torch.tensor([t["sub_idx_2"] for t in transitions], dtype=torch.long, device=device),
    }


class HierarchicalActorCritic(ActorSampling, nn.Module):
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
        self.trade_head = TradeCountHead(hidden)
        self.value_head = nn.Linear(hidden, 1)
        self._head_modules = {
            "vertex": self.vertex_head, "edge": self.edge_head, "hex": self.hex_head,
            "player": self.player_head, "resource": self.resource_head,
            "resource2": self.resource2_head, "discard_index": self.discard_index_head,
        }

    def features(self, obs_batch: torch.Tensor) -> torch.Tensor:
        return self.trunk(obs_batch)

    def value(self, obs_batch: torch.Tensor) -> torch.Tensor:
        """Value-only forward pass (no action sampling). Used by
        `hier_ppo.collect_episode` to bootstrap V(s_T) when an episode is
        truncated by the step cap (see `training/ppo.compute_gae`)."""
        feats = self.features(obs_batch)
        return self.value_head(feats).squeeze(-1)

    def head_logits_batch(self, obs_batch: torch.Tensor):
        """Every head's logits for the whole batch in one pass (moved to
        host NumPy once) -- see ActorSampling."""
        feats = self.features(obs_batch)
        logits = {"type": self.type_head(feats)}
        for name, mod in self._head_modules.items():
            logits[name] = mod(feats)
        values = self.value_head(feats).squeeze(-1)
        return (feats.float().cpu().numpy(), {k: v.float().cpu().numpy() for k, v in logits.items()},
                values.float().cpu().numpy())

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

        trade_lp, trade_ent = trade_logprob_entropy(self.trade_head, feats, tb)
        return logprob + extra_logprob + trade_lp, entropy + extra_entropy + trade_ent, value
