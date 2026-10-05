"""Decision-time search with the learned value function (roadmap Phase 5).

At a MAIN-phase decision the agent:
  1. scores every concrete legal action by the policy's joint log-prob
     (type head + pointer heads) and keeps the top `top_k`;
  2. for each candidate, re-samples everything this seat cannot see
     (`agents.search_heuristic.determinize`: opponents' hands and dev cards,
     deck order -- public counts preserved) `n_det` times, applies the
     action with the real engine, and evaluates the resulting position with
     the model's value head from this seat's point of view;
  3. plays the candidate with the best mean value (+ `prior_weight` x its
     policy log-prob).

Trades (structured templates) and every non-MAIN decision fall back to the
policy itself: a pending proposal's value depends on how others respond,
which a one-step lookahead can't see. Determinization keeps the search
honest -- it never reads the true hidden cards -- so it is a legitimate
deployment-time agent, not an oracle.
"""
from __future__ import annotations

import copy
import random

import numpy as np
import torch

from agents.search_heuristic import determinize
from env.engine import acting_player, is_template, legal_actions, step as engine_step
from env.state import Phase
from training.hier_model import (
    ACTION_TYPE_INDEX, TYPE_TO_HEADS, action_to_indices, group_by_type, stage1_mask, stage2_mask,
)


def _log_softmax_masked(logits: np.ndarray, mask: np.ndarray) -> np.ndarray:
    l = np.where(mask > 0, logits.astype(np.float64), -np.inf)
    m = l.max()
    return l - m - np.log(np.exp(l - m).sum())


def action_logprobs(row_logits: dict[str, np.ndarray], legal: list) -> list[tuple[float, object]]:
    """Joint policy log-prob of every concrete (non-template) legal action."""
    by_type = group_by_type(legal)
    type_mask = np.zeros(len(row_logits["type"]), dtype=np.float32)
    for t in by_type:
        type_mask[ACTION_TYPE_INDEX[t]] = 1.0
    type_lp = _log_softmax_masked(row_logits["type"], type_mask)
    out = []
    for t, acts in by_type.items():
        if any(is_template(a) for a in acts):
            continue
        h1, h2 = TYPE_TO_HEADS[t]
        lp1_all = None
        if h1 is not None:
            lp1_all = _log_softmax_masked(row_logits[h1], stage1_mask(t, acts))
        for a in acts:
            lp = type_lp[ACTION_TYPE_INDEX[t]]
            if h1 is not None:
                i1, i2 = action_to_indices(t, acts, a)
                lp += lp1_all[i1]
                if h2 is not None:
                    lp += _log_softmax_masked(row_logits[h2], stage2_mask(t, acts, i1))[i2]
            out.append((float(lp), a))
    return out


class ValueSearchAgent:
    def __init__(self, player_id: int, rng: random.Random | None = None, model=None,
                 model_kind: str = "gnn", public_hand_features: bool = False,
                 top_k: int = 4, n_det: int = 4, prior_weight: float = 0.0):
        from training.agent import HierarchicalLearnedAgent
        self.player_id = player_id
        self.rng = rng or random.Random()
        self.model = model
        self.model_kind = model_kind
        self.phf = public_hand_features
        self.top_k, self.n_det, self.prior_weight = top_k, n_det, prior_weight
        self.policy = HierarchicalLearnedAgent(player_id, self.rng, model=model, deterministic=True,
                                               model_kind=model_kind, public_hand_features=public_hand_features)
        self.searched = 0
        self.overrode = 0

    def _encode(self, state, pid):
        if self.model_kind == "gnn":
            from training.graph_features import build_graph_observation
            return build_graph_observation(state, pid, public_hand_features=self.phf)
        from env.pettingzoo_env import build_observation
        from training.model import flatten_observation
        return flatten_observation(build_observation(state, pid, [], False, public_hand_features=self.phf))

    def _batch(self, encoded: list):
        if self.model_kind == "gnn":
            return {k: torch.tensor(np.stack([e[k] for e in encoded]), dtype=torch.float32)
                    for k in encoded[0]}
        return torch.tensor(np.stack(encoded), dtype=torch.float32)

    def choose(self, state, legal=None):
        legal = legal if legal is not None else legal_actions(state)
        me = acting_player(state)
        policy_action = self.policy.choose(state, legal)
        if len(legal) <= 1 or state.phase != Phase.MAIN or me != self.player_id \
                or policy_action.type.value in ("propose_trade",):
            return policy_action
        with torch.inference_mode():
            _, logits, _ = self.model.head_logits_batch(self._batch([self._encode(state, me)]))
        row = {k: v[0] for k, v in logits.items()}
        scored = sorted(action_logprobs(row, legal), key=lambda t: -t[0])[: self.top_k]
        if len(scored) < 2:
            return policy_action
        self.searched += 1
        sims, owners, won = [], [], []
        seeds = [self.rng.randrange(2 ** 31) for _ in range(self.n_det)]
        for ci, (_, a) in enumerate(scored):
            for sd in seeds:  # common random numbers across candidates
                rr = random.Random(sd)
                sim = copy.deepcopy(state)
                determinize(sim, me, rr)
                engine_step(sim, a, rng=rr)
                sims.append(self._encode(sim, me))
                owners.append(ci)
                won.append(sim.winner == me)
        with torch.inference_mode():
            values = self.model.value(self._batch(sims)).float().cpu().numpy()
        # a won game is worth exactly the terminal win reward, whatever V says
        values = np.where(np.array(won), 1.0, values)
        means = np.zeros(len(scored))
        for ci, v in zip(owners, values):
            means[ci] += v / self.n_det
        for ci, (lp, _) in enumerate(scored):
            means[ci] += self.prior_weight * lp
        best = scored[int(np.argmax(means))][1]
        if best != policy_action:
            self.overrode += 1
        return best
