"""Value-guided decision-time search (roadmap Phase 5)."""
import math
import random

import numpy as np
import torch

from agents.heuristic import HeuristicAgent
from agents.value_search import ValueSearchAgent, action_logprobs
from env.engine import CatanEngine, is_legal_action, is_template
from training.gnn_model import GraphActorCritic
from training.graph_features import build_graph_observation


def _model():
    torch.manual_seed(0)
    return GraphActorCritic(32, 1).eval()


def test_action_logprobs_match_sampler_and_normalize():
    model = _model()
    eng = CatanEngine(seed=7)
    rng = random.Random(7)
    checked = 0
    while not eng.done and checked < 60:
        legal = eng.legal_actions()
        obs = build_graph_observation(eng.state, eng.acting_player())
        batch = {k: torch.tensor(v).unsqueeze(0) for k, v in obs.items()}
        with torch.inference_mode():
            _, logits, _ = model.head_logits_batch(batch)
        row = {k: v[0] for k, v in logits.items()}
        scored = action_logprobs(row, legal)
        total = sum(math.exp(lp) for lp, _ in scored)
        assert total <= 1 + 1e-6
        if not any(is_template(a) for a in legal):
            assert abs(total - 1.0) < 1e-5   # full mass when no trade templates
        action, logprob, _, _ = model.act(batch, legal, rng=np.random.default_rng(checked))
        if not is_template(action) and action.type.value not in ("propose_trade", "counter_trade"):
            lp = dict((repr(a), l) for l, a in scored)[repr(action)]
            assert abs(lp - logprob) < 1e-4
            checked += 1
        eng.step(HeuristicAgent(eng.acting_player(), rng).choose(eng.state, legal))
    assert checked >= 30


def test_value_search_plays_legal_games_and_searches():
    model = _model()
    eng = CatanEngine(seed=8)
    agents = {0: ValueSearchAgent(0, random.Random(0), model=model)}
    agents.update({p: HeuristicAgent(p, random.Random(p)) for p in (1, 2, 3)})
    steps = 0
    while not eng.done and steps < 4000:
        a = agents[eng.acting_player()].choose(eng.state)
        assert is_legal_action(eng.state, a)
        eng.step(a)
        steps += 1
    assert eng.done and agents[0].searched > 5
