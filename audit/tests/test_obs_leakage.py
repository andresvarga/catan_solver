"""Information-leakage audit (Phase 14/22). Counterfactual construction: two
states that differ ONLY in information hidden from observer A must produce
byte-identical observations for A, under every encoder the project trains
with (flat dict + flatten_observation, GNN graph features), with and without
public_hand_features."""
from __future__ import annotations

import copy
import random

import numpy as np
import pytest

from agents.heuristic import HeuristicAgent
from audit.helpers import RES, fresh_main_state, give, place_settlement
from env.actions import Action, ActionType
from env.engine import CatanEngine, legal_actions, step
from env.pettingzoo_env import build_observation
from env.state import DevCard, Phase
from training.graph_features import build_graph_observation
from training.model import flatten_observation


def mid_game_states(n=40, seed0=13_000_000):
    out = []
    for g in range(n):
        seed = seed0 + g
        eng = CatanEngine(seed=seed)
        agents = {i: HeuristicAgent(i, random.Random(seed * 7 + i)) for i in range(4)}
        target = random.Random(seed).randint(60, 400)
        for _ in range(target):
            if eng.done:
                break
            eng.step(agents[eng.acting_player()].choose(eng.state))
        if not eng.done:
            out.append(eng.state)
    return out


def scramble_hidden(state, observer: int, rng: random.Random):
    """Re-deal everything `observer` cannot see, preserving all public counts."""
    s = copy.deepcopy(state)
    # opponent resource identities: hand sizes are public, and so is the bank
    # (the supply stacks are face-up), hence the opponents' *combined* holding
    # of each resource. What stays hidden is how those cards are split among
    # the opponents -- re-deal exactly that.
    opps = [pid for pid in s.players if pid != observer]
    pool = []
    for pid in opps:
        p = s.players[pid]
        for r in RES:
            pool += [r] * p.resources[r]
            p.resources[r] = 0
    rng.shuffle(pool)
    i = 0
    for pid in opps:
        n = state.players[pid].hand_size()
        for r in pool[i:i + n]:
            s.players[pid].resources[r] += 1
        i += n
    # opponent dev-card identities + deck order (counts public)
    pool = list(s.dev_card_deck)
    counts = {}
    for pid, p in s.players.items():
        if pid == observer:
            continue
        counts[pid] = (p.total_dev_cards(), sum(p.dev_cards_bought_this_turn.values()))
        for c, k in p.dev_cards.items():
            pool += [c] * k
        p.dev_cards = {c: 0 for c in DevCard}
        p.dev_cards_bought_this_turn = {c: 0 for c in DevCard}
    rng.shuffle(pool)
    i = 0
    for pid, (held, bought) in counts.items():
        dealt = pool[i:i + held]; i += held
        for c in dealt:
            s.players[pid].dev_cards[c] += 1
        for c in dealt[:bought]:
            s.players[pid].dev_cards_bought_this_turn[c] += 1
    s.dev_card_deck = pool[i:]
    return s


def obs_all(state, pid, phf):
    acting = (pid == __import__("env.engine", fromlist=["acting_player"]).acting_player(state))
    legal = legal_actions(state) if acting else []
    d = build_observation(state, pid, legal, show_mask=acting, public_hand_features=phf)
    g = build_graph_observation(state, pid, public_hand_features=phf)
    return d, flatten_observation(d), g, [repr(a) for a in legal]


def assert_same(a, b):
    d1, f1, g1, l1 = a
    d2, f2, g2, l2 = b
    for k in d1:
        assert np.array_equal(d1[k], d2[k]), f"flat obs key {k} leaks hidden info"
    assert np.array_equal(f1, f2)
    for k in g1:
        assert np.array_equal(g1[k], g2[k]), f"graph obs key {k} leaks hidden info"
    assert l1 == l2, "legal action list depends on hidden info"


STATES = None


def _states():
    global STATES
    if STATES is None:
        STATES = mid_game_states()
    return STATES


@pytest.mark.parametrize("phf", [False, True])
def test_observation_invariant_to_hidden_info(phf):
    rng = random.Random(0)
    checked = 0
    for st in _states():
        for observer in range(4):
            for _ in range(3):
                alt = scramble_hidden(st, observer, rng)
                assert_same(obs_all(st, observer, phf), obs_all(alt, observer, phf))
                checked += 1
    assert checked > 300


def test_scramble_actually_changes_hidden_state():
    """Guard against a vacuous counterfactual."""
    rng = random.Random(1)
    diffs = 0
    for st in _states():
        alt = scramble_hidden(st, 0, rng)
        diffs += any(alt.players[p].resources != st.players[p].resources for p in (1, 2, 3))
    assert diffs > len(_states()) // 2


def test_public_estimates_after_steal_independent_of_stolen_identity():
    """Two worlds: victim holds {ore:2, wood:1} vs {wood:2, ore:1}. After the
    same steal, every public-estimate row must be identical."""
    ests = []
    for hand in ({"ore": 2, "wood": 1}, {"wood": 2, "ore": 1}):
        s = fresh_main_state(13_100_000)
        hx = next(h for h in s.board.hexes.values() if h.id != s.board.robber_hex)
        place_settlement(s, 1, hx.vertex_ids[0])
        give(s, 1, **hand)
        s.public_resource_estimates[1] = {r: 0.0 for r in RES}  # identities unknown
        s.phase = Phase.MOVE_ROBBER
        step(s, Action(ActionType.MOVE_ROBBER, {"hex_id": hx.id, "victim": 1}), rng=random.Random(3))
        ests.append({p: dict(e) for p, e in s.public_resource_estimates.items()})
    assert ests[0] == ests[1]


def test_public_estimates_after_discard_independent_of_discard_choice():
    ests = []
    for cards in ({"ore": 4}, {"wood": 4}):
        s = fresh_main_state(13_100_001)
        give(s, 2, ore=4, wood=4)
        s.public_resource_estimates[2] = {r: 0.0 for r in RES}
        s.public_resource_estimates[2][RES[0]] = 4.0  # 4 wood publicly known
        s.phase = Phase.DISCARD; s.players_to_discard = [2]; s.discard_amounts = {2: 4}
        from env.board import Resource
        step(s, Action(ActionType.DISCARD, {"cards": {Resource(k): v for k, v in cards.items()}}))
        ests.append({p: dict(e) for p, e in s.public_resource_estimates.items()})
    assert ests[0] == ests[1]


def test_counter_offer_terms_visible_to_proposer():
    obs = []
    from env.board import Resource
    for give_r in (Resource.ORE, Resource.SHEEP):
        s = fresh_main_state(13_100_002)
        give(s, 0, wood=1); give(s, 1, ore=1, sheep=1)
        from env.engine import make_trade
        step(s, make_trade(ActionType.PROPOSE_TRADE, {Resource.WOOD: 1}, {Resource.BRICK: 1},
                           actor=0, target=1))
        step(s, Action(ActionType.COUNTER_TRADE, {"give": {give_r: 1}, "want": {Resource.WOOD: 1}}))
        obs.append(obs_all(s, 0, True))
    with pytest.raises(AssertionError):
        assert_same(obs[0], obs[1])
