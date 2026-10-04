"""Structured domestic trades (audit F-20): multi-card bundles (1-3 cards a
side), targeted or broadcast offers, multi-card counters, engine-side
validation, env submission as Action objects, and the policy's autoregressive
bundle head."""
import random

import numpy as np
import pytest
import torch

from env.actions import Action, ActionType
from env.board import Resource
from env.engine import (
    ALL_OPPONENTS, CatanEngine, acting_player, is_legal_action, is_template, legal_actions,
    make_trade, random_trade, step, trade_bundle_ok,
)
from env.pettingzoo_env import CatanAECEnv
from env.state import MAX_TRADE_PROPOSALS_PER_TURN, Phase, new_game

W, B, S, H, O = Resource.WOOD, Resource.BRICK, Resource.SHEEP, Resource.WHEAT, Resource.ORE


def _main_state(seed=0):
    s = new_game(seed=seed)
    s.phase, s.current_player = Phase.MAIN, 0
    return s


def _give(s, pid, **cards):
    for name, k in cards.items():
        r = Resource(name)
        s.players[pid].resources[r] += k
        s.bank[r] -= k


def _total(s):
    return {r: s.bank[r] + sum(p.resources[r] for p in s.players.values()) for r in Resource}


def test_bundle_validation():
    hand = {W: 3, B: 1, S: 0, H: 0, O: 0}
    assert trade_bundle_ok(hand, {W: 2}, {O: 1})
    assert trade_bundle_ok(hand, {W: 2, B: 1}, {O: 2, H: 1})
    assert not trade_bundle_ok(hand, {W: 4}, {O: 1})          # > 3 cards a side
    assert not trade_bundle_ok(hand, {W: 1}, {O: 4})
    assert not trade_bundle_ok(hand, {}, {O: 1})              # gift request
    assert not trade_bundle_ok(hand, {W: 1}, {})              # gift
    assert not trade_bundle_ok(hand, {W: 1}, {W: 1})          # like-for-like
    assert not trade_bundle_ok(hand, {S: 1}, {O: 1})          # not held
    assert not trade_bundle_ok(hand, {W: -1, B: 1}, {O: 1})   # negative
    assert not trade_bundle_ok(hand, {W: 1.0}, {O: 1})        # non-integer


def test_legal_list_has_one_template_per_trade_type_and_mask_hides_it():
    s = _main_state(1)
    _give(s, 0, wood=2)
    props = [a for a in legal_actions(s) if a.type == ActionType.PROPOSE_TRADE]
    assert len(props) == 1 and is_template(props[0])
    assert props[0].params["target_options"] == [1, 2, 3, ALL_OPPONENTS]
    assert not is_legal_action(s, props[0])
    with pytest.raises(ValueError):
        step(s, props[0])


def test_targeted_multicard_trade_executes_and_conserves():
    s = _main_state(2)
    _give(s, 0, wood=2, brick=1)
    _give(s, 2, ore=2, sheep=1)
    before = _total(s)
    t = make_trade(ActionType.PROPOSE_TRADE, {W: 2, B: 1}, {O: 2}, actor=0, target=2)
    assert is_legal_action(s, t)
    step(s, t)
    assert acting_player(s) == 2 and s.trade_targets_remaining == [2]  # only the target responds
    assert ActionType.ACCEPT_TRADE in {a.type for a in legal_actions(s)}
    step(s, Action(ActionType.ACCEPT_TRADE))
    step(s, Action(ActionType.CONFIRM_TRADE, {"target": 2}))
    assert s.players[0].resources[O] == 2 and s.players[0].resources[W] == 0
    assert s.players[2].resources[W] == 2 and s.players[2].resources[B] == 1 and s.players[2].resources[O] == 0
    assert _total(s) == before and s.phase == Phase.MAIN


def test_broadcast_goes_to_all_in_turn_order_and_accept_requires_payment():
    s = _main_state(3)
    s.current_player = 2
    _give(s, 2, sheep=1)
    _give(s, 0, ore=1)  # only player 0 can pay
    step(s, make_trade(ActionType.PROPOSE_TRADE, {S: 1}, {O: 1}, actor=2, target=ALL_OPPONENTS))
    assert s.pending_trade.targets == [3, 0, 1]
    for responder, can_pay in ((3, False), (0, True), (1, False)):
        assert acting_player(s) == responder
        types = {a.type for a in legal_actions(s)}
        assert (ActionType.ACCEPT_TRADE in types) == can_pay
        step(s, Action(ActionType.ACCEPT_TRADE if can_pay else ActionType.REJECT_TRADE))
    assert [a.params["target"] for a in legal_actions(s) if a.type == ActionType.CONFIRM_TRADE] == [0]


def test_multicard_counter_round_trip():
    s = _main_state(4)
    _give(s, 0, wood=1, wheat=2)
    _give(s, 1, ore=3)
    step(s, make_trade(ActionType.PROPOSE_TRADE, {W: 1}, {O: 1}, actor=0, target=1))
    counter_t = next(a for a in legal_actions(s) if a.type == ActionType.COUNTER_TRADE)
    assert is_template(counter_t)
    counter = make_trade(ActionType.COUNTER_TRADE, {O: 2}, {W: 1, H: 2})
    assert is_legal_action(s, counter)
    assert not is_legal_action(s, make_trade(ActionType.COUNTER_TRADE, {O: 4}, {W: 1}))
    step(s, counter)
    assert acting_player(s) == 0
    step(s, Action(ActionType.ACCEPT_TRADE))
    assert s.players[0].resources[O] == 2 and s.players[0].resources[H] == 0
    assert s.players[1].resources[W] == 1 and s.players[1].resources[H] == 2


def test_proposals_capped_per_turn():
    s = _main_state(5)
    _give(s, 0, wood=3)
    for _ in range(MAX_TRADE_PROPOSALS_PER_TURN):
        step(s, make_trade(ActionType.PROPOSE_TRADE, {W: 1}, {O: 1}, actor=0, target=1))
        step(s, Action(ActionType.REJECT_TRADE))
    assert ActionType.PROPOSE_TRADE not in {a.type for a in legal_actions(s)}
    assert not is_legal_action(s, make_trade(ActionType.PROPOSE_TRADE, {W: 1}, {O: 1}, actor=0, target=1))


def test_trades_illegal_outside_their_phase():
    s = _main_state(6)
    _give(s, 0, wood=1)
    s.phase = Phase.ROLL
    assert not is_legal_action(s, make_trade(ActionType.PROPOSE_TRADE, {W: 1}, {O: 1}, actor=0, target=1))
    s.phase = Phase.MAIN
    assert not is_legal_action(s, make_trade(ActionType.COUNTER_TRADE, {W: 1}, {O: 1}))
    assert not is_legal_action(s, make_trade(ActionType.PROPOSE_TRADE, {W: 1}, {O: 1}, actor=0, target=0))


def test_engine_wrapper_rejects_illegal_actions_without_mutation():
    eng = CatanEngine(seed=7)
    before = repr(eng.state.players)
    with pytest.raises(ValueError):
        eng.step(Action(ActionType.BUILD_CITY, {"vertex_id": 0}))
    assert repr(eng.state.players) == before


def test_env_accepts_trade_actions_and_rejects_bad_ones():
    env = CatanAECEnv(seed=8)
    env.reset(seed=8)
    with pytest.raises(ValueError):  # malformed / unaffordable bundle, wrong phase
        env.step(Action(ActionType.PROPOSE_TRADE, {"give": {W: 9}, "want": {O: 1}, "targets": [1]}))
    rng = random.Random(8)
    submitted = 0
    for _ in range(4000):
        if not env.agents:
            break
        obs, r, term, trunc, _ = env.last()
        if term or trunc:
            env.step(None)
            continue
        legal = env.legal_actions()
        tmpl = next((a for a in legal if is_template(a)), None)
        if tmpl is not None and rng.random() < 0.5:
            with pytest.raises(ValueError):  # templates aren't steppable by index
                env.step(legal.index(tmpl))
            env.step(random_trade(tmpl, rng, actor=tmpl.params["actor"]))
            submitted += 1
        else:
            env.step(int(rng.choice(list(np.flatnonzero(obs["action_mask"])))))
    assert submitted > 10


@pytest.mark.parametrize("model_kind", ["hier", "gnn"])
def test_policy_samples_legal_bundles_and_logprobs_match(model_kind):
    from training.hier_model import HierarchicalActorCritic, prepare_transition_batch
    from training.hier_ppo import collect_episode
    from training.model import observation_dim
    from training.model_adapters import FLAT_ADAPTER, GRAPH_ADAPTER
    torch.manual_seed(0)
    if model_kind == "hier":
        model, adapter = HierarchicalActorCritic(observation_dim(), hidden=32), FLAT_ADAPTER
    else:
        from training.gnn_model import GraphActorCritic
        model, adapter = GraphActorCritic(hidden=32, gnn_layers=1), GRAPH_ADAPTER
    model.eval()
    trs = []
    for seed in range(3):
        data = collect_episode(CatanAECEnv(seed=seed, max_episode_steps=600), model, "cpu", seed,
                               adapter=adapter)
        for t in data.values():
            trs.extend(t)
    trade_rows = [t for t in trs if t["trade_counts"][0] >= 0]
    assert len(trade_rows) > 20, "an untrained policy should propose/counter some trades"
    sizes = {(int(t["trade_counts"][:5].sum()), int(t["trade_counts"][5:].sum())) for t in trade_rows}
    assert any(g > 1 or w > 1 for g, w in sizes), "multi-card bundles should be sampled"
    for t in trade_rows:
        c = t["trade_counts"]
        assert 1 <= c[:5].sum() <= 3 and 1 <= c[5:].sum() <= 3 and not (c[:5] * c[5:]).any()
    obs = adapter.to_batch([t["obs"] for t in trs], "cpu")
    with torch.no_grad():
        lp, ent, _ = model.evaluate_actions(obs, prepare_transition_batch(trs, "cpu"))
    old = torch.tensor([t["logprob"] for t in trs])
    assert torch.allclose(lp, old, atol=1e-4)
    assert torch.isfinite(ent).all()
