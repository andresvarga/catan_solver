"""Tests for the publicly-inferable opponent hand tracker (card counting)
and its flag-gated observation features (`public_hand_features`).

Tracker contract: exact for public-identity flows (production, setup grants,
build costs, trades, Year of Plenty, Monopoly), expected-value updates for
hidden-identity flows (robber-steal identity, discard contents), and the
invariant that estimates are non-negative with sum <= public hand size.
"""
import random

import numpy as np
import torch

from env.actions import Action, ActionType
from env.board import Resource
from agents.random_agent import choose as random_choose
from env.engine import CatanEngine, is_legal_action, legal_actions, step
from env.state import DevCard, Phase, new_game
from env.pettingzoo_env import CatanAECEnv
from training.model import flatten_observation, observation_dim


def _est(state, pid):
    return state.public_resource_estimates[pid]


def _assert_invariants(state):
    for pid, p in state.players.items():
        est = _est(state, pid)
        for r, v in est.items():
            assert v >= -1e-6, f"negative estimate {v} for p{pid} {r}"
        assert sum(est.values()) <= p.hand_size() + 1e-6, \
            f"p{pid} estimate sum {sum(est.values())} exceeds hand {p.hand_size()}"


def test_setup_grants_are_tracked_exactly():
    engine = CatanEngine(randomize_board=True, seed=3)
    rng = random.Random(3)
    while engine.state.phase in (Phase.SETUP_SETTLEMENT, Phase.SETUP_ROAD):
        engine.step(random_choose(engine.state, rng))
    for pid, p in engine.state.players.items():
        for r in Resource:
            assert abs(_est(engine.state, pid)[r] - p.resources[r]) < 1e-9, \
                "setup grants are fully public: estimates must be exact"


def test_build_cost_is_publicly_deducted():
    state = new_game(seed=4)
    state.players[0].resources[Resource.WOOD] = 2
    state.players[0].resources[Resource.BRICK] = 1
    _est(state, 0)[Resource.WOOD] = 2.0
    _est(state, 0)[Resource.BRICK] = 1.0
    state.phase = Phase.MAIN
    state.current_player = 0
    eid = state.board.vertices[0].edge_ids[0]
    state.players[0].settlements.append(0)  # connection for the road
    step(state, Action(ActionType.BUILD_ROAD, {"edge_id": eid}), rng=random.Random(0))
    assert abs(_est(state, 0)[Resource.WOOD] - 1.0) < 1e-9
    assert abs(_est(state, 0)[Resource.BRICK] - 0.0) < 1e-9
    _assert_invariants(state)


def test_robber_steal_moves_expected_distribution():
    state = new_game(seed=8)
    state.players[1].resources[Resource.WOOD] = 2
    state.players[1].resources[Resource.ORE] = 1
    _est(state, 1)[Resource.WOOD] = 2.0
    _est(state, 1)[Resource.ORE] = 1.0
    state.phase = Phase.MOVE_ROBBER
    state.current_player = 0
    target_hex = next(h.id for h in state.board.hexes.values() if h.id != state.board.robber_hex)
    step(state, Action(ActionType.MOVE_ROBBER, {"hex_id": target_hex, "victim": 1}),
         rng=random.Random(0))

    # Victim: 3-card hand lost 1 unidentified card -> estimate scales by 2/3.
    assert abs(_est(state, 1)[Resource.WOOD] - 4 / 3) < 1e-9
    assert abs(_est(state, 1)[Resource.ORE] - 2 / 3) < 1e-9
    # Thief: gains the victim's expected distribution, not the true card.
    assert abs(_est(state, 0)[Resource.WOOD] - 2 / 3) < 1e-9
    assert abs(_est(state, 0)[Resource.ORE] - 1 / 3) < 1e-9
    assert abs(sum(_est(state, 0).values()) - 1.0) < 1e-9  # exactly 1 card's worth
    _assert_invariants(state)


def test_discard_scales_estimate_by_public_count_only():
    state = new_game(seed=9)
    state.players[0].resources[Resource.WOOD] = 4
    state.players[0].resources[Resource.BRICK] = 4
    _est(state, 0)[Resource.WOOD] = 4.0
    _est(state, 0)[Resource.BRICK] = 4.0
    state.phase = Phase.DISCARD
    state.players_to_discard = [0]
    state.discard_amounts = {0: 4}
    # Contents are hidden: even a maximally-lopsided discard only scales the
    # estimate by the public count (4 of 8 cards -> x0.5).
    step(state, Action(ActionType.DISCARD, {"cards": {Resource.WOOD: 4}}), rng=random.Random(0))
    assert abs(_est(state, 0)[Resource.WOOD] - 2.0) < 1e-9
    assert abs(_est(state, 0)[Resource.BRICK] - 2.0) < 1e-9
    assert abs(sum(_est(state, 0).values()) - state.players[0].hand_size()) < 1e-9
    _assert_invariants(state)


def test_monopoly_is_fully_public():
    state = new_game(seed=10)
    state.players[0].dev_cards[DevCard.MONOPOLY] = 1
    state.players[1].resources[Resource.WOOD] = 3
    _est(state, 1)[Resource.WOOD] = 2.0  # partially identified; 1 card unknown
    state.players[2].resources[Resource.WOOD] = 1
    _est(state, 2)[Resource.WOOD] = 1.0
    state.phase = Phase.MAIN
    state.current_player = 0
    step(state, Action(ActionType.PLAY_MONOPOLY, {"resource": Resource.WOOD}), rng=random.Random(0))
    # Victims visibly hand over everything: their wood estimate is exactly 0,
    # and the monopolist is credited the exact public total (4), including
    # the card that had been unknown-identity until now.
    assert _est(state, 1)[Resource.WOOD] == 0.0
    assert _est(state, 2)[Resource.WOOD] == 0.0
    assert abs(_est(state, 0)[Resource.WOOD] - 4.0) < 1e-9
    _assert_invariants(state)


def test_estimates_hold_invariants_across_full_random_games():
    from agents.random_agent import choose
    for seed in range(3):
        engine = CatanEngine(randomize_board=True, seed=seed)
        rng = random.Random(seed)
        steps = 0
        while not engine.done and steps < 2000:
            engine.step(choose(engine.state, rng))
            steps += 1
            _assert_invariants(engine.state)


def test_observation_gating_off_by_default_on_by_flag():
    env_off = CatanAECEnv(randomize_board=False, seed=0)
    env_off.reset(seed=0)
    obs_off = env_off.observe(env_off.agent_selection)
    assert "public_est_resources" not in obs_off
    assert "public_est_resources" not in env_off.observation_space(env_off.agent_selection).spaces
    assert flatten_observation(obs_off).shape[0] == observation_dim()

    env_on = CatanAECEnv(randomize_board=False, seed=0, public_hand_features=True)
    env_on.reset(seed=0)
    rng = random.Random(0)
    for _ in range(30):  # get past setup so estimates are non-trivial
        obs, reward, term, trunc, info = env_on.last()
        if term or trunc:
            break
        env_on.step(random_choose(env_on.engine.state, rng, env_on.legal_actions()))
    obs_on = env_on.observe(env_on.agent_selection)
    assert obs_on["public_est_resources"].shape == (4, 5)
    assert obs_on["public_est_unknown"].shape == (4,)
    assert "public_est_resources" in env_on.observation_space(env_on.agent_selection).spaces
    dim_on = observation_dim(public_hand_features=True)
    assert flatten_observation(obs_on).shape[0] == dim_on
    assert dim_on == observation_dim() + 4 * 5 + 4
    # values come straight from the engine's tracker
    state = env_on.engine.state
    for pid in range(4):
        for i, r in enumerate(Resource):
            assert abs(obs_on["public_est_resources"][pid, i] - _est(state, pid)[r]) < 1e-6


def test_gnn_observation_and_model_with_flag():
    from training.gnn_model import GraphActorCritic
    from training.graph_features import (
        OPPONENT_FEAT_DIM, PUBLIC_HAND_FEAT_DIM, build_graph_observation,
    )

    engine = CatanEngine(randomize_board=True, seed=5)
    rng = random.Random(5)
    for _ in range(12):
        engine.step(random_choose(engine.state, rng))
    state = engine.state

    obs_off = build_graph_observation(state, 0)
    assert obs_off["opponent"].shape == (3, OPPONENT_FEAT_DIM)
    obs_on = build_graph_observation(state, 0, public_hand_features=True)
    assert obs_on["opponent"].shape == (3, OPPONENT_FEAT_DIM + PUBLIC_HAND_FEAT_DIM)
    # base features identical; new columns match the tracker
    assert np.allclose(obs_on["opponent"][:, :OPPONENT_FEAT_DIM], obs_off["opponent"])
    for offset in range(3):
        opp = (0 + 1 + offset) % 4
        for i, r in enumerate(Resource):
            assert abs(obs_on["opponent"][offset, OPPONENT_FEAT_DIM + i] * 19.0 - _est(state, opp)[r]) < 1e-5

    model = GraphActorCritic(hidden=32, gnn_layers=2, public_hand_features=True)
    acts = legal_actions(state)
    obs_t = {k: torch.tensor(v, dtype=torch.float32).unsqueeze(0) for k, v in obs_on.items()}
    action, logprob, value, head_data = model.act(obs_t, acts, deterministic=True)
    assert is_legal_action(state, action)


def test_end_to_end_training_step_with_flag_both_models():
    from training.hier_model import HierarchicalActorCritic
    from training.hier_ppo import collect_rollout, ppo_update
    from training.model_adapters import FLAT_ADAPTER, GRAPH_ADAPTER
    from training.gnn_model import GraphActorCritic

    env = CatanAECEnv(randomize_board=False, allow_trading=False, allow_dev_cards=False,
                       max_episode_steps=80, public_hand_features=True)

    hier = HierarchicalActorCritic(obs_dim=observation_dim(public_hand_features=True), hidden=32)
    transitions, _ = collect_rollout(env, hier, "cpu", num_episodes=2, base_seed=0)
    assert transitions
    stats = ppo_update(hier, torch.optim.Adam(hier.parameters(), lr=3e-4), transitions,
                       epochs=1, minibatch_size=64)
    assert all(np.isfinite(v) for v in stats.values())

    gnn = GraphActorCritic(hidden=32, gnn_layers=2, public_hand_features=True)
    transitions, _ = collect_rollout(env, gnn, "cpu", num_episodes=2, base_seed=10,
                                      adapter=GRAPH_ADAPTER)
    assert transitions
    stats = ppo_update(gnn, torch.optim.Adam(gnn.parameters(), lr=3e-4), transitions,
                       epochs=1, minibatch_size=64, adapter=GRAPH_ADAPTER)
    assert all(np.isfinite(v) for v in stats.values())
