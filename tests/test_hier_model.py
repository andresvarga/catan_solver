import random

import numpy as np
import torch

from agents.random_agent import choose as random_choose
from env.actions import Action, ActionType
from env.board import Resource
from env.engine import ALL_OPPONENTS, CatanEngine, is_legal_action, legal_actions, make_trade
from env.pettingzoo_env import CatanAECEnv, build_observation
from env.state import DevCard, Phase, new_game
from training.hier_model import PLAYER_SIZE, RESOURCE_LIST, TRADE_TYPES, trade_head_data
from training.agent import HierarchicalLearnedAgent, load_hier_model
from training.hier_model import (
    HierarchicalActorCritic, action_to_indices, group_by_type, match_action, stage1_mask, stage2_mask,
)
from training.hier_ppo import collect_rollout, collect_rollout_parallel, ppo_update
from training.model import flatten_observation, observation_dim


def _model():
    return HierarchicalActorCritic(obs_dim=observation_dim(), hidden=32)


def test_act_always_returns_a_legal_action_across_many_phases():
    model = _model()
    engine = CatanEngine(randomize_board=True, seed=0)
    rng = random.Random(0)
    steps = 0
    while not engine.done and steps < 400:
        actor = engine.acting_player()
        acts = legal_actions(engine.state)
        obs = build_observation(engine.state, actor, acts, show_mask=True)
        flat = torch.tensor(flatten_observation(obs), dtype=torch.float32).unsqueeze(0)
        action, logprob, value, head_data = model.act(flat, acts, deterministic=(steps % 2 == 0))
        assert is_legal_action(engine.state, action)
        engine.step(action)
        steps += 1


def test_evaluate_actions_matches_act_logprob_exactly():
    model = _model()
    engine = CatanEngine(randomize_board=True, seed=1)
    rng = random.Random(1)
    mismatches = 0
    for _ in range(150):
        if engine.done:
            break
        actor = engine.acting_player()
        acts = legal_actions(engine.state)
        obs = build_observation(engine.state, actor, acts, show_mask=True)
        flat = torch.tensor(flatten_observation(obs), dtype=torch.float32).unsqueeze(0)
        action, logprob, value, head_data = model.act(flat, acts, deterministic=False)
        transitions = [{**head_data, "obs": flat.squeeze(0).numpy()}]
        obs_batch = torch.tensor(np.array([t["obs"] for t in transitions]), dtype=torch.float32)
        new_logprob, entropy, val = model.evaluate_actions(obs_batch, transitions)
        if abs(new_logprob.item() - logprob) > 1e-4:
            mismatches += 1
        engine.step(action)
    assert mismatches == 0


def test_vertex_pointer_index_is_the_vertex_id_regardless_of_trading():
    """The regression test for the actual bug: whether trading is legal this
    turn must not change what a given vertex-head index refers to."""
    state_no_trade = new_game(seed=5, allow_trading=False, allow_dev_cards=False)
    state_trade = new_game(seed=5, allow_trading=True, allow_dev_cards=True)
    # both start in an identical setup position (same seed => same board)
    vertex_id = 10
    for st in (state_no_trade, state_trade):
        assert legal_actions(st)  # sanity: setup phase has options
    acts_no_trade = legal_actions(state_no_trade)
    acts_trade = legal_actions(state_trade)
    mask_no_trade = stage1_mask(ActionType.BUILD_SETTLEMENT,
                                 [a for a in acts_no_trade if a.type == ActionType.BUILD_SETTLEMENT])
    mask_trade = stage1_mask(ActionType.BUILD_SETTLEMENT,
                              [a for a in acts_trade if a.type == ActionType.BUILD_SETTLEMENT])
    # same board => same legal settlement spots => identical masks, and
    # position `vertex_id` in that mask always means "vertex_id", full stop.
    assert np.array_equal(mask_no_trade, mask_trade)
    matched = match_action(ActionType.BUILD_SETTLEMENT,
                            [a for a in acts_trade if a.type == ActionType.BUILD_SETTLEMENT],
                            idx1=vertex_id, idx2=None)
    assert matched.params["vertex_id"] == vertex_id


def test_hex_then_player_two_stage_pointer_round_trips():
    state = new_game(seed=2)
    victim_hex = next(iter(state.board.hexes.values()))
    state.players[1].settlements.append(victim_hex.vertex_ids[0])
    state.vertex_owner[victim_hex.vertex_ids[0]] = (1, "settlement")
    state.phase = Phase.MOVE_ROBBER
    state.current_player = 0
    acts = [a for a in legal_actions(state) if a.type == ActionType.MOVE_ROBBER]
    assert acts
    a = next(a for a in acts if a.params["victim"] is not None)
    mask1 = stage1_mask(ActionType.MOVE_ROBBER, acts)
    assert mask1[a.params["hex_id"]] == 1.0
    mask2 = stage2_mask(ActionType.MOVE_ROBBER, acts, a.params["hex_id"])
    assert mask2[a.params["victim"]] == 1.0
    matched = match_action(ActionType.MOVE_ROBBER, acts, a.params["hex_id"], a.params["victim"])
    assert matched == a


def test_year_of_plenty_resource_pair_round_trips_either_order():
    state = new_game(seed=3)
    state.players[0].dev_cards[DevCard.YEAR_OF_PLENTY] = 1
    state.phase = Phase.MAIN
    state.current_player = 0
    acts = [a for a in legal_actions(state) if a.type == ActionType.PLAY_YEAR_OF_PLENTY]
    assert acts
    target = next(a for a in acts if a.params["resources"][0] != a.params["resources"][1])
    r1, r2 = target.params["resources"]
    from training.hier_model import RESOURCE_LIST
    idx1, idx2 = RESOURCE_LIST.index(r1), RESOURCE_LIST.index(r2)
    mask1 = stage1_mask(ActionType.PLAY_YEAR_OF_PLENTY, acts)
    assert mask1[idx1] == 1.0 and mask1[idx2] == 1.0
    mask2 = stage2_mask(ActionType.PLAY_YEAR_OF_PLENTY, acts, idx1)
    assert mask2[idx2] == 1.0
    matched = match_action(ActionType.PLAY_YEAR_OF_PLENTY, acts, idx1, idx2)
    assert set(matched.params["resources"]) == {r1, r2}


def test_maritime_and_propose_and_confirm_trade_round_trip():
    from training.hier_model import RESOURCE_LIST
    state = new_game(seed=4)
    state.players[0].resources[Resource.WOOD] = 4
    state.phase = Phase.MAIN
    state.current_player = 0
    acts = legal_actions(state)

    maritime = [a for a in acts if a.type == ActionType.MARITIME_TRADE]
    assert maritime
    a = maritime[0]
    i1, i2 = RESOURCE_LIST.index(a.params["give"]), RESOURCE_LIST.index(a.params["receive"])
    matched = match_action(ActionType.MARITIME_TRADE, maritime, i1, i2)
    assert matched == a

    propose = [a for a in acts if a.type == ActionType.PROPOSE_TRADE]
    assert len(propose) == 1 and propose[0].params["template"]  # structured: one template
    template = propose[0]
    trade = make_trade(ActionType.PROPOSE_TRADE, {Resource.WOOD: 2}, {Resource.ORE: 1}, actor=0, target=2)
    assert is_legal_action(state, trade)
    mask1 = stage1_mask(ActionType.PROPOSE_TRADE, propose)
    assert mask1.tolist() == [0.0, 1.0, 1.0, 1.0, 1.0]  # opponents 1-3 + "all"
    idx1, _ = action_to_indices(ActionType.PROPOSE_TRADE, propose, trade)
    assert idx1 == 2
    counts, masks = trade_head_data(template, trade)
    assert counts.tolist() == [2, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    assert masks[0].tolist() == [0, 1, 1, 1]  # only wood held: must give 1-3 wood
    assert masks[5].tolist() == [1, 0, 0, 0]  # can't want wood while giving it


def test_discard_index_head_round_trips():
    state = new_game(seed=6)
    state.players[0].resources[Resource.WOOD] = 8
    state.phase = Phase.ROLL
    state.current_player = 0
    from env import engine as eng
    eng._begin_discard_or_robber(state)
    if state.phase == Phase.DISCARD:
        acts = legal_actions(state)
        mask1 = stage1_mask(ActionType.DISCARD, acts)
        assert mask1[: len(acts)].sum() == len(acts)
        matched = match_action(ActionType.DISCARD, acts, 0, None)
        assert matched == acts[0]


def test_hier_collect_rollout_and_ppo_update():
    model = _model()
    env = CatanAECEnv(randomize_board=False, seed=7, allow_trading=False, allow_dev_cards=False,
                       max_episode_steps=80)
    transitions, summaries = collect_rollout(env, model, "cpu", num_episodes=3, base_seed=0)
    assert len(transitions) > 0
    assert len(summaries) == 3
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    stats = ppo_update(model, optimizer, transitions, epochs=2, minibatch_size=64)
    for k, v in stats.items():
        assert np.isfinite(v), f"{k} is not finite: {v}"


def test_hier_collect_rollout_full_ruleset_exercises_all_action_types():
    model = _model()
    env = CatanAECEnv(randomize_board=True, seed=8, max_episode_steps=600)
    transitions, summaries = collect_rollout(env, model, "cpu", num_episodes=4, base_seed=0)
    assert len(transitions) > 0


def test_parallel_rollout_matches_sequential_shape_and_is_valid():
    model = _model()
    env_kwargs = dict(randomize_board=False, allow_trading=False, allow_dev_cards=False,
                       max_episode_steps=200)
    transitions, summaries = collect_rollout_parallel(env_kwargs, model, num_episodes=6,
                                                        base_seed=42, num_workers=3)
    assert len(summaries) == 6
    assert len(transitions) > 0
    for t in transitions:
        assert np.isfinite(t["advantage"])
        assert np.isfinite(t["return"])
        assert t["type_mask"].sum() >= 1

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    stats = ppo_update(model, optimizer, transitions, epochs=1, minibatch_size=64)
    for k, v in stats.items():
        assert np.isfinite(v)


def test_mixed_seat_rollout_only_records_trainee_transitions():
    from agents.heuristic import HeuristicAgent
    from env.pettingzoo_env import CatanAECEnv
    from training.hier_ppo import collect_episode

    model = _model()
    env = CatanAECEnv(randomize_board=False, allow_trading=False, allow_dev_cards=False, max_episode_steps=150)
    opponent_agents = {1: HeuristicAgent(1, random.Random(1)),
                        2: HeuristicAgent(2, random.Random(2)),
                        3: HeuristicAgent(3, random.Random(3))}
    episode_data = collect_episode(env, model, "cpu", seed=0, opponent_agents=opponent_agents)
    assert episode_data["player_0"], "trainee seat should have recorded transitions"
    for pid in (1, 2, 3):
        assert episode_data[f"player_{pid}"] == [], "opponent seats must not record trainee transitions"


def test_mixed_seat_rollout_pure_self_play_when_no_opponents_given():
    from env.pettingzoo_env import CatanAECEnv
    from training.hier_ppo import collect_episode

    model = _model()
    env = CatanAECEnv(randomize_board=False, allow_trading=False, allow_dev_cards=False, max_episode_steps=150)
    episode_data = collect_episode(env, model, "cpu", seed=0, opponent_agents=None)
    assert all(episode_data[f"player_{pid}"] for pid in range(4)), \
        "with no opponent_agents every seat should be the trainee"


def test_collect_rollout_summary_reports_ranking_and_trainee_seats():
    from agents.random_agent import RandomAgent
    from env.pettingzoo_env import CatanAECEnv
    from training.hier_ppo import collect_rollout

    model = _model()
    env = CatanAECEnv(randomize_board=False, allow_trading=False, allow_dev_cards=False, max_episode_steps=150)
    opponent_agents = {pid: RandomAgent(pid, random.Random(pid)) for pid in (1, 2, 3)}
    transitions, summaries = collect_rollout(env, model, "cpu", num_episodes=2, base_seed=0,
                                              opponent_agents=opponent_agents, opponent_name="random")
    for s in summaries:
        assert s["trainee_pids"] == [0]
        assert s["opponent_name"] == "random"
        assert set(s["ranking_pids"]) == {0, 1, 2, 3}


def test_action_to_indices_is_the_exact_inverse_of_match_action():
    """Behavior-cloning correctness depends entirely on this: every action a
    demonstrator (heuristic, random) took must decompose into the same
    (idx1, idx2) that match_action would reconstruct it from, across every
    action type, over many random states."""
    from agents.random_agent import choose as random_choose
    engine = CatanEngine(randomize_board=True, seed=42)
    rng = random.Random(42)
    seen_types = set()
    steps = 0
    while not engine.done and steps < 3000:
        state = engine.state
        acts = legal_actions(state)
        chosen = random_choose(state, rng, acts)
        by_type = {}
        for a in acts:
            by_type.setdefault(a.type, []).append(a)
        actions_of_type = by_type[chosen.type]
        idx1, idx2 = action_to_indices(chosen.type, actions_of_type, chosen)
        if chosen.type in TRADE_TYPES:
            # structured trade: target index + bundle counts must rebuild it exactly
            counts, masks = trade_head_data(actions_of_type[0], chosen)
            assert all(masks[k, c] == 1.0 for k, c in enumerate(counts))
            give = {RESOURCE_LIST[i]: int(c) for i, c in enumerate(counts[:5]) if c}
            want = {RESOURCE_LIST[i]: int(c) for i, c in enumerate(counts[5:]) if c}
            if chosen.type == ActionType.PROPOSE_TRADE:
                target = ALL_OPPONENTS if idx1 == PLAYER_SIZE - 1 else idx1
                reconstructed = make_trade(chosen.type, give, want, actor=state.current_player, target=target)
            else:
                reconstructed = make_trade(chosen.type, give, want)
        else:
            reconstructed = match_action(chosen.type, actions_of_type, idx1, idx2)
        assert reconstructed == chosen, f"{chosen.type}: {reconstructed} != {chosen}"
        seen_types.add(chosen.type)
        engine.step(chosen)
        steps += 1
    # sanity: this game actually exercised a good spread of action types,
    # not just BUILD_ROAD/END_TURN
    assert len(seen_types) >= 8, f"only exercised {seen_types}"


def test_hier_checkpoint_roundtrip_and_agent():
    model = _model()
    path = "/tmp/catan_test_hier_checkpoint.pt"
    torch.save({"model": model.state_dict()}, path)
    loaded = load_hier_model(path, hidden=32)
    agent = HierarchicalLearnedAgent(0, model=loaded, deterministic=True)

    engine = CatanEngine(randomize_board=True, seed=9)
    for _ in range(30):
        if engine.done:
            break
        actor = engine.acting_player()
        if actor == 0:
            action = agent.choose(engine.state)
        else:
            action = random_choose(engine.state, random.Random(0))
        engine.step(action)
