import random

import numpy as np
import torch

from env.engine import CatanEngine, is_legal_action, legal_actions
from training.board_topology import (
    EDGE_TO_VERTEX, HEX_TO_VERTEX, NUM_EDGES, NUM_HEXES, NUM_VERTICES,
    VERTEX_TO_EDGE, VERTEX_TO_HEX, VERTEX_TO_VERTEX,
)
from training.gnn_model import GraphActorCritic
from training.graph_features import (
    CONTEXT_FEAT_DIM, EDGE_FEAT_DIM, HEX_FEAT_DIM, NUM_OPPONENTS, OPPONENT_FEAT_DIM,
    PLAYER_FEAT_DIM, VERTEX_FEAT_DIM, build_graph_observation,
)
from training.hier_ppo import collect_rollout, collect_rollout_parallel, ppo_update
from training.model_adapters import GRAPH_ADAPTER


def _model(hidden=32, layers=2):
    return GraphActorCritic(hidden=hidden, gnn_layers=layers)


def test_topology_relation_sizes_and_symmetry():
    assert HEX_TO_VERTEX[0].shape[0] == NUM_HEXES * 6
    assert VERTEX_TO_HEX[0].shape[0] == NUM_HEXES * 6
    assert EDGE_TO_VERTEX[0].shape[0] == NUM_EDGES * 2
    assert VERTEX_TO_VERTEX[0].shape[0] == NUM_EDGES * 2  # each edge => 2 directed vertex-vertex pairs
    assert VERTEX_TO_EDGE[0].shape[0] == EDGE_TO_VERTEX[0].shape[0]
    # every hex/vertex/edge index actually appears in its own relation
    assert set(HEX_TO_VERTEX[0].tolist()) == set(range(NUM_HEXES))
    assert set(np.concatenate([VERTEX_TO_HEX[0], VERTEX_TO_EDGE[0]]).tolist()) == set(range(NUM_VERTICES))
    assert set(EDGE_TO_VERTEX[0].tolist()) == set(range(NUM_EDGES))


def test_graph_observation_shapes_and_finiteness():
    engine = CatanEngine(randomize_board=True, seed=3)
    for _ in range(15):
        engine.step(legal_actions(engine.state)[0])
    actor = engine.acting_player()
    obs = build_graph_observation(engine.state, actor)
    assert obs["hex"].shape == (NUM_HEXES, HEX_FEAT_DIM)
    assert obs["vertex"].shape == (NUM_VERTICES, VERTEX_FEAT_DIM)
    assert obs["edge"].shape == (NUM_EDGES, EDGE_FEAT_DIM)
    assert obs["player"].shape == (PLAYER_FEAT_DIM,)
    assert obs["opponent"].shape == (NUM_OPPONENTS, OPPONENT_FEAT_DIM)
    assert obs["context"].shape == (CONTEXT_FEAT_DIM,)
    for v in obs.values():
        assert np.isfinite(v).all()


def test_graph_observation_relative_seat_encoding():
    state = CatanEngine(randomize_board=True, seed=4).state
    state.players[1].settlements.append(0)  # vertex 0 owned by player 1
    for me in range(4):
        obs = build_graph_observation(state, me)
        expected_slot = 1 if me == 1 else 2 + ((1 - me - 1) % 3)
        row = obs["vertex"][0]
        assert row[expected_slot] == 1.0
        others = [i for i in range(5) if i != expected_slot]
        assert all(row[i] == 0.0 for i in others)


def test_act_returns_legal_actions_across_many_boards():
    model = _model()
    for seed in range(6):
        engine = CatanEngine(randomize_board=True, seed=seed)
        for step in range(60):
            if engine.done:
                break
            actor = engine.acting_player()
            acts = legal_actions(engine.state)
            obs = build_graph_observation(engine.state, actor)
            obs_t = {k: torch.tensor(v, dtype=torch.float32).unsqueeze(0) for k, v in obs.items()}
            action, logprob, value, head_data = model.act(obs_t, acts, deterministic=(step % 2 == 0))
            assert is_legal_action(engine.state, action)
            assert np.isfinite(logprob) and np.isfinite(value)
            engine.step(action)


def test_evaluate_actions_matches_act_logprob_exactly():
    model = _model()
    engine = CatanEngine(randomize_board=True, seed=5)
    mismatches = 0
    for _ in range(120):
        if engine.done:
            break
        actor = engine.acting_player()
        acts = legal_actions(engine.state)
        obs = build_graph_observation(engine.state, actor)
        obs_t = {k: torch.tensor(v, dtype=torch.float32).unsqueeze(0) for k, v in obs.items()}
        action, logprob, value, head_data = model.act(obs_t, acts, deterministic=False)
        transitions = [{**head_data, "obs": obs}]
        batch = {k: torch.tensor(np.array([t["obs"][k] for t in transitions]), dtype=torch.float32) for k in obs}
        new_logprob, entropy, val = model.evaluate_actions(batch, transitions)
        if abs(new_logprob.item() - logprob) > 1e-4:
            mismatches += 1
        engine.step(action)
    assert mismatches == 0


def test_gnn_collect_rollout_and_ppo_update():
    model = _model()
    from env.pettingzoo_env import CatanAECEnv
    env = CatanAECEnv(randomize_board=False, allow_trading=False, allow_dev_cards=False, max_episode_steps=100)
    transitions, summaries = collect_rollout(env, model, "cpu", num_episodes=3, base_seed=0, adapter=GRAPH_ADAPTER)
    assert len(transitions) > 0
    assert len(summaries) == 3
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    stats = ppo_update(model, optimizer, transitions, epochs=2, minibatch_size=64, adapter=GRAPH_ADAPTER)
    for k, v in stats.items():
        assert np.isfinite(v), f"{k} is not finite: {v}"


def test_gnn_parallel_rollout_matches_sequential_shape():
    model = _model()
    env_kwargs = dict(randomize_board=False, allow_trading=False, allow_dev_cards=False, max_episode_steps=100)
    transitions, summaries = collect_rollout_parallel(env_kwargs, model, num_episodes=4, base_seed=10,
                                                       num_workers=2, adapter=GRAPH_ADAPTER)
    assert len(summaries) == 4
    assert len(transitions) > 0
    for t in transitions:
        assert np.isfinite(t["advantage"])
        assert np.isfinite(t["return"])


def test_gnn_checkpoint_roundtrip_and_agent():
    model = _model()
    path = "/tmp/catan_test_gnn_checkpoint.pt"
    torch.save({"model": model.state_dict()}, path)

    from training.agent import HierarchicalLearnedAgent, load_gnn_model
    loaded = load_gnn_model(path, hidden=32, gnn_layers=2)
    agent = HierarchicalLearnedAgent(0, model=loaded, deterministic=True, model_kind="gnn")

    engine = CatanEngine(randomize_board=True, seed=6)
    for _ in range(30):
        if engine.done:
            break
        actor = engine.acting_player()
        if actor == 0:
            action = agent.choose(engine.state)
        else:
            from agents.random_agent import choose
            action = choose(engine.state, random.Random(0))
        engine.step(action)
