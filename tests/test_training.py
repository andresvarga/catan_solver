import random

import numpy as np
import torch

from env.pettingzoo_env import CatanAECEnv, MAX_ACTIONS
from training.agent import LearnedAgent, load_model
from training.model import ActorCritic, flatten_observation, observation_dim
from training.ppo import collect_rollout, compute_gae, ppo_update


def test_gae_treats_truncation_as_terminal_no_double_count():
    """The env pays rank-on-standing terminal rewards when an episode
    truncates, so GAE must NOT also bootstrap a value estimate past the
    boundary -- that summed two estimates of the same future outcome into
    one target. Truncated and terminated endings must produce identical
    targets given identical rewards/values."""
    def episode(terminated, truncated):
        return [
            {"value": 0.5, "reward": 0.0, "done": False, "terminated": False, "truncated": False},
            {"value": 0.7, "reward": 1.0, "done": True, "terminated": terminated, "truncated": truncated},
        ]

    trunc = compute_gae(episode(False, True), gamma=0.99, lam=0.95)
    term = compute_gae(episode(True, False), gamma=0.99, lam=0.95)

    # Last transition's target is exactly reward - value: nothing beyond the
    # episode boundary leaks in.
    assert abs(trunc[-1]["advantage"] - (1.0 - 0.7)) < 1e-9
    assert abs(trunc[-1]["return"] - 1.0) < 1e-9
    for a, b in zip(trunc, term):
        assert abs(a["advantage"] - b["advantage"]) < 1e-9
        assert abs(a["return"] - b["return"]) < 1e-9


def test_flatten_observation_matches_observation_dim():
    dim = observation_dim()
    env = CatanAECEnv(randomize_board=False, seed=0)
    env.reset(seed=0)
    obs = env.observe(env.agent_selection)
    flat = flatten_observation(obs)
    assert flat.shape == (dim,)
    assert flat.dtype == np.float32
    assert np.isfinite(flat).all()


def test_masked_actions_are_always_legal_across_many_samples():
    dim = observation_dim()
    model = ActorCritic(obs_dim=dim, hidden=32)
    env = CatanAECEnv(randomize_board=False, seed=1)
    env.reset(seed=1)
    obs = env.observe(env.agent_selection)
    flat = torch.tensor(flatten_observation(obs), dtype=torch.float32).unsqueeze(0)
    mask = torch.tensor(obs["action_mask"], dtype=torch.float32).unsqueeze(0)
    legal_idxs = set(np.nonzero(obs["action_mask"])[0].tolist())
    for _ in range(200):
        action, logprob, entropy, value = model.act(flat, mask, deterministic=False)
        assert int(action.item()) in legal_idxs
    det_action, *_ = model.act(flat, mask, deterministic=True)
    assert int(det_action.item()) in legal_idxs


def test_collect_rollout_produces_valid_transitions():
    dim = observation_dim()
    model = ActorCritic(obs_dim=dim, hidden=32)
    env = CatanAECEnv(randomize_board=False, seed=2, allow_trading=False, allow_dev_cards=False,
                       max_episode_steps=80)
    transitions, summaries = collect_rollout(env, model, "cpu", num_episodes=3, base_seed=0)
    assert len(transitions) > 0
    assert len(summaries) == 3
    for t in transitions:
        assert t["mask"][t["action"]] == 1
        assert np.isfinite(t["advantage"])
        assert np.isfinite(t["return"])


def test_ppo_update_runs_without_nan_and_updates_params():
    dim = observation_dim()
    model = ActorCritic(obs_dim=dim, hidden=32)
    env = CatanAECEnv(randomize_board=False, seed=3, allow_trading=False, allow_dev_cards=False,
                       max_episode_steps=80)
    transitions, _ = collect_rollout(env, model, "cpu", num_episodes=3, base_seed=100)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    before = [p.clone() for p in model.parameters()]
    stats = ppo_update(model, optimizer, transitions, epochs=2, minibatch_size=64)
    for k, v in stats.items():
        assert np.isfinite(v), f"{k} is not finite: {v}"
    after = list(model.parameters())
    assert any(not torch.equal(b, a) for b, a in zip(before, after))


def test_checkpoint_roundtrip_and_learned_agent():
    dim = observation_dim()
    model = ActorCritic(obs_dim=dim, hidden=32)
    path = "/tmp/catan_test_checkpoint.pt"
    torch.save({"model": model.state_dict()}, path)
    loaded = load_model(path, hidden=32)
    agent = LearnedAgent(0, model=loaded, deterministic=True)

    from env.engine import CatanEngine
    engine = CatanEngine(randomize_board=True, seed=7)
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
