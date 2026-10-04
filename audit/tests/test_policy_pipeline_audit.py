"""Masking -> policy -> action -> storage -> PPO-ratio consistency (Phases 16, 20, 27)."""
from __future__ import annotations

import math
import random

import numpy as np
import pytest
import torch

from env.engine import legal_actions
from env.pettingzoo_env import CatanAECEnv
from training.hier_model import HierarchicalActorCritic, prepare_transition_batch
from training.hier_ppo import collect_episode
from training.model import observation_dim
from training.model_adapters import FLAT_ADAPTER, GRAPH_ADAPTER


def _models():
    torch.manual_seed(0)
    from training.gnn_model import GraphActorCritic
    return [("hier", HierarchicalActorCritic(observation_dim(False), hidden=64), FLAT_ADAPTER),
            ("gnn", GraphActorCritic(hidden=32, gnn_layers=2), GRAPH_ADAPTER)]


@pytest.mark.parametrize("which", [0, 1])
def test_rollout_logprobs_match_evaluate_actions_and_actions_legal(which):
    name, model, adapter = _models()[which]
    model.eval()
    trs = []
    n_dec = 0
    for seed in range(13_700_000, 13_700_006):
        env = CatanAECEnv(seed=seed, max_episode_steps=700)
        data = collect_episode(env, model, "cpu", seed, adapter=adapter)
        for agent, t in data.items():
            assert t, "every seat should record transitions in self-play"
            assert t[-1]["done"], "last transition must be terminal/truncated"
            assert all(not x["done"] for x in t[:-1]), "done only on the final transition"
            trs.extend(t)
    assert len(trs) > 2000
    obs = adapter.to_batch([t["obs"] for t in trs], "cpu")
    tb = prepare_transition_batch(trs, "cpu")
    with torch.no_grad():
        lp, ent, val = model.evaluate_actions(obs, tb)
    old = torch.tensor([t["logprob"] for t in trs])
    assert torch.allclose(lp, old, atol=1e-4), f"max |dlogp| {float((lp-old).abs().max())}"
    assert torch.isfinite(ent).all() and torch.isfinite(val).all()
    v_old = torch.tensor([t["value"] for t in trs])
    assert torch.allclose(val, v_old, atol=1e-4)


def test_masked_heads_never_select_illegal_and_no_nan():
    """10k+ sampled decisions through act() across all phases: the chosen
    Action must be an element of legal_actions()."""
    name, model, adapter = _models()[0]
    model.eval()
    n = 0
    for seed in range(13_800_000, 13_800_010):
        env = CatanAECEnv(seed=seed)
        env.reset(seed=seed)
        while env.agents and n < 12_000:
            obs, r, term, trunc, _ = env.last()
            env.clear_reward(env.agent_selection)
            if term or trunc:
                env.step(None); continue
            legal = env.legal_actions()
            x = adapter.to_single(adapter.encode(env, obs, int(env.agent_selection[-1])), "cpu")
            with torch.inference_mode():
                a, lp, v, hd = model.act(x, legal, deterministic=False)
            assert a in legal and math.isfinite(lp) and lp <= 1e-6
            env.step(legal.index(a))
            n += 1
    assert n >= 10_000


def test_year_of_plenty_pair_has_two_encodings():
    """(r1,r2) and (r2,r1) both map to the same unordered YoP action: the
    policy's probability for that action is split across two index paths,
    but evaluate_actions scores only the sampled path. Correct for PPO (ratio
    of the same path), but BC targets from action_to_indices always use the
    canonical order -- a mild representational redundancy."""
    from env.actions import Action, ActionType
    from env.board import Resource
    from training.hier_model import match_action
    acts = [Action(ActionType.PLAY_YEAR_OF_PLENTY, {"resources": [Resource.WOOD, Resource.ORE]})]
    assert match_action(ActionType.PLAY_YEAR_OF_PLENTY, acts, 0, 4) is \
        match_action(ActionType.PLAY_YEAR_OF_PLENTY, acts, 4, 0)


def test_effective_discount_of_terminal_reward():
    """Measures how much of the terminal reward reaches a player's first
    decisions under gamma=0.99 per own decision (Phase 23 evidence)."""
    from agents.heuristic import HeuristicAgent
    from env.engine import CatanEngine
    lens = []
    for seed in range(13_900_000, 13_900_040):
        eng = CatanEngine(seed=seed)
        ag = {i: HeuristicAgent(i, random.Random(seed + i)) for i in range(4)}
        cnt = [0] * 4
        while not eng.done:
            leg = eng.legal_actions()
            p = eng.acting_player()
            if len(leg) > 1:
                cnt[p] += 1
            eng.step(ag[p].choose(eng.state, leg))
        lens.extend(cnt)
    med = float(np.median(lens))
    # record for the report
    print(f"median decisions/agent/game={med:.0f}, 0.99**med={0.99**med:.3f}")
    assert med > 50
