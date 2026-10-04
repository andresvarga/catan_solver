import numpy as np

from env.pettingzoo_env import CatanAECEnv
from training.model import flatten_observation, observation_dim
from training.ppo import compute_gae


def test_gae_bootstraps_truncation_but_not_termination():
    """Truncation (step cap) pays no reward and is not the end of the game, so
    its final transition bootstraps from V(s_T) (`bootstrap_value`); a real
    termination is absorbing. (Replaces the old rank-on-standing scheme,
    which paid a VP leader the full win reward on truncation -- audit F-06.)"""
    def episode(terminated, truncated, reward, boot=None):
        last = {"value": 0.7, "reward": reward, "done": True,
                "terminated": terminated, "truncated": truncated}
        if boot is not None:
            last["bootstrap_value"] = boot
        return [{"value": 0.5, "reward": 0.0, "done": False, "terminated": False, "truncated": False},
                last]

    term = compute_gae(episode(True, False, 1.0, boot=0.9), gamma=0.99, lam=0.95)
    assert abs(term[-1]["advantage"] - (1.0 - 0.7)) < 1e-9  # bootstrap ignored on termination
    assert abs(term[-1]["return"] - 1.0) < 1e-9

    trunc = compute_gae(episode(False, True, 0.0, boot=0.9), gamma=0.99, lam=0.95)
    assert abs(trunc[-1]["advantage"] - (0.99 * 0.9 - 0.7)) < 1e-9
    assert abs(trunc[-1]["return"] - 0.99 * 0.9) < 1e-9
    # earlier step: delta0 + gamma*lam*A1
    d0 = 0.0 + 0.99 * 0.7 - 0.5
    assert abs(trunc[0]["advantage"] - (d0 + 0.99 * 0.95 * trunc[-1]["advantage"])) < 1e-9

    no_boot = compute_gae(episode(False, True, 0.0), gamma=0.99, lam=0.95)
    assert abs(no_boot[-1]["advantage"] - (0.0 - 0.7)) < 1e-9


def test_default_discount_keeps_credit_for_opening_decisions():
    """~180-230 real decisions per player per game: gamma must leave most of
    the terminal reward visible to the setup placements (audit F-15)."""
    from training.ppo import GAMMA
    assert GAMMA ** 230 > 0.75


def test_collect_episode_skips_forced_moves_and_bootstraps_truncation():
    import torch
    from training.hier_model import HierarchicalActorCritic
    from training.hier_ppo import collect_episode

    torch.manual_seed(0)
    model = HierarchicalActorCritic(obs_dim=observation_dim(), hidden=32).eval()
    for max_steps in (300, None):
        env = CatanAECEnv(seed=11, max_episode_steps=max_steps, allow_trading=False)
        data = collect_episode(env, model, "cpu", 11)
        rewards_seen = {a: 0.0 for a in env.possible_agents}
        for agent, trs in data.items():
            assert trs and trs[-1]["done"] and not any(t["done"] for t in trs[:-1])
            for t in trs:
                # every stored decision had a real choice somewhere in its heads
                choices = (t["type_mask"].sum() > 1
                           or (t["stage1_head"] is not None and t["sub_mask_1"].sum() > 1)
                           or (t["stage2_head"] is not None and t["sub_mask_2"].sum() > 1))
                assert choices
                rewards_seen[agent] += t["reward"]
            if max_steps is not None:
                assert trs[-1]["truncated"] and "bootstrap_value" in trs[-1]
            else:
                assert trs[-1]["terminated"] and "bootstrap_value" not in trs[-1]
        if max_steps is None:  # forced-move rewards are folded in, not dropped
            assert abs(sum(rewards_seen.values())) < 1e-6  # win/loss reward is zero-sum
        else:
            assert all(v == 0.0 for v in rewards_seen.values())


def test_flatten_observation_matches_observation_dim():
    dim = observation_dim()
    env = CatanAECEnv(randomize_board=False, seed=0)
    env.reset(seed=0)
    obs = env.observe(env.agent_selection)
    flat = flatten_observation(obs)
    assert flat.shape == (dim,)
    assert flat.dtype == np.float32
    assert np.isfinite(flat).all()


def test_evaluate_policy_rotates_trainee_seat(monkeypatch):
    """Seat 0 is the strongest seat, so evaluation must rotate the trainee's
    seat (seed % 4) instead of always seating it first (audit F-14)."""
    import training.train_hier as th
    from agents.random_agent import RandomAgent

    seats = []

    class RecordingAgent(RandomAgent):
        def __init__(self, player_id, model=None, **kwargs):
            super().__init__(player_id)
            seats.append(player_id)

    monkeypatch.setattr(th, "HierarchicalLearnedAgent", RecordingAgent)
    res = th.evaluate_policy(model=None, opponent_kind="random", games=8, seed_base=40,
                             allow_trading=False, max_steps=300)
    assert sorted(seats) == [0, 0, 1, 1, 2, 2, 3, 3]
    lo, hi = res["ci95"]
    assert 0.0 <= lo <= res["win_rate"] <= hi <= 1.0 and res["games"] == 8


def test_write_manifest_records_run_and_keeps_history(tmp_path):
    import argparse
    import json
    from training.run_manifest import write_manifest

    args = argparse.Namespace(seed=3, lr=1e-4)
    p = write_manifest(str(tmp_path), args, {"driver": "test"})
    m = json.load(open(p))
    assert m["args"] == {"seed": 3, "lr": 1e-4} and m["driver"] == "test"
    assert m["git_commit"] and "git_dirty_files" in m and m["torch"]
    write_manifest(str(tmp_path), args)
    assert (tmp_path / "manifest.1.json").exists()


def test_parallel_rollouts_reproduce_sequential_exactly():
    """F-22: rollout data is a pure function of (weights, seeds) -- worker
    count and process scheduling must not change it."""
    import random as _random
    import torch
    from agents.heuristic import HeuristicAgent
    from training.hier_model import HierarchicalActorCritic
    from training.hier_ppo import collect_rollout, collect_rollout_parallel

    torch.manual_seed(0)
    model = HierarchicalActorCritic(obs_dim=observation_dim(), hidden=32).eval()
    kwargs = dict(randomize_board=True, max_episode_steps=400)

    def sig(trs):  # chosen actions + rewards must match exactly
        return [(t["reward"], t["type_idx"], t["sub_idx_1"], t["sub_idx_2"],
                 tuple(t["trade_counts"]), t["done"]) for t in trs]

    def close_logprobs(a, b):  # BLAS thread count differs parent vs worker: ~1e-6 noise
        return all(abs(x["logprob"] - y["logprob"]) < 1e-4 for x, y in zip(a, b))

    for opponents in (None, {pid: HeuristicAgent(pid, _random.Random(pid)) for pid in (1, 2, 3)}):
        seq, _ = collect_rollout(CatanAECEnv(**kwargs), model, "cpu", 6, 500, opponent_agents=opponents)
        par2, _ = collect_rollout_parallel(kwargs, model, 6, 500, 2, opponent_agents=opponents)
        par3, _ = collect_rollout_parallel(kwargs, model, 6, 500, 3, opponent_agents=opponents)
        # Phase 4: several games in flight with batched inference -- in-process
        # and inside forked workers -- must not change the data either
        bat1, _ = collect_rollout_parallel(kwargs, model, 6, 500, 1, opponent_agents=opponents,
                                           envs_per_worker=4)
        bat2, _ = collect_rollout_parallel(kwargs, model, 6, 500, 2, opponent_agents=opponents,
                                           envs_per_worker=3)
        assert sig(seq) == sig(par2) == sig(par3) == sig(bat1) == sig(bat2) and len(seq) > 100
        for other in (par2, par3, bat1, bat2):
            assert close_logprobs(seq, other)
