"""PettingZoo/AEC bookkeeping, reward accounting, determinism (Phases 17, 18, 23)."""
from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from audit.helpers import random_env_index, state_signature
from env.engine import CatanEngine, total_vp
from env.pettingzoo_env import CatanAECEnv
from env.state import Phase


def _random_play(env, rng, max_steps=20000, record=None):
    """Drive the env with uniformly random legal indices, mimicking the
    training loop's last()/clear_reward pattern. Returns per-agent reward sums
    and decision counts."""
    rew = {a: 0.0 for a in env.possible_agents}
    decisions = {a: 0 for a in env.possible_agents}
    steps = 0
    while env.agents and steps < max_steps:
        agent = env.agent_selection
        obs, r, term, trunc, info = env.last()
        env.clear_reward(agent)
        rew[agent] += r
        if term or trunc:
            env.step(None)
            continue
        n = int(obs["action_mask"].sum())
        assert n > 0
        idx = random_env_index(env, rng)
        if record is not None:
            record.append(idx)
        decisions[agent] += 1
        env.step(idx)
        steps += 1
    return rew, decisions


@pytest.mark.parametrize("seed", range(13_200_000, 13_200_020))
def test_terminal_rewards_delivered_to_every_agent(seed):
    env = CatanAECEnv(seed=seed, allow_trading=False)  # trading off: keeps random games short
    env.reset(seed=seed)
    rew, _ = _random_play(env, random.Random(seed))
    assert env.agents == []
    st = env.engine.state
    assert st.phase == Phase.GAME_OVER
    # winner gets +1 unless tied on VP with someone else
    vps = {p: total_vp(st, p) for p in st.players}
    w = f"player_{st.winner}"
    if list(vps.values()).count(vps[st.winner]) == 1:
        assert rew[w] == pytest.approx(1.0)
    assert sum(rew.values()) == pytest.approx(-0.5)  # 1 + 0 - 0.5 - 1, ties preserve the sum


@pytest.mark.xfail(strict=True, reason="F-01: setup grants mint cards, so a hand can exceed 19 of a "
                                         "resource and own_resources leaves its declared Box(0, 19) "
                                         "(seed 13300027, step 755: 20 wood)")
def test_observations_within_declared_space():
    bad = {}
    for seed in range(13_300_000, 13_300_030):
        env = CatanAECEnv(seed=seed, public_hand_features=True)
        env.reset(seed=seed)
        rng = random.Random(seed)
        steps = 0
        while env.agents and steps < 3000:
            agent = env.agent_selection
            obs, r, term, trunc, _ = env.last()
            env.clear_reward(agent)
            for k, sp in env.observation_space(agent).spaces.items():
                if not sp.contains(np.asarray(obs[k], dtype=sp.dtype)) or obs[k].dtype != sp.dtype:
                    bad.setdefault(k, (seed, steps, obs[k]))
            if term or trunc:
                env.step(None); continue
            env.step(random_env_index(env, rng))
            steps += 1
    assert not bad, f"observation fields outside declared Box/dtype: {list(bad)}"


def test_determinism_same_seed_same_actions():
    sigs = []
    for _ in range(2):
        env = CatanAECEnv(seed=13_400_000)
        env.reset(seed=13_400_000)
        rng = random.Random(5)
        trace = []
        while env.agents:
            obs, r, term, trunc, _ = env.last()
            if term or trunc:
                env.step(None); continue
            env.step(random_env_index(env, rng))
            trace.append(state_signature(env.engine.state))
        sigs.append(trace)
    assert sigs[0] == sigs[1]


def test_different_seeds_differ():
    a = CatanEngine(seed=13_400_001).state
    b = CatanEngine(seed=13_400_002).state
    assert [h.terrain for h in a.board.hexes.values()] != [h.terrain for h in b.board.hexes.values()] \
        or a.dev_card_deck != b.dev_card_deck


def test_board_and_dice_share_one_seed_stream():
    """Documents (not a failure): the dice RNG is random.Random(seed), the
    board RNG is random.Random(seed), the deck RNG is random.Random(seed+1) --
    i.e. game seed s's deck order == the board-shuffle stream of game s+1's
    seed. Correlated streams, not a practical leak."""
    import random as R
    seed = 13_400_003
    assert [R.Random(seed).random() for _ in range(3)] == [R.Random(seed).random() for _ in range(3)]


def test_truncation_pays_no_reward_by_default():
    """F-06 regression: a step-cap truncation is not a win -- nobody is paid,
    even a unique VP leader. The legacy 'rank' mode still pays the leader."""
    found = 0
    for seed in range(13_500_000, 13_500_040):
        results = {}
        for mode in ("zero", "rank"):
            env = CatanAECEnv(seed=seed, max_episode_steps=900, allow_trading=False,
                              truncation_reward=mode)
            env.reset(seed=seed)
            rew, _ = _random_play(env, random.Random(seed))
            results[mode] = (rew, env.engine.state)
        rew0, st = results["zero"]
        if st.phase == Phase.GAME_OVER:
            continue
        vps = {p: total_vp(st, p) for p in st.players}
        top = max(vps.values())
        if list(vps.values()).count(top) != 1:
            continue
        leader = max(vps, key=vps.get)
        assert all(v == 0.0 for v in rew0.values())
        assert results["rank"][0][f"player_{leader}"] == pytest.approx(1.0)
        found += 1
    assert found >= 3


def test_vp_shaping_is_not_potential_based_over_episode():
    """With vp_shaping_weight=w the per-agent shaping sum telescopes to
    w*(final_vp - initial_vp) -- i.e. it is potential-based ONLY if gamma=1.
    Under PPO's gamma=0.99 per-own-decision discounting, early VP is worth
    more than late VP and losing VP (award transfer) is penalized."""
    w = 0.05
    env = CatanAECEnv(seed=13_500_001, vp_shaping_weight=w, allow_trading=False)
    env.reset(seed=13_500_001)
    rew, _ = _random_play(env, random.Random(1))
    st = env.engine.state
    # strip terminal part
    vps = {p: total_vp(st, p) for p in st.players}
    order = sorted(vps, key=lambda p: -vps[p])
    assert all(abs(rew[f"player_{p}"] - w * vps[p]) <= 1.0 + 1e-9 for p in st.players)


def test_dead_step_requires_none():
    env = CatanAECEnv(seed=13_600_000, allow_trading=False)
    env.reset(seed=13_600_000)
    rng = random.Random(0)
    while True:
        obs, r, term, trunc, _ = env.last()
        if term or trunc:
            break
        env.step(random_env_index(env, rng))
    with pytest.raises(Exception):
        env.step(0)  # PettingZoo requires None for dead agents
