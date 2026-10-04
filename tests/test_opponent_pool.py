import random
from collections import Counter

from agents.opponent_pool import PooledOpponent, builtin_members
from env.engine import CatanEngine, is_legal_action


def test_pool_draw_is_deterministic_per_episode_seed_and_covers_styles():
    opp = PooledOpponent(1, builtin_members(), random.Random(0))
    draws = []
    for seed in range(400):
        opp.reset_episode(seed)
        a = opp.current.name
        opp.reset_episode(seed)
        assert opp.current.name == a  # pure function of the seed
        draws.append(a)
    counts = Counter(draws)
    assert set(counts) == {"heuristic", "honest", "search", "random"}
    assert counts["heuristic"] > counts["random"] and counts["search"] > counts["random"]


def test_pooled_opponents_play_legal_full_games():
    for seed in range(3):
        eng = CatanEngine(seed=seed)
        agents = {pid: PooledOpponent(pid, builtin_members(), random.Random(seed * 10 + pid))
                  for pid in range(4)}
        for pid, a in agents.items():
            a.reset_episode(seed * 100 + pid)
        steps = 0
        while not eng.done and steps < 6000:
            act = agents[eng.acting_player()].choose(eng.state)
            assert is_legal_action(eng.state, act)
            eng.step(act)
            steps += 1
        assert eng.done
