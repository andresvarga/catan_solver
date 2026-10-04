"""Throughput / component timing (audit Phase 32). Single process, 1 torch thread.

    python -m audit.tools.profile_perf --out audit/results/perf.json
"""
from __future__ import annotations

import argparse
import cProfile
import io
import json
import pstats
import random
import time

import torch

from agents.heuristic import HeuristicAgent
from agents.random_agent import RandomAgent
from env.engine import CatanEngine, compute_longest_road_length, legal_actions
from env.pettingzoo_env import CatanAECEnv, build_observation
from training.graph_features import build_graph_observation
from training.model import flatten_observation


def timeit(fn, n):
    t = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t) / n


def sample_states(n_games=20):
    states = []
    import copy
    for g in range(n_games):
        eng = CatanEngine(seed=21_000_000 + g)
        ag = {i: HeuristicAgent(i, random.Random(g * 4 + i)) for i in range(4)}
        k = 0
        while not eng.done:
            eng.step(ag[eng.acting_player()].choose(eng.state))
            k += 1
            if k % 50 == 0:
                states.append(copy.deepcopy(eng.state))
    return states


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(1)
    out = {}

    # raw engine: random-agent games
    t = time.perf_counter(); steps = 0; games = 0
    while time.perf_counter() - t < 20:
        eng = CatanEngine(seed=21_100_000 + games)
        rng = random.Random(games)
        while not eng.done and steps < 10**9:
            leg = eng.legal_actions()
            eng.step(RandomAgent(0, rng).choose(eng.state, leg))
            steps += 1
        games += 1
    dt = time.perf_counter() - t
    out["engine_random_steps_per_s"] = steps / dt
    out["engine_random_games_per_s"] = games / dt

    # heuristic games
    t = time.perf_counter(); steps = 0; games = 0
    while time.perf_counter() - t < 20:
        eng = CatanEngine(seed=21_200_000 + games)
        ag = {i: HeuristicAgent(i, random.Random(games * 4 + i)) for i in range(4)}
        while not eng.done:
            eng.step(ag[eng.acting_player()].choose(eng.state)); steps += 1
        games += 1
    dt = time.perf_counter() - t
    out["engine_heuristic_steps_per_s"] = steps / dt
    out["engine_heuristic_games_per_s"] = games / dt

    states = sample_states()
    out["n_sample_states"] = len(states)
    def per_state(f):
        t = time.perf_counter()
        for s in states:
            f(s)
        return (time.perf_counter() - t) / len(states) * 1e6
    out["us_legal_actions"] = per_state(legal_actions)
    out["us_build_observation"] = per_state(lambda s: build_observation(s, s.current_player, [], False))
    out["us_flatten_observation"] = per_state(
        lambda s: flatten_observation(build_observation(s, s.current_player, [], False)))
    out["us_graph_observation"] = per_state(lambda s: build_graph_observation(s, s.current_player))
    out["us_longest_road_all_players"] = per_state(
        lambda s: [compute_longest_road_length(s, p) for p in range(4)])

    from training.hier_model import HierarchicalActorCritic
    from training.gnn_model import GraphActorCritic
    from training.model import observation_dim
    hier = HierarchicalActorCritic(observation_dim(), 256).eval()
    gnn = GraphActorCritic(128, 3).eval()
    gnn_big = GraphActorCritic(256, 4).eval()
    def act_hier(s):
        leg = legal_actions(s)
        x = torch.tensor(flatten_observation(build_observation(s, s.current_player, leg, True))).unsqueeze(0)
        with torch.inference_mode():
            hier.act(x, leg)
    def act_gnn(m):
        def f(s):
            leg = legal_actions(s)
            g = build_graph_observation(s, s.current_player)
            x = {k: torch.tensor(v).unsqueeze(0) for k, v in g.items()}
            with torch.inference_mode():
                m.act(x, leg)
        return f
    sub = states[:200]
    out["us_policy_act_hier256_cpu"] = sum(timeit(lambda: act_hier(s), 1) for s in sub) / len(sub) * 1e6
    out["us_policy_act_gnn128x3_cpu"] = sum(timeit(lambda: act_gnn(gnn)(s), 1) for s in sub) / len(sub) * 1e6
    out["us_policy_act_gnn256x4_cpu"] = sum(timeit(lambda: act_gnn(gnn_big)(s), 1) for s in sub) / len(sub) * 1e6

    # env step overhead (AEC wrapper, obs built by last())
    env = CatanAECEnv(seed=21_300_000); env.reset(seed=21_300_000)
    rng = random.Random(0); t = time.perf_counter(); n = 0
    while time.perf_counter() - t < 10:
        if not env.agents:
            env.reset(seed=21_300_000 + n)
        obs, r, te, tr, _ = env.last()
        if te or tr:
            env.step(None); continue
        env.step(rng.randrange(len(env.legal_actions()))); n += 1
    out["aec_env_steps_per_s_with_obs"] = n / (time.perf_counter() - t)

    # cProfile of heuristic games -- top functions
    pr = cProfile.Profile(); pr.enable()
    for g in range(15):
        eng = CatanEngine(seed=21_400_000 + g)
        ag = {i: HeuristicAgent(i, random.Random(g * 4 + i)) for i in range(4)}
        while not eng.done:
            eng.step(ag[eng.acting_player()].choose(eng.state))
    pr.disable()
    sio = io.StringIO(); pstats.Stats(pr, stream=sio).sort_stats("cumulative").print_stats(25)
    out["cprofile_heuristic_top"] = sio.getvalue().splitlines()[:60]

    # cProfile of a hier-model self-play rollout (what training does)
    from training.hier_ppo import collect_episode
    from training.model_adapters import FLAT_ADAPTER, GRAPH_ADAPTER
    pr = cProfile.Profile(); pr.enable()
    t = time.perf_counter(); n_tr = 0
    for g in range(3):
        data = collect_episode(CatanAECEnv(seed=g, max_episode_steps=2500), hier, "cpu", 21_500_000 + g)
        n_tr += sum(len(v) for v in data.values())
    pr.disable()
    out["hier_rollout_transitions_per_s_1proc"] = n_tr / (time.perf_counter() - t)
    sio = io.StringIO(); pstats.Stats(pr, stream=sio).sort_stats("tottime").print_stats(20)
    out["cprofile_hier_rollout_tottime"] = sio.getvalue().splitlines()[:50]
    t = time.perf_counter(); n_tr = 0
    for g in range(1):
        data = collect_episode(CatanAECEnv(seed=g, max_episode_steps=2500), gnn_big, "cpu", 21_600_000 + g,
                               adapter=GRAPH_ADAPTER)
        n_tr += sum(len(v) for v in data.values())
    out["gnn256x4_rollout_transitions_per_s_1proc"] = n_tr / (time.perf_counter() - t)

    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({k: v for k, v in out.items() if not k.startswith("cprofile")}, indent=1))


if __name__ == "__main__":
    main()
