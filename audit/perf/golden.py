"""Golden behavioural fingerprints for the simulation-performance work.

Recorded once from the pre-optimization code (`python -m audit.perf.golden --record`) and
re-checked after every optimization (`audit/tests/test_perf_golden.py`). Three layers:

1. engine games (seeded random agents, incl. structured trades): rolling digest of the FULL
   game state (every GameState field except the debug `turn_log`) + the ordered legal-action
   list after every step, with checkpoint digests every 50 steps to localise divergences;
2. AEC env episodes: every observation the acting agent sees (flat dict, flattened vector,
   graph features, with and without public-hand features), rewards, terminations;
3. rollout output of `collect_rollout_parallel` (flat + GNN models, self-play and vs the
   mixed opponent pool): every transition field. Discrete fields, observations and rewards
   must match exactly; log-prob / value / advantage / return within 1e-5 (BLAS batching).
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import random

import numpy as np

GOLDEN_PATH = os.path.join(os.path.dirname(__file__), "..", "results", "perf", "golden.json")


def _h(*parts) -> str:
    m = hashlib.sha1()
    for p in parts:
        if isinstance(p, np.ndarray):
            m.update(str(p.dtype).encode() + str(p.shape).encode())
            m.update(np.ascontiguousarray(p).tobytes())
        elif isinstance(p, (bytes, bytearray)):
            m.update(p)
        else:
            m.update(repr(p).encode())
    return m.hexdigest()


def state_repr(state) -> str:
    d = {f.name: getattr(state, f.name) for f in dataclasses.fields(state) if f.name != "turn_log"}
    d["board"] = (state.board.robber_hex,
                  [(h.terrain, h.number) for h in state.board.hexes.values()],
                  [(v.port, v.port_generic) for v in state.board.vertices.values()])
    return repr(d)


# ---------------------------------------------------------------- 1. engine games
def engine_trace(seed: int) -> dict:
    from agents.random_agent import RandomAgent
    from env.engine import CatanEngine, legal_actions
    eng = CatanEngine(randomize_board=True, seed=seed)
    agents = {i: RandomAgent(i, random.Random(seed * 97 + i)) for i in range(4)}
    rolling, checkpoints, steps = "", [], 0
    while not eng.done and steps < 6000:
        legal = eng.legal_actions()
        eng.step(agents[eng.acting_player()].choose(eng.state, legal))
        steps += 1
        rolling = _h(rolling, state_repr(eng.state), [repr(a) for a in legal_actions(eng.state)])
        if steps % 50 == 0:
            checkpoints.append(rolling)
    return {"seed": seed, "steps": steps, "winner": eng.state.winner, "digest": rolling,
            "checkpoints": checkpoints}


# ---------------------------------------------------------------- 2. env episodes
def env_trace(seed: int, phf: bool) -> dict:
    from env.engine import is_template, random_trade
    from env.pettingzoo_env import CatanAECEnv
    from training.graph_features import build_graph_observation
    from training.model import flatten_observation
    env = CatanAECEnv(seed=seed, public_hand_features=phf)
    env.reset(seed=seed)
    rng = random.Random(seed)
    rolling, checkpoints, steps = "", [], 0
    while env.agents and steps < 6000:
        agent = env.agent_selection
        obs, r, term, trunc, _ = env.last()
        env.clear_reward(agent)
        pid = int(agent.split("_")[1])
        g = build_graph_observation(env.engine.state, pid, public_hand_features=phf)
        rolling = _h(rolling, agent, r, term, trunc,
                     *[obs[k] for k in sorted(obs)], flatten_observation(obs), *[g[k] for k in sorted(g)])
        if term or trunc:
            env.step(None)
            continue
        legal = env.legal_actions()
        i = rng.randrange(len(legal))
        a = legal[i]
        env.step(random_trade(a, rng, actor=a.params["actor"]) if is_template(a) else i)
        steps += 1
        if steps % 50 == 0:
            checkpoints.append(rolling)
    return {"seed": seed, "phf": phf, "steps": steps, "digest": rolling, "checkpoints": checkpoints}


# ---------------------------------------------------------------- 3. rollouts
def _rollout_cfg(name: str):
    import torch
    from agents.opponent_pool import builtin_members
    from training.hier_ppo import RotatingOpponents
    from training.model_adapters import ADAPTERS
    from training.train_hier import build_model
    from training.rl_finetune import _pooled_opponent
    from functools import partial
    torch.manual_seed(0)
    if name == "hier_selfplay":
        return build_model("hier", 64, 1), ADAPTERS["hier"], None, False
    if name == "hier_pool":
        return (build_model("hier", 64, 1), ADAPTERS["hier"],
                RotatingOpponents(partial(_pooled_opponent, builtin_members())), False)
    if name == "gnn_selfplay_phf":
        return build_model("gnn", 32, 2, public_hand_features=True), ADAPTERS["gnn"], None, True
    raise KeyError(name)


ROLLOUTS = {"hier_selfplay": (6, 300_000), "hier_pool": (8, 300_100), "gnn_selfplay_phf": (3, 300_200)}


def rollout_trace(name: str, workers: int = 1, envs_per_worker: int = 1, device: str = "cpu") -> dict:
    from training.hier_ppo import collect_rollout_parallel
    model, adapter, opponents, phf = _rollout_cfg(name)
    n, base = ROLLOUTS[name]
    env_kwargs = dict(randomize_board=True, max_episode_steps=4000, public_hand_features=phf)
    if device != "cpu" and workers <= 1:
        model = model.to(device)
    trs, summ = collect_rollout_parallel(env_kwargs, model, n, base, workers, opponent_agents=opponents,
                                         adapter=adapter, envs_per_worker=envs_per_worker,
                                         inference_device=device)
    exact, numeric = [], []
    for t in trs:
        obs = t["obs"]
        obs_parts = [obs[k] for k in sorted(obs)] if isinstance(obs, dict) else [obs]
        exact.append(_h(*obs_parts, t["type_mask"], t["type_idx"], t["stage1_head"], t["stage2_head"],
                        t["sub_mask_1"], t["sub_idx_1"], t["sub_mask_2"], t["sub_idx_2"],
                        t["trade_counts"], t["trade_masks"], t["reward"], t["done"],
                        t["terminated"], t["truncated"]))
        numeric.append([t["logprob"], t["value"], t["advantage"], t["return"],
                        t.get("bootstrap_value", 0.0)])
    return {"name": name, "transitions": len(trs), "exact": _h(*exact),
            "numeric": numeric, "summaries": [(s["seed"], s["winner"], s["turns"]) for s in summ]}


ENGINE_SEEDS = list(range(70_000_000, 70_000_016))
ENV_SEEDS = [(70_100_000, False), (70_100_001, True), (70_100_002, True), (70_100_003, False)]


def record_all() -> dict:
    return {"engine": [engine_trace(s) for s in ENGINE_SEEDS],
            "env": [env_trace(s, p) for s, p in ENV_SEEDS],
            "rollouts": [rollout_trace(n) for n in ROLLOUTS]}


def compare_rollout(gold: dict, got: dict, tol: float = 1e-5) -> list[str]:
    errs = []
    if gold["transitions"] != got["transitions"]:
        errs.append(f"{gold['name']}: transitions {got['transitions']} != {gold['transitions']}")
    if gold["exact"] != got["exact"]:
        errs.append(f"{gold['name']}: exact-field digest differs")
    if gold["summaries"] != [list(x) for x in got["summaries"]] and gold["summaries"] != got["summaries"]:
        errs.append(f"{gold['name']}: episode summaries differ")
    a, b = np.array(gold["numeric"]), np.array(got["numeric"])
    if a.shape != b.shape or not np.allclose(a, b, atol=tol, rtol=0):
        errs.append(f"{gold['name']}: numeric fields differ (max {np.abs(a - b).max() if a.shape == b.shape else 'shape'})")
    return errs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", action="store_true")
    a = ap.parse_args()
    data = record_all()
    if a.record:
        os.makedirs(os.path.dirname(GOLDEN_PATH), exist_ok=True)
        with open(GOLDEN_PATH, "w") as f:
            json.dump(data, f)
        print(f"recorded {GOLDEN_PATH}: {len(data['engine'])} engine games, {len(data['env'])} env "
              f"episodes, rollouts {[r['transitions'] for r in data['rollouts']]}")


if __name__ == "__main__":
    main()
