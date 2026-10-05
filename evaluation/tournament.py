"""Seat-balanced evaluation (roadmap Phase 3; promoted from audit/tools).

Protocol: every board seed is played four times, once with the candidate in
each seat, against three copies of one opponent style -- so seat and board
effects cancel exactly. Reports win rate with a Wilson 95% CI, per-seat rates,
average VP, game length and timeouts. Two candidates evaluated on the same
seeds can be compared *paired* (`paired_compare`): an exact McNemar test on
the per-game outcome pairs plus a bootstrap CI (resampling board seeds) for
the win-rate difference -- much tighter than comparing two independent CIs.

Agent specs:
    heuristic | honest | search | random
    ckpt:<hier|gnn>:<path>[:hidden[:gnn_layers]][:phf]     (deterministic)
    ckpt-sample:...                                          (sampled actions)
    untrained:<hier|gnn>:<torch seed>
    vsearch:<top_k>:<n_det>:<ckpt spec>   value-guided decision-time search
                                           (agents/value_search.py) around a checkpoint
"""
from __future__ import annotations

import math
import multiprocessing as mp
import random
import time
from dataclasses import dataclass

from env.engine import CatanEngine, total_vp
from env.state import NUM_PLAYERS

MAX_STEPS = 6000


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


# ---------------------------------------------------------------- agents
_MODEL_CACHE: dict[str, object] = {}


@dataclass
class _CkptSpec:
    kind: str
    path: str
    hidden: int
    layers: int
    phf: bool
    sample: bool


def _parse_ckpt(spec: str) -> _CkptSpec:
    head, kind, *rest = spec.split(":")
    phf = bool(rest) and rest[-1] == "phf"
    if phf:
        rest = rest[:-1]
    path = rest[0]
    hidden = int(rest[1]) if len(rest) > 1 else (256 if kind == "hier" else 128)
    layers = int(rest[2]) if len(rest) > 2 else 3
    return _CkptSpec(kind, path, hidden, layers, phf, head == "ckpt-sample")


def _model_for(spec: str):
    if spec in _MODEL_CACHE:
        return _MODEL_CACHE[spec]
    import torch
    if spec.startswith("untrained:"):
        _, kind, tseed = spec.split(":")
        torch.manual_seed(int(tseed))
        from training.train_hier import build_model
        model = build_model(kind, 256 if kind == "hier" else 128, 3)
    else:
        c = _parse_ckpt(spec)
        from training.agent import load_gnn_model, load_hier_model
        model = (load_gnn_model(c.path, hidden=c.hidden, gnn_layers=c.layers, public_hand_features=c.phf)
                 if c.kind == "gnn" else load_hier_model(c.path, hidden=c.hidden, public_hand_features=c.phf))
    model.eval()
    _MODEL_CACHE[spec] = model
    return model


def make_agent(spec: str, pid: int, seed: int):
    rng = random.Random(seed * 97 + pid)
    if spec == "random":
        from agents.random_agent import RandomAgent
        return RandomAgent(pid, rng)
    if spec == "heuristic":
        from agents.heuristic import HeuristicAgent
        return HeuristicAgent(pid, rng)
    if spec == "honest":
        from agents.heuristic import HonestHeuristicAgent
        return HonestHeuristicAgent(pid, rng)
    if spec == "search":
        from agents.search_heuristic import SearchHeuristicAgent
        return SearchHeuristicAgent(pid, rng)
    if spec.startswith("vsearch:"):
        from agents.value_search import ValueSearchAgent
        _, k, d, inner = spec.split(":", 3)
        c = _parse_ckpt(inner)
        return ValueSearchAgent(pid, rng, model=_model_for(inner), model_kind=c.kind,
                                public_hand_features=c.phf, top_k=int(k), n_det=int(d))
    if spec.startswith(("ckpt:", "ckpt-sample:", "untrained:")):
        from training.agent import HierarchicalLearnedAgent
        model = _model_for(spec)
        if spec.startswith("untrained:"):
            kind, sample, phf = spec.split(":")[1], False, False
        else:
            c = _parse_ckpt(spec)
            kind, sample, phf = c.kind, c.sample, c.phf
        return HierarchicalLearnedAgent(pid, rng, model=model, deterministic=not sample,
                                        model_kind=kind, public_hand_features=phf)
    raise ValueError(f"unknown agent spec {spec!r}")


# ---------------------------------------------------------------- games
def play_game(seed: int, cand_seat: int, cand: str, opp: str) -> dict:
    import torch
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed * 4 + cand_seat)  # sampled policies: reproducible per game
        eng = CatanEngine(randomize_board=True, seed=seed)
        agents = {pid: make_agent(cand if pid == cand_seat else opp, pid, seed) for pid in range(NUM_PLAYERS)}
        steps = 0
        while not eng.done and steps < MAX_STEPS:
            eng.step(agents[eng.acting_player()].choose(eng.state))
            steps += 1
    s = eng.state
    return {"seed": seed, "seat": cand_seat, "won": s.winner == cand_seat, "done": eng.done,
            "winner": s.winner, "vp": total_vp(s, cand_seat), "turns": s.turn_number, "steps": steps}


def _play_star(args):
    return play_game(*args)


def _init_worker():
    import torch
    torch.set_num_threads(1)


def run_matchup(cand: str, opp: str, seeds: list[int], workers: int = 8) -> list[dict]:
    """Candidate vs 3x `opp` on every seed from every seat. Model specs are
    loaded once in the parent so forked workers share them copy-on-write."""
    for spec in (cand, opp):
        if spec.startswith("vsearch:"):
            spec = spec.split(":", 3)[3]
        if spec.startswith(("ckpt:", "ckpt-sample:", "untrained:")):
            _model_for(spec)
    jobs = [(seed, seat, cand, opp) for seed in seeds for seat in range(NUM_PLAYERS)]
    if workers <= 1:
        return [play_game(*j) for j in jobs]
    with mp.get_context("fork").Pool(workers, initializer=_init_worker) as pool:
        return pool.map(_play_star, jobs, chunksize=2)


def summarize(records: list[dict]) -> dict:
    n = len(records)
    k = sum(r["won"] for r in records)
    per_seat = {}
    for seat in range(NUM_PLAYERS):
        rs = [r for r in records if r["seat"] == seat]
        ks = sum(r["won"] for r in rs)
        per_seat[seat] = {"n": len(rs), "wins": ks, "rate": ks / max(1, len(rs)), "ci95": wilson(ks, len(rs))}
    return {"games": n, "wins": k, "win_rate": k / n, "ci95": wilson(k, n), "parity": 1 / NUM_PLAYERS,
            "per_seat": per_seat, "avg_vp": sum(r["vp"] for r in records) / n,
            "avg_turns": sum(r["turns"] for r in records) / n,
            "timeouts": sum(not r["done"] for r in records)}


# ---------------------------------------------------------------- paired comparison
def _mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value on discordant counts b (A won, B lost)
    and c (A lost, B won)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def paired_compare(records_a: list[dict], records_b: list[dict], n_boot: int = 2000,
                   seed: int = 0) -> dict:
    """A vs B on identical (seed, seat) games against the same opponent."""
    ka = {(r["seed"], r["seat"]): r["won"] for r in records_a}
    kb = {(r["seed"], r["seat"]): r["won"] for r in records_b}
    keys = sorted(set(ka) & set(kb))
    if not keys:
        raise ValueError("no common (seed, seat) games to pair")
    b = sum(ka[k] and not kb[k] for k in keys)
    c = sum(kb[k] and not ka[k] for k in keys)
    diff = (sum(ka[k] for k in keys) - sum(kb[k] for k in keys)) / len(keys)
    by_seed: dict[int, list[int]] = {}
    for k in keys:
        by_seed.setdefault(k[0], []).append(int(ka[k]) - int(kb[k]))
    seeds = list(by_seed)
    rng = random.Random(seed)
    boots = []
    for _ in range(n_boot):
        sample = [by_seed[rng.choice(seeds)] for _ in seeds]
        tot = sum(sum(x) for x in sample)
        cnt = sum(len(x) for x in sample)
        boots.append(tot / cnt)
    boots.sort()
    return {"games": len(keys), "win_rate_a": sum(ka[k] for k in keys) / len(keys),
            "win_rate_b": sum(kb[k] for k in keys) / len(keys), "diff": diff,
            "diff_ci95": (boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot) - 1]),
            "a_only_wins": b, "b_only_wins": c, "mcnemar_p": _mcnemar_exact(b, c)}


def timed(fn, *a, **kw):
    t0 = time.time()
    out = fn(*a, **kw)
    return out, time.time() - t0
