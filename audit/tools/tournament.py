"""Seat-balanced 1-vs-3 tournament (audit Phases 25, 29-31).

Each board seed is played 4 times, once with the candidate in each seat, so
seat and board effects cancel exactly. Reports overall + per-seat win rates
with Wilson 95% CIs, avg VP, game length and timeouts.

    python -m audit.tools.tournament --candidate heuristic --opponent random --seeds 250
    python -m audit.tools.tournament --candidate ckpt:hier:path.pt --opponent heuristic
    python -m audit.tools.tournament --candidate heuristic --opponent heuristic   # seat bias
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import random
import time

from agents.heuristic import HeuristicAgent
from agents.random_agent import RandomAgent
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


class HonestHeuristic(HeuristicAgent):
    """HeuristicAgent with its two hidden-information reads replaced by
    public-information equivalents (audit control for F-09)."""

    def _robber_score(self, state, action):
        from agents.heuristic import hex_pip
        hex_id, victim = action.params["hex_id"], action.params["victim"]
        if victim is None:
            return -1.0
        p = state.players[victim]
        vis = p.visible_vp() + 2 * (state.longest_road_holder == victim) + 2 * (state.largest_army_holder == victim)
        return hex_pip(state, hex_id) * (1.0 + 0.3 * vis)

    def _maybe_play_monopoly(self, state, mono_actions):
        if not mono_actions:
            return None
        _, missing = self._target(state)
        best, best_haul = None, 0
        for a in mono_actions:
            r = a.params["resource"]
            haul = sum(state.public_resource_estimates[pid][r] for pid in state.players if pid != self.player_id)
            if haul > best_haul:
                best, best_haul = a, haul
        if best is not None and best_haul >= self.monopoly_min_haul and missing.get(best.params["resource"], 0) > 0:
            return best
        return None


_MODEL_CACHE = {}


def make_agent(spec: str, pid: int, seed: int):
    rng = random.Random(seed * 97 + pid)
    if spec == "random":
        return RandomAgent(pid, rng)
    if spec == "heuristic":
        return HeuristicAgent(pid, rng)
    if spec == "honest":
        return HonestHeuristic(pid, rng)
    if spec == "search":
        from agents.search_heuristic import SearchHeuristicAgent
        return SearchHeuristicAgent(pid, rng)
    if spec.startswith("ckpt:") or spec.startswith("untrained:"):
        from training.agent import HierarchicalLearnedAgent
        kind, mk, *rest = spec.split(":")
        path = ":".join(rest)
        key = spec
        if key not in _MODEL_CACHE:
            import torch
            torch.manual_seed(int(path) if kind == "untrained" and path.isdigit() else 0)
            if kind == "untrained":
                from training.train_hier import build_model
                m = build_model(mk, 256 if mk == "hier" else 128, 3)
            else:
                from training.agent import load_gnn_model, load_hier_model
                m = load_hier_model(path) if mk == "hier" else load_gnn_model(path)
            m.eval()
            _MODEL_CACHE[key] = m
        sampled = spec.endswith("#sample")
        return HierarchicalLearnedAgent(pid, rng, model=_MODEL_CACHE[key], deterministic=not sampled,
                                        model_kind=mk)
    raise ValueError(spec)


def play(args):
    seed, cand_seat, cand, opp = args
    eng = CatanEngine(randomize_board=True, seed=seed)
    agents = {pid: make_agent(cand if pid == cand_seat else opp, pid, seed) for pid in range(NUM_PLAYERS)}
    steps = 0
    while not eng.done and steps < MAX_STEPS:
        a = agents[eng.acting_player()].choose(eng.state)
        eng.step(a)
        steps += 1
    s = eng.state
    return {"seed": seed, "seat": cand_seat, "won": s.winner == cand_seat, "done": eng.done,
            "winner": s.winner, "vp": total_vp(s, cand_seat), "turns": s.turn_number, "steps": steps}


def _init():
    import torch
    torch.set_num_threads(1)


def run(cand: str, opp: str, seeds: int, seed_base: int, workers: int):
    jobs = [(seed_base + i, seat, cand, opp) for i in range(seeds) for seat in range(NUM_PLAYERS)]
    t0 = time.time()
    with mp.get_context("fork").Pool(workers, initializer=_init) as pool:
        res = pool.map(play, jobs, chunksize=2)
    dt = time.time() - t0
    n = len(res)
    k = sum(r["won"] for r in res)
    per_seat = {}
    for seat in range(NUM_PLAYERS):
        rs = [r for r in res if r["seat"] == seat]
        ks = sum(r["won"] for r in rs)
        per_seat[seat] = {"n": len(rs), "wins": ks, "rate": ks / len(rs), "ci95": wilson(ks, len(rs))}
    # overall winner-seat distribution (all games, any agent)
    win_seat = {s: sum(1 for r in res if r["winner"] == s) for s in range(NUM_PLAYERS)}
    return {
        "candidate": cand, "opponent": opp, "games": n, "seed_base": seed_base, "seeds": seeds,
        "seconds": round(dt, 1), "win_rate": k / n, "ci95": wilson(k, n),
        "parity": 0.25, "per_seat": per_seat, "winner_seat_all_games": win_seat,
        "avg_vp": sum(r["vp"] for r in res) / n,
        "avg_turns": sum(r["turns"] for r in res) / n,
        "timeouts": sum(not r["done"] for r in res),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--opponent", required=True)
    ap.add_argument("--seeds", type=int, default=250, help="board seeds; games = 4 x seeds")
    ap.add_argument("--seed-base", type=int, default=20_000_000)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--out", type=str, default=None)
    a = ap.parse_args()
    r = run(a.candidate, a.opponent, a.seeds, a.seed_base, a.workers)
    txt = json.dumps(r, indent=1)
    print(txt)
    if a.out:
        with open(a.out, "w") as f:
            f.write(txt)


if __name__ == "__main__":
    main()
