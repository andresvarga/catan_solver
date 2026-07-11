"""Does weighting one resource higher in vertex_production_value actually
produce a stronger heuristic agent? Cheap way to test this empirically
(heuristic-vs-heuristic, no model training) before committing to any
retraining: play a candidate-weighted HeuristicAgent against 3 unweighted
ones, seat rotated per game, and report win rate. A flat/no-op weighting
should land near the demonstrator's own baseline; if wheat/ore-weighted
variants durably beat that baseline, it's evidence the raw-pip formula is
underpricing them (see agents/heuristic.py's vertex_production_value docstring
and the road_quality.py finding that motivated this script), and gives a
concrete weighting to regenerate demonstrations with.

Usage:
  python -m scripts.resource_weight_tournament --games 400 --num-workers 12
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import random

from agents.heuristic import HeuristicAgent
from env.board import HexType
from env.engine import CatanEngine, legal_actions, total_vp
from env.state import NUM_PLAYERS

MAX_STEPS = 4000

CANDIDATES = {
    "flat (baseline sanity check)": None,
    "wheat x1.5": {HexType.WHEAT: 1.5},
    "ore x1.5": {HexType.ORE: 1.5},
    "wheat+ore x1.5": {HexType.WHEAT: 1.5, HexType.ORE: 1.5},
    "wheat+ore x2.0": {HexType.WHEAT: 2.0, HexType.ORE: 2.0},
    "wood x1.5": {HexType.WOOD: 1.5},
    "brick x1.5": {HexType.BRICK: 1.5},
    "sheep x1.5": {HexType.SHEEP: 1.5},
}


def play_game(weights: dict | None, seed: int) -> int:
    """Returns 1 if the weighted seat wins, else 0."""
    engine = CatanEngine(randomize_board=True, seed=seed)
    candidate_seat = seed % NUM_PLAYERS
    agents = {pid: HeuristicAgent(pid, random.Random(seed * 97 + pid))
              for pid in range(NUM_PLAYERS) if pid != candidate_seat}
    agents[candidate_seat] = HeuristicAgent(candidate_seat, random.Random(seed * 97 + candidate_seat),
                                             resource_weights=weights)
    steps = 0
    while not engine.done and steps < MAX_STEPS:
        actor = engine.acting_player()
        acts = legal_actions(engine.state)
        choice = agents[actor].choose(engine.state, acts)
        engine.step(choice)
        steps += 1
    vps = {pid: total_vp(engine.state, pid) for pid in range(NUM_PLAYERS)}
    winner = max(vps, key=lambda p: vps[p])
    return 1 if winner == candidate_seat else 0


_worker_weights = None


def _init_worker(weights):
    global _worker_weights
    _worker_weights = weights


def _worker_play(seeds: list[int]) -> list[int]:
    return [play_game(_worker_weights, s) for s in seeds]


def run_candidate(weights: dict | None, games: int, seed_base: int, num_workers: int) -> float:
    seeds = [seed_base + i for i in range(games)]
    nw = max(1, min(num_workers, games))
    chunks = [seeds[i::nw] for i in range(nw)]
    ctx = mp.get_context("fork")
    with ctx.Pool(processes=nw, initializer=_init_worker, initargs=(weights,)) as pool:
        results = [r for chunk in pool.map(_worker_play, chunks) for r in chunk]
    return sum(results) / len(results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=400)
    parser.add_argument("--seed", type=int, default=6_000_000)
    parser.add_argument("--num-workers", type=int, default=12)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    print(f"{args.games} games/candidate, 25.0% = parity with 3 unweighted opponents\n")
    results = {}
    for label, weights in CANDIDATES.items():
        wr = run_candidate(weights, args.games, args.seed, args.num_workers)
        results[label] = wr
        print(f"  {label:<32s} win_rate={wr:.1%}")

    if args.json:
        print(json.dumps(results))


if __name__ == "__main__":
    main()
