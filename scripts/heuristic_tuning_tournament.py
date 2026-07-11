"""Generalizes resource_weight_tournament.py's method (race a modified
HeuristicAgent against 3 unmodified ones, let win rate -- not guesswork --
decide) to the heuristic's other hand-picked, never-tuned constants:
ROAD_VALUE_THRESHOLD, KNIGHT_VALUE_THRESHOLD, LONGEST_ROAD_PUSH_BONUS,
MONOPOLY_MIN_HAUL (agents/heuristic.py). Every candidate is evaluated on top
of the already-confirmed ore x1.5 resource weighting (see
resource_weight_tournament.py's result and the DAgger round it produced), not
against the plain baseline, since that's the recipe actually in use now and
these constants may interact with it rather than being independent.

Usage:
  python -m scripts.heuristic_tuning_tournament --games 5000 --num-workers 12
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import random

from agents.heuristic import (
    HeuristicAgent, KNIGHT_VALUE_THRESHOLD, LONGEST_ROAD_PUSH_BONUS,
    MONOPOLY_MIN_HAUL, ROAD_VALUE_THRESHOLD,
)
from env.board import HexType
from env.engine import CatanEngine, legal_actions, total_vp
from env.state import NUM_PLAYERS

MAX_STEPS = 4000
ORE_WEIGHT = {HexType.ORE: 1.5}  # the confirmed-good baseline every candidate builds on

BASE_KWARGS = dict(resource_weights=ORE_WEIGHT)

CANDIDATES = {
    "flat (no ore weight, sanity floor)": dict(resource_weights=None),
    "ore x1.5 (current champion recipe)": dict(BASE_KWARGS),
    "road_threshold=3.0 (more permissive)": dict(BASE_KWARGS, road_value_threshold=3.0),
    "road_threshold=7.0 (pickier)": dict(BASE_KWARGS, road_value_threshold=7.0),
    "knight_threshold=3.0 (plays knights more)": dict(BASE_KWARGS, knight_value_threshold=3.0),
    "knight_threshold=9.0 (plays knights less)": dict(BASE_KWARGS, knight_value_threshold=9.0),
    "longest_road_bonus=0.0 (ignore LR race)": dict(BASE_KWARGS, longest_road_push_bonus=0.0),
    "longest_road_bonus=3.0 (push LR harder)": dict(BASE_KWARGS, longest_road_push_bonus=3.0),
    "monopoly_min_haul=1 (plays monopoly more)": dict(BASE_KWARGS, monopoly_min_haul=1),
    "monopoly_min_haul=5 (plays monopoly less)": dict(BASE_KWARGS, monopoly_min_haul=5),
}


def play_game(kwargs: dict, seed: int) -> int:
    """Returns 1 if the candidate seat wins, else 0."""
    engine = CatanEngine(randomize_board=True, seed=seed)
    candidate_seat = seed % NUM_PLAYERS
    agents = {pid: HeuristicAgent(pid, random.Random(seed * 97 + pid))
              for pid in range(NUM_PLAYERS) if pid != candidate_seat}
    agents[candidate_seat] = HeuristicAgent(candidate_seat, random.Random(seed * 97 + candidate_seat),
                                             **kwargs)
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


_worker_kwargs = None


def _init_worker(kwargs):
    global _worker_kwargs
    _worker_kwargs = kwargs


def _worker_play(seeds: list[int]) -> list[int]:
    return [play_game(_worker_kwargs, s) for s in seeds]


def run_candidate(kwargs: dict, games: int, seed_base: int, num_workers: int) -> float:
    seeds = [seed_base + i for i in range(games)]
    nw = max(1, min(num_workers, games))
    chunks = [seeds[i::nw] for i in range(nw)]
    ctx = mp.get_context("fork")
    with ctx.Pool(processes=nw, initializer=_init_worker, initargs=(kwargs,)) as pool:
        results = [r for chunk in pool.map(_worker_play, chunks) for r in chunk]
    return sum(results) / len(results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=7_000_000)
    parser.add_argument("--num-workers", type=int, default=12)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    print(f"{args.games} games/candidate, all seats otherwise plain HeuristicAgent, "
          f"same seed set reused per candidate (paired comparison)\n")
    results = {}
    for label, kwargs in CANDIDATES.items():
        wr = run_candidate(kwargs, args.games, args.seed, args.num_workers)
        results[label] = wr
        print(f"  {label:<40s} win_rate={wr:.1%}")

    if args.json:
        print(json.dumps(results))


if __name__ == "__main__":
    main()
