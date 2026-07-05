"""Tournament runner: seat any mix of agent kinds and report win rate, average
VP, and turns-to-win (§12 of the design doc). Used here to validate that the
Phase 3 heuristic agent is a genuine step up from random play, and later as
the general evaluation harness for trained checkpoints against fixed tiers.
"""
from __future__ import annotations

import argparse
import random
import time
from collections import Counter

from agents.heuristic import HeuristicAgent
from agents.random_agent import RandomAgent
from env.engine import CatanEngine, total_vp
from env.state import NUM_PLAYERS

AGENT_FACTORIES = {"random": RandomAgent, "heuristic": HeuristicAgent}
MAX_STEPS = 4000


def run_match(seed: int, seat_kinds: list[str], max_steps: int = MAX_STEPS):
    engine = CatanEngine(randomize_board=True, seed=seed)
    agents = {i: AGENT_FACTORIES[kind](i, random.Random(seed * 97 + i))
              for i, kind in enumerate(seat_kinds)}
    steps = 0
    while not engine.done and steps < max_steps:
        actor = engine.acting_player()
        action = agents[actor].choose(engine.state)
        engine.step(action)
        steps += 1
    return engine, steps


def run_tournament(seat_kinds: list[str], games: int, base_seed: int = 0,
                    max_steps: int = MAX_STEPS) -> list[dict]:
    records = []
    for i in range(games):
        seed = base_seed + i
        engine, steps = run_match(seed, seat_kinds, max_steps)
        records.append({
            "seed": seed,
            "winner": engine.state.winner,
            "timed_out": not engine.done,
            "turns": engine.state.turn_number,
            "steps": steps,
            "vp": {pid: total_vp(engine.state, pid) for pid in range(NUM_PLAYERS)},
        })
    return records


def summarize(records: list[dict], seat_kinds: list[str]) -> dict:
    per_kind_games = Counter()
    per_kind_wins = Counter()
    per_kind_vp_sum = Counter()
    for r in records:
        for pid, kind in enumerate(seat_kinds):
            per_kind_games[kind] += 1
            per_kind_vp_sum[kind] += r["vp"][pid]
        if r["winner"] is not None:
            per_kind_wins[seat_kinds[r["winner"]]] += 1

    winning_turns = [r["turns"] for r in records if r["winner"] is not None]
    return {
        "games": len(records),
        "timeouts": sum(1 for r in records if r["timed_out"]),
        "win_rate": {kind: per_kind_wins[kind] / per_kind_games[kind] for kind in per_kind_games},
        "avg_vp": {kind: per_kind_vp_sum[kind] / per_kind_games[kind] for kind in per_kind_games},
        "avg_turns_to_finish": sum(winning_turns) / len(winning_turns) if winning_turns else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seats", type=str, default="heuristic,random,random,random",
                         help="comma-separated agent kind per seat, e.g. heuristic,random,random,random")
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    seat_kinds = args.seats.split(",")
    assert len(seat_kinds) == NUM_PLAYERS
    assert all(k in AGENT_FACTORIES for k in seat_kinds)

    start = time.time()
    records = run_tournament(seat_kinds, args.games, base_seed=args.seed)
    elapsed = time.time() - start
    summary = summarize(records, seat_kinds)

    print(f"seats: {seat_kinds}")
    print(f"{summary['games']} games in {elapsed:.1f}s "
          f"({summary['games'] / elapsed:.1f} games/s), timeouts: {summary['timeouts']}")
    print(f"avg turns to finish: {summary['avg_turns_to_finish']:.1f}"
          if summary["avg_turns_to_finish"] else "no games finished")
    print("\nkind       win_rate   avg_vp")
    for kind in summary["win_rate"]:
        print(f"{kind:<10} {summary['win_rate'][kind]:>7.1%}   {summary['avg_vp'][kind]:>5.2f}")


if __name__ == "__main__":
    main()
