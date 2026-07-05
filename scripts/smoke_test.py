"""Run full random-vs-random games end to end to shake out engine bugs before
anything RL-shaped gets built on top. Not a statistical benchmark -- a crash
detector, capped at MAX_STEPS per game so an engine bug that produces an
infinite loop fails loudly instead of hanging."""
from __future__ import annotations

import argparse
import random
import sys
import time
import traceback

from agents.random_agent import choose
from env.engine import CatanEngine
from env.state import Phase

MAX_STEPS = 4000


def run_one(seed: int) -> dict:
    engine = CatanEngine(randomize_board=True, seed=seed)
    rng = random.Random(seed)
    steps = 0
    action_counts: dict[str, int] = {}
    while not engine.done and steps < MAX_STEPS:
        action = choose(engine.state, rng)
        action_counts[action.type.value] = action_counts.get(action.type.value, 0) + 1
        engine.step(action)
        steps += 1
    return {
        "seed": seed,
        "steps": steps,
        "winner": engine.state.winner,
        "turn_number": engine.state.turn_number,
        "timed_out": not engine.done,
        "action_counts": action_counts,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    results = []
    start = time.time()
    for seed in range(args.games):
        try:
            result = run_one(seed)
        except Exception:
            print(f"CRASH on seed {seed}")
            traceback.print_exc()
            sys.exit(1)
        results.append(result)
        if args.verbose:
            print(result["seed"], result["steps"], result["winner"], result["timed_out"])

    elapsed = time.time() - start
    timeouts = [r for r in results if r["timed_out"]]
    steps = [r["steps"] for r in results]
    print(f"\n{len(results)} games in {elapsed:.1f}s ({len(results) / elapsed:.1f} games/s)")
    print(f"timeouts: {len(timeouts)}")
    print(f"steps: min={min(steps)} max={max(steps)} mean={sum(steps) / len(steps):.0f}")
    winners = [r["winner"] for r in results if r["winner"] is not None]
    from collections import Counter
    print(f"winner distribution: {dict(Counter(winners))}")


if __name__ == "__main__":
    main()
