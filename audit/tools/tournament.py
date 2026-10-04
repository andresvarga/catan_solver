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
import multiprocessing as mp
import time

from env.state import NUM_PLAYERS

from evaluation.tournament import make_agent, play_game, wilson  # promoted to evaluation/ (Phase 3)

MAX_STEPS = 6000


def play(args):
    return play_game(*args)


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
