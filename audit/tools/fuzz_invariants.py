"""Randomized invariant fuzzing (audit Phase 19).

Plays many complete games with random-legal or heuristic agents and runs
`audit.helpers.check_invariants` after every engine step. Also cross-checks
the engine's longest-road length against an independent subset-enumeration
reference at game end, and records per-game statistics. First failure of each
invariant kind is saved with seed + action index for reproduction.

    python -m audit.tools.fuzz_invariants --games 1000 --agent random --workers 14
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import random
import time
from collections import Counter

from agents.heuristic import HeuristicAgent
from agents.random_agent import RandomAgent
from audit.helpers import check_invariants, reference_longest_road
from env.engine import CatanEngine, compute_longest_road_length, legal_actions, total_vp
from env.state import Phase

MAX_STEPS = 6000


def play(seed: int, agent_kind: str, conservation: bool, lr_check: bool, relative: bool = False):
    eng = CatanEngine(randomize_board=True, seed=seed)
    cls = RandomAgent if agent_kind == "random" else HeuristicAgent
    agents = {i: cls(i, random.Random(seed * 97 + i)) for i in range(4)}
    violations: dict[str, dict] = {}
    steps = 0
    baseline = None
    type_counts = Counter()
    max_legal = 0
    while not eng.done and steps < MAX_STEPS:
        legal = eng.legal_actions()
        max_legal = max(max_legal, len(legal))
        a = agents[eng.acting_player()].choose(eng.state, legal)
        type_counts[a.type.value] += 1
        eng.step(a)
        steps += 1
        if relative:
            in_setup = eng.state.phase in (Phase.SETUP_SETTLEMENT, Phase.SETUP_ROAD)
            if in_setup:
                errs = []
                baseline = None
            else:
                if baseline is None:
                    baseline = {r: eng.state.bank[r] + sum(p.resources[r] for p in eng.state.players.values())
                                for r in eng.state.bank}
                errs = check_invariants(eng.state, strict_conservation=conservation, expected_totals=baseline)
        else:
            errs = check_invariants(eng.state, strict_conservation=conservation)
        for e in errs:
            key = e.split(":")[0].split("=")[0].strip()
            # normalize player-specific keys
            key = " ".join(w for w in key.split() if not (w.startswith("p") and w[1:2].isdigit()))
            if key not in violations:
                violations[key] = {"msg": e, "seed": seed, "step": steps, "action": repr(a)}
    lr_mismatch = []
    if lr_check:
        s = eng.state
        for pid, p in s.players.items():
            if 0 < len(p.roads) <= 15:
                blocked = set()
                for q in s.players.values():
                    if q.id != pid:
                        blocked |= set(q.settlements) | set(q.cities)
                ref = reference_longest_road(s.board, p.roads, blocked)
                got = compute_longest_road_length(s, pid)
                if ref != got:
                    lr_mismatch.append({"seed": seed, "pid": pid, "ref": ref, "engine": got,
                                        "roads": list(p.roads)})
    s = eng.state
    winner_vp = total_vp(s, s.winner) if s.winner is not None else None
    others_ge10 = [pid for pid in s.players if pid != s.winner and total_vp(s, pid) >= 10]
    return {
        "seed": seed, "done": eng.done, "steps": steps, "turns": s.turn_number,
        "winner": s.winner, "winner_vp": winner_vp,
        "winner_is_current": (s.winner == s.current_player) if s.winner is not None else None,
        "others_ge10": others_ge10,
        "violations": violations, "lr_mismatch": lr_mismatch,
        "types": dict(type_counts), "max_legal": max_legal,
        "bank_end": {r.value: k for r, k in s.bank.items()},
    }


def _worker(args):
    return play(*args)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=100)
    ap.add_argument("--agent", choices=["random", "heuristic"], default="random")
    ap.add_argument("--seed-base", type=int, default=10_000_000)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--no-conservation", action="store_true")
    ap.add_argument("--lr-check", action="store_true")
    ap.add_argument("--relative-conservation", action="store_true",
                    help="conservation measured vs post-setup totals (isolates non-setup leaks)")
    ap.add_argument("--out", type=str, required=True)
    a = ap.parse_args()
    t0 = time.time()
    jobs = [(a.seed_base + i, a.agent, not a.no_conservation, a.lr_check, a.relative_conservation) for i in range(a.games)]
    with mp.get_context("fork").Pool(a.workers) as pool:
        results = pool.map(_worker, jobs, chunksize=4)
    dt = time.time() - t0
    first: dict[str, dict] = {}
    counts = Counter()
    for r in results:
        for k, v in r["violations"].items():
            counts[k] += 1
            first.setdefault(k, v)
    lr = [m for r in results for m in r["lr_mismatch"]]
    summary = {
        "games": a.games, "agent": a.agent, "seed_base": a.seed_base, "seconds": round(dt, 1),
        "finished": sum(r["done"] for r in results),
        "total_steps": sum(r["steps"] for r in results),
        "games_with_violation_kind": dict(counts), "first_violation": first,
        "longest_road_mismatches": len(lr), "longest_road_mismatch_examples": lr[:5],
        "winner_seat": dict(Counter(r["winner"] for r in results)),
        "winner_not_current_player": sum(1 for r in results if r["winner_is_current"] is False),
        "off_turn_win_seeds": [r["seed"] for r in results if r["winner_is_current"] is False][:20],
        "timeouts": sum(1 for r in results if not r["done"]),
        "games_with_nonwinner_ge10": sum(1 for r in results if r["others_ge10"]),
        "winner_vp_hist": dict(Counter(r["winner_vp"] for r in results)),
        "max_legal_actions": max(r["max_legal"] for r in results),
        "mean_turns": sum(r["turns"] for r in results) / len(results),
        "mean_steps": sum(r["steps"] for r in results) / len(results),
        "action_type_totals": dict(sum((Counter(r["types"]) for r in results), Counter())),
        "min_bank_end": {k: min(r["bank_end"][k] for r in results) for k in results[0]["bank_end"]},
        "max_bank_end": {k: max(r["bank_end"][k] for r in results) for k in results[0]["bank_end"]},
    }
    with open(a.out, "w") as f:
        json.dump(summary, f, indent=1, default=str)
    print(json.dumps({k: summary[k] for k in summary if k not in ("action_type_totals",)},
                     indent=1, default=str))


if __name__ == "__main__":
    main()
