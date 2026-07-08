"""Direct robber-decision quality metric (instead of waiting for robber play
to move aggregate win rate).

Plays N games (evaluated agent vs 3 heuristic opponents, seat rotated per
game) and scores every MOVE_ROBBER / PLAY_KNIGHT decision the evaluated
agent makes against the alternatives that were legal at that moment:

- denial_eff:      opponent production pips blocked by the chosen hex,
                   divided by the best blockable pips among all legal
                   options this decision (1.0 = always blocks the most
                   damaging hex available).
- steal_eff:       chosen victim's hand size divided by the largest hand
                   size among all eligible victims across options (steal EV
                   proxy; 0 if a victim was available but none was robbed).
- robs_leader:     fraction of decisions (among those with any eligible
                   victim) where the chosen victim is (tied for) the
                   highest-VP eligible victim.
- self_block_rate: fraction of decisions where the chosen hex also blocks
                   the evaluated agent's own production.

Usage:
  python -m scripts.robber_quality --checkpoint path.pt [--public-hand-features]
  python -m scripts.robber_quality --agent heuristic     # baseline reference
  python -m scripts.robber_quality --agent random        # baseline floor
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import random

from agents.heuristic import HeuristicAgent
from agents.random_agent import RandomAgent
from env.actions import Action, ActionType
from env.board import PIP_COUNT
from env.engine import CatanEngine, legal_actions, total_vp
from env.state import NUM_PLAYERS

ROBBER_TYPES = (ActionType.MOVE_ROBBER, ActionType.PLAY_KNIGHT)
MAX_STEPS = 4000


def blocked_pips(state, hex_id: int, exclude_player: int) -> int:
    """Production pips the robber on `hex_id` denies to players other than
    `exclude_player` (2x for cities)."""
    hx = state.board.hexes[hex_id]
    if hx.number is None:
        return 0
    pips = PIP_COUNT[hx.number]
    total = 0
    for vid in hx.vertex_ids:
        owner, kind = state.vertex_owner.get(vid, (None, None))
        if owner is not None and owner != exclude_player:
            total += pips * (2 if kind == "city" else 1)
    return total


def own_pips(state, hex_id: int, player: int) -> int:
    hx = state.board.hexes[hex_id]
    if hx.number is None:
        return 0
    pips = PIP_COUNT[hx.number]
    total = 0
    for vid in hx.vertex_ids:
        owner, kind = state.vertex_owner.get(vid, (None, None))
        if owner == player:
            total += pips * (2 if kind == "city" else 1)
    return total


def score_decision(state, actor: int, options: list[Action], chosen: Action) -> dict:
    best_block = max(blocked_pips(state, a.params["hex_id"], actor) for a in options)
    chosen_block = blocked_pips(state, chosen.params["hex_id"], actor)

    victim_hand = {}
    victim_vp = {}
    for a in options:
        v = a.params["victim"]
        if v is not None and v not in victim_hand:
            victim_hand[v] = state.players[v].hand_size()
            victim_vp[v] = total_vp(state, v)

    rec = {"denial_eff": None, "steal_eff": None, "robs_leader": None,
           "self_block": 1.0 if own_pips(state, chosen.params["hex_id"], actor) > 0 else 0.0,
           "chosen_block": chosen_block}
    if best_block > 0:
        rec["denial_eff"] = chosen_block / best_block
    if victim_hand:
        chosen_victim = chosen.params["victim"]
        max_hand = max(victim_hand.values())
        if max_hand > 0:
            rec["steal_eff"] = (victim_hand.get(chosen_victim, 0) / max_hand
                                 if chosen_victim is not None else 0.0)
        max_vp = max(victim_vp.values())
        rec["robs_leader"] = (1.0 if chosen_victim is not None
                              and victim_vp[chosen_victim] == max_vp else 0.0)
    return rec


_model_cache: dict[str, object] = {}


def make_evaluated_agent(args, seat: int):
    if args.agent == "random":
        return RandomAgent(seat, random.Random(seat * 7919 + 13))
    if args.agent == "heuristic":
        return HeuristicAgent(seat, random.Random(seat * 7919 + 13))
    from training.agent import HierarchicalLearnedAgent, load_hier_model
    if args.checkpoint not in _model_cache:  # once per worker process, not per game
        _model_cache[args.checkpoint] = load_hier_model(
            args.checkpoint, hidden=args.hidden,
            public_hand_features=args.public_hand_features)
    return HierarchicalLearnedAgent(seat, model=_model_cache[args.checkpoint], deterministic=True,
                                     model_kind=args.model_type,
                                     public_hand_features=args.public_hand_features)


def play_game(args, seed: int) -> list[dict]:
    engine = CatanEngine(randomize_board=True, seed=seed)
    eval_seat = seed % NUM_PLAYERS
    agents = {pid: HeuristicAgent(pid, random.Random(seed * 97 + pid))
              for pid in range(NUM_PLAYERS) if pid != eval_seat}
    agents[eval_seat] = make_evaluated_agent(args, eval_seat)

    records = []
    steps = 0
    while not engine.done and steps < MAX_STEPS:
        actor = engine.acting_player()
        acts = legal_actions(engine.state)
        choice = agents[actor].choose(engine.state, acts)
        if actor == eval_seat and choice.type in ROBBER_TYPES:
            options = [a for a in acts if a.type == choice.type]
            records.append(score_decision(engine.state, actor, options, choice))
        engine.step(choice)
        steps += 1
    return records


_worker_args = None


def _init_worker(args):
    global _worker_args
    import torch
    torch.set_num_threads(1)
    _worker_args = args


def _worker_play(seeds: list[int]) -> list[dict]:
    out = []
    for seed in seeds:
        out.extend(play_game(_worker_args, seed))
    return out


def aggregate(records: list[dict]) -> dict:
    def mean_of(key):
        vals = [r[key] for r in records if r[key] is not None]
        return (sum(vals) / len(vals), len(vals)) if vals else (None, 0)

    out = {"decisions": len(records)}
    for key in ("denial_eff", "steal_eff", "robs_leader", "self_block", "chosen_block"):
        m, n = mean_of(key)
        out[key] = m
        out[f"{key}_n"] = n
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--agent", type=str, default=None, choices=["random", "heuristic"],
                         help="evaluate a fixed agent instead of a checkpoint (baseline)")
    parser.add_argument("--model-type", choices=["hier", "gnn"], default="hier")
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--public-hand-features", action="store_true")
    parser.add_argument("--games", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args()
    assert (args.checkpoint is None) != (args.agent is None), \
        "pass exactly one of --checkpoint / --agent"

    seeds = [args.seed + i for i in range(args.games)]
    if args.num_workers <= 1:
        _init_worker(args)
        records = _worker_play(seeds)
    else:
        nw = max(1, min(args.num_workers, args.games))
        chunks = [seeds[i::nw] for i in range(nw)]
        ctx = mp.get_context("fork")
        with ctx.Pool(processes=nw, initializer=_init_worker, initargs=(args,)) as pool:
            records = [r for chunk in pool.map(_worker_play, chunks) for r in chunk]

    summary = aggregate(records)
    label = args.agent or args.checkpoint
    if args.json:
        print(json.dumps({"label": label, **summary}))
    else:
        print(f"evaluated: {label} ({args.games} games, {summary['decisions']} robber decisions)")
        for key in ("denial_eff", "steal_eff", "robs_leader", "self_block", "chosen_block"):
            v = summary[key]
            print(f"  {key:<14} {v:.3f}  (n={summary[f'{key}_n']})" if v is not None
                  else f"  {key:<14} n/a")


if __name__ == "__main__":
    main()
