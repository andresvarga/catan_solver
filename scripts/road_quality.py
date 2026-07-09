"""Direct settlement/road placement quality metric -- the spatial analog of
scripts/robber_quality.py, built to answer a concrete question about a
specific model rather than wait for aggregate win rate to move: "does this
policy build toward valuable board positions, or just toward *a* legal one?"

Plays N games (evaluated agent vs 3 heuristic opponents, seat rotated per
game) and scores every BUILD_SETTLEMENT / BUILD_ROAD decision the evaluated
agent makes against the alternatives that were legal at that moment, using
the same board-value notions the heuristic demonstrator itself uses
(agents.heuristic.vertex_production_value / best_reachable_vertex_value) --
not a new preference, just a yardstick for behavior that was already supposed
to be learned from the demonstrations:

- settlement_eff: chosen vertex's production value / the best production
                  value among all legal settlement spots this decision
                  (1.0 = always takes the best available spot).
- road_eff:       chosen edge's best-reachable-vertex value (1-hop lookahead
                   from either endpoint) / the best such value among all
                   legal road options this decision (1.0 = always extends
                   toward the most promising frontier available).

Usage:
  python -m scripts.road_quality --checkpoint path.pt --model-type gnn [--public-hand-features]
  python -m scripts.road_quality --agent heuristic     # baseline reference
  python -m scripts.road_quality --agent random        # baseline floor
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import random

from agents.heuristic import HeuristicAgent, best_reachable_vertex_value, vertex_production_value
from agents.random_agent import RandomAgent
from env.actions import Action, ActionType
from env.engine import CatanEngine, legal_actions
from env.state import NUM_PLAYERS

SCORED_TYPES = (ActionType.BUILD_SETTLEMENT, ActionType.BUILD_ROAD)
MAX_STEPS = 4000


def road_frontier_value(state, edge_id: int) -> float:
    v1, v2 = state.board.edges[edge_id].vertex_ids
    return max(best_reachable_vertex_value(state, v1, max_depth=1),
               best_reachable_vertex_value(state, v2, max_depth=1))


def score_decision(state, options: list[Action], chosen: Action) -> dict:
    if chosen.type == ActionType.BUILD_SETTLEMENT:
        values = {a.params["vertex_id"]: vertex_production_value(state, a.params["vertex_id"])
                  for a in options}
        chosen_key = chosen.params["vertex_id"]
        eff_key = "settlement_eff"
    else:
        values = {a.params["edge_id"]: road_frontier_value(state, a.params["edge_id"])
                  for a in options}
        chosen_key = chosen.params["edge_id"]
        eff_key = "road_eff"

    best = max(values.values())
    rec = {"settlement_eff": None, "road_eff": None, "chosen_value": values[chosen_key]}
    if best > 0:
        rec[eff_key] = values[chosen_key] / best
    return rec


_model_cache: dict[str, object] = {}


def make_evaluated_agent(args, seat: int):
    if args.agent == "random":
        return RandomAgent(seat, random.Random(seat * 7919 + 13))
    if args.agent == "heuristic":
        return HeuristicAgent(seat, random.Random(seat * 7919 + 13))
    from training.agent import HierarchicalLearnedAgent, load_gnn_model, load_hier_model
    if args.checkpoint not in _model_cache:  # once per worker process, not per game
        if args.model_type == "gnn":
            _model_cache[args.checkpoint] = load_gnn_model(
                args.checkpoint, hidden=args.hidden, gnn_layers=args.gnn_layers,
                public_hand_features=args.public_hand_features)
        else:
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
        if actor == eval_seat and choice.type in SCORED_TYPES:
            options = [a for a in acts if a.type == choice.type]
            if len(options) > 1:  # a forced single legal option carries no signal
                records.append(score_decision(engine.state, options, choice))
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
    for key in ("settlement_eff", "road_eff"):
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
    parser.add_argument("--gnn-layers", type=int, default=3, help="only used when --model-type gnn")
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
        print(f"evaluated: {label} ({args.games} games, {summary['decisions']} scored decisions)")
        for key in ("settlement_eff", "road_eff"):
            v = summary[key]
            print(f"  {key:<14} {v:.3f}  (n={summary[f'{key}_n']})" if v is not None
                  else f"  {key:<14} n/a")


if __name__ == "__main__":
    main()
