"""Collect (observation, action) demonstrations from heuristic-vs-heuristic
self-play (roadmap §9's "later an imitation-learning demonstration source"),
for behavior-cloning a policy that starts near heuristic-competent instead of
random-competent -- the wall a pure-RL trainee has been unable to climb
because losing ~100% of games against heuristic gives PPO almost no gradient
to work with.

Each of the 4 heuristic seats' decisions is recorded (they're all
heuristic, so every seat is a valid demonstration), encoded into exactly the
head_data shape `HierarchicalActorCritic.act()` produces -- so a
demonstration and a self-generated rollout transition are interchangeable to
`evaluate_actions`/`prepare_transition_batch`. Decisions with only one legal
action are skipped (zero signal; inference-time agents never query the model
for these either).
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import random
import time

import numpy as np

from agents.heuristic import HeuristicAgent
from env.engine import CatanEngine, legal_actions
from env.pettingzoo_env import build_observation
from env.state import NUM_PLAYERS
from training.graph_features import build_graph_observation
from training.hier_model import (
    ACTION_TYPE_INDEX, HEAD_NAME_INDEX, NO_TRADE_COUNTS, NO_TRADE_MASKS, NUM_ACTION_TYPES,
    SUBMASK_PAD, TRADE_TYPES, TYPE_TO_HEADS, _pad, action_to_indices, group_by_type, stage1_mask,
    stage2_mask, trade_head_data,
)
from training.imitation_data import OBS_PREFIX
from training.model import flatten_observation

MAX_STEPS = 4000


def encode_decision(state, actor: int, acts, chosen, public_hand_features: bool,
                     model_type: str = "hier") -> dict:
    """Encode one (state, demonstrated action) pair into the head_data-shaped
    supervised target `evaluate_actions`/`prepare_transition_batch` consume.
    Shared by heuristic-self-play collection and DAgger collection (where the
    demonstrated action comes from the expert but the state came from the
    learner's own rollout). The label fields (everything but `obs`) don't
    depend on which encoder `model_type` selects -- they come from the legal
    actions and the demonstrated action, not the observation."""
    by_type = group_by_type(acts)
    actions_of_type = by_type[chosen.type]
    stage1_head, stage2_head = TYPE_TO_HEADS[chosen.type]
    idx1, idx2 = action_to_indices(chosen.type, actions_of_type, chosen)

    type_mask = np.zeros(NUM_ACTION_TYPES, dtype=np.float32)
    for t in by_type:
        type_mask[ACTION_TYPE_INDEX[t]] = 1.0
    mask1 = _pad(stage1_mask(chosen.type, actions_of_type)) if stage1_head else np.zeros(SUBMASK_PAD, dtype=np.float32)
    mask2 = (_pad(stage2_mask(chosen.type, actions_of_type, idx1)) if stage2_head
             else np.zeros(SUBMASK_PAD, dtype=np.float32))
    # structured trades: the demonstrated concrete bundle vs the listed template
    trade_counts, trade_masks = (trade_head_data(actions_of_type[0], chosen) if chosen.type in TRADE_TYPES
                                 else (NO_TRADE_COUNTS, NO_TRADE_MASKS))

    if model_type == "gnn":
        obs = build_graph_observation(state, actor, public_hand_features=public_hand_features)
    else:
        obs_dict = build_observation(state, actor, acts, show_mask=True,
                                      public_hand_features=public_hand_features)
        obs = flatten_observation(obs_dict)
    return {
        "obs": obs,
        "type_mask": type_mask,
        "type_idx": ACTION_TYPE_INDEX[chosen.type],
        "head1_id": HEAD_NAME_INDEX[stage1_head] if stage1_head else -1,
        "sub_mask_1": mask1,
        "sub_idx_1": idx1 if idx1 is not None else -1,
        "head2_id": HEAD_NAME_INDEX[stage2_head] if stage2_head else -1,
        "sub_mask_2": mask2,
        "sub_idx_2": idx2 if idx2 is not None else -1,
        "trade_counts": trade_counts,
        "trade_masks": trade_masks,
    }


def records_to_arrays(records: list[dict], model_type: str = "hier") -> dict[str, np.ndarray]:
    if model_type == "gnn":
        obs_arrays = {f"{OBS_PREFIX}{k}": np.stack([r["obs"][k] for r in records]).astype(np.float32)
                      for k in records[0]["obs"]}
    else:
        obs_arrays = {"obs": np.stack([r["obs"] for r in records]).astype(np.float32)}
    return {
        **obs_arrays,
        "type_mask": np.stack([r["type_mask"] for r in records]).astype(np.float32),
        "type_idx": np.array([r["type_idx"] for r in records], dtype=np.int64),
        "head1_id": np.array([r["head1_id"] for r in records], dtype=np.int64),
        "sub_mask_1": np.stack([r["sub_mask_1"] for r in records]).astype(np.float32),
        "sub_idx_1": np.array([r["sub_idx_1"] for r in records], dtype=np.int64),
        "head2_id": np.array([r["head2_id"] for r in records], dtype=np.int64),
        "sub_mask_2": np.stack([r["sub_mask_2"] for r in records]).astype(np.float32),
        "sub_idx_2": np.array([r["sub_idx_2"] for r in records], dtype=np.int64),
        "trade_counts": np.stack([r["trade_counts"] for r in records]).astype(np.int64),
        "trade_masks": np.stack([r["trade_masks"] for r in records]).astype(np.float32),
    }


def play_and_record(seed: int, public_hand_features: bool, model_type: str = "hier",
                     resource_weights: dict | None = None) -> list[dict]:
    engine = CatanEngine(randomize_board=True, seed=seed)
    agents = {i: HeuristicAgent(i, random.Random(seed * 97 + i), resource_weights=resource_weights)
              for i in range(NUM_PLAYERS)}
    records = []
    steps = 0
    while not engine.done and steps < MAX_STEPS:
        state = engine.state
        actor = engine.acting_player()
        acts = legal_actions(state)
        chosen = agents[actor].choose(state, acts)
        if len(acts) > 1:
            records.append(encode_decision(state, actor, acts, chosen, public_hand_features, model_type))
        engine.step(chosen)
        steps += 1
    return records


_worker_phf = False
_worker_model_type = "hier"
_worker_resource_weights = None


def _init_worker(public_hand_features: bool, model_type: str = "hier",
                  resource_weights: dict | None = None) -> None:
    global _worker_phf, _worker_model_type, _worker_resource_weights
    _worker_phf = public_hand_features
    _worker_model_type = model_type
    _worker_resource_weights = resource_weights


def _worker_collect(seeds: list[int]) -> list[dict]:
    out = []
    for seed in seeds:
        out.extend(play_and_record(seed, _worker_phf, _worker_model_type, _worker_resource_weights))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=12)
    parser.add_argument("--public-hand-features", action="store_true")
    parser.add_argument("--model-type", choices=["hier", "gnn"], default="hier",
                         help="observation encoding to record: 'hier' (flat vector) or "
                              "'gnn' (graph-encoded board, see training/graph_features.py)")
    parser.add_argument("--ore-weight", type=float, default=1.0,
                         help="resource_weights={ORE: this} for all 4 self-play seats (see "
                              "agents/heuristic.py) -- this dataset has no opponent/demonstrator "
                              "split, every seat's decisions are recorded as demonstrations.")
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    from env.board import HexType
    resource_weights = {HexType.ORE: args.ore_weight} if args.ore_weight != 1.0 else None

    seeds = list(range(args.seed, args.seed + args.games))
    t0 = time.time()
    if args.num_workers <= 1:
        _init_worker(args.public_hand_features, args.model_type, resource_weights)
        records = _worker_collect(seeds)
    else:
        nw = max(1, min(args.num_workers, args.games))
        chunks = [seeds[i::nw] for i in range(nw)]
        ctx = mp.get_context("fork")
        with ctx.Pool(processes=nw, initializer=_init_worker,
                      initargs=(args.public_hand_features, args.model_type, resource_weights)) as pool:
            results = pool.map(_worker_collect, chunks)
        records = [r for chunk in results for r in chunk]
    elapsed = time.time() - t0

    n = len(records)
    print(f"{args.games} games -> {n} decisions in {elapsed:.1f}s ({n/elapsed:.0f} decisions/s)")

    arrays = records_to_arrays(records, args.model_type)
    np.savez(args.out, **arrays)
    size_mb = sum(a.nbytes for a in arrays.values()) / 1e6
    print(f"wrote {args.out} ({size_mb:.0f} MB)")


if __name__ == "__main__":
    main()
