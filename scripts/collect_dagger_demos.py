"""DAgger collection (Ross et al. 2011): roll games out under the *learner's*
policy, but label every one of the learner's decision states with what the
*expert* (HeuristicAgent) would have done there.

Why this exists: plain BC hit 97.5% per-decision accuracy yet only ~9% win
rate vs the heuristic (25% = seat parity), the classic compounding-error
signature -- BC only ever sees states the expert visits, so the clone gets no
supervision on the slightly-off states its own small errors drift it into,
where its error rate is much higher, compounding over a ~250-decision game.
Labeling the learner's own state distribution is the textbook fix (O(eT)
regret instead of O(eT^2)).

Setup mirrors the evaluation condition: the policy occupies one (rotating)
seat against 3 heuristic opponents, so the labeled distribution is exactly
the one the policy faces at eval time. Only the policy seat's decisions are
recorded (the opponents' states are the plain heuristic-self-play
distribution the base demo set already covers). The game continues with the
POLICY's (sampled) action; `expert_prob` optionally executes the expert's
action instead with that probability (the classic beta-mixing knob, default
off -- with a programmatic expert and a warm-started learner, pure
learner-driven rollouts are standard).
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import random
import time

import numpy as np
import torch

from agents.heuristic import HeuristicAgent
from env.engine import CatanEngine, legal_actions
from env.state import NUM_PLAYERS, Phase
from scripts.collect_heuristic_demos import MAX_STEPS, encode_decision, records_to_arrays

# Phases the search-family labelers actually search (agents/search_heuristic.py)
_SEARCHED_PHASES = {Phase.SETUP_SETTLEMENT, Phase.SETUP_ROAD, Phase.DISCARD}


def _labeler_owns_decision(labeler_kind: str, state, expert_action, labeler=None) -> bool:
    """Whether this decision's label reflects the labeler's actual expertise.

    The search/rollout teachers only search deterministic-public-outcome
    decisions; for everything else (robber placement, trade responses,
    knight/monopoly timing, trade proposals) they silently defer to the plain
    heuristic's gates. Recording those deferred choices as imitation targets
    actively OVERWRITES whatever the student already knows there -- measured:
    an expert-iteration chain initialized from an RL-fine-tuned champion
    degraded 58.1% -> 49.6% in 3 rounds, because RL's gains live largely in
    exactly the decisions search cannot own and BC kept reverting them to
    heuristic behavior. So for those labelers, only searched decisions are
    recorded; the plain-heuristic labeler keeps recording everything (its
    students have no better prior to protect). The 'self' labeler -- the
    champion's own deterministic choice at states visited under its own
    (sampled) rollout policy, i.e. a self-distillation anchor -- also records
    everything: it never defers to anything foreign, so there's nothing to
    filter out."""
    if labeler_kind in ("heuristic", "self"):
        return True
    if labeler_kind == "rollout-override":
        # Strictest filter (after 'rollout' with the searched-decision filter
        # STILL degraded the RL champion 58.1% -> 54.2% in one round): record
        # ONLY MAIN decisions where the model-driven rollouts overrode the
        # static search ranking. Those are the only labels that carry
        # information beyond eval_state -- everything else in the teacher is
        # a policy family the RL champion has already moved past.
        return state.phase == Phase.MAIN and getattr(labeler, "last_overrode", False)
    from agents.search_heuristic import SEARCHABLE_MAIN
    if state.phase in _SEARCHED_PHASES:
        return True
    return state.phase == Phase.MAIN and expert_action.type in SEARCHABLE_MAIN


def make_labeler(kind: str, seat: int, rng: random.Random,
                  resource_weights: dict | None, search_depth: int = 2,
                  model=None, model_type: str = "gnn",
                  public_hand_features: bool = True):
    """The expert whose choices become the imitation targets. 'search' is the
    lookahead agent (agents/search_heuristic.py, ~51% win rate vs 3 plain
    heuristics at depth 1, ~58% at depth 2+; depths 2 and 3 measured
    equal-strength on paired seeds, depth 3 just costs ~2.5x more).
    'rollout' is expert iteration: the same search agent, but near-tied
    candidates are settled by determinized rollouts in which THIS seat's
    continuations are played by `model` (the current student) -- the teacher
    is literally search wrapped around the policy being trained, so it
    strengthens automatically as the student improves round over round.
    'self' is a self-distillation anchor: the labeler IS `model`, run
    deterministically, so the recorded targets are exactly the champion's own
    current preferences on states visited under its own (sampled) play --
    broad, representative, and immune to the catastrophic-forgetting failure
    mode of imitating a foreign teacher (see module docstring)."""
    if kind == "self":
        assert model is not None, "labeler 'self' needs the student model"
        from training.agent import HierarchicalLearnedAgent
        return HierarchicalLearnedAgent(seat, deterministic=True, model=model,
                                         model_kind=model_type,
                                         public_hand_features=public_hand_features)
    if kind in ("rollout", "rollout-override"):
        assert model is not None, f"labeler '{kind}' needs the student model"
        from agents.search_heuristic import RolloutSearchAgent
        from training.agent import HierarchicalLearnedAgent

        def factory(pid):
            return HierarchicalLearnedAgent(pid, model=model, deterministic=True,
                                             model_kind=model_type,
                                             public_hand_features=public_hand_features)
        kwargs = {"search_depth": search_depth, "rollouts": 6, "rollout_margin": 6.0,
                  "rollout_top_m": 4, "rollout_agent_factory": factory}
        if resource_weights is not None:
            kwargs["resource_weights"] = resource_weights
        return RolloutSearchAgent(seat, rng, **kwargs)
    if kind == "search":
        from agents.search_heuristic import SearchHeuristicAgent
        kwargs = {"search_depth": search_depth}
        if resource_weights is not None:
            kwargs["resource_weights"] = resource_weights
        return SearchHeuristicAgent(seat, rng, **kwargs)
    return HeuristicAgent(seat, rng, resource_weights=resource_weights)


def play_and_record_dagger(seed: int, public_hand_features: bool, model,
                            expert_prob: float = 0.0, model_type: str = "hier",
                            resource_weights: dict | None = None,
                            labeler_kind: str = "heuristic",
                            labeler_search_depth: int = 2) -> list[dict]:
    from training.agent import HierarchicalLearnedAgent

    torch.manual_seed(seed)  # policy sampling reproducible per game
    engine = CatanEngine(randomize_board=True, seed=seed)
    policy_seat = seed % NUM_PLAYERS
    mix_rng = random.Random(seed * 131 + 7)

    # Opponent seats stay the plain (unweighted) heuristic -- this mirrors the
    # eval condition exactly (see module docstring), which must not shift
    # underneath a labeler change. Only the labeler -- whose choices become
    # the imitation target -- gets `resource_weights`.
    agents = {pid: HeuristicAgent(pid, random.Random(seed * 97 + pid))
              for pid in range(NUM_PLAYERS) if pid != policy_seat}
    policy = HierarchicalLearnedAgent(policy_seat, model=model, deterministic=False,
                                       model_kind=model_type,
                                       public_hand_features=public_hand_features)
    # A dedicated expert instance for labeling: HeuristicAgent keeps small
    # per-turn internal state, so the labeler must not be one of the playing
    # opponents (whose seats/perspectives differ anyway).
    labeler = make_labeler(labeler_kind, policy_seat, random.Random(seed * 173 + 11),
                            resource_weights, labeler_search_depth,
                            model=model, model_type=model_type,
                            public_hand_features=public_hand_features)

    records = []
    steps = 0
    while not engine.done and steps < MAX_STEPS:
        state = engine.state
        actor = engine.acting_player()
        acts = legal_actions(state)
        if actor == policy_seat:
            expert_action = labeler.choose(state, acts)
            if len(acts) > 1 and _labeler_owns_decision(labeler_kind, state, expert_action,
                                                          labeler):
                records.append(encode_decision(state, actor, acts, expert_action,
                                                public_hand_features, model_type))
            if expert_prob > 0 and mix_rng.random() < expert_prob:
                execute = expert_action
            else:
                execute = policy.choose(state, acts)
        else:
            execute = agents[actor].choose(state, acts)
        engine.step(execute)
        steps += 1
    return records


_w_model = None
_w_phf = False
_w_expert_prob = 0.0
_w_model_type = "hier"
_w_resource_weights = None
_w_labeler_kind = "heuristic"
_w_labeler_depth = 2


def _init_worker(model, phf: bool, expert_prob: float, model_type: str = "hier",
                  resource_weights: dict | None = None,
                  labeler_kind: str = "heuristic", labeler_search_depth: int = 2) -> None:
    global _w_model, _w_phf, _w_expert_prob, _w_model_type, _w_resource_weights, \
        _w_labeler_kind, _w_labeler_depth
    torch.set_num_threads(1)
    model.eval()
    _w_model, _w_phf, _w_expert_prob, _w_model_type = model, phf, expert_prob, model_type
    _w_resource_weights = resource_weights
    _w_labeler_kind = labeler_kind
    _w_labeler_depth = labeler_search_depth


def _worker_collect(seeds: list[int]) -> list[dict]:
    out = []
    for seed in seeds:
        out.extend(play_and_record_dagger(seed, _w_phf, _w_model, _w_expert_prob, _w_model_type,
                                           _w_resource_weights, _w_labeler_kind,
                                           _w_labeler_depth))
    return out


def collect_dagger(model, games: int, base_seed: int, public_hand_features: bool,
                    num_workers: int = 12, expert_prob: float = 0.0,
                    model_type: str = "hier", resource_weights: dict | None = None,
                    labeler_kind: str = "heuristic",
                    labeler_search_depth: int = 2) -> dict[str, np.ndarray]:
    seeds = list(range(base_seed, base_seed + games))
    if num_workers <= 1:
        _init_worker(model, public_hand_features, expert_prob, model_type, resource_weights,
                      labeler_kind, labeler_search_depth)
        records = _worker_collect(seeds)
    else:
        nw = max(1, min(num_workers, games))
        chunks = [seeds[i::nw] for i in range(nw)]
        ctx = mp.get_context("fork")
        with ctx.Pool(processes=nw, initializer=_init_worker,
                      initargs=(model, public_hand_features, expert_prob, model_type,
                                resource_weights, labeler_kind,
                                labeler_search_depth)) as pool:
            records = [r for chunk in pool.map(_worker_collect, chunks) for r in chunk]
    return records_to_arrays(records, model_type)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--gnn-layers", type=int, default=3)
    parser.add_argument("--model-type", choices=["hier", "gnn"], default="hier")
    parser.add_argument("--public-hand-features", action="store_true")
    parser.add_argument("--games", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--expert-prob", type=float, default=0.0)
    parser.add_argument("--ore-weight", type=float, default=1.0,
                         help="resource_weights={ORE: this} for the labeler only (see "
                              "agents/heuristic.py) -- opponent seats stay unweighted.")
    parser.add_argument("--labeler", choices=["heuristic", "search", "rollout", "rollout-override", "self"], default="heuristic",
                         help="'search' = lookahead expert (agents/search_heuristic.py); "
                              "'rollout' = expert iteration (search + determinized rollouts "
                              "with the checkpoint model as own-seat rollout policy)")
    parser.add_argument("--labeler-search-depth", type=int, default=2,
                         help="search labeler only: same-turn lookahead depth")
    parser.add_argument("--num-workers", type=int, default=12)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    from env.board import HexType
    resource_weights = {HexType.ORE: args.ore_weight} if args.ore_weight != 1.0 else None

    from training.agent import load_gnn_model, load_hier_model
    if args.model_type == "gnn":
        model = load_gnn_model(args.checkpoint, hidden=args.hidden, gnn_layers=args.gnn_layers,
                                public_hand_features=args.public_hand_features)
    else:
        model = load_hier_model(args.checkpoint, hidden=args.hidden,
                                 public_hand_features=args.public_hand_features)
    t0 = time.time()
    arrays = collect_dagger(model, args.games, args.seed, args.public_hand_features,
                             num_workers=args.num_workers, expert_prob=args.expert_prob,
                             model_type=args.model_type, resource_weights=resource_weights,
                             labeler_kind=args.labeler,
                             labeler_search_depth=args.labeler_search_depth)
    n = arrays["type_idx"].shape[0]
    print(f"{args.games} learner-rollout games -> {n} expert-labeled decisions "
          f"in {time.time()-t0:.1f}s")
    np.savez(args.out, **arrays)
    size_mb = sum(a.nbytes for a in arrays.values()) / 1e6
    print(f"wrote {args.out} ({size_mb:.0f} MB)")


if __name__ == "__main__":
    main()
