"""League-based self-play training CLI (roadmap phase 5, design doc §7).

Builds on `training/train_hier.py`'s single-policy PPO loop by adding a
persisted `League` (training/league.py) of opponents: historical snapshots,
main exploiters, and the phase-3 heuristic/random agents as permanent free
diversity. Each iteration is either pure self-play (fast, all 4 seats are the
trainee) or a match against one PFSP-sampled league opponent (1 trainee seat,
3 opponent seats) -- `--self-play-prob` controls the mix. This is exactly the
mitigation the design doc calls for against pure-self-play's known failure
modes (oscillation, catastrophic forgetting, overfitting to one style).

Promotion is gated on a pre-registered statistical test (win rate threshold +
exact binomial p-value), not a judgment call after looking at the numbers --
see `League.promotion_test`. Historical snapshots are kept forever once
demoted, so an old exploit can't quietly resurface.

Compute-scoped simplification: exploiters here are trained *sequentially*
(pause main training, run a short side-session against the frozen current
main, resume) rather than on separate concurrent hardware the way
AlphaStar/OpenAI Five's league does -- a deliberate, documented scope
decision for a single CPU machine, not an oversight.
"""
from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import os
import random
import time
from collections import OrderedDict, deque

import numpy as np
import torch

from agents.heuristic import HeuristicAgent
from agents.random_agent import RandomAgent
from env.engine import CatanEngine, total_vp
from env.pettingzoo_env import CatanAECEnv
from env.state import NUM_PLAYERS
from training.agent import HierarchicalLearnedAgent, load_gnn_model, load_hier_model
from training.hier_model import HierarchicalActorCritic
from training.hier_ppo import (
    GAE_LAMBDA, GAMMA, collect_rollout, collect_rollout_parallel, compute_holdout_nll,
    load_bc_anchor, ppo_update, reseed_forked_worker,
)
from training.league import League
from training.model import observation_dim
from training.model_adapters import ADAPTERS
from training.run_manifest import write_manifest
from training.train_hier import build_model, evaluate_policy, move_optimizer_state

MAX_EVAL_STEPS = 4000
_MODEL_CACHE_MAX = 16


_LOADED_MODEL_CACHE: "OrderedDict[tuple[str, str, int, int], object]" = OrderedDict()


def load_model_for(model_type: str, checkpoint_path: str, hidden: int, gnn_layers: int,
                   public_hand_features: bool = False):
    """Cached by (model_type, checkpoint_path, hidden, gnn_layers): every
    checkpoint file is written once under a unique name (historical/main/
    main_exploiter snapshots each get their own filename per iteration) and
    never mutated afterward, so it's safe -- and much cheaper -- to load a
    given checkpoint from disk only once instead of every time an opponent is
    sampled or evaluated against. LRU-bounded: a long run accumulates
    hundreds of league members, and an unbounded cache would keep every one
    of their models resident forever."""
    key = (model_type, checkpoint_path, hidden, gnn_layers, public_hand_features)
    if key in _LOADED_MODEL_CACHE:
        _LOADED_MODEL_CACHE.move_to_end(key)
        return _LOADED_MODEL_CACHE[key]
    if model_type == "gnn":
        model = load_gnn_model(checkpoint_path, hidden=hidden, gnn_layers=gnn_layers,
                                public_hand_features=public_hand_features)
    else:
        model = load_hier_model(checkpoint_path, hidden=hidden,
                                 public_hand_features=public_hand_features)
    _LOADED_MODEL_CACHE[key] = model
    if len(_LOADED_MODEL_CACHE) > _MODEL_CACHE_MAX:
        _LOADED_MODEL_CACHE.popitem(last=False)
    return model


def env_kwargs_from_args(args: argparse.Namespace) -> dict:
    return dict(
        randomize_board=args.randomize_board,
        allow_trading=not args.no_trading,
        allow_dev_cards=not args.no_dev_cards,
        vp_shaping_weight=args.vp_shaping_weight,
        max_episode_steps=args.max_episode_steps,
        public_hand_features=args.public_hand_features,
        truncation_reward=getattr(args, "truncation_reward", "zero"),
        terminal_reward=getattr(args, "terminal_reward", "win_loss"),
    )


def annealed_bc_coef(initial: float, final: float | None, iteration: int,
                      start_iteration: int, total_iterations: int) -> float:
    """Linear anchor-coefficient schedule over *this invocation's* iterations
    (resume-aware: progress is measured from start_iteration, not absolute
    league iteration). final=None means no annealing."""
    if final is None:
        return initial
    if total_iterations <= 1:
        return final
    progress = (iteration - start_iteration) / (total_iterations - 1)
    progress = min(1.0, max(0.0, progress))
    return initial + (final - initial) * progress


def episode_rating_teams(summary: dict, main_name: str, opponent_name: str) -> list[str]:
    """Collapses one episode's finishing order into a clean 2-team
    [winner_name, loser_name] comparison for `League.update_ratings`, rather
    than feeding the opponent's (typically 3) identical seats in as if they
    were independent competitors -- see the call site's comment for the
    rating-collapse bug that caused."""
    trainee_pid = summary["trainee_pids"][0]
    ranking = summary["ranking_pids"]
    trainee_rank = ranking.index(trainee_pid)
    opponent_best_rank = min(ranking.index(pid) for pid in ranking if pid != trainee_pid)
    if trainee_rank <= opponent_best_rank:
        return [main_name, opponent_name]
    return [opponent_name, main_name]


def save_checkpoint(model: HierarchicalActorCritic, checkpoint_dir: str, name: str, iteration: int,
                     optimizer: torch.optim.Optimizer | None = None) -> str:
    path = os.path.join(checkpoint_dir, f"{name}.pt")
    ckpt = {"model": model.state_dict(), "iteration": iteration}
    if optimizer is not None:
        ckpt["optimizer"] = optimizer.state_dict()
    torch.save(ckpt, path)
    return path


def make_seat_agents(member, seats: list[int], args: argparse.Namespace, rng: random.Random,
                      shared_model=None) -> dict[int, object]:
    """One agent instance per requested seat -- never share a single
    HeuristicAgent/HierarchicalLearnedAgent instance across seats, since
    HeuristicAgent tracks per-player state (its own hand/target) internally.
    Pass `shared_model` (an already-loaded model for `member`) to avoid
    re-loading the same checkpoint from disk on every call, e.g. across many
    games against the same frozen opponent."""
    if member.role == "random":
        return {pid: RandomAgent(pid, random.Random(rng.randint(0, 2**31 - 1))) for pid in seats}
    if member.role == "heuristic":
        return {pid: HeuristicAgent(pid, random.Random(rng.randint(0, 2**31 - 1))) for pid in seats}
    model = shared_model if shared_model is not None else load_model_for(
        args.model_type, member.checkpoint_path, args.hidden, args.gnn_layers,
        public_hand_features=args.public_hand_features)
    return {pid: HierarchicalLearnedAgent(pid, model=model, deterministic=False, model_kind=args.model_type,
                                           public_hand_features=args.public_hand_features)
            for pid in seats}


def _play_vs_member_game(model, member, shared_model, args: argparse.Namespace,
                          env_kwargs: dict, seed: int) -> tuple[bool, bool]:
    """One game: trainee (deterministic) vs. 3 copies of `member`, seat
    rotated by seed to cancel position bias. Returns (won, finished).

    Reseeds torch's global RNG from `seed` before playing: a model-based
    opponent (`deterministic=False` in make_seat_agents) samples its actions
    from torch's global RNG rather than a per-agent seeded one, so without
    this reset each game's outcome would depend on however much prior
    sampling happened earlier in the process -- reproducible for a single
    serial run, but not equal to a parallel run's per-worker RNG stream (each
    worker forks from the same initial state, then diverges independently).
    This makes each game a pure function of its own seed either way.

    The reseed happens inside `torch.random.fork_rng()` so it can't clobber
    the caller's global torch RNG -- in serial mode this runs in the training
    process, and without the fork every eval pass would silently reset the
    training loop's own sampling stream."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        engine = CatanEngine(randomize_board=env_kwargs["randomize_board"], seed=seed,
                              allow_trading=env_kwargs["allow_trading"], allow_dev_cards=env_kwargs["allow_dev_cards"])
        trainee_seat = seed % NUM_PLAYERS
        opponent_seats = [p for p in range(NUM_PLAYERS) if p != trainee_seat]
        agents = make_seat_agents(member, opponent_seats, args, random.Random(seed), shared_model=shared_model)
        agents[trainee_seat] = HierarchicalLearnedAgent(trainee_seat, model=model, deterministic=True,
                                                         model_kind=args.model_type,
                                                         public_hand_features=args.public_hand_features)
        steps = 0
        while not engine.done and steps < MAX_EVAL_STEPS:
            actor = engine.acting_player()
            engine.step(agents[actor].choose(engine.state))
            steps += 1
        finished = engine.done
        won = finished and engine.state.winner == trainee_seat
    return won, finished


# -- parallel promotion/tier evaluation --------------------------------------
# Same fork-based worker-pool pattern as training/hier_ppo.py's
# collect_rollout_parallel and training/train_hier.py's evaluate_policy:
# games vs. a frozen league member are independent and read-only w.r.t. both
# models, so this removes promotion eval (100 games by default) and tier eval
# from being a synchronous stall in the training loop.
_eval2_worker_model = None
_eval2_worker_member = None
_eval2_worker_shared_model = None
_eval2_worker_args: argparse.Namespace | None = None
_eval2_worker_env_kwargs: dict = {}


def _init_eval2_worker(model, member, shared_model, args: argparse.Namespace, env_kwargs: dict) -> None:
    global _eval2_worker_model, _eval2_worker_member, _eval2_worker_shared_model
    global _eval2_worker_args, _eval2_worker_env_kwargs
    torch.set_num_threads(1)
    reseed_forked_worker()
    model.eval()
    if shared_model is not None:
        shared_model.eval()
    _eval2_worker_model = model
    _eval2_worker_member = member
    _eval2_worker_shared_model = shared_model
    _eval2_worker_args = args
    _eval2_worker_env_kwargs = env_kwargs


def _eval2_worker_play(seeds: list[int]) -> tuple[int, int]:
    wins = finished_n = 0
    for seed in seeds:
        won, finished = _play_vs_member_game(_eval2_worker_model, _eval2_worker_member,
                                              _eval2_worker_shared_model, _eval2_worker_args,
                                              _eval2_worker_env_kwargs, seed)
        wins += int(won)
        finished_n += int(finished)
    return wins, finished_n


def evaluate_vs_member(model, league: League, member_name: str, games: int,
                        seed_base: int, env_kwargs: dict, args: argparse.Namespace,
                        num_workers: int = 1) -> tuple[int, int, int]:
    """Trainee (deterministic) vs. 3 copies of `member_name`, seat rotated
    per game to avoid position bias. Returns (wins, games, finished)."""
    member = league.members[member_name]
    # Load the opponent's checkpoint once for the whole eval, not once per
    # game -- the checkpoint is frozen for the duration of this call, so
    # re-loading it `games` times (100 by default) was pure wasted disk I/O
    # and deserialization, blocking the training loop synchronously.
    shared_model = None
    if member.role not in ("random", "heuristic"):
        shared_model = load_model_for(args.model_type, member.checkpoint_path, args.hidden, args.gnn_layers,
                                       public_hand_features=args.public_hand_features)

    seeds = [seed_base + i for i in range(games)]

    if num_workers <= 1:
        wins, finished = 0, 0
        for seed in seeds:
            won, fin = _play_vs_member_game(model, member, shared_model, args, env_kwargs, seed)
            wins += int(won)
            finished += int(fin)
    else:
        nw = max(1, min(num_workers, games))
        chunks = [seeds[i::nw] for i in range(nw)]
        chunks = [c for c in chunks if c]
        ctx = mp.get_context("fork")
        with ctx.Pool(processes=len(chunks), initializer=_init_eval2_worker,
                      initargs=(model, member, shared_model, args, env_kwargs)) as pool:
            results = pool.map(_eval2_worker_play, chunks)
        wins = sum(r[0] for r in results)
        finished = sum(r[1] for r in results)

    return wins, games, finished


def run_exploiter_session(league: League, args: argparse.Namespace, iteration: int, env_kwargs: dict,
                           device: str) -> None:
    """Fork the current main's weights, train briefly against nothing but the
    frozen current main (no PFSP diversity -- that's the point: find and
    punish main's specific weaknesses), then add the result to the league."""
    main_member = league.main()
    if main_member is None:
        return

    adapter = ADAPTERS[args.model_type]
    exploiter_model = build_model(args.model_type, args.hidden, args.gnn_layers,
                                   public_hand_features=args.public_hand_features)
    ckpt = torch.load(main_member.checkpoint_path, map_location="cpu")
    exploiter_model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    optimizer = torch.optim.Adam(exploiter_model.parameters(), lr=args.lr)
    frozen_main = load_model_for(args.model_type, main_member.checkpoint_path, args.hidden, args.gnn_layers,
                                  public_hand_features=args.public_hand_features)

    rng = random.Random(iteration * 7919)
    base_seed = 500_000 + iteration * 1000
    for _ in range(args.exploiter_iterations):
        trainee_seat = rng.randrange(NUM_PLAYERS)
        opponent_seats = [p for p in range(NUM_PLAYERS) if p != trainee_seat]
        opponent_agents = {pid: HierarchicalLearnedAgent(pid, model=frozen_main, deterministic=False,
                                                          model_kind=args.model_type,
                                                          public_hand_features=args.public_hand_features)
                            for pid in opponent_seats}
        if args.num_workers > 1:
            transitions, _ = collect_rollout_parallel(env_kwargs, exploiter_model, args.episodes_per_iter,
                                                        base_seed, args.num_workers,
                                                        opponent_agents=opponent_agents, opponent_name=main_member.name,
                                                        adapter=adapter, gamma=args.gamma, lam=args.gae_lambda)
        else:
            env = CatanAECEnv(**env_kwargs)
            transitions, _ = collect_rollout(env, exploiter_model, "cpu", args.episodes_per_iter, base_seed,
                                              opponent_agents=opponent_agents, opponent_name=main_member.name,
                                              adapter=adapter, gamma=args.gamma, lam=args.gae_lambda)
        base_seed += args.episodes_per_iter
        exploiter_model.to(device)
        ppo_update(exploiter_model, optimizer, transitions, epochs=args.epochs, minibatch_size=args.minibatch_size,
                   adapter=adapter, device=device, clip_ratio=args.clip_ratio, value_coef=args.value_coef,
                   entropy_coef=args.entropy_coef, max_grad_norm=args.max_grad_norm,
                   target_kl=(args.target_kl if args.target_kl >= 0 else None))
        exploiter_model.to("cpu")

    name = f"main_exploiter_iter{iteration}"
    path = save_checkpoint(exploiter_model, args.checkpoint_dir, name, iteration, optimizer=optimizer)
    league.add_member(name, "main_exploiter", checkpoint_path=path, iteration=iteration,
                       mu=main_member.mu, sigma=main_member.sigma)
    print(f"  trained {name} against frozen {main_member.name} for {args.exploiter_iterations} iterations")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--episodes-per-iter", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=512)
    parser.add_argument("--max-episode-steps", type=int, default=4000,
                         help="step cap; heuristic games take ~1,100 steps (p90 ~1,450), so a "
                              "cap below ~2,500 truncates most games (audit F-06)")
    parser.add_argument("--terminal-reward", choices=["win_loss", "rank"], default="win_loss",
                         help="game-end reward: 'win_loss' (+1 winner, -1/3 each loser; default) or "
                              "legacy 'rank' placement reward {1, 0, -0.5, -1} (audit F-16)")
    parser.add_argument("--truncation-reward", choices=["zero", "rank"], default="zero",
                         help="what a step-cap truncation pays: 'zero' (default; GAE bootstraps V(s_T)) or 'rank' (legacy rank-on-standing, pays the VP leader a full win -- audit F-06)")
    parser.add_argument("--vp-shaping-weight", type=float, default=0.05)
    parser.add_argument("--no-trading", action="store_true")
    parser.add_argument("--no-dev-cards", action="store_true")
    parser.add_argument("--randomize-board", action="store_true", default=True)
    parser.add_argument("--public-hand-features", action="store_true",
                         help="expose the engine's publicly-inferable per-player resource "
                              "estimates (card counting) as observation features. Widens the "
                              "observation, so every checkpoint in a league directory must be "
                              "trained with the same setting of this flag.")
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--self-play-prob", type=float, default=0.5,
                         help="fraction of iterations that are pure self-play vs. a sampled league opponent")
    parser.add_argument("--snapshot-every", type=int, default=25,
                         help="add the current trainee to the league as a historical member every N iterations")
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--eval-games", type=int, default=30)
    parser.add_argument("--promotion-games", type=int, default=100)
    parser.add_argument("--promotion-win-rate", type=float, default=0.30,
                         help="minimum win rate in the 1-vs-3 promotion match (parity = 0.25); the "
                              "result must also be significantly above parity (binomial test)")
    parser.add_argument("--promotion-alpha", type=float, default=0.05)
    parser.add_argument("--bc-anchor-dataset", type=str, default=None,
                         help="demonstration .npz (scripts/collect_heuristic_demos.py) used as a "
                              "BC anchor: each PPO gradient step adds bc-anchor-coef * NLL of the "
                              "demonstrated actions, so the policy can only drift from the "
                              "demonstrations where reward justifies it. hier model only.")
    parser.add_argument("--bc-anchor-coef", type=float, default=0.2)
    parser.add_argument("--bc-anchor-coef-final", type=float, default=None,
                         help="if set, linearly anneal the anchor coefficient from "
                              "--bc-anchor-coef down to this value over the run's iterations. "
                              "Rationale: a constant anchor confines the policy to the BC basin "
                              "(polish-then-plateau); annealing lets late-stage RL leave the "
                              "basin gradually from a polished starting point instead of being "
                              "either imprisoned (constant) or shredded (no anchor).")
    parser.add_argument("--bc-anchor-samples", type=int, default=200_000,
                         help="subsample the anchor dataset to this many decisions (device memory)")
    parser.add_argument("--bc-anchor-minibatch", type=int, default=512)
    parser.add_argument("--bc-holdout-dataset", type=str, default=None,
                         help="a SEPARATE demonstration .npz (disjoint game seeds from the "
                              "anchor/BC-pretraining dataset) never trained on. Its NLL under "
                              "the current policy is measured every iteration as a drift/"
                              "erosion signal, and the checkpoint with the lowest value seen is "
                              "kept as <checkpoint-dir>/best_bc_holdout.pt -- early-stopped "
                              "selection instead of trusting a fixed final iteration count, "
                              "since a constant/low BC-anchor coefficient was observed to erode "
                              "the policy slowly over thousands of iterations after an earlier "
                              "peak (see README/experiment notes).")
    parser.add_argument("--bc-holdout-samples", type=int, default=None,
                         help="subsample the holdout dataset to this many decisions (device "
                              "memory); default None uses the whole file.")
    parser.add_argument("--select-best-opponent", type=str, default="heuristic",
                         choices=["random", "heuristic"],
                         help="checkpoint selection and early stopping are driven by this "
                              "opponent's win rate from the periodic eval block -- i.e. the "
                              "actual task metric, not a proxy. (A per-iteration behavior-NLL "
                              "proxy was tried and found unreliable: divergence from a "
                              "demonstrator's behavior isn't the same as getting worse, since "
                              "exceeding the demonstrator requires diverging from it.)")
    parser.add_argument("--best-eval-window", type=int, default=3,
                         help="checkpoint selection compares a moving average of the last N "
                              "eval points (not the single latest one) against the best moving "
                              "average seen so far, to avoid locking onto a lucky small-sample "
                              "eval -- eval_games default 30 has ~6-point win-rate noise.")
    parser.add_argument("--early-stop-patience", type=int, default=0,
                         help="stop training if the smoothed vs-select-best-opponent win rate "
                              "hasn't improved for this many consecutive eval checkpoints. "
                              "0 disables early stopping.")
    parser.add_argument("--early-stop-min-delta", type=float, default=0.0,
                         help="minimum increase in the smoothed win rate to count as an "
                              "improvement (resets the early-stopping patience counter); "
                              "guards against noise-sized upticks resetting patience forever.")
    parser.add_argument("--final-eval-games", type=int, default=200,
                         help="at the end of training, re-evaluate best_eval.pt and the final "
                              "model with this many games each (0 disables). Necessary even "
                              "with a full selection window: taking the max of a noisy "
                              "statistic across dozens of eval checkpoints over a long run "
                              "systematically overestimates the true value at whichever point "
                              "wins (regression-to-the-mean, aka winner's curse) -- observed "
                              "directly, where a smoothed training-time win rate reading from a "
                              "genuine full 3-eval window (not a partial-window artifact) came "
                              "in more than 2x higher than a larger re-evaluation of the same "
                              "checkpoint showed. A big final confirmatory eval on the "
                              "shortlisted candidate(s) is the standard fix, not a bigger "
                              "training-time window -- no amount of window smoothing removes "
                              "bias introduced by searching over many candidates.")
    parser.add_argument("--exploiter-every", type=int, default=0,
                         help="0 disables exploiter side-sessions")
    parser.add_argument("--exploiter-iterations", type=int, default=20)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints_league")
    parser.add_argument("--init-checkpoint", type=str, default=None)
    parser.add_argument("--model-type", choices=["hier", "gnn"], default="hier",
                         help="'hier' = flat-vector trunk (phase 4), 'gnn' = graph-encoded "
                              "board with pointer heads over node embeddings (phase 6)")
    parser.add_argument("--gnn-layers", type=int, default=3, help="only used when --model-type gnn")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gamma", type=float, default=GAMMA)
    parser.add_argument("--gae-lambda", type=float, default=GAE_LAMBDA)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--target-kl", type=float, default=0.02,
                         help="early-stop a PPO update's epoch loop if the mean approx_kl for an "
                              "epoch exceeds this; pass a negative value to disable")
    parser.add_argument("--device", type=str, default="auto",
                         help="device for the PPO gradient step ('auto', 'cpu', 'cuda'). Rollout "
                              "collection always stays on CPU regardless of this flag -- forked "
                              "parallel workers (--num-workers > 1) can't inherit a CUDA context, "
                              "and single-sample env-step inference is dominated by Python/env "
                              "overhead anyway, so only the batched PPO update moves to the GPU.")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    print(f"PPO update device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    write_manifest(args.checkpoint_dir, args, {"driver": "league_train"})
    league = League.load(args.checkpoint_dir)
    adapter = ADAPTERS[args.model_type]

    model = build_model(args.model_type, args.hidden, args.gnn_layers,
                         public_hand_features=args.public_hand_features)
    init_ckpt = None
    if args.init_checkpoint is not None:
        init_ckpt = torch.load(args.init_checkpoint, map_location="cpu")
        model.load_state_dict(init_ckpt["model"] if "model" in init_ckpt else init_ckpt)
        print(f"initialized from {args.init_checkpoint}")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    if init_ckpt is not None and isinstance(init_ckpt, dict) and "optimizer" in init_ckpt:
        optimizer.load_state_dict(init_ckpt["optimizer"])
        move_optimizer_state(optimizer, device)
        print("restored optimizer state from init checkpoint")

    if not league.members:
        league.add_member("random", "random")
        league.add_member("heuristic", "heuristic")
        init_path = save_checkpoint(model, args.checkpoint_dir, "main_iter0", 0, optimizer=optimizer)
        league.add_member("main_iter0", "main", checkpoint_path=init_path, iteration=0)
        league.save()
        print("initialized new league: random, heuristic, main_iter0")
    else:
        print(f"resumed league with {len(league.members)} members, main={league.main_name}")

    bc_anchor = None
    if args.bc_anchor_dataset:
        assert args.model_type == "hier", \
            "--bc-anchor-dataset currently supports the hier model only (flat-obs demonstrations)"
        bc_anchor = load_bc_anchor(args.bc_anchor_dataset, device,
                                    max_samples=args.bc_anchor_samples, seed=args.seed)
        expected_dim = observation_dim(public_hand_features=args.public_hand_features)
        assert bc_anchor["obs"].shape[1] == expected_dim, (
            f"anchor obs width {bc_anchor['obs'].shape[1]} != model obs width {expected_dim} -- "
            f"collect the demonstrations with a matching --public-hand-features setting")
        print(f"BC anchor: {bc_anchor['type_idx'].shape[0]} demos on {device}, "
              f"coef={args.bc_anchor_coef}")

    bc_holdout = None
    best_holdout_nll = math.inf
    best_holdout_iteration = None
    if args.bc_holdout_dataset:
        assert args.model_type == "hier", "--bc-holdout-dataset currently supports the hier model only"
        bc_holdout = load_bc_anchor(args.bc_holdout_dataset, device, max_samples=args.bc_holdout_samples)
        expected_dim = observation_dim(public_hand_features=args.public_hand_features)
        assert bc_holdout["obs"].shape[1] == expected_dim, (
            f"holdout obs width {bc_holdout['obs'].shape[1]} != model obs width {expected_dim}")
        print(f"BC holdout: {bc_holdout['type_idx'].shape[0]} demos on {device} "
              f"(early-stop selection signal)")

    # Checkpoint selection / early stopping driven by the actual eval metric
    # (--select-best-opponent's win rate), not a training-loss proxy -- see
    # the flag's help text. State resets on process restart (not persisted
    # across --init-checkpoint resumes), same scope as the bc_holdout tracking.
    recent_eval_scores: deque[float] = deque(maxlen=args.best_eval_window)
    best_eval_score = -math.inf
    best_eval_iteration = None
    stale_evals = 0
    early_stopped = False

    env_kwargs = env_kwargs_from_args(args)
    rng = random.Random(args.seed)

    # Resume-safe iteration numbering: the league persists the last completed
    # iteration, so a rerun continues from there instead of restarting at 1
    # and silently overwriting historical_iter25 etc. with different weights
    # while their old ratings survive. --iterations is "how many more".
    start_iteration = league.last_iteration + 1
    if league.last_iteration:
        print(f"resuming at iteration {start_iteration} (league had completed {league.last_iteration})")
    base_seed = args.seed + (start_iteration - 1) * args.episodes_per_iter

    for iteration in range(start_iteration, start_iteration + args.iterations):
        t0 = time.time()
        use_self_play = rng.random() < args.self_play_prob
        opponent_agents, opponent_name, trainee_seat = None, None, None
        if not use_self_play:
            opp_member = league.sample_opponent(rng)
            opponent_name = opp_member.name
            trainee_seat = rng.randrange(NUM_PLAYERS)
            opponent_seats = [p for p in range(NUM_PLAYERS) if p != trainee_seat]
            opponent_agents = make_seat_agents(opp_member, opponent_seats, args, rng)

        if args.num_workers > 1:
            transitions, summaries = collect_rollout_parallel(
                env_kwargs, model, args.episodes_per_iter, base_seed, args.num_workers,
                opponent_agents=opponent_agents, opponent_name=opponent_name, adapter=adapter,
                gamma=args.gamma, lam=args.gae_lambda)
        else:
            env = CatanAECEnv(**env_kwargs)
            transitions, summaries = collect_rollout(
                env, model, "cpu", args.episodes_per_iter, base_seed,
                opponent_agents=opponent_agents, opponent_name=opponent_name, adapter=adapter,
                gamma=args.gamma, lam=args.gae_lambda)
        base_seed += args.episodes_per_iter

        bc_coef_now = annealed_bc_coef(args.bc_anchor_coef, args.bc_anchor_coef_final,
                                        iteration, start_iteration, args.iterations)
        model.to(device)
        stats = ppo_update(model, optimizer, transitions, epochs=args.epochs, minibatch_size=args.minibatch_size,
                            adapter=adapter, device=device, clip_ratio=args.clip_ratio,
                            value_coef=args.value_coef, entropy_coef=args.entropy_coef,
                            max_grad_norm=args.max_grad_norm,
                            target_kl=(args.target_kl if args.target_kl >= 0 else None),
                            bc_dataset=bc_anchor, bc_coef=bc_coef_now,
                            bc_minibatch_size=args.bc_anchor_minibatch)

        holdout_nll, is_best_holdout = None, False
        if bc_holdout is not None:
            holdout_nll = compute_holdout_nll(model, bc_holdout)
            if holdout_nll < best_holdout_nll:
                best_holdout_nll, best_holdout_iteration, is_best_holdout = holdout_nll, iteration, True
                save_checkpoint(model, args.checkpoint_dir, "best_bc_holdout", iteration, optimizer=optimizer)

        model.to("cpu")
        elapsed = time.time() - t0

        # Rating update: the trainee's current (unsnapshotted) strength is
        # approximated by whatever "main" currently is -- a live proxy that
        # gets a fresh true value each time a new snapshot is promoted.
        #
        # Collapsed to a clean 2-team comparison (trainee vs. opponent)
        # rather than feeding the opponent's 3 identical seats into
        # `trueskill.rate()` as if they were 3 independent competitors: that
        # produced 3 rating updates for the same dict key per game, with only
        # the *last* one (the opponent's worst-placed seat, since 3 identical
        # seats can't all rank 1st) surviving the overwrite -- a systematic
        # pessimistic bias that was dragging every rated member's mu toward
        # large negative numbers regardless of who was actually winning.
        if opponent_name is not None and league.main_name is not None:
            for s in summaries:
                league.update_ratings(episode_rating_teams(s, league.main_name, opponent_name))

        mean_turns = sum(s["turns"] for s in summaries) / len(summaries)
        finish_rate = sum(1 for s in summaries if s["winner"] is not None) / len(summaries)
        mode = "self-play" if use_self_play else f"vs {opponent_name}"
        bc_str = f" bc={stats['bc_loss']:.3f}@{bc_coef_now:.3f}" if "bc_loss" in stats else ""
        holdout_str = f" holdout={holdout_nll:.3f}{'*' if is_best_holdout else ''}" if holdout_nll is not None else ""
        print(f"iter {iteration:4d} | {mode:<20s} | {len(transitions):5d} steps | {elapsed:5.1f}s | "
              f"turns={mean_turns:5.1f} finish={finish_rate:4.0%} | "
              f"pol={stats['policy_loss']:+.4f} val={stats['value_loss']:.4f} "
              f"ent={stats['entropy']:.3f} kl={stats['approx_kl']:.4f} "
              f"gnorm={stats['grad_norm']:.2f} ev={stats['explained_variance']:.2f}{bc_str}{holdout_str}")

        # Crash-resilient checkpoint + persisted progress marker every
        # iteration -- both cheap (a few MB / a small JSON) relative to
        # losing up to eval_every iterations on an interruption.
        save_checkpoint(model, args.checkpoint_dir, "latest", iteration, optimizer=optimizer)
        league.last_iteration = iteration
        league.save()

        if args.snapshot_every and iteration % args.snapshot_every == 0:
            name = f"historical_iter{iteration}"
            path = save_checkpoint(model, args.checkpoint_dir, name, iteration, optimizer=optimizer)
            seed_mu = league.rating(league.main_name).mu if league.main_name else None
            seed_sigma = league.rating(league.main_name).sigma if league.main_name else None
            league.add_member(name, "historical", checkpoint_path=path, iteration=iteration,
                               mu=seed_mu, sigma=seed_sigma)
            print(f"  snapshotted {name}")

        if args.exploiter_every and iteration % args.exploiter_every == 0:
            run_exploiter_session(league, args, iteration, env_kwargs, device)

        last_iteration_of_run = iteration == start_iteration + args.iterations - 1
        if iteration % args.eval_every == 0 or last_iteration_of_run:
            if league.main_name is not None:
                # Seed bases stride by the game count so successive evals use
                # disjoint game sets (plain `+ iteration` made consecutive
                # evals share most of their seeds).
                wins, games, finished = evaluate_vs_member(
                    model, league, league.main_name, args.promotion_games,
                    seed_base=700_000 + iteration * args.promotion_games,
                    env_kwargs=env_kwargs, args=args,
                    num_workers=args.num_workers)
                promoted = league.promotion_test(wins, games, args.promotion_win_rate, args.promotion_alpha)
                print(f"  vs main ({league.main_name}): {wins}/{games} wins "
                      f"({wins / games:.0%}){' -- PROMOTED' if promoted else ''}")
                if promoted:
                    name = f"main_iter{iteration}"
                    path = save_checkpoint(model, args.checkpoint_dir, name, iteration, optimizer=optimizer)
                    league.add_member(name, "historical", checkpoint_path=path, iteration=iteration,
                                       mu=league.rating(league.main_name).mu, sigma=league.rating(league.main_name).sigma)
                    league.promote(name, iteration)

            select_res = None
            for opp in ("random", "heuristic"):
                res = evaluate_policy(model, opp, args.eval_games,
                                       seed_base=2_000_000 + iteration * args.eval_games,
                                       model_kind=args.model_type, num_workers=args.num_workers,
                                       public_hand_features=args.public_hand_features)
                print(f"  eval vs {opp:<10} win_rate={res['win_rate']:.0%} "
                      f"finish_rate={res['finish_rate']:.0%} avg_vp={res['avg_vp']:.2f}")
                if opp == args.select_best_opponent:
                    select_res = res

            ratings_str = ", ".join(f"{m.name}={league.rating(m.name).mu:.1f}"
                                     for m in sorted(league.members.values(), key=lambda m: -league.rating(m.name).mu)[:6])
            print(f"  top ratings: {ratings_str}")
            league.save()

            recent_eval_scores.append(select_res["win_rate"])
            if len(recent_eval_scores) < args.best_eval_window:
                # Window not yet full: deliberately withhold from the best/
                # stale comparison below rather than compare a partial (1- or
                # 2-sample) average against later full-window averages. An
                # early run once locked its "best" onto a 2-sample average
                # from the first two evals -- less diluted, so more likely to
                # be an extreme value than any later fully-windowed average
                # -- and then early-stopped 1000 iterations later having
                # never revisited a checkpoint that outright beat it on a
                # rigorous 240-game eval. Every comparison must be the same
                # effective sample size.
                print(f"  eval vs {args.select_best_opponent} raw={select_res['win_rate']:.1%} "
                      f"(warming up selection window {len(recent_eval_scores)}/{args.best_eval_window})")
            else:
                smoothed = sum(recent_eval_scores) / len(recent_eval_scores)
                if smoothed > best_eval_score + args.early_stop_min_delta:
                    best_eval_score, best_eval_iteration, stale_evals = smoothed, iteration, 0
                    save_checkpoint(model, args.checkpoint_dir, "best_eval", iteration, optimizer=optimizer)
                    print(f"  new best_eval vs {args.select_best_opponent}: smoothed={smoothed:.1%} "
                          f"(raw={select_res['win_rate']:.1%}, window={len(recent_eval_scores)}) -> best_eval.pt")
                else:
                    stale_evals += 1
                    patience_str = str(args.early_stop_patience) if args.early_stop_patience else "off"
                    print(f"  eval vs {args.select_best_opponent} smoothed={smoothed:.1%} "
                          f"(best={best_eval_score:.1%} @ iter {best_eval_iteration}, "
                          f"stale {stale_evals}/{patience_str})")
                    if args.early_stop_patience and stale_evals >= args.early_stop_patience:
                        print(f"  early stopping: no improvement in smoothed vs-{args.select_best_opponent} "
                              f"win rate for {stale_evals} consecutive evals "
                              f"(best={best_eval_score:.1%} at iter {best_eval_iteration})")
                        early_stopped = True

        if early_stopped:
            break

    if best_eval_iteration is not None:
        print(f"best_eval: vs {args.select_best_opponent} smoothed_win_rate={best_eval_score:.1%} "
              f"at iter {best_eval_iteration} -> {os.path.join(args.checkpoint_dir, 'best_eval.pt')}")
    if bc_holdout is not None:
        print(f"best_bc_holdout: NLL={best_holdout_nll:.4f} at iter {best_holdout_iteration} "
              f"-> {os.path.join(args.checkpoint_dir, 'best_bc_holdout.pt')}")

    if args.final_eval_games > 0:
        print(f"\nconfirmatory eval ({args.final_eval_games} games vs {args.select_best_opponent} -- "
              f"the trustworthy number; training-time smoothed readings above are optimistic, "
              f"biased by having been the max over many noisy eval checkpoints):")
        candidates = [("latest.pt (final)", model)]
        if best_eval_iteration is not None:
            best_model = build_model(args.model_type, args.hidden, args.gnn_layers,
                                      public_hand_features=args.public_hand_features)
            ckpt = torch.load(os.path.join(args.checkpoint_dir, "best_eval.pt"), map_location="cpu")
            best_model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
            candidates.append((f"best_eval.pt (iter {best_eval_iteration})", best_model))
        confirmed = []
        for label, m in candidates:
            res = evaluate_policy(m, args.select_best_opponent, args.final_eval_games,
                                   seed_base=9_000_000, model_kind=args.model_type,
                                   num_workers=args.num_workers,
                                   public_hand_features=args.public_hand_features)
            print(f"  {label:<28s} win_rate={res['win_rate']:.1%} avg_vp={res['avg_vp']:.2f}")
            confirmed.append((label, res["win_rate"]))
        best_label, _ = max(confirmed, key=lambda x: x[1])
        print(f"  -> recommended: {best_label}")


if __name__ == "__main__":
    main()
