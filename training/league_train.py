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
import multiprocessing as mp
import os
import random
import time

import numpy as np
import torch

from agents.heuristic import HeuristicAgent
from agents.random_agent import RandomAgent
from env.engine import CatanEngine, total_vp
from env.pettingzoo_env import CatanAECEnv
from env.state import NUM_PLAYERS
from training.agent import HierarchicalLearnedAgent, load_gnn_model, load_hier_model
from training.hier_model import HierarchicalActorCritic
from training.hier_ppo import GAE_LAMBDA, GAMMA, collect_rollout, collect_rollout_parallel, ppo_update
from training.league import League
from training.model import observation_dim
from training.model_adapters import ADAPTERS
from training.train_hier import build_model, evaluate_policy, move_optimizer_state

MAX_EVAL_STEPS = 4000


_LOADED_MODEL_CACHE: dict[tuple[str, str, int, int], object] = {}


def load_model_for(model_type: str, checkpoint_path: str, hidden: int, gnn_layers: int):
    """Cached by (model_type, checkpoint_path, hidden, gnn_layers): every
    checkpoint file is written once under a unique name (historical/main/
    main_exploiter snapshots each get their own filename per iteration) and
    never mutated afterward, so it's safe -- and much cheaper -- to load a
    given checkpoint from disk only once per process instead of every time an
    opponent is sampled or evaluated against."""
    key = (model_type, checkpoint_path, hidden, gnn_layers)
    if key not in _LOADED_MODEL_CACHE:
        if model_type == "gnn":
            _LOADED_MODEL_CACHE[key] = load_gnn_model(checkpoint_path, hidden=hidden, gnn_layers=gnn_layers)
        else:
            _LOADED_MODEL_CACHE[key] = load_hier_model(checkpoint_path, hidden=hidden)
    return _LOADED_MODEL_CACHE[key]


def env_kwargs_from_args(args: argparse.Namespace) -> dict:
    return dict(
        randomize_board=args.randomize_board,
        allow_trading=not args.no_trading,
        allow_dev_cards=not args.no_dev_cards,
        vp_shaping_weight=args.vp_shaping_weight,
        max_episode_steps=args.max_episode_steps,
    )


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
        args.model_type, member.checkpoint_path, args.hidden, args.gnn_layers)
    return {pid: HierarchicalLearnedAgent(pid, model=model, deterministic=False, model_kind=args.model_type)
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
    This makes each game a pure function of its own seed either way."""
    torch.manual_seed(seed)
    engine = CatanEngine(randomize_board=env_kwargs["randomize_board"], seed=seed,
                          allow_trading=env_kwargs["allow_trading"], allow_dev_cards=env_kwargs["allow_dev_cards"])
    trainee_seat = seed % NUM_PLAYERS
    opponent_seats = [p for p in range(NUM_PLAYERS) if p != trainee_seat]
    agents = make_seat_agents(member, opponent_seats, args, random.Random(seed), shared_model=shared_model)
    agents[trainee_seat] = HierarchicalLearnedAgent(trainee_seat, model=model, deterministic=True,
                                                     model_kind=args.model_type)
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
        shared_model = load_model_for(args.model_type, member.checkpoint_path, args.hidden, args.gnn_layers)

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
    exploiter_model = build_model(args.model_type, args.hidden, args.gnn_layers)
    ckpt = torch.load(main_member.checkpoint_path, map_location="cpu")
    exploiter_model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    optimizer = torch.optim.Adam(exploiter_model.parameters(), lr=args.lr)
    frozen_main = load_model_for(args.model_type, main_member.checkpoint_path, args.hidden, args.gnn_layers)

    rng = random.Random(iteration * 7919)
    base_seed = 500_000 + iteration * 1000
    for _ in range(args.exploiter_iterations):
        trainee_seat = rng.randrange(NUM_PLAYERS)
        opponent_seats = [p for p in range(NUM_PLAYERS) if p != trainee_seat]
        opponent_agents = {pid: HierarchicalLearnedAgent(pid, model=frozen_main, deterministic=False,
                                                          model_kind=args.model_type)
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
    parser.add_argument("--max-episode-steps", type=int, default=800)
    parser.add_argument("--vp-shaping-weight", type=float, default=0.05)
    parser.add_argument("--no-trading", action="store_true")
    parser.add_argument("--no-dev-cards", action="store_true")
    parser.add_argument("--randomize-board", action="store_true", default=True)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--self-play-prob", type=float, default=0.5,
                         help="fraction of iterations that are pure self-play vs. a sampled league opponent")
    parser.add_argument("--snapshot-every", type=int, default=25,
                         help="add the current trainee to the league as a historical member every N iterations")
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--eval-games", type=int, default=30)
    parser.add_argument("--promotion-games", type=int, default=100)
    parser.add_argument("--promotion-win-rate", type=float, default=0.55)
    parser.add_argument("--promotion-alpha", type=float, default=0.05)
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
    league = League.load(args.checkpoint_dir)
    adapter = ADAPTERS[args.model_type]

    model = build_model(args.model_type, args.hidden, args.gnn_layers)
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

    env_kwargs = env_kwargs_from_args(args)
    rng = random.Random(args.seed)
    base_seed = args.seed

    for iteration in range(1, args.iterations + 1):
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

        model.to(device)
        stats = ppo_update(model, optimizer, transitions, epochs=args.epochs, minibatch_size=args.minibatch_size,
                            adapter=adapter, device=device, clip_ratio=args.clip_ratio,
                            value_coef=args.value_coef, entropy_coef=args.entropy_coef,
                            max_grad_norm=args.max_grad_norm,
                            target_kl=(args.target_kl if args.target_kl >= 0 else None))
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
        print(f"iter {iteration:4d} | {mode:<20s} | {len(transitions):5d} steps | {elapsed:5.1f}s | "
              f"turns={mean_turns:5.1f} finish={finish_rate:4.0%} | "
              f"pol={stats['policy_loss']:+.4f} val={stats['value_loss']:.4f} "
              f"ent={stats['entropy']:.3f} kl={stats['approx_kl']:.4f}")

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

        if iteration % args.eval_every == 0 or iteration == args.iterations:
            if league.main_name is not None:
                wins, games, finished = evaluate_vs_member(
                    model, league, league.main_name, args.promotion_games,
                    seed_base=700_000 + iteration, env_kwargs=env_kwargs, args=args,
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

            for opp in ("random", "heuristic"):
                res = evaluate_policy(model, opp, args.eval_games, seed_base=900_000 + iteration,
                                       model_kind=args.model_type, num_workers=args.num_workers)
                print(f"  eval vs {opp:<10} win_rate={res['win_rate']:.0%} "
                      f"finish_rate={res['finish_rate']:.0%} avg_vp={res['avg_vp']:.2f}")

            ratings_str = ", ".join(f"{m.name}={league.rating(m.name).mu:.1f}"
                                     for m in sorted(league.members.values(), key=lambda m: -league.rating(m.name).mu)[:6])
            print(f"  top ratings: {ratings_str}")
            league.save()


if __name__ == "__main__":
    main()
