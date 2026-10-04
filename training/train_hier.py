"""PPO self-play training CLI using the hierarchical/pointer action head
(training/hier_model.py, training/hier_ppo.py) -- supersedes the old flat
index-into-legal_actions model (training/model.ActorCritic, since removed),
which was shown (see README "Status") to collapse once trading was enabled.
Evaluates against
both the fixed random/heuristic tiers in the *full* ruleset (§12's "unmoving
yardstick") and, importantly, under the *same* curriculum settings training
is currently using -- that same-distribution/full-ruleset split is exactly
what isolated the flat model's failure mode last time, so it's built in here
rather than reconstructed ad hoc.
"""
from __future__ import annotations

import argparse
import math
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
from training.agent import HierarchicalLearnedAgent
from training.hier_model import HierarchicalActorCritic
from training.hier_ppo import (
    GAE_LAMBDA, GAMMA, collect_rollout, collect_rollout_parallel, ppo_update, reseed_forked_worker,
)
from training.model import observation_dim
from training.model_adapters import ADAPTERS
from training.run_manifest import write_manifest

OPPONENT_FACTORIES = {"random": RandomAgent, "heuristic": HeuristicAgent}


def build_model(model_type: str, hidden: int, gnn_layers: int,
                public_hand_features: bool = False):
    if model_type == "gnn":
        from training.gnn_model import GraphActorCritic
        return GraphActorCritic(hidden=hidden, gnn_layers=gnn_layers,
                                 public_hand_features=public_hand_features)
    return HierarchicalActorCritic(obs_dim=observation_dim(public_hand_features), hidden=hidden)


def move_optimizer_state(optimizer: torch.optim.Optimizer, device: str) -> None:
    """Move an optimizer's internal per-parameter state tensors (Adam's
    exp_avg/exp_avg_sq) to `device`. `model.to(device)` only moves the
    model's own parameters/buffers -- it has no effect on the optimizer's
    separate state dict, so a restored (e.g. via --init-checkpoint) optimizer
    whose state was saved on one device will otherwise device-mismatch the
    moment .step() runs against gradients computed on a different one, which
    is exactly what happens here since rollout always runs on CPU while the
    PPO update moves the model to `device` and back every iteration."""
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)


def env_kwargs_from_args(args: argparse.Namespace) -> dict:
    return dict(
        randomize_board=args.randomize_board,
        allow_trading=not args.no_trading,
        allow_dev_cards=not args.no_dev_cards,
        vp_shaping_weight=args.vp_shaping_weight,
        max_episode_steps=args.max_episode_steps,
        public_hand_features=args.public_hand_features,
        truncation_reward=getattr(args, "truncation_reward", "zero"),
    )


def _play_eval_game(model, opponent_cls, seed: int, randomize_board: bool, allow_trading: bool,
                     allow_dev_cards: bool, max_steps: int, model_kind: str,
                     public_hand_features: bool = False) -> tuple[bool, bool, int]:
    """Trainee (deterministic) vs. 3 copies of `opponent_cls`, the trainee's
    seat rotated by seed (`seed % 4`). Seats are not symmetric -- seat 0 wins
    ~28% of heuristic-vs-heuristic games vs ~23-25% for the others -- so a
    fixed seat biases the win rate (audit F-14). Use a game count divisible
    by 4 to balance seats exactly. Returns (won, finished, vp)."""
    engine = CatanEngine(randomize_board=randomize_board, seed=seed,
                          allow_trading=allow_trading, allow_dev_cards=allow_dev_cards)
    seat = seed % NUM_PLAYERS
    agents = {seat: HierarchicalLearnedAgent(seat, model=model, deterministic=True, model_kind=model_kind,
                                              public_hand_features=public_hand_features)}
    for pid in range(NUM_PLAYERS):
        if pid != seat:
            agents[pid] = opponent_cls(pid, random.Random(seed * 97 + pid))
    steps = 0
    while not engine.done and steps < max_steps:
        actor = engine.acting_player()
        action = agents[actor].choose(engine.state)
        engine.step(action)
        steps += 1
    finished = engine.done
    won = finished and engine.state.winner == seat
    return won, finished, total_vp(engine.state, seat)


def wilson_ci(wins: int, games: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a win rate."""
    if games == 0:
        return 0.0, 1.0
    p = wins / games
    d = 1 + z * z / games
    c = (p + z * z / (2 * games)) / d
    h = z * math.sqrt(p * (1 - p) / games + z * z / (4 * games * games)) / d
    return c - h, c + h


# -- parallel evaluation -----------------------------------------------------
# Same fork-based worker-pool pattern as training/hier_ppo.py's
# collect_rollout_parallel: eval games are independent and read-only w.r.t.
# the model (deterministic policy, no gradient), so they parallelize the same
# way rollout collection does, removing eval as a synchronous stall in the
# training loop.
_eval_worker_model = None
_eval_worker_opponent_cls = None
_eval_worker_kwargs: dict = {}


def _init_eval_worker(model, opponent_cls, kwargs: dict) -> None:
    global _eval_worker_model, _eval_worker_opponent_cls, _eval_worker_kwargs
    torch.set_num_threads(1)
    reseed_forked_worker()
    model.eval()
    _eval_worker_model = model
    _eval_worker_opponent_cls = opponent_cls
    _eval_worker_kwargs = kwargs


def _eval_worker_play(seeds: list[int]) -> tuple[int, int, int]:
    wins = finished_n = vp_sum = 0
    for seed in seeds:
        won, finished, vp = _play_eval_game(_eval_worker_model, _eval_worker_opponent_cls, seed,
                                             **_eval_worker_kwargs)
        wins += int(won)
        finished_n += int(finished)
        vp_sum += vp
    return wins, finished_n, vp_sum


def evaluate_policy(model, opponent_kind: str, games: int, seed_base: int,
                     randomize_board: bool = True, allow_trading: bool = True,
                     allow_dev_cards: bool = True, max_steps: int = 4000,
                     model_kind: str = "hier", num_workers: int = 1,
                     public_hand_features: bool = False) -> dict:
    opponent_cls = OPPONENT_FACTORIES[opponent_kind]
    kwargs = dict(randomize_board=randomize_board, allow_trading=allow_trading,
                  allow_dev_cards=allow_dev_cards, max_steps=max_steps, model_kind=model_kind,
                  public_hand_features=public_hand_features)
    seeds = [seed_base + i for i in range(games)]

    if num_workers <= 1:
        wins, finished, vp_sum = 0, 0, 0
        for seed in seeds:
            won, fin, vp = _play_eval_game(model, opponent_cls, seed, **kwargs)
            wins += int(won)
            finished += int(fin)
            vp_sum += vp
    else:
        num_workers = max(1, min(num_workers, games))
        chunks = [seeds[i::num_workers] for i in range(num_workers)]
        chunks = [c for c in chunks if c]
        ctx = mp.get_context("fork")
        with ctx.Pool(processes=len(chunks), initializer=_init_eval_worker,
                      initargs=(model, opponent_cls, kwargs)) as pool:
            results = pool.map(_eval_worker_play, chunks)
        wins = sum(r[0] for r in results)
        finished = sum(r[1] for r in results)
        vp_sum = sum(r[2] for r in results)

    return {"win_rate": wins / games, "wins": wins, "games": games, "ci95": wilson_ci(wins, games),
            "finish_rate": finished / games, "avg_vp": vp_sum / games}


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
    parser.add_argument("--truncation-reward", choices=["zero", "rank"], default="zero",
                         help="what a step-cap truncation pays: 'zero' (default; GAE bootstraps V(s_T)) or 'rank' (legacy rank-on-standing, pays the VP leader a full win -- audit F-06)")
    parser.add_argument("--vp-shaping-weight", type=float, default=0.05)
    parser.add_argument("--no-trading", action="store_true")
    parser.add_argument("--no-dev-cards", action="store_true")
    parser.add_argument("--randomize-board", action="store_true", default=False)
    parser.add_argument("--public-hand-features", action="store_true",
                         help="expose the engine's publicly-inferable per-player resource "
                              "estimates (card counting) as observation features. Widens the "
                              "observation, so checkpoints are only compatible across runs "
                              "using the same setting of this flag.")
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--eval-games", type=int, default=30)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints_hier")
    parser.add_argument("--init-checkpoint", type=str, default=None,
                         help="resume/fine-tune from a prior checkpoint's weights, e.g. to "
                              "advance a curriculum stage rather than restarting from scratch")
    parser.add_argument("--num-workers", type=int, default=1,
                         help="parallel rollout-collection processes (fork-based); "
                              "1 = sequential, no multiprocessing overhead. ~12 is a good "
                              "default on a 16-core machine -- see README for the benchmark.")
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
    write_manifest(args.checkpoint_dir, args, {"driver": "train_hier"})
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
        # Restore Adam's moment estimates too -- without this, every curriculum-
        # stage resume via --init-checkpoint restarted them at zero, causing a
        # transient effective-LR spike right after every resume.
        optimizer.load_state_dict(init_ckpt["optimizer"])
        move_optimizer_state(optimizer, device)
        print("restored optimizer state from init checkpoint")
    env_kwargs = env_kwargs_from_args(args)
    env = CatanAECEnv(**env_kwargs) if args.num_workers <= 1 else None

    base_seed = args.seed
    for iteration in range(1, args.iterations + 1):
        t0 = time.time()
        if args.num_workers <= 1:
            transitions, summaries = collect_rollout(env, model, "cpu", args.episodes_per_iter, base_seed,
                                                       adapter=adapter, gamma=args.gamma, lam=args.gae_lambda)
        else:
            transitions, summaries = collect_rollout_parallel(
                env_kwargs, model, args.episodes_per_iter, base_seed, args.num_workers, adapter=adapter,
                gamma=args.gamma, lam=args.gae_lambda)
        base_seed += args.episodes_per_iter
        model.to(device)
        stats = ppo_update(model, optimizer, transitions, epochs=args.epochs,
                            minibatch_size=args.minibatch_size, adapter=adapter, device=device,
                            clip_ratio=args.clip_ratio, value_coef=args.value_coef,
                            entropy_coef=args.entropy_coef, max_grad_norm=args.max_grad_norm,
                            target_kl=(args.target_kl if args.target_kl >= 0 else None))
        model.to("cpu")
        elapsed = time.time() - t0

        mean_turns = sum(s["turns"] for s in summaries) / len(summaries)
        finish_rate = sum(1 for s in summaries if s["winner"] is not None) / len(summaries)
        print(f"iter {iteration:4d} | {len(transitions):5d} steps | {elapsed:5.1f}s | "
              f"turns={mean_turns:5.1f} finish={finish_rate:4.0%} | "
              f"pol={stats['policy_loss']:+.4f} val={stats['value_loss']:.4f} "
              f"ent={stats['entropy']:.3f} kl={stats['approx_kl']:.4f}")

        # Crash-resilient checkpoint: a few MB and <100ms per iteration, vs.
        # losing up to eval_every iterations of progress on an interruption.
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "iteration": iteration, "args": vars(args)},
                   os.path.join(args.checkpoint_dir, "latest.pt"))

        if iteration % args.eval_every == 0 or iteration == args.iterations:
            for opp in ("random", "heuristic"):
                # Seed bases stride by eval_games so successive evals use
                # disjoint game sets (plain `+ iteration` made consecutive
                # evals share most of their seeds).
                same = evaluate_policy(model, opp, args.eval_games,
                                        seed_base=800_000 + iteration * args.eval_games,
                                        randomize_board=args.randomize_board,
                                        allow_trading=not args.no_trading,
                                        allow_dev_cards=not args.no_dev_cards,
                                        model_kind=args.model_type, num_workers=args.num_workers,
                                        public_hand_features=args.public_hand_features)
                full = evaluate_policy(model, opp, args.eval_games,
                                        seed_base=2_000_000 + iteration * args.eval_games,
                                        model_kind=args.model_type, num_workers=args.num_workers,
                                        public_hand_features=args.public_hand_features)
                print(f"  eval vs {opp:<10} [same-dist] win_rate={same['win_rate']:.0%} "
                      f"(95% CI {same['ci95'][0]:.0%}-{same['ci95'][1]:.0%}) "
                      f"finish_rate={same['finish_rate']:.0%} avg_vp={same['avg_vp']:.2f}  "
                      f"[full-ruleset] win_rate={full['win_rate']:.0%} "
                      f"(95% CI {full['ci95'][0]:.0%}-{full['ci95'][1]:.0%}) "
                      f"finish_rate={full['finish_rate']:.0%} avg_vp={full['avg_vp']:.2f}")
            ckpt_path = os.path.join(args.checkpoint_dir, f"iter_{iteration}.pt")
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "iteration": iteration, "args": vars(args)}, ckpt_path)
            print(f"  saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
