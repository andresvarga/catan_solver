"""PPO self-play training CLI (roadmap phase 4).

One shared policy plays all four seats (`training/ppo.py`), against a flat-
vector model (`training/model.py`). Curriculum flags let this run against a
reduced action space first (§8: fixed board, no trading, no dev cards)
before graduating to the full ruleset. Periodic evaluation is against the
*fixed* random/heuristic tiers from phase 3 in the full ruleset, regardless
of what curriculum stage training is currently in -- exactly the "decoupled
from the moving target" evaluation practice from §12.
"""
from __future__ import annotations

import argparse
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
from training.agent import LearnedAgent
from training.model import ActorCritic, observation_dim
from training.ppo import GAE_LAMBDA, GAMMA, collect_rollout, ppo_update

OPPONENT_FACTORIES = {"random": RandomAgent, "heuristic": HeuristicAgent}


def make_env(args: argparse.Namespace) -> CatanAECEnv:
    return CatanAECEnv(
        randomize_board=args.randomize_board,
        allow_trading=not args.no_trading,
        allow_dev_cards=not args.no_dev_cards,
        vp_shaping_weight=args.vp_shaping_weight,
        max_episode_steps=args.max_episode_steps,
    )


def evaluate_policy(model: ActorCritic, opponent_kind: str, games: int, seed_base: int,
                     max_steps: int = 4000) -> dict:
    """Learned agent (deterministic, seat 0) vs. three fixed-tier opponents,
    always in the *full* ruleset -- an unmoving yardstick, per §12."""
    opponent_cls = OPPONENT_FACTORIES[opponent_kind]
    wins, finished, vp_sum = 0, 0, 0
    for i in range(games):
        seed = seed_base + i
        engine = CatanEngine(randomize_board=True, seed=seed)
        agents = {0: LearnedAgent(0, model=model, deterministic=True)}
        for pid in range(1, NUM_PLAYERS):
            agents[pid] = opponent_cls(pid, random.Random(seed * 97 + pid))
        steps = 0
        while not engine.done and steps < max_steps:
            actor = engine.acting_player()
            action = agents[actor].choose(engine.state)
            engine.step(action)
            steps += 1
        if engine.done:
            finished += 1
            if engine.state.winner == 0:
                wins += 1
        vp_sum += total_vp(engine.state, 0)
    return {"win_rate": wins / games, "finish_rate": finished / games, "avg_vp": vp_sum / games}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--episodes-per-iter", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=512)
    parser.add_argument("--max-episode-steps", type=int, default=600)
    parser.add_argument("--vp-shaping-weight", type=float, default=0.05)
    parser.add_argument("--no-trading", action="store_true")
    parser.add_argument("--no-dev-cards", action="store_true")
    parser.add_argument("--randomize-board", action="store_true", default=False)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--eval-games", type=int, default=30)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
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
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    obs_dim = observation_dim()
    model = ActorCritic(obs_dim=obs_dim, hidden=args.hidden)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    env = make_env(args)

    base_seed = args.seed
    for iteration in range(1, args.iterations + 1):
        t0 = time.time()
        transitions, summaries = collect_rollout(env, model, "cpu", args.episodes_per_iter, base_seed,
                                                   gamma=args.gamma, lam=args.gae_lambda)
        base_seed += args.episodes_per_iter
        stats = ppo_update(model, optimizer, transitions, epochs=args.epochs,
                            minibatch_size=args.minibatch_size, clip_ratio=args.clip_ratio,
                            value_coef=args.value_coef, entropy_coef=args.entropy_coef,
                            max_grad_norm=args.max_grad_norm,
                            target_kl=(args.target_kl if args.target_kl >= 0 else None))
        elapsed = time.time() - t0

        mean_turns = sum(s["turns"] for s in summaries) / len(summaries)
        finish_rate = sum(1 for s in summaries if s["winner"] is not None) / len(summaries)
        print(f"iter {iteration:4d} | {len(transitions):5d} steps | {elapsed:5.1f}s | "
              f"turns={mean_turns:5.1f} finish={finish_rate:4.0%} | "
              f"pol={stats['policy_loss']:+.4f} val={stats['value_loss']:.4f} "
              f"ent={stats['entropy']:.3f} kl={stats['approx_kl']:.4f}")

        if iteration % args.eval_every == 0 or iteration == args.iterations:
            for opp in ("random", "heuristic"):
                res = evaluate_policy(model, opp, args.eval_games, seed_base=900_000 + iteration)
                print(f"  eval vs {opp:<10} win_rate={res['win_rate']:.0%} "
                      f"finish_rate={res['finish_rate']:.0%} avg_vp={res['avg_vp']:.2f}")
            ckpt_path = os.path.join(args.checkpoint_dir, f"iter_{iteration}.pt")
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "iteration": iteration, "args": vars(args)}, ckpt_path)
            print(f"  saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
