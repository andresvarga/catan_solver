"""PPO fine-tuning with one trainee seat (rotated every batch) against a
mixed opponent pool by default (agents/opponent_pool.py: heuristic, honest
heuristic, search agent, random, optional past checkpoints), or -- with
`--opponent-pool heuristic` -- the original condition of 3 plain
HeuristicAgents. Training on one fixed bot optimises exploiting that bot;
the pool trades a little of that for robustness to other styles. The
in-loop evaluation is still vs 3 heuristics (seat-rotated).

Why this exists (and why it isn't league_train): imitation has been squeezed
dry -- the student matches the strongest buildable demonstrator (~55%
confirmed vs 3 heuristics), and every teacher upgrade tried past that
(deeper same-turn search, determinized rollouts, expert-iteration rollouts)
lands within +1.5pp of the depth-2 search teacher because they all share the
same heuristic leaf evaluator. RL replaces that evaluator with the ground
truth (win/loss), so it is the one axis that can pass the teacher rather
than approach it. The league's robustness machinery (TrueSkill, promotion,
snapshot opponents) is deliberately not used here: the target metric IS
"win rate vs 3 plain heuristics", so training on exactly that condition is
optimizing the objective, not overfitting to a proxy.

Guardrails carried over from the league loop:
- BC anchor (hier_ppo.load_bc_anchor): each gradient step penalizes NLL of
  demonstrated actions, so PPO can only leave the cloned policy where the
  surrogate gain beats the penalty (protects skills the sparse reward would
  take many episodes to rediscover).
- holdout NLL drift monitor + fixed-seed eval every --eval-every iterations;
  best-eval checkpoint kept alongside latest.pt (crash-resumable).
- KL early stop per update (target_kl), lr well below the BC lr.

Seat rotation: Catan seats are not symmetric (setup order), and the eval
protocol (train_hier.evaluate_policy) rotates the model's seat by seed -- so
each iteration splits its episode budget evenly across the 4 trainee seats.
`best_eval.pt` is the max of many noisy in-loop reads (winner's curse): treat
it as a candidate and confirm it on fresh seed bases before claiming a gain.
"""
from __future__ import annotations

import argparse
import os
import random
import time

import numpy as np
import torch

from evaluation.seeds import check_training_seeds, load_registry
from agents.heuristic import HeuristicAgent
from agents.opponent_pool import PooledOpponent, builtin_members, checkpoint_member
from env.state import NUM_PLAYERS
from training.hier_ppo import (
    collect_rollout_parallel, compute_holdout_nll, load_bc_anchor, ppo_update,
)
from training.imitation_data import split_by_game
from training.model_adapters import ADAPTERS
from training.run_manifest import write_manifest
from training.ppo import GAE_LAMBDA, GAMMA
from training.train_hier import build_model, evaluate_policy, wilson_ci


def fixed_seed_eval(model, games_per_set: int, num_workers: int,
                     public_hand_features: bool, seed_bases: tuple[int, int],
                     model_type: str) -> tuple[float, float, tuple[float, float]]:
    """Seat-rotated (evaluate_policy rotates the trainee by seed % 4) win
    rate, avg VP and 95% Wilson CI pooled over both seed bases."""
    wins = vp = 0.0
    for base in seed_bases:
        res = evaluate_policy(model, "heuristic", games_per_set, seed_base=base,
                               model_kind=model_type, num_workers=num_workers,
                               public_hand_features=public_hand_features)
        wins += res["wins"]
        vp += res["avg_vp"] * games_per_set
    total = 2 * games_per_set
    return wins / total, vp / total, wilson_ci(int(wins), total)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--init-checkpoint", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--model-type", choices=["hier", "gnn"], default="gnn")
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--gnn-layers", type=int, default=4)
    parser.add_argument("--public-hand-features", action="store_true")
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--episodes-per-iter", type=int, default=48,
                         help="split evenly across the 4 trainee seats")
    parser.add_argument("--num-workers", type=int, default=12)
    parser.add_argument("--lr", type=float, default=1e-4,
                         help="well below BC's 1e-3: the policy starts good")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--target-kl", type=float, default=0.02)
    parser.add_argument("--gamma", type=float, default=GAMMA)
    parser.add_argument("--gae-lambda", type=float, default=GAE_LAMBDA)
    parser.add_argument("--max-episode-steps", type=int, default=4000,
                         help="step cap; heuristic games take ~1,100 steps (p90 ~1,450), so a "
                              "cap below ~2,500 truncates most games (audit F-06)")
    parser.add_argument("--terminal-reward", choices=["win_loss", "rank"], default="win_loss",
                         help="game-end reward: 'win_loss' (+1 winner, -1/3 each loser; default) or "
                              "legacy 'rank' placement reward {1, 0, -0.5, -1} (audit F-16)")
    parser.add_argument("--truncation-reward", choices=["zero", "rank"], default="zero",
                         help="what a step-cap truncation pays: 'zero' (default; GAE bootstraps V(s_T)) or 'rank' (legacy rank-on-standing, pays the VP leader a full win -- audit F-06)")
    parser.add_argument("--bc-anchor-dataset", type=str, default=None)
    parser.add_argument("--bc-anchor-coef", type=float, default=0.2)
    parser.add_argument("--bc-anchor-samples", type=int, default=100_000)
    parser.add_argument("--bc-anchor-minibatch", type=int, default=256,
                         help="anchor rows per gradient step; its activations coexist "
                              "with the policy minibatch's on the GPU, so keep it <= "
                              "--minibatch-size on the 6GB card")
    parser.add_argument("--bc-holdout-samples", type=int, default=20_000,
                         help="rows reserved from the anchor dataset (disjoint "
                              "from the anchor subsample) for the drift monitor")
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--eval-games", type=int, default=120,
                         help="per seed base (two bases); keep divisible by 4 so the seat "
                              "rotation (seed %% 4) is balanced")
    parser.add_argument("--eval-seed-bases", type=int, nargs=2,
                         default=load_registry()["sets"]["inloop"]["bases"],
                         help="in-loop selection signal only (default: the registry's 'inloop' "
                              "set) -- final claims need scripts/evaluate_candidate.py on a "
                              "registered confirmation set")
    parser.add_argument("--opponent-pool", choices=["mixed", "heuristic"], default="mixed",
                         help="'mixed' (default): each opponent seat draws a style per episode from "
                              "{heuristic, honest heuristic, search, random} (+ --pool-checkpoint); "
                              "'heuristic': legacy 3x HeuristicAgent (the eval condition itself)")
    parser.add_argument("--pool-weights", type=str, default=None,
                         help="override pool weights, e.g. 'heuristic=0.5,search=0.5'")
    parser.add_argument("--pool-checkpoint", action="append", default=[],
                         help="add a frozen past checkpoint (same --model-type/--hidden) to the pool; "
                              "repeatable")
    parser.add_argument("--pool-checkpoint-weight", type=float, default=0.15)
    parser.add_argument("--envs-per-worker", type=int, default=4,
                         help="games each rollout worker plays concurrently, with one batched forward "
                              "pass per round (Phase 4); data is identical for any value")
    parser.add_argument("--rollout-device", type=str, default="cpu",
                         help="device for rollout inference: 'cpu' (forked workers) or 'cuda' "
                              "(spawned GPU workers; ~3-7x faster for the GNN, slower for the flat "
                              "model). With --num-workers 1 it runs in-process.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    os.makedirs(args.out_dir, exist_ok=True)
    write_manifest(args.out_dir, args, {"driver": "rl_finetune"})
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    adapter = ADAPTERS[args.model_type]

    model = build_model(args.model_type, args.hidden, args.gnn_layers,
                         public_hand_features=args.public_hand_features)
    ckpt = torch.load(args.init_checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    model.eval()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    pool_members = None
    if args.opponent_pool == "mixed":
        weights = None
        if args.pool_weights:
            weights = {k: float(v) for k, v in (kv.split("=") for kv in args.pool_weights.split(","))}
        pool_members = builtin_members(weights)
        for path in args.pool_checkpoint:
            pool_members.append(checkpoint_member(path, args.model_type, args.hidden, args.gnn_layers,
                                                  args.pool_checkpoint_weight,
                                                  public_hand_features=args.public_hand_features))
        print("opponent pool: " + ", ".join(f"{m.name}={m.weight:g}" for m in pool_members), flush=True)

    env_kwargs = dict(randomize_board=True, allow_trading=True, allow_dev_cards=True,
                      vp_shaping_weight=0.0, max_episode_steps=args.max_episode_steps,
                      public_hand_features=args.public_hand_features,
                      truncation_reward=args.truncation_reward,
                      terminal_reward=args.terminal_reward)

    bc_anchor = holdout = None
    if args.bc_anchor_dataset:
        # holdout first (deterministic rows), anchor sampled from the rest --
        # disjoint by construction so holdout NLL measures drift, not fit
        full = np.load(args.bc_anchor_dataset)
        n_full = full["type_idx"].shape[0]
        # whole games held out when the dataset records game ids (audit F-21)
        hold_idx, anchor_pool = split_by_game(
            n_full, full["game_id"] if "game_id" in full.files else None,
            args.bc_holdout_samples, seed=args.seed)
        anchor_idx = anchor_pool[:args.bc_anchor_samples]
        def take(idx):
            sub = {k: torch.as_tensor(full[k][np.sort(idx)], device=device) for k in full.files}
            obs = {k[len("obs_"):]: v for k, v in sub.items() if k.startswith("obs_")}
            rest = {k: v for k, v in sub.items() if not k.startswith("obs_")}
            return ({"obs": obs, **rest}) if obs else \
                {"obs": sub["obs"], **{k: v for k, v in sub.items() if k != "obs"}}
        bc_anchor = take(anchor_idx)
        holdout = take(hold_idx)
        # batch_size sized for the h=256/L=4 GNN on the 6GB card (the 8192
        # default is for the small flat model and OOMs here)
        nll0 = compute_holdout_nll(model.to(device), holdout, batch_size=1024)
        model.to("cpu")
        print(f"bc anchor: {len(anchor_idx)} rows, holdout {len(hold_idx)} rows, "
              f"init holdout NLL={nll0:.4f}", flush=True)

    win0, vp0, ci0 = fixed_seed_eval(model, args.eval_games, args.num_workers,
                                      args.public_hand_features, tuple(args.eval_seed_bases),
                                      args.model_type)
    print(f"iter 0 (init): win_rate={win0:.1%} (95% CI {ci0[0]:.1%}-{ci0[1]:.1%}) avg_vp={vp0:.2f}",
          flush=True)
    best_win = win0
    history = [(0, win0, vp0)]

    per_seat = max(1, args.episodes_per_iter // NUM_PLAYERS)
    check_training_seeds(args.seed * 1_000_000,
                         args.seed * 1_000_000 + args.iterations * 1_000 + 3 * 250 + per_seat,
                         "rl_finetune rollout")
    for it in range(1, args.iterations + 1):
        t0 = time.time()
        transitions = []
        finished = wins = 0
        style_games: dict[str, int] = {}
        style_wins: dict[str, int] = {}
        for seat in range(NUM_PLAYERS):
            if pool_members is None:  # legacy: 3x the evaluation heuristic
                opponents = {pid: HeuristicAgent(pid, random.Random(args.seed * 917 + it * 31 + pid))
                             for pid in range(NUM_PLAYERS) if pid != seat}
            else:
                opponents = {pid: PooledOpponent(pid, pool_members,
                                                 random.Random(args.seed * 917 + it * 31 + pid))
                             for pid in range(NUM_PLAYERS) if pid != seat}
            base_seed = args.seed * 1_000_000 + it * 1_000 + seat * 250
            trs, summaries = collect_rollout_parallel(
                env_kwargs, model, per_seat, base_seed, args.num_workers,
                opponent_agents=opponents, opponent_name=args.opponent_pool, adapter=adapter,
                gamma=args.gamma, lam=args.gae_lambda,
                envs_per_worker=args.envs_per_worker, inference_device=args.rollout_device)
            transitions.extend(trs)
            for s in summaries:
                finished += 0 if s["truncated"] else 1
                wins += 1 if s["winner"] == seat else 0
                for style in set(s.get("opponent_styles", {}).values()):
                    style_games[style] = style_games.get(style, 0) + 1
                    style_wins[style] = style_wins.get(style, 0) + (s["winner"] == seat)
        t_roll = time.time() - t0

        model.to(device).train()
        stats = ppo_update(model, optimizer, transitions,
                            clip_ratio=args.clip_ratio, value_coef=args.value_coef,
                            entropy_coef=args.entropy_coef, epochs=args.epochs,
                            minibatch_size=args.minibatch_size, device=device,
                            adapter=adapter, target_kl=args.target_kl,
                            bc_dataset=bc_anchor, bc_coef=args.bc_anchor_coef,
                            bc_minibatch_size=args.bc_anchor_minibatch)
        nll = compute_holdout_nll(model, holdout, batch_size=1024) \
            if holdout is not None else float("nan")
        model.to("cpu").eval()

        n_ep = per_seat * NUM_PLAYERS
        bc_str = f" bc={stats['bc_loss']:.3f}" if "bc_loss" in stats else ""
        print(f"iter {it:3d}: {len(transitions)} transitions, rollout wins {wins}/{n_ep} "
              f"(finished {finished}/{n_ep}) | pi={stats['policy_loss']:.4f} "
              f"v={stats['value_loss']:.4f} kl={stats['approx_kl']:.4f}{bc_str} "
              f"gnorm={stats['grad_norm']:.2f} ev={stats['explained_variance']:.2f} "
              f"holdout_nll={nll:.4f} | roll {t_roll:.0f}s upd {time.time()-t0-t_roll:.0f}s",
              flush=True)
        if pool_members is not None:  # trainee win rate in games where each style sat at the table
            print("          pool: " + "  ".join(
                f"{k.split('/')[-1]}={style_wins[k] / max(1, style_games[k]):.0%} (n={style_games[k]})"
                for k in sorted(style_games)), flush=True)
        torch.save({"model": model.state_dict(), "iteration": it},
                   os.path.join(args.out_dir, "latest.pt"))

        if it % args.eval_every == 0:
            win, vp, ci = fixed_seed_eval(model, args.eval_games, args.num_workers,
                                           args.public_hand_features, tuple(args.eval_seed_bases),
                                           args.model_type)
            history.append((it, win, vp))
            marker = ""
            if win > best_win:
                best_win = win
                torch.save({"model": model.state_dict(), "iteration": it, "win_rate": win},
                           os.path.join(args.out_dir, "best_eval.pt"))
                marker = "  <-- new best, saved best_eval.pt"
            print(f"iter {it:3d}: EVAL win_rate={win:.1%} (95% CI {ci[0]:.1%}-{ci[1]:.1%}) "
                  f"avg_vp={vp:.2f}{marker}", flush=True)

    print("\niter | win_rate | avg_vp")
    for it, win, vp in history:
        print(f"{it:4d} | {win:8.1%} | {vp:.2f}")


if __name__ == "__main__":
    main()
