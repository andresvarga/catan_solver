"""DAgger driver (roadmap: close the imitation gap before RL).

Each round: (1) roll games under the current policy vs 3 heuristics and label
every policy-seat decision with the expert's choice (scripts/
collect_dagger_demos.py), (2) aggregate with the base heuristic-self-play
demos and all prior rounds, (3) retrain from the current weights on the
aggregated set, (4) evaluate on the fixed-seed protocol. The aggregated
dataset stays on CPU (it grows past GPU memory); minibatches move to the
device per step.

Selection note: the per-round evaluation IS the selection signal here (it's
the task metric, evaluated on fixed seed sets) -- but with few rounds the
winner's-curse bias is small and the final confirmatory numbers are printed
for every round side by side rather than auto-picking silently.
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch

from scripts.collect_dagger_demos import collect_dagger
from training.hier_model import HierarchicalActorCritic
from training.model import observation_dim
from training.train_hier import evaluate_policy


def load_arrays(path: str) -> dict[str, np.ndarray]:
    npz = np.load(path)
    return {k: npz[k] for k in npz.files}


def concat_arrays(parts: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def to_cpu_tensors(arrays: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
    return {k: torch.as_tensor(v) for k, v in arrays.items()}


def train_epochs(model, data: dict[str, torch.Tensor], device: str, epochs: int,
                  batch_size: int, lr: float, entropy_coef: float, seed: int) -> None:
    """BC training loop (same objective as training/bc_pretrain.py) over
    CPU-resident data, minibatches moved to `device` per step."""
    model.to(device).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    n = data["type_idx"].shape[0]
    rng = np.random.RandomState(seed)
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        order = rng.permutation(n)
        losses = []
        for start in range(0, n, batch_size):
            mb = torch.as_tensor(order[start:start + batch_size], dtype=torch.long)
            obs = data["obs"][mb].to(device, non_blocking=True)
            tb = {k: v[mb].to(device, non_blocking=True) for k, v in data.items() if k != "obs"}
            logprob, entropy, _ = model.evaluate_actions(obs, tb)
            loss = -logprob.mean() - entropy_coef * entropy.mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            losses.append(loss.item())
        print(f"    epoch {epoch:2d}/{epochs} loss={np.mean(losses):.4f} ({time.time()-t0:.1f}s)",
              flush=True)
    model.to("cpu").eval()


def fixed_seed_eval(model, games_per_set: int, num_workers: int,
                     public_hand_features: bool,
                     seed_bases: tuple[int, int] = (555_000, 777_000)) -> tuple[float, float]:
    wins = vp = 0.0
    for base in seed_bases:
        res = evaluate_policy(model, "heuristic", games_per_set, seed_base=base,
                               model_kind="hier", num_workers=num_workers,
                               public_hand_features=public_hand_features)
        wins += res["win_rate"] * games_per_set
        vp += res["avg_vp"] * games_per_set
    total = 2 * games_per_set
    return wins / total, vp / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--init-checkpoint", type=str, required=True,
                         help="BC-pretrained starting policy")
    parser.add_argument("--base-dataset", type=str, required=True,
                         help="heuristic-self-play demos .npz (round-0 training data)")
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--games-per-round", type=int, default=400)
    parser.add_argument("--expert-prob", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=5e-4,
                         help="lower than fresh-BC's 1e-3: every round warm-starts")
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--public-hand-features", action="store_true")
    parser.add_argument("--eval-games", type=int, default=120,
                         help="per fixed seed set (two sets, see --eval-seed-bases)")
    parser.add_argument("--eval-seed-bases", type=int, nargs=2, default=[555_000, 777_000],
                         help="two eval seed bases. Use a range no prior experiment has "
                              "selected against -- repeatedly comparing/selecting on the same "
                              "seed sets overfits the leaderboard to them (observed: a "
                              "checkpoint chosen as project-best on much-reused sets dropped "
                              "8pp on genuinely fresh seeds). Reserve yet another untouched "
                              "range for the final cross-experiment comparison.")
    parser.add_argument("--num-workers", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)

    obs_dim = observation_dim(public_hand_features=args.public_hand_features)
    model = HierarchicalActorCritic(obs_dim=obs_dim, hidden=args.hidden)
    ckpt = torch.load(args.init_checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    model.eval()

    print(f"device={device}  rounds={args.rounds}  games/round={args.games_per_round}")
    parts = [load_arrays(args.base_dataset)]
    print(f"base dataset: {parts[0]['type_idx'].shape[0]} decisions")

    win0, vp0 = fixed_seed_eval(model, args.eval_games, args.num_workers,
                                 args.public_hand_features, tuple(args.eval_seed_bases))
    print(f"round 0 (init): win_rate={win0:.1%} avg_vp={vp0:.2f}", flush=True)
    history = [(0, win0, vp0, parts[0]["type_idx"].shape[0])]

    for rnd in range(1, args.rounds + 1):
        t0 = time.time()
        base_seed = args.seed + 1_000_000 + rnd * 10_000
        new = collect_dagger(model, args.games_per_round, base_seed,
                              args.public_hand_features, num_workers=args.num_workers,
                              expert_prob=args.expert_prob)
        np.savez(os.path.join(args.out_dir, f"dagger_round{rnd}.npz"), **new)
        parts.append(new)
        agg = concat_arrays(parts)
        n = agg["type_idx"].shape[0]
        print(f"round {rnd}: +{new['type_idx'].shape[0]} labeled decisions "
              f"(aggregate {n}) in {time.time()-t0:.1f}s", flush=True)

        data = to_cpu_tensors(agg)
        train_epochs(model, data, device, args.epochs, args.batch_size, args.lr,
                      args.entropy_coef, seed=args.seed + rnd)
        del data

        torch.save({"model": model.state_dict(), "round": rnd},
                   os.path.join(args.out_dir, f"dagger_round{rnd}.pt"))
        win, vp = fixed_seed_eval(model, args.eval_games, args.num_workers,
                                   args.public_hand_features, tuple(args.eval_seed_bases))
        history.append((rnd, win, vp, n))
        print(f"round {rnd}: win_rate={win:.1%} avg_vp={vp:.2f} "
              f"({time.time()-t0:.1f}s total)", flush=True)

    print("\nround | aggregate decisions | win_rate vs heuristic | avg_vp")
    for rnd, win, vp, n in history:
        print(f"{rnd:5d} | {n:19d} | {win:20.1%} | {vp:.2f}")
    best = max(history, key=lambda h: h[1])
    print(f"best round: {best[0]} ({best[1]:.1%}) -> "
          f"{os.path.join(args.out_dir, f'dagger_round{best[0]}.pt') if best[0] else args.init_checkpoint}")


if __name__ == "__main__":
    main()
