"""Behavior-cloning pretraining CLI: warm-start `HierarchicalActorCritic` on
heuristic-vs-heuristic demonstrations (scripts/collect_heuristic_demos.py)
before handing it to league_train's PPO loop.

BC loss is exactly the negative log-likelihood of the demonstrated action
under the current policy -- which is precisely what
`HierarchicalActorCritic.evaluate_actions` already computes as its `logprob`
return (type head + whichever sub-heads were active), since a demonstration
and a self-generated rollout transition share the same head_data shape. So
this reuses `evaluate_actions` directly rather than reimplementing per-head
cross-entropy: `loss = -mean(logprob) - entropy_coef * mean(entropy)`, with a
small entropy bonus purely to keep the policy from collapsing to
deterministic point masses it can't recover from during the RL fine-tune
that follows.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from training.hier_model import HierarchicalActorCritic
from training.model import observation_dim


def load_dataset(path: str, device: str) -> dict[str, torch.Tensor]:
    npz = np.load(path)
    return {k: torch.as_tensor(npz[k], device=device) for k in npz.files}


def evaluate_bc(model: HierarchicalActorCritic, data: dict[str, torch.Tensor],
                 idx: np.ndarray, batch_size: int = 4096) -> dict[str, float]:
    model.eval()
    total_logprob, total_correct_type, n = 0.0, 0, 0
    with torch.inference_mode():
        for start in range(0, len(idx), batch_size):
            mb = torch.as_tensor(idx[start:start + batch_size], dtype=torch.long, device=data["obs"].device)
            obs_batch = data["obs"][mb]
            tb = {k: v[mb] for k, v in data.items() if k != "obs"}
            logprob, entropy, value = model.evaluate_actions(obs_batch, tb)
            total_logprob += float(logprob.sum())
            type_logits = model.type_head(model.features(obs_batch)).masked_fill(tb["type_mask"] == 0, -1e9)
            total_correct_type += int((type_logits.argmax(-1) == tb["type_idx"]).sum())
            n += len(mb)
    model.train()
    return {"mean_logprob": total_logprob / n, "type_accuracy": total_correct_type / n}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--val-frac", type=float, default=0.05)
    parser.add_argument("--public-hand-features", action="store_true")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    print(f"device: {device}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data = load_dataset(args.dataset, device)
    n = data["obs"].shape[0]
    perm = np.random.permutation(n)
    n_val = int(n * args.val_frac)
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    print(f"loaded {n} demonstrations ({len(train_idx)} train / {len(val_idx)} val)")

    obs_dim = observation_dim(public_hand_features=args.public_hand_features)
    assert data["obs"].shape[1] == obs_dim, (
        f"dataset obs width {data['obs'].shape[1]} != expected {obs_dim} for "
        f"public_hand_features={args.public_hand_features} -- collect the dataset with "
        f"a matching --public-hand-features setting.")
    model = HierarchicalActorCritic(obs_dim=obs_dim, hidden=args.hidden).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    val0 = evaluate_bc(model, data, val_idx)
    print(f"init (random weights): val_logprob={val0['mean_logprob']:.3f} "
          f"type_acc={val0['type_accuracy']:.3f}")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        np.random.shuffle(train_idx)
        losses = []
        for start in range(0, len(train_idx), args.batch_size):
            mb = torch.as_tensor(train_idx[start:start + args.batch_size], dtype=torch.long, device=device)
            obs_batch = data["obs"][mb]
            tb = {k: v[mb] for k, v in data.items() if k != "obs"}
            logprob, entropy, value = model.evaluate_actions(obs_batch, tb)
            loss = -logprob.mean() - args.entropy_coef * entropy.mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            losses.append(loss.item())

        val = evaluate_bc(model, data, val_idx)
        print(f"epoch {epoch:2d} | train_loss={np.mean(losses):.4f} | "
              f"val_logprob={val['mean_logprob']:.3f} val_type_acc={val['type_accuracy']:.3f} | "
              f"{time.time()-t0:.1f}s")

    model.to("cpu")
    torch.save({"model": model.state_dict(), "bc_dataset": args.dataset, "bc_epochs": args.epochs},
               args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
