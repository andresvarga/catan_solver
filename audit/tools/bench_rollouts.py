"""Phase 4 rollout throughput benchmark (self-play, all seats trainee).

    python -m audit.tools.bench_rollouts --out audit/results/bench_rollouts.json
"""
from __future__ import annotations

import argparse
import json
import time

import torch

from training.hier_ppo import collect_rollout_parallel
from training.model_adapters import ADAPTERS
from training.train_hier import build_model

ENV = dict(randomize_board=True, max_episode_steps=4000)


def bench(kind, hidden, layers, workers, envs, device, episodes, base=24_000_000):
    torch.manual_seed(0)
    model = build_model(kind, hidden, layers).eval().to(device)
    adapter = ADAPTERS[kind]
    t = time.perf_counter()
    trs, summ = collect_rollout_parallel(ENV, model, episodes, base, workers, adapter=adapter,
                                         envs_per_worker=envs, inference_device=device)
    dt = time.perf_counter() - t
    return {"model": f"{kind}{hidden}" + (f"x{layers}" if kind == "gnn" else ""), "workers": workers,
            "envs_per_worker": envs, "device": device, "episodes": episodes, "transitions": len(trs),
            "seconds": round(dt, 2), "transitions_per_s": round(len(trs) / dt, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu-workers-only", action="store_true")
    a = ap.parse_args()
    cuda = torch.cuda.is_available()
    plan = [
        ("hier", 256, 3, 1, 1, "cpu", 8), ("hier", 256, 3, 1, 16, "cpu", 16),
        ("hier", 256, 3, 12, 1, "cpu", 48), ("hier", 256, 3, 12, 4, "cpu", 48),
        ("gnn", 128, 3, 1, 1, "cpu", 2), ("gnn", 128, 3, 12, 1, "cpu", 24), ("gnn", 128, 3, 12, 4, "cpu", 48),
        ("gnn", 256, 4, 1, 1, "cpu", 1), ("gnn", 256, 4, 12, 1, "cpu", 12), ("gnn", 256, 4, 12, 4, "cpu", 24),
    ]
    if cuda:
        plan += [("hier", 256, 3, 1, 64, "cuda", 64),
                 ("gnn", 128, 3, 1, 32, "cuda", 32), ("gnn", 128, 3, 1, 64, "cuda", 64),
                 ("gnn", 256, 4, 1, 32, "cuda", 32), ("gnn", 256, 4, 1, 64, "cuda", 64),
                 ("gnn", 256, 4, 1, 128, "cuda", 128)]
    if cuda and a.gpu_workers_only:
        plan = [("gnn", 256, 4, 4, 32, "cuda", 128), ("gnn", 256, 4, 6, 24, "cuda", 144),
                ("gnn", 128, 3, 4, 32, "cuda", 128), ("hier", 256, 3, 6, 32, "cuda", 192)]
    rows = []
    for p in plan:
        r = bench(*p)
        rows.append(r)
        print(json.dumps(r), flush=True)
    with open(a.out, "w") as f:
        json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
