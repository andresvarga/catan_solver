"""Low-noise A/B benchmark for individual optimizations: one process, torch threads = 1,
`collect_episodes_batched` with N games in flight; reports env steps/s and transitions/s
over several trials (same seeds every trial and every code version).

    python -m audit.perf.ab --model hier --envs 16 --episodes 16 --trials 5 --label O0-baseline
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="hier")
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--phf", action="store_true")
    ap.add_argument("--envs", type=int, default=16)
    ap.add_argument("--episodes", type=int, default=16)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--opponents", default="selfplay")
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--label", required=True)
    a = ap.parse_args()
    torch.set_num_threads(1)
    from audit.perf.bench_sim import _model, _opponents
    import training.hier_ppo as hp
    from training.model_adapters import ADAPTERS
    model = _model(a.model, a.hidden, a.layers, a.phf).to(a.device).eval()
    kw = dict(randomize_board=True, max_episode_steps=4000, public_hand_features=a.phf)
    opp = _opponents(a.opponents)
    seeds = list(range(77_000_000, 77_000_000 + a.episodes))
    list(hp.collect_episodes_batched(kw, model, a.device, [76_900_000], opp, ADAPTERS[a.model], 1))  # warm-up
    steps_s, tr_s = [], []
    for _ in range(a.trials):
        t0 = time.perf_counter()
        n_st = n_tr = 0
        for r in hp.collect_episodes_batched(kw, model, a.device, seeds, opp, ADAPTERS[a.model], a.envs):
            n_st += r.env._step_count
            n_tr += sum(len(v) for v in r.data.values())
        dt = time.perf_counter() - t0
        steps_s.append(n_st / dt)
        tr_s.append(n_tr / dt)
    row = {"label": a.label, "config": f"{a.model}{a.hidden}x{a.layers}{'-phf' if a.phf else ''} "
                                       f"{a.device} envs={a.envs} opp={a.opponents}",
           "steps_per_s_mean": round(statistics.mean(steps_s)), "steps_per_s_sd": round(statistics.stdev(steps_s)),
           "transitions_per_s_mean": round(statistics.mean(tr_s)), "env_steps": n_st, "transitions": n_tr}
    print(json.dumps(row))
    with open(os.path.join(os.path.dirname(__file__), "..", "results", "perf", "ab.jsonl"), "a") as f:
        f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
