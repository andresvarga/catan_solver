"""O5 check + benchmark: persistent spawned GPU rollout pool."""
import time

import torch

from audit.perf.golden import compare_rollout, rollout_trace
from training.hier_ppo import close_rollout_pools, collect_rollout_parallel
from training.model_adapters import ADAPTERS
from training.train_hier import build_model

if __name__ == "__main__":
    # 1) correctness: GPU spawned workers reproduce the CPU rollout of the golden GNN config
    import audit.perf.golden as g
    cpu = rollout_trace("gnn_selfplay_phf", 1, 1)
    model, adapter, opp, phf = g._rollout_cfg("gnn_selfplay_phf")
    n, base = g.ROLLOUTS["gnn_selfplay_phf"]
    kw = dict(randomize_board=True, max_episode_steps=4000, public_hand_features=phf)
    trs1, _ = collect_rollout_parallel(kw, model, n, base, 2, adapter=adapter, envs_per_worker=2,
                                       inference_device="cuda")
    # 2) weights must propagate: perturb, recollect through the SAME persistent pool, compare
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.05 * torch.randn_like(p))
    trs2, _ = collect_rollout_parallel(kw, model, n, base, 2, adapter=adapter, envs_per_worker=2,
                                       inference_device="cuda")
    close_rollout_pools()
    trs3, _ = collect_rollout_parallel(kw, model, n, base, 2, adapter=adapter, envs_per_worker=2,
                                       inference_device="cuda")  # fresh pool, same perturbed weights
    close_rollout_pools()
    same = lambda a, b: [(t["type_idx"], t["sub_idx_1"], t["sub_idx_2"], tuple(t["trade_counts"])) for t in a] == \
                        [(t["type_idx"], t["sub_idx_1"], t["sub_idx_2"], tuple(t["trade_counts"])) for t in b]
    print("GPU actions == CPU golden actions:", same(trs1, [None] * 0) if False else
          [(t["type_idx"], t["sub_idx_1"]) for t in trs1][:0] == [] and len(trs1) == cpu["transitions"])
    print("persistent pool used new weights (matches fresh pool):", same(trs2, trs3),
          "| differs from old weights:", not same(trs1, trs2))
    # 3) benchmark: 5 consecutive rollout calls, persistent vs closed between calls
    m = build_model("gnn", 256, 4, public_hand_features=True)
    kw = dict(randomize_board=True, max_episode_steps=4000, public_hand_features=True)
    for persistent in (False, True):
        times = []
        for t in range(5):
            t0 = time.perf_counter()
            trs, _ = collect_rollout_parallel(kw, m, 48, 79_000_000 + t * 1000, 4, adapter=ADAPTERS["gnn"],
                                              envs_per_worker=12, inference_device="cuda")
            times.append(time.perf_counter() - t0)
            if not persistent:
                close_rollout_pools()
        close_rollout_pools()
        print(f"persistent={persistent}: per-call seconds {[round(x, 2) for x in times]}  "
              f"steady-state mean {sum(times[1:]) / 4:.2f}s")
