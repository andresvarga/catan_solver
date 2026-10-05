"""Exclusive-time component breakdown of the training rollout hot loop (single process,
torch threads = 1), plus call counts, GC activity and top allocators.

    python -m audit.perf.profile_components --model hier --envs 1 --episodes 8
"""
from __future__ import annotations

import argparse
import collections
import gc
import json
import time
import tracemalloc

import torch

TIMES = collections.Counter()
CALLS = collections.Counter()
_STACK: list[list] = []


def timed(label, fn):
    def wrapper(*a, **k):
        frame = [0.0]
        _STACK.append(frame)
        t0 = time.perf_counter()
        try:
            return fn(*a, **k)
        finally:
            dt = time.perf_counter() - t0
            _STACK.pop()
            TIMES[label] += dt - frame[0]  # exclusive of timed children
            CALLS[label] += 1
            if _STACK:
                _STACK[-1][0] += dt
    wrapper.__wrapped__ = fn
    return wrapper


def patch(obj, name, label):
    setattr(obj, name, timed(label, getattr(obj, name)))


def install():
    import env.engine as en
    import env.pettingzoo_env as pe
    import env.public_beliefs as pb
    import training.graph_features as gf
    import training.hier_model as hm
    import training.hier_ppo as hp
    import training.model as tm
    import training.ppo as tp
    import training.model_adapters as ma
    # engine: rules transition, legality enumeration (all call sites), validation
    patch(en, "step", "engine.step (rules)")
    patch(en, "legal_actions", "engine.legal_actions")
    patch(en, "is_legal_action", "engine.is_legal_action (validation)")
    patch(en, "new_game", "reset: new_game/generate_board")
    pe.engine_legal_actions = en.legal_actions
    pe.is_legal_action = en.is_legal_action
    # env wrapper
    patch(pe.CatanAECEnv, "step", "env.step (bookkeeping)")
    patch(pe.CatanAECEnv, "last", "env.last")
    patch(pe.CatanAECEnv, "reset", "env.reset")
    patch(pe.CatanAECEnv, "__init__", "env.__init__")
    patch(pe, "build_observation", "obs: build_observation (dict)")
    patch(pb, "expected_dev_cards", "obs: public beliefs")
    pe.expected_dev_cards = pb.expected_dev_cards
    gf.expected_dev_cards = pb.expected_dev_cards
    patch(gf, "build_graph_observation", "obs: build_graph_observation")
    ma.build_graph_observation = gf.build_graph_observation
    patch(tm, "flatten_observation", "obs: flatten_observation")
    ma.flatten_observation = tm.flatten_observation
    # policy
    patch(hm.ActorSampling, "act_batch", "policy: act_batch (python/glue)")
    for cls in _model_classes():
        patch(cls, "head_logits_batch", "policy: forward (head_logits_batch)")
        patch(cls, "value", "policy: value (truncation bootstrap)")
    patch(hm, "sample_decision", "policy: numpy sampling (sample_decision)")
    patch(hm, "decode_trade", "policy: trade bundle decode")
    patch(ma, "_flat_to_batch", "policy: obs -> tensor")
    patch(ma, "_graph_to_batch", "policy: obs -> tensor")
    ma.FLAT_ADAPTER.to_batch = ma._flat_to_batch
    ma.GRAPH_ADAPTER.to_batch = ma._graph_to_batch
    ma.FLAT_ADAPTER.encode = timed("obs: adapter.encode (glue)", ma.FLAT_ADAPTER.encode)
    ma.GRAPH_ADAPTER.encode = timed("obs: adapter.encode (glue)", ma.GRAPH_ADAPTER.encode)
    # rollout bookkeeping
    patch(hp.EpisodeRunner, "advance", "runner.advance (bookkeeping)")
    patch(hp.EpisodeRunner, "apply", "runner.apply (bookkeeping)")
    patch(hp, "_episode_opponents", "runner: opponent copies")
    patch(tp, "compute_gae", "storage: compute_gae")
    hp.compute_gae = tp.compute_gae


def _model_classes():
    from training.gnn_model import GraphActorCritic
    from training.hier_model import HierarchicalActorCritic
    return [HierarchicalActorCritic, GraphActorCritic]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="hier")
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--phf", action="store_true")
    ap.add_argument("--envs", type=int, default=1)
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--opponents", default="selfplay")
    ap.add_argument("--alloc", action="store_true", help="tracemalloc top allocators (slow)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    torch.set_num_threads(1)
    from audit.perf.bench_sim import _model, _opponents
    import training.hier_ppo as hp
    from training.model_adapters import ADAPTERS
    model = _model(a.model, a.hidden, a.layers, a.phf).to(a.device).eval()
    install()
    env_kwargs = dict(randomize_board=True, max_episode_steps=4000, public_hand_features=a.phf)
    opp = _opponents(a.opponents)
    # warm-up
    list(hp.collect_episodes_batched(env_kwargs, model, a.device, [73_000_000], opp, ADAPTERS[a.model], 1))
    TIMES.clear(); CALLS.clear()
    gc_stats = {"collections": [0, 0, 0], "seconds": 0.0}
    gc_t = [0.0]

    def gc_cb(phase, info):
        if phase == "start":
            gc_t[0] = time.perf_counter()
        else:
            gc_stats["collections"][info["generation"]] += 1
            gc_stats["seconds"] += time.perf_counter() - gc_t[0]
    gc.callbacks.append(gc_cb)
    if a.alloc:
        tracemalloc.start(1)
    t0 = time.perf_counter()
    n_tr = n_steps = 0
    for r in hp.collect_episodes_batched(env_kwargs, model, a.device,
                                         list(range(73_100_000, 73_100_000 + a.episodes)), opp,
                                         ADAPTERS[a.model], num_envs=a.envs):
        n_tr += sum(len(v) for v in r.data.values())
        n_steps += r.env._step_count
    wall = time.perf_counter() - t0
    gc.callbacks.remove(gc_cb)
    top_alloc = []
    if a.alloc:
        snap = tracemalloc.take_snapshot()
        tracemalloc.stop()
        for s in snap.statistics("lineno")[:15]:
            top_alloc.append(f"{s.size / 2**20:7.2f} MB {s.count:8d} blocks  {s.traceback}")
    accounted = sum(TIMES.values())
    rows = sorted(((k, v, CALLS[k]) for k, v in TIMES.items()), key=lambda r: -r[1])
    print(f"model={a.model}{a.hidden}x{a.layers} envs={a.envs} episodes={a.episodes}  wall={wall:.2f}s  "
          f"transitions={n_tr} ({n_tr / wall:.0f}/s)  env steps={n_steps} ({n_steps / wall:.0f}/s)")
    print(f"{'component (exclusive)':48s} {'% wall':>7s} {'calls':>9s} {'us/call':>9s} {'us/env-step':>11s}")
    for k, v, c in rows:
        print(f"{k:48s} {v / wall:7.1%} {c:9d} {v / c * 1e6:9.1f} {v / n_steps * 1e6:11.1f}")
    print(f"{'(untimed: loop glue, numpy/env misc)':48s} {(wall - accounted) / wall:7.1%}")
    print(f"GC: {gc_stats['collections']} collections (gen0/1/2), {gc_stats['seconds']:.3f}s "
          f"({gc_stats['seconds'] / wall:.1%} of wall)")
    for line in top_alloc:
        print(line)
    if a.out:
        with open(a.out, "w") as f:
            json.dump({"args": vars(a), "wall": wall, "transitions": n_tr, "env_steps": n_steps,
                       "components": {k: {"seconds": v, "calls": c} for k, v, c in rows},
                       "gc": gc_stats, "top_alloc": top_alloc}, f, indent=1)


if __name__ == "__main__":
    main()
