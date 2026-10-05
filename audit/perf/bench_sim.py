"""Reproducible simulation-throughput benchmarks (simulation performance audit).

    python -m audit.perf.bench_sim engine  --procs 1,2,4,8,16
    python -m audit.perf.bench_sim env     --procs 1,8,16
    python -m audit.perf.bench_sim rollout --model hier --workers 1,4,8,12,16 --envs 1,4,16
    python -m audit.perf.bench_sim rollout --model gnn --hidden 256 --layers 4 --device cuda ...
    python -m audit.perf.bench_sim ipc / startup

Every rollout trial = one `collect_rollout_parallel` call (exactly what training does per
iteration), after one warm-up call. Throughput is reported per trial and summarised
(mean/median/min/max/std). A sampler thread records the CPU% and memory (PSS, so forked
copy-on-write pages aren't double counted) of the whole process tree.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import pickle
import random
import statistics
import threading
import time

import psutil
import torch

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "perf")


# ---------------------------------------------------------------- resource sampler
class TreeSampler:
    def __init__(self, interval: float = 0.25, pss: bool = True):
        self.interval, self.pss = interval, pss
        self.cpu, self.mem, self.nproc, self.nthreads = [], [], [], []
        self._stop = threading.Event()
        self.root = psutil.Process()

    def _procs(self):
        try:
            return [self.root] + self.root.children(recursive=True)
        except psutil.Error:
            return [self.root]

    def _run(self):
        seen: dict[int, psutil.Process] = {}
        while not self._stop.is_set():
            procs = self._procs()
            cpu = mem = threads = 0.0
            for p in procs:
                try:
                    q = seen.setdefault(p.pid, p)
                    cpu += q.cpu_percent(None)
                    mem += (q.memory_full_info().pss if self.pss else q.memory_info().rss)
                    threads += q.num_threads()
                except psutil.Error:
                    pass
            self.cpu.append(cpu)
            self.mem.append(mem)
            self.nproc.append(len(procs))
            self.nthreads.append(threads)
            time.sleep(self.interval)

    def __enter__(self):
        for p in self._procs():
            try:
                p.cpu_percent(None)
            except psutil.Error:
                pass
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join()

    def summary(self) -> dict:
        cpu = self.cpu[2:] or self.cpu  # first samples have no cpu delta yet
        return {"cpu_pct_mean": round(statistics.mean(cpu), 1) if cpu else None,
                "mem_mb_peak": round(max(self.mem) / 2 ** 20, 1) if self.mem else None,
                "mem_mb_mean": round(statistics.mean(self.mem) / 2 ** 20, 1) if self.mem else None,
                "procs_peak": max(self.nproc) if self.nproc else None,
                "threads_peak": max(self.nthreads) if self.nthreads else None}


def stats(xs: list[float]) -> dict:
    return {"mean": round(statistics.mean(xs), 1), "median": round(statistics.median(xs), 1),
            "min": round(min(xs), 1), "max": round(max(xs), 1),
            "std": round(statistics.stdev(xs), 1) if len(xs) > 1 else 0.0}


# ---------------------------------------------------------------- W1 engine / W2 env
def _engine_games(args):
    from agents.random_agent import RandomAgent
    from env.engine import CatanEngine
    base, n = args
    steps = 0
    lengths = []
    for g in range(n):
        seed = base + g
        eng = CatanEngine(randomize_board=True, seed=seed)
        agents = {i: RandomAgent(i, random.Random(seed * 97 + i)) for i in range(4)}
        k = 0
        while not eng.done and k < 6000:
            eng.step(agents[eng.acting_player()].choose(eng.state, eng.legal_actions()))
            k += 1
        steps += k
        lengths.append(k)
    return steps, lengths


def _env_games(args):
    from env.engine import is_template, random_trade
    from env.pettingzoo_env import CatanAECEnv
    base, n = args
    steps = 0
    lengths = []
    env = CatanAECEnv()
    for g in range(n):
        seed = base + g
        env.reset(seed=seed)
        rng = random.Random(seed)
        k = 0
        while env.agents and k < 6000:
            obs, r, term, trunc, _ = env.last()
            env.clear_reward(env.agent_selection)
            if term or trunc:
                env.step(None)
                continue
            legal = env.legal_actions()
            i = rng.randrange(len(legal))
            a = legal[i]
            env.step(random_trade(a, rng, actor=a.params["actor"]) if is_template(a) else i)
            k += 1
        steps += k
        lengths.append(k)
    return steps, lengths


def bench_games(kind: str, procs: int, games_per_proc: int, trials: int, base: int) -> dict:
    fn = _engine_games if kind == "engine" else _env_games
    fn((base - 1000, 1))  # warm-up (imports, caches)
    rates, grates, lengths = [], [], []
    samp = None
    for t in range(trials):
        jobs = [(base + t * 100_000 + p * 1000, games_per_proc) for p in range(procs)]
        with TreeSampler() as samp:
            t0 = time.perf_counter()
            if procs == 1:
                res = [fn(jobs[0])]
            else:
                with mp.get_context("fork").Pool(procs) as pool:
                    res = pool.map(fn, jobs)
            dt = time.perf_counter() - t0
        steps = sum(r[0] for r in res)
        rates.append(steps / dt)
        grates.append(procs * games_per_proc / dt)
        lengths += [x for r in res for x in r[1]]
    return {"workload": kind, "procs": procs, "games_per_proc": games_per_proc,
            "steps_per_s": stats(rates), "games_per_s": stats(grates),
            "steps_per_game_mean": round(statistics.mean(lengths), 1), **samp.summary()}


# ---------------------------------------------------------------- W3-W5 rollouts
def _model(kind, hidden, layers, phf):
    from training.train_hier import build_model
    torch.manual_seed(0)
    return build_model(kind, hidden, layers, public_hand_features=phf)


def _opponents(name):
    if name == "selfplay":
        return None
    from functools import partial
    from agents.opponent_pool import builtin_members
    from training.hier_ppo import RotatingOpponents
    from training.rl_finetune import _heuristic_opponent, _pooled_opponent
    if name == "heuristic":
        return RotatingOpponents(partial(_heuristic_opponent))
    return RotatingOpponents(partial(_pooled_opponent, builtin_members()))


def bench_rollout(kind, hidden, layers, phf, opponents, workers, envs, device, episodes,
                  trials, base) -> dict:
    from training.hier_ppo import collect_rollout_parallel
    from training.model_adapters import ADAPTERS
    import training.hier_ppo as hp
    model = _model(kind, hidden, layers, phf)
    adapter = ADAPTERS[kind]
    opp = _opponents(opponents)
    env_kwargs = dict(randomize_board=True, max_episode_steps=4000, public_hand_features=phf)
    # count engine steps per episode via the summary (fork workers inherit this patch)
    orig = hp._episode_summary

    def summ(env, seed, opponent_agents, opponent_name):
        d = orig(env, seed, opponent_agents, opponent_name)
        d["env_steps"] = env._step_count
        return d
    hp._episode_summary = summ
    collect_rollout_parallel(env_kwargs, model, max(workers, 2), base - 50_000, workers, opponent_agents=opp,
                             adapter=adapter, envs_per_worker=envs, inference_device=device)  # warm-up
    tr_rates, st_rates, g_rates, walls = [], [], [], []
    samp = None
    steps_known = True
    for t in range(trials):
        with TreeSampler() as samp:
            t0 = time.perf_counter()
            trs, summaries = collect_rollout_parallel(
                env_kwargs, model, episodes, base + t * 10_000, workers, opponent_agents=opp,
                adapter=adapter, envs_per_worker=envs, inference_device=device)
            dt = time.perf_counter() - t0
        walls.append(dt)
        tr_rates.append(len(trs) / dt)
        g_rates.append(len(summaries) / dt)
        if all("env_steps" in s for s in summaries):
            st_rates.append(sum(s["env_steps"] for s in summaries) / dt)
        else:
            steps_known = False  # spawned workers don't inherit the patch
    hp._episode_summary = orig
    return {"workload": "rollout", "model": f"{kind}{hidden}x{layers}", "phf": phf, "opponents": opponents,
            "workers": workers, "envs_per_worker": envs, "device": device, "episodes": episodes,
            "transitions_per_s": stats(tr_rates), "games_per_s": stats(g_rates),
            "env_steps_per_s": stats(st_rates) if steps_known and st_rates else None,
            "call_seconds": stats(walls), **samp.summary()}


# ---------------------------------------------------------------- IPC / startup
def bench_ipc(kind, hidden, layers, phf, episodes: int = 4) -> dict:
    """Size and (de)serialization time of one worker's result payload."""
    import training.hier_ppo as hp
    from training.model_adapters import ADAPTERS
    model = _model(kind, hidden, layers, phf).eval()
    env_kwargs = dict(randomize_board=True, max_episode_steps=4000, public_hand_features=phf)
    hp._init_worker(model, env_kwargs, None, None, ADAPTERS[kind], num_envs=4)
    res = hp._worker_collect(list(range(71_000_000, 71_000_000 + episodes)))
    n_tr = sum(len(r[1]) for r in res)
    t0 = time.perf_counter(); blob = pickle.dumps(res, protocol=pickle.HIGHEST_PROTOCOL); t1 = time.perf_counter()
    pickle.loads(blob); t2 = time.perf_counter()
    return {"workload": "ipc", "model": f"{kind}{hidden}x{layers}", "phf": phf, "transitions": n_tr,
            "payload_mb": round(len(blob) / 2 ** 20, 2), "bytes_per_transition": round(len(blob) / n_tr),
            "pickle_us_per_transition": round((t1 - t0) / n_tr * 1e6, 2),
            "unpickle_us_per_transition": round((t2 - t1) / n_tr * 1e6, 2)}


def _noop(x):
    return x


def bench_startup(kind, hidden, layers, phf, workers: int, device: str) -> dict:
    """Pool creation + initializer (+ model copy) cost paid by every rollout call."""
    import copy
    from concurrent.futures import ProcessPoolExecutor
    import training.hier_ppo as hp
    from training.model_adapters import ADAPTERS
    model = _model(kind, hidden, layers, phf)
    env_kwargs = dict(randomize_board=True, max_episode_steps=4000, public_hand_features=phf)
    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        if device == "cpu":
            ctx, wm = mp.get_context("fork"), model
        else:
            ctx, wm = mp.get_context("spawn"), copy.deepcopy(model).to("cpu")
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx, initializer=hp._init_worker,
                                 initargs=(wm, env_kwargs, None, None, ADAPTERS[kind], 0.999, 0.98, 1,
                                           device)) as pool:
            list(pool.map(_noop, range(workers)))
        times.append(time.perf_counter() - t0)
    return {"workload": "startup", "model": f"{kind}{hidden}x{layers}", "workers": workers, "device": device,
            "seconds": stats(times)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["engine", "env", "rollout", "ipc", "startup"])
    ap.add_argument("--procs", default="1")
    ap.add_argument("--games", type=int, default=20, help="games per process (engine/env)")
    ap.add_argument("--model", default="hier")
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--phf", action="store_true")
    ap.add_argument("--opponents", default="selfplay", choices=["selfplay", "heuristic", "pool"])
    ap.add_argument("--workers", default="1")
    ap.add_argument("--envs", default="1")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--episodes", type=int, default=48)
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--base", type=int, default=72_000_000)
    ap.add_argument("--tag", default="baseline")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rows = []
    if a.what in ("engine", "env"):
        for p in map(int, a.procs.split(",")):
            rows.append(bench_games(a.what, p, a.games, a.trials, a.base))
            print(json.dumps(rows[-1]), flush=True)
    elif a.what == "rollout":
        for w in map(int, a.workers.split(",")):
            for e in map(int, a.envs.split(",")):
                rows.append(bench_rollout(a.model, a.hidden, a.layers, a.phf, a.opponents, w, e, a.device,
                                          a.episodes, a.trials, a.base))
                print(json.dumps(rows[-1]), flush=True)
    elif a.what == "ipc":
        rows.append(bench_ipc(a.model, a.hidden, a.layers, a.phf))
        print(json.dumps(rows[-1]))
    else:
        for w in map(int, a.workers.split(",")):
            rows.append(bench_startup(a.model, a.hidden, a.layers, a.phf, w, a.device))
            print(json.dumps(rows[-1]), flush=True)
    for r in rows:
        r["tag"] = a.tag
    out = a.out or os.path.join(OUT_DIR, f"{a.tag}_{a.what}.jsonl")
    with open(out, "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
