"""Self-play PPO using the hierarchical/pointer action head (training/hier_model.py)
instead of the flat index-into-legal_actions scheme. Structurally mirrors
training/ppo.py (same AEC turn-order credit-assignment trick -- see that
module's docstring); the difference is entirely in what gets stored per
transition and how the policy is queried/evaluated.

Supports league play (roadmap phase 5, design doc §7): any subset of the 4
seats can be handed to a frozen `opponent_agents[pid]` (anything with a
`.choose(state) -> Action` method -- `HierarchicalLearnedAgent`,
`HeuristicAgent`, `RandomAgent` all already qualify). Only the seats *not* in
`opponent_agents` are the trainee, and only those get transitions recorded
for the PPO update -- opponent seats are just environment dynamics from the
trainee's point of view. `opponent_agents=None`/`{}` reproduces plain
self-play (every seat is the trainee), so this is a strict generalization of
the phase-4 loop, not a parallel code path.

Also supports either model family (roadmap phase 6) via `ModelAdapter`
(training/model_adapters.py): `HierarchicalActorCritic`'s flat-vector trunk
and `GraphActorCritic`'s GNN encoder both expose the same `.act()` /
`.evaluate_actions()` contract, so the only thing that differs is how an
observation gets encoded and how a batch of them gets stacked into tensors
-- everything below is otherwise identical for both, hence one shared file
rather than a duplicated `gnn_ppo.py`.
"""
from __future__ import annotations

import atexit
import copy
import multiprocessing as mp
import os
import random
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool

import numpy as np
import torch
import torch.nn as nn

from env.engine import total_vp
from env.pettingzoo_env import CatanAECEnv
from training.hier_model import prepare_transition_batch
from training.model_adapters import FLAT_ADAPTER, ModelAdapter
from training.ppo import GAE_LAMBDA, GAMMA, compute_gae


class RotatingOpponents:
    """`opponent_agents` that depends on the episode seed: the trainee sits in
    seat `seed % 4` and every other seat gets `make_opponent(pid)`. Lets one
    rollout call cover all four trainee seats (one worker pool per
    iteration instead of four). Picklable if `make_opponent` is."""

    def __init__(self, make_opponent):
        self.make_opponent = make_opponent

    def __call__(self, seed: int) -> dict[int, object]:
        trainee = seed % 4
        return {pid: self.make_opponent(pid) for pid in range(4) if pid != trainee}


def _episode_opponents(opponent_agents, seed: int) -> dict[int, object]:
    """Per-episode opponent instances seeded from the episode seed. Shallow
    copies (models and pool members stay shared) with fresh RNG state, so
    episodes running concurrently in one process never share an RNG stream.
    `opponent_agents` is a {seat: agent} dict, None, or a callable
    seed -> dict (e.g. RotatingOpponents)."""
    if callable(opponent_agents):
        opponent_agents = opponent_agents(seed)
    out = {}
    for seat, agent in (opponent_agents or {}).items():
        a = copy.copy(agent)
        s = seed * 1_000_003 + seat
        if hasattr(a, "reset_episode"):  # e.g. agents.opponent_pool.PooledOpponent
            a.reset_episode(s)
        elif getattr(a, "rng", None) is not None:
            a.rng = random.Random(s)
        out[seat] = a
    return out


class EpisodeRunner:
    """One episode as a resumable state machine: `advance()` plays forced
    moves, opponent seats and dead-agent steps until a trainee decision is
    needed (returning its encoded observation + legal actions) or the game
    ends (returning None); `apply()` records the model's decision and steps.

    `skip_forced`: decisions with exactly one legal action (ROLL_DICE with no
    playable dev card, a lone discard/robber option, ...) carry no policy
    gradient but, if stored, each still costs one step of discount -- ~20% of
    all transitions (audit F-15). They are played without querying the
    model, and reward earned across them accrues to the agent's previous real
    decision. On truncation each trainee's last transition records
    `bootstrap_value` = V(s_T) for compute_gae.

    All randomness comes from `seed` (board/dice via the env, policy sampling
    via a per-episode NumPy Generator, opponents via `_episode_opponents`),
    so an episode is a pure function of (weights, seed) regardless of how
    many episodes are batched together or which process plays it (F-22)."""

    def __init__(self, env: CatanAECEnv, model, device: str, seed: int,
                 opponent_agents: dict[int, object] | None, adapter: ModelAdapter,
                 skip_forced: bool = True):
        self.env, self.model, self.device, self.seed = env, model, device, seed
        self.adapter, self.skip_forced = adapter, skip_forced
        self.opponents = _episode_opponents(opponent_agents, seed)
        self.rng = np.random.default_rng(seed)
        self.pending: dict[str, dict] = {}
        self.data: dict[str, list] = {a: [] for a in env.possible_agents}
        self._decision = None  # (agent, encoded, legal) awaiting apply()
        env.reset(seed=seed)

    def _finalize(self, agent: str, terminated: bool, truncated: bool) -> None:
        p = self.pending.pop(agent)
        self.data[agent].append({**p, "terminated": terminated, "truncated": truncated,
                                 "done": terminated or truncated})

    def advance(self):
        env = self.env
        while env.agents:
            agent = env.agent_selection
            pid = int(agent.split("_")[1])
            # Skip building PettingZoo's flat observation dict when nothing
            # reads it (graph adapter, opponent seats) -- ~15% of GNN rollout time.
            needs_obs = self.adapter.needs_raw_obs and pid not in self.opponents
            obs, reward, term, trunc, info = env.last(observe=needs_obs)
            env.clear_reward(agent)
            if agent in self.pending:
                self.pending[agent]["reward"] += reward
            if term or trunc:
                if agent in self.pending:
                    if trunc and not term:
                        encoded = self.adapter.encode(env, obs, pid)
                        with torch.inference_mode():
                            self.pending[agent]["bootstrap_value"] = float(
                                self.model.value(self.adapter.to_single(encoded, self.device)).reshape(-1)[0])
                    self._finalize(agent, term, trunc)
                env.step(None)
                continue
            legal = env.legal_actions()
            if pid in self.opponents:
                env.step(_env_action(legal, self.opponents[pid].choose(env.engine.state, legal)))
                continue
            if self.skip_forced and len(legal) == 1:
                env.step(0)  # forced: no transition; reward accrues to the pending one
                continue
            if agent in self.pending:
                self._finalize(agent, False, False)
            encoded = self.adapter.encode(env, obs, pid)
            self._decision = (agent, encoded, legal)
            return encoded, legal
        return None

    def apply(self, action, logprob: float, value: float, head_data: dict) -> None:
        agent, encoded, legal = self._decision
        self._decision = None
        self.pending[agent] = {"obs": encoded, "logprob": logprob, "value": value,
                               "reward": 0.0, **head_data}
        self.env.step(_env_action(legal, action))


def collect_episode(env: CatanAECEnv, model, device: str, seed: int,
                     opponent_agents: dict[int, object] | None = None,
                     adapter: ModelAdapter = FLAT_ADAPTER,
                     skip_forced: bool = True) -> dict:
    """Play one episode; returns each trainee seat's own transition list
    (see EpisodeRunner)."""
    runner = EpisodeRunner(env, model, device, seed, opponent_agents, adapter, skip_forced)
    while (need := runner.advance()) is not None:
        encoded, legal = need
        action, logprob, value, head_data = model.act(adapter.to_single(encoded, device), legal,
                                                      rng=runner.rng)
        runner.apply(action, logprob, value, head_data)
    return runner.data


def collect_episodes_batched(env_kwargs: dict, model, device: str, seeds: list[int],
                             opponent_agents: dict[int, object] | None = None,
                             adapter: ModelAdapter = FLAT_ADAPTER, num_envs: int = 16,
                             skip_forced: bool = True):
    """Play `seeds` with up to `num_envs` games in flight, one batched
    forward pass per round of pending trainee decisions (Phase 4). Yields
    each finished EpisodeRunner (.seed, .data, .env, .opponents); per-episode
    results are identical to `collect_episode` on the same seed."""
    queue = list(seeds)
    active: list[EpisodeRunner] = []

    def start():
        seed = queue.pop(0)
        return EpisodeRunner(CatanAECEnv(**env_kwargs), model, device, seed, opponent_agents,
                             adapter, skip_forced)

    while queue or active:
        while queue and len(active) < num_envs:
            active.append(start())
        waiting, needs = [], []
        for r in active:
            need = r.advance()
            if need is None:
                yield r
            else:
                waiting.append(r)
                needs.append(need)
        active = waiting
        if not waiting:
            continue
        batch = adapter.to_batch([enc for enc, _ in needs], device)
        results = model.act_batch(batch, [legal for _, legal in needs], rngs=[r.rng for r in waiting])
        for r, (action, logprob, value, head_data) in zip(waiting, results):
            r.apply(action, logprob, value, head_data)


def _env_action(legal: list, action):
    """Index into the legal list when the chosen action is listed; otherwise
    (a structured trade built from a template) the concrete Action itself,
    which CatanAECEnv.step validates."""
    for i, a in enumerate(legal):
        if a is action:
            return i
    try:
        return legal.index(action)
    except ValueError:
        return action


def _episode_summary(env: CatanAECEnv, seed: int, opponent_agents: dict[int, object] | None,
                      opponent_name: str | None) -> dict:
    state = env.engine.state
    ranking_pids = sorted(state.players.keys(), key=lambda pid: total_vp(state, pid), reverse=True)
    opponent_agents = opponent_agents or {}
    return {
        "seed": seed,
        "turns": state.turn_number,
        "winner": state.winner,
        "truncated": state.phase.name != "GAME_OVER",
        "ranking_pids": ranking_pids,
        "trainee_pids": [pid for pid in state.players if pid not in opponent_agents],
        "opponent_name": opponent_name,
        # which style sat in each opponent seat (pooled opponents), for
        # per-style win-rate breakdowns
        "opponent_styles": {pid: getattr(getattr(a, "current", None), "name", type(a).__name__)
                            for pid, a in opponent_agents.items()},
    }


def _runner_output(r: "EpisodeRunner", opponent_name, gamma, lam) -> tuple[int, list[dict], dict]:
    transitions: list[dict] = []
    for agent, trs in r.data.items():
        if trs:
            transitions.extend(compute_gae(trs, gamma=gamma, lam=lam))
    return r.seed, transitions, _episode_summary(r.env, r.seed, r.opponents, opponent_name)


def _assemble(episodes) -> tuple[list[dict], list[dict]]:
    # Seed order == the order sequential collection produces, so the PPO
    # update sees identical data however episodes were split or batched (F-22).
    all_transitions: list[dict] = []
    all_summaries: list[dict] = []
    for _, transitions, summary in sorted(episodes, key=lambda ep: ep[0]):
        all_transitions.extend(transitions)
        all_summaries.append(summary)
    return all_transitions, all_summaries


def collect_rollout(env: CatanAECEnv, model, device: str,
                     num_episodes: int, base_seed: int,
                     opponent_agents: dict[int, object] | None = None,
                     opponent_name: str | None = None,
                     adapter: ModelAdapter = FLAT_ADAPTER,
                     gamma: float = GAMMA, lam: float = GAE_LAMBDA) -> tuple[list[dict], list[dict]]:
    """Sequential, one game at a time on `env` (reference implementation;
    collect_rollout_parallel produces identical data faster)."""
    episodes = []
    for i in range(num_episodes):
        r = EpisodeRunner(env, model, device, base_seed + i, opponent_agents, adapter)
        while (need := r.advance()) is not None:
            encoded, legal = need
            r.apply(*model.act(adapter.to_single(encoded, device), legal, rng=r.rng))
        episodes.append(_runner_output(r, opponent_name, gamma, lam))
    return _assemble(episodes)


# -- parallel rollout collection -------------------------------------------
# Uses fork (default on Linux) so the model -- already updated in the parent
# after each PPO step -- is inherited by worker processes via copy-on-write
# memory rather than being pickled through `initargs` on every iteration.
# Each iteration forks a fresh pool so workers always see the latest weights;
# fork itself is cheap (no re-importing torch/modules), so the main cost
# stays in the actual rollout collection, not process startup.
#
# Known tradeoff: forking a process that already has PyTorch's internal
# thread pool running triggers Python's "fork() in a multi-threaded process
# may deadlock" warning. We accept this rather than switching to
# `forkserver` (which forks from a clean single-threaded server instead, but
# loses the free copy-on-write model sharing and would need to pickle the
# model every iteration instead). In practice `torch.set_num_threads(1)` at
# the top of `_init_worker` shrinks the risk window immediately, and this has
# run correctly across every benchmark and test in this repo -- but if you
# ever see a hang here on a different machine/torch build, `forkserver` with
# explicit model pickling is the safe fallback.
_worker_model = None
_worker_env_kwargs: dict | None = None
_worker_opponent_agents: dict[int, object] | None = None
_worker_opponent_name: str | None = None
_worker_adapter: ModelAdapter = FLAT_ADAPTER
_worker_gamma: float = GAMMA
_worker_lam: float = GAE_LAMBDA
_worker_num_envs: int = 1
_worker_device: str = "cpu"


def reseed_forked_worker(opponent_agents: dict[int, object] | None = None) -> None:
    """Give this forked worker its own RNG streams for anything NOT covered by
    collect_episode's per-episode seeding (e.g. ad-hoc sampling in eval
    helpers). Rollout data itself no longer depends on this. fork copies the parent's
    torch RNG state byte-for-byte, so without this every worker's
    `dist.sample()`/`multinomial` stream is identical -- correlated
    exploration noise across supposedly independent workers. Same story for
    opponent agents' `random.Random` instances, which were constructed in the
    parent and inherited by every worker. PID-derived seeds keep the streams
    distinct per worker (per-run reproducibility of parallel rollouts was
    already off the table -- fork inherits whatever parent RNG state existed
    at pool creation)."""
    pid_salt = os.getpid() * 0x9E3779B1
    torch.manual_seed((torch.initial_seed() ^ pid_salt) % (2 ** 63))
    if opponent_agents and not callable(opponent_agents):  # callables build fresh, seeded agents per episode
        for seat, agent in opponent_agents.items():
            rng = getattr(agent, "rng", None)
            if rng is not None:
                rng.seed(pid_salt + seat)


def _init_worker(model, env_kwargs: dict, opponent_agents: dict[int, object] | None,
                  opponent_name: str | None, adapter: ModelAdapter,
                  gamma: float = GAMMA, lam: float = GAE_LAMBDA, num_envs: int = 1,
                  device: str = "cpu") -> None:
    global _worker_model, _worker_env_kwargs, _worker_opponent_agents, _worker_opponent_name
    global _worker_adapter, _worker_gamma, _worker_lam, _worker_num_envs, _worker_device
    torch.set_num_threads(1)  # avoid N workers each spawning their own thread pool
    reseed_forked_worker(opponent_agents)
    model.to(device).eval()
    _worker_model = model
    _worker_device = device
    _worker_env_kwargs = env_kwargs
    _worker_opponent_agents = opponent_agents
    _worker_opponent_name = opponent_name
    _worker_adapter = adapter
    _worker_gamma = gamma
    _worker_lam = lam
    _worker_num_envs = num_envs


def _worker_collect(seeds: list[int]) -> list[tuple[int, list[dict], dict]]:
    """Per-episode (seed, transitions, summary) so the parent can reassemble
    results in seed order, independent of how seeds were split across
    workers or batched within one."""
    return [_runner_output(r, _worker_opponent_name, _worker_gamma, _worker_lam)
            for r in collect_episodes_batched(_worker_env_kwargs, _worker_model, _worker_device, seeds,
                                              _worker_opponent_agents, _worker_adapter,
                                              num_envs=_worker_num_envs)]


def collect_rollout_parallel(env_kwargs: dict, model,
                              num_episodes: int, base_seed: int, num_workers: int,
                              opponent_agents: dict[int, object] | None = None,
                              opponent_name: str | None = None,
                              adapter: ModelAdapter = FLAT_ADAPTER,
                              gamma: float = GAMMA, lam: float = GAE_LAMBDA,
                              envs_per_worker: int = 1,
                              inference_device: str = "cpu") -> tuple[list[dict], list[dict]]:
    """Rollouts across `num_workers` processes, each running
    `envs_per_worker` games concurrently with one batched forward pass per
    round. CPU inference uses forked workers (model shared copy-on-write);
    GPU inference (`inference_device="cuda"`) uses *spawned* workers, each
    with its own CUDA context and a copy of the weights -- the fast path for
    the GNN, whose batch-1 CPU forward is ~6 ms but ~0.14 ms/sample batched
    on a GPU. With num_workers == 1 everything runs in this process (the
    model must already be on `inference_device`). Output is identical in
    every configuration (F-22)."""
    seeds = list(range(base_seed, base_seed + num_episodes))
    if num_workers <= 1:
        model.to(inference_device).eval()
        episodes = [_runner_output(r, opponent_name, gamma, lam)
                    for r in collect_episodes_batched(env_kwargs, model, inference_device, seeds,
                                                      opponent_agents, adapter, num_envs=envs_per_worker)]
        return _assemble(episodes)
    num_workers = max(1, min(num_workers, num_episodes))
    chunks = [seeds[i::num_workers] for i in range(num_workers)]
    chunks = [c for c in chunks if c]

    if inference_device != "cpu":
        return _assemble(_collect_spawned(model, env_kwargs, chunks, opponent_agents, opponent_name,
                                          adapter, gamma, lam, envs_per_worker, inference_device))
    ctx, worker_model = mp.get_context("fork"), model
    # concurrent.futures rather than multiprocessing.Pool: a worker that dies
    # (e.g. failing to initialize) raises BrokenProcessPool instead of being
    # silently respawned forever.
    for attempt in range(3):
        try:
            with ProcessPoolExecutor(max_workers=len(chunks), mp_context=ctx, initializer=_init_worker,
                                     initargs=(worker_model, env_kwargs, opponent_agents, opponent_name,
                                               adapter, gamma, lam, envs_per_worker,
                                               inference_device)) as pool:
                results = list(pool.map(_worker_collect, chunks))
            break
        except BrokenProcessPool:
            # A worker died abruptly (seen once: a PyTorch-internal clock
            # assertion in a spawned CUDA worker). Rollouts are a pure function
            # of (weights, seeds), so retrying yields identical data.
            if attempt == 2:
                raise
            print(f"rollout worker pool broke; retrying ({attempt + 1}/2)", flush=True)
    return _assemble(ep for chunk in results for ep in chunk)


# -- persistent spawned (GPU-inference) workers -------------------------------
# A CUDA context can't survive fork, so GPU-inference workers are spawned -- and
# spawning + CUDA initialisation costs ~2.5 s per call (performance audit),
# ~40% of a typical rl_finetune rollout. These pools are therefore created once
# per (workers, device, architecture) and reused; every call ships the current
# weights (state_dict) and that call's settings with the task. Forked CPU pools
# stay per-call: forking is ~0.1 s and inherits the current weights for free.
_SPAWN_POOLS: dict[tuple, ProcessPoolExecutor] = {}


def _arch_key(model) -> tuple:
    return (type(model).__name__,) + tuple((k, tuple(v.shape)) for k, v in model.state_dict().items())


def _init_spawned_worker(model_template, device: str) -> None:
    global _worker_model, _worker_device
    torch.set_num_threads(1)
    _worker_model = model_template.to(device).eval()
    _worker_device = device


def _spawned_task(payload) -> list[tuple[int, list[dict], dict]]:
    global _worker_env_kwargs, _worker_opponent_agents, _worker_opponent_name
    global _worker_adapter, _worker_gamma, _worker_lam, _worker_num_envs
    (state_dict, env_kwargs, opponents, opponent_name, adapter, gamma, lam, num_envs, seeds) = payload
    _worker_model.load_state_dict(state_dict)
    _worker_model.eval()
    _worker_env_kwargs, _worker_opponent_agents, _worker_opponent_name = env_kwargs, opponents, opponent_name
    _worker_adapter, _worker_gamma, _worker_lam, _worker_num_envs = adapter, gamma, lam, num_envs
    return _worker_collect(seeds)


def _collect_spawned(model, env_kwargs, chunks, opponent_agents, opponent_name, adapter, gamma, lam,
                     envs_per_worker, device):
    key = (len(chunks), device, _arch_key(model))
    state_dict = {k: v.detach().to("cpu") for k, v in model.state_dict().items()}
    payloads = [(state_dict, env_kwargs, opponent_agents, opponent_name, adapter, gamma, lam,
                 envs_per_worker, c) for c in chunks]
    for attempt in range(3):
        pool = _SPAWN_POOLS.get(key)
        if pool is None:
            pool = ProcessPoolExecutor(max_workers=len(chunks), mp_context=mp.get_context("spawn"),
                                       initializer=_init_spawned_worker,
                                       initargs=(copy.deepcopy(model).to("cpu"), device))
            _SPAWN_POOLS[key] = pool
        try:
            return [ep for chunk in pool.map(_spawned_task, payloads) for ep in chunk]
        except BrokenProcessPool:
            # a worker died (seen once: a PyTorch-internal clock assertion in a
            # spawned CUDA worker); rollouts are deterministic, so rebuild + retry
            _SPAWN_POOLS.pop(key, None)
            pool.shutdown(wait=False, cancel_futures=True)
            if attempt == 2:
                raise
            print(f"rollout worker pool broke; retrying ({attempt + 1}/2)", flush=True)


def close_rollout_pools() -> None:
    """Shut down persistent GPU rollout workers (frees their CUDA contexts,
    ~134 MiB each plus the model)."""
    for pool in _SPAWN_POOLS.values():
        pool.shutdown(wait=True, cancel_futures=True)
    _SPAWN_POOLS.clear()


atexit.register(close_rollout_pools)


def load_bc_anchor(path: str, device: str, max_samples: int | None = None,
                    seed: int = 0) -> dict[str, torch.Tensor]:
    """Load a demonstration dataset (scripts/collect_heuristic_demos.py .npz)
    as device-resident tensors for use as a BC anchor in `ppo_update`. The
    arrays are already in the exact prepared-transition-batch layout
    `evaluate_actions` consumes. `max_samples` subsamples (without
    replacement, seeded) to bound device memory -- ~2.7 KB/sample."""
    npz = np.load(path)
    n = npz["type_idx"].shape[0]
    idx = None
    if max_samples is not None and n > max_samples:
        idx = np.random.RandomState(seed).choice(n, max_samples, replace=False)
    return {k: torch.as_tensor(npz[k] if idx is None else npz[k][idx], device=device)
            for k in npz.files}


def compute_holdout_nll(model, holdout: dict[str, torch.Tensor], batch_size: int = 8192) -> float:
    """Mean negative log-likelihood of a held-out demonstration set under the
    current policy -- the generalization signal for detecting BC-anchor
    erosion and selecting an early-stopped checkpoint. Deliberately
    independent of whatever subsample `load_bc_anchor` is using as the
    anchor term: the anchor dataset is what the loss directly optimizes
    (so its NLL improving is partly just fitting), while this dataset is
    never trained on, so its NLL rising is a real drift signal. Chunked to
    bound peak memory on large holdout sets."""
    model.eval()
    n = holdout["type_idx"].shape[0]
    total = 0.0
    with torch.inference_mode():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            tb = {k: v[start:end] for k, v in holdout.items() if k != "obs"}
            # obs is a flat Tensor (hier) or dict[str, Tensor] (gnn)
            obs = holdout["obs"]
            obs_mb = {k: v[start:end] for k, v in obs.items()} if isinstance(obs, dict) \
                else obs[start:end]
            logprob, _, _ = model.evaluate_actions(obs_mb, tb)
            total += float(logprob.sum())
    model.train()
    return -total / n


def _index_batch(batch, idx):
    """Index into an already-batched observation (a plain Tensor for
    FLAT_ADAPTER, or a dict[str, Tensor] for GRAPH_ADAPTER) without rebuilding
    it from the underlying Python/numpy observations. `idx` is a long Tensor
    (or anything torch fancy-indexing accepts)."""
    if isinstance(batch, dict):
        return {k: v[idx] for k, v in batch.items()}
    return batch[idx]


def ppo_update(model, optimizer: torch.optim.Optimizer, transitions: list[dict],
               clip_ratio: float = 0.2, value_coef: float = 0.5, entropy_coef: float = 0.01,
               epochs: int = 4, minibatch_size: int = 256, max_grad_norm: float = 0.5,
               device: str = "cpu", adapter: ModelAdapter = FLAT_ADAPTER,
               target_kl: float | None = 0.02,
               bc_dataset: dict[str, torch.Tensor] | None = None, bc_coef: float = 0.0,
               bc_minibatch_size: int = 512, train_policy: bool = True) -> dict:
    """`bc_dataset`/`bc_coef`: optional BC anchor (see `load_bc_anchor`).
    Each gradient step adds `bc_coef * NLL(demonstrated actions)` on a fresh
    random demo minibatch. Near the BC optimum this gradient is ~zero, and it
    grows as the policy drifts from the demonstrated behavior -- so PPO can
    only move the policy away from the demonstrations where the clipped
    surrogate gain outweighs the anchor penalty (the AlphaStar-style guard
    against RL fine-tuning destroying cloned skills). The demo forward pass
    shares no rows with the rollout minibatch, so the value/entropy terms are
    untouched by anchor rows."""
    advantages_all = np.array([t["advantage"] for t in transitions], dtype=np.float32)
    advantages_all = (advantages_all - advantages_all.mean()) / (advantages_all.std() + 1e-8)

    # Build the whole rollout's tensors once, not once per minibatch per
    # epoch -- observations (this used to rebuild a Python list -> np.array
    # -> torch.tensor, or a whole dict-of-arrays for the GNN adapter, on
    # every single minibatch), the per-transition head data (masks/indices/
    # active-head ids for evaluate_actions), and the PPO scalars. Minibatches
    # then just index into device-resident tensors.
    obs_batch_full = adapter.to_batch([t["obs"] for t in transitions], device)
    eval_batch_full = prepare_transition_batch(transitions, device)
    old_logprobs_full = torch.tensor([t["logprob"] for t in transitions], dtype=torch.float32, device=device)
    advantages_full = torch.as_tensor(advantages_all, device=device)
    returns_full = torch.tensor([t["return"] for t in transitions], dtype=torch.float32, device=device)

    n = len(transitions)
    idx = np.arange(n)
    stats = {"policy_loss": [], "value_loss": [], "entropy": [], "approx_kl": [], "clip_frac": [],
             "grad_norm": []}
    use_bc = bc_dataset is not None and bc_coef > 0
    if use_bc:
        stats["bc_loss"] = []
        n_demo = bc_dataset["type_idx"].shape[0]

    for _ in range(epochs):
        np.random.shuffle(idx)
        epoch_kls = []
        for start in range(0, n, minibatch_size):
            mb = idx[start:start + minibatch_size]
            mb_t = torch.as_tensor(mb, dtype=torch.long, device=device)
            obs_batch = _index_batch(obs_batch_full, mb_t)
            eval_batch = {k: v[mb_t] for k, v in eval_batch_full.items()}
            old_logprob = old_logprobs_full[mb_t]
            adv = advantages_full[mb_t]
            ret = returns_full[mb_t]

            logprob, entropy, value = model.evaluate_actions(obs_batch, eval_batch)
            ratio = torch.exp(logprob - old_logprob)
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * adv
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = ((value - ret) ** 2).mean()
            entropy_loss = -entropy.mean()
            if train_policy:
                loss = policy_loss + value_coef * value_loss + entropy_coef * entropy_loss
            else:
                # critic warm-up: fit V to real returns before its advantages
                # are allowed to move the (cloned) policy
                loss = value_coef * value_loss

            if use_bc and train_policy:
                demo_idx = torch.randint(n_demo, (min(bc_minibatch_size, n_demo),), device=device)
                demo_tb = {k: v[demo_idx] for k, v in bc_dataset.items() if k != "obs"}
                demo_logprob, _, _ = model.evaluate_actions(
                    _index_batch(bc_dataset["obs"], demo_idx), demo_tb)
                bc_loss = -demo_logprob.mean()
                loss = loss + bc_coef * bc_loss
                stats["bc_loss"].append(bc_loss.item())

            optimizer.zero_grad()
            loss.backward()
            if not train_policy:
                # warm-up touches only the value head: gradients into the
                # shared trunk would shift the policy's outputs too
                for name, p in model.named_parameters():
                    if not name.startswith("value_head"):
                        p.grad = None
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)  # pre-clip norm
            optimizer.step()
            stats["grad_norm"].append(float(grad_norm))

            with torch.no_grad():
                approx_kl = (old_logprob - logprob).mean().item()
                clip_frac = ((ratio - 1.0).abs() > clip_ratio).float().mean().item()
            stats["policy_loss"].append(policy_loss.item())
            stats["value_loss"].append(value_loss.item())
            stats["entropy"].append(entropy.mean().item())
            stats["approx_kl"].append(approx_kl)
            stats["clip_frac"].append(clip_frac)
            epoch_kls.append(approx_kl)

        # Stop before a further epoch would push the policy too far from the
        # one that collected this rollout -- see training/ppo.py's ppo_update
        # for the same guard and the reasoning behind the abs().
        if target_kl is not None and abs(float(np.mean(epoch_kls))) > target_kl:
            break

    out = {k: float(np.mean(v)) for k, v in stats.items() if v}
    # Explained variance of the rollout's value predictions w.r.t. their GAE
    # return targets: ~0 means the critic explains nothing, 1 is perfect.
    values = np.array([t["value"] for t in transitions], dtype=np.float64)
    returns = np.array([t["return"] for t in transitions], dtype=np.float64)
    var = returns.var()
    out["explained_variance"] = float(1.0 - (returns - values).var() / var) if var > 1e-12 else float("nan")
    return out


def fit_value_head_lstsq(model, fit_transitions: list[dict], heldout_transitions: list[dict],
                         adapter: ModelAdapter, device: str = "cpu", ridge: float = 0.3,
                         batch_size: int = 2048) -> dict:
    """Closed-form critic initialisation: ridge regression of Monte-Carlo
    returns (compute transitions with lam=1.0 so targets don't depend on the
    untrained critic) onto the frozen trunk features, written into
    `model.value_head`. A cloned policy's critic otherwise starts with
    negative explained variance, and its noisy advantages drag the policy
    down for the first tens of iterations (pilot 2026-10-04). Returns
    explained variance on `heldout_transitions` before and after."""
    from training.imitation_data import model_features

    def feats_and_targets(trs):
        xs, ys = [], []
        model.to(device).eval()
        with torch.inference_mode():
            for i in range(0, len(trs), batch_size):
                chunk = trs[i:i + batch_size]
                xs.append(model_features(model, adapter.to_batch([t["obs"] for t in chunk], device)).double().cpu())
                ys.append(torch.tensor([t["return"] for t in chunk], dtype=torch.float64))
        return torch.cat(xs), torch.cat(ys)

    def ev(x, y):
        with torch.inference_mode():
            pred = model.value_head(x.to(device=device, dtype=torch.float32)).squeeze(-1).double().cpu()
        return float(1 - (y - pred).var() / y.var()) if y.var() > 0 else float("nan")

    x_fit, y_fit = feats_and_targets(fit_transitions)
    x_hold, y_hold = feats_and_targets(heldout_transitions)
    before = ev(x_hold, y_hold)
    xa = torch.cat([x_fit, torch.ones(len(x_fit), 1, dtype=torch.float64)], dim=1)
    # Ridge strength relative to the feature scale (lambda = ridge * mean
    # diagonal of X^T X): returns are nearly constant within a player's game,
    # so a weakly-regularised fit memorises trajectories and extrapolates
    # wildly on held-out games.
    gram = xa.T @ xa
    lam = ridge * float(torch.diagonal(gram)[:-1].mean())
    reg = lam * torch.eye(xa.shape[1], dtype=torch.float64)
    reg[-1, -1] = 0.0  # don't shrink the bias
    w = torch.linalg.solve(gram + reg, xa.T @ y_fit)
    with torch.no_grad():
        model.value_head.weight.copy_(w[:-1].to(model.value_head.weight.dtype).view(1, -1))
        model.value_head.bias.copy_(w[-1:].to(model.value_head.bias.dtype))
    return {"ev_before": before, "ev_after": ev(x_hold, y_hold),
            "fit_rows": len(x_fit), "heldout_rows": len(x_hold)}
