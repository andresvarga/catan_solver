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

import multiprocessing as mp
import os

import numpy as np
import torch
import torch.nn as nn

from env.engine import total_vp
from env.pettingzoo_env import CatanAECEnv
from training.hier_model import prepare_transition_batch
from training.model_adapters import FLAT_ADAPTER, ModelAdapter
from training.ppo import GAE_LAMBDA, GAMMA, compute_gae


def collect_episode(env: CatanAECEnv, model, device: str, seed: int,
                     opponent_agents: dict[int, object] | None = None,
                     adapter: ModelAdapter = FLAT_ADAPTER,
                     skip_forced: bool = True) -> dict:
    """Play one episode, returning each trainee seat's own transition list.

    `skip_forced`: decisions with exactly one legal action (ROLL_DICE with no
    playable dev card, a lone discard/robber option, ...) carry no policy
    gradient but, if stored, each still costs one step of discount -- ~20% of
    all transitions (audit F-15). With skip_forced they are played without
    querying the model, and any reward earned across them accrues to the
    agent's previous real decision. On truncation each trainee's last
    transition records `bootstrap_value` = V(s_T) for compute_gae."""
    opponent_agents = opponent_agents or {}
    env.reset(seed=seed)
    pending: dict[str, dict] = {}
    episode_data: dict[str, list] = {a: [] for a in env.possible_agents}

    def finalize(agent: str, terminated: bool, truncated: bool) -> None:
        p = pending.pop(agent)
        episode_data[agent].append({**p, "terminated": terminated, "truncated": truncated,
                                    "done": terminated or truncated})

    while env.agents:
        agent = env.agent_selection
        pid = int(agent.split("_")[1])
        # Skip building PettingZoo's own flat observation dict whenever
        # nothing below will read it: the graph adapter derives everything
        # from env.engine.state instead (profiling showed this is ~15% of
        # GNN rollout time otherwise spent building a value nothing ever
        # reads), and an opponent-controlled seat picks its action straight
        # from env.engine.state too, regardless of adapter.
        needs_obs = adapter.needs_raw_obs and pid not in opponent_agents
        obs, reward, term, trunc, info = env.last(observe=needs_obs)
        env.clear_reward(agent)
        done = term or trunc

        if agent in pending:
            pending[agent]["reward"] += reward

        if done:
            if agent in pending:
                if trunc and not term:
                    encoded = adapter.encode(env, obs, pid)
                    with torch.inference_mode():
                        pending[agent]["bootstrap_value"] = float(
                            model.value(adapter.to_single(encoded, device)).reshape(-1)[0])
                finalize(agent, term, trunc)
            action_idx = None
        elif pid in opponent_agents:
            # Pass the env's cached legal-action list through so the agent
            # doesn't re-enumerate it, and so its chosen Action is (usually)
            # the same object -- making the .index() lookup an identity scan.
            legal = env.legal_actions()
            concrete_action = opponent_agents[pid].choose(env.engine.state, legal)
            action_idx = _env_action(legal, concrete_action)
        else:
            legal = env.legal_actions()
            if skip_forced and len(legal) == 1:
                action_idx = 0  # forced move: no transition, reward accrues to the pending one
            else:
                if agent in pending:
                    finalize(agent, False, False)
                encoded = adapter.encode(env, obs, pid)
                obs_t = adapter.to_single(encoded, device)
                with torch.inference_mode():
                    concrete_action, logprob, value, head_data = model.act(obs_t, legal, deterministic=False)
                action_idx = _env_action(legal, concrete_action)
                pending[agent] = {"obs": encoded, "logprob": logprob, "value": value,
                                  "reward": 0.0, **head_data}
        env.step(action_idx)

    return episode_data


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
    }


def collect_rollout(env: CatanAECEnv, model, device: str,
                     num_episodes: int, base_seed: int,
                     opponent_agents: dict[int, object] | None = None,
                     opponent_name: str | None = None,
                     adapter: ModelAdapter = FLAT_ADAPTER,
                     gamma: float = GAMMA, lam: float = GAE_LAMBDA) -> tuple[list[dict], list[dict]]:
    all_transitions = []
    episode_summaries = []
    for i in range(num_episodes):
        seed = base_seed + i
        episode_data = collect_episode(env, model, device, seed, opponent_agents=opponent_agents, adapter=adapter)
        for agent, transitions in episode_data.items():
            if transitions:
                all_transitions.extend(compute_gae(transitions, gamma=gamma, lam=lam))
        episode_summaries.append(_episode_summary(env, seed, opponent_agents, opponent_name))
    return all_transitions, episode_summaries


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


def reseed_forked_worker(opponent_agents: dict[int, object] | None = None) -> None:
    """Give this forked worker its own RNG streams. fork copies the parent's
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
    if opponent_agents:
        for seat, agent in opponent_agents.items():
            rng = getattr(agent, "rng", None)
            if rng is not None:
                rng.seed(pid_salt + seat)


def _init_worker(model, env_kwargs: dict, opponent_agents: dict[int, object] | None,
                  opponent_name: str | None, adapter: ModelAdapter,
                  gamma: float = GAMMA, lam: float = GAE_LAMBDA) -> None:
    global _worker_model, _worker_env_kwargs, _worker_opponent_agents, _worker_opponent_name
    global _worker_adapter, _worker_gamma, _worker_lam
    torch.set_num_threads(1)  # avoid N workers each spawning their own thread pool
    reseed_forked_worker(opponent_agents)
    model.eval()
    _worker_model = model
    _worker_env_kwargs = env_kwargs
    _worker_opponent_agents = opponent_agents
    _worker_opponent_name = opponent_name
    _worker_adapter = adapter
    _worker_gamma = gamma
    _worker_lam = lam


def _worker_collect(seeds: list[int]) -> tuple[list[dict], list[dict]]:
    env = CatanAECEnv(**_worker_env_kwargs)
    transitions: list[dict] = []
    summaries: list[dict] = []
    for seed in seeds:
        episode_data = collect_episode(env, _worker_model, "cpu", seed,
                                        opponent_agents=_worker_opponent_agents, adapter=_worker_adapter)
        for agent, trs in episode_data.items():
            if trs:
                transitions.extend(compute_gae(trs, gamma=_worker_gamma, lam=_worker_lam))
        summaries.append(_episode_summary(env, seed, _worker_opponent_agents, _worker_opponent_name))
    return transitions, summaries


def collect_rollout_parallel(env_kwargs: dict, model,
                              num_episodes: int, base_seed: int, num_workers: int,
                              opponent_agents: dict[int, object] | None = None,
                              opponent_name: str | None = None,
                              adapter: ModelAdapter = FLAT_ADAPTER,
                              gamma: float = GAMMA, lam: float = GAE_LAMBDA) -> tuple[list[dict], list[dict]]:
    seeds = list(range(base_seed, base_seed + num_episodes))
    num_workers = max(1, min(num_workers, num_episodes))
    chunks = [seeds[i::num_workers] for i in range(num_workers)]
    chunks = [c for c in chunks if c]

    ctx = mp.get_context("fork")
    with ctx.Pool(processes=len(chunks), initializer=_init_worker,
                  initargs=(model, env_kwargs, opponent_agents, opponent_name, adapter, gamma, lam)) as pool:
        results = pool.map(_worker_collect, chunks)

    all_transitions: list[dict] = []
    all_summaries: list[dict] = []
    for transitions, summaries in results:
        all_transitions.extend(transitions)
        all_summaries.extend(summaries)
    return all_transitions, all_summaries


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
               bc_minibatch_size: int = 512) -> dict:
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
    stats = {"policy_loss": [], "value_loss": [], "entropy": [], "approx_kl": [], "clip_frac": []}
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
            loss = policy_loss + value_coef * value_loss + entropy_coef * entropy_loss

            if use_bc:
                demo_idx = torch.randint(n_demo, (min(bc_minibatch_size, n_demo),), device=device)
                demo_tb = {k: v[demo_idx] for k, v in bc_dataset.items() if k != "obs"}
                demo_logprob, _, _ = model.evaluate_actions(
                    _index_batch(bc_dataset["obs"], demo_idx), demo_tb)
                bc_loss = -demo_logprob.mean()
                loss = loss + bc_coef * bc_loss
                stats["bc_loss"].append(bc_loss.item())

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

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

    return {k: float(np.mean(v)) for k, v in stats.items()}
