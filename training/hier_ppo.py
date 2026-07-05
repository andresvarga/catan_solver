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

import numpy as np
import torch
import torch.nn as nn

from env.engine import total_vp
from env.pettingzoo_env import CatanAECEnv
from training.model_adapters import FLAT_ADAPTER, ModelAdapter
from training.ppo import GAE_LAMBDA, GAMMA, compute_gae


def collect_episode(env: CatanAECEnv, model, device: str, seed: int,
                     opponent_agents: dict[int, object] | None = None,
                     adapter: ModelAdapter = FLAT_ADAPTER) -> dict:
    opponent_agents = opponent_agents or {}
    env.reset(seed=seed)
    pending: dict[str, dict] = {}
    episode_data: dict[str, list] = {a: [] for a in env.possible_agents}

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
            p = pending.pop(agent)
            bootstrap_value = 0.0
            if trunc and not term:
                # Truncated (hit max_episode_steps), not a true win/loss: bootstrap
                # with the model's own value estimate of the final observation
                # instead of treating it as if there were no future reward.
                encoded_final = adapter.encode(env, obs, pid)
                obs_final_t = adapter.to_single(encoded_final, device)
                with torch.no_grad():
                    bootstrap_value = float(model.value(obs_final_t).item())
            episode_data[agent].append({
                **p, "reward": reward, "terminated": term, "truncated": trunc,
                "done": done, "bootstrap_value": bootstrap_value,
            })

        if done:
            action_idx = None
        elif pid in opponent_agents:
            legal = env.legal_actions()
            concrete_action = opponent_agents[pid].choose(env.engine.state)
            action_idx = legal.index(concrete_action)
        else:
            legal = env.legal_actions()
            encoded = adapter.encode(env, obs, pid)
            obs_t = adapter.to_single(encoded, device)
            with torch.no_grad():
                concrete_action, logprob, value, head_data = model.act(obs_t, legal, deterministic=False)
            action_idx = legal.index(concrete_action)
            pending[agent] = {"obs": encoded, "logprob": logprob, "value": value, **head_data}
        env.step(action_idx)

    return episode_data


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


def _init_worker(model, env_kwargs: dict, opponent_agents: dict[int, object] | None,
                  opponent_name: str | None, adapter: ModelAdapter,
                  gamma: float = GAMMA, lam: float = GAE_LAMBDA) -> None:
    global _worker_model, _worker_env_kwargs, _worker_opponent_agents, _worker_opponent_name
    global _worker_adapter, _worker_gamma, _worker_lam
    torch.set_num_threads(1)  # avoid N workers each spawning their own thread pool
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


def _index_batch(batch, idx: np.ndarray):
    """Index into an already-batched observation (a plain Tensor for
    FLAT_ADAPTER, or a dict[str, Tensor] for GRAPH_ADAPTER) without rebuilding
    it from the underlying Python/numpy observations."""
    if isinstance(batch, dict):
        return {k: v[idx] for k, v in batch.items()}
    return batch[idx]


def ppo_update(model, optimizer: torch.optim.Optimizer, transitions: list[dict],
               clip_ratio: float = 0.2, value_coef: float = 0.5, entropy_coef: float = 0.01,
               epochs: int = 4, minibatch_size: int = 256, max_grad_norm: float = 0.5,
               device: str = "cpu", adapter: ModelAdapter = FLAT_ADAPTER,
               target_kl: float | None = 0.02) -> dict:
    old_logprobs_all = np.array([t["logprob"] for t in transitions], dtype=np.float32)
    advantages_all = np.array([t["advantage"] for t in transitions], dtype=np.float32)
    returns_all = np.array([t["return"] for t in transitions], dtype=np.float32)
    advantages_all = (advantages_all - advantages_all.mean()) / (advantages_all.std() + 1e-8)

    # Build the whole rollout's observation batch once, not once per
    # minibatch per epoch -- this used to rebuild (Python list -> np.array ->
    # torch.tensor, or a whole dict-of-arrays for the GNN adapter) on every
    # single minibatch, `epochs` times over.
    obs_batch_full = adapter.to_batch([t["obs"] for t in transitions], device)

    n = len(transitions)
    idx = np.arange(n)
    stats = {"policy_loss": [], "value_loss": [], "entropy": [], "approx_kl": [], "clip_frac": []}

    for _ in range(epochs):
        np.random.shuffle(idx)
        epoch_kls = []
        for start in range(0, n, minibatch_size):
            mb = idx[start:start + minibatch_size]
            mb_transitions = [transitions[i] for i in mb]
            obs_batch = _index_batch(obs_batch_full, mb)
            old_logprob = torch.tensor(old_logprobs_all[mb], dtype=torch.float32, device=device)
            adv = torch.tensor(advantages_all[mb], dtype=torch.float32, device=device)
            ret = torch.tensor(returns_all[mb], dtype=torch.float32, device=device)

            logprob, entropy, value = model.evaluate_actions(obs_batch, mb_transitions)
            ratio = torch.exp(logprob - old_logprob)
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * adv
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = ((value - ret) ** 2).mean()
            entropy_loss = -entropy.mean()
            loss = policy_loss + value_coef * value_loss + entropy_coef * entropy_loss

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
