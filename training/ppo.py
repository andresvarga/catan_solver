"""Self-play PPO for the Catan AEC env (roadmap phase 4).

One shared policy plays all four seats. AEC turn order interleaves agents, so
credit assignment has to reconstruct each agent's *own* decision sequence
before running GAE -- reward earned between two of an agent's own decisions
(from opponents acting, dice production, robber steals, etc.) all belongs to
the transition that led into that gap. We replicate the same
"zero-cumulative-reward-on-read" trick PettingZoo's own
`TerminateIllegalWrapper` uses, rather than depending on a wrapper, so
`last()` always returns "reward since I was last asked to act."

No centralized critic yet (MAPPO's CTDE piece from §6) -- the value head
here only sees the same redacted per-agent observation the actor does. That
simplification is fine for a first working loop; wiring in ground-truth
critic inputs is a natural next step once this baseline trains end to end.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from env.pettingzoo_env import CatanAECEnv
from training.model import ActorCritic, flatten_observation

GAMMA = 0.99
GAE_LAMBDA = 0.95


def collect_episode(env: CatanAECEnv, model: ActorCritic, device: str, seed: int) -> dict:
    env.reset(seed=seed)
    pending: dict[str, dict] = {}
    episode_data: dict[str, list] = {a: [] for a in env.possible_agents}

    while env.agents:
        agent = env.agent_selection
        obs, reward, term, trunc, info = env.last()
        env.clear_reward(agent)
        done = term or trunc

        if agent in pending:
            p = pending.pop(agent)
            bootstrap_value = 0.0
            if trunc and not term:
                # Truncated (hit max_episode_steps), not a true win/loss: bootstrap
                # with the model's own value estimate of the final observation
                # instead of treating it as if there were no future reward.
                flat_final = flatten_observation(obs)
                obs_final_t = torch.tensor(flat_final, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    bootstrap_value = float(model.value(obs_final_t).item())
            episode_data[agent].append({
                **p, "reward": reward, "terminated": term, "truncated": trunc,
                "done": done, "bootstrap_value": bootstrap_value,
            })

        if done:
            action = None
        else:
            flat = flatten_observation(obs)
            mask = obs["action_mask"]
            obs_t = torch.tensor(flat, dtype=torch.float32, device=device).unsqueeze(0)
            mask_t = torch.tensor(mask, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                action_t, logprob_t, _, value_t = model.act(obs_t, mask_t)
            action = int(action_t.item())
            pending[agent] = {
                "obs": flat, "mask": mask, "action": action,
                "logprob": float(logprob_t.item()), "value": float(value_t.item()),
            }
        env.step(action)

    return episode_data


def compute_gae(transitions: list[dict], gamma: float = GAMMA, lam: float = GAE_LAMBDA) -> list[dict]:
    if not transitions:
        return transitions
    # Bootstrap the last transition with the model's own value estimate if the
    # episode ended by truncation rather than a true terminal state (see
    # collect_episode's "bootstrap_value"); defaults to 0.0 for a real terminal.
    final_bootstrap = transitions[-1].get("bootstrap_value", 0.0)
    values = [t["value"] for t in transitions] + [final_bootstrap]
    advantages = [0.0] * len(transitions)
    gae = 0.0
    for t in reversed(range(len(transitions))):
        terminated = transitions[t].get("terminated", transitions[t]["done"])
        mask = 0.0 if terminated else 1.0
        delta = transitions[t]["reward"] + gamma * values[t + 1] * mask - values[t]
        gae = delta + gamma * lam * mask * gae
        advantages[t] = gae
    for t, adv in zip(transitions, advantages):
        t["advantage"] = adv
        t["return"] = adv + t["value"]
    return transitions


def collect_rollout(env: CatanAECEnv, model: ActorCritic, device: str,
                     num_episodes: int, base_seed: int,
                     gamma: float = GAMMA, lam: float = GAE_LAMBDA) -> tuple[list[dict], list[dict]]:
    """Returns (flat list of processed transitions ready for PPO, per-episode summaries)."""
    all_transitions = []
    episode_summaries = []
    for i in range(num_episodes):
        seed = base_seed + i
        episode_data = collect_episode(env, model, device, seed)
        for agent, transitions in episode_data.items():
            if transitions:
                all_transitions.extend(compute_gae(transitions, gamma=gamma, lam=lam))
        episode_summaries.append({
            "seed": seed,
            "turns": env.engine.state.turn_number,
            "winner": env.engine.state.winner,
            "truncated": env.engine.state.phase.name != "GAME_OVER",
        })
    return all_transitions, episode_summaries


def ppo_update(model: ActorCritic, optimizer: torch.optim.Optimizer, transitions: list[dict],
               clip_ratio: float = 0.2, value_coef: float = 0.5, entropy_coef: float = 0.01,
               epochs: int = 4, minibatch_size: int = 256, max_grad_norm: float = 0.5,
               device: str = "cpu", target_kl: float | None = 0.02) -> dict:
    obs = torch.tensor(np.array([t["obs"] for t in transitions]), dtype=torch.float32, device=device)
    masks = torch.tensor(np.array([t["mask"] for t in transitions]), dtype=torch.float32, device=device)
    actions = torch.tensor([t["action"] for t in transitions], dtype=torch.long, device=device)
    old_logprobs = torch.tensor([t["logprob"] for t in transitions], dtype=torch.float32, device=device)
    advantages = torch.tensor([t["advantage"] for t in transitions], dtype=torch.float32, device=device)
    returns = torch.tensor([t["return"] for t in transitions], dtype=torch.float32, device=device)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    n = obs.shape[0]
    idx = np.arange(n)
    stats = {"policy_loss": [], "value_loss": [], "entropy": [], "approx_kl": [], "clip_frac": []}

    for _ in range(epochs):
        np.random.shuffle(idx)
        epoch_kls = []
        for start in range(0, n, minibatch_size):
            mb = idx[start:start + minibatch_size]
            logprob, entropy, value = model.evaluate_actions(obs[mb], masks[mb], actions[mb])
            ratio = torch.exp(logprob - old_logprobs[mb])
            surr1 = ratio * advantages[mb]
            surr2 = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * advantages[mb]
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = ((value - returns[mb]) ** 2).mean()
            entropy_loss = -entropy.mean()
            loss = policy_loss + value_coef * value_loss + entropy_coef * entropy_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                approx_kl = (old_logprobs[mb] - logprob).mean().item()
                clip_frac = ((ratio - 1.0).abs() > clip_ratio).float().mean().item()
            stats["policy_loss"].append(policy_loss.item())
            stats["value_loss"].append(value_loss.item())
            stats["entropy"].append(entropy.mean().item())
            stats["approx_kl"].append(approx_kl)
            stats["clip_frac"].append(clip_frac)
            epoch_kls.append(approx_kl)

        # Stop before a further epoch would push the policy too far from the
        # one that collected this rollout, at which point the importance-
        # sampling ratios (and thus the clipped surrogate) stop being a
        # trustworthy estimate. abs() because this estimator (old - new
        # logprob) can be negative for a policy that's merely reinforcing its
        # existing behavior, which isn't the runaway-divergence case this
        # guards against.
        if target_kl is not None and abs(float(np.mean(epoch_kls))) > target_kl:
            break

    return {k: float(np.mean(v)) for k, v in stats.items()}
