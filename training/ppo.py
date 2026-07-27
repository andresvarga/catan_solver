"""Shared GAE utility for the Catan PPO loops.

This module used to hold a full self-play PPO loop for a flat-action-space
model (`collect_episode`/`collect_rollout`/`ppo_update` against
`training/model.ActorCritic`, roadmap phase 4). That model and loop were
superseded by the pointer-based hierarchical/GNN models and their own loop
(`training/hier_ppo.py`) and removed; `compute_gae` survives because both
`training/hier_ppo.py` and `training/rl_finetune.py` still depend on it.

AEC turn order interleaves agents, so credit assignment has to reconstruct
each agent's *own* decision sequence before running GAE -- reward earned
between two of an agent's own decisions (from opponents acting, dice
production, robber steals, etc.) all belongs to the transition that led into
that gap. `training/hier_ppo.py` replicates the same
"zero-cumulative-reward-on-read" trick PettingZoo's own
`TerminateIllegalWrapper` uses, rather than depending on a wrapper, so
`last()` always returns "reward since I was last asked to act."
"""
from __future__ import annotations

GAMMA = 0.99
GAE_LAMBDA = 0.95


def compute_gae(transitions: list[dict], gamma: float = GAMMA, lam: float = GAE_LAMBDA) -> list[dict]:
    """Truncation is treated exactly like termination here (mask on `done`,
    no bootstrap value): the env already pays out rank-based terminal rewards
    on truncation ("rank on current standing"), and that rank reward *is* the
    estimate of the episode's remaining value. The previous scheme paid the
    rank reward AND bootstrapped V(s_final) on top -- two estimates of the
    same future outcome summed into one target, systematically inflating
    value targets exactly in the regime (early training, low finish rate)
    where truncation dominates."""
    if not transitions:
        return transitions
    values = [t["value"] for t in transitions] + [0.0]  # final entry is always masked out
    advantages = [0.0] * len(transitions)
    gae = 0.0
    for t in reversed(range(len(transitions))):
        mask = 0.0 if transitions[t]["done"] else 1.0
        delta = transitions[t]["reward"] + gamma * values[t + 1] * mask - values[t]
        gae = delta + gamma * lam * mask * gae
        advantages[t] = gae
    for t, adv in zip(transitions, advantages):
        t["advantage"] = adv
        t["return"] = adv + t["value"]
    return transitions
