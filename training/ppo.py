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

# Discounting is per *own non-forced decision*. A player makes ~180-230 real
# decisions per game, so gamma=0.99 passed only ~10% of the terminal reward
# back to the opening placements (audit F-15); 0.999 passes ~80%.
GAMMA = 0.999
GAE_LAMBDA = 0.98


def compute_gae(transitions: list[dict], gamma: float = GAMMA, lam: float = GAE_LAMBDA) -> list[dict]:
    """GAE over one agent's own decision sequence.

    Termination (a real game end) is absorbing: no bootstrap. Truncation (step
    cap) is NOT the end of the game: the env pays no reward for it
    (`truncation_reward="zero"`), and the final transition bootstraps from
    `bootstrap_value` = V(s_T), which `hier_ppo.collect_episode` records for
    every trainee seat at truncation. (The old scheme paid a rank-on-standing
    reward instead and treated truncation as terminal -- that paid a VP leader
    the full win reward, an incentive to stall; audit F-06.) A truncated
    transition without `bootstrap_value` bootstraps from 0."""
    if not transitions:
        return transitions
    advantages = [0.0] * len(transitions)
    gae = 0.0
    for t in reversed(range(len(transitions))):
        tr = transitions[t]
        if tr["done"]:
            truncated_only = tr.get("truncated", False) and not tr.get("terminated", False)
            next_value = tr.get("bootstrap_value", 0.0) if truncated_only else 0.0
            delta = tr["reward"] + gamma * next_value - tr["value"]
            gae = delta
        else:
            delta = tr["reward"] + gamma * transitions[t + 1]["value"] - tr["value"]
            gae = delta + gamma * lam * gae
        advantages[t] = gae
    for t, adv in zip(transitions, advantages):
        t["advantage"] = adv
        t["return"] = adv + t["value"]
    return transitions
