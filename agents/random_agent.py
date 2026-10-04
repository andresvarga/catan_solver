"""Random legal-action agent. Used for engine smoke-testing and as the
opponent-pool floor in later evaluation (§12 of the design doc).

Trades are structured: the legal list holds one PROPOSE_TRADE / COUNTER_TRADE
*template* each, which this agent expands into a random legal bundle
(`engine.random_trade`). Templates are downweighted so a random game spends
most of its steps building rather than haggling, while still exercising the
trade sub-protocol regularly (proposals are capped per turn by the engine).
"""
from __future__ import annotations

import random

from env.actions import Action, ActionType
from env.engine import acting_player, is_template, legal_actions, random_trade
from env.state import GameState

_LOW_WEIGHT_TYPES = {ActionType.PROPOSE_TRADE, ActionType.COUNTER_TRADE}
_LOW_WEIGHT = 0.25


def choose(state: GameState, rng: random.Random, legal: list[Action] | None = None) -> Action:
    """`legal` lets a caller that already enumerated this step's legal
    actions (e.g. the rollout loop, via the env's cache) pass them in instead
    of paying for a second enumeration."""
    actions = legal if legal is not None else legal_actions(state)
    if not actions:
        raise RuntimeError(f"No legal actions in phase {state.phase}")
    weights = [_LOW_WEIGHT if a.type in _LOW_WEIGHT_TYPES else 1.0 for a in actions]
    chosen = rng.choices(actions, weights=weights, k=1)[0]
    if is_template(chosen):  # structured trade: pick a random concrete bundle
        chosen = random_trade(chosen, rng, actor=acting_player(state))
    return chosen


class RandomAgent:
    """Same `.choose(state)` interface as `HeuristicAgent`, for tournament
    scripts that mix agent types by seat."""

    def __init__(self, player_id: int, rng: random.Random | None = None):
        self.player_id = player_id
        self.rng = rng or random.Random()

    def choose(self, state: GameState, legal: list[Action] | None = None) -> Action:
        return choose(state, self.rng, legal)
