"""Random legal-action agent. Used for engine smoke-testing and as the
opponent-pool floor in later evaluation (§12 of the design doc).

Uniform-random over the raw legal-action list isn't a useful smoke test on
its own: PROPOSE_TRADE/COUNTER_TRADE alone make up most of the enumerated
MAIN-phase action list (§3 flags this exact flat-enumeration blowup), so a
truly uniform sampler spends nearly all of its steps haggling instead of
building and rarely reaches a finished game. Downweighting those two action
types keeps the agent "random" for every other decision while still
exercising the trade sub-protocol occasionally.
"""
from __future__ import annotations

import random

from env.actions import Action, ActionType
from env.engine import legal_actions
from env.state import GameState

_LOW_WEIGHT_TYPES = {ActionType.PROPOSE_TRADE, ActionType.COUNTER_TRADE}
_LOW_WEIGHT = 0.05


def choose(state: GameState, rng: random.Random, legal: list[Action] | None = None) -> Action:
    """`legal` lets a caller that already enumerated this step's legal
    actions (e.g. the rollout loop, via the env's cache) pass them in instead
    of paying for a second enumeration."""
    actions = legal if legal is not None else legal_actions(state)
    if not actions:
        raise RuntimeError(f"No legal actions in phase {state.phase}")
    weights = [_LOW_WEIGHT if a.type in _LOW_WEIGHT_TYPES else 1.0 for a in actions]
    return rng.choices(actions, weights=weights, k=1)[0]


class RandomAgent:
    """Same `.choose(state)` interface as `HeuristicAgent`, for tournament
    scripts that mix agent types by seat."""

    def __init__(self, player_id: int, rng: random.Random | None = None):
        self.player_id = player_id
        self.rng = rng or random.Random()

    def choose(self, state: GameState, legal: list[Action] | None = None) -> Action:
        return choose(state, self.rng, legal)
