"""A seat filled from a mixed pool of opponent styles (roadmap Phase 2).

Training against three copies of one fixed bot optimises "beat that bot",
which a policy can do by exploiting its quirks. `PooledOpponent` instead
draws, at the start of every episode, which style occupies its seat -- the
plain heuristic, the hidden-information-free heuristic, the depth-2 search
agent, a random agent, or a frozen past checkpoint -- so the trainee meets a
mix of play styles and table compositions.

The draw is a pure function of the episode seed: `hier_ppo.collect_episode`
calls `reset_episode(seed)` on any opponent that has it, so rollouts stay
reproducible (audit F-22).
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable

from agents.heuristic import HeuristicAgent, HonestHeuristicAgent
from agents.random_agent import RandomAgent

# name -> sampling weight. Weighted toward the stronger, more realistic styles;
# random keeps a little exposure to chaotic tables.
DEFAULT_POOL_WEIGHTS = {"heuristic": 0.35, "honest": 0.2, "search": 0.35, "random": 0.1}


@dataclass
class PoolMember:
    name: str
    weight: float
    factory: Callable[[int, random.Random], object]  # (player_id, rng) -> agent


# Factories are module-level / picklable so a pool can be shipped to spawned
# (GPU-inference) rollout workers, not only forked ones.
def _make_heuristic(pid, rng):
    return HeuristicAgent(pid, rng)


def _make_honest(pid, rng):
    return HonestHeuristicAgent(pid, rng)


def _make_search(pid, rng):
    from agents.search_heuristic import SearchHeuristicAgent
    return SearchHeuristicAgent(pid, rng)


def _make_random(pid, rng):
    return RandomAgent(pid, rng)


_BUILTIN_FACTORIES = {"heuristic": _make_heuristic, "honest": _make_honest,
                      "search": _make_search, "random": _make_random}


def builtin_members(weights: dict[str, float] | None = None) -> list[PoolMember]:
    weights = DEFAULT_POOL_WEIGHTS if weights is None else weights
    return [PoolMember(n, w, _BUILTIN_FACTORIES[n]) for n, w in weights.items() if w > 0]


class _CheckpointFactory:
    """Picklable factory for a frozen-checkpoint opponent. The model is loaded
    lazily, once per process, and shared by every seat that draws it."""
    _cache: dict = {}

    def __init__(self, path, model_kind, hidden, gnn_layers, public_hand_features):
        self.args = (path, model_kind, hidden, gnn_layers, public_hand_features)

    def model(self):
        if self.args not in _CheckpointFactory._cache:
            from training.agent import load_gnn_model, load_hier_model
            path, kind, hidden, layers, phf = self.args
            _CheckpointFactory._cache[self.args] = (
                load_gnn_model(path, hidden=hidden, gnn_layers=layers, public_hand_features=phf)
                if kind == "gnn" else load_hier_model(path, hidden=hidden, public_hand_features=phf))
        return _CheckpointFactory._cache[self.args]

    def __call__(self, pid, rng):
        from training.agent import HierarchicalLearnedAgent
        _, kind, _, _, phf = self.args
        return HierarchicalLearnedAgent(pid, rng, model=self.model(), deterministic=True,
                                        model_kind=kind, public_hand_features=phf)


def checkpoint_member(path: str, model_kind: str, hidden: int, gnn_layers: int,
                      weight: float, public_hand_features: bool = False) -> PoolMember:
    """A frozen past checkpoint as a pool member (deterministic play)."""
    factory = _CheckpointFactory(path, model_kind, hidden, gnn_layers, public_hand_features)
    factory.model()  # fail fast on a bad path / architecture mismatch
    return PoolMember(f"ckpt:{path}", weight, factory)


class PooledOpponent:
    """Same `.choose(state, legal)` interface as the other agents; the
    concrete agent behind it is re-drawn by `reset_episode`."""

    def __init__(self, player_id: int, members: list[PoolMember], rng: random.Random | None = None):
        if not members:
            raise ValueError("opponent pool is empty")
        self.player_id = player_id
        self.members = members
        self.rng = rng or random.Random()
        self.current: PoolMember | None = None
        self._agent = None
        self.reset_episode(self.rng.randrange(2 ** 31))

    def reset_episode(self, seed: int) -> None:
        r = random.Random(seed)
        self.current = r.choices(self.members, weights=[m.weight for m in self.members], k=1)[0]
        self._agent = self.current.factory(self.player_id, random.Random(r.randrange(2 ** 31)))

    def choose(self, state, legal=None):
        return self._agent.choose(state, legal)
