"""Wraps a trained ActorCritic checkpoint behind the same `.choose(state)`
interface as `HeuristicAgent`/`RandomAgent`, so scripts/evaluate.py can seat
a trained policy in a tournament without any special-casing."""
from __future__ import annotations

import random

import torch

from env.engine import acting_player, legal_actions
from env.pettingzoo_env import build_observation
from env.state import GameState
from training.model import ActorCritic, flatten_observation, observation_dim


class LearnedAgent:
    def __init__(self, player_id: int, rng: random.Random | None = None,
                 model: ActorCritic | None = None, checkpoint_path: str | None = None,
                 deterministic: bool = True):
        self.player_id = player_id
        self.rng = rng or random.Random()
        self.deterministic = deterministic
        if model is not None:
            self.model = model
        elif checkpoint_path is not None:
            self.model = load_model(checkpoint_path)
        else:
            raise ValueError("LearnedAgent needs either model= or checkpoint_path=")
        self.model.eval()

    def choose(self, state: GameState):
        actions = legal_actions(state)
        if not actions:
            raise RuntimeError(f"No legal actions in phase {state.phase}")
        if len(actions) == 1:
            return actions[0]
        actor = acting_player(state)
        obs = build_observation(state, actor, actions, show_mask=True)
        flat = torch.tensor(flatten_observation(obs), dtype=torch.float32).unsqueeze(0)
        mask = torch.tensor(obs["action_mask"], dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            action_t, *_ = self.model.act(flat, mask, deterministic=self.deterministic)
        return actions[int(action_t.item())]


def load_model(checkpoint_path: str, hidden: int = 256) -> ActorCritic:
    obs_dim = observation_dim()
    model = ActorCritic(obs_dim=obs_dim, hidden=hidden)
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict["model"] if "model" in state_dict else state_dict)
    model.eval()
    return model


class HierarchicalLearnedAgent:
    """Same interface as `LearnedAgent`, backed by either pointer-based model:
    `HierarchicalActorCritic` (flat-vector trunk) or `GraphActorCritic`
    (roadmap phase 6's GNN encoder) -- pass `model_kind="gnn"` for the
    latter. `model.act()` already returns a concrete `Action` for both, so
    there's no index-lookup step here the way there is for the flat
    (non-hierarchical) model / the PettingZoo wrapper's `Discrete(400)`
    contract."""

    def __init__(self, player_id: int, rng: random.Random | None = None,
                 model=None, checkpoint_path: str | None = None, deterministic: bool = True,
                 model_kind: str = "hier"):
        self.player_id = player_id
        self.rng = rng or random.Random()
        self.deterministic = deterministic
        self.model_kind = model_kind
        if model is not None:
            self.model = model
        elif checkpoint_path is not None:
            loader = load_gnn_model if model_kind == "gnn" else load_hier_model
            self.model = loader(checkpoint_path)
        else:
            raise ValueError("HierarchicalLearnedAgent needs either model= or checkpoint_path=")
        self.model.eval()

    def choose(self, state: GameState):
        actions = legal_actions(state)
        if not actions:
            raise RuntimeError(f"No legal actions in phase {state.phase}")
        if len(actions) == 1:
            return actions[0]
        actor = acting_player(state)
        if self.model_kind == "gnn":
            from training.graph_features import build_graph_observation
            encoded = build_graph_observation(state, actor)
            obs_t = {k: torch.tensor(v, dtype=torch.float32).unsqueeze(0) for k, v in encoded.items()}
        else:
            obs = build_observation(state, actor, actions, show_mask=True)
            obs_t = torch.tensor(flatten_observation(obs), dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            action, *_ = self.model.act(obs_t, actions, deterministic=self.deterministic)
        return action


def load_hier_model(checkpoint_path: str, hidden: int = 256):
    from training.hier_model import HierarchicalActorCritic
    obs_dim = observation_dim()
    model = HierarchicalActorCritic(obs_dim=obs_dim, hidden=hidden)
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict["model"] if "model" in state_dict else state_dict)
    model.eval()
    return model


def load_gnn_model(checkpoint_path: str, hidden: int = 128, gnn_layers: int = 3):
    from training.gnn_model import GraphActorCritic
    model = GraphActorCritic(hidden=hidden, gnn_layers=gnn_layers)
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict["model"] if "model" in state_dict else state_dict)
    model.eval()
    return model
