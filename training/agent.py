"""Wraps a trained hierarchical/GNN checkpoint behind the same `.choose(state)`
interface as `HeuristicAgent`/`RandomAgent`, so evaluation/replay/training
code can seat a trained policy without any special-casing. (The original
flat-action-space `LearnedAgent`/`load_model`, wrapping the now-removed
`training.model.ActorCritic`, was superseded by `HierarchicalLearnedAgent`
below and removed.)"""
from __future__ import annotations

import random

import torch

from env.engine import acting_player, legal_actions
from env.pettingzoo_env import build_observation
from env.state import GameState
from training.model import flatten_observation, observation_dim


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
                 model_kind: str = "hier", public_hand_features: bool = False):
        self.player_id = player_id
        self.rng = rng or random.Random()
        self.deterministic = deterministic
        self.model_kind = model_kind
        # Must match the flag the model was trained with -- it changes the
        # observation width (a mismatch fails loudly at the first forward).
        self.public_hand_features = public_hand_features
        if model is not None:
            self.model = model
        elif checkpoint_path is not None:
            if model_kind == "gnn":
                self.model = load_gnn_model(checkpoint_path, public_hand_features=public_hand_features)
            else:
                self.model = load_hier_model(checkpoint_path, public_hand_features=public_hand_features)
        else:
            raise ValueError("HierarchicalLearnedAgent needs either model= or checkpoint_path=")
        self.model.eval()

    def choose(self, state: GameState, legal: list | None = None):
        actions = legal if legal is not None else legal_actions(state)
        if not actions:
            raise RuntimeError(f"No legal actions in phase {state.phase}")
        if len(actions) == 1:
            return actions[0]
        actor = acting_player(state)
        if self.model_kind == "gnn":
            from training.graph_features import build_graph_observation
            encoded = build_graph_observation(state, actor,
                                               public_hand_features=self.public_hand_features)
            obs_t = {k: torch.tensor(v, dtype=torch.float32).unsqueeze(0) for k, v in encoded.items()}
        else:
            obs = build_observation(state, actor, actions, show_mask=True,
                                     public_hand_features=self.public_hand_features)
            obs_t = torch.tensor(flatten_observation(obs), dtype=torch.float32).unsqueeze(0)
        with torch.inference_mode():
            action, *_ = self.model.act(obs_t, actions, deterministic=self.deterministic)
        return action


def load_hier_model(checkpoint_path: str, hidden: int = 256, public_hand_features: bool = False):
    from training.hier_model import HierarchicalActorCritic
    obs_dim = observation_dim(public_hand_features=public_hand_features)
    model = HierarchicalActorCritic(obs_dim=obs_dim, hidden=hidden)
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict["model"] if "model" in state_dict else state_dict)
    model.eval()
    return model


def load_gnn_model(checkpoint_path: str, hidden: int = 128, gnn_layers: int = 3,
                   public_hand_features: bool = False):
    from training.gnn_model import GraphActorCritic
    model = GraphActorCritic(hidden=hidden, gnn_layers=gnn_layers,
                              public_hand_features=public_hand_features)
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    # strict=False tolerates checkpoints saved before vertex_aux_head existed
    # (it's a training-only probe, unused by act()/choose()) -- but any other
    # missing/unexpected key means a real architecture mismatch, so surface it.
    result = model.load_state_dict(state_dict["model"] if "model" in state_dict else state_dict,
                                    strict=False)
    unexpected_missing = [k for k in result.missing_keys if not k.startswith("vertex_aux_head")]
    assert not unexpected_missing and not result.unexpected_keys, (
        f"missing={unexpected_missing} unexpected={result.unexpected_keys}")
    model.eval()
    return model
