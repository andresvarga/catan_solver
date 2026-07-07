"""Adapters so the same rollout-collection/PPO code in `training/hier_ppo.py`
works for both `HierarchicalActorCritic` (flat-vector trunk) and
`GraphActorCritic` (roadmap phase 6's GNN encoder). The two models differ
only in what an "encoded observation" looks like (one flat array vs. a dict
of per-node/player/opponent/context arrays) and how a batch of them gets
stacked into tensors -- everything else in the rollout/PPO loop (masking,
GAE, the clipped surrogate) is identical and stays in `hier_ppo.py` unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import torch

from training.graph_features import build_graph_observation
from training.model import flatten_observation


@dataclass
class ModelAdapter:
    encode: Callable[[Any, dict, int], Any]
    to_single: Callable[[Any, str], Any]
    to_batch: Callable[[list, str], Any]
    # Whether `encode` actually reads the `obs` dict PettingZoo's own
    # `last()` builds, or (like the graph adapter) derives everything from
    # `env.engine.state` instead and ignores it entirely. Lets the rollout
    # loop call `env.last(observe=False)` to skip that build for adapters
    # that never use it -- profiling showed it's ~15% of GNN rollout time
    # for a value nothing downstream ever reads.
    needs_raw_obs: bool = True


def _flat_encode(env, obs, pid):
    return flatten_observation(obs)


def _flat_to_single(encoded, device):
    return torch.tensor(encoded, dtype=torch.float32, device=device).unsqueeze(0)


def _flat_to_batch(encoded_list, device):
    return torch.tensor(np.array(encoded_list), dtype=torch.float32, device=device)


FLAT_ADAPTER = ModelAdapter(encode=_flat_encode, to_single=_flat_to_single, to_batch=_flat_to_batch)


def _graph_encode(env, obs, pid):
    return build_graph_observation(env.engine.state, pid,
                                    public_hand_features=env.public_hand_features)


def _graph_to_single(encoded, device):
    return {k: torch.tensor(v, dtype=torch.float32, device=device).unsqueeze(0) for k, v in encoded.items()}


def _graph_to_batch(encoded_list, device):
    keys = encoded_list[0].keys()
    return {k: torch.tensor(np.array([e[k] for e in encoded_list]), dtype=torch.float32, device=device)
            for k in keys}


GRAPH_ADAPTER = ModelAdapter(encode=_graph_encode, to_single=_graph_to_single, to_batch=_graph_to_batch,
                              needs_raw_obs=False)

ADAPTERS = {"hier": FLAT_ADAPTER, "gnn": GRAPH_ADAPTER}
