"""Shared dataset loading / minibatch construction for BC-style training
(training/bc_pretrain.py, training/dagger.py) across both model encoders.

Demonstration .npz files store the observation either as a single "obs"
array (flat/hier encoder) or as "obs_<field>" arrays, one per
training.graph_features.build_graph_observation field (gnn encoder: hex,
vertex, edge, player, opponent, context, self_id). The label fields
(type_mask, type_idx, head*_id, sub_mask_*, sub_idx_*) are identical either
way -- they come from the legal-action set and the demonstrated action, not
from how the observation was encoded.
"""
from __future__ import annotations

import numpy as np
import torch

LABEL_KEYS = ("type_mask", "type_idx", "head1_id", "sub_mask_1", "sub_idx_1",
              "head2_id", "sub_mask_2", "sub_idx_2")
OPTIONAL_LABEL_KEYS = ("trade_counts", "trade_masks")
OBS_PREFIX = "obs_"


def is_graph_dataset(data) -> bool:
    keys = data.files if hasattr(data, "files") else data.keys()
    return any(k.startswith(OBS_PREFIX) for k in keys)


def load_dataset(path: str, device: str) -> dict[str, torch.Tensor]:
    npz = np.load(path)
    return {k: torch.as_tensor(npz[k], device=device) for k in npz.files}


def batch_obs(data: dict[str, torch.Tensor], idx: torch.Tensor, device):
    """Either a Tensor (hier) or dict[str, Tensor] (gnn), moved to `device`."""
    if is_graph_dataset(data):
        return {k[len(OBS_PREFIX):]: v[idx].to(device) for k, v in data.items()
                if k.startswith(OBS_PREFIX)}
    return data["obs"][idx].to(device)


def batch_labels(data: dict[str, torch.Tensor], idx: torch.Tensor, device) -> dict[str, torch.Tensor]:
    """Label tensors for a minibatch. The structured-trade labels
    (trade_counts/trade_masks) are optional so datasets recorded before
    structured trades still load (they then carry no trade-bundle term)."""
    return {k: data[k][idx].to(device) for k in LABEL_KEYS + OPTIONAL_LABEL_KEYS if k in data}


def model_features(model, obs_batch):
    """The trunk output before the type/value heads, for either encoder."""
    if hasattr(model, "encode"):  # GraphActorCritic
        return model.encode(obs_batch)[0]
    return model.features(obs_batch)  # HierarchicalActorCritic
