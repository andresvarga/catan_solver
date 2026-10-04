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


def split_by_game(n_rows: int, game_ids: np.ndarray | None, holdout: float | int,
                  seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """(holdout_idx, rest_idx) with whole games held out (audit F-21).

    Decisions within one game are strongly correlated, so a random row split
    leaks near-duplicates of training rows into validation and makes
    validation NLL/accuracy optimistic. `holdout` is a fraction (< 1) or a
    row count. Rows with game_id < 0 (unknown, e.g. older datasets) are split
    row-wise as before."""
    rng = np.random.RandomState(seed)
    target = int(round(n_rows * holdout)) if holdout < 1 else int(holdout)
    if game_ids is None:
        game_ids = np.full(n_rows, -1, dtype=np.int64)
    game_ids = np.asarray(game_ids)
    known = np.flatnonzero(game_ids >= 0)
    unknown = np.flatnonzero(game_ids < 0)
    hold: list[np.ndarray] = []
    if len(known):
        games = np.unique(game_ids[known])
        rng.shuffle(games)
        frac_known = len(known) / n_rows
        want_known = int(round(target * frac_known))
        taken = 0
        chosen = []
        for g in games:
            if taken >= want_known:
                break
            chosen.append(g)
            taken += int((game_ids == g).sum())
        hold.append(np.flatnonzero(np.isin(game_ids, chosen)))
        target -= taken
    if len(unknown) and target > 0:
        hold.append(rng.permutation(unknown)[:target])
    hold_idx = np.sort(np.concatenate(hold)) if hold else np.zeros(0, dtype=np.int64)
    rest_idx = np.setdiff1d(np.arange(n_rows), hold_idx)
    return hold_idx, rng.permutation(rest_idx)
