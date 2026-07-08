"""Tests for the GNN encoder path through the imitation pipeline (demo
collection, BC pretraining, DAgger) -- previously flat-vector ('hier') only."""
import numpy as np
import torch

from scripts.collect_dagger_demos import collect_dagger, play_and_record_dagger
from scripts.collect_heuristic_demos import play_and_record, records_to_arrays
from training.dagger import concat_arrays
from training.gnn_model import GraphActorCritic
from training.imitation_data import batch_labels, batch_obs, is_graph_dataset, model_features
from training.train_hier import build_model

GRAPH_FIELDS = {"hex", "vertex", "edge", "player", "opponent", "context", "self_id"}


def _gnn_model():
    torch.manual_seed(0)
    m = GraphActorCritic(hidden=32, gnn_layers=2)
    m.eval()
    return m


def test_heuristic_demos_record_graph_observations():
    records = play_and_record(seed=6, public_hand_features=False, model_type="gnn")
    assert len(records) > 20
    assert set(records[0]["obs"].keys()) == GRAPH_FIELDS
    arrays = records_to_arrays(records, model_type="gnn")
    assert is_graph_dataset(arrays)
    assert "obs" not in arrays
    for f in GRAPH_FIELDS:
        assert f"obs_{f}" in arrays
    assert arrays["obs_hex"].shape[0] == len(records)


def test_gnn_demos_are_trainable_via_evaluate_actions():
    records = play_and_record(seed=6, public_hand_features=False, model_type="gnn")
    arrays = records_to_arrays(records, model_type="gnn")
    data = {k: torch.as_tensor(v) for k, v in arrays.items()}
    idx = torch.arange(len(records))
    model = _gnn_model()
    obs_batch = batch_obs(data, idx, "cpu")
    assert set(obs_batch.keys()) == GRAPH_FIELDS
    labels = batch_labels(data, idx, "cpu")
    logprob, entropy, value = model.evaluate_actions(obs_batch, labels)
    loss = -logprob.mean()
    loss.backward()
    assert torch.isfinite(loss)
    feats = model_features(model, obs_batch)
    assert feats.shape[0] == len(records)


def test_dagger_gnn_collection_and_aggregation():
    model = _gnn_model()
    records = play_and_record_dagger(seed=15, public_hand_features=False, model=model,
                                      model_type="gnn")
    assert len(records) > 20
    assert set(records[0]["obs"].keys()) == GRAPH_FIELDS

    arrays = collect_dagger(model, games=3, base_seed=200, public_hand_features=False,
                             num_workers=2, model_type="gnn")
    assert is_graph_dataset(arrays)
    assert arrays["type_idx"].shape[0] > 20

    base = records_to_arrays(play_and_record(seed=6, public_hand_features=False, model_type="gnn"),
                              model_type="gnn")
    agg = concat_arrays([base, arrays])
    assert agg["obs_hex"].shape[0] == base["obs_hex"].shape[0] + arrays["obs_hex"].shape[0]


def test_build_model_gnn_matches_dataset_encoding():
    model = build_model("gnn", hidden=32, gnn_layers=2, public_hand_features=False)
    assert isinstance(model, GraphActorCritic)
