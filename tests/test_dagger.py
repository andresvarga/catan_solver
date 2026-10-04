"""Tests for DAgger collection (scripts/collect_dagger_demos.py) and the
aggregation pieces of training/dagger.py."""
import numpy as np
import torch

from scripts.collect_dagger_demos import collect_dagger, play_and_record_dagger
from scripts.collect_heuristic_demos import play_and_record, records_to_arrays
from training.dagger import concat_arrays
from training.hier_model import HierarchicalActorCritic
from training.model import observation_dim

DATA_KEYS = {"obs", "type_mask", "type_idx", "head1_id", "sub_mask_1", "sub_idx_1",
             "head2_id", "sub_mask_2", "sub_idx_2", "trade_counts", "trade_masks"}


def _model():
    torch.manual_seed(0)
    m = HierarchicalActorCritic(obs_dim=observation_dim(), hidden=32)
    m.eval()
    return m


def test_dagger_records_share_the_demo_schema_and_are_trainable():
    model = _model()
    records = play_and_record_dagger(seed=5, public_hand_features=False, model=model)
    assert len(records) > 20, "a full game should produce many policy-seat decisions"
    arrays = records_to_arrays(records)
    assert set(arrays) == DATA_KEYS
    assert arrays["obs"].shape[1] == observation_dim()
    assert ((arrays["type_idx"] >= 0) & (arrays["type_idx"] < 19)).all()
    # every recorded target must be reachable under its own mask -- the label
    # is the expert's action, which was drawn from the same legal set
    tm = arrays["type_mask"][np.arange(len(records)), arrays["type_idx"]]
    assert (tm == 1.0).all(), "expert-labeled type must be legal in the recorded mask"
    has_head1 = arrays["head1_id"] >= 0
    m1 = arrays["sub_mask_1"][np.arange(len(records)), np.maximum(arrays["sub_idx_1"], 0)]
    assert (m1[has_head1] == 1.0).all(), "stage-1 label must be legal under its mask"
    # trainable end to end: one BC gradient step on the labels
    data = {k: torch.as_tensor(v) for k, v in arrays.items()}
    logprob, entropy, _ = model.evaluate_actions(data["obs"],
                                                  {k: v for k, v in data.items() if k != "obs"})
    loss = -logprob.mean()
    loss.backward()
    assert torch.isfinite(loss)


def test_dagger_labels_come_from_expert_not_policy():
    """With a random-weights policy driving the rollout, the labeled actions
    must reflect the expert's preferences, not the policy's: heuristic play
    almost never proposes trades at random-policy rates and heavily favors
    builds/end_turn. Sanity-check the label distribution is expert-shaped by
    comparing against labels from a pure heuristic-self-play game."""
    model = _model()
    dagger_recs = []
    for seed in (11, 12, 13):
        dagger_recs.extend(play_and_record_dagger(seed, False, model))
    demo_recs = []
    for seed in (11, 12, 13):
        demo_recs.extend(play_and_record(seed, False))

    def type_hist(recs):
        h = np.zeros(19)
        for r in recs:
            h[r["type_idx"]] += 1
        return h / h.sum()

    hd, hb = type_hist(dagger_recs), type_hist(demo_recs)
    # distributions won't be identical (different visited states) but should
    # be far closer to each other than to uniform-over-legal
    l1 = np.abs(hd - hb).sum()
    assert l1 < 0.8, f"label distribution should be expert-shaped (L1 to demos {l1:.2f})"


def test_collect_dagger_parallel_matches_schema():
    model = _model()
    arrays = collect_dagger(model, games=4, base_seed=100, public_hand_features=False,
                             num_workers=2)
    assert set(arrays) == DATA_KEYS
    assert arrays["type_idx"].shape[0] > 50


def test_concat_arrays_aggregates_rounds():
    model = _model()
    a = records_to_arrays(play_and_record_dagger(21, False, model))
    b = records_to_arrays(play_and_record_dagger(22, False, model))
    agg = concat_arrays([a, b])
    assert agg["obs"].shape[0] == a["obs"].shape[0] + b["obs"].shape[0]
    assert set(agg) == DATA_KEYS
