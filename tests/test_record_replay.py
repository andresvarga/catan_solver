"""Tests for the replay recorder (scripts/record_replay.py)."""
import json
import os

from env.engine import CatanEngine
from scripts.record_replay import board_geometry, record_game, write_replay
from training.gnn_model import GraphActorCritic
from training.hier_model import HierarchicalActorCritic
from training.model import observation_dim


def test_board_geometry_covers_every_element():
    board = CatanEngine(randomize_board=True, seed=4).state.board
    geo = board_geometry(board)
    assert len(geo["hexes"]) == 19
    assert len(geo["vertices"]) == 54
    assert len(geo["edges"]) == 72
    for h in geo["hexes"]:
        assert len(h["corners"]) == 6
    # every edge endpoint has a position
    for a, b in geo["edges"].values():
        assert a in geo["vertices"] and b in geo["vertices"]


def test_record_game_produces_valid_replay_and_html():
    model = HierarchicalActorCritic(obs_dim=observation_dim(), hidden=32)
    model.eval()
    replay = record_game(model, seat=0, seed=3, opponents="random",
                          public_hand_features=False, max_steps=250,
                          checkpoint_label="test")
    # serializable end to end
    payload = json.dumps(replay)
    assert len(replay["frames"]) >= 2
    f0 = replay["frames"][0]
    assert f0["act"] is None and f0["s"]["turn"] == 0
    # every non-initial frame has an actor, action label, and a state snapshot
    diags = 0
    for f in replay["frames"][1:]:
        assert f["a"] in (0, 1, 2, 3)
        assert isinstance(f["act"], str) and f["act"]
        assert len(f["s"]["p"]) == 4 and len(f["s"]["bank"]) == 5
        assert len(f["s"]["p"][0]) == 7  # 5 resources + knights + vp
        assert len(f["s"]["dev"]) == 4 and len(f["s"]["dev"][0]) == 5
        if f["s"]["tr"] is not None:
            tr = f["s"]["tr"]
            assert len(tr["give"]) == 5 and len(tr["want"]) == 5
            assert tr["pr"] in (0, 1, 2, 3)
        if f["d"] is not None:
            diags += 1
            d = f["d"]
            assert 0.0 <= d["im"] <= 1.0 + 1e-6
            assert 0.0 <= d["cp"] <= 1.0 + 1e-6
            assert d["n"] > 1
            assert d["top"] and all(len(t) == 2 for t in d["top"])
    assert diags > 0, "expected at least one instrumented model decision"

    out = "/tmp/catan_test_replay.html"
    write_replay(replay, out)
    html = open(out).read()
    assert "__REPLAY_JSON__" not in html
    assert '"frames"' in html and "const R =" in html
    assert "<!--ARTIFACT_START-->" in html and "<!--ARTIFACT_END-->" in html
    os.remove(out)


def test_record_game_works_with_the_gnn_encoder():
    """The recorder's diagnostics (trunk forward, stage-1 pointer logits) take
    a different path per encoder (per-head linear layers vs. embedding-dot
    pointer heads) -- regression coverage for that branch, not just hier."""
    model = GraphActorCritic(hidden=32, gnn_layers=2)
    model.eval()
    replay = record_game(model, seat=0, seed=3, opponents="random",
                          public_hand_features=False, max_steps=200,
                          checkpoint_label="test-gnn", model_type="gnn")
    json.dumps(replay)  # still serializable
    diags = [f["d"] for f in replay["frames"] if f["d"] is not None]
    assert diags, "expected at least one instrumented model decision"
    for d in diags:
        assert 0.0 <= d["im"] <= 1.0 + 1e-6
        assert 0.0 <= d["cp"] <= 1.0 + 1e-6
        assert d["top"] and all(len(t) == 2 for t in d["top"])
