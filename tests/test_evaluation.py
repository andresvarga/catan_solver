"""Evaluation protocol (roadmap Phase 3): seed registry, seat-balanced
matchups, paired comparison, game-level BC holdout split."""
import numpy as np
import pytest

from evaluation.seeds import check_training_seeds, eval_ranges, load_registry, log_usage, seed_set, usage_count
from evaluation.tournament import paired_compare, run_matchup, summarize, wilson
from training.imitation_data import split_by_game


def test_registry_sets_are_disjoint_and_clear_of_reserved_ranges():
    reg = load_registry()
    ranges = eval_ranges(reg)
    for i, (n1, a1, b1) in enumerate(ranges):
        for n2, a2, b2 in ranges[i + 1:]:
            assert b1 < a2 or b2 < a1, f"{n1} overlaps {n2}"
        for r in reg["reserved"]:
            assert b1 < r["start"] or r["end"] < a1, f"{n1} overlaps reserved {r['use']}"


def test_seed_set_interleaves_bases_and_respects_count():
    s = seed_set("inloop", 3)
    assert s == [40_000_000, 40_100_000, 40_000_001, 40_100_001, 40_000_002, 40_100_002]
    assert len(seed_set("selection")) == load_registry()["sets"]["selection"]["count"]
    with pytest.raises(KeyError):
        seed_set("nope")


def test_training_seed_overlap_is_refused():
    check_training_seeds(0, 100_000)  # typical training range: fine
    with pytest.raises(ValueError):
        check_training_seeds(49_999_000, 50_000_010, "test")


def test_usage_log_counts_prior_uses(tmp_path):
    path = str(tmp_path / "usage.jsonl")
    assert log_usage("confirm_1", {"candidate": "a"}, path) == 0
    assert log_usage("confirm_1", {"candidate": "b"}, path) == 1
    assert usage_count("confirm_1", path) == 2 and usage_count("confirm_2", path) == 0


def test_matchup_is_seat_balanced_and_worker_count_invariant():
    seeds = [70_000_000 + i for i in range(3)]
    r1 = run_matchup("heuristic", "honest", seeds, workers=1)
    r3 = run_matchup("heuristic", "honest", seeds, workers=3)
    key = lambda rs: sorted((r["seed"], r["seat"], r["won"], r["vp"], r["steps"]) for r in rs)
    assert key(r1) == key(r3)
    summ = summarize(r1)
    assert summ["games"] == 12 and all(summ["per_seat"][s]["n"] == 3 for s in range(4))
    lo, hi = summ["ci95"]
    assert lo <= summ["win_rate"] <= hi


def test_paired_compare_statistics():
    recs = lambda wins: [{"seed": i // 4, "seat": i % 4, "won": w} for i, w in enumerate(wins)]
    a = recs([True] * 30 + [False] * 70)
    same = paired_compare(a, a)
    assert same["diff"] == 0 and same["mcnemar_p"] == 1.0
    b = recs([False] * 30 + [False] * 70)  # A wins 30 games B lost, never the reverse
    cmp = paired_compare(a, b)
    assert cmp["a_only_wins"] == 30 and cmp["b_only_wins"] == 0
    assert cmp["diff"] == pytest.approx(0.30) and cmp["mcnemar_p"] < 1e-6
    assert cmp["diff_ci95"][0] > 0
    assert wilson(0, 10)[0] == pytest.approx(0.0, abs=1e-12) and wilson(10, 10)[1] == pytest.approx(1.0)


def test_split_by_game_never_straddles_games():
    game_ids = np.repeat(np.arange(50), 20)  # 50 games x 20 rows
    hold, rest = split_by_game(len(game_ids), game_ids, 0.1, seed=0)
    assert set(game_ids[hold]).isdisjoint(set(game_ids[rest]))
    assert len(hold) == 100 and len(hold) + len(rest) == 1000
    # unknown game ids fall back to a row split
    hold2, rest2 = split_by_game(100, None, 0.2, seed=0)
    assert len(hold2) == 20 and len(rest2) == 80


def test_evaluate_candidate_cli_smoke(tmp_path):
    from scripts.evaluate_candidate import main
    out = main(["--candidate", "heuristic", "--opponents", "random", "--seed-set", "selection",
                "--seeds", "2", "--baseline", "honest", "--workers", "2",
                "--out", str(tmp_path / "r.json")])
    res = out["results"]["random"]
    assert res["candidate"]["games"] == 8 and "paired" in res
