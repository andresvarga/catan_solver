import random

from training.league import League, binomial_test_pvalue


def test_binomial_test_pvalue_matches_known_values():
    # a fair coin landing heads 50/100 should have p-value ~0.5+ (not significant)
    assert binomial_test_pvalue(50, 100, 0.5) > 0.4
    # 90/100 heads should be extremely significant evidence of bias
    assert binomial_test_pvalue(90, 100, 0.5) < 1e-6
    # exactly n successes out of n is the smallest possible p-value for that n
    assert binomial_test_pvalue(100, 100, 0.5) < binomial_test_pvalue(60, 100, 0.5)


def test_promotion_test_requires_both_win_rate_and_significance():
    """Promotion matches are 1 candidate vs 3 copies of the main: parity is
    25%, so significance is tested against 0.25 (audit F-17)."""
    league = League("/tmp/catan_league_test_promo")
    assert league.promotion_test(wins=28, games=100) is False   # below the 30% threshold
    assert league.promotion_test(wins=4, games=10) is False     # 40%, but n=10 isn't significant vs 25%
    assert league.promotion_test(wins=45, games=120) is True    # 37.5%, clearly above parity
    assert league.promotion_test(wins=31, games=100) is False   # clears threshold, not significant
    # the old p0=0.5 test would have rejected all of these
    assert league.promotion_test(wins=45, games=120, parity=0.5) is False


def test_add_member_and_main_pointer():
    league = League("/tmp/catan_league_test_members")
    league.add_member("m0", "main", checkpoint_path="a.pt")
    assert league.main_name == "m0"
    assert league.main().name == "m0"
    league.add_member("hist1", "historical", checkpoint_path="b.pt")
    assert league.main_name == "m0"  # adding a non-main member doesn't move the pointer


def test_promote_demotes_previous_main():
    league = League("/tmp/catan_league_test_promote")
    league.add_member("m0", "main", checkpoint_path="a.pt")
    league.add_member("m1", "historical", checkpoint_path="b.pt")
    league.promote("m1", iteration=5)
    assert league.main_name == "m1"
    assert league.members["m1"].role == "main"
    assert league.members["m0"].role == "historical"


def test_sample_opponent_excludes_current_main_and_random_by_default():
    league = League("/tmp/catan_league_test_sample")
    league.add_member("m0", "main", checkpoint_path="a.pt")
    league.add_member("random", "random")
    league.add_member("heuristic", "heuristic")
    rng = random.Random(0)
    for _ in range(30):
        opp = league.sample_opponent(rng)
        assert opp.name != "m0"
        assert opp.role != "random"


def test_sample_opponent_falls_back_to_uniform_without_a_main():
    league = League("/tmp/catan_league_test_sample_nomain")
    league.add_member("heuristic", "heuristic")
    league.add_member("hist1", "historical", checkpoint_path="a.pt")
    rng = random.Random(0)
    seen = set()
    for _ in range(30):
        seen.add(league.sample_opponent(rng).name)
    assert seen == {"heuristic", "hist1"}


def test_update_ratings_rewards_winner_and_penalizes_loser():
    league = League("/tmp/catan_league_test_ratings")
    league.add_member("a", "main", checkpoint_path="a.pt")
    league.add_member("b", "historical", checkpoint_path="b.pt")
    before_a, before_b = league.rating("a").mu, league.rating("b").mu
    league.update_ratings(["a", "b"])  # a wins
    assert league.rating("a").mu > before_a
    assert league.rating("b").mu < before_b
    assert league.members["a"].games_played == 1
    assert league.members["b"].games_played == 1


def test_update_ratings_ignores_unknown_names_but_does_not_crash():
    league = League("/tmp/catan_league_test_ratings_unknown")
    league.add_member("a", "main", checkpoint_path="a.pt")
    league.update_ratings(["a", "not_a_member"])  # should just no-op (fewer than 2 known)


def test_save_and_load_round_trip():
    league = League("/tmp/catan_league_test_persist")
    league.add_member("m0", "main", checkpoint_path="a.pt", iteration=3)
    league.add_member("heuristic", "heuristic")
    league.update_ratings(["m0", "heuristic"])
    league.save()

    loaded = League.load("/tmp/catan_league_test_persist")
    assert loaded.main_name == "m0"
    assert set(loaded.members.keys()) == {"m0", "heuristic"}
    assert loaded.members["m0"].games_played == 1
    assert abs(loaded.rating("m0").mu - league.rating("m0").mu) < 1e-9
