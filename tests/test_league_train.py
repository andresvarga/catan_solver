import shutil
import subprocess
import sys

from training.league import League
from training.league_train import episode_rating_teams

CHECKPOINT_DIR = "/tmp/catan_league_train_integration_test"


def test_episode_rating_teams_is_a_clean_two_way_comparison():
    # trainee (pid 0) finished 1st, opponent's best seat (pid 1) finished 2nd
    summary = {"trainee_pids": [0], "ranking_pids": [0, 1, 2, 3]}
    assert episode_rating_teams(summary, "main", "opp") == ["main", "opp"]

    # trainee finished last -- opponent's best seat clearly beat it
    summary = {"trainee_pids": [0], "ranking_pids": [1, 2, 3, 0]}
    assert episode_rating_teams(summary, "main", "opp") == ["opp", "main"]

    # trainee is pid 2, finishing 2nd (rank index 1); opponent's best seat is
    # pid 1, finishing 1st (rank index 0) -- opponent's best seat still beats
    # the trainee even though the trainee didn't finish last
    summary = {"trainee_pids": [2], "ranking_pids": [1, 2, 0, 3]}
    assert episode_rating_teams(summary, "main", "opp") == ["opp", "main"]

    # trainee is pid 2, finishing 1st -- beats every opponent seat
    summary = {"trainee_pids": [2], "ranking_pids": [2, 1, 0, 3]}
    assert episode_rating_teams(summary, "main", "opp") == ["main", "opp"]


def test_repeated_multi_seat_games_no_longer_crash_ratings_toward_negative_infinity():
    """Regression test for the duplicate-seat overwrite bug: feeding an
    opponent's 3 identical seats into trueskill.rate() as independent
    competitors, then writing the result back to the same dict key 3 times,
    silently kept only the *worst*-placed seat's update -- a systematic
    pessimistic bias regardless of who actually won. With the fix, a trainee
    that wins every game should see its rating climb, not collapse."""
    league = League("/tmp/catan_league_ratings_regression")
    league.add_member("main", "main", checkpoint_path="a.pt")
    league.add_member("opp", "historical", checkpoint_path="b.pt")

    # trainee (pid 0) wins every game; opponent occupies pids 1,2,3
    summary = {"trainee_pids": [0], "ranking_pids": [0, 1, 2, 3]}
    for _ in range(30):
        league.update_ratings(episode_rating_teams(summary, "main", "opp"))

    assert league.rating("main").mu > 25.0, "a consistent winner's rating should rise, not crash"
    assert league.rating("opp").mu < 25.0, "a consistent loser's rating should fall, but stay bounded"
    assert league.rating("opp").mu > -5.0, "should not collapse toward large negative numbers"


def test_league_train_runs_end_to_end_and_builds_a_league():
    shutil.rmtree(CHECKPOINT_DIR, ignore_errors=True)
    result = subprocess.run(
        [sys.executable, "-m", "training.league_train",
         "--iterations", "4", "--episodes-per-iter", "4", "--hidden", "16",
         "--max-episode-steps", "120", "--no-trading", "--no-dev-cards",
         "--self-play-prob", "0.5", "--snapshot-every", "2", "--eval-every", "2",
         "--eval-games", "2", "--promotion-games", "2",
         "--exploiter-every", "2", "--exploiter-iterations", "1",
         "--checkpoint-dir", CHECKPOINT_DIR],
        capture_output=True, text=True, timeout=180,
    )
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    assert "initialized new league" in result.stdout
    assert "snapshotted historical_iter2" in result.stdout
    assert "eval vs random" in result.stdout
    assert "eval vs heuristic" in result.stdout

    from training.league import League
    league = League.load(CHECKPOINT_DIR)
    assert league.main_name is not None
    role_names = {m.role for m in league.members.values()}
    assert {"random", "heuristic", "historical"}.issubset(role_names)
