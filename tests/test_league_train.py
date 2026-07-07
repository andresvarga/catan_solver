import os
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
         # window=1: every eval is immediately eligible for best-checkpoint
         # comparison, since this test only produces 2 eval checkpoints
         # total (iter 2, iter 4) -- not enough to fill the default window=3.
         # The window-warmup behavior itself is covered by
         # test_best_eval_selection_waits_for_a_full_window below.
         "--best-eval-window", "1",
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
    # win-rate-driven checkpoint selection runs by default (no flag needed)
    assert "new best_eval vs heuristic" in result.stdout
    assert "best_eval: vs heuristic" in result.stdout
    assert os.path.exists(os.path.join(CHECKPOINT_DIR, "best_eval.pt"))


def _max_iter_reached(stdout: str) -> int:
    import re
    nums = [int(m) for m in re.findall(r"^iter\s+(\d+)\s+\|", stdout, re.MULTILINE)]
    return max(nums) if nums else 0


def test_best_eval_selection_waits_for_a_full_window():
    """Regression test for a real bug: with window=3, the very first eval
    (a 1-sample average) must NOT be eligible to become the recorded best --
    an early run let a 2-sample average lock in as "best" before the window
    ever filled, and a rigorous re-evaluation later showed a *worse*
    checkpoint had won purely because early partial-window averages are
    less diluted (so more likely to be an extreme value) than any later
    full-window comparison. Every comparison must use the same sample size."""
    ckpt_dir = "/tmp/catan_league_train_window_warmup_test"
    shutil.rmtree(ckpt_dir, ignore_errors=True)
    result = subprocess.run(
        [sys.executable, "-m", "training.league_train",
         "--iterations", "8", "--episodes-per-iter", "4", "--hidden", "16",
         "--max-episode-steps", "120", "--no-trading", "--no-dev-cards",
         "--self-play-prob", "0.5", "--snapshot-every", "0", "--eval-every", "2",
         "--eval-games", "2", "--promotion-games", "2",
         "--best-eval-window", "3",
         "--checkpoint-dir", ckpt_dir],
        capture_output=True, text=True, timeout=180,
    )
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    # 4 eval checkpoints happen (iter 2,4,6,8); the first 2 must be pure
    # warm-up (window not yet at 3) with no best-eval claim at all.
    assert "warming up selection window 1/3" in result.stdout
    assert "warming up selection window 2/3" in result.stdout
    lines = result.stdout.splitlines()
    warmup2_idx = next(i for i, l in enumerate(lines) if "warming up selection window 2/3" in l)
    first_best_idx = next((i for i, l in enumerate(lines) if "new best_eval" in l), None)
    assert first_best_idx is not None, "expected a best_eval claim once the window fills"
    assert first_best_idx > warmup2_idx, "best_eval must not be claimed before the window is full"
    shutil.rmtree(ckpt_dir, ignore_errors=True)


def test_confirmatory_final_eval_runs_and_recommends_a_candidate():
    """The training-time smoothed win rate is a biased (winner's-curse) signal
    -- observed directly in practice, where a smoothed reading from a genuine
    full eval window still overstated true quality by more than 2x. The
    final confirmatory eval (larger sample, run once at the end rather than
    searched over) is what should actually be trusted."""
    ckpt_dir = "/tmp/catan_league_train_confirmatory_eval_test"
    shutil.rmtree(ckpt_dir, ignore_errors=True)
    result = subprocess.run(
        [sys.executable, "-m", "training.league_train",
         "--iterations", "8", "--episodes-per-iter", "4", "--hidden", "16",
         "--max-episode-steps", "120", "--no-trading", "--no-dev-cards",
         "--self-play-prob", "0.5", "--snapshot-every", "0", "--eval-every", "2",
         "--eval-games", "2", "--promotion-games", "2",
         "--best-eval-window", "3", "--final-eval-games", "4",
         "--checkpoint-dir", ckpt_dir],
        capture_output=True, text=True, timeout=180,
    )
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    assert "confirmatory eval (4 games vs heuristic" in result.stdout
    assert "latest.pt (final)" in result.stdout
    assert "best_eval.pt (iter" in result.stdout
    assert "-> recommended:" in result.stdout
    shutil.rmtree(ckpt_dir, ignore_errors=True)


def test_early_stopping_halts_before_the_requested_iteration_count():
    """--early-stop-patience 1 with an impossibly high --early-stop-min-delta
    guarantees every eval after the warmup window fills counts as 'stale',
    so the run must stop well before completing all requested iterations."""
    ckpt_dir = "/tmp/catan_league_train_early_stop_test"
    shutil.rmtree(ckpt_dir, ignore_errors=True)
    result = subprocess.run(
        [sys.executable, "-m", "training.league_train",
         "--iterations", "40", "--episodes-per-iter", "4", "--hidden", "16",
         "--max-episode-steps", "120", "--no-trading", "--no-dev-cards",
         "--self-play-prob", "0.5", "--snapshot-every", "0", "--eval-every", "2",
         "--eval-games", "2", "--promotion-games", "2",
         "--best-eval-window", "2", "--early-stop-patience", "1", "--early-stop-min-delta", "2.0",
         "--checkpoint-dir", ckpt_dir],
        capture_output=True, text=True, timeout=180,
    )
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    assert "early stopping: no improvement" in result.stdout
    assert _max_iter_reached(result.stdout) < 40, "run should have stopped well short of the ceiling"
    shutil.rmtree(ckpt_dir, ignore_errors=True)
