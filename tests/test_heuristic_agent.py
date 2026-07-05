from scripts.evaluate import run_tournament, summarize


def test_heuristic_beats_random_decisively():
    seat_kinds = ["heuristic", "random", "random", "random"]
    records = run_tournament(seat_kinds, games=40, base_seed=1000)
    summary = summarize(records, seat_kinds)
    assert summary["timeouts"] == 0
    assert summary["win_rate"]["heuristic"] >= 0.8
    assert summary["avg_vp"]["heuristic"] > summary["avg_vp"]["random"]


def test_heuristic_vs_heuristic_is_roughly_balanced_across_seats():
    """No systematic seat-order bug should let one heuristic-controlled seat
    dominate the other three heuristic-controlled seats."""
    seat_kinds = ["heuristic", "heuristic", "heuristic", "heuristic"]
    records = run_tournament(seat_kinds, games=60, base_seed=2000)
    wins = [0, 0, 0, 0]
    for r in records:
        if r["winner"] is not None:
            wins[r["winner"]] += 1
    assert sum(wins) >= 55  # allow a couple of step-cap timeouts
    assert max(wins) / sum(wins) < 0.55


def test_heuristic_never_crashes_or_times_out_at_scale():
    seat_kinds = ["heuristic", "heuristic", "heuristic", "heuristic"]
    records = run_tournament(seat_kinds, games=50, base_seed=3000)
    assert all(not r["timed_out"] for r in records)
