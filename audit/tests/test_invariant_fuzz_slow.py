"""Slow randomized invariant fuzzing (roadmap Phase 1): full games with the
invariant checker after every step -- strict 95-card conservation (19 per
resource, hands + bank), piece/occupancy/distance/connectivity invariants,
dev-card and knight conservation, award thresholds -- plus the independent
Longest Road oracle at game end and the own-turn-win rule.

    python -m pytest -m slow
"""
import multiprocessing as mp

import pytest

from audit.tools.fuzz_invariants import play

pytestmark = pytest.mark.slow


def _run(jobs):
    with mp.get_context("fork").Pool(min(12, mp.cpu_count())) as pool:
        return pool.starmap(play, jobs, chunksize=4)


@pytest.mark.parametrize("agent, games, base", [("random", 1000, 16_000_000), ("heuristic", 200, 16_100_000)])
def test_invariants_hold_across_many_games(agent, games, base):
    results = _run([(base + i, agent, True, True) for i in range(games)])
    violations = {k: v for r in results for k, v in r["violations"].items()}
    assert not violations, f"first violations: {violations}"
    assert not [m for r in results for m in r["lr_mismatch"]]
    assert all(r["done"] for r in results), "every game must finish"
    assert all(r["winner_is_current"] for r in results), "wins only on the winner's own turn"
