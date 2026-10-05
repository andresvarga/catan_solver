"""Simulation behaviour must be identical to the pre-optimization golden fingerprints
(audit/results/perf/golden.json, recorded by `python -m audit.perf.golden --record` before
any performance change). Every performance optimization must keep this green."""
import json

import pytest

from audit.perf.golden import (
    ENGINE_SEEDS, ENV_SEEDS, GOLDEN_PATH, ROLLOUTS, compare_rollout, engine_trace, env_trace,
    rollout_trace,
)

GOLD = json.load(open(GOLDEN_PATH))


def _first_divergence(gold, got):
    for i, (a, b) in enumerate(zip(gold["checkpoints"], got["checkpoints"])):
        if a != b:
            return f"first divergence before step {(i + 1) * 50}"
    return "diverges after last checkpoint"


@pytest.mark.parametrize("i", range(len(ENGINE_SEEDS)))
def test_engine_games_identical(i):
    gold, got = GOLD["engine"][i], engine_trace(ENGINE_SEEDS[i])
    assert (got["steps"], got["winner"]) == (gold["steps"], gold["winner"])
    assert got["digest"] == gold["digest"], _first_divergence(gold, got)


@pytest.mark.parametrize("i", range(len(ENV_SEEDS)))
def test_env_episodes_identical(i):
    gold, got = GOLD["env"][i], env_trace(*ENV_SEEDS[i])
    assert got["steps"] == gold["steps"]
    assert got["digest"] == gold["digest"], _first_divergence(gold, got)


@pytest.mark.parametrize("name", list(ROLLOUTS))
@pytest.mark.parametrize("workers, envs", [(1, 1), (2, 3)])
def test_rollouts_identical(name, workers, envs):
    gold = next(r for r in GOLD["rollouts"] if r["name"] == name)
    assert compare_rollout(gold, rollout_trace(name, workers, envs)) == []


@pytest.mark.skipif(not __import__("torch").cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("workers, envs", [(1, 4), (2, 2)])
def test_gpu_rollouts_match_golden(workers, envs):
    """GPU inference (in-process and persistent spawned workers) reproduces the CPU
    golden rollout: same actions/observations/rewards, numerics within 1e-4."""
    from training.hier_ppo import close_rollout_pools
    gold = next(r for r in GOLD["rollouts"] if r["name"] == "gnn_selfplay_phf")
    try:
        assert compare_rollout(gold, rollout_trace("gnn_selfplay_phf", workers, envs, device="cuda"),
                               tol=1e-4) == []
    finally:
        close_rollout_pools()
