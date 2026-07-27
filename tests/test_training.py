import numpy as np

from env.pettingzoo_env import CatanAECEnv
from training.model import flatten_observation, observation_dim
from training.ppo import compute_gae


def test_gae_treats_truncation_as_terminal_no_double_count():
    """The env pays rank-on-standing terminal rewards when an episode
    truncates, so GAE must NOT also bootstrap a value estimate past the
    boundary -- that summed two estimates of the same future outcome into
    one target. Truncated and terminated endings must produce identical
    targets given identical rewards/values."""
    def episode(terminated, truncated):
        return [
            {"value": 0.5, "reward": 0.0, "done": False, "terminated": False, "truncated": False},
            {"value": 0.7, "reward": 1.0, "done": True, "terminated": terminated, "truncated": truncated},
        ]

    trunc = compute_gae(episode(False, True), gamma=0.99, lam=0.95)
    term = compute_gae(episode(True, False), gamma=0.99, lam=0.95)

    # Last transition's target is exactly reward - value: nothing beyond the
    # episode boundary leaks in.
    assert abs(trunc[-1]["advantage"] - (1.0 - 0.7)) < 1e-9
    assert abs(trunc[-1]["return"] - 1.0) < 1e-9
    for a, b in zip(trunc, term):
        assert abs(a["advantage"] - b["advantage"]) < 1e-9
        assert abs(a["return"] - b["return"]) < 1e-9


def test_flatten_observation_matches_observation_dim():
    dim = observation_dim()
    env = CatanAECEnv(randomize_board=False, seed=0)
    env.reset(seed=0)
    obs = env.observe(env.agent_selection)
    flat = flatten_observation(obs)
    assert flat.shape == (dim,)
    assert flat.dtype == np.float32
    assert np.isfinite(flat).all()
