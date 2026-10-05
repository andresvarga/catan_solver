"""Performance encoders must be bit-identical to the reference encoders
(simulation performance audit)."""
import copy
import random

import numpy as np
import pytest

from agents.random_agent import RandomAgent
from env.engine import CatanEngine, legal_actions
from env.pettingzoo_env import build_observation
from training.model import encode_flat, flatten_observation


def _states(n_games=6, every=7):
    out = []
    for g in range(n_games):
        eng = CatanEngine(seed=74_000_000 + g)
        agents = {i: RandomAgent(i, random.Random(g * 10 + i)) for i in range(4)}
        k = 0
        while not eng.done and k < 2500:
            eng.step(agents[eng.acting_player()].choose(eng.state))
            k += 1
            if k % every == 0:
                out.append(copy.deepcopy(eng.state))
    return out


STATES = _states()


@pytest.mark.parametrize("phf", [False, True])
def test_encode_flat_bit_identical(phf):
    checked = 0
    for s in STATES:
        for pid in range(4):
            ref = flatten_observation(build_observation(s, pid, legal_actions(s), True, public_hand_features=phf))
            got = encode_flat(s, pid, phf)
            assert got.dtype == ref.dtype and got.shape == ref.shape
            assert np.array_equal(got.view(np.uint32), ref.view(np.uint32)), \
                f"mismatch at {np.flatnonzero(got != ref)[:5]}"
            checked += 1
    assert checked > 1000
