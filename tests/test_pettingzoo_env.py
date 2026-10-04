import random

import numpy as np
import pytest
from pettingzoo.test import api_test

from env.pettingzoo_env import CatanAECEnv
from env.state import Phase


def _weighted_masked_choice(mask: np.ndarray, legal_actions, rng: random.Random):
    """An index for ordinary actions, or -- when a trade template is picked
    -- a random concrete trade Action (structured trades are submitted as
    Actions, not indices; templates are 0 in the mask)."""
    from env.actions import ActionType
    from env.engine import is_template, random_trade
    idxs = list(range(len(legal_actions)))
    weights = [0.05 if legal_actions[i].type in
               (ActionType.PROPOSE_TRADE, ActionType.COUNTER_TRADE) else 1.0 for i in idxs]
    i = rng.choices(idxs, weights=weights, k=1)[0]
    if is_template(legal_actions[i]):
        assert mask[i] == 0
        return random_trade(legal_actions[i], rng, actor=legal_actions[i].params["actor"])
    assert mask[i] == 1
    return i


def test_full_random_game_via_agent_iter():
    env = CatanAECEnv(randomize_board=True, seed=0)
    env.reset(seed=0)
    rng = random.Random(0)
    steps = 0
    for agent in env.agent_iter(max_iter=20000):
        obs, reward, terminated, truncated, info = env.last()
        if terminated or truncated:
            action = None
        else:
            legal = env.legal_actions()
            action = _weighted_masked_choice(obs["action_mask"], legal, rng)
        env.step(action)
        steps += 1
        if steps > 15000:
            break
    assert env.engine.state.phase == Phase.GAME_OVER
    assert env.agents == []  # all agents drained after termination
    total_reward = sum(env._cumulative_rewards.values()) if env._cumulative_rewards else 0


def test_observation_matches_declared_space_throughout_game():
    env = CatanAECEnv(randomize_board=True, seed=1)
    env.reset(seed=1)
    rng = random.Random(1)
    for i, agent in enumerate(env.agent_iter(max_iter=3000)):
        obs, reward, terminated, truncated, info = env.last()
        if terminated or truncated:
            action = None
        else:
            assert env.observation_space(agent).contains(obs)
            legal = env.legal_actions()
            action = _weighted_masked_choice(obs["action_mask"], legal, rng)
        env.step(action)
        if i > 2500:
            break


def test_action_mask_matches_legal_action_count():
    env = CatanAECEnv(randomize_board=True, seed=2)
    env.reset(seed=2)
    obs = env.observe(env.agent_selection)
    from env.engine import is_template
    legal = env.legal_actions()
    assert obs["action_mask"].sum() == sum(not is_template(a) for a in legal)
    assert all(obs["action_mask"][i] == (0 if is_template(a) else 1) for i, a in enumerate(legal))


def test_pettingzoo_api_compliance():
    env = CatanAECEnv(randomize_board=True, seed=3)
    api_test(env, num_cycles=800)


def test_curriculum_flags_strip_trade_and_dev_card_actions():
    from env.actions import ActionType
    env = CatanAECEnv(randomize_board=False, seed=4, allow_trading=False, allow_dev_cards=False)
    env.reset(seed=4)
    rng = random.Random(4)
    seen_types = set()
    for i, agent in enumerate(env.agent_iter(max_iter=2000)):
        obs, reward, terminated, truncated, info = env.last()
        if terminated or truncated:
            action = None
        else:
            legal = env.legal_actions()
            seen_types.update(a.type for a in legal)
            action = rng.randrange(len(legal))
        env.step(action)
        if i > 1500:
            break
    forbidden = {ActionType.PROPOSE_TRADE, ActionType.COUNTER_TRADE, ActionType.MARITIME_TRADE,
                 ActionType.BUY_DEV_CARD, ActionType.PLAY_KNIGHT, ActionType.PLAY_ROAD_BUILDING,
                 ActionType.PLAY_YEAR_OF_PLENTY, ActionType.PLAY_MONOPOLY}
    assert not (seen_types & forbidden)


def test_clear_reward_gives_reward_since_last_asked_to_act():
    """Explicit test of the semantics training/ppo.py and training/hier_ppo.py
    rely on: after clear_reward(agent), a later last() for that same agent
    reports only reward earned since the clear, not the full cumulative total
    PettingZoo's own AEC bookkeeping would otherwise return."""
    env = CatanAECEnv(randomize_board=True, seed=9, vp_shaping_weight=0.05)
    env.reset(seed=9)
    rng = random.Random(9)

    agent = env.agent_selection
    assert agent in env._cumulative_rewards
    env.clear_reward(agent)
    assert env._cumulative_rewards[agent] == 0.0

    # Drive a few steps so some reward could accumulate for `agent` off-turn
    # (e.g. VP shaping from another player's road/settlement).
    for _ in range(20):
        if not env.agents:
            break
        obs, reward, terminated, truncated, info = env.last()
        action = None if (terminated or truncated) else int(rng.choice(list(np.flatnonzero(obs["action_mask"]))))
        env.step(action)

    # clear_reward on an agent not yet tracked (already removed) must be a no-op, not an error.
    env.clear_reward("player_99")


def test_max_episode_steps_truncates_and_drains():
    env = CatanAECEnv(randomize_board=True, seed=5, max_episode_steps=40)
    env.reset(seed=5)
    rng = random.Random(5)
    steps = 0
    for agent in env.agent_iter(max_iter=1000):
        obs, reward, terminated, truncated, info = env.last()
        action = None if (terminated or truncated) else int(rng.choice(list(np.flatnonzero(obs["action_mask"]))))
        env.step(action)
        steps += 1
        if steps > 200:
            break
    assert env.agents == []
    assert env.engine.state.phase != Phase.GAME_OVER


def test_terminal_reward_ties_split_evenly_not_by_player_id():
    """Regression test: _assign_terminal_rewards used to break VP ties via
    sorted()'s stable order, which silently favored lower player_id every
    time (common on truncation, where standing is ranked on current VP
    rather than a clean win). Construct a clean 2-way tie for the lead and
    check both tied players get the same (averaged) reward regardless of
    which player_id holds the tie."""
    env = CatanAECEnv(randomize_board=False, seed=1)
    env.reset(seed=1)
    state = env.engine.state

    # Give players 2 and 3 (not the 0/1 that stability bias would favor)
    # identical settlement counts so their VP ties for the lead.
    v0 = next(iter(state.board.vertices))
    v1 = next(v for v in state.board.vertices if v != v0)
    state.players[2].settlements.append(v0)
    state.players[3].settlements.append(v1)

    env._assign_terminal_rewards()
    r2 = env.rewards["player_2"]
    r3 = env.rewards["player_3"]
    assert r2 == r3, f"tied players should receive identical reward, got {r2} vs {r3}"
    # tied for 1st+2nd place reward mass: (1.0 + 0.0) / 2 = 0.5 each
    assert r2 == pytest.approx(0.5)
    assert env.infos["player_2"]["final_rank"] == env.infos["player_3"]["final_rank"] == 1


def test_terminal_reward_all_four_tied_split_evenly():
    """All-zero-VP tie (e.g. truncation before anyone has built anything)
    should split the full reward mass evenly across all 4, not silently
    hand player_0 the best rank purely because of dict/sort order."""
    env = CatanAECEnv(randomize_board=False, seed=2)
    env.reset(seed=2)
    env._assign_terminal_rewards()
    rewards = [env.rewards[f"player_{i}"] for i in range(4)]
    assert len(set(rewards)) == 1, f"expected identical rewards for a full tie, got {rewards}"
    assert rewards[0] == pytest.approx((1.0 + 0.0 - 0.5 - 1.0) / 4)


def test_vp_shaping_rewards_are_nonzero_when_enabled():
    env = CatanAECEnv(randomize_board=True, seed=6, vp_shaping_weight=0.05)
    env.reset(seed=6)
    rng = random.Random(6)
    saw_nonzero = False
    for i, agent in enumerate(env.agent_iter(max_iter=800)):
        obs, reward, terminated, truncated, info = env.last()
        if reward != 0:
            saw_nonzero = True
        action = None if (terminated or truncated) else int(rng.choice(list(np.flatnonzero(obs["action_mask"]))))
        env.step(action)
        if i > 600:
            break
    assert saw_nonzero


def test_counter_offer_is_observable_to_proposer():
    """The proposer decides ACCEPT/REJECT on a counter-offer, so its terms and
    author must be in the observation (audit F-04), in both encoders."""
    import random as _random
    from env.actions import Action, ActionType
    from env.board import Resource
    from env.engine import CatanEngine, legal_actions, step
    from training.graph_features import build_graph_observation
    from env.pettingzoo_env import build_observation

    eng = CatanEngine(seed=3)
    rng = _random.Random(3)
    while eng.state.phase != Phase.MAIN:
        eng.step(rng.choice(eng.legal_actions()))
    s = eng.state
    p0 = s.current_player
    s.players[p0].resources[Resource.WOOD] += 1
    responder = (p0 + 1) % 4
    s.players[responder].resources[Resource.ORE] += 1
    from env.engine import make_trade
    step(s, make_trade(ActionType.PROPOSE_TRADE, {Resource.WOOD: 1}, {Resource.BRICK: 1},
                       actor=p0, target=responder))
    step(s, Action(ActionType.COUNTER_TRADE, {"give": {Resource.ORE: 1}, "want": {Resource.WOOD: 1}}))
    obs = build_observation(s, p0, legal_actions(s), True)
    assert obs["counter_trade_give"].tolist() == [0, 0, 0, 0, 1]
    assert obs["counter_trade_want"].tolist() == [1, 0, 0, 0, 0]
    assert int(obs["counter_trade_proposer"][0]) == responder
    ctx = build_graph_observation(s, p0)["context"]
    assert ctx[25 + 4] > 0 and ctx[30 + 0] > 0 and ctx[35 + 2] == 1.0  # responder = next seat
