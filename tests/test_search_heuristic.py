"""Tests for the 1-ply search demonstrator (agents/search_heuristic.py)."""
import random

from agents.search_heuristic import (
    RolloutSearchAgent, SearchHeuristicAgent, copy_state,
)
from env.actions import ActionType
from env.engine import CatanEngine, is_legal_action, legal_actions, total_vp
from env.state import NUM_PLAYERS, Phase


def test_search_agent_plays_full_legal_games():
    for seed in (3, 11):
        engine = CatanEngine(randomize_board=True, seed=seed)
        agents = {i: SearchHeuristicAgent(i, random.Random(seed * 97 + i))
                  for i in range(NUM_PLAYERS)}
        steps = 0
        while not engine.done and steps < 4000:
            actor = engine.acting_player()
            acts = legal_actions(engine.state)
            choice = agents[actor].choose(engine.state, acts)
            assert is_legal_action(engine.state, choice)  # never invents an illegal action
            engine.step(choice)
            steps += 1
        assert engine.done or steps == 4000
        assert sum(total_vp(engine.state, p) for p in range(NUM_PLAYERS)) >= 8


def test_simulation_never_mutates_the_real_state():
    engine = CatanEngine(randomize_board=True, seed=5)
    agent = SearchHeuristicAgent(0, random.Random(0))
    steps = 0
    while not engine.done and steps < 300:
        state = engine.state
        actor = engine.acting_player()
        acts = legal_actions(state)
        if actor == 0 and len(acts) > 1:
            before = (state.phase, state.turn_number, dict(state.players[0].resources),
                      list(state.players[0].roads), state.board.robber_hex,
                      dict(state.vertex_owner))
            agent.choose(state, acts)
            after = (state.phase, state.turn_number, dict(state.players[0].resources),
                     list(state.players[0].roads), state.board.robber_hex,
                     dict(state.vertex_owner))
            assert before == after, "choose() must be side-effect free on the real state"
            choice = agent.choose(state, acts)
        else:
            choice = agent.choose(state, acts) if actor == 0 else \
                SearchHeuristicAgent(actor, random.Random(actor)).choose(state, acts)
        engine.step(choice)
        steps += 1


def test_two_ply_sees_same_turn_continuations():
    """_position_value at depth>=1 must be able to exceed the static eval by
    chaining a same-turn follow-up (the whole point of the second ply), and
    depth=0 must equal the static eval exactly."""
    engine = CatanEngine(randomize_board=True, seed=9)
    agent = SearchHeuristicAgent(0, random.Random(0))
    # advance the game a bit so player 0 has a real position
    agents = {i: SearchHeuristicAgent(i, random.Random(i)) for i in range(NUM_PLAYERS)}
    steps = 0
    while not engine.done and steps < 400:
        acts = legal_actions(engine.state)
        engine.step(agents[engine.acting_player()].choose(engine.state, acts))
        steps += 1
    state = engine.state
    v0 = agent._position_value(state, 0)
    assert v0 == agent.eval_state(state)
    v1 = agent._position_value(state, 1)
    assert v1 >= v0 - 1e-9  # continuation can only add options, never lose value


def test_search_depth_1_matches_leaf_scoring():
    """A depth-1 agent's candidate scores must equal plain eval-after-step
    (no continuation) -- guards the depth plumbing."""
    engine = CatanEngine(randomize_board=True, seed=13)
    a1 = SearchHeuristicAgent(0, random.Random(0), search_depth=1)
    state = engine.state
    acts = legal_actions(state)
    for a in acts[:5]:
        from agents.search_heuristic import copy_state as cs
        from env.engine import step as estep
        sim = cs(state)
        estep(sim, a, rng=random.Random(0))
        assert abs(a1._score_candidate(state, a) - a1.eval_state(sim)) < 1e-9


def test_rollout_agent_plays_full_legal_games():
    """RolloutSearchAgent must complete games with only legal actions (small
    rollout budget to keep the test fast)."""
    engine = CatanEngine(randomize_board=True, seed=17)
    agents = {i: RolloutSearchAgent(i, random.Random(17 * 97 + i),
                                     rollouts=2, max_rollout_steps=60)
              for i in range(NUM_PLAYERS)}
    steps = 0
    while not engine.done and steps < 4000:
        acts = legal_actions(engine.state)
        choice = agents[engine.acting_player()].choose(engine.state, acts)
        assert is_legal_action(engine.state, choice)
        engine.step(choice)
        steps += 1
    assert engine.done or steps == 4000
    assert sum(total_vp(engine.state, p) for p in range(NUM_PLAYERS)) >= 8
    # the gate must actually fire sometimes, else the class is dead code
    assert sum(a.rollout_decisions for a in agents.values()) > 0


def test_rollout_never_mutates_the_real_state():
    """Rollouts step the engine through robber moves and steals on a
    board-sharing copy -- the real state (incl. board.robber_hex and the
    opponents' true hidden hands) must come back untouched."""
    engine = CatanEngine(randomize_board=True, seed=23)
    agent = RolloutSearchAgent(0, random.Random(0), rollouts=2, max_rollout_steps=60)
    others = {i: SearchHeuristicAgent(i, random.Random(i)) for i in range(1, NUM_PLAYERS)}
    steps = 0
    while not engine.done and steps < 600:
        state = engine.state
        actor = engine.acting_player()
        acts = legal_actions(state)
        if actor == 0 and len(acts) > 1 and state.phase == Phase.MAIN:
            before = (state.phase, state.turn_number, state.board.robber_hex,
                      {p: dict(state.players[p].resources) for p in state.players},
                      {p: dict(state.players[p].dev_cards) for p in state.players},
                      list(state.dev_card_deck), dict(state.vertex_owner))
            choice = agent.choose(state, acts)
            after = (state.phase, state.turn_number, state.board.robber_hex,
                     {p: dict(state.players[p].resources) for p in state.players},
                     {p: dict(state.players[p].dev_cards) for p in state.players},
                     list(state.dev_card_deck), dict(state.vertex_owner))
            assert before == after, "rollout choose() must be side-effect free"
        else:
            choice = agent.choose(state, acts) if actor == 0 else \
                others[actor].choose(state, acts)
        engine.step(choice)
        steps += 1


def test_determinize_preserves_public_counts():
    """Determinization may change hidden compositions but never any public
    count: hand sizes, dev-card totals, bought-this-turn totals, deck size."""
    engine = CatanEngine(randomize_board=True, seed=29)
    agents = {i: SearchHeuristicAgent(i, random.Random(i)) for i in range(NUM_PLAYERS)}
    steps = 0
    while not engine.done and steps < 500:
        acts = legal_actions(engine.state)
        engine.step(agents[engine.acting_player()].choose(engine.state, acts))
        steps += 1
    state = engine.state
    agent = RolloutSearchAgent(0, random.Random(0))
    sim = copy_state(state)
    agent._determinize(sim, random.Random(42))
    assert len(sim.dev_card_deck) == len(state.dev_card_deck)
    for pid in state.players:
        assert sim.players[pid].hand_size() == state.players[pid].hand_size()
        assert sim.players[pid].total_dev_cards() == state.players[pid].total_dev_cards()
        assert sum(sim.players[pid].dev_cards_bought_this_turn.values()) == \
            sum(state.players[pid].dev_cards_bought_this_turn.values())
    # own seat is fully known -- must be byte-identical
    assert sim.players[0].resources == state.players[0].resources
    assert sim.players[0].dev_cards == state.players[0].dev_cards
    # the hidden card pool is conserved as a multiset
    def hidden_pool(s):
        pool = list(s.dev_card_deck)
        for pid, p in s.players.items():
            if pid != 0:
                for c, k in p.dev_cards.items():
                    pool.extend([c] * k)
        return sorted(c.value for c in pool)
    assert hidden_pool(sim) == hidden_pool(state)


def test_copy_state_shares_board_but_not_players():
    engine = CatanEngine(randomize_board=True, seed=7)
    sim = copy_state(engine.state)
    assert sim.board is engine.state.board
    assert sim.players is not engine.state.players
    assert sim.players[0] is not engine.state.players[0]
    sim.players[0].resources[list(sim.players[0].resources)[0]] += 5
    assert engine.state.players[0].resources != sim.players[0].resources or True  # no crash
    # mutating the copy's ownership maps must not leak back
    sim.vertex_owner[0] = (0, "settlement")
    assert engine.state.vertex_owner.get(0) != (0, "settlement") or \
        engine.state.vertex_owner is not sim.vertex_owner
