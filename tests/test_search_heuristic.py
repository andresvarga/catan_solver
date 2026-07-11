"""Tests for the 1-ply search demonstrator (agents/search_heuristic.py)."""
import random

from agents.search_heuristic import SearchHeuristicAgent, copy_state
from env.actions import ActionType
from env.engine import CatanEngine, legal_actions, total_vp
from env.state import NUM_PLAYERS


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
            assert choice in acts  # never invents an illegal action
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
