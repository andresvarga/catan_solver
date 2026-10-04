import random

import pytest

from env.actions import Action, ActionType
from env.board import Resource
from env.engine import (
    CatanEngine, acting_player, award_knight, can_build_city, can_build_road,
    can_build_settlement, check_win, compute_longest_road_length, distribute_resources,
    legal_actions, recompute_longest_road, step, total_vp, trade_ratio_for,
    ALL_OPPONENTS, is_template, make_trade,
)
from env.state import BUILDING_COSTS, DevCard, MAX_TRADE_PROPOSALS_PER_TURN, Phase, new_game


def play_setup_phase(engine: CatanEngine, rng: random.Random) -> None:
    """Fast-forward through the 8 setup placements with legal random choices."""
    while engine.state.phase in (Phase.SETUP_SETTLEMENT, Phase.SETUP_ROAD):
        actions = engine.legal_actions()
        engine.step(rng.choice(actions))


def test_new_game_with_no_seed_gives_different_dev_card_orders_across_calls():
    """Regression test companion to test_board.py's board-layout version:
    the dev-card deck shuffle used to fall back to a fixed seed too."""
    orders = set()
    for _ in range(8):
        state = new_game(seed=None)
        orders.add(tuple(state.dev_card_deck))
    assert len(orders) > 1, "expected different dev-card orders across repeated seed=None calls"


def test_setup_phase_order_and_resource_grant():
    engine = CatanEngine(randomize_board=True, seed=10)
    rng = random.Random(10)
    play_setup_phase(engine, rng)
    assert engine.state.phase == Phase.ROLL
    assert engine.state.current_player == 0
    for p in engine.state.players.values():
        assert len(p.settlements) == 2
        assert len(p.roads) == 2
        # second settlement should have granted at least 1 resource (unless
        # placed adjacent only to other placed settlements, astronomically
        # unlikely on turn 1 but not otherwise impossible) -- just assert
        # hands aren't uniformly empty across all four players.
    total_cards = sum(p.hand_size() for p in engine.state.players.values())
    assert total_cards > 0


def test_settlement_distance_rule():
    state = new_game(seed=1)
    v0 = 0
    assert can_build_settlement(state, 0, v0, require_road=False)
    state.players[0].settlements.append(v0)
    state.vertex_owner[v0] = (0, "settlement")
    assert not can_build_settlement(state, 1, v0, require_road=False)
    for adj in state.board.vertices[v0].adjacent_vertex_ids:
        assert not can_build_settlement(state, 1, adj, require_road=False)


def test_build_road_requires_connection():
    state = new_game(seed=2)
    v0 = 0
    state.players[0].settlements.append(v0)
    touching_edges = state.board.vertices[v0].edge_ids
    other_edges = [e for e in state.board.edges if e not in touching_edges]
    assert can_build_road(state, 0, touching_edges[0])
    # find an edge with neither endpoint touching player 0's network
    far_edge = None
    for eid in other_edges:
        a, b = state.board.edges[eid].vertex_ids
        if v0 not in (a, b) and a not in state.board.vertices[v0].adjacent_vertex_ids \
           and b not in state.board.vertices[v0].adjacent_vertex_ids:
            far_edge = eid
            break
    assert far_edge is not None
    assert not can_build_road(state, 0, far_edge)


def test_build_city_requires_own_settlement():
    state = new_game(seed=3)
    v0 = 0
    assert not can_build_city(state, 0, v0)
    state.players[0].settlements.append(v0)
    assert can_build_city(state, 0, v0)
    state.players[1].settlements.append(1)
    assert not can_build_city(state, 0, 1)


def test_piece_limits():
    state = new_game(seed=4)
    state.players[0].settlements = list(range(5))
    assert not can_build_settlement(state, 0, 10, require_road=False)
    state.players[0].cities = list(range(4))
    state.players[0].settlements = [50]
    assert not can_build_city(state, 0, 50)


def test_dice_production_settlement_and_city():
    state = new_game(seed=5)
    hx = next(h for h in state.board.hexes.values() if h.number == 8)
    vid = hx.vertex_ids[0]
    state.players[0].settlements.append(vid)
    state.vertex_owner[vid] = (0, "settlement")
    vid2 = hx.vertex_ids[3]
    state.players[1].cities.append(vid2)
    state.vertex_owner[vid2] = (1, "city")
    from env.board import HEX_TO_RESOURCE
    resource = HEX_TO_RESOURCE[hx.terrain]
    before_p0 = state.players[0].resources[resource]
    before_p1 = state.players[1].resources[resource]
    distribute_resources(state, 8)
    assert state.players[0].resources[resource] == before_p0 + 1
    assert state.players[1].resources[resource] == before_p1 + 2


def test_dice_production_skips_robbed_hex():
    state = new_game(seed=6)
    hx = next(h for h in state.board.hexes.values() if h.number == 9)
    vid = hx.vertex_ids[0]
    state.players[0].settlements.append(vid)
    state.vertex_owner[vid] = (0, "settlement")
    state.board.robber_hex = hx.id
    from env.board import HEX_TO_RESOURCE
    resource = HEX_TO_RESOURCE[hx.terrain]
    before = state.players[0].resources[resource]
    distribute_resources(state, 9)
    assert state.players[0].resources[resource] == before


def test_bank_shortage_rule_blocks_everyone():
    state = new_game(seed=7)
    hx = next(h for h in state.board.hexes.values() if h.number == 5)
    from env.board import HEX_TO_RESOURCE
    resource = HEX_TO_RESOURCE[hx.terrain]
    state.bank[resource] = 1
    state.players[0].cities.append(hx.vertex_ids[0])  # wants 2
    state.vertex_owner[hx.vertex_ids[0]] = (0, "city")
    before_bank = state.bank[resource]
    distribute_resources(state, 5)
    assert state.bank[resource] == before_bank  # nobody got any
    assert state.players[0].resources[resource] == 0


def test_move_robber_steals_from_victim():
    state = new_game(seed=8)
    hx_id = 0
    victim_vertex = state.board.hexes[hx_id].vertex_ids[0]
    state.players[1].settlements.append(victim_vertex)
    state.vertex_owner[victim_vertex] = (1, "settlement")
    state.players[1].resources[Resource.WOOD] = 3
    state.phase = Phase.MOVE_ROBBER
    state.current_player = 0
    rng = random.Random(0)
    step(state, Action(ActionType.MOVE_ROBBER, {"hex_id": hx_id, "victim": 1}), rng=rng)
    assert state.board.robber_hex == hx_id
    assert state.players[1].hand_size() == 2
    assert state.players[0].hand_size() == 1


def test_discard_flow_after_seven():
    state = new_game(seed=9)
    state.players[0].resources[Resource.WOOD] = 8
    state.phase = Phase.ROLL
    state.current_player = 0
    rng = random.Random(0)
    # force a 7 by monkeypatching roll_two_dice via direct phase transition
    from env import engine as eng
    eng._begin_discard_or_robber(state)
    assert state.phase in (Phase.DISCARD, Phase.MOVE_ROBBER)
    if state.phase == Phase.DISCARD:
        assert acting_player(state) == 0
        assert state.discard_amounts[0] == 4
        actions = legal_actions(state)
        assert all(a.type == ActionType.DISCARD for a in actions)
        assert all(sum(a.params["cards"].values()) == 4 for a in actions)
        step(state, actions[0], rng=rng)
        assert 0 not in state.players_to_discard
        assert state.phase == Phase.MOVE_ROBBER


def test_longest_road_minimum_length_and_award():
    state = new_game(seed=11)
    # build a path of 5 connected roads for player 0 starting from vertex 0
    v = 0
    roads = []
    visited = {v}
    for _ in range(5):
        candidates = [e for e in state.board.vertices[v].edge_ids]
        chosen = None
        for eid in candidates:
            a, b = state.board.edges[eid].vertex_ids
            nxt = b if a == v else a
            if nxt not in visited:
                chosen = (eid, nxt)
                break
        assert chosen is not None
        eid, nxt = chosen
        roads.append(eid)
        state.players[0].roads.append(eid)
        visited.add(nxt)
        v = nxt
    length = compute_longest_road_length(state, 0)
    assert length == 5
    recompute_longest_road(state)
    assert state.longest_road_holder == 0


def test_largest_army_requires_three_knights_and_strict_transfer():
    state = new_game(seed=12)
    award_knight(state, 0)
    award_knight(state, 0)
    assert state.largest_army_holder is None
    award_knight(state, 0)
    assert state.largest_army_holder == 0
    award_knight(state, 1)
    award_knight(state, 1)
    award_knight(state, 1)
    assert state.largest_army_holder == 0  # tie: incumbent keeps it
    award_knight(state, 1)
    assert state.largest_army_holder == 1  # strictly exceeded


def test_win_condition_via_settlements_and_bonuses():
    state = new_game(seed=13)
    state.players[0].settlements = list(range(4))  # 4 VP
    state.players[0].cities = [10, 11, 12]  # 6 VP
    assert total_vp(state, 0) == 10
    check_win(state)
    assert state.winner == 0
    assert state.phase == Phase.GAME_OVER


def test_hidden_victory_point_card_can_trigger_win():
    state = new_game(seed=14)
    state.players[0].settlements = list(range(3))  # 3 VP
    state.players[0].cities = [10, 11, 12]  # 6 VP => 9 total so far
    assert total_vp(state, 0) == 9
    state.players[0].dev_cards[DevCard.VICTORY_POINT] = 1  # 10th VP, hidden until now
    check_win(state)
    assert state.winner == 0


def test_counter_offer_is_decided_by_original_proposer():
    state = new_game(seed=16)
    state.players[0].resources[Resource.WOOD] = 3
    state.players[1].resources[Resource.BRICK] = 3
    state.phase = Phase.TRADE_RESPONSE
    state.current_player = 0
    state.pending_trade = __import__("env.state", fromlist=["TradeOffer"]).TradeOffer(
        proposer=0, give={Resource.WOOD: 1}, want={Resource.SHEEP: 1}, targets=[1])
    state.trade_targets_remaining = [1]
    assert acting_player(state) == 1  # target 1 must respond first

    rng = random.Random(0)
    step(state, Action(ActionType.COUNTER_TRADE,
                        {"give": {Resource.BRICK: 1}, "want": {Resource.WOOD: 1}}), rng=rng)
    # after the counter, the ORIGINAL proposer (0) must accept/reject it, not
    # player 1 (the counter-offerer) accepting their own counter.
    assert acting_player(state) == 0
    actions = legal_actions(state)
    assert {a.type for a in actions} == {ActionType.ACCEPT_TRADE, ActionType.REJECT_TRADE}

    before_0_wood, before_0_brick = state.players[0].resources[Resource.WOOD], state.players[0].resources[Resource.BRICK]
    before_1_wood, before_1_brick = state.players[1].resources[Resource.WOOD], state.players[1].resources[Resource.BRICK]
    step(state, Action(ActionType.ACCEPT_TRADE, {}), rng=rng)
    assert state.players[0].resources[Resource.BRICK] == before_0_brick + 1
    assert state.players[0].resources[Resource.WOOD] == before_0_wood - 1
    assert state.players[1].resources[Resource.WOOD] == before_1_wood + 1
    assert state.players[1].resources[Resource.BRICK] == before_1_brick - 1
    assert state.phase == Phase.MAIN


def test_maritime_trade_ratio_prefers_best_port():
    state = new_game(seed=15)
    vid = next(iter(state.board.vertices))
    v = state.board.vertices[vid]
    v.port = Resource.WOOD
    v.port_generic = False
    state.players[0].settlements.append(vid)
    assert trade_ratio_for(state, 0, Resource.WOOD) == 2
    assert trade_ratio_for(state, 0, Resource.ORE) == 4


def test_propose_trade_is_capped_per_turn_and_resets_on_end_turn():
    state = new_game(seed=17)
    for r in Resource:
        state.players[0].resources[r] = 3
    state.phase = Phase.MAIN
    state.current_player = 0
    rng = random.Random(0)

    def propose_and_reject_all(st):
        actions = [a for a in legal_actions(st) if a.type == ActionType.PROPOSE_TRADE]
        assert actions, "expected PROPOSE_TRADE to still be legal"
        assert is_template(actions[0])  # structured trades: one template entry
        step(st, make_trade(ActionType.PROPOSE_TRADE, {Resource.WOOD: 1}, {Resource.ORE: 1},
                            actor=st.current_player, target=ALL_OPPONENTS), rng=rng)
        while st.phase == Phase.TRADE_RESPONSE:
            resp_actions = legal_actions(st)
            reject = next((a for a in resp_actions if a.type == ActionType.REJECT_TRADE), None)
            step(st, reject or resp_actions[0], rng=rng)

    for _ in range(MAX_TRADE_PROPOSALS_PER_TURN):
        propose_and_reject_all(state)

    assert state.trades_proposed_this_turn == MAX_TRADE_PROPOSALS_PER_TURN
    assert not [a for a in legal_actions(state) if a.type == ActionType.PROPOSE_TRADE]

    end_turn = next(a for a in legal_actions(state) if a.type == ActionType.END_TURN)
    step(state, end_turn, rng=rng)
    assert state.trades_proposed_this_turn == 0
    state.phase = Phase.MAIN  # skip past ROLL for this test; we're checking the reset, not turn flow
    for r in Resource:
        state.players[state.current_player].resources[r] = 3
    assert [a for a in legal_actions(state) if a.type == ActionType.PROPOSE_TRADE]


def test_full_random_games_never_crash_and_terminate(monkeypatch=None):
    from agents.random_agent import choose
    for seed in range(15):
        engine = CatanEngine(randomize_board=True, seed=seed)
        rng = random.Random(seed)
        steps = 0
        while not engine.done and steps < 4000:
            engine.step(choose(engine.state, rng))
            steps += 1
        assert steps < 4000, f"seed {seed} did not terminate"
        assert engine.state.winner is not None
        assert total_vp(engine.state, engine.state.winner) >= 10


def test_road_cannot_continue_through_opponent_settlement():
    """Official rule: an opponent's settlement/city blocks road continuation
    through its vertex. Regression test for the engine allowing builds that
    compute_longest_road_length would then refuse to count."""
    state = new_game(seed=20)
    # interior-ish vertex with >= 3 edges so there's an e2 distinct from e1
    vid = next(v.id for v in state.board.vertices.values() if len(v.edge_ids) == 3)
    v = state.board.vertices[vid]
    e1, e2 = v.edge_ids[0], v.edge_ids[1]
    # player 0 has a road ending at vid via e1
    state.players[0].roads.append(e1)
    state.road_owner[e1] = 0
    assert can_build_road(state, 0, e2), "sanity: continuation legal with vertex unoccupied"

    # opponent settlement on vid blocks continuing through it
    state.players[1].settlements.append(vid)
    state.vertex_owner[vid] = (1, "settlement")
    assert not can_build_road(state, 0, e2)

    # ... but the player's OWN settlement there allows it
    state.players[1].settlements.remove(vid)
    state.vertex_owner[vid] = (0, "settlement")
    state.players[0].settlements.append(vid)
    assert can_build_road(state, 0, e2)


def test_road_owner_index_stays_in_sync_with_player_road_lists():
    """`state.road_owner` is the O(1) legality index the engine maintains
    alongside `players[*].roads`; if the two ever diverge, road/settlement
    legality silently rots. Play full random games and check the invariant."""
    from agents.random_agent import choose
    for seed in range(5):
        engine = CatanEngine(randomize_board=True, seed=seed)
        rng = random.Random(seed)
        steps = 0
        while not engine.done and steps < 4000:
            engine.step(choose(engine.state, rng))
            steps += 1
        expected = {eid: pid for pid, p in engine.state.players.items() for eid in p.roads}
        assert engine.state.road_owner == expected
        total_roads = sum(len(p.roads) for p in engine.state.players.values())
        assert len(engine.state.road_owner) == total_roads  # no edge owned twice
