"""Audit rule tests (Phases 3-13, 35-36). Deterministic, hand-constructed
states. Tests marked xfail(strict=True) reproduce CONFIRMED deviations from
the official 2020 CATAN base rules (audit/results/catan_rules_2020.txt); they
will start failing (XPASS -> error) once the bug is fixed, prompting removal
of the marker. IDs (F-xx) refer to CATAN_MARL_AUDIT.md."""
from __future__ import annotations

import random
from collections import Counter

import pytest

from audit.helpers import (
    RES, check_invariants, edge_between, fresh_main_state, give, legal_set, path_edges,
    place_road, place_settlement, reference_longest_road, simple_path, state_signature,
    disjoint_path, closed_nbhd,
)
from env.actions import Action, ActionType
from env.board import HexType, Resource, generate_board
from env.engine import (
    CatanEngine, acting_player, compute_longest_road_length, distribute_resources,
    legal_actions, recompute_longest_road, step, total_vp,
)
from env.state import DevCard, Phase, STANDARD_DEV_CARD_COUNTS, new_game

W, B, S, H, O = Resource.WOOD, Resource.BRICK, Resource.SHEEP, Resource.WHEAT, Resource.ORE


def types(state):
    return Counter(a.type for a in legal_actions(state))


# ------------------------------------------------------------------ board
@pytest.mark.parametrize("seed", range(10_000_000, 10_000_200))
def test_board_composition(seed):
    b = generate_board(randomize=True, seed=seed)
    assert len(b.hexes) == 19 and len(b.vertices) == 54 and len(b.edges) == 72
    assert Counter(h.terrain for h in b.hexes.values()) == {
        HexType.WOOD: 4, HexType.BRICK: 3, HexType.SHEEP: 4, HexType.WHEAT: 4,
        HexType.ORE: 3, HexType.DESERT: 1}
    assert sorted(h.number for h in b.hexes.values() if h.number) == \
        [2, 3, 3, 4, 4, 5, 5, 6, 6, 8, 8, 9, 9, 10, 10, 11, 11, 12]
    assert b.hexes[b.robber_hex].terrain == HexType.DESERT
    ports = [v for v in b.vertices.values() if v.port is not None or v.port_generic]
    assert len(ports) == 18
    kinds = Counter(("generic" if v.port_generic else v.port.value) for v in ports)
    assert kinds == {"generic": 8, "wood": 2, "brick": 2, "sheep": 2, "wheat": 2, "ore": 2}


def test_board_red_numbers_not_adjacent():
    """F-10 regression: no 6/8 tokens on edge-sharing hexes (geometric check,
    independent of board.py's axial adjacency), random and fixed boards."""
    boards = [generate_board(randomize=True, seed=sd) for sd in range(10_000_000, 10_001_000)]
    boards.append(generate_board(randomize=False))
    for b in boards:
        hv = {h.id: set(h.vertex_ids) for h in b.hexes.values()}
        red = [h.id for h in b.hexes.values() if h.number in (6, 8)]
        assert not any(len(hv[a] & hv[c]) >= 2 for i, a in enumerate(red) for c in red[i + 1:])


# ------------------------------------------------------------------ setup
def test_setup_snake_order_and_start():
    eng = CatanEngine(seed=10_000_001)
    rng = random.Random(1)
    order = []
    while eng.state.phase in (Phase.SETUP_SETTLEMENT, Phase.SETUP_ROAD):
        if eng.state.phase == Phase.SETUP_SETTLEMENT:
            order.append(eng.acting_player())
        acts = eng.legal_actions()
        if eng.state.phase == Phase.SETUP_ROAD:
            v = eng.state.just_placed_settlement_vertex
            assert all(v in eng.state.board.edges[a.params["edge_id"]].vertex_ids for a in acts)
        eng.step(rng.choice(acts))
    assert order == [0, 1, 2, 3, 3, 2, 1, 0]
    assert eng.state.phase == Phase.ROLL and eng.state.current_player == 0
    for pid, p in eng.state.players.items():
        assert len(p.settlements) == 2 and len(p.roads) == 2
        # second settlement's adjacent producing hexes = starting hand
        second = p.settlements[1]
        expected = Counter()
        for h in eng.state.board.vertices[second].hex_ids:
            hx = eng.state.board.hexes[h]
            if hx.number is not None:
                expected[Resource(hx.terrain.value)] += 1
        assert {r: k for r, k in p.resources.items() if k} == dict(expected)
        assert total_vp(eng.state, pid) == 2


def test_setup_grant_debits_bank():
    eng = CatanEngine(seed=10_000_002)
    rng = random.Random(2)
    while eng.state.phase in (Phase.SETUP_SETTLEMENT, Phase.SETUP_ROAD):
        eng.step(rng.choice(eng.legal_actions()))
    for r in RES:
        held = sum(p.resources[r] for p in eng.state.players.values())
        assert held + eng.state.bank[r] == 19


def test_setup_distance_rule_enforced():
    eng = CatanEngine(seed=10_000_003)
    a = eng.legal_actions()[0]
    eng.step(a)
    v = a.params["vertex_id"]
    eng.step(eng.legal_actions()[0])
    nbrs = set(eng.state.board.vertices[v].adjacent_vertex_ids) | {v}
    for act in eng.legal_actions():
        assert act.params["vertex_id"] not in nbrs


# ------------------------------------------------------------------ turn machine
def test_roll_only_once_and_main_has_no_roll():
    eng = CatanEngine(seed=10_000_004)
    rng = random.Random(4)
    while eng.state.phase != Phase.ROLL:
        eng.step(rng.choice(eng.legal_actions()))
    assert [a.type for a in eng.legal_actions()] == [ActionType.ROLL_DICE]
    eng.step(eng.legal_actions()[0])
    while eng.state.phase != Phase.MAIN:
        eng.step(rng.choice(eng.legal_actions()))
    assert ActionType.ROLL_DICE not in types(eng.state)


def test_knight_playable_before_roll():
    s = fresh_main_state(10_000_005)
    s.phase = Phase.ROLL
    s.players[0].dev_cards[DevCard.KNIGHT] = 1
    assert ActionType.PLAY_KNIGHT in types(s)
    assert legal_actions(s)[0].type == ActionType.ROLL_DICE
    step(s, next(a for a in legal_actions(s) if a.type == ActionType.PLAY_KNIGHT), rng=random.Random(0))
    assert s.phase == Phase.ROLL and s.players[0].knights_played == 1
    assert [a.type for a in legal_actions(s)] == [ActionType.ROLL_DICE]  # one dev card per turn


def test_road_building_before_roll_then_roll():
    s = fresh_main_state(10_000_025)
    s.phase = Phase.ROLL
    place_settlement(s, 0, 0)
    s.players[0].dev_cards[DevCard.ROAD_BUILDING] = 1
    step(s, Action(ActionType.PLAY_ROAD_BUILDING))
    assert s.phase == Phase.ROLL and s.free_roads_remaining == 2
    for _ in range(2):
        acts = legal_actions(s)
        assert acts and all(a.type == ActionType.BUILD_ROAD for a in acts)
        step(s, acts[0])
    assert len(s.players[0].roads) == 2 and s.free_roads_remaining == 0
    assert [a.type for a in legal_actions(s)] == [ActionType.ROLL_DICE]
    assert check_invariants(s) == []


def test_end_turn_rotates_and_resets():
    s = fresh_main_state(10_000_006)
    s.players[0].played_dev_card_this_turn = True
    s.trades_proposed_this_turn = 2
    step(s, Action(ActionType.END_TURN))
    assert s.current_player == 1 and s.phase == Phase.ROLL
    assert not s.players[0].played_dev_card_this_turn and s.trades_proposed_this_turn == 0


# ------------------------------------------------------------------ production
def _producing_setup(seed=10_000_007):
    s = fresh_main_state(seed)
    hx = next(h for h in s.board.hexes.values() if h.number == 8)
    v0, v1 = hx.vertex_ids[0], hx.vertex_ids[2]  # distance-2 apart on the same hex
    place_settlement(s, 1, v0)
    place_settlement(s, 2, v1, kind="city")
    return s, hx


def test_production_settlement_city_robber():
    s, hx = _producing_setup()
    r = Resource(hx.terrain.value)
    before1, before2 = s.players[1].resources[r], s.players[2].resources[r]
    gains = distribute_resources(s, 8)
    # other 8-hex may also pay the same players; isolate this hex's contribution
    other = [h for h in s.board.hexes.values() if h.number == 8 and h.id != hx.id]
    if not any(v in s.vertex_owner for h in other for v in h.vertex_ids):
        assert s.players[1].resources[r] - before1 == 1
        assert s.players[2].resources[r] - before2 == 2
    s2, hx2 = _producing_setup()
    s2.board.robber_hex = hx2.id
    g = distribute_resources(s2, 8)
    assert g[1][Resource(hx2.terrain.value)] == 0 or any(
        v in s2.vertex_owner for h in other for v in h.vertex_ids)


def test_bank_shortage_single_player_gets_remainder():
    s, hx = _producing_setup()
    r = Resource(hx.terrain.value)
    # only player 2 (city, wants 2) produces; bank has 1 left
    s.players[1].settlements.clear(); del s.vertex_owner[hx.vertex_ids[0]]
    for h in s.board.hexes.values():  # neutralize the other 8 hex
        if h.number == 8 and h.id != hx.id:
            h.number = 99
    s.players[3].resources[r] += s.bank[r] - 1; s.bank[r] = 1
    distribute_resources(s, 8)
    assert s.players[2].resources[r] == 1


def test_bank_shortage_multi_player_nobody_gets():
    s, hx = _producing_setup()
    r = Resource(hx.terrain.value)
    for h in s.board.hexes.values():
        if h.number == 8 and h.id != hx.id:
            h.number = 99
    s.players[3].resources[r] += s.bank[r] - 2; s.bank[r] = 2  # need 3
    distribute_resources(s, 8)
    assert s.players[1].resources[r] == 0 and s.players[2].resources[r] == 0


# ------------------------------------------------------------------ building
def test_building_costs_and_limits():
    s = fresh_main_state(10_000_008)
    place_settlement(s, 0, 0)
    e = s.board.vertices[0].edge_ids[0]
    place_road(s, 0, e)
    assert ActionType.BUILD_ROAD not in types(s)
    give(s, 0, wood=1, brick=1)
    assert ActionType.BUILD_ROAD in types(s)
    give(s, 0, wheat=2, ore=3)
    acts = [a for a in legal_actions(s) if a.type == ActionType.BUILD_CITY]
    assert [a.params["vertex_id"] for a in acts] == [0]
    step(s, acts[0])
    assert s.players[0].cities == [0] and s.players[0].settlements == []
    assert total_vp(s, 0) == 2 and s.players[0].resources[O] == 0
    # piece limit: 15 roads
    s.players[0].roads = list(range(15)); s.road_owner = {e2: 0 for e2 in range(15)}
    give(s, 0, wood=1, brick=1)
    assert ActionType.BUILD_ROAD not in types(s)


def test_opponent_settlement_blocks_road_continuation():
    s = fresh_main_state(10_000_009)
    rng = random.Random(9)
    path = simple_path(s.board, 10, 2, rng)  # v0 - v1 - v2
    place_settlement(s, 0, path[0])
    place_road(s, 0, edge_between(s.board, path[0], path[1]))
    place_settlement(s, 1, path[1])  # opponent on the far end of my road
    give(s, 0, wood=1, brick=1)
    blocked_edge = edge_between(s.board, path[1], path[2])
    assert all(a.params["edge_id"] != blocked_edge for a in legal_actions(s)
               if a.type == ActionType.BUILD_ROAD)


def test_settlement_requires_own_road_and_distance():
    s = fresh_main_state(10_000_010)
    rng = random.Random(10)
    p = simple_path(s.board, 20, 3, rng)
    place_settlement(s, 0, p[0])
    for e in path_edges(s.board, p):
        place_road(s, 0, e)
    give(s, 0, wood=1, brick=1, sheep=1, wheat=1)
    verts = {a.params["vertex_id"] for a in legal_actions(s) if a.type == ActionType.BUILD_SETTLEMENT}
    assert p[1] not in verts  # distance rule
    assert verts <= {p[2], p[3]}
    assert p[2] in verts or p[3] in verts


# ------------------------------------------------------------------ robber / 7
def test_seven_discard_threshold_and_order():
    s = fresh_main_state(10_000_011)
    s.phase = Phase.ROLL
    give(s, 0, wood=4, brick=4)        # 8 -> discard 4
    give(s, 2, sheep=7)                # 7 -> no discard
    give(s, 3, ore=3, wheat=3, wood=3) # 9 -> discard 4

    class SevenRng(random.Random):
        def randint(self, a, b):
            return 3 if not hasattr(self, "_x") else 4
    rng = random.Random()
    rng.randint = lambda a, b, it=iter([3, 4]): next(it)
    step(s, Action(ActionType.ROLL_DICE), rng=rng)
    assert s.phase == Phase.DISCARD and s.players_to_discard == [0, 3]
    assert s.discard_amounts == {0: 4, 3: 4}
    assert all(sum(a.params["cards"].values()) == 4 for a in legal_actions(s))
    step(s, legal_actions(s)[0])
    assert acting_player(s) == 3
    step(s, legal_actions(s)[0])
    assert s.phase == Phase.MOVE_ROBBER and acting_player(s) == 0
    assert all(a.params["hex_id"] != s.board.robber_hex for a in legal_actions(s))


@pytest.mark.xfail(strict=True, reason="F-12: DISCARD enumeration capped at 100 combos -- some legal "
                                         "discards are unrepresentable for large hands")
def test_discard_all_combinations_representable():
    s = fresh_main_state(10_000_012)
    s.phase = Phase.DISCARD
    give(s, 0, wood=4, brick=4, sheep=4, wheat=4, ore=4)
    s.players_to_discard = [0]; s.discard_amounts = {0: 10}
    # number of multisets of size 10 bounded by 4 per type = 101 (exceeds the cap)
    assert len(legal_actions(s)) == 101


def test_robber_steal_moves_one_card_and_victims_adjacent():
    s = fresh_main_state(10_000_013)
    hx = next(h for h in s.board.hexes.values() if h.id != s.board.robber_hex)
    place_settlement(s, 1, hx.vertex_ids[0])
    give(s, 1, ore=3)
    s.phase = Phase.MOVE_ROBBER
    acts = [a for a in legal_actions(s) if a.params["hex_id"] == hx.id]
    assert [a.params["victim"] for a in acts] == [1]
    step(s, acts[0], rng=random.Random(0))
    assert s.players[0].resources[O] == 1 and s.players[1].resources[O] == 2
    assert s.board.robber_hex == hx.id and s.phase == Phase.MAIN


def test_steal_uniform_over_cards():
    counts = Counter()
    for i in range(4000):
        s = fresh_main_state(10_000_014)
        hx = next(h for h in s.board.hexes.values() if h.id != s.board.robber_hex)
        place_settlement(s, 1, hx.vertex_ids[0])
        give(s, 1, ore=3, wood=1)
        s.phase = Phase.MOVE_ROBBER
        step(s, Action(ActionType.MOVE_ROBBER, {"hex_id": hx.id, "victim": 1}), rng=random.Random(i))
        counts[next(r for r in RES if s.players[0].resources[r])] += 1
    assert abs(counts[O] / 4000 - 0.75) < 0.03


# ------------------------------------------------------------------ dev cards
def test_dev_deck_composition():
    s = new_game(seed=10_000_015)
    assert Counter(s.dev_card_deck) == STANDARD_DEV_CARD_COUNTS
    assert len(s.dev_card_deck) == 25


def test_dev_card_not_playable_turn_bought_and_one_per_turn():
    s = fresh_main_state(10_000_016)
    s.dev_card_deck = [DevCard.KNIGHT] * 3
    give(s, 0, sheep=1, wheat=1, ore=1)
    step(s, Action(ActionType.BUY_DEV_CARD))
    assert ActionType.PLAY_KNIGHT not in types(s)
    step(s, Action(ActionType.END_TURN))
    for _ in range(3):  # cycle back to p0
        s.phase = Phase.MAIN
        step(s, Action(ActionType.END_TURN))
    s.phase = Phase.MAIN
    s.players[0].dev_cards[DevCard.MONOPOLY] = 1
    assert ActionType.PLAY_KNIGHT in types(s)
    step(s, next(a for a in legal_actions(s) if a.type == ActionType.PLAY_KNIGHT))
    assert ActionType.PLAY_MONOPOLY not in types(s)


def test_vp_card_counts_immediately_and_can_win():
    s = fresh_main_state(10_000_017)
    for v in (0, 2, 4, 6, 8):
        pass
    s.players[0].dev_cards[DevCard.VICTORY_POINT] = 9
    s.dev_card_deck = [DevCard.VICTORY_POINT]
    give(s, 0, sheep=1, wheat=1, ore=1)
    step(s, Action(ActionType.BUY_DEV_CARD))
    assert s.winner == 0 and s.phase == Phase.GAME_OVER
    assert legal_actions(s) == []


def test_monopoly_and_year_of_plenty():
    s = fresh_main_state(10_000_018)
    s.players[0].dev_cards[DevCard.MONOPOLY] = 1
    give(s, 1, ore=2); give(s, 2, ore=3, wood=1)
    step(s, Action(ActionType.PLAY_MONOPOLY, {"resource": O}))
    assert s.players[0].resources[O] == 5 and s.players[1].resources[O] == 0
    assert check_invariants(s) == []
    s2 = fresh_main_state(10_000_019)
    s2.players[0].dev_cards[DevCard.YEAR_OF_PLENTY] = 1
    s2.bank[O] = 1
    combos = [tuple(sorted(r.value for r in a.params["resources"])) for a in legal_actions(s2)
              if a.type == ActionType.PLAY_YEAR_OF_PLENTY]
    assert ("ore", "ore") not in combos and len(combos) == 14


def test_road_building_does_not_lock_turn():
    s = fresh_main_state(10_000_020)
    place_settlement(s, 0, 0)
    s.players[0].roads = []
    # 14 roads already used elsewhere -> only 1 piece left
    others = [e for e in s.board.edges if 0 not in s.board.edges[e].vertex_ids][:14]
    for e in others:
        place_road(s, 0, e)
    s.players[0].dev_cards[DevCard.ROAD_BUILDING] = 1
    give(s, 0, wheat=2, ore=3)
    step(s, Action(ActionType.PLAY_ROAD_BUILDING))
    road = next(a for a in legal_actions(s) if a.type == ActionType.BUILD_ROAD)
    step(s, road)
    assert ActionType.BUILD_CITY in types(s)


# ------------------------------------------------------------------ trading
def test_maritime_ratios_by_port():
    s = fresh_main_state(10_000_021)
    gen = next(v for v in s.board.vertices.values() if v.port_generic)
    spec = next(v for v in s.board.vertices.values() if v.port == W)
    give(s, 0, wood=3, brick=3, sheep=4)
    m = lambda: {(a.params["give"], a.params["receive"]) for a in legal_actions(s)
                 if a.type == ActionType.MARITIME_TRADE}
    assert {g for g, _ in m()} == {S}
    place_settlement(s, 0, gen.id)
    assert {g for g, _ in m()} == {W, B, S}
    place_settlement(s, 0, spec.id) if spec.id not in s.board.vertices[gen.id].adjacent_vertex_ids else None
    step(s, Action(ActionType.MARITIME_TRADE, {"give": B, "receive": O}))
    assert s.players[0].resources[B] == 0 and s.players[0].resources[O] == 1


def test_maritime_trade_requires_bank_stock():
    s = fresh_main_state(10_000_022)
    give(s, 0, wood=4)
    s.players[1].resources[O] += s.bank[O]; s.bank[O] = 0
    assert not any(a.type == ActionType.MARITIME_TRADE and a.params["receive"] == O
                   for a in legal_actions(s))


def test_trade_protocol_atomic_and_only_current_player_proposes():
    s = fresh_main_state(10_000_023)
    give(s, 0, wood=1); give(s, 2, ore=1)
    from env.engine import ALL_OPPONENTS, make_trade
    step(s, make_trade(ActionType.PROPOSE_TRADE, {W: 1}, {O: 1}, actor=0, target=ALL_OPPONENTS))
    assert acting_player(s) == 1
    step(s, Action(ActionType.REJECT_TRADE))   # p1 has no ore: cannot accept
    step(s, Action(ActionType.ACCEPT_TRADE))   # p2 accepts
    step(s, Action(ActionType.REJECT_TRADE))   # p3
    assert acting_player(s) == 0 and {a.type for a in legal_actions(s)} == \
        {ActionType.CONFIRM_TRADE, ActionType.CANCEL_TRADE}
    step(s, Action(ActionType.CONFIRM_TRADE, {"target": 2}))
    assert s.players[0].resources[O] == 1 and s.players[2].resources[W] == 1
    assert check_invariants(s) == []


def test_accept_requires_ability_to_pay():
    """F-13 regression."""
    from env.engine import make_trade
    s = fresh_main_state(10_000_024)
    give(s, 0, wood=1)
    step(s, make_trade(ActionType.PROPOSE_TRADE, {W: 1}, {O: 1}, actor=0, target=1))
    assert ActionType.ACCEPT_TRADE not in types(s)


# ------------------------------------------------------------------ longest road
def _lr_state(seed=10_000_030):
    return fresh_main_state(seed)


def test_longest_road_shapes_vs_reference():
    s = _lr_state()
    b = s.board
    rng = random.Random(30)
    for trial in range(300):
        s = _lr_state()
        start = rng.randrange(54)
        # random connected road network of 5-12 edges grown from start
        net = set()
        frontier = [start]
        target = rng.randint(5, 12)
        while len(net) < target:
            v = rng.choice(frontier)
            e = rng.choice(b.vertices[v].edge_ids)
            if e in net:
                continue
            net.add(e)
            frontier.extend(b.edges[e].vertex_ids)
        for e in net:
            place_road(s, 0, e)
        blocked = set()
        verts = sorted({v for e in net for v in b.edges[e].vertex_ids})
        for v in rng.sample(verts, k=rng.randint(0, 2)):
            if all(adj not in s.vertex_owner for adj in b.vertices[v].adjacent_vertex_ids):
                place_settlement(s, 1, v)
                blocked.add(v)
        assert compute_longest_road_length(s, 0) == reference_longest_road(b, list(net), blocked), trial


def test_longest_road_award_transfer_and_ties():
    s = _lr_state(10_000_031)
    b = s.board
    rng = random.Random(31)
    p0 = simple_path(b, 0, 5, rng)
    for e in path_edges(b, p0):
        place_road(s, 0, e)
    recompute_longest_road(s)
    assert s.longest_road_holder == 0 and total_vp(s, 0) == 2
    used = closed_nbhd(b, p0)
    p1 = disjoint_path(b, 5, rng, used)
    for e in path_edges(b, p1):
        place_road(s, 1, e)
    recompute_longest_road(s)
    assert s.longest_road_holder == 0, "tie keeps holder"
    # break p0's road in the middle: 0 drops to 2/3 -> p1 sole leader takes it
    place_settlement(s, 2, p0[2])
    recompute_longest_road(s)
    assert s.road_lengths[0] < 5 and s.longest_road_holder == 1


def test_longest_road_broken_tie_among_others_sets_aside():
    s = _lr_state(10_000_032)
    b = s.board
    rng = random.Random(32)
    p0 = simple_path(b, 0, 6, rng)
    for e in path_edges(b, p0):
        place_road(s, 0, e)
    used = closed_nbhd(b, p0)
    p1 = disjoint_path(b, 5, rng, used)
    used |= closed_nbhd(b, p1)
    p2 = disjoint_path(b, 5, rng, used)
    for e in path_edges(b, p1): place_road(s, 1, e)
    for e in path_edges(b, p2): place_road(s, 2, e)
    recompute_longest_road(s)
    assert s.longest_road_holder == 0
    place_settlement(s, 3, p0[3])
    recompute_longest_road(s)
    assert s.longest_road_holder is None


# ------------------------------------------------------------------ largest army / VP
def test_largest_army_threshold_tie_and_transfer():
    s = fresh_main_state(10_000_040)
    from env.engine import award_knight
    award_knight(s, 1); award_knight(s, 1)
    assert s.largest_army_holder is None
    award_knight(s, 1)
    assert s.largest_army_holder == 1
    for _ in range(3):
        award_knight(s, 2)
    assert s.largest_army_holder == 1
    award_knight(s, 2)
    assert s.largest_army_holder == 2 and total_vp(s, 2) == 2


def test_cannot_win_on_another_players_turn():
    s = fresh_main_state(10_000_050)
    b = s.board
    rng = random.Random(50)
    p0 = simple_path(b, 0, 6, rng)  # holder p1 with 6
    for e in path_edges(b, p0):
        place_road(s, 1, e)
    used = closed_nbhd(b, p0)
    p2 = disjoint_path(b, 5, rng, used)
    for e in path_edges(b, p2):
        place_road(s, 2, e)
    recompute_longest_road(s)
    assert s.longest_road_holder == 1
    s.players[2].dev_cards[DevCard.VICTORY_POINT] = 8  # 8 hidden VP -> LR would make 10
    # current player 0 builds a settlement breaking p1's road (connected via own road)
    mid = p0[3]
    nb = next(v for v in b.vertices[mid].adjacent_vertex_ids if v not in p0)
    place_road(s, 0, edge_between(b, mid, nb))
    give(s, 0, wood=1, brick=1, sheep=1, wheat=1)
    act = Action(ActionType.BUILD_SETTLEMENT, {"vertex_id": mid})
    if act in legal_actions(s):
        step(s, act)
    else:
        pytest.skip("construction not legal on this board")
    assert s.longest_road_holder == 2
    assert s.winner is None and s.phase != Phase.GAME_OVER  # not p2's turn: no win yet
    # p0 and p1 end their turns; p2 claims the win at the start of its own turn
    step(s, Action(ActionType.END_TURN))
    assert s.winner is None and s.current_player == 1
    s.phase = Phase.MAIN
    step(s, Action(ActionType.END_TURN))
    assert s.winner == 2 and s.phase == Phase.GAME_OVER


# ------------------------------------------------------------------ adversarial engine API
def test_engine_step_rejects_illegal_action():
    """F-05 regression: the validated entry points agents and training use
    (CatanEngine.step, CatanAECEnv.step) reject illegal actions before
    mutating anything. The raw functional `engine.step` stays unchecked by
    design (hot simulation loops in the search agents feed it only legal
    actions)."""
    eng = CatanEngine(seed=10_000_060)
    s = eng.state
    s.phase = Phase.MAIN
    s.turn_number = 1
    place_settlement(s, 0, 0)
    before = state_signature(s)
    for bad in (Action(ActionType.BUILD_CITY, {"vertex_id": 0}),        # no resources
                Action(ActionType.BUILD_SETTLEMENT, {"vertex_id": 0}),  # occupied
                Action(ActionType.ROLL_DICE),                           # wrong phase
                Action(ActionType.MOVE_ROBBER, {"hex_id": 0, "victim": None})):
        with pytest.raises(ValueError):
            eng.step(bad)
    assert state_signature(s) == before


def test_env_rejects_out_of_range_index_without_mutation():
    from env.pettingzoo_env import CatanAECEnv
    env = CatanAECEnv(seed=10_000_061)
    env.reset(seed=10_000_061)
    before = state_signature(env.engine.state)
    n = len(env.legal_actions())
    for bad in (-1, n, 399, 10_000):
        with pytest.raises(ValueError):
            env.step(bad)
    assert state_signature(env.engine.state) == before
