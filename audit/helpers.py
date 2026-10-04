"""Shared audit utilities: invariant checker, independent rule oracles, and
state-construction helpers. Audit-only -- imports production code but never
modifies it."""
from __future__ import annotations

import copy
import random
from collections import Counter, defaultdict

from env.actions import Action, ActionType
from env.board import HEX_TO_RESOURCE, Resource
from env.engine import (
    CatanEngine, acting_player, legal_actions, step as engine_step, total_vp,
)
from env.state import (
    BUILDING_COSTS, DevCard, NUM_PLAYERS, Phase, STANDARD_DEV_CARD_COUNTS, new_game,
)

RES = list(Resource)
TOTAL_PER_RESOURCE = 19
TOTAL_DEV = sum(STANDARD_DEV_CARD_COUNTS.values())


# --------------------------------------------------------------------------
# Invariants
# --------------------------------------------------------------------------

def check_invariants(state, strict_conservation: bool = True,
                     expected_totals: dict | None = None) -> list[str]:
    """Return a list of violated invariants (empty = OK). `expected_totals`
    (resource -> hands+bank) overrides the 19-per-resource rule, so drift can
    be measured relative to a post-setup baseline."""
    errs: list[str] = []
    b = state.board
    # non-negative hands/bank, conservation
    for r in RES:
        if state.bank[r] < 0:
            errs.append(f"bank[{r.value}]={state.bank[r]} < 0")
        held = 0
        for pid, p in state.players.items():
            if p.resources[r] < 0:
                errs.append(f"p{pid}.{r.value}={p.resources[r]} < 0")
            held += p.resources[r]
        target = expected_totals[r] if expected_totals else TOTAL_PER_RESOURCE
        if strict_conservation and held + state.bank[r] != target:
            errs.append(f"conservation {r.value}: hands {held} + bank {state.bank[r]} != {target}")
    # dev cards
    dev_total = len(state.dev_card_deck)
    played_est = 0
    for pid, p in state.players.items():
        for c, k in p.dev_cards.items():
            if k < 0:
                errs.append(f"p{pid}.dev[{c.value}]={k} < 0")
            dev_total += k
        for c, k in p.dev_cards_bought_this_turn.items():
            if k > p.dev_cards[c]:
                errs.append(f"p{pid} bought_this_turn[{c.value}]={k} > held {p.dev_cards[c]}")
        played_est += p.knights_played
    if dev_total > TOTAL_DEV:
        errs.append(f"dev cards in deck+hands {dev_total} > 25")
    knights_total = (Counter(state.dev_card_deck)[DevCard.KNIGHT]
                     + sum(p.dev_cards[DevCard.KNIGHT] for p in state.players.values())
                     + sum(p.knights_played for p in state.players.values()))
    if knights_total != 14:
        errs.append(f"knight conservation {knights_total} != 14")
    # pieces
    for pid, p in state.players.items():
        if len(p.settlements) > 5: errs.append(f"p{pid} settlements {len(p.settlements)} > 5")
        if len(p.cities) > 4: errs.append(f"p{pid} cities {len(p.cities)} > 4")
        if len(p.roads) > 15: errs.append(f"p{pid} roads {len(p.roads)} > 15")
        if len(set(p.roads)) != len(p.roads): errs.append(f"p{pid} duplicate road ids")
    # occupancy consistency
    vo = {}
    for pid, p in state.players.items():
        for v in p.settlements:
            if v in vo: errs.append(f"vertex {v} doubly occupied")
            vo[v] = (pid, "settlement")
        for v in p.cities:
            if v in vo: errs.append(f"vertex {v} doubly occupied")
            vo[v] = (pid, "city")
    if vo != dict(state.vertex_owner):
        errs.append("vertex_owner cache out of sync with player lists")
    ro = {}
    for pid, p in state.players.items():
        for e in p.roads:
            if e in ro: errs.append(f"edge {e} doubly occupied")
            ro[e] = pid
    if ro != dict(state.road_owner):
        errs.append("road_owner cache out of sync with player lists")
    # distance rule
    for v in vo:
        for adj in b.vertices[v].adjacent_vertex_ids:
            if adj in vo:
                errs.append(f"distance rule violated at {v}-{adj}")
    # road connectivity: every road touches own building or own road through a
    # vertex not occupied by an opponent (checked loosely: touches own network)
    for pid, p in state.players.items():
        own_v = set(p.settlements) | set(p.cities)
        for e in p.roads:
            a, c = b.edges[e].vertex_ids
            ok = False
            for x in (a, c):
                if x in own_v:
                    ok = True
                for e2 in b.vertices[x].edge_ids:
                    if e2 != e and ro.get(e2) == pid:
                        ok = True
            if not ok:
                errs.append(f"p{pid} road {e} disconnected")
    # robber
    if state.board.robber_hex not in b.hexes:
        errs.append("robber off-board")
    # current player / phase
    if state.current_player not in range(NUM_PLAYERS):
        errs.append("bad current_player")
    if state.phase != Phase.GAME_OVER:
        if not legal_actions(state):
            errs.append(f"no legal actions in non-terminal phase {state.phase}")
    if state.winner is not None and state.phase != Phase.GAME_OVER:
        errs.append("winner set but phase not GAME_OVER")
    # award consistency
    if state.longest_road_holder is not None and state.road_lengths.get(state.longest_road_holder, 0) < 5:
        errs.append("longest road holder below 5")
    if state.largest_army_holder is not None and \
            state.players[state.largest_army_holder].knights_played < 3:
        errs.append("largest army holder below 3 knights")
    return errs


# --------------------------------------------------------------------------
# Independent Longest Road reference (edge-subset characterization)
# --------------------------------------------------------------------------

def reference_longest_road(board, player_edges: list[int], blocked: set[int]) -> int:
    """Longest trail via subset enumeration -- independent of the engine's DFS.

    A set S of the player's edges can be traversed as one trail iff S is
    connected and has 0 or 2 odd-degree vertices (Euler). An opponent-occupied
    vertex v may only be a trail endpoint: deg_S(v)==1, or deg_S(v)==2 with S a
    closed circuit starting/ending at v. Exponential -- only for <= ~16 edges.
    """
    edges = list(player_edges)
    n = len(edges)
    if n == 0:
        return 0
    ends = [board.edges[e].vertex_ids for e in edges]
    best = 0
    for mask in range(1, 1 << n):
        k = bin(mask).count("1")
        if k <= best:
            continue
        deg: dict[int, int] = defaultdict(int)
        adj: dict[int, list[int]] = defaultdict(list)
        sel = [i for i in range(n) if mask >> i & 1]
        for i in sel:
            a, c = ends[i]
            deg[a] += 1; deg[c] += 1
            adj[a].append(c); adj[c].append(a)
        odd = [v for v, d in deg.items() if d % 2]
        if len(odd) not in (0, 2):
            continue
        # connectivity
        start = ends[sel[0]][0]
        seen = {start}; stack = [start]
        while stack:
            x = stack.pop()
            for y in adj[x]:
                if y not in seen:
                    seen.add(y); stack.append(y)
        if len(seen) != len(deg):
            continue
        ok = True
        circuit_anchor_used = False
        for v in blocked:
            d = deg.get(v, 0)
            if d <= 1:
                continue
            # a closed circuit may start/end at ONE blocked vertex
            if d == 2 and not odd and not circuit_anchor_used:
                circuit_anchor_used = True
                continue
            ok = False
            break
        if ok:
            best = k
    return best


# --------------------------------------------------------------------------
# Construction helpers
# --------------------------------------------------------------------------

def fresh_main_state(seed: int = 0, trading=True, dev=True):
    """A game state in MAIN for player 0 with no buildings at all (setup
    skipped) -- for hand-built scenarios."""
    s = new_game(randomize_board=True, seed=seed, allow_trading=trading, allow_dev_cards=dev)
    s.phase = Phase.MAIN
    s.current_player = 0
    s.turn_number = 1
    return s


def place_settlement(s, pid, vid, kind="settlement"):
    getattr(s.players[pid], "cities" if kind == "city" else "settlements").append(vid)
    s.vertex_owner[vid] = (pid, kind)


def place_road(s, pid, eid):
    s.players[pid].roads.append(eid)
    s.road_owner[eid] = pid


def give(s, pid, **res):
    """Move resources bank -> player (conservation-preserving)."""
    for name, k in res.items():
        r = Resource(name)
        s.players[pid].resources[r] += k
        s.bank[r] -= k


def edge_between(board, a, b):
    for e in board.vertices[a].edge_ids:
        if set(board.edges[e].vertex_ids) == {a, b}:
            return e
    raise KeyError((a, b))


def path_edges(board, vertices: list[int]) -> list[int]:
    return [edge_between(board, vertices[i], vertices[i + 1]) for i in range(len(vertices) - 1)]


def simple_path(board, start: int, length: int, rng: random.Random, avoid: set[int] = frozenset()):
    """Random vertex-simple path of `length` edges starting at `start`."""
    for _ in range(500):
        path = [start]
        while len(path) < length + 1:
            nxt = [v for v in board.vertices[path[-1]].adjacent_vertex_ids
                   if v not in path and v not in avoid]
            if not nxt:
                break
            path.append(rng.choice(nxt))
        if len(path) == length + 1:
            return path
    raise RuntimeError("no path")


def legal_set(state) -> set[str]:
    return {repr(a) for a in legal_actions(state)}


def snapshot(state):
    """Deep copy for before/after comparison (board included)."""
    return copy.deepcopy(state)


def state_signature(state) -> tuple:
    """Hashable summary of everything mutable in a GameState."""
    return (
        tuple(sorted((r.value, k) for r, k in state.bank.items())),
        tuple((pid, tuple(sorted((r.value, k) for r, k in p.resources.items())),
               tuple(sorted((c.value, k) for c, k in p.dev_cards.items())),
               tuple(sorted(p.settlements)), tuple(sorted(p.cities)), tuple(sorted(p.roads)),
               p.knights_played, p.played_dev_card_this_turn)
              for pid, p in sorted(state.players.items())),
        tuple(c.value for c in state.dev_card_deck),
        state.current_player, state.phase.name, state.board.robber_hex,
        state.longest_road_holder, state.largest_army_holder, state.winner, state.turn_number,
        state.dice_roll,
    )


def disjoint_path(board, length: int, rng: random.Random, avoid: set[int]):
    """Vertex-simple path of `length` edges using no vertex in `avoid`."""
    starts = [v for v in range(len(board.vertices)) if v not in avoid]
    rng.shuffle(starts)
    for st in starts:
        try:
            return simple_path(board, st, length, rng, avoid=avoid)
        except RuntimeError:
            continue
    raise RuntimeError("no disjoint path")


def closed_nbhd(board, verts) -> set[int]:
    out = set(verts)
    for v in verts:
        out |= set(board.vertices[v].adjacent_vertex_ids)
    return out
