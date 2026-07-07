"""Greedy-expansion + needs-based trade heuristic agent (roadmap phase 3).

Serves three roles from the design doc: a curriculum-stage baseline (§8), an
evaluation-tier opponent distinct from random (§12), and later an imitation-
learning demonstration source (§9). It scores every legal action against a
hand-crafted notion of value rather than searching -- no lookahead beyond a
2-hop BFS for road targeting.
"""
from __future__ import annotations

import random
from collections import deque

from env.actions import Action, ActionType
from env.board import PIP_COUNT, Resource
from env.engine import (
    legal_actions, total_vp, trade_ratio_for, vertex_distance_ok,
)
from env.state import BUILDING_COSTS, GameState, MIN_LONGEST_ROAD, Phase

ROAD_VALUE_THRESHOLD = 5.0
KNIGHT_VALUE_THRESHOLD = 6.0
LONGEST_ROAD_PUSH_BONUS = 1.5
MONOPOLY_MIN_HAUL = 3


def vertex_production_value(state: GameState, vertex_id: int) -> float:
    v = state.board.vertices[vertex_id]
    resources_seen = set()
    value = 0.0
    for hex_id in v.hex_ids:
        hx = state.board.hexes[hex_id]
        if hx.number is None:
            continue
        value += PIP_COUNT[hx.number]
        resources_seen.add(hx.terrain)
    value += 0.5 * len(resources_seen)
    if v.port_generic:
        value += 0.5
    elif v.port is not None:
        value += 1.0
    return value


def best_reachable_vertex_value(state: GameState, from_vertex: int, max_depth: int = 2) -> float:
    """BFS outward (ignoring who owns what along the way) for the best
    currently-distance-legal settlement spot within `max_depth` vertex-hops --
    a stand-in for "is this road pointing somewhere worthwhile."""
    best = 0.0
    visited = {from_vertex}
    frontier = deque([(from_vertex, 0)])
    while frontier:
        v, d = frontier.popleft()
        if d >= max_depth:
            continue
        for adj in state.board.vertices[v].adjacent_vertex_ids:
            if adj in visited:
                continue
            visited.add(adj)
            if vertex_distance_ok(state, adj):
                best = max(best, vertex_production_value(state, adj))
            frontier.append((adj, d + 1))
    return best


def hex_pip(state: GameState, hex_id: int) -> int:
    number = state.board.hexes[hex_id].number
    return PIP_COUNT.get(number, 0)


MAX_TRADE_PROPOSALS_PER_TURN = 2


class HeuristicAgent:
    def __init__(self, player_id: int, rng: random.Random | None = None):
        self.player_id = player_id
        self.rng = rng or random.Random()
        self._trade_turn_key: tuple[int, int] | None = None
        self._trade_proposals_this_turn = 0

    # -- dispatch -----------------------------------------------------------
    def choose(self, state: GameState, legal: list[Action] | None = None) -> Action:
        """`legal` lets a caller that already enumerated this step's legal
        actions (e.g. the rollout loop, via the env's cache) pass them in
        instead of paying for a second enumeration."""
        actions = legal if legal is not None else legal_actions(state)
        if not actions:
            raise RuntimeError(f"No legal actions in phase {state.phase}")
        if len(actions) == 1:
            return actions[0]
        phase = state.phase
        if phase == Phase.SETUP_SETTLEMENT:
            return self._best(actions, lambda a: vertex_production_value(state, a.params["vertex_id"]))
        if phase == Phase.SETUP_ROAD:
            return self._best(actions, lambda a: self._road_reach_score(state, a.params["edge_id"]))
        if phase == Phase.DISCARD:
            return self._choose_discard(state, actions)
        if phase == Phase.MOVE_ROBBER:
            return self._best(actions, lambda a: self._robber_score(state, a))
        if phase == Phase.TRADE_RESPONSE:
            return self._choose_trade_response(state, actions)
        if phase == Phase.MAIN:
            return self._choose_main(state, actions)
        return actions[0]  # ROLL: only ROLL_DICE is legal

    def _best(self, actions: list[Action], score_fn) -> Action:
        return max(actions, key=lambda a: (score_fn(a), self.rng.random()))

    # -- shared valuation -----------------------------------------------------
    def _road_reach_score(self, state: GameState, edge_id: int) -> float:
        a, b = state.board.edges[edge_id].vertex_ids
        return max(best_reachable_vertex_value(state, a), best_reachable_vertex_value(state, b))

    def _road_score(self, state: GameState, edge_id: int) -> float:
        score = self._road_reach_score(state, edge_id)
        player = state.players[self.player_id]
        if state.longest_road_holder != self.player_id and len(player.roads) + 1 >= MIN_LONGEST_ROAD:
            score += LONGEST_ROAD_PUSH_BONUS
        return score

    def _robber_score(self, state: GameState, action: Action) -> float:
        hex_id, victim = action.params["hex_id"], action.params["victim"]
        if victim is None:
            return -1.0
        return hex_pip(state, hex_id) * (1.0 + 0.3 * total_vp(state, victim))

    def _target(self, state: GameState) -> tuple[str, dict[Resource, int]]:
        """What the agent is working toward next: city upgrade > new
        settlement > dev card, restricted to piece-limit-legal options.
        Drives trade/discard decisions, not build legality (legal_actions
        already enforces that)."""
        player = state.players[self.player_id]
        if len(player.cities) < 4 and player.settlements:
            return "city", self._missing(state, BUILDING_COSTS["city"])
        if len(player.settlements) < 5:
            return "settlement", self._missing(state, BUILDING_COSTS["settlement"])
        return "dev_card", self._missing(state, BUILDING_COSTS["dev_card"])

    def _missing(self, state: GameState, cost: dict[Resource, int]) -> dict[Resource, int]:
        hand = state.players[self.player_id].resources
        return {r: max(0, amt - hand.get(r, 0)) for r, amt in cost.items()}

    # -- MAIN phase -----------------------------------------------------------
    def _choose_main(self, state: GameState, actions: list[Action]) -> Action:
        turn_key = (state.turn_number, state.current_player)
        if turn_key != self._trade_turn_key:
            self._trade_turn_key = turn_key
            self._trade_proposals_this_turn = 0

        by_type: dict[ActionType, list[Action]] = {}
        for a in actions:
            by_type.setdefault(a.type, []).append(a)

        if ActionType.BUILD_CITY in by_type:
            return self._best(by_type[ActionType.BUILD_CITY],
                               lambda a: vertex_production_value(state, a.params["vertex_id"]))

        if ActionType.BUILD_SETTLEMENT in by_type:
            return self._best(by_type[ActionType.BUILD_SETTLEMENT],
                               lambda a: vertex_production_value(state, a.params["vertex_id"]))

        if ActionType.BUILD_ROAD in by_type:
            best_road = self._best(by_type[ActionType.BUILD_ROAD],
                                    lambda a: self._road_score(state, a.params["edge_id"]))
            if self._road_score(state, best_road.params["edge_id"]) >= ROAD_VALUE_THRESHOLD:
                return best_road

        knight = self._maybe_play_knight(state, by_type.get(ActionType.PLAY_KNIGHT, []))
        if knight is not None:
            return knight

        mono = self._maybe_play_monopoly(state, by_type.get(ActionType.PLAY_MONOPOLY, []))
        if mono is not None:
            return mono

        yop = self._maybe_play_year_of_plenty(state, by_type.get(ActionType.PLAY_YEAR_OF_PLENTY, []))
        if yop is not None:
            return yop

        if ActionType.BUY_DEV_CARD in by_type:
            return by_type[ActionType.BUY_DEV_CARD][0]

        trade = self._maybe_maritime_trade(state, by_type.get(ActionType.MARITIME_TRADE, []))
        if trade is not None:
            return trade

        propose = self._maybe_propose_trade(state, by_type.get(ActionType.PROPOSE_TRADE, []))
        if propose is not None:
            return propose

        return by_type[ActionType.END_TURN][0]

    def _maybe_play_knight(self, state: GameState, knight_actions: list[Action]) -> Action | None:
        if not knight_actions:
            return None
        player = state.players[self.player_id]
        holder = state.largest_army_holder
        pushes_largest_army = (
            player.knights_played + 1 >= 3
            and (holder is None or player.knights_played + 1 > state.players[holder].knights_played)
        )
        best = self._best(knight_actions, lambda a: self._robber_score(state, a))
        if pushes_largest_army or self._robber_score(state, best) >= KNIGHT_VALUE_THRESHOLD:
            return best
        return None

    def _maybe_play_monopoly(self, state: GameState, mono_actions: list[Action]) -> Action | None:
        if not mono_actions:
            return None
        _, missing = self._target(state)
        best, best_haul = None, 0
        for a in mono_actions:
            r = a.params["resource"]
            haul = sum(state.players[pid].resources[r] for pid in state.players if pid != self.player_id)
            if haul > best_haul:
                best, best_haul = a, haul
        if best is not None and best_haul >= MONOPOLY_MIN_HAUL and missing.get(best.params["resource"], 0) > 0:
            return best
        return None

    def _maybe_play_year_of_plenty(self, state: GameState, yop_actions: list[Action]) -> Action | None:
        if not yop_actions:
            return None
        _, missing = self._target(state)
        needed = {r for r, amt in missing.items() if amt > 0}
        if not needed:
            return None
        best, best_score = None, 0
        for a in yop_actions:
            score = sum(1 for r in a.params["resources"] if r in needed)
            if score > best_score:
                best, best_score = a, score
        return best

    def _maybe_maritime_trade(self, state: GameState, trade_actions: list[Action]) -> Action | None:
        if not trade_actions:
            return None
        _, missing = self._target(state)
        needed = {r for r, amt in missing.items() if amt > 0}
        if not needed:
            return None
        hand = state.players[self.player_id].resources
        for a in trade_actions:
            give_r, want_r = a.params["give"], a.params["receive"]
            if want_r not in needed:
                continue
            ratio = trade_ratio_for(state, self.player_id, give_r)
            surplus = hand.get(give_r, 0) - missing.get(give_r, 0)
            if surplus >= ratio:
                return a
        return None

    def _maybe_propose_trade(self, state: GameState, propose_actions: list[Action]) -> Action | None:
        if not propose_actions:
            return None
        if self._trade_proposals_this_turn >= MAX_TRADE_PROPOSALS_PER_TURN:
            # a rejected/countered-and-rejected proposal returns to MAIN with
            # the agent's hand unchanged, so without this cap the identical
            # highest-scoring offer would be re-proposed forever.
            return None
        _, missing = self._target(state)
        needed = {r for r, amt in missing.items() if amt > 0}
        if not needed:
            return None
        hand = state.players[self.player_id].resources
        candidates = []
        for a in propose_actions:
            give_r = next(iter(a.params["give"]))
            want_r = next(iter(a.params["want"]))
            if want_r not in needed:
                continue
            surplus = hand.get(give_r, 0) - missing.get(give_r, 0) - 1  # keep a spare
            if surplus >= 0:
                candidates.append(a)
        if candidates:
            self._trade_proposals_this_turn += 1
            return self.rng.choice(candidates)
        return None

    # -- discard --------------------------------------------------------------
    def _choose_discard(self, state: GameState, actions: list[Action]) -> Action:
        """Keep resources needed for the current target; discard surplus,
        breaking ties toward keeping diversity (don't dump an entire type)."""
        _, missing = self._target(state)
        needed_types = {r for r, amt in missing.items() if amt == 0}  # already-sufficient types we still use
        hand = state.players[self.player_id].resources

        def badness(action: Action) -> float:
            cards = action.params["cards"]
            score = 0.0
            for r, amt in cards.items():
                remaining_after = hand.get(r, 0) - amt
                need_weight = 2.0 if missing.get(r, 0) > 0 else 1.0
                score += amt * need_weight
                if remaining_after < 0:
                    score += 100  # infeasible, shouldn't happen
            return score

        return min(actions, key=lambda a: (badness(a), self.rng.random()))

    # -- trade response ---------------------------------------------------------
    def _trade_desirability(self, state: GameState, gain: dict[Resource, int],
                             cost: dict[Resource, int]) -> float:
        _, missing = self._target(state)
        hand = state.players[self.player_id].resources
        gain_score = sum(min(amt, missing.get(r, 0)) for r, amt in gain.items()) * 2.0
        gain_score += sum(gain.values()) * 0.1
        cost_score = 0.0
        for r, amt in cost.items():
            surplus = hand.get(r, 0) - missing.get(r, 0)
            cost_score += amt * (0.1 if surplus >= amt else 1.5)
        return gain_score - cost_score

    def _pick_counter(self, state: GameState, counter_actions: list[Action]) -> Action | None:
        _, missing = self._target(state)
        needed = {r for r, amt in missing.items() if amt > 0}
        hand = state.players[self.player_id].resources
        for a in counter_actions:
            give_r = next(iter(a.params["give"]))
            want_r = next(iter(a.params["want"]))
            if want_r not in needed:
                continue
            surplus = hand.get(give_r, 0) - missing.get(give_r, 0) - 1
            if surplus >= 0:
                return a
        return None

    def _choose_trade_response(self, state: GameState, actions: list[Action]) -> Action:
        by_type: dict[ActionType, list[Action]] = {}
        for a in actions:
            by_type.setdefault(a.type, []).append(a)

        if ActionType.CONFIRM_TRADE in by_type or ActionType.CANCEL_TRADE in by_type:
            if ActionType.CONFIRM_TRADE in by_type:
                return by_type[ActionType.CONFIRM_TRADE][0]  # we proposed it; any accepter is fine
            return by_type[ActionType.CANCEL_TRADE][0]

        if state.trade_counter_context is not None:
            # responding to a counter-offer made to us (only accept/reject exist)
            ctx = state.trade_counter_context
            desirability = self._trade_desirability(state, gain=ctx.give, cost=ctx.want)
            if desirability > 0:
                return by_type[ActionType.ACCEPT_TRADE][0]
            return by_type[ActionType.REJECT_TRADE][0]

        # responding to the original proposal: accepting means we give `want`
        # and receive `give` (see TradeOffer/_execute_trade semantics)
        offer = state.pending_trade
        desirability = self._trade_desirability(state, gain=offer.give, cost=offer.want)
        if desirability > 0.5:
            return by_type[ActionType.ACCEPT_TRADE][0]
        if ActionType.COUNTER_TRADE in by_type:
            counter = self._pick_counter(state, by_type[ActionType.COUNTER_TRADE])
            if counter is not None:
                return counter
        return by_type[ActionType.REJECT_TRADE][0]
