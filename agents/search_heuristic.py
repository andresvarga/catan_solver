"""1-ply search demonstrator (teacher upgrade after the DAgger plateau).

The plain HeuristicAgent scores candidate actions with static formulas
(vertex_production_value etc.) and never looks at the state an action
actually produces. DAgger against it converged at ~28% -- the model absorbed
everything that teacher could show it, so further gains need a stronger
teacher, not more data (measured: 6 extra DAgger rounds / ~6000 games moved
nothing on fresh seeds).

This agent upgrades exactly the decisions where lookahead is well-defined:
actions whose outcome is deterministic and public (builds, maritime trades,
year-of-plenty, discards, setup placements, end-turn). Each candidate is
simulated with the real engine `step` on a copy of the state (board shared,
it's immutable) and the resulting position is scored by `eval_state`. This
makes consequences the static formulas can't see -- longest-road flips, port
acquisition, whether a discard actually protects the next build -- part of
the choice. Action types whose outcome is hidden or interactive (dev-card
draws, robber steals, monopoly, player trades) keep the base heuristic's
behavior: simulating them would either leak hidden information into the
demonstration policy (which the student, seeing only public state, could
never reproduce) or require modeling opponent responses.
"""
from __future__ import annotations

import copy
import random

from agents.heuristic import HeuristicAgent, vertex_production_value
from env.actions import Action, ActionType
from env.board import HexType
from env.engine import pay, step as engine_step, total_vp, vertex_distance_ok
from env.state import BUILDING_COSTS, GameState, Phase

# Deterministic, public-outcome types -- safe and meaningful to simulate.
# BUY_DEV_CARD is searchable via a payment-only simulation plus a flat
# expected-value term (see _score_candidate); the draw itself is never
# simulated because deck order is hidden information.
SEARCHABLE_MAIN = {
    ActionType.BUILD_ROAD, ActionType.BUILD_SETTLEMENT, ActionType.BUILD_CITY,
    ActionType.MARITIME_TRADE, ActionType.PLAY_YEAR_OF_PLENTY, ActionType.END_TURN,
    ActionType.BUY_DEV_CARD,
}
ORE_WEIGHT = {HexType.ORE: 1.5}  # same confirmed weighting the labeler already uses

# eval_state weights -- overridable per instance (weights=dict) so
# scripts can tournament-sweep them empirically instead of hand-guessing.
DEFAULT_EVAL_WEIGHTS = {
    "vp": 10.0,           # per victory point
    "production": 0.5,    # per pip-value of own settlements (cities x2)
    "expansion": 0.5,     # best reachable settlement spot (see _expansion_value)
    "missing": 0.4,       # per resource still missing toward the next build target.
                          # Tournament-swept (two independent 3-4k-game seed sets):
                          # 0.4 and 0.2 both beat the original 0.7 by +1.2-1.8pp;
                          # 0.0 collapses to 20% (the term is what makes the agent
                          # save toward builds at all), so stay off that cliff.
    "road_length": 0.15,  # per own longest-road length
    "hand_risk": 0.10,    # per card above 7 (robber-discard exposure)
    "opp_vp": 2.0,        # per best-opponent victory point
    "dev_ev": 3.0,        # expected value of an unseen dev card (5/25 are 1 VP,
                          # 14/25 knights; scored as a flat bonus when search
                          # considers BUY_DEV_CARD, since actually drawing in
                          # simulation would leak the deck order)
}


def copy_state(state: GameState) -> GameState:
    """Deep copy sharing the board (deep-copying 19 hexes/54 vertices/72
    edges per candidate would dominate the search cost). Safe ONLY because no
    searchable action mutates the board: the one board-level mutable field is
    robber_hex, and MOVE_ROBBER/PLAY_KNIGHT are excluded from search (their
    steal outcome is hidden-information anyway)."""
    return copy.deepcopy(state, memo={id(state.board): state.board})


class SearchHeuristicAgent(HeuristicAgent):
    def __init__(self, player_id: int, rng: random.Random | None = None,
                 resource_weights: dict | None = ORE_WEIGHT,
                 weights: dict | None = None):
        super().__init__(player_id, rng, resource_weights=resource_weights)
        self._sim_rng = random.Random(0)  # searchable actions never consult it
        self.w = dict(DEFAULT_EVAL_WEIGHTS)
        if weights:
            self.w.update(weights)

    # -- position evaluation --------------------------------------------------
    def _expansion_value(self, state: GameState) -> float:
        """Value of the best settlement spot the road network can reach: full
        value on a network vertex, discounted 0.6 one road away, 0.35 two
        roads away. Without this term a 1-ply search is blind to why roads
        are worth their cost (a road produces nothing immediately) and
        degenerates into hoarding -- measured 18% vs the 25% seat-parity
        floor before it existed. The depth-2 ring lets the search see the
        first road of a two-road plan, which depth-1 could not."""
        me = state.players[self.player_id]
        network: set[int] = set()
        for eid in me.roads:
            network.update(state.board.edges[eid].vertex_ids)
        best = 0.0

        def spot_value(vid: int, discount: float) -> float:
            if vid not in state.vertex_owner and vertex_distance_ok(state, vid):
                return discount * vertex_production_value(state, vid, self.resource_weights)
            return 0.0

        ring1: set[int] = set()
        for vid in network:
            best = max(best, spot_value(vid, 1.0))
            for adj in state.board.vertices[vid].adjacent_vertex_ids:
                if adj not in network:
                    ring1.add(adj)
                    best = max(best, spot_value(adj, 0.6))
        for vid in ring1:
            for adj in state.board.vertices[vid].adjacent_vertex_ids:
                if adj not in network and adj not in ring1:
                    best = max(best, spot_value(adj, 0.35))
        return best

    def eval_state(self, state: GameState) -> float:
        me = state.players[self.player_id]
        w = self.w
        v = w["vp"] * total_vp(state, self.player_id)

        prod = sum(vertex_production_value(state, vid, self.resource_weights)
                   for vid in me.settlements)
        prod += 2.0 * sum(vertex_production_value(state, vid, self.resource_weights)
                          for vid in me.cities)
        v += w["production"] * prod

        if len(me.settlements) < 5:  # spot value only matters while expandable
            v += w["expansion"] * self._expansion_value(state)

        # closeness to the next build target (same target logic the base
        # heuristic's trading already uses)
        _, missing = self._target(state)
        v -= w["missing"] * sum(missing.values())

        v += w["road_length"] * state.road_lengths.get(self.player_id, 0)
        v -= w["hand_risk"] * max(0, me.hand_size() - 7)   # robber-discard exposure

        opp_best = max(total_vp(state, pid) for pid in state.players if pid != self.player_id)
        v -= w["opp_vp"] * opp_best
        return v

    def _score_candidate(self, state: GameState, a: Action) -> float:
        if a.type == ActionType.BUY_DEV_CARD:
            # Simulate the payment only; actually drawing in simulation would
            # condition the choice on deck order (hidden information). The
            # card's value enters as a flat expectation term instead.
            sim = copy_state(state)
            pay(sim, self.player_id, BUILDING_COSTS["dev_card"])
            return self.eval_state(sim) + self.w["dev_ev"]
        sim = copy_state(state)
        engine_step(sim, a, rng=self._sim_rng)
        return self.eval_state(sim)

    def _search(self, state: GameState, candidates: list[Action]) -> Action:
        best_action, best_score = None, None
        for a in candidates:
            score = self._score_candidate(state, a)
            if best_score is None or score > best_score + 1e-9 or \
               (abs(score - best_score) <= 1e-9 and self.rng.random() < 0.5):
                best_action, best_score = a, score
        return best_action

    # -- dispatch -------------------------------------------------------------
    def choose(self, state: GameState, legal: list[Action] | None = None) -> Action:
        from env.engine import legal_actions
        actions = legal if legal is not None else legal_actions(state)
        if not actions:
            raise RuntimeError(f"No legal actions in phase {state.phase}")
        if len(actions) == 1:
            return actions[0]

        phase = state.phase
        if phase in (Phase.SETUP_SETTLEMENT, Phase.SETUP_ROAD, Phase.DISCARD):
            return self._search(state, actions)
        if phase == Phase.MAIN:
            searchable = [a for a in actions if a.type in SEARCHABLE_MAIN]
            base_choice = super().choose(state, actions)
            if base_choice.type not in SEARCHABLE_MAIN:
                # base heuristic wants a stochastic/interactive action (knight,
                # dev card, monopoly, trade proposal) -- trust its gates there.
                return base_choice
            if not searchable:
                return base_choice
            return self._search(state, searchable)
        # ROLL / MOVE_ROBBER / TRADE_RESPONSE etc.: base behavior
        return super().choose(state, actions)
