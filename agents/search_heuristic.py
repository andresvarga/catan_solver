"""Search demonstrator (teacher upgrade after the DAgger plateau).

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

Depth (`search_depth`): at depth 2+ the search continues through the
agent's OWN same-turn follow-ups -- after simulating a candidate it also
considers the best next searchable action from the resulting position
(maritime-trade-then-build-city, road-then-settlement), beam-limited to
`beam_width` continuations. The second ply is never an opponent reply:
opponents' turns open with a dice roll and their hands are hidden, so
searching into them would break the same no-hidden-information rule."""
from __future__ import annotations

import copy
import random

from agents.heuristic import HeuristicAgent, vertex_production_value
from env.actions import Action, ActionType
from env.board import HexType, Resource
from env.engine import (
    acting_player, legal_actions, pay, step as engine_step, total_vp, vertex_distance_ok,
)
from env.state import (
    BUILDING_COSTS, GameState, Phase, empty_dev_hand, empty_hand,
)

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


# Same-turn continuation candidates at ply 2+. END_TURN is excluded (its
# value is already the "stop here" baseline every continuation must beat);
# BUY_DEV_CARD is excluded (it's a leaf -- see _score_candidate).
CONTINUATION_TYPES = SEARCHABLE_MAIN - {ActionType.END_TURN, ActionType.BUY_DEV_CARD}


class SearchHeuristicAgent(HeuristicAgent):
    def __init__(self, player_id: int, rng: random.Random | None = None,
                 resource_weights: dict | None = ORE_WEIGHT,
                 weights: dict | None = None,
                 search_depth: int = 2, beam_width: int = 6):
        super().__init__(player_id, rng, resource_weights=resource_weights)
        self._sim_rng = random.Random(0)  # searchable actions never consult it
        self.search_depth = search_depth
        self.beam_width = beam_width
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

    def _position_value(self, state: GameState, depth: int) -> float:
        """Value of a position the agent acts from: its static eval, or --
        while `depth` allows and it's still this agent's MAIN turn -- the
        best value reachable by continuing with further same-turn searchable
        actions (beam-limited by static eval). Never recurses into another
        player's turn or a non-MAIN phase."""
        v = self.eval_state(state)
        if depth <= 0 or state.winner is not None or state.phase != Phase.MAIN \
                or acting_player(state) != self.player_id:
            return v
        followups = [a for a in legal_actions(state) if a.type in CONTINUATION_TYPES]
        if not followups:
            return v
        scored = []
        for a in followups:
            sim = copy_state(state)
            engine_step(sim, a, rng=self._sim_rng)
            scored.append((self.eval_state(sim), sim))
        scored.sort(key=lambda t: -t[0])
        for leaf_val, sim in scored[:self.beam_width]:
            v = max(v, self._position_value(sim, depth - 1) if depth > 1 else leaf_val)
        return v

    def _score_candidate(self, state: GameState, a: Action) -> float:
        if a.type == ActionType.BUY_DEV_CARD:
            # Simulate the payment only; actually drawing in simulation would
            # condition the choice on deck order (hidden information). The
            # card's value enters as a flat expectation term instead. Leaf:
            # no continuation, since the post-payment state is already an
            # approximation (the real one holds an extra unseen card).
            sim = copy_state(state)
            pay(sim, self.player_id, BUILDING_COSTS["dev_card"])
            return self.eval_state(sim) + self.w["dev_ev"]
        sim = copy_state(state)
        engine_step(sim, a, rng=self._sim_rng)
        return self._position_value(sim, self.search_depth - 1)

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


class RolloutSearchAgent(SearchHeuristicAgent):
    """Adds determinized-rollout evaluation on top of the same-turn search.

    The same-turn search's blind spot is everything that happens after
    END_TURN: opponent builds that take contested spots, the robber landing
    on us, dice variance between "bank toward a city" and "build the road
    now". Deepening the same-turn search cannot see any of that (depths 2
    and 3 measured equal-strength), so the upgrade is a different axis:
    for the top few near-tied candidates, play out one full table rotation
    with the real engine and average the resulting positions.

    Hidden information is handled by determinization, which is what keeps
    the demonstration policy imitable: before each rollout, everything this
    seat cannot see -- opponents' hand compositions, opponents' unplayed dev
    cards, the deck order -- is re-sampled from public knowledge (hand/card
    counts are public; `state.public_resource_estimates` is the engine's
    card-counting prior). The rollout then runs on a sampled world, so its
    outcome is a function of public state + own hand only, never of the true
    hidden cards. This also makes BUY_DEV_CARD honestly simulable inside a
    rollout: drawing from a re-shuffled deck leaks nothing.

    Rollouts are paired (common random numbers): the same K determinization/
    dice seeds are reused across all finalists, so most of the sampling noise
    cancels in the comparison. Cost control: rollouts trigger only when >=2
    candidates sit within `rollout_margin` eval-points of the best -- the
    clear-cut majority of decisions stay pure same-turn search."""

    def __init__(self, player_id: int, rng: random.Random | None = None,
                 resource_weights: dict | None = ORE_WEIGHT,
                 weights: dict | None = None,
                 search_depth: int = 2, beam_width: int = 6,
                 rollouts: int = 6, rollout_top_m: int = 3,
                 rollout_margin: float = 3.0, rollout_rotations: int = 1,
                 max_rollout_steps: int = 150, win_bonus: float = 150.0,
                 rollout_agent_factory=None):
        super().__init__(player_id, rng, resource_weights=resource_weights,
                          weights=weights, search_depth=search_depth,
                          beam_width=beam_width)
        self.rollouts = rollouts
        self.rollout_top_m = rollout_top_m
        self.rollout_margin = rollout_margin
        self.rollout_rotations = rollout_rotations
        self.max_rollout_steps = max_rollout_steps
        self.win_bonus = win_bonus
        # Expert iteration hook: `rollout_agent_factory(pid) -> agent` replaces
        # the rollout policy for THIS seat only (e.g. with the current trained
        # model -- at deployment this seat's future actions are the model's,
        # so a model rollout policy is the accurate self-model). Opponent
        # seats always stay plain HeuristicAgent: in the eval/collection
        # condition the opponents literally are plain heuristics, so that is
        # the true opponent model, not an approximation. The agent is built
        # once and reused across rollouts, so the factory should return a
        # per-turn-stateless agent (deterministic model agents qualify).
        self._rollout_agent_factory = rollout_agent_factory
        self._rollout_self_agent = None
        # observability counters (tournament scripts read these)
        self.decisions = 0
        self.rollout_decisions = 0
        # per-decision flags (DAgger label filtering reads these): whether the
        # last MAIN-phase choose() ran rollouts, and whether the rollout
        # verdict OVERRODE the static search's top candidate -- overridden
        # decisions are the only ones whose label carries information beyond
        # eval_state (namely, what the model-driven rollouts saw).
        self.last_gated = False
        self.last_overrode = False

    # -- determinization ------------------------------------------------------
    def _determinize(self, sim: GameState, rr: random.Random) -> None:
        """Re-sample everything this seat cannot see, preserving all public
        counts. Mutates `sim` (a copy) only."""
        me = self.player_id

        # Opponents' unplayed dev cards + the deck form one hidden pool.
        # Redeal preserving each opponent's total and bought-this-turn count
        # (both public), remainder becomes the deck (size preserved).
        pool = list(sim.dev_card_deck)
        opp_counts: dict[int, tuple[int, int]] = {}
        for pid, p in sim.players.items():
            if pid == me:
                continue
            opp_counts[pid] = (p.total_dev_cards(),
                               sum(p.dev_cards_bought_this_turn.values()))
            for card, k in p.dev_cards.items():
                pool.extend([card] * k)
            p.dev_cards = empty_dev_hand()
            p.dev_cards_bought_this_turn = empty_dev_hand()
        rr.shuffle(pool)
        i = 0
        for pid, (held, bought) in opp_counts.items():
            p = sim.players[pid]
            dealt = pool[i:i + held]
            i += held
            for card in dealt:
                p.dev_cards[card] += 1
            for card in dealt[:bought]:
                p.dev_cards_bought_this_turn[card] += 1
        sim.dev_card_deck = pool[i:]

        # Opponents' resource hands: counts are public; composition is
        # sampled around the engine's card-counting prior (integer part =
        # publicly certain cards, remainder ~ fractional mass + uniform floor).
        resources = list(Resource)
        for pid, p in sim.players.items():
            if pid == me:
                continue
            n = p.hand_size()
            est = sim.public_resource_estimates.get(pid, {})
            hand = empty_hand()
            known = 0
            for r in resources:
                k = min(n - known, int(est.get(r, 0.0)))
                hand[r] = k
                known += k
            weights = {r: (est.get(r, 0.0) - int(est.get(r, 0.0))) + 0.25
                       for r in resources}
            for _ in range(n - known):
                total = sum(weights.values())
                x = rr.random() * total
                for r in resources:
                    x -= weights[r]
                    if x <= 0:
                        hand[r] += 1
                        break
                else:
                    hand[resources[-1]] += 1
            p.resources = hand

    # -- rollout --------------------------------------------------------------
    def _rollout_value(self, state: GameState, action: Action, seed: int) -> float:
        """Execute `action` in a determinized copy, then play everyone with
        the plain heuristic until the start of this seat's next turn (or a
        winner / step cap), and evaluate. The board is shared with the real
        state and rollouts DO move the robber, so robber_hex is restored
        before returning."""
        rr = random.Random(seed)
        sim = copy_state(state)
        saved_robber = state.board.robber_hex
        try:
            self._determinize(sim, rr)
            engine_step(sim, action, rng=rr)
            agents = {
                pid: HeuristicAgent(
                    pid, random.Random(rr.randrange(2 ** 31)),
                    resource_weights=self.resource_weights if pid == self.player_id else None)
                for pid in sim.players
            }
            if self._rollout_agent_factory is not None:
                if self._rollout_self_agent is None:
                    self._rollout_self_agent = self._rollout_agent_factory(self.player_id)
                agents[self.player_id] = self._rollout_self_agent
            rotations_seen = 0
            for _ in range(self.max_rollout_steps):
                if sim.winner is not None:
                    return self._terminal_value(sim)
                # each arrival at our own pre-roll marks one full rotation;
                # stop at the requested horizon (arrival is counted once --
                # the next step is the roll itself, which leaves Phase.ROLL)
                if sim.current_player == self.player_id and sim.phase == Phase.ROLL:
                    rotations_seen += 1
                    if rotations_seen >= self.rollout_rotations:
                        break
                acts = legal_actions(sim)
                act = acts[0] if len(acts) == 1 else \
                    agents[acting_player(sim)].choose(sim, acts)
                engine_step(sim, act, rng=rr)
            return self._terminal_value(sim)
        finally:
            state.board.robber_hex = saved_robber

    def _terminal_value(self, sim: GameState) -> float:
        v = self.eval_state(sim)
        if sim.winner == self.player_id:
            v += self.win_bonus
        elif sim.winner is not None:
            v -= self.win_bonus
        return v

    def choose(self, state: GameState, legal: list[Action] | None = None) -> Action:
        # reset per-decision flags here, not in _search: MAIN choose() can
        # return via the base heuristic without ever reaching _search, which
        # would otherwise leave stale flags for label filtering to misread
        self.last_gated = False
        self.last_overrode = False
        return super().choose(state, legal)

    # -- search with rollout tie-break -----------------------------------------
    def _search(self, state: GameState, candidates: list[Action]) -> Action:
        """Same-turn search first; when the top candidates are near-tied
        (within rollout_margin) the decision goes to paired determinized
        rollouts. Non-MAIN phases (setup, discard) keep the pure search:
        their rollouts would be dominated by whole-game noise."""
        if state.phase != Phase.MAIN:
            return super()._search(state, candidates)
        self.decisions += 1
        scored = sorted(((self._score_candidate(state, a), a) for a in candidates),
                        key=lambda t: -t[0])
        best_score = scored[0][0]
        finalists = [a for s, a in scored[:self.rollout_top_m]
                     if s >= best_score - self.rollout_margin]
        if len(finalists) < 2:
            return scored[0][1]
        self.rollout_decisions += 1
        self.last_gated = True
        seeds = [self.rng.randrange(2 ** 31) for _ in range(self.rollouts)]
        best_action, best_mean = finalists[0], None
        for a in finalists:
            mean = sum(self._rollout_value(state, a, sd) for sd in seeds) / len(seeds)
            if best_mean is None or mean > best_mean:
                best_action, best_mean = a, mean
        self.last_overrode = best_action is not scored[0][1]
        return best_action
