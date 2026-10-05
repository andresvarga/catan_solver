"""Core Catan rules engine: legality, state transitions, scoring.

Operates on a plain `GameState` (env/state.py) via `legal_actions(state)` and
`step(state, action)`. No RL-specific concepts (observations, rewards, agent
loop) live here -- that's the PettingZoo wrapper's job. Simplifications/rule
variants worth flagging explicitly:
- dev cards may be played before rolling (ROLL phase) or after (MAIN), as
  in the official rules; Road Building's roads are placed immediately, and
  any that cannot be placed are forfeited.
- domestic trades are *structured*: `legal_actions` lists one template
  Action per trade type (PROPOSE_TRADE / COUNTER_TRADE with
  params {"template": True, ...}) instead of enumerating every bundle; the
  agent builds a concrete give/want bundle (1-`MAX_TRADE_CARDS_PER_SIDE`
  cards each side, disjoint resources, addressed to one opponent or all) and
  `is_legal_action` validates it. Templates themselves are not steppable.
- PROPOSE_TRADE is capped at `MAX_TRADE_PROPOSALS_PER_TURN` (state.py) per
  player per turn. This is a deliberate rule variant, not standard Catan: a
  trained policy that hasn't yet learned when to stop negotiating can
  otherwise burn a game's entire step budget cycling propose/respond/confirm
  without ever reaching END_TURN (observed directly -- a self-play episode
  spent 624/800 steps in TRADE_RESPONSE across 156 completed trades and
  advanced all of two turns). The cap bounds that worst case structurally
  instead of hoping training eventually discourages it.
"""
from __future__ import annotations

import random
from collections import Counter, defaultdict
from itertools import combinations_with_replacement

from env.actions import Action, ActionType
from env.board import Resource
from env.state import (
    BUILDING_COSTS, DevCard, GameState, MAX_TRADE_CARDS_PER_SIDE, MAX_TRADE_PROPOSALS_PER_TURN,
    MIN_LARGEST_ARMY,
    MIN_LONGEST_ROAD, NUM_PLAYERS, Phase, STARTING_CITIES, STARTING_ROADS, STARTING_SETTLEMENTS,
    TradeOffer, WINNING_VP, empty_hand, new_game,
)

SETUP_ORDER = [0, 1, 2, 3, 3, 2, 1, 0]


# --------------------------------------------------------------------------
# Query helpers
# --------------------------------------------------------------------------

def owner_of_vertex(state: GameState, vertex_id: int) -> tuple[int | None, str | None]:
    return state.vertex_owner.get(vertex_id, (None, None))


def vertex_distance_ok(state: GameState, vertex_id: int) -> bool:
    owner, _ = owner_of_vertex(state, vertex_id)
    if owner is not None:
        return False
    for adj in state.board.vertices[vertex_id].adjacent_vertex_ids:
        if owner_of_vertex(state, adj)[0] is not None:
            return False
    return True


def road_edge_free(state: GameState, edge_id: int) -> bool:
    return edge_id not in state.road_owner


def player_touches_edge(state: GameState, player_id: int, vertex_id: int) -> bool:
    player = state.players[player_id]
    if vertex_id in player.settlements or vertex_id in player.cities:
        return True
    # An opponent's settlement/city on this vertex blocks road continuation
    # through it (official rule): the player's roads may *end* here, but a
    # new road can't connect to the network via this vertex -- it would have
    # to connect through its other endpoint instead. This mirrors the
    # opponent-vertex cutoff compute_longest_road_length already applies, so
    # build legality and road scoring finally agree.
    owner, _ = owner_of_vertex(state, vertex_id)
    if owner is not None and owner != player_id:
        return False
    road_owner = state.road_owner
    for eid in state.board.vertices[vertex_id].edge_ids:
        if road_owner.get(eid) == player_id:
            return True
    return False


def can_build_road(state: GameState, player_id: int, edge_id: int) -> bool:
    if not road_edge_free(state, edge_id):
        return False
    if len(state.players[player_id].roads) >= STARTING_ROADS:
        return False
    a, b = state.board.edges[edge_id].vertex_ids
    return player_touches_edge(state, player_id, a) or player_touches_edge(state, player_id, b)


def can_build_settlement(state: GameState, player_id: int, vertex_id: int,
                          require_road: bool = True) -> bool:
    if len(state.players[player_id].settlements) >= STARTING_SETTLEMENTS:
        return False
    if not vertex_distance_ok(state, vertex_id):
        return False
    if require_road:
        road_owner = state.road_owner
        if not any(road_owner.get(eid) == player_id
                   for eid in state.board.vertices[vertex_id].edge_ids):
            return False
    return True


def can_build_city(state: GameState, player_id: int, vertex_id: int) -> bool:
    if len(state.players[player_id].cities) >= STARTING_CITIES:
        return False
    return vertex_id in state.players[player_id].settlements


def has_resources(hand: dict[Resource, int], cost: dict[Resource, int]) -> bool:
    return all(hand.get(r, 0) >= amt for r, amt in cost.items())


def pay(state: GameState, player_id: int, cost: dict[Resource, int]) -> None:
    player = state.players[player_id]
    for r, amt in cost.items():
        player.resources[r] -= amt
        state.bank[r] += amt
    _public_spend(state, player_id, cost)  # building costs are public


# --------------------------------------------------------------------------
# Public hand-estimate bookkeeping (card counting -- see GameState field doc)
# --------------------------------------------------------------------------

def _public_gain(state: GameState, pid: int, resource: Resource, amt: float) -> None:
    state.public_resource_estimates[pid][resource] += amt


def _public_spend(state: GameState, pid: int, cost: dict[Resource, int]) -> None:
    """Publicly-identified loss (build cost, trade give, ...): subtract what
    the estimate can cover; any shortfall must have been paid from unknown-
    identity cards, which the derived unknown mass absorbs automatically."""
    est = state.public_resource_estimates[pid]
    for r, amt in cost.items():
        est[r] = max(0.0, est[r] - amt)
    _public_clamp(state, pid)


def _public_clamp(state: GameState, pid: int) -> None:
    """Re-impose sum(estimates) <= hand_size. Expectation updates for
    hidden-identity events can leave the estimate over-claiming after a later
    exactly-known spend (e.g. we credited a thief 0.5 expected sheep but the
    stolen card was really wood, which they then visibly spent)."""
    est = state.public_resource_estimates[pid]
    total = sum(est.values())
    hand = state.players[pid].hand_size()
    if total > hand:
        scale = 0.0 if total <= 0 else hand / total
        for r in est:
            est[r] *= scale


def _public_unidentified_loss(state: GameState, pid: int, k: int, hand_before: int) -> None:
    """k cards of publicly-unknown identity left the hand (discard contents,
    robber-steal victim): each card in the hand was equally likely, so the
    whole estimate scales down proportionally."""
    if hand_before <= 0:
        return
    scale = max(0.0, 1.0 - k / hand_before)
    est = state.public_resource_estimates[pid]
    for r in est:
        est[r] *= scale


def _public_steal(state: GameState, victim: int, thief: int, victim_hand_before: int) -> None:
    """One unidentified card moved victim -> thief: the thief's estimate
    gains the victim's expected per-resource distribution (the residual
    probability mass -- the victim's own unknown-identity share -- lands in
    the thief's unknown mass automatically via hand size)."""
    if victim_hand_before <= 0:
        return
    victim_est = state.public_resource_estimates[victim]
    thief_est = state.public_resource_estimates[thief]
    for r, amt in victim_est.items():
        thief_est[r] += amt / victim_hand_before
    _public_unidentified_loss(state, victim, 1, victim_hand_before)


def trade_ratio_for(state: GameState, player_id: int, resource: Resource) -> int:
    """Best bank/port exchange ratio a player currently has for `resource`."""
    player = state.players[player_id]
    owned_vertices = list(player.settlements) + list(player.cities)
    best = 4
    for vid in owned_vertices:
        v = state.board.vertices[vid]
        if v.port_generic:
            best = min(best, 3)
        elif v.port == resource:
            best = min(best, 2)
    return best


def eligible_robber_victims(state: GameState, hex_id: int, mover: int) -> list[int]:
    victims = set()
    for vid in state.board.hexes[hex_id].vertex_ids:
        owner, _ = owner_of_vertex(state, vid)
        if owner is not None and owner != mover:
            victims.add(owner)
    return sorted(victims)


# --------------------------------------------------------------------------
# Longest road / largest army
# --------------------------------------------------------------------------

def compute_longest_road_length(state: GameState, player_id: int) -> int:
    player = state.players[player_id]
    player_edges = set(player.roads)
    if not player_edges:
        return 0
    adjacency: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for eid in player_edges:
        a, b = state.board.edges[eid].vertex_ids
        adjacency[a].append((eid, b))
        adjacency[b].append((eid, a))

    opponent_vertices = set()
    for p in state.players.values():
        if p.id == player_id:
            continue
        opponent_vertices.update(p.settlements)
        opponent_vertices.update(p.cities)

    best = 0

    def dfs(vertex: int, visited: set[int]) -> None:
        nonlocal best
        best = max(best, len(visited))
        if vertex in opponent_vertices and visited:
            return
        for eid, nxt in adjacency[vertex]:
            if eid not in visited:
                visited.add(eid)
                dfs(nxt, visited)
                visited.remove(eid)

    for start in list(adjacency.keys()):
        dfs(start, set())
    return best


def recompute_longest_road(state: GameState, players: list[int] | None = None) -> None:
    """Ties favor the incumbent holder; a new player only takes the bonus by
    being the *unique* strict leader. A tie among non-holders leaves the
    bonus unclaimed, per official rules.

    `players` scopes whose road-network DFS actually gets recomputed this
    call: only a newly built settlement/city can sever an opponent's road
    network (by planting a blocking vertex), so a plain road build only ever
    needs to recompute the *acting* player's own length -- every other
    player's cached `state.road_lengths` entry is still valid and reused
    as-is. Pass None (the default) to recompute everyone, e.g. after a
    settlement is built."""
    target_players = list(state.players.keys()) if players is None else players
    for pid in target_players:
        state.road_lengths[pid] = compute_longest_road_length(state, pid)

    lengths = state.road_lengths
    candidates = {pid: l for pid, l in lengths.items() if l >= MIN_LONGEST_ROAD}

    if not candidates:
        state.longest_road_holder = None
        state.longest_road_length = 0
        return

    max_len = max(candidates.values())
    leaders = [pid for pid, l in candidates.items() if l == max_len]
    holder = state.longest_road_holder

    if holder is not None and holder in leaders:
        state.longest_road_length = max_len
    elif len(leaders) == 1:
        state.longest_road_holder = leaders[0]
        state.longest_road_length = max_len
    else:
        state.longest_road_holder = None
        state.longest_road_length = 0


def award_knight(state: GameState, player_id: int) -> None:
    player = state.players[player_id]
    player.knights_played += 1
    holder = state.largest_army_holder
    if player.knights_played >= MIN_LARGEST_ARMY:
        if holder is None or player.knights_played > state.players[holder].knights_played:
            state.largest_army_holder = player_id


def total_vp(state: GameState, player_id: int) -> int:
    player = state.players[player_id]
    vp = player.visible_vp() + player.hidden_vp()
    if state.longest_road_holder == player_id:
        vp += 2
    if state.largest_army_holder == player_id:
        vp += 2
    return vp


def check_win(state: GameState) -> None:
    """Official rule: you can only win during your own turn. Only the turn
    owner is checked here; a player who reaches 10 VP during someone else's
    turn (e.g. by inheriting Longest Road when a third player's settlement
    breaks the holder's road) claims the win at the start of their own next
    turn -- END_TURN calls this again for the incoming player."""
    pid = state.current_player
    if total_vp(state, pid) >= WINNING_VP:
        state.winner = pid
        state.phase = Phase.GAME_OVER


# --------------------------------------------------------------------------
# Dice / production
# --------------------------------------------------------------------------

def roll_two_dice(rng: random.Random) -> tuple[int, int]:
    return rng.randint(1, 6), rng.randint(1, 6)


def distribute_resources(state: GameState, total: int) -> dict[int, dict[Resource, int]]:
    from env.board import HEX_TO_RESOURCE

    gains = {pid: empty_hand() for pid in state.players}
    for hx in state.board.hexes.values():
        if hx.number != total or hx.id == state.board.robber_hex:
            continue
        resource = HEX_TO_RESOURCE[hx.terrain]
        for vid in hx.vertex_ids:
            owner, kind = owner_of_vertex(state, vid)
            if owner is None:
                continue
            gains[owner][resource] += 2 if kind == "city" else 1

    # Official shortage rule: if the supply can't cover everyone's production
    # of a resource, nobody receives it -- unless only one player is owed it,
    # in which case that player takes whatever remains.
    for resource in Resource:
        requested = sum(g[resource] for g in gains.values())
        if requested > state.bank[resource]:
            owed = [pid for pid, g in gains.items() if g[resource] > 0]
            for pid, g in gains.items():
                g[resource] = state.bank[resource] if len(owed) == 1 and pid in owed else 0

    for pid, g in gains.items():
        for resource, amt in g.items():
            if amt:
                state.players[pid].resources[resource] += amt
                state.bank[resource] -= amt
                _public_gain(state, pid, resource, amt)  # dice production is public
    return gains


# --------------------------------------------------------------------------
# Acting player (handles sub-protocols that aren't the turn owner)
# --------------------------------------------------------------------------

def acting_player(state: GameState) -> int:
    if state.phase == Phase.DISCARD and state.players_to_discard:
        return state.players_to_discard[0]
    if state.phase == Phase.TRADE_RESPONSE:
        if state.trade_counter_context is not None:
            # the counter's *target* (the original proposer) must accept/reject
            # it -- trade_counter_context.proposer is the counter-offerer, who
            # already acted to create it.
            return state.pending_trade.proposer
        if state.trade_targets_remaining:
            return state.trade_targets_remaining[0]
        return state.pending_trade.proposer
    return state.current_player


# --------------------------------------------------------------------------
# Legal actions
# --------------------------------------------------------------------------

def _resource_combinations(total: int, hand: dict[Resource, int], cap: int = 300):
    resources = list(Resource)

    def rec(idx, remaining, current):
        if remaining == 0:
            yield dict(current)
            return
        if idx == len(resources):
            return
        r = resources[idx]
        max_take = min(hand.get(r, 0), remaining)
        for take in range(max_take + 1):
            current[r] = take
            yield from rec(idx + 1, remaining - take, current)
        current.pop(r, None)

    count = 0
    for combo in rec(0, total, {}):
        yield combo
        count += 1
        if count >= cap:
            return


def legal_actions(state: GameState) -> list[Action]:
    if state.phase == Phase.GAME_OVER:
        return []

    actor = acting_player(state)
    actions: list[Action] = []

    if state.phase == Phase.SETUP_SETTLEMENT:
        for vid in state.board.vertices:
            if can_build_settlement(state, actor, vid, require_road=False):
                actions.append(Action(ActionType.BUILD_SETTLEMENT, {"vertex_id": vid}))
        return actions

    if state.phase == Phase.SETUP_ROAD:
        v = state.just_placed_settlement_vertex
        for eid in state.board.vertices[v].edge_ids:
            if road_edge_free(state, eid):
                actions.append(Action(ActionType.BUILD_ROAD, {"edge_id": eid}))
        return actions

    if state.phase == Phase.ROLL:
        # Official rule: development cards may be played at any time during
        # your turn, "even before you roll the dice" (e.g. a knight to move
        # the robber off your own hex before production). ROLL_DICE stays
        # first so simple agents that take actions[0] in ROLL keep rolling.
        if state.free_roads_remaining > 0:  # pre-roll Road Building in progress
            return _free_road_actions(state, actor)
        return [Action(ActionType.ROLL_DICE)] + _dev_card_play_actions(state, actor)

    if state.phase == Phase.DISCARD:
        amount = state.discard_amounts[actor]
        hand = state.players[actor].resources
        for combo in _resource_combinations(amount, hand, cap=100):
            actions.append(Action(ActionType.DISCARD, {"cards": combo}))
        return actions

    if state.phase == Phase.MOVE_ROBBER:
        for hx in state.board.hexes.values():
            if hx.id == state.board.robber_hex:
                continue
            victims = eligible_robber_victims(state, hx.id, actor)
            if victims:
                for v in victims:
                    actions.append(Action(ActionType.MOVE_ROBBER, {"hex_id": hx.id, "victim": v}))
            else:
                actions.append(Action(ActionType.MOVE_ROBBER, {"hex_id": hx.id, "victim": None}))
        return actions

    if state.phase == Phase.TRADE_RESPONSE:
        if state.trade_counter_context is not None:
            # original proposer decides on the counter; accepting is only
            # possible if they can pay what the counter asks for
            ctx = state.trade_counter_context
            if has_resources(state.players[actor].resources, ctx.want):
                actions.append(Action(ActionType.ACCEPT_TRADE, {}))
            actions.append(Action(ActionType.REJECT_TRADE, {}))
            return actions
        if state.trade_targets_remaining:
            offer = state.pending_trade
            hand = state.players[actor].resources
            if has_resources(hand, offer.want):  # can only accept what you can pay
                actions.append(Action(ActionType.ACCEPT_TRADE, {}))
            actions.append(Action(ActionType.REJECT_TRADE, {}))
            if sum(hand.values()) > 0:
                actions.append(trade_template(state, actor, ActionType.COUNTER_TRADE))
            return actions
        # proposer confirms one of the accepted offers, or cancels
        for target in state.trade_accepted:
            actions.append(Action(ActionType.CONFIRM_TRADE, {"target": target}))
        actions.append(Action(ActionType.CANCEL_TRADE, {}))
        return actions

    # ROLL/MAIN shared: dev-card plays + MAIN-only actions
    player = state.players[actor]
    hand = player.resources

    if state.phase == Phase.MAIN:
        if state.free_roads_remaining > 0:
            # Road Building's roads are placed immediately; step() clears
            # free_roads_remaining as soon as no placement is possible, so
            # this list is never empty and the turn can't lock.
            return _free_road_actions(state, actor)
        else:
            if has_resources(hand, BUILDING_COSTS["road"]):
                for eid in state.board.edges:
                    if can_build_road(state, actor, eid):
                        actions.append(Action(ActionType.BUILD_ROAD, {"edge_id": eid}))
            if has_resources(hand, BUILDING_COSTS["settlement"]):
                for vid in state.board.vertices:
                    if can_build_settlement(state, actor, vid):
                        actions.append(Action(ActionType.BUILD_SETTLEMENT, {"vertex_id": vid}))
            if has_resources(hand, BUILDING_COSTS["city"]):
                for vid in state.board.vertices:
                    if can_build_city(state, actor, vid):
                        actions.append(Action(ActionType.BUILD_CITY, {"vertex_id": vid}))
            if state.allow_dev_cards and has_resources(hand, BUILDING_COSTS["dev_card"]) and state.dev_card_deck:
                actions.append(Action(ActionType.BUY_DEV_CARD))

            if state.allow_trading:
                for give_r in Resource:
                    ratio = trade_ratio_for(state, actor, give_r)
                    if hand.get(give_r, 0) >= ratio:
                        for want_r in Resource:
                            if want_r != give_r and state.bank[want_r] >= 1:  # supply must have it
                                actions.append(Action(ActionType.MARITIME_TRADE,
                                                       {"give": give_r, "receive": want_r}))

                if state.trades_proposed_this_turn < MAX_TRADE_PROPOSALS_PER_TURN \
                        and sum(hand.values()) > 0:
                    actions.append(trade_template(state, actor, ActionType.PROPOSE_TRADE))

        actions.extend(_dev_card_play_actions(state, actor))
        actions.append(Action(ActionType.END_TURN))

    return actions


def _free_road_actions(state: GameState, actor: int) -> list[Action]:
    return [Action(ActionType.BUILD_ROAD, {"edge_id": eid, "free": True})
            for eid in state.board.edges if can_build_road(state, actor, eid)]


def _dev_card_play_actions(state: GameState, actor: int) -> list[Action]:
    """Playable development cards for the turn owner (ROLL or MAIN phase):
    at most one per turn, never one bought this turn, VP cards never
    'played' (they count automatically)."""
    actions: list[Action] = []
    player = state.players[actor]
    if not state.allow_dev_cards or player.played_dev_card_this_turn:
        return actions
    avail = lambda c: player.dev_cards.get(c, 0) - player.dev_cards_bought_this_turn.get(c, 0)
    if avail(DevCard.KNIGHT) > 0:
        for hx in state.board.hexes.values():
            if hx.id == state.board.robber_hex:
                continue
            victims = eligible_robber_victims(state, hx.id, actor)
            if victims:
                for v in victims:
                    actions.append(Action(ActionType.PLAY_KNIGHT, {"hex_id": hx.id, "victim": v}))
            else:
                actions.append(Action(ActionType.PLAY_KNIGHT, {"hex_id": hx.id, "victim": None}))
    if avail(DevCard.ROAD_BUILDING) > 0 and state.free_roads_remaining == 0:
        actions.append(Action(ActionType.PLAY_ROAD_BUILDING))
    if avail(DevCard.YEAR_OF_PLENTY) > 0:
        for combo in combinations_with_replacement(list(Resource), 2):
            needed = Counter(combo)
            if all(state.bank[r] >= n for r, n in needed.items()):
                actions.append(Action(ActionType.PLAY_YEAR_OF_PLENTY, {"resources": list(combo)}))
    if avail(DevCard.MONOPOLY) > 0:
        for r in Resource:
            actions.append(Action(ActionType.PLAY_MONOPOLY, {"resource": r}))
    return actions


def _settle_free_roads(state: GameState, actor: int) -> None:
    """Forfeit any remaining Road Building roads that can no longer be placed
    (no road pieces left, or no legal edge) -- otherwise the turn would be
    stuck offering only END_TURN (MAIN) or nothing at all (ROLL)."""
    if state.free_roads_remaining > 0 and not any(
            can_build_road(state, actor, eid) for eid in state.board.edges):
        state.free_roads_remaining = 0


# --------------------------------------------------------------------------
# Step
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# Structured domestic trades
# --------------------------------------------------------------------------

TRADE_TYPES = (ActionType.PROPOSE_TRADE, ActionType.COUNTER_TRADE)
ALL_OPPONENTS = "all"


def trade_template(state: GameState, actor: int, kind: ActionType) -> Action:
    """The single legal-list entry standing in for every concrete bundle of
    `kind`. Carries what a bundle builder needs: the actor's own hand (its
    own private info) and, for proposals, who it may be addressed to."""
    params = {"template": True, "actor": actor,
              "hand": {r: k for r, k in state.players[actor].resources.items()}}
    if kind == ActionType.PROPOSE_TRADE:
        params["target_options"] = opponents_in_turn_order(actor) + [ALL_OPPONENTS]
    return Action(kind, params)


def is_template(action: Action) -> bool:
    return bool(action.params.get("template"))


def opponents_in_turn_order(pid: int) -> list[int]:
    return [(pid + k) % NUM_PLAYERS for k in range(1, NUM_PLAYERS)]


def trade_bundle_ok(hand: dict[Resource, int], give: dict, want: dict) -> bool:
    """1..MAX_TRADE_CARDS_PER_SIDE cards each side, non-negative integer
    counts, no resource on both sides (no like-for-like swaps or gifts), and
    the giver actually holds `give`."""
    if not all(isinstance(r, Resource) and isinstance(k, int) and not isinstance(k, bool) and k >= 0
               for side in (give, want) for r, k in side.items()):
        return False
    g = {r: k for r, k in give.items() if k}
    w = {r: k for r, k in want.items() if k}
    if not (1 <= sum(g.values()) <= MAX_TRADE_CARDS_PER_SIDE):
        return False
    if not (1 <= sum(w.values()) <= MAX_TRADE_CARDS_PER_SIDE):
        return False
    if set(g) & set(w):
        return False
    return has_resources(hand, g)


def make_trade(kind: ActionType, give: dict, want: dict, actor: int | None = None,
               target: int | str | None = None) -> Action:
    """Build a concrete trade Action. PROPOSE_TRADE needs `actor` and a
    `target` (an opponent id or ALL_OPPONENTS); COUNTER_TRADE has none (it
    always goes back to the proposer)."""
    params = {"give": {r: k for r, k in give.items() if k},
              "want": {r: k for r, k in want.items() if k}}
    if kind == ActionType.PROPOSE_TRADE:
        params["targets"] = opponents_in_turn_order(actor) if target == ALL_OPPONENTS else [target]
    return Action(kind, params)


def _concrete_trade_legal(state: GameState, action: Action) -> bool:
    if state.phase == Phase.GAME_OVER:
        return False
    actor = acting_player(state)
    p = action.params
    if not isinstance(p.get("give"), dict) or not isinstance(p.get("want"), dict):
        return False
    hand = state.players[actor].resources
    if action.type == ActionType.PROPOSE_TRADE:
        if state.phase != Phase.MAIN or state.free_roads_remaining > 0 or not state.allow_trading:
            return False
        if state.trades_proposed_this_turn >= MAX_TRADE_PROPOSALS_PER_TURN:
            return False
        targets = p.get("targets")
        opps = opponents_in_turn_order(actor)
        if not isinstance(targets, list) or not (targets == opps or (len(targets) == 1 and targets[0] in opps)):
            return False
        return trade_bundle_ok(hand, p["give"], p["want"])
    # COUNTER_TRADE: a responder still owing a reply to the original offer
    if state.phase != Phase.TRADE_RESPONSE or state.trade_counter_context is not None \
            or not state.trade_targets_remaining:
        return False
    return trade_bundle_ok(hand, p["give"], p["want"])


def is_legal_action(state: GameState, action: Action) -> bool:
    """Single legality oracle: concrete trades are validated structurally,
    everything else must appear in `legal_actions(state)`. Templates are
    never steppable."""
    if is_template(action):
        return False
    if action.type in TRADE_TYPES:
        return _concrete_trade_legal(state, action)
    return action in legal_actions(state)


def random_trade(template: Action, rng: random.Random, actor: int | None = None) -> Action:
    """A random legal bundle for `template` (random agent, fuzzing)."""
    hand = template.params["hand"]
    pool = [r for r, k in hand.items() for _ in range(min(k, MAX_TRADE_CARDS_PER_SIDE))]
    n_give = rng.randint(1, min(MAX_TRADE_CARDS_PER_SIDE, len(pool)))
    give: dict[Resource, int] = {}
    for r in rng.sample(pool, n_give):
        give[r] = give.get(r, 0) + 1
    rest = [r for r in Resource if r not in give]
    want: dict[Resource, int] = {}
    for _ in range(rng.randint(1, MAX_TRADE_CARDS_PER_SIDE)):
        r = rng.choice(rest)
        want[r] = want.get(r, 0) + 1
    target = rng.choice(template.params["target_options"]) \
        if template.type == ActionType.PROPOSE_TRADE else None
    return make_trade(template.type, give, want, actor=actor, target=target)


def _begin_discard_or_robber(state: GameState) -> None:
    to_discard = [pid for pid, p in state.players.items() if p.hand_size() > 7]
    if to_discard:
        state.players_to_discard = to_discard
        state.discard_amounts = {pid: state.players[pid].hand_size() // 2 for pid in to_discard}
        state.phase = Phase.DISCARD
    else:
        state.phase = Phase.MOVE_ROBBER


def _apply_move_robber(state: GameState, mover: int, hex_id: int, victim: int | None,
                        rng: random.Random) -> None:
    state.board.robber_hex = hex_id
    if victim is not None:
        vhand = state.players[victim].resources
        pool = [r for r, cnt in vhand.items() for _ in range(cnt)]
        if pool:
            # Public bookkeeping first (it must see the pre-steal hand size);
            # the stolen card's identity is hidden from everyone but the two
            # parties, so only the expected distribution moves.
            _public_steal(state, victim, mover, len(pool))
            stolen = rng.choice(pool)
            vhand[stolen] -= 1
            state.players[mover].resources[stolen] += 1


def _execute_trade(state: GameState, giver: int, receiver: int,
                    give: dict[Resource, int], want: dict[Resource, int]) -> None:
    for r, amt in give.items():
        state.players[giver].resources[r] -= amt
        state.players[receiver].resources[r] += amt
        _public_gain(state, receiver, r, amt)  # executed trades are public
    for r, amt in want.items():
        state.players[receiver].resources[r] -= amt
        state.players[giver].resources[r] += amt
        _public_gain(state, giver, r, amt)
    _public_spend(state, giver, give)
    _public_spend(state, receiver, want)


def step(state: GameState, action: Action, rng: random.Random | None = None) -> None:
    if rng is None:
        rng = random.Random()
    if is_template(action):
        raise ValueError(f"{action.type.value} template is not steppable -- build a concrete "
                         "bundle (engine.make_trade) instead")
    actor = acting_player(state)
    t = action.type
    p = action.params

    if state.phase == Phase.SETUP_SETTLEMENT:
        vid = p["vertex_id"]
        state.players[actor].settlements.append(vid)
        state.vertex_owner[vid] = (actor, "settlement")
        state.just_placed_settlement_vertex = vid
        state.phase = Phase.SETUP_ROAD
        return

    if state.phase == Phase.SETUP_ROAD:
        eid = p["edge_id"]
        state.players[actor].roads.append(eid)
        state.road_owner[eid] = actor
        idx = state.setup_order_index
        if idx >= NUM_PLAYERS:  # second pass: grant starting resources
            v = state.just_placed_settlement_vertex
            for hx_id in state.board.vertices[v].hex_ids:
                hx = state.board.hexes[hx_id]
                if hx.number is not None:
                    from env.board import HEX_TO_RESOURCE
                    resource = HEX_TO_RESOURCE[hx.terrain]
                    state.players[actor].resources[resource] += 1
                    state.bank[resource] -= 1  # cards come from the supply (19 of each)
                    _public_gain(state, actor, resource, 1)  # setup grants are public
        recompute_longest_road(state)
        state.setup_order_index += 1
        if state.setup_order_index >= len(SETUP_ORDER):
            state.phase = Phase.ROLL
            state.current_player = 0
            state.turn_number = 1
        else:
            state.current_player = SETUP_ORDER[state.setup_order_index]
            state.phase = Phase.SETUP_SETTLEMENT
        return

    if t == ActionType.ROLL_DICE:
        d1, d2 = roll_two_dice(rng)
        state.dice_roll = (d1, d2)
        total = d1 + d2
        state.players[actor].has_rolled_this_turn = True
        if total == 7:
            _begin_discard_or_robber(state)
        else:
            distribute_resources(state, total)
            state.phase = Phase.MAIN
        return

    if t == ActionType.DISCARD:
        cards = p["cards"]
        hand_before = state.players[actor].hand_size()
        for r, amt in cards.items():
            state.players[actor].resources[r] -= amt
            state.bank[r] += amt
        # Discard *count* is public, contents are not.
        _public_unidentified_loss(state, actor, sum(cards.values()), hand_before)
        state.players_to_discard.pop(0)
        if not state.players_to_discard:
            state.phase = Phase.MOVE_ROBBER
        return

    if t == ActionType.MOVE_ROBBER:
        _apply_move_robber(state, actor, p["hex_id"], p["victim"], rng)
        state.phase = Phase.MAIN
        check_win(state)
        return

    if t == ActionType.BUILD_ROAD:
        eid = p["edge_id"]
        free = state.free_roads_remaining > 0
        if free:
            state.free_roads_remaining -= 1
        else:
            pay(state, actor, BUILDING_COSTS["road"])
        state.players[actor].roads.append(eid)
        state.road_owner[eid] = actor
        # A standalone road build (no settlement placed in this same step) can
        # only ever extend the *acting* player's own network -- it can't sever
        # anyone else's, since severing requires a new blocking vertex, which
        # only a settlement/city places. Safe to recompute just this player.
        recompute_longest_road(state, players=[actor])
        if free:
            _settle_free_roads(state, actor)
        check_win(state)
        return

    if t == ActionType.BUILD_SETTLEMENT:
        vid = p["vertex_id"]
        pay(state, actor, BUILDING_COSTS["settlement"])
        state.players[actor].settlements.append(vid)
        state.vertex_owner[vid] = (actor, "settlement")
        recompute_longest_road(state)
        check_win(state)
        return

    if t == ActionType.BUILD_CITY:
        vid = p["vertex_id"]
        pay(state, actor, BUILDING_COSTS["city"])
        state.players[actor].settlements.remove(vid)
        state.players[actor].cities.append(vid)
        state.vertex_owner[vid] = (actor, "city")
        check_win(state)
        return

    if t == ActionType.BUY_DEV_CARD:
        pay(state, actor, BUILDING_COSTS["dev_card"])
        card = state.dev_card_deck.pop()
        state.players[actor].dev_cards[card] += 1
        state.players[actor].dev_cards_bought_this_turn[card] += 1
        state.players[actor].last_dev_purchase_turn = state.turn_number
        check_win(state)
        return

    if t == ActionType.PLAY_KNIGHT:
        player = state.players[actor]
        player.dev_cards[DevCard.KNIGHT] -= 1
        player.dev_cards_played[DevCard.KNIGHT] += 1
        player.played_dev_card_this_turn = True
        award_knight(state, actor)
        _apply_move_robber(state, actor, p["hex_id"], p["victim"], rng)
        check_win(state)
        return

    if t == ActionType.PLAY_ROAD_BUILDING:
        player = state.players[actor]
        player.dev_cards[DevCard.ROAD_BUILDING] -= 1
        player.dev_cards_played[DevCard.ROAD_BUILDING] += 1
        player.played_dev_card_this_turn = True
        state.free_roads_remaining = 2
        _settle_free_roads(state, actor)
        return

    if t == ActionType.PLAY_YEAR_OF_PLENTY:
        player = state.players[actor]
        player.dev_cards[DevCard.YEAR_OF_PLENTY] -= 1
        player.dev_cards_played[DevCard.YEAR_OF_PLENTY] += 1
        player.played_dev_card_this_turn = True
        for r in p["resources"]:
            player.resources[r] += 1
            state.bank[r] -= 1
            _public_gain(state, actor, r, 1)  # announced publicly
        return

    if t == ActionType.PLAY_MONOPOLY:
        player = state.players[actor]
        player.dev_cards[DevCard.MONOPOLY] -= 1
        player.dev_cards_played[DevCard.MONOPOLY] += 1
        player.played_dev_card_this_turn = True
        r = p["resource"]
        for other_id, other in state.players.items():
            if other_id == actor:
                continue
            taken = other.resources[r]
            other.resources[r] = 0
            player.resources[r] += taken
            # Monopoly is fully public: each victim visibly hands over their
            # entire holding of r, so their r-estimate collapses to exactly 0
            # (this also retroactively reveals how many of their unknown
            # cards were r -- the derived unknown mass absorbs it) and the
            # monopolist is credited the exact amount.
            state.public_resource_estimates[other_id][r] = 0.0
            _public_clamp(state, other_id)
            _public_gain(state, actor, r, taken)
        return

    if t == ActionType.MARITIME_TRADE:
        give_r, want_r = p["give"], p["receive"]
        ratio = trade_ratio_for(state, actor, give_r)
        state.players[actor].resources[give_r] -= ratio
        state.bank[give_r] += ratio
        state.players[actor].resources[want_r] += 1
        state.bank[want_r] -= 1
        # Bank trades are public. Gain before spend: _public_spend ends with
        # the sum<=hand clamp, which must run after ALL of this event's
        # updates -- a gain applied post-clamp could push the estimate back
        # above the (already final) hand size.
        _public_gain(state, actor, want_r, 1)
        _public_spend(state, actor, {give_r: ratio})
        return

    if t == ActionType.PROPOSE_TRADE:
        state.trades_proposed_this_turn += 1
        state.last_trade_offer[actor] = (dict(p["give"]), dict(p["want"]), state.turn_number)
        state.pending_trade = TradeOffer(proposer=actor, give=dict(p["give"]), want=dict(p["want"]),
                                          targets=list(p["targets"]))
        state.trade_targets_remaining = list(p["targets"])
        state.trade_accepted = []
        state.phase = Phase.TRADE_RESPONSE
        return

    if t == ActionType.ACCEPT_TRADE:
        if state.trade_counter_context is not None:
            ctx = state.trade_counter_context
            if has_resources(state.players[ctx.proposer].resources, ctx.give) and \
               has_resources(state.players[state.pending_trade.proposer].resources, ctx.want):
                _execute_trade(state, ctx.proposer, state.pending_trade.proposer, ctx.give, ctx.want)
            state.trade_counter_context = None
            state.pending_trade = None
            state.trade_targets_remaining = []
            state.trade_accepted = []
            state.phase = Phase.MAIN
            return
        state.trade_accepted.append(actor)
        state.trade_targets_remaining.pop(0)
        if not state.trade_targets_remaining:
            _resolve_trade_confirmation_phase(state)
        return

    if t == ActionType.REJECT_TRADE:
        if state.trade_counter_context is not None:
            state.trade_counter_context = None
            state.pending_trade = None
            state.trade_targets_remaining = []
            state.trade_accepted = []
            state.phase = Phase.MAIN
            return
        state.trade_targets_remaining.pop(0)
        if not state.trade_targets_remaining:
            _resolve_trade_confirmation_phase(state)
        return

    if t == ActionType.COUNTER_TRADE:
        offer = state.pending_trade
        state.last_trade_offer[actor] = (dict(p["give"]), dict(p["want"]), state.turn_number)
        state.trade_counter_context = TradeOffer(proposer=actor, give=p["give"], want=p["want"],
                                                  targets=[offer.proposer], round=offer.round + 1)
        state.trade_targets_remaining.pop(0)
        return

    if t == ActionType.CONFIRM_TRADE:
        target = p["target"]
        offer = state.pending_trade
        if has_resources(state.players[offer.proposer].resources, offer.give) and \
           has_resources(state.players[target].resources, offer.want):
            _execute_trade(state, offer.proposer, target, offer.give, offer.want)
        state.pending_trade = None
        state.trade_accepted = []
        state.phase = Phase.MAIN
        return

    if t == ActionType.CANCEL_TRADE:
        state.pending_trade = None
        state.trade_accepted = []
        state.phase = Phase.MAIN
        return

    if t == ActionType.END_TURN:
        player = state.players[actor]
        player.played_dev_card_this_turn = False
        player.has_rolled_this_turn = False
        player.dev_cards_bought_this_turn = {c: 0 for c in DevCard}
        state.free_roads_remaining = 0
        state.trades_proposed_this_turn = 0
        state.current_player = (actor + 1) % NUM_PLAYERS
        state.turn_number += 1
        state.phase = Phase.ROLL
        # a player who reached 10 VP during another player's turn claims the
        # win now, at the start of their own turn (see check_win)
        check_win(state)
        return

    raise ValueError(f"Unhandled action type {t}")


def _resolve_trade_confirmation_phase(state: GameState) -> None:
    if not state.trade_accepted:
        state.pending_trade = None
        state.phase = Phase.MAIN
    # else: stays in TRADE_RESPONSE; acting_player() now returns the proposer
    # to CONFIRM_TRADE or CANCEL_TRADE (empty trade_targets_remaining + non-empty accepted)


class CatanEngine:
    """Thin stateful convenience wrapper around the functional engine above."""

    def __init__(self, randomize_board: bool = True, seed: int | None = None,
                 allow_trading: bool = True, allow_dev_cards: bool = True):
        self.allow_trading = allow_trading
        self.allow_dev_cards = allow_dev_cards
        self.rng = random.Random(seed)
        self.state = new_game(randomize_board=randomize_board, seed=seed,
                               allow_trading=allow_trading, allow_dev_cards=allow_dev_cards)

    def reset(self, randomize_board: bool = True, seed: int | None = None) -> GameState:
        self.rng = random.Random(seed)
        self.state = new_game(randomize_board=randomize_board, seed=seed,
                               allow_trading=self.allow_trading, allow_dev_cards=self.allow_dev_cards)
        return self.state

    def acting_player(self) -> int:
        return acting_player(self.state)

    def legal_actions(self) -> list[Action]:
        return legal_actions(self.state)

    def step(self, action: Action, validate: bool = True) -> None:
        """`validate` (default on) rejects illegal actions with ValueError
        before anything is mutated; the module-level `step` stays unchecked
        for hot simulation loops that only ever feed it legal actions."""
        if validate and not is_legal_action(self.state, action):
            raise ValueError(f"illegal action {action!r} in phase {self.state.phase.name}")
        step(self.state, action, rng=self.rng)

    @property
    def done(self) -> bool:
        return self.state.phase == Phase.GAME_OVER
