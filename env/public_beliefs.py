"""Belief features computable from public information plus the observer's own
hand (roadmap Phase 5, "public-event history / belief features").

Development cards: the deck's composition is known (14 knights, 5 VP, 2 each
of the progress cards), every play is announced, and every player sees their
own hand. So the cards the observer cannot see -- the deck plus opponents'
unplayed cards -- have a known composition, and an opponent holding `h`
unplayed cards holds `h * unknown[c] / pool` cards of type `c` in
expectation. This is the right prior for "how close is that player to 10 VP"
(hidden VP cards) and "how likely is a knight next turn".

Trade offers: a player's most recent proposal/counter reveals what they need.

Everything here must depend only on public state + the observer's hand --
audit/tests/test_obs_leakage.py checks that by re-dealing hidden cards.
"""
from __future__ import annotations

from env.state import DevCard, GameState, NUM_PLAYERS, STANDARD_DEV_CARD_COUNTS

AGE_CAP = 40  # turns; ages are reported as min(age, AGE_CAP) / AGE_CAP


def unknown_dev_composition(state: GameState, observer: int) -> dict[DevCard, int]:
    """Counts of each card type among cards the observer cannot see (deck +
    opponents' unplayed cards)."""
    me = state.players[observer]
    unknown = {}
    for c, total in STANDARD_DEV_CARD_COUNTS.items():
        played = sum(p.dev_cards_played[c] for p in state.players.values())
        unknown[c] = total - played - me.dev_cards[c]
    return unknown


def expected_dev_cards(state: GameState, observer: int, pid: int) -> dict[DevCard, float]:
    """Expected unplayed cards of each type held by `pid` from `observer`'s
    point of view (the observer's own hand is known exactly)."""
    p = state.players[pid]
    if pid == observer:
        return {c: float(k) for c, k in p.dev_cards.items()}
    unknown = unknown_dev_composition(state, observer)
    pool = sum(unknown.values())
    held = p.total_dev_cards()
    if pool <= 0 or held == 0:
        return {c: 0.0 for c in DevCard}
    return {c: held * unknown[c] / pool for c in DevCard}


def turns_since_dev_purchase(state: GameState, pid: int) -> float:
    """Normalized age of `pid`'s last dev-card purchase (1.0 = never/long ago)."""
    last = state.players[pid].last_dev_purchase_turn
    if last < 0:
        return 1.0
    return min(state.turn_number - last, AGE_CAP) / AGE_CAP


def last_offer(state: GameState, pid: int):
    """(give dict, want dict, normalized age) of `pid`'s latest public trade
    offer; empty dicts and age 1.0 if none."""
    rec = state.last_trade_offer.get(pid)
    if rec is None:
        return {}, {}, 1.0
    give, want, turn = rec
    return give, want, min(state.turn_number - turn, AGE_CAP) / AGE_CAP


def seat_order(observer: int) -> list[int]:
    """Absolute seats in observer-relative order: me, then turn order."""
    return [(observer + k) % NUM_PLAYERS for k in range(NUM_PLAYERS)]


def expected_dev_cards_all(state: GameState, observer: int) -> dict[int, dict[DevCard, float]]:
    """`expected_dev_cards(state, observer, pid)` for every seat, computing the
    unknown-card composition once (identical arithmetic, ~4x less work)."""
    unknown = unknown_dev_composition(state, observer)
    pool = sum(unknown.values())
    out = {}
    for pid, p in state.players.items():
        if pid == observer:
            out[pid] = {c: float(k) for c, k in p.dev_cards.items()}
            continue
        held = p.total_dev_cards()
        if pool <= 0 or held == 0:
            out[pid] = {c: 0.0 for c in DevCard}
        else:
            out[pid] = {c: held * unknown[c] / pool for c in DevCard}
    return out
