"""Public-event belief features (roadmap Phase 5) and observer-relative flat
encoding (audit F-26)."""
import numpy as np
import pytest

from env.engine import legal_actions
from env.pettingzoo_env import NUM_VERTICES, build_observation
from env.public_beliefs import expected_dev_cards, last_offer, turns_since_dev_purchase
from env.state import DevCard, Phase, new_game
from training.model import flatten_observation


def _state():
    s = new_game(seed=21)
    s.phase, s.current_player, s.turn_number = Phase.MAIN, 0, 30
    return s


def test_expected_dev_cards_from_public_counts_and_own_hand():
    s = _state()
    # observer 0 holds 2 VP cards; 6 knights have been played publicly
    s.players[0].dev_cards[DevCard.VICTORY_POINT] = 2
    s.players[1].dev_cards_played[DevCard.KNIGHT] = 6
    s.players[1].knights_played = 6
    # opponent 2 holds 3 unplayed cards (true identities irrelevant to the belief)
    s.players[2].dev_cards[DevCard.MONOPOLY] = 3
    s.dev_card_deck = s.dev_card_deck[:10]
    exp = expected_dev_cards(s, 0, 2)
    # unknown pool: VP 5-2=3, knights 14-6=8, ...; pool size = deck 10 + opponents' held 3 = 13
    assert sum(exp.values()) == pytest.approx(3.0)
    unknown_total = 25 - 6 - 2  # minus played, minus observer's own cards
    assert exp[DevCard.VICTORY_POINT] == pytest.approx(3 * 3 / unknown_total)
    assert exp[DevCard.KNIGHT] == pytest.approx(3 * 8 / unknown_total)
    # the belief must not depend on what opponent 2 actually holds
    s.players[2].dev_cards[DevCard.MONOPOLY] = 0
    s.players[2].dev_cards[DevCard.VICTORY_POINT] = 3
    assert expected_dev_cards(s, 0, 2) == pytest.approx(exp)
    # own hand is known exactly
    assert expected_dev_cards(s, 0, 0)[DevCard.VICTORY_POINT] == 2


def test_purchase_age_and_last_offer():
    s = _state()
    assert turns_since_dev_purchase(s, 1) == 1.0
    s.players[1].last_dev_purchase_turn = 26
    assert turns_since_dev_purchase(s, 1) == pytest.approx(4 / 40)
    from env.board import Resource
    s.last_trade_offer[3] = ({Resource.WOOD: 2}, {Resource.ORE: 1}, 28)
    give, want, age = last_offer(s, 3)
    assert give == {Resource.WOOD: 2} and want == {Resource.ORE: 1} and age == pytest.approx(2 / 40)


def test_flat_encoding_is_observer_relative():
    s = _state()
    v = 10
    s.vertex_owner[v] = (2, "settlement")
    s.players[2].settlements.append(v)
    as_owner = flatten_observation(build_observation(s, 2, [], False))
    as_other = flatten_observation(build_observation(s, 3, [], False))
    owner_block = slice(19 * 3, 19 * 3 + NUM_VERTICES)  # after hex terrain/number/robber
    assert as_owner[owner_block][v] == pytest.approx(0.25)   # "mine"
    assert as_other[owner_block][v] == pytest.approx(1.0)    # seat 2 is 3rd after seat 3
    obs = build_observation(s, 0, legal_actions(s), True)
    assert obs["bank"].tolist() == [s.bank[r] for r in s.bank] and obs["observer"][0] == 0
