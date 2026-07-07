"""Game state: dev cards, per-player state, and the full world state."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto

from env.board import Board, Resource, generate_board

NUM_PLAYERS = 4
STARTING_SETTLEMENTS = 5
STARTING_CITIES = 4
STARTING_ROADS = 15
WINNING_VP = 10
MIN_LONGEST_ROAD = 5
MIN_LARGEST_ARMY = 3
MAX_TRADE_PROPOSALS_PER_TURN = 2  # engine-level negotiation-efficiency bound; see engine.py legal_actions


class DevCard(Enum):
    KNIGHT = "knight"
    ROAD_BUILDING = "road_building"
    YEAR_OF_PLENTY = "year_of_plenty"
    MONOPOLY = "monopoly"
    VICTORY_POINT = "victory_point"


STANDARD_DEV_CARD_COUNTS = {
    DevCard.KNIGHT: 14,
    DevCard.ROAD_BUILDING: 2,
    DevCard.YEAR_OF_PLENTY: 2,
    DevCard.MONOPOLY: 2,
    DevCard.VICTORY_POINT: 5,
}

BUILDING_COSTS = {
    "road": {Resource.WOOD: 1, Resource.BRICK: 1},
    "settlement": {Resource.WOOD: 1, Resource.BRICK: 1, Resource.SHEEP: 1, Resource.WHEAT: 1},
    "city": {Resource.WHEAT: 2, Resource.ORE: 3},
    "dev_card": {Resource.SHEEP: 1, Resource.WHEAT: 1, Resource.ORE: 1},
}


class Phase(Enum):
    SETUP_SETTLEMENT = auto()
    SETUP_ROAD = auto()
    ROLL = auto()
    DISCARD = auto()
    MOVE_ROBBER = auto()
    MAIN = auto()
    TRADE_RESPONSE = auto()
    GAME_OVER = auto()


def empty_hand() -> dict[Resource, int]:
    return {r: 0 for r in Resource}


def empty_dev_hand() -> dict[DevCard, int]:
    return {c: 0 for c in DevCard}


@dataclass
class PlayerState:
    id: int
    resources: dict[Resource, int] = field(default_factory=empty_hand)
    dev_cards: dict[DevCard, int] = field(default_factory=empty_dev_hand)
    dev_cards_bought_this_turn: dict[DevCard, int] = field(default_factory=empty_dev_hand)
    settlements: list[int] = field(default_factory=list)  # vertex ids
    cities: list[int] = field(default_factory=list)  # vertex ids
    roads: list[int] = field(default_factory=list)  # edge ids
    knights_played: int = 0
    played_dev_card_this_turn: bool = False
    has_rolled_this_turn: bool = False

    def hand_size(self) -> int:
        return sum(self.resources.values())

    def total_dev_cards(self) -> int:
        return sum(self.dev_cards.values())

    def visible_vp(self) -> int:
        """VP from settlements/cities/road/army bonuses only (no hidden VP cards)."""
        return len(self.settlements) + 2 * len(self.cities)

    def hidden_vp(self) -> int:
        return self.dev_cards.get(DevCard.VICTORY_POINT, 0)


@dataclass
class TradeOffer:
    proposer: int
    give: dict[Resource, int]
    want: dict[Resource, int]
    targets: list[int]
    responses: dict[int, str] = field(default_factory=dict)  # player_id -> accept/reject/counter
    round: int = 0


@dataclass
class GameState:
    board: Board
    players: dict[int, PlayerState]
    bank: dict[Resource, int]
    dev_card_deck: list[DevCard]
    current_player: int
    phase: Phase
    turn_number: int = 0
    dice_roll: tuple[int, int] | None = None
    longest_road_holder: int | None = None
    longest_road_length: int = 0
    road_lengths: dict[int, int] = field(default_factory=dict)
    vertex_owner: dict[int, tuple[int, str]] = field(default_factory=dict)  # vertex_id -> (player_id, "settlement"|"city")
    road_owner: dict[int, int] = field(default_factory=dict)  # edge_id -> player_id; kept in sync by engine.step, mirrors players[*].roads
    # Publicly-inferable per-player resource estimates (card counting): every
    # resource flow except robber-steal identity and discard contents is
    # public in Catan, so an attentive player can track everyone's hand to
    # within the uncertainty those two events introduce. Maintained by
    # engine.step as expected values -- exact for public flows, proportional
    # expectation updates for hidden ones. Invariant: non-negative, and
    # sum(estimates[pid]) <= players[pid].hand_size() (the remainder is
    # "unknown-identity" mass). Exposed as observation features only when
    # the env's `public_hand_features` flag is on.
    public_resource_estimates: dict[int, dict[Resource, float]] = field(
        default_factory=lambda: {pid: {r: 0.0 for r in Resource} for pid in range(NUM_PLAYERS)})
    largest_army_holder: int | None = None
    setup_round: int = 0  # 0 = first pass, 1 = second (reverse) pass
    setup_order_index: int = 0
    just_placed_settlement_vertex: int | None = None  # for setup road placement
    players_to_discard: list[int] = field(default_factory=list)
    discard_amounts: dict[int, int] = field(default_factory=dict)
    pending_trade: TradeOffer | None = None
    trade_targets_remaining: list[int] = field(default_factory=list)
    trade_accepted: list[int] = field(default_factory=list)
    trade_counter_context: TradeOffer | None = None
    trades_proposed_this_turn: int = 0
    free_roads_remaining: int = 0
    winner: int | None = None
    turn_log: list[str] = field(default_factory=list)
    # curriculum flags (§8 of the design doc): disabling these strips the
    # corresponding action types out of legal_actions() entirely, rather than
    # leaving them legal-but-discouraged, so an early-curriculum policy has a
    # genuinely smaller action space to learn over.
    allow_trading: bool = True
    allow_dev_cards: bool = True


def new_game(randomize_board: bool = True, seed: int | None = None,
             allow_trading: bool = True, allow_dev_cards: bool = True) -> GameState:
    board = generate_board(randomize=randomize_board, seed=seed)
    players = {i: PlayerState(id=i) for i in range(NUM_PLAYERS)}
    bank = {r: 19 for r in Resource}
    deck: list[DevCard] = []
    for card, count in STANDARD_DEV_CARD_COUNTS.items():
        deck.extend([card] * count)
    import random as _random
    # No fixed fallback here (unlike generate_board's randomize=False case) --
    # dev-card order has no "reproducible curriculum layout" use case, so
    # seed=None should just mean "genuinely random," not "always this order."
    deck_rng = _random.Random(seed + 1) if seed is not None else _random.Random()
    deck_rng.shuffle(deck)
    return GameState(
        board=board,
        players=players,
        bank=bank,
        dev_card_deck=deck,
        current_player=0,
        phase=Phase.SETUP_SETTLEMENT,
        allow_trading=allow_trading,
        allow_dev_cards=allow_dev_cards,
    )
