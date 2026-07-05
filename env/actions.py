"""Action type inventory. An Action is (type, params-dict); engine.legal_actions()
enumerates concrete legal Actions for the player to move (§3 of the design doc
factors these into head-wise masks later, in the PettingZoo wrapper -- the
engine itself deals in whole concrete actions since that's what correctness
testing needs)."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from env.board import Resource


class ActionType(Enum):
    ROLL_DICE = "roll_dice"
    BUILD_ROAD = "build_road"
    BUILD_SETTLEMENT = "build_settlement"
    BUILD_CITY = "build_city"
    BUY_DEV_CARD = "buy_dev_card"
    PLAY_KNIGHT = "play_knight"
    PLAY_ROAD_BUILDING = "play_road_building"
    PLAY_YEAR_OF_PLENTY = "play_year_of_plenty"
    PLAY_MONOPOLY = "play_monopoly"
    MOVE_ROBBER = "move_robber"
    DISCARD = "discard"
    MARITIME_TRADE = "maritime_trade"
    PROPOSE_TRADE = "propose_trade"
    ACCEPT_TRADE = "accept_trade"
    REJECT_TRADE = "reject_trade"
    COUNTER_TRADE = "counter_trade"
    CONFIRM_TRADE = "confirm_trade"
    CANCEL_TRADE = "cancel_trade"
    END_TURN = "end_turn"


@dataclass(frozen=True)
class Action:
    type: ActionType
    params: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"{self.type.value}({self.params})"
