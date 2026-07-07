"""PettingZoo AEC wrapper around the Catan engine (§1 of the design doc).

Turn ownership follows `engine.acting_player`, not simple round-robin cycling
-- discard, robber, and trade-response sub-protocols hand control to whichever
player owes the next decision, then return it to the normal turn owner. This
is the "embedded sub-game" pattern described in §1: one AEC stepping contract,
no separate simultaneous-move API.

Action representation is a deliberate placeholder, not the final §3 design:
`action_space(agent) = Discrete(MAX_ACTIONS)` where index i means "the i-th
entry of the legal-action list returned for this observation" (the same
"index into legal_actions" contract many imperfect-information-game RL
wrappers use, e.g. OpenSpiel's). The autoregressive, pointer-based factorized
policy head described in §3/§5 is a model-side concern for later phases; it
will consume `Action` objects the same way this wrapper's `_legal_cache` does,
so nothing here needs to change when that head is built -- only how an agent
picks among `legal_actions` will.
"""
from __future__ import annotations

import numpy as np
from gymnasium import spaces
from pettingzoo.utils.env import AECEnv

from env.actions import Action
from env.board import HexType, Resource
from env.engine import (
    CatanEngine, acting_player as engine_acting_player, legal_actions as engine_legal_actions,
    total_vp,
)
from env.state import DevCard, NUM_PLAYERS, Phase, PlayerState

MAX_ACTIONS = 400
NUM_HEXES = 19
NUM_VERTICES = 54
NUM_EDGES = 72
RESOURCE_LIST = list(Resource)
DEV_CARD_LIST = list(DevCard)
PHASE_LIST = list(Phase)
RESOURCE_INDEX = {r: i for i, r in enumerate(Resource)}
HEXTYPE_INDEX = {t: i for i, t in enumerate(HexType)}
PHASE_INDEX = {p: i for i, p in enumerate(Phase)}


def _agent_name(pid: int) -> str:
    return f"player_{pid}"


def _agent_id(agent: str) -> int:
    return int(agent.split("_")[1])


class CatanAECEnv(AECEnv):
    metadata = {"name": "catan_v0", "is_parallelizable": False}

    def __init__(self, randomize_board: bool = True, seed: int | None = None,
                 allow_trading: bool = True, allow_dev_cards: bool = True,
                 vp_shaping_weight: float = 0.0, max_episode_steps: int | None = None,
                 public_hand_features: bool = False):
        super().__init__()
        self._randomize_board = randomize_board
        self._init_seed = seed
        self._allow_trading = allow_trading
        self._allow_dev_cards = allow_dev_cards
        self._vp_shaping_weight = vp_shaping_weight
        self._max_episode_steps = max_episode_steps
        # Expose the engine's publicly-inferable per-player resource
        # estimates (card counting) as observation features. Off by default:
        # it widens the observation, so a model must be built with the same
        # flag it will be run with (checkpoints trained without it keep
        # their dimensions). Public attribute -- model adapters read it.
        self.public_hand_features = public_hand_features
        self._step_count = 0
        self.possible_agents = [_agent_name(i) for i in range(NUM_PLAYERS)]
        self.agents = self.possible_agents[:]

        obs_space = self._build_observation_space()
        self.observation_spaces = {a: obs_space for a in self.possible_agents}
        self.action_spaces = {a: spaces.Discrete(MAX_ACTIONS) for a in self.possible_agents}

        self.engine: CatanEngine | None = None
        self._legal_cache: list[Action] = []

    # -- gymnasium/pettingzoo space plumbing -----------------------------
    def _build_observation_space(self) -> spaces.Dict:
        extra = {}
        if self.public_hand_features:
            extra = {
                "public_est_resources": spaces.Box(0, 19, (NUM_PLAYERS, 5), dtype=np.float32),
                "public_est_unknown": spaces.Box(0, 40, (NUM_PLAYERS,), dtype=np.float32),
            }
        return spaces.Dict({
            **extra,
            "hex_terrain": spaces.Box(0, 5, (NUM_HEXES,), dtype=np.int8),
            "hex_number": spaces.Box(0, 12, (NUM_HEXES,), dtype=np.int8),
            "robber": spaces.Box(0, 1, (NUM_HEXES,), dtype=np.int8),
            "vertex_owner": spaces.Box(-1, NUM_PLAYERS - 1, (NUM_VERTICES,), dtype=np.int8),
            "vertex_type": spaces.Box(0, 2, (NUM_VERTICES,), dtype=np.int8),
            "vertex_port_generic": spaces.Box(0, 1, (NUM_VERTICES,), dtype=np.int8),
            "vertex_port_resource": spaces.Box(-1, 4, (NUM_VERTICES,), dtype=np.int8),
            "edge_owner": spaces.Box(-1, NUM_PLAYERS - 1, (NUM_EDGES,), dtype=np.int8),
            "own_resources": spaces.Box(0, 19, (5,), dtype=np.int16),
            "own_dev_cards": spaces.Box(0, 25, (5,), dtype=np.int16),
            "own_dev_cards_playable": spaces.Box(0, 25, (5,), dtype=np.int16),
            "public_hand_size": spaces.Box(0, 40, (NUM_PLAYERS,), dtype=np.int16),
            "public_visible_vp": spaces.Box(0, 12, (NUM_PLAYERS,), dtype=np.int8),
            "public_settlements": spaces.Box(0, 5, (NUM_PLAYERS,), dtype=np.int8),
            "public_cities": spaces.Box(0, 4, (NUM_PLAYERS,), dtype=np.int8),
            "public_roads": spaces.Box(0, 15, (NUM_PLAYERS,), dtype=np.int8),
            "public_knights_played": spaces.Box(0, 14, (NUM_PLAYERS,), dtype=np.int8),
            "public_dev_card_count": spaces.Box(0, 25, (NUM_PLAYERS,), dtype=np.int8),
            "longest_road_holder": spaces.Box(-1, NUM_PLAYERS - 1, (1,), dtype=np.int8),
            "largest_army_holder": spaces.Box(-1, NUM_PLAYERS - 1, (1,), dtype=np.int8),
            "current_player": spaces.Box(0, NUM_PLAYERS - 1, (1,), dtype=np.int8),
            "acting_player": spaces.Box(0, NUM_PLAYERS - 1, (1,), dtype=np.int8),
            "phase": spaces.Box(0, len(PHASE_LIST) - 1, (1,), dtype=np.int8),
            "dice_roll": spaces.Box(0, 6, (2,), dtype=np.int8),
            "pending_trade_give": spaces.Box(0, 19, (5,), dtype=np.int16),
            "pending_trade_want": spaces.Box(0, 19, (5,), dtype=np.int16),
            "pending_trade_proposer": spaces.Box(-1, NUM_PLAYERS - 1, (1,), dtype=np.int8),
            "action_mask": spaces.Box(0, 1, (MAX_ACTIONS,), dtype=np.int8),
        })

    # No lru_cache here: caching a bound method on `self` would pin every env
    # instance in the cache forever (a leak when envs are created per
    # iteration). The spaces dicts already hold the one shared space object,
    # so a plain lookup returns a stable identity anyway.
    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    # -- lifecycle --------------------------------------------------------
    def reset(self, seed: int | None = None, options: dict | None = None) -> None:
        use_seed = seed if seed is not None else self._init_seed
        self.engine = CatanEngine(randomize_board=self._randomize_board, seed=use_seed,
                                   allow_trading=self._allow_trading, allow_dev_cards=self._allow_dev_cards)
        self.agents = self.possible_agents[:]
        self.rewards = {a: 0.0 for a in self.agents}
        self._cumulative_rewards = {a: 0.0 for a in self.agents}
        self.terminations = {a: False for a in self.agents}
        self.truncations = {a: False for a in self.agents}
        self.infos = {a: {} for a in self.agents}
        self._prev_vp = {pid: total_vp(self.engine.state, pid) for pid in self.engine.state.players}
        self._step_count = 0
        self._refresh_legal_cache()
        self.agent_selection = _agent_name(engine_acting_player(self.engine.state))

    def _refresh_legal_cache(self) -> None:
        self._legal_cache = engine_legal_actions(self.engine.state)
        if len(self._legal_cache) > MAX_ACTIONS:
            raise RuntimeError(
                f"legal_actions() returned {len(self._legal_cache)} > MAX_ACTIONS={MAX_ACTIONS}; "
                "raise the padded action space size."
            )

    def legal_actions(self) -> list[Action]:
        """The concrete Action objects the current `action_mask` indexes into."""
        return self._legal_cache

    def clear_reward(self, agent: str) -> None:
        """Zero `agent`'s accumulated reward right after it's been read via
        `last()`, so a later `last()` call only reflects reward earned since
        this point -- the "reward since I was last asked to act" pattern PPO
        rollout collection relies on. A documented, stable method instead of
        training code (training/ppo.py, training/hier_ppo.py) reaching into
        PettingZoo's own private `_cumulative_rewards` bookkeeping directly."""
        if agent in self._cumulative_rewards:
            self._cumulative_rewards[agent] = 0.0

    # -- stepping -----------------------------------------------------------
    def step(self, action: int) -> None:
        agent = self.agent_selection
        if self.terminations[agent] or self.truncations[agent]:
            return self._was_dead_step(action)

        if action < 0 or action >= len(self._legal_cache):
            raise ValueError(f"action index {action} out of range for "
                              f"{len(self._legal_cache)} legal actions")
        concrete = self._legal_cache[action]

        self.rewards = {a: 0.0 for a in self.agents}
        self.engine.step(concrete)
        self._step_count += 1

        state = self.engine.state
        if self._vp_shaping_weight:
            self._apply_vp_shaping(state)

        if state.phase == Phase.GAME_OVER:
            self._assign_terminal_rewards()
            self.terminations = {a: True for a in self.agents}
        elif self._max_episode_steps is not None and self._step_count >= self._max_episode_steps:
            self._assign_terminal_rewards()  # rank on current standing, not a real win
            self.truncations = {a: True for a in self.agents}
            self._refresh_legal_cache()
        else:
            self._refresh_legal_cache()

        self._accumulate_rewards()
        if not (all(self.terminations.values()) or all(self.truncations.values())):
            self.agent_selection = _agent_name(engine_acting_player(state))

    def _apply_vp_shaping(self, state) -> None:
        """Potential-based-in-spirit dense shaping (§4, stage A/B): a small
        per-step reward for each player's own VP delta, so credit assignment
        doesn't rely solely on the sparse terminal outcome. Applied to every
        player, not just whoever acted, since a road/settlement placement can
        change someone else's longest-road VP off-turn."""
        for pid in state.players:
            new_vp = total_vp(state, pid)
            delta = new_vp - self._prev_vp[pid]
            if delta:
                self.rewards[_agent_name(pid)] += delta * self._vp_shaping_weight
            self._prev_vp[pid] = new_vp

    def _assign_terminal_rewards(self) -> None:
        state = self.engine.state
        vps = {pid: total_vp(state, pid) for pid in state.players}
        ranking = sorted(state.players.keys(), key=lambda pid: vps[pid], reverse=True)
        rank_reward = {0: 1.0, 1: 0.0, 2: -0.5, 3: -1.0}

        # Group by VP so ties split the reward mass evenly instead of
        # falling back to sorted()'s stable order, which silently favored
        # lower player_id on every tie -- common on truncation, where
        # standing is ranked on current VP rather than a clean win, and a
        # real-terminal tie for 2nd/3rd/4th is still possible on a clean win.
        groups: list[list[int]] = []
        for pid in ranking:
            if groups and vps[pid] == vps[groups[-1][0]]:
                groups[-1].append(pid)
            else:
                groups.append([pid])

        rank = 0
        for group in groups:
            avg_reward = sum(rank_reward.get(rank + i, -1.0) for i in range(len(group))) / len(group)
            for pid in group:
                self.rewards[_agent_name(pid)] += avg_reward
                self.infos[_agent_name(pid)]["final_rank"] = rank + 1
                self.infos[_agent_name(pid)]["final_vp"] = vps[pid]
            rank += len(group)

    # -- observation --------------------------------------------------------
    def observe(self, agent: str) -> dict:
        pid = _agent_id(agent)
        return build_observation(self.engine.state, pid, self._legal_cache,
                                  show_mask=(agent == self.agent_selection),
                                  public_hand_features=self.public_hand_features)

    def render(self):
        state = self.engine.state
        print(f"turn {state.turn_number} phase={state.phase.name} "
              f"current={state.current_player} acting={engine_acting_player(state)}")
        for pid, p in state.players.items():
            print(f"  p{pid}: vp={total_vp(state, pid)} hand={p.hand_size()} "
                  f"settlements={len(p.settlements)} cities={len(p.cities)} roads={len(p.roads)}")

    def close(self):
        pass


def _static_board_arrays(board) -> dict[str, np.ndarray]:
    """Arrays derived purely from the immutable board layout (terrain, number
    tokens, ports) -- identical for every observation of the same board, so
    they're computed once and cached on the Board object. Callers receive
    copies, so downstream mutation can't corrupt the cache."""
    cached = getattr(board, "_flat_obs_static", None)
    if cached is None:
        hex_terrain = np.zeros(NUM_HEXES, dtype=np.int8)
        hex_number = np.zeros(NUM_HEXES, dtype=np.int8)
        for hx in board.hexes.values():
            hex_terrain[hx.id] = HEXTYPE_INDEX[hx.terrain]
            hex_number[hx.id] = hx.number or 0
        vertex_port_generic = np.zeros(NUM_VERTICES, dtype=np.int8)
        vertex_port_resource = np.full(NUM_VERTICES, -1, dtype=np.int8)
        for vid, v in board.vertices.items():
            if v.port_generic:
                vertex_port_generic[vid] = 1
            if v.port is not None:
                vertex_port_resource[vid] = RESOURCE_INDEX[v.port]
        cached = {
            "hex_terrain": hex_terrain,
            "hex_number": hex_number,
            "vertex_port_generic": vertex_port_generic,
            "vertex_port_resource": vertex_port_resource,
        }
        board._flat_obs_static = cached
    return {k: v.copy() for k, v in cached.items()}


def public_hand_estimate_arrays(state) -> tuple[np.ndarray, np.ndarray]:
    """(NUM_PLAYERS, 5) publicly-inferable per-player resource estimates and
    (NUM_PLAYERS,) unknown-identity card counts (public hand size minus the
    identified estimate mass). Includes every seat -- the observer's own row
    is what *opponents* can infer about them, which is itself strategically
    useful (e.g. how attractive a robber target the observer looks)."""
    est = np.zeros((NUM_PLAYERS, 5), dtype=np.float32)
    unknown = np.zeros(NUM_PLAYERS, dtype=np.float32)
    for opid, e in state.public_resource_estimates.items():
        row = est[opid]
        for i, r in enumerate(RESOURCE_LIST):
            row[i] = e[r]
        unknown[opid] = max(0.0, state.players[opid].hand_size() - float(row.sum()))
    return est, unknown


def build_observation(state, pid: int, legal_cache: list[Action], show_mask: bool,
                       public_hand_features: bool = False) -> dict:
    """Builds the same observation dict `CatanAECEnv.observe` returns, but
    directly from a raw `GameState` -- lets a standalone agent (e.g.
    `training.agent.LearnedAgent`) query a trained policy outside the AEC
    wrapper, using the exact same encoding path training used."""
    board = state.board
    static = _static_board_arrays(board)

    robber = np.zeros(NUM_HEXES, dtype=np.int8)
    robber[board.robber_hex] = 1

    vertex_owner = np.full(NUM_VERTICES, -1, dtype=np.int8)
    vertex_type = np.zeros(NUM_VERTICES, dtype=np.int8)
    for other_pid, p in state.players.items():
        for vid in p.settlements:
            vertex_owner[vid] = other_pid
            vertex_type[vid] = 1
        for vid in p.cities:
            vertex_owner[vid] = other_pid
            vertex_type[vid] = 2

    edge_owner = np.full(NUM_EDGES, -1, dtype=np.int8)
    for other_pid, p in state.players.items():
        for eid in p.roads:
            edge_owner[eid] = other_pid

    me: PlayerState = state.players[pid]
    own_resources = np.array([me.resources[r] for r in RESOURCE_LIST], dtype=np.int16)
    own_dev_cards = np.array([me.dev_cards[c] for c in DEV_CARD_LIST], dtype=np.int16)
    own_playable = np.array(
        [me.dev_cards[c] - me.dev_cards_bought_this_turn[c] for c in DEV_CARD_LIST],
        dtype=np.int16,
    )

    public_hand_size = np.zeros(NUM_PLAYERS, dtype=np.int16)
    public_visible_vp = np.zeros(NUM_PLAYERS, dtype=np.int8)
    public_settlements = np.zeros(NUM_PLAYERS, dtype=np.int8)
    public_cities = np.zeros(NUM_PLAYERS, dtype=np.int8)
    public_roads = np.zeros(NUM_PLAYERS, dtype=np.int8)
    public_knights = np.zeros(NUM_PLAYERS, dtype=np.int8)
    public_dev_count = np.zeros(NUM_PLAYERS, dtype=np.int8)
    for other_pid, p in state.players.items():
        public_hand_size[other_pid] = p.hand_size()
        public_visible_vp[other_pid] = p.visible_vp()
        public_settlements[other_pid] = len(p.settlements)
        public_cities[other_pid] = len(p.cities)
        public_roads[other_pid] = len(p.roads)
        public_knights[other_pid] = p.knights_played
        public_dev_count[other_pid] = p.total_dev_cards()

    dice = state.dice_roll or (0, 0)
    pending_give = np.zeros(5, dtype=np.int16)
    pending_want = np.zeros(5, dtype=np.int16)
    pending_proposer = -1
    if state.pending_trade is not None:
        for r, amt in state.pending_trade.give.items():
            pending_give[RESOURCE_INDEX[r]] = amt
        for r, amt in state.pending_trade.want.items():
            pending_want[RESOURCE_INDEX[r]] = amt
        pending_proposer = state.pending_trade.proposer

    mask = np.zeros(MAX_ACTIONS, dtype=np.int8)
    if show_mask:
        mask[: len(legal_cache)] = 1

    extra = {}
    if public_hand_features:
        est, unknown = public_hand_estimate_arrays(state)
        extra = {"public_est_resources": est, "public_est_unknown": unknown}

    return {
        **extra,
        "hex_terrain": static["hex_terrain"],
        "hex_number": static["hex_number"],
        "robber": robber,
        "vertex_owner": vertex_owner,
        "vertex_type": vertex_type,
        "vertex_port_generic": static["vertex_port_generic"],
        "vertex_port_resource": static["vertex_port_resource"],
        "edge_owner": edge_owner,
        "own_resources": own_resources,
        "own_dev_cards": own_dev_cards,
        "own_dev_cards_playable": own_playable,
        "public_hand_size": public_hand_size,
        "public_visible_vp": public_visible_vp,
        "public_settlements": public_settlements,
        "public_cities": public_cities,
        "public_roads": public_roads,
        "public_knights_played": public_knights,
        "public_dev_card_count": public_dev_count,
        "longest_road_holder": np.array([state.longest_road_holder
                                          if state.longest_road_holder is not None else -1], dtype=np.int8),
        "largest_army_holder": np.array([state.largest_army_holder
                                          if state.largest_army_holder is not None else -1], dtype=np.int8),
        "current_player": np.array([state.current_player], dtype=np.int8),
        "acting_player": np.array([engine_acting_player(state)], dtype=np.int8),
        "phase": np.array([PHASE_INDEX[state.phase]], dtype=np.int8),
        "dice_roll": np.array(dice, dtype=np.int8),
        "pending_trade_give": pending_give,
        "pending_trade_want": pending_want,
        "pending_trade_proposer": np.array([pending_proposer], dtype=np.int8),
        "action_mask": mask,
    }
