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
    CatanEngine, acting_player as engine_acting_player, is_legal_action, is_template,
    legal_actions as engine_legal_actions, total_vp,
)
from env.public_beliefs import expected_dev_cards, last_offer, turns_since_dev_purchase
from env.state import DevCard, MAX_TRADE_PROPOSALS_PER_TURN, NUM_PLAYERS, Phase, PlayerState

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
                 public_hand_features: bool = False, truncation_reward: str = "zero",
                 terminal_reward: str = "win_loss"):
        super().__init__()
        # What a real game end pays. "win_loss" (default): +1 to the winner,
        # -1/3 to each loser -- zero-sum, and exactly the objective evaluation
        # measures (win rate). "rank": legacy placement reward {+1, 0, -0.5,
        # -1} by VP, which trades win probability for 2nd place (audit F-16).
        if terminal_reward not in ("win_loss", "rank"):
            raise ValueError(f"terminal_reward must be 'win_loss' or 'rank', got {terminal_reward!r}")
        self._terminal_reward = terminal_reward
        # What a step-cap truncation pays. "zero" (default): nobody is paid --
        # a truncated game has no winner, and the training loop bootstraps
        # V(s_T) instead (training/ppo.compute_gae). "rank": legacy behaviour,
        # rank-on-current-VP rewards, which pays the VP leader exactly what a
        # real win pays and so rewards stalling while ahead (audit F-06).
        if truncation_reward not in ("zero", "rank"):
            raise ValueError(f"truncation_reward must be 'zero' or 'rank', got {truncation_reward!r}")
        self._truncation_reward = truncation_reward
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
            # Live counter-offer (TRADE_RESPONSE, proposer deciding): what the
            # counter-offerer gives / wants and who they are. Public to the
            # whole table, and the proposer cannot evaluate ACCEPT without it.
            "counter_trade_give": spaces.Box(0, 19, (5,), dtype=np.int16),
            "counter_trade_want": spaces.Box(0, 19, (5,), dtype=np.int16),
            "counter_trade_proposer": spaces.Box(-1, NUM_PLAYERS - 1, (1,), dtype=np.int8),
            # who the live proposal is addressed to (absolute seats) and how
            # many of the turn owner's proposals are already used
            "pending_trade_targets": spaces.Box(0, 1, (NUM_PLAYERS,), dtype=np.int8),
            "trades_proposed_this_turn": spaces.Box(0, MAX_TRADE_PROPOSALS_PER_TURN, (1,), dtype=np.int8),
            # whose observation this is (encoders make seats observer-relative)
            "observer": spaces.Box(0, NUM_PLAYERS - 1, (1,), dtype=np.int8),
            # public supply and deck size
            "bank": spaces.Box(0, 19, (5,), dtype=np.int16),
            "dev_deck_size": spaces.Box(0, 25, (1,), dtype=np.int16),
            # public-event beliefs per seat (env/public_beliefs.py): expected
            # hidden VP cards / knights, age of last dev purchase (0-1), last
            # trade offer give/want and its age (0-1)
            "public_expected_vp": spaces.Box(0, 5, (NUM_PLAYERS,), dtype=np.float32),
            "public_expected_knights": spaces.Box(0, 14, (NUM_PLAYERS,), dtype=np.float32),
            "public_dev_purchase_age": spaces.Box(0, 1, (NUM_PLAYERS,), dtype=np.float32),
            "public_last_offer_give": spaces.Box(0, 3, (NUM_PLAYERS, 5), dtype=np.int8),
            "public_last_offer_want": spaces.Box(0, 3, (NUM_PLAYERS, 5), dtype=np.int8),
            "public_last_offer_age": spaces.Box(0, 1, (NUM_PLAYERS,), dtype=np.float32),
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
    def step(self, action) -> None:
        """`action` is either an int index into `legal_actions()` (the
        Discrete contract; trade *templates* are not valid indices and are
        0 in `action_mask`), or a concrete `Action` -- the only way to submit
        a structured trade (engine.make_trade). Anything illegal raises
        ValueError before the game state is touched."""
        agent = self.agent_selection
        if self.terminations[agent] or self.truncations[agent]:
            return self._was_dead_step(action)

        if isinstance(action, Action):
            if not is_legal_action(self.engine.state, action):
                raise ValueError(f"illegal action {action!r}")
            concrete = action
        else:
            if action < 0 or action >= len(self._legal_cache):
                raise ValueError(f"action index {action} out of range for "
                                  f"{len(self._legal_cache)} legal actions")
            concrete = self._legal_cache[action]
            if is_template(concrete):
                raise ValueError(f"index {action} is a {concrete.type.value} template; submit a "
                                  "concrete trade Action (engine.make_trade) instead")

        self.rewards = {a: 0.0 for a in self.agents}
        # already validated above (index into this step's legal list, or
        # is_legal_action for a submitted Action): skip the engine's own check,
        # which would re-enumerate every legal action (performance audit O1)
        self.engine.step(concrete, validate=False)
        self._step_count += 1

        state = self.engine.state
        if self._vp_shaping_weight:
            self._apply_vp_shaping(state)

        if state.phase == Phase.GAME_OVER:
            if self._terminal_reward == "win_loss":
                self._assign_win_loss_rewards()
            else:
                self._assign_terminal_rewards()
            self.terminations = {a: True for a in self.agents}
        elif self._max_episode_steps is not None and self._step_count >= self._max_episode_steps:
            if self._truncation_reward == "rank":
                self._assign_terminal_rewards()  # legacy: rank on current standing
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

    def _assign_win_loss_rewards(self) -> None:
        """+1 winner, -1/3 each other player (sums to zero). Final rank/VP
        infos are still recorded for logging."""
        state = self.engine.state
        for pid in state.players:
            self.rewards[_agent_name(pid)] += 1.0 if pid == state.winner else -1.0 / 3.0
        self._record_final_standing()

    def _record_final_standing(self) -> None:
        state = self.engine.state
        vps = {pid: total_vp(state, pid) for pid in state.players}
        order = sorted(state.players, key=lambda pid: (pid != state.winner, -vps[pid]))
        for rank, pid in enumerate(order, start=1):
            self.infos[_agent_name(pid)]["final_rank"] = rank
            self.infos[_agent_name(pid)]["final_vp"] = vps[pid]

    def _assign_terminal_rewards(self) -> None:
        """Legacy rank-based reward (terminal_reward="rank", and
        truncation_reward="rank")."""
        state = self.engine.state
        vps = {pid: total_vp(state, pid) for pid in state.players}
        ranking = sorted(state.players.keys(), key=lambda pid: vps[pid], reverse=True)
        if state.winner is not None:
            # The winner is 1st outright even if another player is level on VP
            # (possible: someone can sit on 10+ VP reached during another
            # player's turn, waiting to claim it on their own turn).
            ranking.remove(state.winner)
            ranking.insert(0, state.winner)
        rank_reward = {0: 1.0, 1: 0.0, 2: -0.5, 3: -1.0}

        # Group by VP so ties split the reward mass evenly instead of
        # falling back to sorted()'s stable order, which silently favored
        # lower player_id on every tie -- common on truncation, where
        # standing is ranked on current VP rather than a clean win, and a
        # real-terminal tie for 2nd/3rd/4th is still possible on a clean win.
        groups: list[list[int]] = []
        for pid in ranking:
            if groups and vps[pid] == vps[groups[-1][0]] and groups[-1][0] != state.winner:
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
    `training.agent.HierarchicalLearnedAgent`) query a trained policy outside
    the AEC wrapper, using the exact same encoding path training used."""
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

    counter_give = np.zeros(5, dtype=np.int16)
    counter_want = np.zeros(5, dtype=np.int16)
    counter_proposer = -1
    if state.trade_counter_context is not None:
        for r, amt in state.trade_counter_context.give.items():
            counter_give[RESOURCE_INDEX[r]] = amt
        for r, amt in state.trade_counter_context.want.items():
            counter_want[RESOURCE_INDEX[r]] = amt
        counter_proposer = state.trade_counter_context.proposer

    exp_vp = np.zeros(NUM_PLAYERS, dtype=np.float32)
    exp_kn = np.zeros(NUM_PLAYERS, dtype=np.float32)
    dev_age = np.zeros(NUM_PLAYERS, dtype=np.float32)
    offer_give = np.zeros((NUM_PLAYERS, 5), dtype=np.int8)
    offer_want = np.zeros((NUM_PLAYERS, 5), dtype=np.int8)
    offer_age = np.zeros(NUM_PLAYERS, dtype=np.float32)
    for other_pid in state.players:
        exp = expected_dev_cards(state, pid, other_pid)
        exp_vp[other_pid] = exp[DevCard.VICTORY_POINT]
        exp_kn[other_pid] = exp[DevCard.KNIGHT]
        dev_age[other_pid] = turns_since_dev_purchase(state, other_pid)
        g, w, age = last_offer(state, other_pid)
        for r, k in g.items():
            offer_give[other_pid, RESOURCE_INDEX[r]] = k
        for r, k in w.items():
            offer_want[other_pid, RESOURCE_INDEX[r]] = k
        offer_age[other_pid] = age

    pending_targets = np.zeros(NUM_PLAYERS, dtype=np.int8)
    if state.pending_trade is not None:
        pending_targets[state.pending_trade.targets] = 1

    mask = np.zeros(MAX_ACTIONS, dtype=np.int8)
    if show_mask:
        # templates aren't steppable by index (see CatanAECEnv.step)
        mask[: len(legal_cache)] = [0 if is_template(a) else 1 for a in legal_cache]

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
        "counter_trade_give": counter_give,
        "counter_trade_want": counter_want,
        "counter_trade_proposer": np.array([counter_proposer], dtype=np.int8),
        "pending_trade_targets": pending_targets,
        "trades_proposed_this_turn": np.array([state.trades_proposed_this_turn], dtype=np.int8),
        "observer": np.array([pid], dtype=np.int8),
        "bank": np.array([state.bank[r] for r in RESOURCE_LIST], dtype=np.int16),
        "dev_deck_size": np.array([len(state.dev_card_deck)], dtype=np.int16),
        "public_expected_vp": exp_vp,
        "public_expected_knights": exp_kn,
        "public_dev_purchase_age": dev_age,
        "public_last_offer_give": offer_give,
        "public_last_offer_want": offer_want,
        "public_last_offer_age": offer_age,
        "action_mask": mask,
    }
