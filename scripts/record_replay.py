"""Record a checkpoint playing a full game and emit a self-contained HTML
replay viewer (scripts/replay_template.html + embedded JSON) -- watch the
policy play in real time, step through decisions, and inspect per-decision
model diagnostics.

Diagnostics captured at every model decision (the "what is it doing wrong"
instrumentation):
- value estimate, masked action-type distribution (top alternatives), entropy
- *attempted illegal mass*: the pre-mask softmax probability the type head
  put on action types that were illegal this step. The action mask makes
  actually-illegal moves impossible, so this is the observable version of
  "the model tried an illegal move" -- large values mean the policy's raw
  preferences disagree with the rules (e.g. the documented END_TURN/trade
  logit pathology).
- the same for the chosen type's stage-1 pointer head (e.g. mass on vertices
  it can't legally build on).

Usage:
  python -m scripts.record_replay --checkpoint path.pt [--public-hand-features] \
      --seat 0 --opponents heuristic --seed 5 --out replay.html
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import random

import numpy as np
import torch

from agents.heuristic import HeuristicAgent
from agents.random_agent import RandomAgent
from env.actions import Action, ActionType
from env.board import HexType, Resource, _hex_corners
from env.engine import CatanEngine, legal_actions, total_vp
from env.pettingzoo_env import build_observation
from env.state import DevCard, NUM_PLAYERS
from training.graph_features import build_graph_observation
from training.hier_model import (
    ACTION_TYPES, ACTION_TYPE_INDEX, NEG_INF, TYPE_TO_HEADS, group_by_type, stage1_mask,
)
from training.model import flatten_observation

RESOURCE_LIST = list(Resource)
DEV_CARD_LIST = list(DevCard)  # knight, road_building, year_of_plenty, monopoly, victory_point
MAX_STEPS = 4000
TEMPLATE_PATH = os.path.join(os.path.dirname(__file__), "replay_template.html")


# -- board geometry -----------------------------------------------------------

def board_geometry(board) -> dict:
    """Pixel-space geometry for rendering: hex corner polygons and vertex/edge
    positions, derived from the same corner math generate_board used (hex
    .vertex_ids are stored in corner order, so corners map 1:1 to vertices)."""
    hexes = []
    vertex_pos: dict[int, tuple[float, float]] = {}
    for hx in board.hexes.values():
        corners = _hex_corners(hx.q, hx.r)
        for vid, pt in zip(hx.vertex_ids, corners):
            vertex_pos.setdefault(vid, pt)
        cx = sum(p[0] for p in corners) / 6.0
        cy = sum(p[1] for p in corners) / 6.0
        hexes.append({"id": hx.id, "terrain": hx.terrain.value, "number": hx.number,
                      "cx": round(cx, 2), "cy": round(cy, 2),
                      "corners": [[round(x, 2), round(y, 2)] for x, y in corners]})
    edges = {eid: list(e.vertex_ids) for eid, e in board.edges.items()}
    ports = []
    for vid, v in board.vertices.items():
        if v.port_generic:
            ports.append({"vid": vid, "kind": "any"})
        elif v.port is not None:
            ports.append({"vid": vid, "kind": v.port.value})
    return {
        "hexes": hexes,
        "vertices": {vid: [round(x, 2), round(y, 2)] for vid, (x, y) in vertex_pos.items()},
        "edges": edges,
        "ports": ports,
    }


# -- action display ------------------------------------------------------------

def action_label(a: Action) -> str:
    p = a.params
    t = a.type
    if t == ActionType.BUILD_SETTLEMENT:
        return f"build settlement @v{p['vertex_id']}"
    if t == ActionType.BUILD_CITY:
        return f"upgrade to city @v{p['vertex_id']}"
    if t == ActionType.BUILD_ROAD:
        return f"build road @e{p['edge_id']}" + (" (free)" if p.get("free") else "")
    if t in (ActionType.MOVE_ROBBER, ActionType.PLAY_KNIGHT):
        verb = "move robber" if t == ActionType.MOVE_ROBBER else "play knight, robber"
        victim = f", rob P{p['victim']}" if p.get("victim") is not None else ", rob nobody"
        return f"{verb} -> hex {p['hex_id']}{victim}"
    if t == ActionType.MARITIME_TRADE:
        return f"bank trade: give {p['give'].value} -> get {p['receive'].value}"
    if t == ActionType.PROPOSE_TRADE:
        give = ",".join(f"{r.value}x{n}" for r, n in p["give"].items())
        want = ",".join(f"{r.value}x{n}" for r, n in p["want"].items())
        return f"propose trade: {give} for {want}"
    if t == ActionType.COUNTER_TRADE:
        give = ",".join(f"{r.value}x{n}" for r, n in p["give"].items())
        want = ",".join(f"{r.value}x{n}" for r, n in p["want"].items())
        return f"counter: {give} for {want}"
    if t == ActionType.CONFIRM_TRADE:
        return f"confirm trade with P{p['target']}"
    if t == ActionType.DISCARD:
        cards = ",".join(f"{r.value}x{n}" for r, n in p["cards"].items() if n)
        return f"discard {cards}"
    if t == ActionType.PLAY_MONOPOLY:
        return f"play monopoly: {p['resource'].value}"
    if t == ActionType.PLAY_YEAR_OF_PLENTY:
        return "play year of plenty: " + ",".join(r.value for r in p["resources"])
    return t.value.replace("_", " ")


def action_target(a: Action) -> dict | None:
    """Board element the action points at, for the viewer's highlight ring."""
    p = a.params
    if "vertex_id" in p:
        return {"v": p["vertex_id"]}
    if "edge_id" in p:
        return {"e": p["edge_id"]}
    if "hex_id" in p:
        return {"h": p["hex_id"]}
    return None


def _res_vec(d: dict) -> list[int]:
    return [d.get(r, 0) for r in RESOURCE_LIST]


def snapshot_state(state) -> dict:
    players = []
    devs = []
    for pid in range(NUM_PLAYERS):
        p = state.players[pid]
        players.append([
            *[p.resources[r] for r in RESOURCE_LIST],
            p.knights_played, total_vp(state, pid),
        ])
        devs.append([p.dev_cards[c] for c in DEV_CARD_LIST])

    trade = None
    if state.pending_trade is not None:
        offer = state.pending_trade
        trade = {
            "pr": offer.proposer,
            "give": _res_vec(offer.give),
            "want": _res_vec(offer.want),
            "rem": list(state.trade_targets_remaining),
            "acc": list(state.trade_accepted),
            "ctr": None,
        }
        if state.trade_counter_context is not None:
            ctx = state.trade_counter_context
            trade["ctr"] = {"pr": ctx.proposer, "give": _res_vec(ctx.give),
                             "want": _res_vec(ctx.want)}

    return {
        "b": [[vid, owner, 2 if kind == "city" else 1]
              for vid, (owner, kind) in sorted(state.vertex_owner.items())],
        "r": [[eid, pid] for eid, pid in sorted(state.road_owner.items())],
        "rob": state.board.robber_hex,
        "p": players,
        "dev": devs,
        "tr": trade,
        "bank": [state.bank[r] for r in RESOURCE_LIST],
        "turn": state.turn_number,
        "cur": state.current_player,
        "lr": state.longest_road_holder if state.longest_road_holder is not None else -1,
        "la": state.largest_army_holder if state.largest_army_holder is not None else -1,
    }


# -- model diagnostics -----------------------------------------------------------

def _encode_for_diagnostics(state, actor: int, acts: list[Action], public_hand_features: bool,
                             model_type: str):
    """Returns (obs_batch, encode_fn) where encode_fn(model) -> (type_logits, value,
    stage1_logits_fn). Isolates the only two encoder-specific things
    `model_diagnostics` needs: how the observation is batched, and how a
    stage-1 head's logits are fetched (a flat dict lookup for the hier
    trunk's per-head linear layers vs. a shared pointer-logit method for the
    GNN trunk, which scores against node embeddings instead)."""
    if model_type == "gnn":
        obs = build_graph_observation(state, actor, public_hand_features=public_hand_features)
        return {k: torch.tensor(v, dtype=torch.float32).unsqueeze(0) for k, v in obs.items()}
    obs = build_observation(state, actor, acts, show_mask=True,
                             public_hand_features=public_hand_features)
    return torch.tensor(flatten_observation(obs), dtype=torch.float32).unsqueeze(0)


def model_diagnostics(model, state, seat: int, acts: list[Action], chosen: Action,
                       public_hand_features: bool, model_type: str = "hier") -> dict:
    obs_batch = _encode_for_diagnostics(state, seat, acts, public_hand_features, model_type)
    with torch.inference_mode():
        if model_type == "gnn":
            feats, hex_emb, vertex_emb, edge_emb, opp_emb = model.encode(obs_batch)
            self_id = obs_batch["self_id"]
        else:
            feats = model.features(obs_batch)
        value = float(model.value_head(feats).squeeze())
        type_logits = model.type_head(feats).squeeze(0)

        by_type = group_by_type(acts)
        legal_idx = {ACTION_TYPE_INDEX[t] for t in by_type}
        pre_mask = torch.softmax(type_logits, dim=-1)
        illegal_mass = float(sum(pre_mask[i] for i in range(len(ACTION_TYPES))
                                  if i not in legal_idx))
        illegal_top = sorted(
            ((ACTION_TYPES[i].value, float(pre_mask[i])) for i in range(len(ACTION_TYPES))
             if i not in legal_idx),
            key=lambda x: -x[1])[:3]

        masked = type_logits.clone()
        for i in range(len(ACTION_TYPES)):
            if i not in legal_idx:
                masked[i] = NEG_INF
        mprobs = torch.softmax(masked, dim=-1)
        entropy = float(-(mprobs[mprobs > 0] * torch.log(mprobs[mprobs > 0])).sum())
        top = sorted(((ACTION_TYPES[i].value, float(mprobs[i])) for i in legal_idx),
                     key=lambda x: -x[1])[:4]
        chosen_prob = float(mprobs[ACTION_TYPE_INDEX[chosen.type]])

        # stage-1 pointer head for the chosen type: pre-mask mass on illegal ids
        illegal_mass_s1 = None
        head_name, _ = TYPE_TO_HEADS[chosen.type]
        if head_name is not None:
            mask1 = stage1_mask(chosen.type, by_type[chosen.type])
            if model_type == "gnn":
                logits1 = model._head_logits(head_name, feats, hex_emb, vertex_emb, edge_emb,
                                              opp_emb, self_id).squeeze(0)[: len(mask1)]
            else:
                logits1 = model._head_modules[head_name](feats).squeeze(0)[: len(mask1)]
            probs1 = torch.softmax(logits1, dim=-1)
            illegal_mass_s1 = float(sum(float(probs1[i]) for i in range(len(mask1))
                                         if mask1[i] == 0))

    return {
        "v": round(value, 3), "H": round(entropy, 3), "n": len(acts),
        "cp": round(chosen_prob, 3),
        "top": [[t, round(pr, 3)] for t, pr in top],
        "im": round(illegal_mass, 3),
        "imt": [[t, round(pr, 3)] for t, pr in illegal_top if pr >= 0.005],
        "im1": round(illegal_mass_s1, 3) if illegal_mass_s1 is not None else None,
    }


# -- recording -----------------------------------------------------------------

def record_game(model, seat: int, seed: int, opponents: str,
                 public_hand_features: bool, max_steps: int = MAX_STEPS,
                 checkpoint_label: str = "", model_type: str = "hier") -> dict:
    from training.agent import HierarchicalLearnedAgent

    engine = CatanEngine(randomize_board=True, seed=seed)
    opp_cls = HeuristicAgent if opponents == "heuristic" else RandomAgent
    agents = {pid: opp_cls(pid, random.Random(seed * 97 + pid))
              for pid in range(NUM_PLAYERS) if pid != seat}
    agents[seat] = HierarchicalLearnedAgent(seat, model=model, deterministic=True,
                                             model_kind=model_type,
                                             public_hand_features=public_hand_features)

    frames = [{"a": None, "ph": engine.state.phase.name, "act": None, "at": None,
               "dice": None, "tg": None, "s": snapshot_state(engine.state), "d": None}]
    steps = 0
    while not engine.done and steps < max_steps:
        state = engine.state
        actor = engine.acting_player()
        phase = state.phase.name
        acts = legal_actions(state)
        chosen = agents[actor].choose(state, acts)

        diag = None
        if actor == seat and len(acts) > 1:
            diag = model_diagnostics(model, state, seat, acts, chosen, public_hand_features,
                                      model_type)

        engine.step(chosen)
        steps += 1
        dice = list(state.dice_roll) if chosen.type == ActionType.ROLL_DICE else None
        frames.append({
            "a": actor, "ph": phase, "act": action_label(chosen),
            "at": chosen.type.value, "dice": dice, "tg": action_target(chosen),
            "s": snapshot_state(engine.state), "d": diag,
        })

    state = engine.state
    return {
        "meta": {
            "checkpoint": checkpoint_label,
            "seat": seat,
            "seed": seed,
            "opponents": opponents,
            "public_hand_features": public_hand_features,
            "generated": datetime.datetime.now().isoformat(timespec="seconds"),
            "result": {
                "finished": engine.done,
                "winner": state.winner,
                "turns": state.turn_number,
                "steps": steps,
                "vps": [total_vp(state, pid) for pid in range(NUM_PLAYERS)],
            },
        },
        "seats": ["model" if pid == seat else opponents for pid in range(NUM_PLAYERS)],
        "board": board_geometry(state.board),
        "frames": frames,
    }


def write_replay(replay: dict, out_path: str) -> None:
    with open(TEMPLATE_PATH) as f:
        template = f.read()
    payload = json.dumps(replay, separators=(",", ":"))
    html = template.replace("__REPLAY_JSON__", payload)
    with open(out_path, "w") as f:
        f.write(html)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--model-type", choices=["hier", "gnn"], default="hier")
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--gnn-layers", type=int, default=3, help="only used when --model-type gnn")
    parser.add_argument("--public-hand-features", action="store_true")
    parser.add_argument("--seat", type=int, default=0)
    parser.add_argument("--opponents", choices=["heuristic", "random"], default="heuristic")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    parser.add_argument("--out", type=str, default="replay.html")
    args = parser.parse_args()

    from training.agent import load_gnn_model, load_hier_model
    if args.model_type == "gnn":
        model = load_gnn_model(args.checkpoint, hidden=args.hidden, gnn_layers=args.gnn_layers,
                                public_hand_features=args.public_hand_features)
    else:
        model = load_hier_model(args.checkpoint, hidden=args.hidden,
                                 public_hand_features=args.public_hand_features)

    replay = record_game(model, args.seat, args.seed, args.opponents,
                          args.public_hand_features, max_steps=args.max_steps,
                          checkpoint_label=os.path.basename(args.checkpoint),
                          model_type=args.model_type)
    write_replay(replay, args.out)

    r = replay["meta"]["result"]
    n_model_decisions = sum(1 for f in replay["frames"] if f["d"] is not None)
    max_im = max((f["d"]["im"] for f in replay["frames"] if f["d"]), default=0.0)
    print(f"recorded seed={args.seed}: {r['steps']} steps, {r['turns']} turns, "
          f"winner={'P' + str(r['winner']) if r['winner'] is not None else 'none (truncated)'}, "
          f"VPs={r['vps']}")
    print(f"model decisions instrumented: {n_model_decisions} "
          f"(max attempted-illegal type mass: {max_im:.1%})")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
