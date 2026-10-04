# catan-marl

A rules-correct, 4-player standard Catan simulator with a PettingZoo AEC
interface, built as the environment layer (roadmap phases 1-2) for the
self-play multi-agent RL system described in the accompanying design doc.

## Setup

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For GPU-accelerated training, replace the CPU torch wheel with a CUDA build
matching your driver (check `nvidia-smi`'s "CUDA Version" for the max
supported toolkit), e.g.:

```
pip install --force-reinstall --index-url https://download.pytorch.org/whl/cu126 torch==2.12.1
```

`training/train_hier.py` and `training/league_train.py` take a `--device`
flag (`auto` / `cpu` / `cuda`, default `auto`). Only the batched PPO gradient
step runs on the chosen device -- rollout collection always stays on CPU,
since `--num-workers > 1` forks worker processes and a CUDA context can't
survive a fork, and single-sample env-step inference is dominated by
Python/env overhead anyway (GPU wouldn't help there).

## What's here

- `env/board.py` — hex/vertex/edge graph generation, ports, standard resource
  and number-token distribution.
- `env/state.py` — game state dataclasses, dev cards, building costs.
- `env/engine.py` — the rules engine: legality, turn/phase state machine,
  dice production, robber, trading, longest road / largest army, win check.
  Also maintains publicly-inferable per-player resource estimates (card
  counting: every flow except robber-steal identity and discard contents is
  public in Catan) — exposed as observation features via the
  `--public-hand-features` training flag / `CatanAECEnv(public_hand_features=True)`.
  Off by default; the flag widens the observation, so checkpoints are only
  compatible across runs with the same setting — train an A/B pair to
  measure its effect on robber targeting and trade evaluation.
- `env/actions.py` — the action-type inventory.
- `env/pettingzoo_env.py` — PettingZoo AEC wrapper (`CatanAECEnv`) with a
  structured `Dict` observation and a masked `Discrete(400)` action space
  (index into the current legal-action list — see the module docstring for
  why this is a placeholder for the factorized policy head described in the
  design doc's action-space section, not the final RL-facing encoding).
  Domestic trades are *structured*: the legal list carries one
  PROPOSE_TRADE / COUNTER_TRADE template (masked out of the index space), and
  a concrete bundle built with `env.engine.make_trade` is passed to
  `env.step(action)` as an `Action`, validated by `engine.is_legal_action`.
- `agents/random_agent.py` — legal-action agent (function + `RandomAgent`
  class) for smoke testing and as the evaluation-tier floor.
- `agents/heuristic.py` — greedy-expansion + needs-based trade agent
  (roadmap phase 3): production-value settlement/city placement, a 2-hop
  BFS road-targeting heuristic, priority-ordered build/dev-card decisions,
  opportunistic knight play for largest army and robber blocking, and
  needs-based trade proposals/accepts/counters/discards scored against
  whatever the agent is currently saving up for.
- `scripts/smoke_test.py` — runs N full random games end to end and reports
  crash/timeout rate, step counts, and winner distribution.
- `scripts/evaluate.py` — tournament runner: seat any mix of agent kinds
  (`--seats heuristic,random,random,random`), reports win rate/avg VP/turns
  to finish per agent kind (§12 of the design doc).
- `scripts/record_replay.py` — plays a checkpoint through a full game and
  writes a self-contained HTML replay viewer (board, hands, action log,
  play/pause/step/seek) with per-decision model diagnostics: value estimate,
  masked type distribution, and *attempted-illegal probability mass* (the
  pre-mask probability the policy put on rule-breaking options -- masking
  makes illegal moves impossible to execute, so this is the observable form
  of "the model tried an illegal move"). `--out replay.html`, open in any
  browser; "Next flagged" jumps straight to suspicious decisions.
- `tests/` — pytest suite: board invariants, rules/legality, scoring,
  PettingZoo API compliance (via `pettingzoo.test.api_test`), heuristic-
  vs-random / heuristic-vs-heuristic tournament checks, and the PPO/model
  pipeline (masking, GAE, checkpoint round-trip).
- `training/model.py` — flat-vector observation encoder (`flatten_observation`,
  reused by every model below) + the original `ActorCritic` MLP with a flat
  masked-categorical `Discrete(400)` head (phase 4's v0 sanity check — see
  "Status" for why it was superseded).
- `training/hier_model.py` — `HierarchicalActorCritic`: the pointer-based
  policy head that replaced the flat one (action-type categorical, then
  pointer heads over stable vertex/edge/hex/resource/player IDs). This is the
  model everything below actually trains.
- `training/ppo.py` — shared GAE utility (`compute_gae`) used by
  `training/hier_ppo.py`, the self-play/fine-tuning PPO loop for the
  hierarchical and GNN models; it also has fork-based parallel rollout
  collection (`collect_rollout_parallel`, ~6-7x throughput on a 16-core
  machine) and supports mixed-seat episodes (trainee vs. a frozen league
  opponent occupying the other seats). (The original flat-action-space model
  and its PPO loop/CLI, `training/model.ActorCritic` and `training/train.py`,
  were superseded by the pointer-based hierarchical model below and removed.)
- `training/train_hier.py` — single-policy training CLI (hierarchical
  model), with curriculum flags, checkpointing, and evaluation against both
  the fixed random/heuristic tiers and the current training-distribution
  settings.
- `training/league.py` — `League`: a persisted population registry (roadmap
  phase 5) with TrueSkill ratings fed by each game's full finishing order,
  PFSP-style opponent sampling, and a pre-registered statistical promotion
  test (win-rate threshold + exact binomial p-value, no scipy needed).
- `training/league_train.py` — the league training CLI: alternates pure
  self-play with matches against a PFSP-sampled league opponent, snapshots
  historical checkpoints periodically, runs promotion evals, and can spin up
  short "main exploiter" side-sessions against the frozen current main.
- `training/agent.py` — `LearnedAgent` / `HierarchicalLearnedAgent`, wrapping
  a checkpoint behind the same `.choose(state)` interface as the other
  agents, for tournaments and league opponent seats alike.

## Running things

```
python3 -m pytest tests/ -q
python3 -m scripts.smoke_test --games 200
python3 -m scripts.evaluate --seats heuristic,random,random,random --games 150

# single-policy hierarchical-model training (phase 4)
python3 -m training.train_hier --iterations 150 --num-workers 12 --eval-every 25

# league training (phase 5) -- self-play + PFSP opponents + snapshots/promotion/exploiters
python3 -m training.league_train --iterations 200 --num-workers 12 \
  --init-checkpoint <a training_hier checkpoint>.pt --eval-every 25
```

## Status

Environment + rules engine + PettingZoo wrapper (phases 1-2), the heuristic
agent + evaluation harness (phase 3), self-play PPO (phase 4), and a league
system (phase 5) are done and tested (69 passing tests).

The heuristic agent wins 100% of 150 games seated 1-vs-3 against random,
and 50% of 150 games seated 2-vs-2 against random, while staying balanced
across seats in heuristic-vs-heuristic play.

**Flat-model PPO baseline (phase 4, v1):** the loop worked and the policy
learned a real strategy within a restricted (no-trade/no-dev-card) training
distribution, but **collapsed to avg VP 2.0 the instant trading was
enabled** — isolating the cause pinned it on the action representation, not
the board: `Discrete(400)`'s "index into whatever `legal_actions()` returns
this step" has no stable meaning once the action list's composition changes
(e.g. trade actions appearing). This is exactly the flat-encoding weakness
the design doc's action-space section warned about.

**Hierarchical/pointer model (the fix):** replacing the flat head with a
type-then-pointer design (stable vertex/edge/hex/resource IDs) fixed it —
confirmed by a direct regression test and by training resolving faster and
to a better ceiling than the flat model ever reached, even at 1/3 the
iterations. Full-ruleset transfer still failed at first for a *different*,
narrower reason: `END_TURN`'s logit gets trained pathologically negative
during no-trade curriculum training (rarely optimal there) while trade-type
logits sit at random init (masked out, zero gradient) — so once trading
turns on, argmax prefers the untrained-but-mildly-positive trade type over
the trained-very-negative `END_TURN`, looping forever. Fine-tuning with
trading/dev cards/random board all enabled fixed *that* too — full-ruleset
win rate against random went from a flat 0% at every stage-1 checkpoint to
**50-80%** within 300 more iterations.

**Two efficiency fixes, both validated:** (1) an engine-level cap on
`PROPOSE_TRADE` per turn (`MAX_TRADE_PROPOSALS_PER_TURN` in `env/state.py`)
— a deliberate rule variant, not standard Catan — after a self-play episode
was observed spending 624/800 steps in trade negotiation across 156 trades
while advancing all of two turns; the cap took the same checkpoint from 2
turns/episode to 83+. (2) fork-based parallel rollout collection, ~6-7x
throughput on a 16-core machine (23.9s → 3.6s per iteration at 12 workers).

**League system (phase 5):** `training/league.py` + `training/league_train.py`
add PFSP opponent sampling, historical snapshotting, TrueSkill ratings, a
statistically-gated promotion test, and periodic main-exploiter side-training
— all mechanically validated in a 200-iteration run (self-play/opponent
mixing, snapshots, exploiter sessions, promotion evaluation all fired
correctly). That run also caught a real bug: opponents occupying 3 seats
were fed into `trueskill.rate()` as 3 independent competitors sharing one
dict key, and only the last (worst-placed) write survived — a systematic
pessimistic bias that was dragging every rated member's mu toward large
negative numbers regardless of who was winning. Fixed by collapsing each
episode to a clean trainee-vs-opponent 2-team comparison
(`episode_rating_teams`), with a regression test. The bug never touched
actual policy training (PPO trains on game rewards, not ratings) — only the
rating/reporting numbers and PFSP sampling weights were affected. No
promotion occurred in that 200-iteration run (35% win rate vs. the initial
main in a 1-vs-3, short of the 55%-with-significance bar) — an appropriately
cautious result given the bar is deliberately strict, not evidence the
system is broken.

## Known simplifications (documented in code, worth revisiting later)

- Domestic trading is a bounded variant of free negotiation: bundles of 1-3
  cards per side (`MAX_TRADE_CARDS_PER_SIDE`), no resource on both sides,
  addressed to one opponent or all; responders accept (only if they can
  pay), reject, or make one counter-offer of the same shape; at most
  `MAX_TRADE_PROPOSALS_PER_TURN` (4) proposals per turn. The policy decodes a
  bundle autoregressively (`training/hier_model.TradeCountHead`).
- `legal_actions()` caps DISCARD combinations at 100; `HierarchicalActorCritic`
  uses pointer/multi-discrete heads instead of enumeration for everything
  except DISCARD, which stays a flat index specifically because it's a
  homogeneous per-phase list (no cross-type drift risk).
- Port placement is evenly spaced around the boundary ring rather than
  replicating one specific real-world board's exact port corners.
- League exploiters are trained *sequentially* (pause main training, run a
  short side-session, resume) rather than concurrently on separate hardware
  the way AlphaStar/OpenAI Five's league does — a deliberate scope decision
  for a single CPU machine.
- The rating update treats the actively-training model's strength as
  whatever the current "main" member's rating is (a live proxy), since the
  trainee itself isn't a stable, snapshotted league member between
  promotions.
