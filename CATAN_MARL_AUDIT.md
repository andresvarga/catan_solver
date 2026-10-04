# Catan MARL Audit

Audit date: 2026-10-04 · Repository: `catan-marl/` · Branch `master` @ `6e6949a` (clean except two pre-existing,
uncommitted user edits in `scripts/collect_dagger_demos.py`, `training/dagger.py`, left untouched).
Environment: Python 3.14.4, torch 2.12.1+cu126, gymnasium 1.3.0, pettingzoo 1.26.1, numpy 2.5.0,
Linux 7.0.0-34, 16 cores, RTX 3060 Laptop (6 GB). Full baseline: `audit/results/baseline.md`.

All audit code lives in `audit/` (tests in `audit/tests/`, tools in `audit/tools/`, raw outputs in
`audit/results/`). **No production file was modified.** Rule questions were resolved against the official
2020 CATAN base rules & almanac downloaded from catan.com (`audit/results/catan_rules_2020.pdf`/`.txt`).
Every confirmed bug has a strict-xfail test that reproduces it and will flip when it is fixed.

Labels: **CONFIRMED** (reproduced with a test or recorded seed), **LIKELY**, **POSSIBLE**, **NOT REPRODUCED**.

---

## Fix Status (updated 2026-10-04)

The findings below describe the code **as audited** (`6e6949a`). Since then:

| Finding | Status | Where |
|---|---|---|
| F-03 off-turn win | **Fixed**: only the turn owner can win; a player who reaches 10 VP on another player's turn wins at the start of their own turn | commit `537a79a` |
| F-04 counter-offer invisible | **Fixed**: counter give/want/author in both encoders | `537a79a` |
| F-06 truncation pays a win | **Fixed**: 0 reward + V(s_T) bootstrap by default; step caps default 4000 | `537a79a` |
| F-07 no pre-roll dev cards | **Fixed** | `537a79a` |
| F-08 Road Building lock | **Fixed** (unplaceable free roads are forfeited) | `537a79a` |
| F-14 seat-0-only eval | **Fixed**: seat rotated by seed, Wilson CIs printed | `537a79a` |
| F-15 short credit horizon | **Fixed**: γ=0.999, λ=0.98, forced moves not stored | `537a79a` |
| F-18 lost runs | **Mitigated**: `manifest.json` per run (old artifacts unrecoverable) | `537a79a` |
| F-20 1:1-only trading | **Fixed (bounded variant)**: 1-3 cards per side, targeted or broadcast, multi-card counters, 4 proposals/turn, structured actions + autoregressive bundle head | working tree (uncommitted) |
| F-13 ACCEPT without paying | **Fixed** as part of F-20 | working tree |
| F-05 no step validation | **Partly fixed**: `CatanEngine.step` and `CatanAECEnv.step` validate; raw `engine.step` stays unchecked for simulation speed | working tree |
| F-01, F-02, F-10, F-11, F-12, F-16, F-17, F-19, F-21–F-29 | Open | — |

## Executive Summary

**Overall:** the engine is much better than most hobby Catan simulators. Information hiding is clean, action
masking is airtight, and Longest Road is correct (checked against an independent oracle on ~52,000 networks).
I found no P0 issue: nothing lets the agent see hidden information, take an illegal action through the policy,
or farm reward in a way that would invalidate past results.

It is still **not "rules-correct standard Catan"**, as the README claims. The confirmed deviations are:

- resource non-conservation and a bank that can go negative;
- wins awarded off-turn;
- no pre-roll development cards;
- a turn lock after Road Building;
- 1:1-only player trading with a 2-proposal-per-turn cap;
- 6/8 tokens adjacent on about 87% of boards.

The ML-side problems that matter most are:

- **Evaluation and selection are biased.** In-loop evals always seat the model at seat 0, which is measurably
  the best seat, while the code and memory describe them as seat-rotated.
- **Credit assignment is very weak.** γ=0.99 per own decision over about 226 decisions per player per game
  passes only ~10% of the terminal reward back to the opening placements.
- **The reward does not target winning.** It is rank- and VP-based, and truncation pays a VP leader the full
  win reward. With the CLI's own defaults, 90–98% of games are truncated.
- **Counter-offers are invisible to the policy** that must accept or reject them.
- **The claimed champion cannot be verified.** Its weights and training pools no longer exist on disk.

| Area | Grade | One-line justification |
|---|:-:|---|
| Catan Rules Correctness | **C+** | Core mechanics right; 10 confirmed deviations, 3 affect strategy (pre-roll dev cards, trading model, Road Building lock) |
| Environment/API Correctness | **B+** | Official `api_test` + `seed_test` pass; reward bookkeeping correct; `step()` has no legality guard |
| Information Integrity | **A−** | Counterfactual leakage tests pass under every encoder; minus: counter-offer omitted (under-information, not leakage); baseline heuristics read hidden state |
| Action Space | **B** | Pointer heads always produce legal actions; 1:1-trade-only + discard cap + 2-proposal cap make some legal Catan actions unrepresentable |
| Reward Design | **C** | Rank reward ≠ win objective; truncation pays win reward; γ horizon far too short for Catan |
| MARL Architecture | **B−** | PPO math verified (log-prob/value recomputation exact); no centralized critic, no memory; GNN encoder is sound |
| Self-Play | **C−** | Champion line is BC + PPO vs a *fixed* heuristic opponent; league promotion test is practically unreachable |
| Evaluation | **C** | Good habits in memory (fresh seeds, n≥600), but in-repo evals are seat-0-only, no CIs, random-opponent tier is uninformative |
| Reproducibility | **D+** | Sequential training bit-reproducible; parallel rollouts (all real runs) are not; champion artifacts deleted |
| Testing | **B−** | 121 tests, mostly behavioural; no conservation/off-turn-win/counter-visibility tests; one test encodes an incomplete rule |
| Performance | **C+** | Engine fast (~38k steps/s/core); batch-1 policy inference dominates (GNN rollout ~30× slower than flat model) |
| Maintainability | **B** | Clear modules, unusually good docstrings; functional engine with no guards; README stale (says 69 tests) |

---

## Critical Findings (P0)

None confirmed. No evidence of hidden-information leakage, illegal actions reachable by the policy, evaluation
contamination by gradient updates, or reward farming that would invalidate existing results. Several P1 items
below **would** be P0 if the CLI defaults were used for a headline run (see F-06).

---

## High-Priority Findings (P1)

### F-14 — In-loop evaluation is seat-0 only; seat 0 is the strongest seat — CONFIRMED
- **Evidence:** `training/train_hier.py:_play_eval_game` (lines 74-95): `agents = {0: HierarchicalLearnedAgent(...)}`,
  docstring "Trainee (deterministic, seat 0)". `training/rl_finetune.py:fixed_seed_eval` → `evaluate_policy`
  inherits this, while the module docstring (lines 25-27) says "the eval protocol rotates the model's seat by
  seed". `best_eval.pt` is selected on this number (lines 196-206).
- **Reproduction:** `audit/tools/tournament.py` heuristic×4, 1000 board seeds × 4 seats (4000 games, base 20.1M):
  seat 0 wins **28.3%** of all games, seats 1/2/3 = 24.5/22.9/24.3% (χ²(3)=25.7, p≈1e-5). The search agent vs
  3 heuristics: 64.4% from seat 0 vs 54.8–57.2% from seats 1–3 (`t_search_vs_heuristic.json`).
- **Impact:** In-loop win rates are inflated relative to rotated evaluation by roughly +3–7pp, depending on
  the agent. Selecting `best_eval.pt` from 40 noisy seat-0 looks compounds the winner's curse already noted in
  memory. In-loop numbers cannot be compared with seat-rotated confirmatories.
- **Fix:** rotate `trainee_seat = seed % 4` in `_play_eval_game` (as `league_train._play_vs_member_game`
  already does), or play every seed from all 4 seats (as `audit/tools/tournament.py` does).
- **Regression test:** assert that `evaluate_policy` reports per-seat counts with each seat at n/4.

### F-15 — Discounting destroys long-horizon credit — CONFIRMED (measurement)
- **Evidence:** `training/ppo.py` `GAMMA=0.99`, `GAE_LAMBDA=0.95`, applied per *own* transition
  (`hier_ppo.collect_episode` records every decision, including forced ones).
- **Reproduction:** `audit/tests/test_policy_pipeline_audit.py::test_effective_discount_of_terminal_reward`
  measured a median of 230 non-forced decisions per player per heuristic game. Self-play rollouts with the
  audit checkpoint averaged 226 transitions per agent per episode, 19.9% of them forced single-option moves.
  0.99^226 ≈ **0.10**. GAE(λ=0.95) shrinks the signal further: (0.99·0.95)^k reaches about 1e-5 after 226 steps.
- **Impact:** The opening settlements, the most consequential decisions in Catan, receive about one tenth of
  the terminal signal, and in effect nothing through GAE. The policy is pushed toward short-term VP and
  production, so value estimates for early states are dominated by bias. This is the most plausible
  structural cause of the RL plateau recorded in memory, where three RL legs ended null.
- **Fix:** γ≥0.999 (or γ=1 with a terminal-only reward), λ≈0.98–1.0; drop forced single-action transitions
  from the trajectory (they carry no policy gradient but still consume discount); or discount per *turn*
  instead of per decision.
- **Regression test:** assert that γ^(median own-decisions) ≥ 0.5 in the training config.

### F-06 — Truncation pays the full win reward; default step caps truncate most games — CONFIRMED
- **Evidence:** `env/pettingzoo_env.py:194-196` calls `_assign_terminal_rewards()` on truncation ("rank on
  current standing"). `rank_reward={0:1.0,…}` gives the VP leader exactly the reward of a real win.
  `compute_gae` treats truncation as terminal with no bootstrap (`training/ppo.py:25-33`). Defaults:
  `train_hier --max-episode-steps 600`, `rl_finetune --max-episode-steps 800`.
- **Reproduction:** `test_env_api_audit.py::test_truncation_pays_win_reward_to_vp_leader`: a leader with <10 VP
  receives +1.0. Heuristic games last a median of **1,109 steps** (p10 806, p90 1452; 400 games). **90.5% exceed
  800 steps, 98% exceed 600, 0% exceed 2500.**
- **Impact:** With default flags, training optimises "lead on VP at step 600/800". That is a different game
  from Catan: it favours early settlements over cities, development cards, and Largest Army. Memory says the
  champion legs used 2500, which is safe. The project's documented quick-start (`README` →
  `train_hier --iterations 150 …`) uses the 600 default.
- **Fix:** raise defaults to ≥4000 (or no cap); on truncation give 0 reward and bootstrap V(s_T), the
  standard treatment; or a small draw penalty. Never pay +1 without a win.
- **Regression test:** assert `rewards[leader] < 1.0` when truncated without a winner.

### F-04 — Counter-offer terms are invisible to the player who must accept/reject them — CONFIRMED
- **Evidence:** `env/state.py:134` `trade_counter_context`. Neither `env/pettingzoo_env.build_observation`
  (lines 360-369) nor `training/graph_features.build_graph_observation` (lines 166-173) reads it; both encode
  only `pending_trade`. `engine.acting_player` hands the decision to the proposer
  (`engine.py:351-355`), and ACCEPT executes the counter's terms (`engine.py:757-767`).
- **Reproduction:** `test_obs_leakage.py::test_counter_offer_terms_visible_to_proposer` (xfail). Two counters
  with different terms produce byte-identical observations for the proposer.
- **Impact:** A learned proposer accepts or rejects counters blind. Heuristic opponents read the counter
  directly from `state` (`agents/heuristic.py:369-374`), so learned agents are systematically disadvantaged.
  In self-play, counter-trading becomes noise rather than a skill.
- **Fix:** add `counter_give`, `counter_want`, and the relative seat of `counter_proposer` to both encoders.
- **Regression test:** the xfail above, flipped to a pass.

### F-07 — Development cards cannot be played before rolling — CONFIRMED (documented simplification)
- **Evidence:** `engine.legal_actions` returns `[ROLL_DICE]` only in `Phase.ROLL` (line 410-411). Official rules:
  "You may play a development card at any time, even before you roll the dice."
- **Reproduction:** `test_rules_audit.py::test_knight_playable_before_roll` (xfail).
- **Impact:** The common opening move of knighting the robber off your own hex before rolling is impossible,
  which removes real strategy and changes Largest Army timing.
- **Fix:** in `Phase.ROLL`, add the dev-card plays alongside ROLL_DICE. A knight played pre-roll moves the
  robber and returns to ROLL. A pre-roll Road Building or Year of Plenty needs ROLL-phase handling.

### F-20 — Player trading is heavily simplified — CONFIRMED (documented)
- **Evidence:** `engine.legal_actions` enumerates only 1-for-1 proposals to *all* opponents (lines 490-498).
  Counters are 1-for-1 too (lines 442-448). `MAX_TRADE_PROPOSALS_PER_TURN = 2` (`state.py:16`). The policy
  heads mirror this (`hier_model.RESOURCE_PAIR_TYPES`).
- **Impact:** Multi-card trades (2:1 asks, 1-for-2 offers), targeted offers, and open negotiation cannot be
  represented. Domestic trade is the central social mechanic of Catan, and the learned game is "Catan with
  vestigial trading". Results must not be described as standard Catan.
- **Fix (research-level):** factorised give/want multisets (per-resource count heads), target-player selection,
  and an explicit budget (e.g. ≤3 proposals) declared as the variant.

### F-18 — Claimed champion and training pools no longer exist — CONFIRMED
- **Evidence:** memory records the champion at `~/.claude/jobs/aa3968d4/tmp/imitation_gnn/big/rl_long2/best_eval.pt`
  plus several pools and anchors. `find /home/andres -name '*.pt'` finds no Catan checkpoints, and the
  `.gitignore`d `checkpoints_*` directories are absent.
- **Impact:** The headline result (57.2% vs 3 heuristics) cannot be re-verified, re-evaluated per seat, or
  used as an init. The whole BC → DAgger → RL lineage would have to be regenerated.
- **Fix:** keep experiment outputs under the repo (git-ignored) or another durable path, with a manifest
  (args, git SHA, seeds, eval JSON) per checkpoint.

### F-03 — A player can win on another player's turn — CONFIRMED
- **Evidence:** `engine.check_win` (lines 298-303) scans every player in id order after any VP-changing step.
  Official rules: "You can only win during your turn. If … you have 10 victory points during another player's
  turn, you must wait until your next turn."
- **Reproduction:** `test_rules_audit.py::test_cannot_win_on_another_players_turn` (xfail) is a deterministic
  construction. Player 0's settlement breaks player 1's road, Longest Road passes to player 2 (with 8 hidden
  VP), and player 2 is declared winner during player 0's turn. Fuzzing found 15/10,000 random games
  (`off_turn_win_seeds` in `fuzz_random_10000_rel.json`, e.g. seed 11000205) and 4/2,000 heuristic games
  (e.g. seed 12000927).
- **Impact:** Rare (~0.2%), but it is a wrong victory condition and gives a terminal reward to the wrong
  player. Rated P1 rather than P0 only because of its frequency.
- **Fix:** `check_win` should consider only `state.current_player`, and must also run at the start of each
  player's turn (on ROLL) so a pending off-turn 10 VP is claimed then.

---

## Medium Findings (P2)

| ID | Finding | Status | Evidence / reproduction | Impact | Fix |
|---|---|---|---|---|---|
| F-01 | Setup starting resources created from nothing (bank not debited) | CONFIRMED | `engine.py:592-600` (no `state.bank[r] -= 1`); `test_setup_grant_debits_bank`; fuzz: conservation broken in 89/100 games from step ~10; bank reaches 24/19 | Up to 12 extra cards in circulation; bank-shortage rule (F-11) almost never fires; public card-count totals wrong | debit bank in setup grant |
| F-02 | Maritime trade ignores bank stock → bank negative, card minted | CONFIRMED | `engine.py:481-488,732-738`; `test_maritime_trade_requires_bank_stock`; 65/10,000 random games, 1/2,000 heuristic (seed 12001242 step 906) | Duplication path (rare); invalid state | require `bank[receive] >= 1` |
| F-05 | `engine.step()` performs no legality validation | CONFIRMED | `engine.py:572-823`; `test_engine_step_rejects_illegal_action` (BUILD_CITY with no cards → negative hand, VP +1) | Not reachable via env/policy (index-into-legal-list) but every agent/eval/search path calls the engine directly; a buggy agent silently corrupts state | `step(..., validate=True)` default checking `action in legal_actions` (or a cheaper per-type predicate) |
| F-08 | Road Building locks the turn when <2 roads placeable | CONFIRMED | `engine.py:460-465` (`free_roads_remaining>0` hides everything else); `test_road_building_does_not_lock_turn` | Can't build/trade after RB with 1 piece left or 1 legal edge; also cannot decline 2nd road except by END_TURN | clear `free_roads_remaining` when no legal free road; allow skipping |
| F-10 | 6/8 tokens adjacent on ~87% of boards | CONFIRMED | `board.generate_board` shuffles tokens unconstrained; 868/1000 boards; `test_board_red_numbers_not_adjacent` | Training distribution ≠ official variable set-up; red-number clusters inflate robber/expansion value | reject/resample until no adjacent 6/8 |
| F-13 | ACCEPT_TRADE legal for a responder who can't pay; CONFIRM then voids silently | CONFIRMED | `engine.py:437-439,794-803`; `test_accept_requires_ability_to_pay`, `test_griefing_accept_cannot_burn_proposal_slots` | Griefing vector in self-play (burn the proposer's 2 proposals); noise in trade learning | only offer ACCEPT if responder holds `want` |
| F-16 | Reward optimises rank/VP, not wins | CONFIRMED (design) | rank reward {1,0,−0.5,−1} + VP shaping (`pettingzoo_env.py:205-243`); `train_hier --vp-shaping-weight 0.05` default | 2nd vs 4th place differs by 1.0 — agent pays win probability for placement; win rate is the reported metric | terminal win/loss (+1/−1/3 per loser) for strength runs; anneal shaping to 0 |
| F-17 | League promotion test practically unreachable | CONFIRMED (code) | `league.promotion_test` requires ≥55% wins and binomial p<0.05 vs p0=**0.5** in a 1-vs-3 game where parity is 25% | Main never changes ⇒ PFSP/history machinery idles; README's 200-iter run had no promotion | test vs p0=0.25 (or a head-to-head 2-seat format) |
| F-19 | "vs random" evaluation is uninformative | CONFIRMED | Untrained hier model 10.5% [8.8,12.5] vs 3 random; after **2** PPO iterations (96 self-play games) 80–90%; heuristic 100% (1000/1000) | Win rate vs random saturates almost immediately; can't rank policies | drop as a strength tier; keep as smoke test |
| F-22 | Parallel rollouts not reproducible | CONFIRMED | `hier_ppo.reseed_forked_worker` salts with `os.getpid()`; `audit/tools/repro_check.sh`: sequential same-seed runs bit-identical, `--num-workers 4` same-seed runs differ in metrics and weights | Every real training run is unrepeatable | salt with worker index + iteration + base seed |
| F-24 | Dangerous CLI defaults | CONFIRMED | `train_hier`: `--randomize-board` default False (one fixed board for every game), `--max-episode-steps 600` (98% truncation, F-06), `--num-workers 1` | README's quick-start trains on a single board where almost every game is truncated | defaults: random board, ≥4000 steps |
| F-25 | GNN rollout throughput ~20× below flat model; GNN unusable on CPU for updates | CONFIRMED | see Performance; memory records an 18 h CPU PPO update | Limits GNN RL to ~10⁶ steps/h | batched inference across envs (vectorised workers + one GPU inference server) |
| F-12 | DISCARD enumeration capped at 100 combos (deterministic order) | CONFIRMED | `engine.py:413-418`, `hier_model.DISCARD_INDEX_SIZE=100`; `test_discard_all_combinations_representable` (4 of each, discard 10 ⇒ 101 legal) | Rare (needs ~18+ cards), but truncation drops the *last* combos in enumeration order (those discarding the most ore) | per-resource count heads for discard |
| F-26 | Flat model encodes owners as absolute-seat scalars | CONFIRMED (design) | `training/model.flatten_observation` lines 28,32: `(owner+1)/4` | Flat model must learn seat-specific decoding; GNN path is relative (fine) | relative one-hot ownership |

## Low-Priority Findings (P3)

| ID | Finding | Status | Evidence | Notes |
|---|---|---|---|---|
| F-09 | Heuristic baseline & search teacher read hidden information | CONFIRMED, **not material** | `agents/heuristic.py:151` (`total_vp` incl. hidden VP for robber target), `:242` (true opponent hands for Monopoly); `search_heuristic.eval_state:160` (opponents' hidden VP) | `HonestHeuristic` (public-info replacements, `audit/tools/tournament.py`) vs 3 originals: **25.3% [24.0, 26.6]** over 4000 seat-balanced games = parity ⇒ cheating gives no measurable edge. Still fix for hygiene; a BC teacher must be imitable |
| F-11 | Bank-shortage single-player exception missing | CONFIRMED | `engine.py:328-332`; `test_bank_shortage_single_player_gets_remainder`; existing `tests/test_engine.py::test_bank_shortage_rule_blocks_everyone` encodes the incomplete rule | rare |
| F-21 | BC validation split by decision, not by game | CONFIRMED | `training/bc_pretrain.py:88-89` random row permutation | val NLL optimistic (correlated rows) |
| F-23 | RNG stream coupling | CONFIRMED | dice `Random(seed)` = board `Random(seed)`; deck `Random(seed+1)` = next game's board stream | not a practical leak; use `SeedSequence.spawn` |
| F-27 | Bank stock and deck size absent from observations | CONFIRMED | `build_observation` | both are public in Catan; deck size derivable, bank not (because of F-01) |
| F-28 | Port positions evenly spaced, not official frame | CONFIRMED (documented) | `board._assign_ports` spacing 30//9=3 → one 6-edge gap | minor distribution shift |
| F-29 | README stale/overclaims | CONFIRMED | "rules-correct, 4-player standard Catan"; "69 passing tests" (actual 121) | |

---

## Architecture Overview

```
env/board.py      generate_board()  19 hex / 54 vertex / 72 edge graph, ports, tokens (topology ids are seed-invariant)
      ↓
env/state.py      GameState, PlayerState, Phase{SETUP_SETTLEMENT,SETUP_ROAD,ROLL,DISCARD,MOVE_ROBBER,MAIN,TRADE_RESPONSE,GAME_OVER}
      ↓
env/engine.py     legal_actions(state) → [Action];  step(state, action, rng) (functional, unvalidated);  acting_player()
                  production, robber, trade protocol, longest road DFS, largest army, check_win, public card-count estimates
      ↓
env/pettingzoo_env.py  CatanAECEnv (AEC): agent_selection = acting_player; Discrete(400) index into legal list;
                  rewards = rank-based terminal (+ optional VP shaping); truncation = rank on standing
      ↓
Observation       flat: build_observation → training/model.flatten_observation  (absolute seat ids)
                  graph: training/graph_features.build_graph_observation (relative seats, per-node features)
      ↓
Policy            training/hier_model.HierarchicalActorCritic (MLP trunk)  |  training/gnn_model.GraphActorCritic
                  (R-GCN-style message passing over training/board_topology); type head → pointer heads
                  (vertex/edge/hex/player/resource/resource2/discard_index); masks built from the legal list
      ↓
Action            match_action() picks the concrete legal Action → env.step(legal.index(action))
      ↓
Rollout/storage   training/hier_ppo.collect_episode (per-agent "reward since last acted"), fork-parallel workers
      ↓
Returns           training/ppo.compute_gae per agent trajectory, truncation = terminal
      ↓
Update            hier_ppo.ppo_update: clipped surrogate (joint factorised log-prob), MSE value, summed head entropy,
                  adv-normalisation, grad-clip 0.5, target-KL early stop, optional BC-anchor NLL
      ↓
Drivers           train_hier (self-play) · league_train (PFSP league) · bc_pretrain / dagger (imitation of
                  agents/search_heuristic) · rl_finetune (PPO vs 3 fixed HeuristicAgents, BC anchor)
      ↓
Checkpoint        torch.save({"model","optimizer","iteration","args"}) latest.pt / iter_N.pt / best_eval.pt
      ↓
Evaluation        train_hier.evaluate_policy (seat 0 only!), league_train (seat-rotated), scripts/evaluate.py,
                  ad-hoc confirmatories (not in repo)
```

**What the project actually optimises (Phase 2).** The design calls this "self-play MARL", but the strongest
line in practice (memory) is: imitation of a hand-built search heuristic (BC+DAgger), then PPO fine-tuning
against **three fixed copies of `HeuristicAgent`**, with the BC anchor. The objective actually measured is
"win rate in a 1-vs-3 game against one specific heuristic opponent". This is a strong-policy-vs-fixed-bot
objective, not equilibrium play, not robustness to diverse opponents, and not human-level play on standard
Catan. The README's "rules-correct, 4-player standard Catan" overstates the simulator (see Known
Simplifications). No document claims a "solved" game, which is appropriate.

**State machine (reconstructed from `engine.step`).**
```
SETUP_SETTLEMENT ⇄ SETUP_ROAD  (order 0,1,2,3,3,2,1,0; 2nd road grants resources)  → ROLL(p0)
ROLL --roll≠7--> MAIN
ROLL --roll=7--> DISCARD (each >7 player, in seat order) --> MOVE_ROBBER --> MAIN
      (no discarders)  ----------------------------------> MOVE_ROBBER
MAIN --PROPOSE_TRADE--> TRADE_RESPONSE (targets in seat order: ACCEPT/REJECT/COUNTER)
        COUNTER --> proposer ACCEPT/REJECT counter --> MAIN
        all responded, ≥1 accept --> proposer CONFIRM(target)/CANCEL --> MAIN ; none --> MAIN
MAIN --PLAY_ROAD_BUILDING--> MAIN[free roads only + END_TURN]   (F-08)
MAIN --END_TURN--> ROLL(next player)
any VP-changing step --check_win (any player!)--> GAME_OVER     (F-03)
```
Phases are explicit (`Phase` enum). I found no reachable state with multiple rolls, a skipped mandatory
discard or robber move, a stuck state, or a skipped player: 12,000 fuzzed games with invariant checks after
every step, plus `test_roll_only_once_and_main_has_no_roll` and `test_seven_discard_threshold_and_order`.

---

## Catan Rules Coverage

| Rule / Mechanic | Implemented | Tested (audit) | Correct | Notes |
|---|:-:|:-:|:-:|---|
| 19 hexes, 4/3/4/4/3/1 terrain, 18 tokens | ✔ | ✔ 200 seeds | ✔ | |
| Topology 54 vertices / 72 edges, adjacency | ✔ | ✔ | ✔ | degrees {2:18,3:36}; ids stable across seeds |
| Robber starts on desert | ✔ | ✔ | ✔ | |
| 9 harbors (4×3:1, 5×2:1), 2 coastal vertices each | ✔ | ✔ | ✔ | positions not official (F-28) |
| 6/8 not adjacent (random set-up) | ✘ | ✔ | ✘ | F-10 |
| Snake setup order, settlement→adjacent road | ✔ | ✔ | ✔ | |
| Setup distance rule | ✔ | ✔ | ✔ | |
| 2nd-settlement starting resources | ✔ | ✔ | ✘ | bank not debited (F-01) |
| Roll once; 7 → discard ⌊n/2⌋ if >7; all players; before robber | ✔ | ✔ | ✔ | |
| Robber must move; victims adjacent; random steal | ✔ | ✔ (4000 steals, χ ok) | ✔ | 0-card victim selectable (allowed) |
| Production incl. city ×2, robber block, desert | ✔ | ✔ | ✔ | |
| Bank shortage | partial | ✔ | ✘ | single-player exception missing (F-11) |
| Road/settlement/city/dev costs | ✔ | ✔ | ✔ | |
| Piece limits 15/5/4, settlement returned on city | ✔ | ✔ | ✔ | |
| Road connectivity, opponent-vertex blocking | ✔ | ✔ | ✔ | |
| Dev deck 14/5/2/2/2 | ✔ | ✔ | ✔ | |
| Not playable turn bought; 1 per turn; VP exempt | ✔ | ✔ | ✔ | |
| Play dev card before rolling | ✘ | ✔ | ✘ | F-07 |
| Knight → robber + Largest Army | ✔ | ✔ | ✔ | |
| Road Building | ✔ | ✔ | partial | turn lock (F-08) |
| Year of Plenty (bank-limited) | ✔ | ✔ | ✔ | |
| Monopoly | ✔ | ✔ | ✔ | |
| Maritime 4:1 / 3:1 / 2:1 | ✔ | ✔ | partial | no bank-stock check (F-02) |
| Domestic trade | simplified | ✔ | ✘ (variant) | 1:1, broadcast, 2/turn, ACCEPT w/o cards (F-13, F-20) |
| Longest Road length (trails, branches, loops, blocking) | ✔ | ✔ oracle | ✔ | 0 mismatches / ~48k networks + 300 hand-built |
| Longest Road ≥5, ties keep holder, broken→set aside on tie | ✔ | ✔ | ✔ | matches almanac "Special Case" |
| Largest Army ≥3 played knights, strict transfer | ✔ | ✔ | ✔ | |
| VP sources incl. hidden VP cards; ≥10 wins | ✔ | ✔ | ✔ | |
| Win only on own turn | ✘ | ✔ | ✘ | F-03 |
| No mutation after GAME_OVER | ✔ | ✔ | ✔ | legal_actions empty; env dead-step requires None |

**Longest Road** (Phase 11): `compute_longest_road_length` is an edge-simple trail DFS from every vertex. It
refuses to pass *through* an opponent-occupied vertex but allows a trail to start or end there. The audit's
oracle (`audit/helpers.reference_longest_road`) is independent of the DFS. It enumerates edge subsets and
accepts those that are connected with 0 or 2 odd-degree vertices, where an opponent-blocked vertex may only be
an endpoint and a circuit may be anchored at a single blocked vertex. Results:
- Engine and oracle agree on every end-of-game network in 10,000 random and 2,000 heuristic games, plus 300
  random hand-built networks (5–12 edges, 0–2 blockers).
- The one early disagreement (seed 10000789) was a bug in **my** oracle: it allowed a circuit through two
  blocked vertices. The engine was right.
- Transfers, tie retention, and the "broken road, other players tie → set aside" case are tested explicitly.

Verdict: **correct**.

---

## Observation-Space Audit

Counterfactual method (`audit/tests/test_obs_leakage.py`), run on 40 mid-game heuristic states × 4 observers ×
3 re-deals:
1. Re-deal everything observer A cannot see: opponents' resource identities (hand sizes kept), opponents'
   dev-card identities (counts and bought-this-turn counts kept), and deck order.
2. Assert that A's flat dict, flattened vector, graph features, **and legal-action list** are byte-identical.

This passes with `public_hand_features` both off and on. A guard test confirms the re-deal really changes the
hidden state. Dynamic tests confirm the public card-count estimates are independent of a stolen card's
identity and of discard contents.

| Feature | Public | Private to player | Hidden | Included? | Correct? |
|---|:-:|:-:|:-:|:-:|:-:|
| Terrain, numbers, ports, robber | ✔ | | | ✔ | ✔ |
| Buildings / roads (owner, type) | ✔ | | | ✔ (flat: absolute seat; graph: relative) | ✔ |
| Own resources, own dev cards (+ playable) | | ✔ | | ✔ | ✔ |
| Own hidden VP | | ✔ | | graph ✔ / flat via dev cards | ✔ |
| Opponent hand sizes, dev-card counts, knights, piece counts, visible VP, awards | ✔ | | | ✔ | ✔ |
| Opponent resource identities | | | ✔ | ✘ (only public card-count estimate, opt-in) | ✔ |
| Opponent dev-card identities / hidden VP | | | ✔ | ✘ | ✔ |
| Deck order, RNG state, future dice | | | ✔ | ✘ | ✔ |
| Stolen-card identity (to third parties) | | | ✔ | ✘ (estimate uses expectation) | ✔ |
| Dice result, phase, current/acting player | ✔ | | | ✔ | ✔ |
| Pending proposal (give/want/proposer) | ✔ | | | ✔ | ✔ |
| **Counter-offer terms** | ✔ | | | **✘** | ✘ (F-04) |
| Bank stock | ✔ | | | ✘ | omission (F-27) |
| Dev deck size | ✔ | | | ✘ (derivable) | ok |
| Action history / previous robber targets / trade history | ✔ | | | ✘ (only via card-count estimate) | omission |

**Actor/critic boundary (Phase 22):** actor and critic share one trunk and one input. There is no
centralised critic and no privileged tensor anywhere in the rollout or update path. Training rollouts
(`collect_episode`) build the observation from `env.engine.state` through the same `build_graph_observation`
used at evaluation. **No leakage.**

**Partial observability (Phase 21):** no recurrence or history. The opt-in public card-count estimates are a
hand-built belief state over resources, and they are the right minimal fix. Still missing: inferring
opponents' dev cards from purchase timing, robber/trade history, and counter-offers (F-04). A recurrent or
transformer policy is not obviously required. Adding a short public-event history (last K robber targets,
trades, dev purchases) is the cheaper next step.

---

## Action-Space Audit

| Action Category | Encoding (policy) | Legal mask | Tested | Notes |
|---|---|---|---|---|
| Setup settlement / road | type + vertex / edge pointer | from legal list | ✔ | |
| Roll dice | type only | ✔ | ✔ | forced (single option) |
| Build road / settlement / city | type + edge / vertex pointer | ✔ | ✔ | |
| Buy dev card | type | ✔ | ✔ | |
| Knight / Move robber | type + hex + player(victim or "none") | stage-2 conditioned on hex | ✔ | |
| Road Building / YoP / Monopoly | type / resource pair / resource | ✔ | ✔ | YoP has two index paths per unordered pair (benign) |
| Discard | flat index into ≤100 enumerated combos | ✔ | ✔ | F-12 cap |
| Maritime trade | give + receive resources | ✔ | ✔ | F-02 bank stock |
| Propose / counter | 1:1 give + want | ✔ | ✔ | F-20 unrepresentable multi-card trades |
| Accept / reject / confirm / cancel | type / player pointer | ✔ | ✔ | F-13 |
| End turn | type | ✔ | ✔ | |

- Every action chosen by `act()` comes from `legal_actions()` (`match_action` searches the legal list):
  12,000 sampled decisions, 0 illegal, all log-probs finite
  (`test_masked_heads_never_select_illegal_and_no_nan`).
- Masked logits use −1e9 *before* softmax/sampling, consistently in `act` and `evaluate_actions`. Rollout
  log-probs and values match the PPO recomputation within 1e-4 for both models (≥2,000 transitions each,
  `test_rollout_logprobs_match_evaluate_actions_and_actions_legal`).
- The env's `Discrete(400)` mask is `index < len(legal)`. That's fine for the API, but its meaning is unstable
  across steps (the README already documents this). The pointer models don't use it.
- Out-of-range indices raise `ValueError` with no state change. Illegal `Action` objects passed straight to
  `engine.step` are **accepted and corrupt state** (F-05).
- Max legal-list length seen in 12,000 games was 100 (the discard cap), well under 400.

---

## Reward Audit

For the agent of player *i* (`env/pettingzoo_env.py`):

r_t^i = w · ΔVP_t^i  +  1[terminal or truncated] · R_rank(i)

R_rank: 1st **+1.0**, 2nd **0.0**, 3rd **−0.5**, 4th **−1.0**. Groups tied on VP share the mean (rank
derives from total VP, hidden VP included, not from `state.winner`). w = `vp_shaping_weight` (`train_hier`
default 0.05; `rl_finetune` uses 0).

| Event | Reward |
|---|---:|
| Win (reach 10 VP on any turn — F-03) | +1.0 |
| Truncation while VP leader | **+1.0** (F-06) |
| 2nd / 3rd / 4th at end | 0 / −0.5 / −1.0 |
| Any VP gained / lost (settlement, city, LR/LA gain or loss, VP card) | ±w per VP |
| Road, dev purchase, resource gain, trade, illegal action, turn length | 0 |

Reward-hacking analysis:
- **VP shaping** telescopes to w·(final VP), so it can't be farmed. Award ping-pong is zero-sum. Under γ<1 it
  rewards *early* VP over late VP, which pushes toward settlement spam rather than cities and development
  cards.
- **Rank reward** values 2nd over 4th by a full win's margin, so the agent will trade win probability for
  placement.
- **Truncation** pays the win reward to the leader, so stalling while ahead is rewarded. Stalling is bounded
  inside a turn (a maximally stalling policy is forced to end its turn within 19 steps; see
  `test_turn_cannot_be_stalled_indefinitely`). Across turns it is only mildly exploitable at a 2500-step cap,
  but dominant at the 600/800 defaults (F-06).
- **No illegal-action penalties** exist and none are needed, since actions are always legal.
- **Trade loops:** proposals are capped and maritime trades always lose cards. No resource duplication along
  any trade-protocol path (`test_no_resource_duplication_via_trade_paths`). The one minting path is F-02.

Does training reward track win rate? In pure self-play the per-episode return is fixed by construction
(rank rewards sum to −0.5, plus symmetric shaping). Mean training reward therefore **cannot** rise, and
`train_hier` doesn't log it. Only held-out evaluations measure progress (see Training Results).

---

## MARL Algorithm Audit

- **Algorithm:** PPO with **full parameter sharing across all four seats** (one network acts for every seat
  using its own seat-relative view). It is independent-learner PPO, not MAPPO: no centralised critic, no
  recurrence.
- **Equations checked line by line** (`training/ppo.py`, `training/hier_ppo.py`):
  - GAE: δ_t = r_t + γV_{t+1}(1−d_t) − V_t; A_t = δ_t + γλ(1−d_t)A_{t+1}; R_t = A_t + V_t. Correct.
  - Clipped surrogate on the joint factorised log-prob (type + active sub-heads). Correct, verified
    numerically.
  - Unclipped MSE value loss; entropy = sum of entropies of active heads (so it scales with the number of
    active heads, a mild bias toward exploring pointer decisions); advantage normalisation per rollout;
    grad-clip 0.5; KL early stop on |mean approx_kl|. Adam.
- **Trajectory segmentation:**
  - Each agent's transitions form a separate GAE sequence, and reward between two of its own decisions is
    attributed to the earlier one through `last()` + `clear_reward`.
  - Every agent's final transition has `done=True` and receives its terminal reward; sums match the rank
    table (`test_terminal_rewards_delivered_to_every_agent`, 20 games).
  - No cross-player or cross-episode contamination was found.
- **Issues:** γ horizon (F-15); truncation (F-06); forced single-option transitions stored and discounted
  (19.9% of rollout transitions); no gradient-norm logging; and `rl_finetune` selects best checkpoints on a
  biased, noisy metric (F-14).

---

## Self-Play Audit

- **`train_hier`** is pure shared-parameter self-play. It has no opponent pool, which makes it vulnerable to
  cycling and forgetting.
- **`league_train`** adds PFSP, snapshots, and exploiters, but the promotion gate is effectively unreachable
  (F-17), so the main never changes.
- **`rl_finetune`**, which produced the champion, is not self-play: it is PPO against **three frozen
  `HeuristicAgent`s**. "Improvement" in that line means exploiting one fixed opponent. No evidence exists in the
  repo for robustness against other styles (search agent, earlier checkpoints, mixed tables).
- The self-play mini-run (below) shows non-monotonic progress against fixed references.

---

## Controlled Mini Training Experiment & Checkpoint Ladder (Phases 31, 40)

The historical checkpoints are gone (F-18), so I trained fresh ones. Command (user-approved compute):
`training.train_hier --model-type hier --hidden 256 --randomize-board --iterations 250 --episodes-per-iter 48
--num-workers 12 --eval-every 25 --eval-games 60 --max-episode-steps 2500 --seed 0`. It ran pure
shared-parameter self-play, full rules, random boards, VP shaping 0.05 (default), and a GPU update.
Checkpoints are in `audit/results/ckpts/run_a/`; the log is `audit/results/logs/train_run_a.log`.

| iter | transitions | turns/game | finish | value loss | entropy | approx KL | in-loop vs random (seat 0; same-dist / full) | in-loop vs heuristic (seat 0) |
|---:|---:|---:|---:|---:|---:|---:|---|---|
| 1 | 82,823 | 284 | 85% | 0.0115 | 1.520 | 0.0055 | — | — |
| 25 | 53,369 | 189 | 100% | 0.0174 | 1.443 | 0.0091 | 50% / 48% | 2% / 2% |
| 50 | 59,775 | 162 | 100% | 0.0126 | 1.409 | 0.0092 | 52% / 65% | 2% / 2% |
| 75 | 57,391 | 156 | 98% | 0.0094 | 1.369 | 0.0090 | 73% / 75% | 5% / 0% |
| 100 | 56,422 | 139 | 100% | 0.0090 | 1.282 | 0.0097 | 73% / 75% | 3% / 2% |
| 125 | 45,739 | 135 | 100% | 0.0097 | 1.193 | 0.0095 | 82% / 75% | 2% / 0% |
| 150 | 44,758 | 130 | 100% | 0.0102 | 1.139 | 0.0090 | 85% / 77% | 2% / 2% |
| 175 | 50,557 | 130 | 100% | 0.0093 | 1.111 | 0.0093 | 83% / 85% | 0% / 5% |
| 200 | 50,971 | 142 | 100% | 0.0098 | 1.072 | 0.0106 | 83% / 92% | 3% / 2% |
| 225 | 45,060 | 138 | 100% | 0.0100 | 1.032 | 0.0103 | 85% / 92% | 10% / 5% |
| 250 | 47,922 | 128 | 100% | 0.0101 | 1.078 | 0.0102 | 90% / 90% | 3% / 3% |

Pipeline health:
- No NaNs or exploding values.
- Entropy declines smoothly (1.52 → 1.08), with no collapse to a single action.
- KL sits in the 0.005–0.011 band, so target-KL early-stopping rarely triggers.
- Value loss is flat at ~0.01: the value head can't explain much, which is consistent with the γ-limited
  return targets (F-15).
- Games get shorter (284 → 128 turns), so the policy learns to finish games.

Self-play learning is real but shallow. Win rate against random rises from ~50% to ~90%, while win rate
against the heuristic stays at 0–10% throughout. No "suspiciously fast" learning against a meaningful
opponent was observed. (The 80–90% vs random seen after 2 iterations in a separate GNN timing run is
F-19's point: that tier saturates almost immediately.)

**Checkpoint ladder.** `audit/tools/run_ladder.sh` on fresh seed bases 30.0M/30.1M/30.2M. Every board seed is
played 4× with the candidate rotated through all seats; Wilson 95% CIs.

| Checkpoint | vs 3 heuristics (1,000 games) | vs 3 random (400) | **iter_250** vs 3× this ckpt (400) | per-seat vs heuristic (s0..s3) |
|---:|---|---|---|---|
| 25 | 1.9% [1.2, 2.9] | 43.2% [38, 48] | **60.3%** [55, 65] | 3.2 / 0.8 / 2.0 / 1.6 |
| 50 | 2.3% [1.5, 3.4] | 54.0% [49, 59] | 36.2% [32, 41] | 3.6 / 2.8 / 0.8 / 2.0 |
| 75 | 2.7% [1.9, 3.9] | 66.5% [62, 71] | 44.8% [40, 50] | 3.6 / 2.8 / 1.6 / 2.8 |
| 100 | 3.7% [2.7, 5.1] | 81.8% [78, 85] | 32.0% [28, 37] | 4.8 / 2.4 / 5.2 / 2.4 |
| 125 | 4.4% [3.3, 5.9] | 80.2% [76, 84] | 33.3% [29, 38] | 6.8 / 3.2 / 3.6 / 4.0 |
| 150 | 2.6% [1.8, 3.8] | 88.7% [85, 91] | 36.2% [32, 41] | 2.4 / 2.4 / 3.2 / 2.4 |
| 175 | 3.5% [2.5, 4.8] | 84.3% [80, 87] | 28.5% [24, 33] | 5.2 / 3.2 / 2.8 / 2.8 |
| 200 | 2.6% [1.8, 3.8] | 87.7% [84, 91] | 31.5% [27, 36] | 2.0 / 2.8 / 2.8 / 2.8 |
| 225 | 3.4% [2.4, 4.7] | 89.7% [86, 92] | 23.3% [19, 28] | 5.2 / 3.2 / 2.8 / 2.4 |
| 250 | 3.6% [2.6, 4.9] | 85.5% [82, 89] | 25.0% [21, 29] *(self = parity check ✓)* | 3.2 / 3.6 / 3.6 / 4.0 |

Reading:
1. **Later checkpoints do beat earlier ones**, and the latest model's edge over older selves shrinks
   monotonically to parity. Self-play is producing real improvement against its own lineage, with no sign of
   cycling inside 250 iterations.
2. That improvement **barely transfers**. Against the fixed heuristic, win rate moves from 1.9% to ~3.5%:
   statistically detectable (CIs at iterations 25 and 125 don't overlap), but ~20pp below parity after 13M
   transitions.
3. **The in-loop metric is not trustworthy for selection.** Its best vs-heuristic reading was iter 225
   (10%, seat 0, n=60). The seat-balanced n=1,000 value is 3.4%, indistinguishable from neighbouring
   checkpoints. Selecting on the in-loop maximum would have picked noise (F-14, winner's curse).
4. **Self-play "training reward" can't be correlated with held-out win rate**, because the self-play return
   is constant by construction (see Reward Audit). Training-side proxies that did move (shorter games, falling
   entropy, rising vs-random rate) track the ladder's random column, not the heuristic column.
5. **No seat dependence** is visible at these low rates. This flat model sees absolute seat ids (F-26), so
   seat effects may appear at higher strength. Re-check them with the GNN encoder.

---

## Evaluation Audit

Good practice already in the user's protocol (memory): fresh never-reused seed bases, n≥600, paired seeds,
and recognition of winner's-curse selection. Problems:

1. In-repo in-loop evaluation is seat 0 only (F-14).
2. No CIs are printed anywhere in-repo. A 120- or 240-game eval has a 95% CI of about ±6–9pp at 25–60%.
3. The opponent tier is a single heuristic. The random tier is uninformative (F-19).
4. `best_eval.pt` = the maximum of ~40 noisy reads, which is selection bias.
5. Eval policies run deterministically (argmax) while training samples. This is fine, but should be stated.
6. Training and evaluation seeds are disjoint in all drivers (train_hier: training 0…N, evals 800k / 2M + it·k;
   rl_finetune: training ≤ ~1M, evals 4.7M/4.8M). There's no gradient path from eval games, and models have no
   dropout or batch-norm, so `eval()` mode is moot.

**Seat effects (heuristic×4, 4000 games):** 28.3 / 24.5 / 22.9 / 24.3%. Seat 0 is about 3pp above parity.

---

## Performance Results

Measured with `audit/tools/profile_perf.py`: single process, 1 torch thread, after training had stopped
(`audit/results/perf.json`). Training throughput comes from the audit mini-run log (12 fork workers + GPU
update, partly under contention from concurrent audit jobs).

| Metric | Value |
|---|---:|
| Engine, random agents | **38,400 steps/s**, 26 games/s per core |
| Engine, heuristic agents | 34,400 steps/s, 31 games/s per core |
| AEC env step incl. `last()` observation | 19,500 steps/s per core |
| `legal_actions` | 13 µs / call |
| `build_observation` (flat dict) / + `flatten_observation` | 41 µs / 62 µs |
| `build_graph_observation` | 57 µs |
| Longest Road, all 4 players | 64 µs |
| Policy `act()` hier h=256 (CPU, batch 1) | **347 µs** |
| Policy `act()` GNN h=128 L=3 / h=256 L=4 (CPU, batch 1) | **2.5 ms / 7.4 ms** |
| Self-play rollout, hier, 1 process | 417 transitions/s |
| Self-play rollout, GNN 256×4, 1 process | 133 transitions/s |
| `train_hier` hier h=256, 12 workers + GPU PPO (mini-run, 250 iters) | **~4,500 transitions/s** (13.2M in 2,906 s) |
| `train_hier` GNN h=128 L=3, 12 workers (timing run) | ~290 transitions/s (≈80k per 270 s iteration) |

**Bottlenecks** (cProfile of a hier rollout, `perf.json`):
1. **Batch-1 policy inference** dominates. It costs 6× the env work for the flat model and 50–130× for the GNN.
2. Enum hashing (`Resource`/`ActionType` dict keys): ~15% of rollout time.
3. `legal_actions` and observation building, ~25% combined.

Longest Road, rendering, logging, and deep copies are **not** bottlenecks. The engine itself is not the
problem. The fix is architectural: step N environments in lockstep per worker and run one batched forward
pass (on GPU for the GNN), which typically gains 10–50× at batch 64–256. Only after that are integer-indexed
arrays instead of Enum-keyed dicts worth doing (~1.3×).

**Scalability** (measured throughput; "transitions" = per-agent decisions, ≈ env steps):

| Env steps | hier h=256 (4.5k/s) | GNN 256×4 rollout only (12 × 133 ≈ 1.6k/s) | GNN 256×4 after batched inference (est. 20k/s) |
|---|---:|---:|---:|
| 1M | 4 min | 10 min | <1 min |
| 10M | 37 min | 1.7 h | 8 min |
| 100M | 6.2 h | 17 h (+ GPU update time) | 1.4 h |
| 1B | 2.6 days | ~7 days (+ updates) | ~14 h |

Storage per checkpoint: hier ~3 MB (248k params, with Adam state); GNN 256×4 ~47 MB with Adam (3.9M
params × 4 B × 3). One checkpoint every 25 iterations of a 2,000-iteration run is ~4 GB for the GNN.
In-memory rollout buffer: GNN observations are 1,420 floats (5.7 KB) per transition, so a 100k-transition
iteration is ~570 MB of host RAM before tensorisation, which is the limiting factor on the 6 GB GPU.
Evaluation: a 1,000-game seat-balanced eval of the hier model vs heuristics takes 20 s on 14 cores.
The GNN is roughly 5–20× slower, inference-bound.

---

## Testing Results

Existing suite: **121 passed** in 155 s (`audit/results/baseline_pytest.txt`). It is mostly behavioural, not
just smoke tests.

| Subsystem | Existing tests | Quality | Missing cases (now covered by audit tests unless noted) |
|---|---:|---|---|
| Board | 11 | good | red-number adjacency |
| Engine rules | 21 | good | conservation, maritime bank stock, off-turn win, RB lock, pre-roll dev, ACCEPT w/o cards, shortage exception (existing test encodes incomplete rule) |
| Longest Road | 2 | weak (min length, blocking) | independent oracle, loops, ties, transfers |
| PettingZoo | 10 | good (api_test) | seed_test, obs-in-space under public features, dead-step |
| Public hands / leakage | 9 | good | counterfactual observation invariance (all encoders), counter-offer visibility |
| Models / PPO | 15+8+2 | good | rollout↔recompute log-prob equality, discount horizon |
| League / BC / DAgger | 9+6+5+4+7 | exercise code paths | promotion statistics vs parity |
| Evaluation | 0 | — | seat rotation (not added; recommend) |

Audit suite (`.venv/bin/python -m pytest audit/tests`): **264 passed, 12 xfailed (strict)** in 54 s. Every
xfail was re-run with `--runxfail` to confirm that it fails on its intended assertion, not on an incidental
error.

| Audit test file | Covers |
|---|---|
| `test_rules_audit.py` | board (200 seeds), setup, turn machine, production, building, robber/7, dev cards, maritime/domestic trade, Longest Road oracle + transfers, Largest Army, off-turn win, engine validation |
| `test_obs_leakage.py` | counterfactual hidden-info invariance (flat + graph + legal list, both feature flags), steal/discard estimate independence, counter-offer visibility |
| `test_env_api_audit.py` | terminal reward delivery (20 games), obs ⊂ declared space, determinism, truncation reward, shaping, dead-step |
| `test_policy_pipeline_audit.py` | rollout↔`evaluate_actions` log-prob/value equality (hier + GNN), 12k masked samples, YoP encoding, discount horizon |
| `test_exploits_audit.py` | stall bound, trade-path conservation (54 protocol paths), ACCEPT griefing, dev replay, robber self-steal |

Randomised invariant fuzzing (`audit/tools/fuzz_invariants.py`, invariants checked after **every** step):

| Run | Games | Steps | Violations (games) | LR oracle mismatches | Off-turn wins |
|---|---:|---:|---|---:|---:|
| random, absolute conservation | 100 | 132,651 | conservation 89–100% (F-01) | 0 | 0 |
| random, relative conservation | 1,000 | 1.42M | bank<0: 5 (F-02) | 0 | 3 |
| random, relative conservation | 10,000 | 14.3M | bank<0: 65 (F-02); 1 timeout at 6000 steps | 0 | 15 |
| heuristic, relative conservation | 2,000 | 2.23M | bank<0: 1 (F-02) | 0 | 4 |

No other invariant ever fired: piece limits, occupancy, distance rule, road connectivity, robber, dev-card
and knight conservation, cache sync, award thresholds, and non-empty legal actions.

---

## Reproducibility Results

`audit/tools/repro_check.sh` (hier h=64, 3 iterations × 4 episodes):

| Check | Result |
|---|---|
| Same seed, sequential rollouts, CPU update — metrics | **identical** |
| Same seed, sequential — final weights | **bit-identical** |
| Different seed — metrics | differ (randomness live) |
| Same seed, `--num-workers 4` — metrics / weights | **differ** (F-22) |
| Engine: same seed + same actions | identical trajectory (`test_determinism_same_seed_same_actions`) |
| PettingZoo `seed_test` | pass |

The guarantee: engine games and sequential training are reproducible, but every real (parallel) training run
is not. GPU updates were not tested for determinism. `torch.use_deterministic_algorithms` is never set.

---

## Exploitability / Reward-Hacking Risks

| Exploit attempt | Result |
|---|---|
| Illegal action through policy/env | closed (mask-by-construction; bad indices raise) |
| Illegal action through `engine.step` | **open** (F-05) — not policy-reachable |
| Build without paying / duplicate resources | closed, except **maritime trade with empty bank mints a card** (F-02) and setup grants (F-01) |
| Replay dev cards / play VP card / same-turn play | closed |
| Longest Road / Largest Army farming | closed (shaping telescopes) |
| Infinite in-turn loop / stall | closed (max 19 steps/turn under stalling policy) |
| Stall-to-truncation while leading | **open** under default caps (F-06) |
| Griefing: ACCEPT without cards to burn proposals | **open** (F-13) |
| Hidden info via robber/trade/malformed actions | closed (masks independent of hidden info) |
| Off-turn win | **open**, rare (F-03) |

---

## Known Simplifications From Standard Catan

1. Domestic trades are 1-for-1 only, always broadcast to all opponents, capped at 2 per turn; counters are 1:1
   and invisible to the proposer (F-04, F-20).
2. No development cards before the roll (F-07).
3. Road Building locks the rest of the turn (F-08).
4. Bank supply is inflated by setup grants (F-01); maritime trades ignore supply (F-02); no single-player
   shortage exception (F-11).
5. Win checked for every player at any time (F-03).
6. Board: unconstrained 6/8 placement (F-10); evenly spaced harbors (F-28).
7. Discard options truncated at 100 combinations (F-12).
8. 4 players only; no 5–6 player or expansion rules (appropriate scope).

These should be stated as "Catan variant V1" in the README and in any reported result.

---

## Recommended Fixes

In priority order:
- F-14: seat rotation.
- F-06: truncation reward and defaults.
- F-15: γ/λ.
- F-04: counter visibility.
- F-03: own-turn win.
- F-01/F-02: conservation.
- F-07: pre-roll dev cards.
- F-08: Road Building lock.
- F-13: ACCEPT gating.
- F-05: validation flag.
- F-22: worker seeding.
- F-10: token placement.
- F-17: promotion p0.
- F-24: defaults.

The rule fixes change the game, so all existing checkpoints and their numbers become stale once they land.
That is acceptable here, since the champion is already lost (F-18).

## Recommended New Tests

All of these exist in `audit/tests/`. Promote them into `tests/` when the fixes land:
- conservation after setup and after every step;
- `test_maritime_trade_requires_bank_stock`;
- `test_cannot_win_on_another_players_turn`;
- `test_counter_offer_terms_visible_to_proposer`;
- `test_knight_playable_before_roll`;
- `test_road_building_does_not_lock_turn`;
- the counterfactual observation-invariance test;
- the Longest Road oracle test;
- the rollout/recompute log-prob test;
- a 1,000-game invariant fuzz as a slow-marked test.

Still to add:
- seat-rotation assertion for `evaluate_policy`;
- truncation-reward test;
- GPU determinism test.

## Recommended Training Improvements

1. **Objective:** win/loss terminal reward (+1, −1/3 each loser) for strength runs; drop the rank reward, or
   anneal it.
2. **Horizon:** γ=0.999–1.0, λ=0.98–1.0, skip forced transitions, consider per-turn discounting.
3. **Truncation:** bootstrap V(s_T) with 0 reward, or pay 0 and cap at ≥4000 steps.
4. **Opponents:** a fixed mixed pool for both training and evaluation (heuristic, search, honest-heuristic,
   ≥3 historical checkpoints), so the target is no longer one exploitable bot. Fix promotion p0 (F-17).
5. **Search at inference:** memory shows the determinised-rollout wrapper adds ~+2pp at deploy time. Combining
   it with a properly long-horizon value function (point 2) is the most promising path to "stronger than the
   teacher", rather than a larger network.

## Recommended Evaluation Protocol

For each candidate:
- **Pool:** fixed opponent pool {heuristic, honest-heuristic, search (depth 2), previous champion, two older
  checkpoints}.
- **Seeds:** a held-out seed list (e.g. 50,000,000+, never used by any trainer). Each seed is played **four
  times, once per seat**, using `audit/tools/tournament.py`.
- **Sample size:** ≥1,000 seeds (4,000 games) per opponent for claims, or 250 seeds for screening.
- **Primary metric:** win rate with a Wilson 95% CI. Comparisons between two candidates use paired seeds and
  a paired test (McNemar, or bootstrap over seeds).
- **Secondary metrics:** per-seat win rate, average VP, game length, timeouts, build mix, trade rate.
- **Selection:** candidates are chosen on a *separate* selection seed set and confirmed on the held-out set.
  Never select on the confirmation set.

## Roadmap

### Immediate — Before Any More Serious Training
| Item | Impact | Difficulty |
|---|:-:|:-:|
| Seat-rotate `evaluate_policy` / `fixed_seed_eval`; print Wilson CIs; stop selecting `best_eval` on the in-loop max (F-14) | HIGH | LOW |
| Truncation: no win reward on truncation (0 + bootstrap), defaults ≥4000 steps, `--randomize-board` on by default (F-06, F-24) | HIGH | LOW |
| Raise γ/λ (≥0.999 / ≥0.98) and drop forced single-action transitions (F-15) | HIGH | LOW |
| Add counter-offer terms to both observation encoders (F-04) | HIGH | LOW |
| Durable checkpoint storage + per-run manifest (git SHA, args, seeds, eval JSON) (F-18) | HIGH | LOW |
| Deterministic worker seeding (base seed + iteration + worker index) (F-22) | MEDIUM | LOW |

### Phase 1 — Environment Correctness
| Item | Impact | Difficulty |
|---|:-:|:-:|
| Own-turn-only win, claimed at start of turn (F-03) | HIGH | LOW |
| Debit bank on setup grant; bank-stock check on maritime trade; single-player shortage exception (F-01, F-02, F-11) | MEDIUM | LOW |
| Pre-roll development cards (F-07) | HIGH | MEDIUM |
| Road Building: allow continuing the turn / skipping (F-08) | MEDIUM | LOW |
| ACCEPT only if responder can pay (F-13) | MEDIUM | LOW |
| `step(validate=True)` guard (F-05) | MEDIUM | LOW |
| No adjacent 6/8 in board generation (F-10) | MEDIUM | LOW |
| Promote `audit/tests` into `tests/`; add slow-marked 1,000-game invariant fuzz to CI | MEDIUM | LOW |
| Richer domestic trade (multi-card, targeted) with an explicit negotiation budget (F-20) | HIGH | HIGH |

### Phase 2 — Training Correctness
| Item | Impact | Difficulty |
|---|:-:|:-:|
| Win/loss terminal objective for strength runs; anneal VP shaping (F-16) | HIGH | LOW |
| Fixed mixed opponent pool for RL instead of 3× one heuristic | HIGH | MEDIUM |
| Fix league promotion test (p0=0.25 or 2-seat format) so the league actually turns over (F-17) | MEDIUM | LOW |
| Log gradient norms, explained variance, per-head entropy | LOW | LOW |
| Re-derive the BC → DAgger → RL lineage on the corrected engine (artifacts lost anyway) | HIGH | MEDIUM |

### Phase 3 — Evaluation
| Item | Impact | Difficulty |
|---|:-:|:-:|
| Adopt `audit/tools/tournament.py` protocol (all-seats per seed, held-out seeds, CIs, paired comparisons) | HIGH | LOW |
| Opponent pool: heuristic, honest-heuristic, search, ≥3 historical checkpoints; drop random as a strength tier (F-19) | HIGH | LOW |
| Separate selection vs confirmation seed sets; a seed registry file in the repo | MEDIUM | LOW |
| Group BC validation split by game (F-21) | LOW | LOW |

### Phase 4 — Performance
| Item | Impact | Difficulty |
|---|:-:|:-:|
| Vectorised rollouts: N envs per worker + batched (GPU) inference — the dominant cost (F-25) | HIGH | MEDIUM |
| Integer-indexed arrays instead of Enum-keyed dicts in engine/obs | MEDIUM | MEDIUM |
| Incremental legal-action / observation caches | LOW | MEDIUM |

### Phase 5 — Stronger Catan AI
| Item | Impact | Difficulty |
|---|:-:|:-:|
| Long-horizon value function + determinised search at inference (memory: rollout wrapper already +~2pp at deploy) | HIGH | MEDIUM |
| Public-event history / belief features (dev-card purchase timing, robber/trade history) before any recurrent model | MEDIUM | MEDIUM |
| Population-based / league self-play with an exploitability probe (train a fresh exploiter vs the champion) | HIGH | HIGH |
| Learned negotiation once trading is un-simplified | HIGH | HIGH |

---

## Answers to the 25 Questions

1. **Is the simulated game legal standard Catan?** No. It is a close variant with 1:1-only broadcast
   trading capped at 2 proposals per turn, no pre-roll dev cards, a Road Building turn lock, off-turn wins,
   setup-grant and maritime conservation bugs, and unconstrained 6/8 placement. The core mechanics are
   correct, including production, robber, discard, dev cards, Longest Road, and Largest Army.
2. **Can any player observe information a real player could not know?** No. Counterfactual tests over every
   encoder, both feature flags, and the legal-action list found no leakage. (The *heuristic opponents* do read
   hidden state, with no measurable benefit: F-09.)
3. **Can the policy execute an illegal action?** No. Actions are chosen from the legal list (0 illegal in
   12,000 samples) and bad env indices raise. `engine.step` itself would accept illegal actions (F-05), but the
   policy can't reach that path.
4. **Can any legal Catan action not be represented?** Yes: multi-card or targeted domestic trades, pre-roll
   dev cards, discards beyond the 100-combination cap, and continuing a turn after Road Building.
5. **Is Longest Road unquestionably correct?** Yes, within what can be tested. Zero mismatches against an
   independent subset-enumeration oracle on ~52,000 networks plus 300 hand-built graphs; tie, transfer, and
   set-aside cases match the official almanac.
6. **Is the robber implemented correctly?** Yes: discard threshold and amount, ordering, mandatory move,
   victim eligibility, and uniform random steal (4,000-trial test). The steal identity is hidden from
   third parties.
7. **Are development cards correctly hidden and timed?** Hidden: yes. Timing: mostly. Not playable on the
   turn bought, one per turn, VP cards count immediately. Pre-roll play is missing (F-07).
8. **Is trading faithfully represented?** No (F-20, F-04, F-13). Maritime trading is right except for the
   bank-stock check (F-02).
9. **Are victory points and termination correct?** VP accounting yes. Termination no: off-turn wins occur in
   about 0.2% of games (F-03). Every agent is terminated correctly and nothing mutates after GAME_OVER.
10. **Is the PettingZoo/MARL interface correct?** Yes. Official `api_test` and `seed_test` pass, reward
    bookkeeping is verified, and observations stay within the declared spaces. Non-standard but documented:
    the `clear_reward` helper and the `Discrete(400)` index-into-legal-list encoding.
11. **Is deterministic evaluation possible?** Yes for engine games and single-process evaluation. No for
    parallel training rollouts (F-22).
12. **Does the reward optimise winning?** No. It optimises rank and VP, and pays truncation like a win
    (F-06, F-16).
13. **Can the agent exploit reward shaping?** Only mildly. VP shaping telescopes, so it can't be farmed. The
    real exploit is stalling into truncation while leading under the default step caps.
14. **Can the agent exploit simulator bugs?** Marginally: minting a card through maritime trade with an empty
    bank (F-02), griefing trade proposals (F-13), and rare off-turn wins (F-03). No free building, no dev-card
    replay, and no infinite loops.
15. **Does higher training reward correlate with held-out win rate?** It can't be measured. Self-play return
    is constant by construction and isn't logged. Training proxies tracked the vs-random tier, not the
    vs-heuristic tier.
16. **Does a later checkpoint reliably beat earlier ones?** Yes, against its own lineage: 60% → 25% (parity)
    across iterations 25 → 250. But the gain against a fixed heuristic is tiny (1.9% → 3.6%).
17. **Are results robust across seats?** Seat 0 is the best seat (+3.3pp for heuristic×4; +7pp for search vs
    heuristic). In-loop evaluations use seat 0 only, so they are biased (F-14).
18. **Are results robust across board seeds?** Evaluations use fresh random boards. The training distribution
    differs from official boards (F-10). Per-board variance is averaged out by the all-seats-per-seed protocol.
19. **Is self-play producing genuine improvement rather than cycling?** In the mini-run it improves
    monotonically against its own history (no cycling in 250 iterations), but it transfers poorly to a
    differently-styled opponent. The production line isn't self-play at all.
20. **Is the algorithm appropriate for partial observability?** Adequate for now. A hand-built belief
    (card-count estimates) is used and there is no memory. The bigger limits are the missing counter-offer
    terms and missing public history, not the absence of an LSTM.
21. **Are actor and critic information boundaries correct?** Yes. They share inputs, and no privileged
    state exists anywhere.
22. **Are training and evaluation isolated?** Yes for seeds and gradients. Selection is not isolated:
    `best_eval` is chosen on the same in-loop seeds that it reports.
23. **Are reported win rates statistically meaningful?** In-repo, no: no CIs, 120–240 games, seat 0 only.
    The memory-recorded confirmatory protocol (fresh seeds, n≥600–3,000, paired) is sound, but those
    results can no longer be re-checked (F-18).
24. **What prevents a substantially stronger agent?** In order:
    - credit assignment (γ horizon, F-15);
    - training against a single fixed opponent with a rank/VP objective;
    - a simplified trading game with blind counters;
    - inference-bound throughput, which caps GNN RL at ~10⁶–10⁷ steps per day;
    - noisy, seat-biased selection.

    Network size is not the bottleneck.
25. **Five highest-value next steps:**
    1. Fix the evaluation protocol: all-seats-per-seed, CIs, no in-loop-max selection.
    2. Use γ≈0.999+ with a win/loss reward and bootstrapped truncation.
    3. Fix the rule and observation bugs (F-01/02/03/04/07/08/13) and rebuild the lineage on the corrected
       engine.
    4. Vectorise rollouts with batched GPU inference.
    5. Train and evaluate against a fixed, diverse opponent pool, then add determinised search at inference
       on top of the better value function.
