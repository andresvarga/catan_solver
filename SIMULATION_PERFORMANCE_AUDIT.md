# Simulation Performance Audit — catan-marl

Date: 2026-10-04 · Code: `46f5c47` (before) → `19eb9a4` + the O3 bit-exactness fix (after) ·
Tooling: `audit/perf/` · Raw results: `audit/results/perf/`

## 1. Executive Summary

The goal was to increase useful environment steps per second without changing game behavior or
training data. Every kept change reproduces the pre-optimization golden fingerprints. Engine
state, observations and actions match exactly. With the O3 fix, rollouts also match with zero
numeric tolerance: log-probs, values and advantages are bit-identical.

| Workload (Ryzen 7 5800H + RTX 3060 Laptop) | Before | After | Speed-up |
|---|---|---|---|
| Flat (hier 256×3) rollout, 1 process, 16 envs, env steps/s | 3,401 | 6,964 (O6)¹ | **2.05×** |
| Flat rollout, best parallel config (12 workers × 16 envs), env steps/s | 17,280 | 24,763 | **1.43×** |
| Flat rollout, best config, completed games/s | 10.9 | 15.6 | 1.43× |
| GNN 256×4 + phf, 4 GPU-inference workers × 24 envs, 192 episodes, transitions/s | 3,551 | 3,869 | 1.09× |
| GNN 256×4 + phf, 1 process, CUDA, 32 envs, env steps/s | 2,466 | 2,567–2,646 | 1.04–1.07× |

¹ Single process, measured on an idle machine. The final bit-exact O3′ changes this by +0.5%, which is noise (§14).

Why the gains are uneven:

- **The CPU flat path was dominated by Python observation encoding.** That cost is now mostly
  gone, and trade-bundle decoding and NumPy sampling overhead were cut as well.
- **Multi-worker scaling is limited by the hardware, not by IPC or startup.** Throughput levels
  off at about 12 processes on 8 physical cores, and the laptop's clock speed drops under full
  load. Per-process speed-ups therefore shrink from about 2× to about 1.4× at full machine load.
- **The GNN path is bound by the GPU forward pass and by GPU batch fill.** It gained little from
  the CPU work. Its main remaining lever is keeping inference batches full (§9, §16).

Negative or neutral results, all measured:

- threads instead of processes: slower;
- per-step logging: none exists;
- GC tuning: GC costs about 0% of run time;
- computing beliefs once in the GNN observation: neutral end-to-end;
- persistent GPU worker pool: about 6% per rollout call, smaller than expected.

## 2. Current Architecture (traced, not assumed)

```
train_hier / rl_finetune / league_train
 └ hier_ppo.collect_rollout_parallel(seeds…)
    ├ CPU inference: ProcessPoolExecutor(fork) created per call      (~0.1 s for 12 workers)
    ├ GPU inference: persistent spawned pool, weights shipped per call (was: new spawn per call, 2.5 s)
    └ _worker_collect → collect_episodes_batched(N envs in flight)
        └ EpisodeRunner(CatanAECEnv) — new env object per episode (~0.9 ms, 0.6 µs/step amortised)
           advance(): env.last → env.legal_actions → adapter.encode
               flat: encode_flat(GameState)           (was: build_observation dict + flatten_observation)
               gnn : build_graph_observation(GameState)
           model.act_batch: head_logits_batch (torch) → NumPy sample_decision → decode_trade
           apply(): env.step(idx) → engine.step(validate=False)   (was: validate=True re-enumerated legal actions)
        → transitions list[dict] → compute_gae → pickle → parent _assemble (seed order)
```

Rollouts are a pure function of (weights, seeds). Each episode has its own NumPy Generator and
its own opponent copies, and episodes are assembled in seed order. As a result, the worker count
and the envs-per-worker setting change throughput but never the data. The golden test checks
this property at configurations (1,1) and (2,3), and on GPU at (1,4) and (2,2).

## 3. Hardware Specification

| | |
|---|---|
| CPU | AMD Ryzen 7 5800H, 8 cores / 16 threads, 16 MB L3, laptop power envelope (clock drops under load) |
| RAM | 38 GB |
| GPU | NVIDIA RTX 3060 Laptop, 6 GB |
| Software | Python 3.14.4, torch 2.12.1+cu126, numpy 2.5.0; psutil 7.2.2 and py-spy 0.4.2 (dev-only, not in requirements.txt) |

## 4. Baseline Performance (code at `46f5c47`)

| Workload | Env steps/s | Games/s | Steps/game | Notes |
|---|---|---|---|---|
| W1 engine only, random agents, 1 process | 30,611 (sd 706) | 28.5 | 1,092 | simulator ceiling |
| W2 AEC env, random legal indices, 1 process | 11,391 (sd 58) | 7.4 | 1,558 | env wrapper + observation dict = 2.7× engine cost |
| W3 flat rollout, 1 worker, 1 / 4 / 16 envs | 2,119 / 2,703 / 3,042 | 1.3 / 1.7 / 1.9 | | in-process path does not pin torch threads (CPU ≈ 800%) |
| W3 flat rollout, 12 workers × 16 envs | 17,280 (sd 1,563) | 10.9 | | PSS 12.2 GB |
| W5 GNN 256×4 phf, 4 GPU workers × 24 envs, 192 episodes | 3,551 transitions/s | 3.0 | | |

Every run excludes warm-up. Each figure is the mean of 3–5 trials; std, min and max are in
the jsonl files.

## 5. Profiling Results

Exclusive-time component profile from `audit/perf/profile_components.py` (single process, one
torch thread).

**Flat model, 16 envs, baseline:**

| Component | % of wall | Per-call cost |
|---|---|---|
| flatten_observation | 18.5% | 102 µs |
| trade-bundle decode | 16.7% | 280 µs per trade decision; about ⅓ of decisions are trades |
| build_observation (dict) | 14.0% | 56 µs |
| forward pass | 9.5% | |
| legal_actions | 7.9% | 43% of the calls only re-validated an already-validated action |
| sample_decision (NumPy) | 7.3% | |
| public beliefs | 5.7% | |
| env bookkeeping | 3.4% | |
| engine.step (rules) | 2.5% | 10 µs |
| GC | ≈0% | |

**Flat model, 16 envs, after O1–O4** (`afterO4_components_hier_envs16.json`):

| Component | % of wall | Per-call cost |
|---|---|---|
| forward | 18.4% | 234 µs per batch |
| trade decode | 17.5% | 173 µs |
| encode_flat | 15.3% | 50 µs; was 158 µs as dict + flatten |
| sample_decision | 12.7% | |
| legal_actions | 8.6% | |
| act_batch glue | 4.8% | |
| env.step | 4.7% | |
| engine.step | 4.4% | |
| the rest | under 4% each | |

The profile is now flat. The biggest remaining items are the neural forward pass and the trade
decoder; O6 then trimmed the decoder further.

**GNN 256×4 phf, CUDA, 32 envs, baseline:**

- The forward pass is 42.5% of wall time, at 4.25 ms per batch.
- Trade decode is 15.5%.
- The average batch was only about 15.8 because of a drain at the end of each run (§9).
- The act_batch glue cost 333 µs per call, which included copying the trade-head weights from
  GPU to CPU on every call. O3 fixed that copy.

## 6. Concurrency Scaling (flat model, CPU inference, env steps/s)

| Workers × envs | Before e=1 | e=4 | e=16 | After e=1 | e=4 | e=16 | Speed-up (e=16) |
|---|---|---|---|---|---|---|---|
| 1 | 2,119 | 2,703 | 3,042 | 3,421 | 4,975 | 6,271 | 2.06× |
| 2 | 3,926 | 5,076 | 5,726 | 5,987 | 8,456 | 10,123 | 1.77× |
| 4 | 7,151 | 8,892 | 9,774 | 10,462 | 14,526 | 17,014 | 1.74× |
| 8 | 10,957 | 14,364 | 15,979 | 16,263 | 21,693 | 23,880 | 1.49× |
| 12 | 12,363 | 15,850 | 17,280 | 18,442 | 22,806 | **24,763** | 1.43× |
| 16 | 12,078 | 15,704 | 16,906 | 17,864 | 20,831 | 21,951 | 1.30× |

After the optimizations, scaling efficiency at 16 envs is:

| Workers | Efficiency |
|---|---|
| 2 | 81% |
| 4 | 68% |
| 8 | 48% |
| 12 | 33% |

The flattening has three hardware causes:

1. there are 8 physical cores, and SMT siblings add little to this integer- and Python-heavy load;
2. the laptop lowers its clock speed when all cores are busy;
3. the work per call is fixed, so the last games to finish leave workers idle. CPU use reaches
   only 600–1,000% of the 1,600% available.

Sixteen processes are slower than 12.

**Threads versus processes** (4 envs per thread): 1 thread 3,516 · 2 threads 3,261 · 4 threads
3,392 · 8 threads 2,819 env steps/s. The GIL serializes the work, so threads give no benefit and
add overhead. Processes are the right model.

## 7. Bottleneck Analysis — top hotspots, in order of measured share

1. **Observation encoding (flat): 32.5% before.** A dict was built and then flattened, with
   per-part `astype` and concatenate calls. Fixed by O2.
2. **Trade-bundle decode: 16–17%, on both model paths.** Each trade is decoded as 10
   sequential head steps, each step doing NumPy calls of 1–10 µs. Fixed by O3 and O6.
3. **The neural forward pass:** 9.5%, rising to 18% after the other fixes, on CPU; 42.5% on
   GPU. This is irreducible without changing the model, apart from batching (GPU).
4. **Small NumPy sampling (sample_decision): 7–13%.** About 10 µs of call overhead per head
   with fewer than 8 options. Fixed by O3.
5. **Redundant legality enumeration: about 3.4% of wall** (43% of `legal_actions` calls).
   Fixed by O1.

**Not bottlenecks:**

- engine rules: 2.5–4.4%;
- reset and env construction: under 0.5%;
- GC: 0%;
- IPC: below 1% of worker time (§10);
- logging: none (§11).

## 8. Memory Analysis

| Item | Size |
|---|---|
| Live environment (engine + state + caches) | ≈99.4 KiB |
| Transition in memory (flat) | 4,152 B: observation 2,052 B (float32), two padded float32 sub-masks 2×400 B, dict overhead |
| Pickled transition | flat 3,235 B · GNN 7,392 B |
| Process memory (PSS) | about 0.96 GB per worker process |
| Episode buffers | about 30 MB peak per episode collected per worker, so peak memory grows with episodes per call (12 workers × 16 envs, 384 episodes: 11.7 GB) |
| Idle GPU worker CUDA context | 134 MiB each, before the model |

Memory per collected episode, not per environment, is what limits scaling up the episodes per
call. Compact storage would cut it by about 2–4× (P1).

## 9. Policy Inference Analysis

- **CPU (flat):** inference is batched across the envs in flight. Going from 1 to 16 envs
  speeds a worker up 1.83× after the optimizations, and 1.44× before.
- **GPU (GNN):** throughput depends on batch fill. Game lengths vary a lot (for an untrained
  self-play policy: mean 1,663 steps, median 1,532, p90 2,566, p95 3,076, p99 3,456, max 4,000
  (the cap), coefficient of variation 0.41). When episodes per worker ≈ slots, the batch drains
  as games finish:

  | Episodes per call | 4 slots | 12 slots | 24 slots (transitions/s, 4 workers, after) |
  |---|---|---|---|
  | 48 | 1,447 | 2,420 | 2,407 |
  | 192 | 1,615 | 3,140 | 3,869 |

  A slot that waits for the slowest game is used about 54% of the time (mean/p95 length).
  `collect_episodes_batched` already refills a slot as soon as its game ends. What wastes
  throughput is the drain at the end of each call. The fix is more episodes per call relative to
  slots, or overlapping consecutive calls (P1).

- **Forward cost per batch** on GPU is 4.25 ms at a batch of about 16, which is mostly launch
  overhead. A larger batch is nearly free: 24 slots × 4 workers saturated better than 12.

## 10. Multiprocessing / IPC Analysis

- **Pool startup** with fork (CPU) is about 0.1 s for 12 workers, which is negligible. With
  spawn plus CUDA initialization it was 2.5 s per call (4 workers). O5 now pays that once.
- **Result IPC** costs 15.4 µs to pickle and 16.6 µs to unpickle per flat transition; for GNN,
  34 µs and 33 µs. Compared with about 150–300 µs of simulation per transition, that is under
  10% on the parent's critical path and about 1% per worker. Moving to columnar arrays (P1)
  would remove most of it.
- **Weights:** fork shares them copy-on-write. The persistent spawned pool sends a CPU
  state_dict per call: about 6 MB for GNN 256×4, under 0.1 s.

## 11. Logging Analysis

There is no per-step logging in the env, engine, policy or rollout loop. The only `print`
calls are in `render()` (not called during training) and the pool-retry message.
`GameState.turn_log` exists but is never appended to on the hot path. **Result: nothing to
remove (negative result).**

## 12. Allocation / Copying Analysis

**Before:** each flat observation took about 15 small array allocations and a concatenate, plus
a GPU→CPU copy of the trade head per act_batch call on GPU. Trade decode allocated mask and
prefix arrays per step.

**After:**

- **O2:** `encode_flat` writes into one float32 buffer and divides once, using a cached static
  board part per Board.
- **O3:** the trade weights are cached on the head, keyed by each parameter's version counter.
- **Transitions still hold float32 padded masks.** A uint8, unpadded form would be 3–4× smaller
  (P1, not done).

GC: 11 gen-1 collections and 0 gen-2 in 2.7 s, 0.000 s in total. `gc.freeze` or threshold
tuning has nothing to win.

## 13. Optimization Opportunities (ranked by measured share × feasibility)

| # | Opportunity | Evidence | Status |
|---|---|---|---|
| O2 | Fused flat encoder straight from GameState | 32.5% of wall | **done, +47.6%** |
| O3 | Pure-Python sampler for small heads + cached trade weights | 7.3% + about 5% of decode | **done, +26.8%** |
| O1 | Skip re-validation of already-validated actions | 3.4% | **done, +5.0%** |
| O6 | Skip MLP on forced trade steps (single legal option) | part of the 17.5% decode | **done, +4.2%** |
| O5 | Persistent spawned GPU pool | 2.5 s per call | **done, about 6% per call** |
| O4 | Compute beliefs once per observer in the graph observation | 11% of graph-obs time | done, neutral end-to-end |
| — | Compact columnar transitions (uint8 masks, arrays) | 4 KB per transition; about 30 MB per episode buffered | P1 |
| — | Overlap rollout calls / avoid the end-of-call drain | GPU fill 54% | P1 |
| — | Batched legal_actions / vector env in a compiled language | engine is 2.5–4.4% after; env wrapper 2.7× engine | P2/P3 (Rust experiment underway on `rust-simulation-engine`) |

## 14. Optimizations Tested (one at a time; before/after mandatory)

Single process, one torch thread, warm-up excluded, 5 trials. The benchmark is
`audit/perf/ab.py` and its log is `audit/results/perf/ab.jsonl`.

| Step | Change | Flat CPU 16 envs, env steps/s (sd) | Δ vs prev | GNN CUDA 32 envs, env steps/s (sd) | Δ vs prev | Golden |
|---|---|---|---|---|---|---|
| O0 | baseline | 3,401 (10) | — | 2,466 (16) | — | recorded |
| O1 | `engine.step(validate=False)` for env-validated actions | 3,571 (15) | +5.0% | 2,509 (90) | +1.7% | ✓ |
| O2 | `encode_flat` fused encoder (bit-exact) | 5,271 (44) | +47.6% | n/a (GNN path) | — | ✓ |
| O3 | small-head Python sampler + trade-MLP weight cache | 6,685 (57) | +26.8% | 2,646 (31) | +5.5% | ✓ |
| O4 | `expected_dev_cards_all` in graph features | — | — | 2,567 (27) | −3.0% (noise level; micro: 53→47 µs per obs, −11%) | ✓ |
| O6 | forced trade steps skip MLP (still consume the RNG draw) | 6,964 (30) | +4.2% | not rerun | — | ✓ |
| O3′ | O3 made bit-exact: normalizer summed by NumPy | 6,124 (53) vs 6,094 (161) for the sequential sum, paired and interleaved² | +0.5% (noise) | — | — | ✓ tol = 0 |

² Measured while another session's `check_rust_parity.py` held a core at 100% (load average about 5), so it was run as a paired, interleaved A/B in one process. Its absolute numbers are below the O6 row for that reason, not because of the change. An unpaired run under the same load gave 6,243.

O5 (persistent spawned pool, GPU only) was tested on 5 consecutive calls (48 GNN episodes,
4 workers × 12 envs). Per call it went from 28.59 s to 26.89 s, −6%. Correctness:
`test_gpu_rollouts_match_golden`, plus a weights-propagation check (perturbed weights through
the persistent pool = a fresh pool).

O6's first A/B (4,994, sd 518) ran while another benchmark was using the CPU. The clean rerun
is the number reported.

Rejected or negative:

- **Threads instead of processes:** −20% at 8 threads.
- **GC tuning:** GC time is 0%.
- **Logging removal:** there is no logging to remove.
- **First version of the O3 sampler:** it used a sequential Python `sum()` for the normalizer.
  In 9% of random cases the log-prob differed from NumPy's SIMD-ordered sum in the last bit;
  sampled actions were identical in all 200k cases. That was within the golden 1e-5 tolerance
  but not bit-exact, so the sum was moved back to `np.add.reduce`: 0/200k mismatches, 2.7 µs
  versus 9.6 µs for the original.

## 15. Correctness Regression

The golden suite (`audit/perf/golden.py`, `audit/tests/test_perf_golden.py`, 28 tests) was
recorded from the pre-optimization code:

- **16 engine games:** rolling SHA-1 of the full GameState plus the ordered legal-action list
  after every step, with 50-step checkpoints to locate any divergence. Exact.
- **4 AEC env episodes:** every observation (flat dict, flattened vector, graph features), with
  and without public-hand features, plus rewards and terminations. Exact.
- **3 rollouts:** flat self-play, flat versus a mixed opponent pool, GNN+phf. Each is checked
  at worker/env configurations (1,1) and (2,3), and on CUDA at (1,4) and (2,2). Every transition
  field is compared: discrete fields exactly, numeric fields within 1e-5 (1e-4 on GPU).
- **Extra check after O3′:** all 3 rollouts at (1,1) match the recording at **tolerance 0**.
- `tests/test_fast_encoder.py`: `encode_flat` equals `flatten_observation(build_observation())`
  bit-for-bit (uint32 view) on more than 1,000 states × observers, in both phf modes.

Full suites at `19eb9a4`, in a clean worktree: `pytest` gave **462 passed, 1 xfailed** and `pytest -m slow` gave **2 passed**. The O3′ edit was checked afterwards with golden and fast-encoder tests: **30 passed**, plus the zero-tolerance rollout check.

## 16. Recommended Architecture

- **Keep the current design:** process-parallel workers, each running N envs in flight with
  batched inference, seed-ordered assembly. Rollouts as a pure function of (weights, seeds)
  is the basis of the regression guarantee.
- **CPU (flat models):** fork per call is cheap enough at 0.1 s; no persistent pool is needed.
- **GPU (GNN models):** keep the persistent spawned pool (O5). Raise episodes per call to at
  least 8× the slot count, or overlap call k+1's start with call k's drain, to keep batches full.
- **Next steps, in order:**
  1. compact columnar transition storage;
  2. drain overlap;
  3. a compiled engine plus batched legal-action and encode functions, only if training moves
     to many more cores. Once the encoder is fixed, the Python env wrapper is the next fixed cost.

## 17. Optimal Parallelism (this machine)

| Path | Best config | Throughput |
|---|---|---|
| Flat, CPU inference | **12 workers × 16 envs** (8 × 16 is within 4% and uses 30% less RAM) | 24,763 env steps/s, 15.6 games/s |
| GNN 256×4, GPU inference | **4 workers × 24 envs, ≥192 episodes per call** | 3,869 transitions/s, 3.3 games/s, 13.8 GB PSS |

## 18. Projected Throughput

Based on measured games per second; the step cap is 4,000 and the bench workload is self-play
with an untrained policy.

| Games | Flat before (10.9/s) | Flat after (15.6/s) | GNN after (3.3/s) |
|---|---|---|---|
| 1 M | 25.5 h | 17.8 h | 3.5 d |
| 10 M | 10.6 d | 7.4 d | 35 d |
| 100 M | 106 d | 74 d | 351 d |

Caveat: trained policies play shorter games (fewer steps per game), so their games-per-second
figures will be higher.

## 19. Prioritized Roadmap

- **P0 (done):** O1–O6 and O3′. All are behavior-identical, with the golden suite as the
  regression gate.
- **P1:**
  - compact transitions and IPC (uint8 unpadded masks, columnar NumPy per worker): 3–4× less
    RAM per episode, more episodes per call;
  - overlap or pipeline rollout calls to remove the GPU end-of-call drain;
  - pin `torch.set_num_threads(1)` on the in-process rollout path.
- **P2:**
  - vectorized trade decode across a batch (all trade decisions in one matmul per step);
  - a cached `legal_actions` delta per step;
  - a cheaper env wrapper (`env.last` and reward-dict rebuilds).
- **P3:** a compiled engine (Rust, being explored on `rust-simulation-engine`) with batched
  legal-action and encode functions. Worth it only once Python glue dominates at the target core
  count; the golden suite is ready to gate it.
