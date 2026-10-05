#!/bin/bash
# AFTER CPU sweep (same grid as baseline), then BEFORE GPU configs on the baseline worktree.
cd /home/andres/claude/catan-marl
for w in 1 2 4 8 12 16; do for e in 1 4 16; do eps=$(( w*e*2 > 48 ? w*e*2 : 48 ));
  .venv/bin/python -m audit.perf.bench_sim rollout --model hier --hidden 256 --layers 3 --workers $w --envs $e --episodes $eps --trials 3 --tag after > /dev/null 2>&1; done; done
cd /tmp/claude-1000/-home-andres-claude/a35c09b6-85e2-4a1c-a6e0-c0d7291de9b4/scratchpad/baseline_wt
for eps in 48 192; do
  .venv/bin/python -m audit.perf.bench_sim rollout --model gnn --hidden 256 --layers 4 --phf --device cuda --workers 4 --envs 4,12,24 --episodes $eps --trials 3 --tag before_gpu_slots --out /home/andres/claude/catan-marl/audit/results/perf/before_gpu_slots_rollout.jsonl > /dev/null 2>&1; done
echo SWEEPS-DONE
