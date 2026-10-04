#!/bin/bash
# Phase 25/31: checkpoint ladder on fresh seeds (>= 30M), seat-balanced.
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
D=audit/results/ckpts/run_a
OUT=audit/results/ladder
mkdir -p $OUT
for it in 25 50 75 100 125 150 175 200 225 250; do
  [ -f $D/iter_$it.pt ] || continue
  $PY -m audit.tools.tournament --candidate ckpt:hier:$D/iter_$it.pt --opponent heuristic --seeds 250 --seed-base 30000000 --workers 14 --out $OUT/iter_${it}_vs_heuristic.json > /dev/null
  $PY -m audit.tools.tournament --candidate ckpt:hier:$D/iter_$it.pt --opponent random --seeds 100 --seed-base 30100000 --workers 14 --out $OUT/iter_${it}_vs_random.json > /dev/null
  # later-vs-earlier: candidate = latest, opponents = 3 copies of this checkpoint
  $PY -m audit.tools.tournament --candidate ckpt:hier:$D/iter_250.pt --opponent ckpt:hier:$D/iter_$it.pt --seeds 100 --seed-base 30200000 --workers 14 --out $OUT/latest_vs_iter_${it}.json > /dev/null
  echo "done $it"
done
$PY - <<'EOF'
import json, os
print(f"{'iter':>5} | {'vs heur (95% CI)':>22} | {'vs random':>18} | {'latest vs iter_k x3':>20} | seat0..3 vs heur")
for it in range(25, 251, 25):
    f = f"audit/results/ladder/iter_{it}_vs_heuristic.json"
    if not os.path.exists(f): continue
    h = json.load(open(f)); r = json.load(open(f"audit/results/ladder/iter_{it}_vs_random.json"))
    l = json.load(open(f"audit/results/ladder/latest_vs_iter_{it}.json"))
    ps = " ".join(f"{h['per_seat'][str(s)]['rate']:.3f}" for s in range(4))
    print(f"{it:5d} | {h['win_rate']:.3f} [{h['ci95'][0]:.3f},{h['ci95'][1]:.3f}] | {r['win_rate']:.3f} [{r['ci95'][0]:.2f},{r['ci95'][1]:.2f}] | {l['win_rate']:.3f} [{l['ci95'][0]:.2f},{l['ci95'][1]:.2f}] | {ps}")
EOF
