#!/bin/bash
# Phase 39: run a tiny train_hier config twice with the same seed (sequential
# rollouts, CPU update), once with a different seed, and twice with parallel
# workers; compare the printed per-iteration metrics.
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
OUT=audit/results/repro
mkdir -p $OUT
common="--model-type hier --hidden 64 --randomize-board --iterations 3 --episodes-per-iter 4 --eval-every 1000 --max-episode-steps 2500 --device cpu"
for tag in seq_a seq_b; do
  $PY -m training.train_hier $common --num-workers 1 --seed 7 --checkpoint-dir $OUT/$tag 2>&1 | grep '^iter' > $OUT/$tag.txt
done
$PY -m training.train_hier $common --num-workers 1 --seed 8 --checkpoint-dir $OUT/seq_seed8 2>&1 | grep '^iter' > $OUT/seq_seed8.txt
for tag in par_a par_b; do
  $PY -m training.train_hier $common --num-workers 4 --seed 7 --checkpoint-dir $OUT/$tag 2>&1 | grep '^iter' > $OUT/$tag.txt
done
strip() { sed -E 's/\|\s+[0-9.]+s \|/| T |/' "$1"; }
echo "seq same-seed identical:   $(cmp -s <(strip $OUT/seq_a.txt) <(strip $OUT/seq_b.txt) && echo YES || echo NO)"
echo "seq different seed differs: $(cmp -s <(strip $OUT/seq_a.txt) <(strip $OUT/seq_seed8.txt) && echo NO || echo YES)"
echo "parallel same-seed identical: $(cmp -s <(strip $OUT/par_a.txt) <(strip $OUT/par_b.txt) && echo YES || echo NO)"
$PY - <<'EOF'
import torch
a = torch.load("audit/results/repro/seq_a/latest.pt", weights_only=False)["model"]
b = torch.load("audit/results/repro/seq_b/latest.pt", weights_only=False)["model"]
print("seq weights bit-identical:", all(torch.equal(a[k], b[k]) for k in a))
a = torch.load("audit/results/repro/par_a/latest.pt", weights_only=False)["model"]
b = torch.load("audit/results/repro/par_b/latest.pt", weights_only=False)["model"]
print("parallel weights bit-identical:", all(torch.equal(a[k], b[k]) for k in a))
EOF
for f in $OUT/*.txt; do echo "== $f"; cat $f; done
