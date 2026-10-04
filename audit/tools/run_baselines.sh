#!/bin/bash
# Baseline tournaments (fresh seed bases >= 20M; each seed played in all 4 seats)
set -x
PY=.venv/bin/python
$PY -m audit.tools.tournament --candidate heuristic --opponent random    --seeds 250  --seed-base 20000000 --workers 4 --out audit/results/t_heuristic_vs_random.json
$PY -m audit.tools.tournament --candidate heuristic --opponent heuristic --seeds 1000 --seed-base 20100000 --workers 4 --out audit/results/t_heuristic_x4_seatbias.json
$PY -m audit.tools.tournament --candidate honest    --opponent heuristic --seeds 1000 --seed-base 20200000 --workers 4 --out audit/results/t_honest_vs_heuristic.json
$PY -m audit.tools.tournament --candidate search    --opponent heuristic --seeds 250  --seed-base 20300000 --workers 4 --out audit/results/t_search_vs_heuristic.json
