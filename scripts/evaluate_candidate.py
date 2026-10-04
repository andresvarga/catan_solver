"""Standard evaluation protocol (roadmap Phase 3; CATAN_MARL_AUDIT.md
"Recommended Evaluation Protocol").

Evaluates a candidate against a fixed opponent pool on a named, registered
seed set (evaluation/seed_registry.json), every seed from all four seats, and
optionally compares it *paired* against a baseline on the same games.

    # screen a checkpoint on the selection set (250 board seeds = 1,000 games/opponent)
    python -m scripts.evaluate_candidate --candidate ckpt:gnn:runs/x/latest.pt:256:4:phf --seeds 250

    # confirm a claim against the current champion on a fresh confirmation set
    python -m scripts.evaluate_candidate --candidate ckpt:gnn:new.pt:256:4:phf \\
        --baseline ckpt:gnn:champ.pt:256:4:phf --seed-set confirm_1 --out confirm.json

Random agents are deliberately not a default opponent: any trained policy
beats them almost immediately (audit F-19), so they can't rank policies.
"""
from __future__ import annotations

import argparse
import json
import sys

from evaluation.seeds import log_usage, seed_set
from evaluation.tournament import paired_compare, run_matchup, summarize, timed

DEFAULT_OPPONENTS = "heuristic,honest,search"


def _fmt_ci(ci) -> str:
    return f"[{ci[0]:.1%}, {ci[1]:.1%}]"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--candidate", required=True, help="agent spec (see evaluation/tournament.py)")
    ap.add_argument("--opponents", default=DEFAULT_OPPONENTS,
                    help=f"comma-separated opponent specs (default {DEFAULT_OPPONENTS}); "
                         "past checkpoints via ckpt:...")
    ap.add_argument("--seed-set", default="selection",
                    help="registered seed set: inloop | selection | confirm_1..5")
    ap.add_argument("--seeds", type=int, default=None, help="use only the first N seeds per base")
    ap.add_argument("--baseline", default=None, help="agent spec to compare against, paired on the same games")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--note", default="", help="free text recorded in the seed-usage log")
    ap.add_argument("--out", default=None, help="write full results JSON here")
    a = ap.parse_args(argv)

    seeds = seed_set(a.seed_set, a.seeds)
    if a.seed_set.startswith("confirm"):
        prior = log_usage(a.seed_set, {"candidate": a.candidate, "baseline": a.baseline,
                                       "opponents": a.opponents, "seeds": len(seeds), "note": a.note})
        if prior:
            print(f"WARNING: confirmation set {a.seed_set!r} has been used {prior} time(s) before; "
                  "results on a reused confirmation set are selection-biased -- prefer a fresh one.",
                  file=sys.stderr)

    opponents = [o for o in a.opponents.split(",") if o]
    out = {"candidate": a.candidate, "baseline": a.baseline, "seed_set": a.seed_set,
           "board_seeds": len(seeds), "results": {}}
    print(f"seed set {a.seed_set}: {len(seeds)} board seeds x 4 seats = {4 * len(seeds)} games per opponent\n")
    print("| opponent | candidate win rate (95% CI) | per seat s0/s1/s2/s3 | avg VP | turns |"
          + (" baseline | diff (95% CI) | McNemar p |" if a.baseline else ""))
    print("|---|---|---|---:|---:|" + ("---|---|---:|" if a.baseline else ""))
    for opp in opponents:
        recs, secs = timed(run_matchup, a.candidate, opp, seeds, a.workers)
        summ = summarize(recs)
        entry = {"candidate": summ, "seconds": round(secs, 1)}
        row = (f"| {opp} | {summ['win_rate']:.1%} {_fmt_ci(summ['ci95'])} | "
               + "/".join(f"{summ['per_seat'][s]['rate']:.0%}" for s in range(4))
               + f" | {summ['avg_vp']:.2f} | {summ['avg_turns']:.0f} |")
        if a.baseline:
            base = run_matchup(a.baseline, opp, seeds, a.workers)
            cmp = paired_compare(recs, base)
            entry["baseline"] = summarize(base)
            entry["paired"] = cmp
            row += (f" {cmp['win_rate_b']:.1%} | {cmp['diff']:+.1%} {_fmt_ci(cmp['diff_ci95'])} | "
                    f"{cmp['mcnemar_p']:.3g} |")
        out["results"][opp] = entry
        print(row, flush=True)
    if a.out:
        with open(a.out, "w") as f:
            json.dump(out, f, indent=1, default=str)
    return out


if __name__ == "__main__":
    main()
