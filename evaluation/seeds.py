"""Evaluation seed registry (roadmap Phase 3).

`seed_registry.json` names the board-seed sets used for evaluation so that
(a) training never trains on evaluation boards, (b) candidate *selection* and
*confirmation* use different seeds, and (c) reuse of confirmation sets is
visible: every use is appended to `seed_usage.jsonl`.
"""
from __future__ import annotations

import json
import os
import subprocess
import time

_DIR = os.path.dirname(os.path.abspath(__file__))
REGISTRY_PATH = os.path.join(_DIR, "seed_registry.json")
USAGE_PATH = os.path.join(_DIR, "seed_usage.jsonl")


def load_registry(path: str = REGISTRY_PATH) -> dict:
    with open(path) as f:
        return json.load(f)


def seed_set(name: str, count: int | None = None, registry: dict | None = None) -> list[int]:
    """The board seeds of a named set (optionally only the first `count` per
    base). Seeds are interleaved across bases so a prefix stays balanced."""
    reg = registry or load_registry()
    if name not in reg["sets"]:
        raise KeyError(f"unknown seed set {name!r}; known: {sorted(reg['sets'])}")
    spec = reg["sets"][name]
    n = spec["count"] if count is None else min(count, spec["count"])
    return [base + i for i in range(n) for base in spec["bases"]]


def eval_ranges(registry: dict | None = None) -> list[tuple[str, int, int]]:
    reg = registry or load_registry()
    return [(name, b, b + spec["count"] - 1) for name, spec in reg["sets"].items() for b in spec["bases"]]


def check_training_seeds(lo: int, hi: int, what: str = "training", registry: dict | None = None) -> None:
    """Raise if the inclusive seed range [lo, hi] used for `what` overlaps any
    evaluation set -- training on evaluation boards would inflate results."""
    for name, a, b in eval_ranges(registry):
        if lo <= b and a <= hi:
            raise ValueError(f"{what} seeds [{lo}, {hi}] overlap evaluation seed set {name!r} "
                             f"[{a}, {b}]; choose a different --seed")


def usage_count(name: str, path: str = USAGE_PATH) -> int:
    if not os.path.exists(path):
        return 0
    with open(path) as f:
        return sum(1 for line in f if line.strip() and json.loads(line).get("set") == name)


def log_usage(name: str, note: dict, path: str = USAGE_PATH) -> int:
    """Append a usage record; returns how many times the set was used before."""
    prior = usage_count(name, path)
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=_DIR, capture_output=True,
                             text=True, timeout=10).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        sha = None
    with open(path, "a") as f:
        f.write(json.dumps({"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "set": name, "git": sha,
                            **note}) + "\n")
    return prior
