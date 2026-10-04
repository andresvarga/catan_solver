"""Per-run manifest written next to a run's checkpoints (audit F-18).

The project's former champion was lost because its checkpoints, training pools
and the record of how they were produced lived only in an ephemeral job
directory. Every training driver now writes `manifest.json` into its output
directory at startup: git commit + dirty files, the exact command line and
parsed arguments, library versions, host and start time. It is cheap and makes
any surviving checkpoint re-creatable and re-evaluable.
"""
from __future__ import annotations

import json
import os
import platform
import socket
import subprocess
import sys
import time

# Paths that are routinely cleaned up -- outputs written here are at risk.
_EPHEMERAL_PREFIXES = ("/tmp/", "/var/tmp/", os.path.expanduser("~/.claude/jobs/"))


def _git(*args: str) -> str | None:
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True,
                              timeout=10, check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def write_manifest(out_dir: str, args, extra: dict | None = None) -> str:
    """Write `<out_dir>/manifest.json` (an existing one is kept as
    `manifest.<n>.json`, so resumed runs keep their history). Returns the path."""
    import numpy as np
    import torch

    os.makedirs(out_dir, exist_ok=True)
    abs_dir = os.path.abspath(out_dir)
    ephemeral = abs_dir.startswith(_EPHEMERAL_PREFIXES)
    if ephemeral:
        print(f"WARNING: writing run outputs under an ephemeral path ({abs_dir}); "
              "checkpoints there may be deleted -- copy anything worth keeping.", file=sys.stderr)
    manifest = {
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "out_dir": abs_dir,
        "ephemeral_out_dir": ephemeral,
        "argv": sys.argv,
        "args": vars(args) if hasattr(args, "__dict__") else dict(args),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty_files": (_git("status", "--porcelain") or "").splitlines(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda_available": torch.cuda.is_available(),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        **(extra or {}),
    }
    path = os.path.join(out_dir, "manifest.json")
    if os.path.exists(path):
        n = 1
        while os.path.exists(os.path.join(out_dir, f"manifest.{n}.json")):
            n += 1
        os.replace(path, os.path.join(out_dir, f"manifest.{n}.json"))
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    return path
