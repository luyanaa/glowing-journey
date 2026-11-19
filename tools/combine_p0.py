#!/usr/bin/env python3
"""Combine per-session P0 summaries and recompute the evoked shuffle gate."""
import argparse
import json
import os

import numpy as np


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("summaries", nargs="+")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    runs = []
    for path in args.summaries:
        with open(path) as fh:
            data = json.load(fh)
        runs.extend(data.get("runs", []))
    intact = [r["metrics"]["evoked_hop1_r2"] for r in runs if r["tag"] == "intact"]
    shuffled = [r["metrics"]["evoked_hop1_r2"] for r in runs if r["tag"].startswith("shuffle")]
    intact = [x for x in intact if np.isfinite(x)]
    shuffled = [x for x in shuffled if np.isfinite(x)]
    mi, si = float(np.mean(intact)), float(np.std(intact, ddof=1)) if len(intact) > 1 else 0.0
    ms, ss = float(np.mean(shuffled)), float(np.std(shuffled, ddof=1)) if len(shuffled) > 1 else 0.0
    delta = mi - ms
    sem = float(np.sqrt(si ** 2 / max(len(intact), 1) + ss ** 2 / max(len(shuffled), 1)))
    gate = {
        "metric": "evoked_hop1_r2",
        "n_intact": len(intact),
        "n_shuffle": len(shuffled),
        "intact_mean": mi,
        "intact_std": si,
        "shuffle_mean": ms,
        "shuffle_std": ss,
        "delta_mean": delta,
        "pooled_sem": sem,
        "pass": bool(delta > 0.0 and delta > sem),
        "rule": "delta_mean > 0 and delta_mean > pooled SEM",
    }
    out = {"gate": gate, "runs": runs}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps(gate, indent=2))


if __name__ == "__main__":
    main()
