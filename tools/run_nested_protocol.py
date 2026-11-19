#!/usr/bin/env python3
"""Run a nested recording-grouped CV fold with seed variance and connectome shuffles.

The outer test fold is never used for model selection.  Inside each outer train
fold, a recording-grouped validation split drives early stopping.  Main results
are reported as mean ± std over training seeds; the connectome-shuffle gate
compares evoked 1-hop R² against 3 degree-preserving permutations.
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from utils import RandiTrialDataset, nested_grouped_splits


def mean_std(values):
    arr = np.asarray(values, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=1)) if arr.size > 1 else 0.0


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True)
    p.add_argument("--source", default="randi", choices=["randi", "toyoshima"])
    p.add_argument("--out", required=True)
    p.add_argument("--epochs", default=20, type=int)
    p.add_argument("--batch-size", default=128, type=int)
    p.add_argument("--substeps", default=2, type=int)
    p.add_argument("--lr", default=0.001, type=float)
    p.add_argument("--folds", default=5, type=int)
    p.add_argument("--fold-indices", default="0")
    p.add_argument("--seeds", default="42,43,44")
    p.add_argument("--shuffle-seeds", default="7,8,9")
    p.add_argument("--shuffle-train-seed", default=42, type=int)
    p.add_argument("--per-neuron-kinetics", action="store_true")
    p.add_argument("--per-edge-release", action="store_true")
    p.add_argument("--causal-preprocess", action="store_true")
    p.add_argument("--python", default=sys.executable)
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    dataset = RandiTrialDataset.load(args.cache)
    folds = nested_grouped_splits(dataset, n_splits=args.folds, val_size=0.2, random_state=42)
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    shuffle_seeds = [int(x) for x in args.shuffle_seeds.split(",") if x.strip()]
    fold_indices = [int(x) for x in args.fold_indices.split(",") if x.strip()]
    records = []

    for fi in fold_indices:
        fold = folds[fi]
        fold_json = os.path.join(args.out, f"fold_{fi}.json")
        with open(fold_json, "w") as fh:
            json.dump(fold, fh)
        print(f"fold {fi}: train={len(fold['train'])} val={len(fold['val'])} test={len(fold['test'])}", flush=True)

        runs = [("intact", seed, None) for seed in seeds]
        runs += [(f"shuffle{s}", args.shuffle_train_seed, s) for s in shuffle_seeds]
        for tag, seed, shuffle_seed in runs:
            run_dir = os.path.join(args.out, f"fold{fi}", f"{tag}_seed{seed}")
            os.makedirs(run_dir, exist_ok=True)
            cmd = [
                args.python, "pyro_svi_vae.py", str(args.epochs), str(args.epochs),
                "--source", args.source,
                "--dataset-cache", args.cache,
                "--split-file", fold_json,
                "--seed", str(seed),
                "--batch-size", str(args.batch_size),
                "--substeps", str(args.substeps),
                "--lr", str(args.lr),
                "--use-prestim-encoder",
                "--snapshot-dir", run_dir,
            ]
            if args.causal_preprocess:
                cmd.append("--causal-preprocess")
            if args.per_neuron_kinetics:
                cmd.append("--per-neuron-kinetics")
            if args.per_edge_release:
                cmd.append("--per-edge-release")
            if shuffle_seed is not None:
                cmd.extend(["--shuffle-connectome", str(shuffle_seed)])
            log_path = os.path.join(run_dir, "train.log")
            with open(log_path, "w") as log:
                proc = subprocess.run(cmd, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, text=True)
            if proc.returncode != 0:
                raise RuntimeError(f"training failed: {tag} seed={seed}; see {log_path}")
            ckpt = os.path.join(run_dir, f"vae_epoch_{args.epochs}.pt")
            eval_path = os.path.join(run_dir, "eval16.json")
            eval_cmd = [
                args.python, "tools/eval_checkpoint.py",
                "--cache", args.cache,
                "--split-file", fold_json,
                "--checkpoint", ckpt,
                "--out", eval_path,
                "--num-samples", "16",
            ]
            if args.per_neuron_kinetics:
                eval_cmd.append("--per-neuron-kinetics")
            if args.per_edge_release:
                eval_cmd.append("--per-edge-release")
            if shuffle_seed is not None:
                eval_cmd.extend(["--shuffle-seed", str(shuffle_seed)])
            proc = subprocess.run(eval_cmd, cwd=REPO, capture_output=True, text=True)
            if proc.returncode != 0:
                raise RuntimeError(f"evaluation failed: {tag} seed={seed}\n{proc.stderr[-4000:]}")
            with open(eval_path) as fh:
                ev = json.load(fh)
            ev["tag"] = tag
            ev["train_seed"] = seed
            records.append(ev)
        # End run loop
    # End fold loop

    intact = [r["metrics"].get("evoked_hop1_r2", float("nan")) for r in records if r["tag"] == "intact"]
    shuffled = [r["metrics"].get("evoked_hop1_r2", float("nan")) for r in records if r["tag"].startswith("shuffle")]
    intact = [x for x in intact if np.isfinite(x)]
    shuffled = [x for x in shuffled if np.isfinite(x)]
    if intact and shuffled:
        mi, si = mean_std(intact)
        ms, ss = mean_std(shuffled)
        delta = mi - ms
        pooled_sem = float(np.sqrt(si ** 2 / max(len(intact), 1) + ss ** 2 / max(len(shuffled), 1)))
        gate = {
            "metric": "evoked_hop1_r2",
            "intact_mean": mi,
            "intact_std": si,
            "shuffle_mean": ms,
            "shuffle_std": ss,
            "delta_mean": delta,
            "pooled_sem": pooled_sem,
            "pass": bool(delta > 0.0 and delta > pooled_sem),
            "rule": "delta_mean > 0 and delta_mean > pooled SEM (one-sided ~84%)",
        }
    else:
        gate = {"metric": "evoked_hop1_r2", "pass": False, "reason": "no paired runs"}
    summary = {"gate": gate, "runs": records}
    with open(os.path.join(args.out, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(gate, indent=2), flush=True)


if __name__ == "__main__":
    main()
