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
from pathlib import Path
import subprocess
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from utils import RandiTrialDataset, nested_grouped_splits


def mean_std(values):
    arr = np.asarray(values, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=1)) if arr.size > 1 else 0.0


def _resolve_init_checkpoint(init_from_dir: str, init_runs: list, fi: int, tag: str, seed: int) -> str:
    """Resolve the Stage-1 checkpoint for `(fi, tag, seed)` without multi-fold key collisions."""
    same_pair = [r for r in init_runs if r.get("tag") == tag and int(r.get("train_seed", -1)) == int(seed)]
    fold_matches = [
        r for r in same_pair
        if r.get("fold") == fi
        or os.path.basename(str(r.get("split_file", ""))) == f"fold_{fi}.json"
        or f"/fold{fi}/" in str(r.get("checkpoint", "")).replace("\\", "/")
    ]
    if len(fold_matches) == 1:
        ckpt_ref = fold_matches[0]["checkpoint"]
    elif not fold_matches and len(same_pair) == 1:
        ckpt_ref = same_pair[0]["checkpoint"]
    elif not same_pair:
        raise RuntimeError(f"No Stage-1 checkpoint entry for fold={fi} {(tag, int(seed))} in {init_from_dir}")
    else:
        raise RuntimeError(
            f"Ambiguous Stage-1 checkpoint entries ({len(fold_matches) or len(same_pair)}) "
            f"for fold={fi} {(tag, int(seed))} in {init_from_dir}"
        )
    cand = os.path.join(init_from_dir, f"fold{fi}", f"{tag}_seed{seed}", os.path.basename(ckpt_ref))
    stage1_ckpt = cand if os.path.isfile(cand) else ckpt_ref
    if not os.path.isfile(stage1_ckpt):
        raise FileNotFoundError(f"Resolved Stage-1 checkpoint missing: {stage1_ckpt}")
    return stage1_ckpt


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True)
    p.add_argument("--source", default="randi", choices=["randi", "toyoshima"])
    p.add_argument("--out", required=True)
    p.add_argument("--epochs", default=20, type=int)
    p.add_argument("--batch-size", default=128, type=int)
    p.add_argument("--substeps", default=2, type=int)
    p.add_argument("--lr", default=0.005, type=float)
    p.add_argument("--save-every", default=5, type=int)
    p.add_argument("--val-metric", default="post_inlier_r2", type=str,
                   help="Validation metric used to select the checkpoint (e.g. post_inlier_r2, evoked_hop1_r2, stim_r2)")
    p.add_argument("--no-prestim-encoder", action="store_true")
    p.add_argument("--folds", default=5, type=int)
    p.add_argument("--fold-indices", default="0")
    p.add_argument("--seeds", default="42,43,44")
    p.add_argument("--shuffle-seeds", default="7,8,9")
    p.add_argument("--shuffle-train-seed", default=42, type=int)
    p.add_argument("--per-neuron-kinetics", action="store_true")
    p.add_argument("--per-edge-release", action="store_true")
    p.add_argument("--per-neuron-stim-gain", action="store_true")
    p.add_argument("--wide-amp-priors", action="store_true")
    p.add_argument("--evoked-syn-priors", action="store_true")
    p.add_argument("--init-from-dir", default=None, type=str,
                   help="Directory with Stage-1 summary.json and fold{fi}/{tag}_seed{seed}/ checkpoints for Stage-2 warm start")
    p.add_argument("--freeze-non-prestim", action="store_true")
    p.add_argument("--stim-actuator", action="store_true")
    p.add_argument("--stim-tau-rise", default=0.08, type=float)
    p.add_argument("--stim-tau-decay", default=0.4, type=float)
    p.add_argument("--hop1-loss-weight", default=1.0, type=float)
    p.add_argument("--stim-loss-weight", default=1.0, type=float)
    p.add_argument("--causal-preprocess", action="store_true")
    p.add_argument("--python", default=sys.executable)
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    dataset = RandiTrialDataset.load(args.cache)
    folds = nested_grouped_splits(dataset, n_splits=args.folds, val_size=0.2, random_state=42)
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    shuffle_seeds = [int(x) for x in args.shuffle_seeds.split(",") if x.strip()]
    fold_indices = [int(x) for x in args.fold_indices.split(",") if x.strip()]
    init_runs = []
    if args.init_from_dir:
        init_summary = os.path.join(args.init_from_dir, "summary.json")
        if not os.path.isfile(init_summary):
            raise FileNotFoundError(f"Stage-1 summary not found: {init_summary}")
        with open(init_summary) as fh:
            init_runs = json.load(fh).get("runs", [])
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
            save_every = max(1, min(int(args.save_every), int(args.epochs)))
            cmd = [
                args.python, "pyro_svi_vae.py", str(args.epochs), str(save_every),
                "--source", args.source,
                "--dataset-cache", args.cache,
                "--split-file", fold_json,
                "--seed", str(seed),
                "--batch-size", str(args.batch_size),
                "--substeps", str(args.substeps),
                "--lr", str(args.lr),
                "--snapshot-dir", run_dir,
            ]
            if not args.no_prestim_encoder:
                cmd.append("--use-prestim-encoder")
            if args.causal_preprocess:
                cmd.append("--causal-preprocess")
            if args.per_neuron_kinetics:
                cmd.append("--per-neuron-kinetics")
            if args.per_edge_release:
                cmd.append("--per-edge-release")
            if args.per_neuron_stim_gain:
                cmd.append("--per-neuron-stim-gain")
            if args.wide_amp_priors:
                cmd.append("--wide-amp-priors")
            if args.evoked_syn_priors:
                cmd.append("--evoked-syn-priors")
            if args.init_from_dir:
                stage1_ckpt = _resolve_init_checkpoint(args.init_from_dir, init_runs, fi, tag, int(seed))
                cmd.extend(["--init-from", stage1_ckpt])
            if args.freeze_non_prestim:
                cmd.append("--freeze-non-prestim")
            if args.stim_actuator:
                cmd.append("--stim-actuator")
                cmd.extend(["--stim-tau-rise", str(args.stim_tau_rise), "--stim-tau-decay", str(args.stim_tau_decay)])
            if args.hop1_loss_weight != 1.0:
                cmd.extend(["--hop1-loss-weight", str(args.hop1_loss_weight)])
            if args.stim_loss_weight != 1.0:
                cmd.extend(["--stim-loss-weight", str(args.stim_loss_weight)])
            if shuffle_seed is not None:
                cmd.extend(["--shuffle-connectome", str(shuffle_seed)])
            log_path = os.path.join(run_dir, "train.log")
            with open(log_path, "w") as log:
                proc = subprocess.run(cmd, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, text=True)
            if proc.returncode != 0:
                raise RuntimeError(f"training failed: {tag} seed={seed}; see {log_path}")
            # Select the best validation checkpoint, not blindly the last
            # epoch.  The outer test fold is never read here.
            candidates = []
            for metric_path in Path(run_dir).glob("metrics_epoch_*.json"):
                with open(metric_path) as mf:
                    md = json.load(mf)
                epoch = int(md.get("epoch", 0))
                val = md.get("val", {}) or {}
                score = val.get(args.val_metric)
                if score is None:
                    score = val.get("evoked_hop1_r2", val.get("post_inlier_r2", val.get("r2")))
                if score is not None and np.isfinite(float(score)):
                    candidates.append((float(score), epoch, metric_path))
            if not candidates:
                raise RuntimeError(f"no validation metrics in {run_dir}")
            _, best_epoch, _ = max(candidates, key=lambda x: (x[0], x[1]))
            ckpt = os.path.join(run_dir, f"vae_epoch_{best_epoch}.pt")
            if not os.path.isfile(ckpt):
                raise RuntimeError(f"selected checkpoint missing: {ckpt}")
            eval_path = os.path.join(run_dir, "eval16.json")
            eval_cmd = [
                args.python, "tools/eval_checkpoint.py",
                "--cache", args.cache,
                "--split-file", fold_json,
                "--checkpoint", ckpt,
                "--out", eval_path,
                "--substeps", str(args.substeps),
                "--num-samples", "16",
            ]
            if args.no_prestim_encoder:
                eval_cmd.append("--no-prestim-encoder")
            if args.per_neuron_kinetics:
                eval_cmd.append("--per-neuron-kinetics")
            if args.per_edge_release:
                eval_cmd.append("--per-edge-release")
            if args.per_neuron_stim_gain:
                eval_cmd.append("--per-neuron-stim-gain")
            if args.wide_amp_priors:
                eval_cmd.append("--wide-amp-priors")
            if args.evoked_syn_priors:
                eval_cmd.append("--evoked-syn-priors")
            if args.stim_actuator:
                eval_cmd.append("--stim-actuator")
                eval_cmd.extend(["--stim-tau-rise", str(args.stim_tau_rise), "--stim-tau-decay", str(args.stim_tau_decay)])
            if shuffle_seed is not None:
                eval_cmd.extend(["--shuffle-seed", str(shuffle_seed)])
            proc = subprocess.run(eval_cmd, cwd=REPO, capture_output=True, text=True)
            if proc.returncode != 0:
                raise RuntimeError(f"evaluation failed: {tag} seed={seed}\n{proc.stderr[-4000:]}")
            with open(eval_path) as fh:
                ev = json.load(fh)
            ev["fold"] = fi
            ev["tag"] = tag
            ev["train_seed"] = seed
            ev["selected_epoch"] = best_epoch
            ev["selected_val_metric"] = args.val_metric
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
            "pass": bool(mi > 0.0 and delta > 0.0 and delta > pooled_sem),
            "rule": "intact_mean > 0 and delta_mean > 0 and delta_mean > pooled SEM (one-sided ~84%)",
        }
    else:
        gate = {"metric": "evoked_hop1_r2", "pass": False, "reason": "no paired runs"}
    summary = {"gate": gate, "runs": records}
    with open(os.path.join(args.out, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(gate, indent=2), flush=True)


if __name__ == "__main__":
    main()
