#!/usr/bin/env python3
"""Evaluate a trained VAE checkpoint on a fixed split file with 16 posterior samples."""
import argparse
import json
import os
import sys

import numpy as np
import pyro
import torch
from torch.utils.data import DataLoader, Subset

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import c302
import vae
from pyro_svi_vae import evaluate_predictions
from utils import RandiTrialDataset


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True)
    p.add_argument("--split-file", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--shuffle-seed", default=None, type=int)
    p.add_argument("--no-prestim-encoder", action="store_true")
    p.add_argument("--per-neuron-kinetics", action="store_true")
    p.add_argument("--per-edge-release", action="store_true")
    p.add_argument("--per-neuron-stim-gain", action="store_true")
    p.add_argument("--wide-amp-priors", action="store_true")
    p.add_argument("--evoked-syn-priors", action="store_true")
    p.add_argument("--direct-global-guide", action="store_true")
    p.add_argument("--use-extrasynaptic", action="store_true")
    p.add_argument("--stim-actuator", action="store_true")
    p.add_argument("--stim-tau-rise", default=0.08, type=float)
    p.add_argument("--stim-tau-decay", default=0.4, type=float)
    p.add_argument("--substeps", default=2, type=int)
    p.add_argument("--eval-seed", default=42, type=int,
                   help="RNG seed for deterministic posterior predictive draws")
    p.add_argument("--num-samples", default=16, type=int)
    p.add_argument("--pre-vols", default=10, type=int)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = RandiTrialDataset.load(args.cache)
    with open(args.split_file) as fh:
        split = json.load(fh)
    test_idx = [int(i) for i in split["test"]]
    loader = DataLoader(Subset(dataset, test_idx), batch_size=64, shuffle=False)

    step = c302.readConnectome(
        os.path.join(REPO, "data/CElegansNeuronTables.xls"),
        use_extrasynaptic=args.use_extrasynaptic,
        per_edge_release=args.per_edge_release,
        per_neuron_stim_gain=args.per_neuron_stim_gain,
        wide_amp_priors=args.wide_amp_priors,
        evoked_syn_priors=args.evoked_syn_priors,
        shuffle_seed=args.shuffle_seed,
    )
    use_prestim = not args.no_prestim_encoder
    rec = c302.RecurrentNematode(
        step,
        dt_obs=0.5,
        substeps=args.substeps,
        use_prestim_encoder=use_prestim,
        pre_vols=args.pre_vols,
        prestim_rank=16,
        variational_prestim=bool(args.direct_global_guide and use_prestim),
        per_neuron_kinetics=args.per_neuron_kinetics,
        per_edge_release=args.per_edge_release,
        stim_actuator=args.stim_actuator,
        stim_tau_rise=args.stim_tau_rise,
        stim_tau_decay=args.stim_tau_decay,
    )
    model = vae.VAElegans(
        rec,
        rec.model.Neuron.neuronSize,
        len(rec.model.synapse.Wicks_SRC),
        len(rec.model.synapse.Gap_Junction_SRC),
        direct_global_guide=args.direct_global_guide,
    ).to(device)
    snap = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(snap["MODEL_STATE"])
    pyro.get_param_store().set_state(snap["PARAM_STORE"])
    model.eval()

    # Always score against the intact biological adjacency; a shuffled model is a
    # control, not a new connectome definition.
    intact = c302.readConnectome(os.path.join(REPO, "data/CElegansNeuronTables.xls"))
    adjacency = intact.synapse.adjacency_matrix(intact.neuronSize)
    np_adjacency = intact.synapse.extrasynaptic_adjacency_matrix(intact.neuronSize)
    np.random.seed(args.eval_seed)
    torch.manual_seed(args.eval_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.eval_seed)
    pyro.set_rng_seed(args.eval_seed)
    metrics = evaluate_predictions(
        model,
        loader,
        device,
        guide=model.guide,
        num_samples=args.num_samples,
        pre_vols=args.pre_vols,
        adjacency=adjacency,
        np_adjacency=np_adjacency,
    )
    result = {
        "metrics": metrics,
        "checkpoint": args.checkpoint,
        "split_file": args.split_file,
        "shuffle_seed": args.shuffle_seed,
        "wide_amp_priors": bool(args.wide_amp_priors),
        "evoked_syn_priors": bool(args.evoked_syn_priors),
        "direct_global_guide": bool(args.direct_global_guide),
        "use_extrasynaptic": bool(args.use_extrasynaptic),
        "substeps": args.substeps,
        "eval_seed": args.eval_seed,
        "num_samples": args.num_samples,
        "test_trials": len(test_idx),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(result, fh, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
